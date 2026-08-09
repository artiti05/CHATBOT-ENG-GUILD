import os
import re
import math
import time
import requests
from typing import Dict, Any, List, Tuple, Optional
import chromadb
from chromadb.config import Settings as ChromaSettings

try:
    from sentence_transformers import CrossEncoder
    HAS_RERANKER = True
except ImportError:
    HAS_RERANKER = False

from core.config import (
    CHROMA_PERSIST_DIR, CHROMA_COLLECTION_NAME, BGE_RERANKER_MODEL_NAME,
    OLLAMA_URL, OLLAMA_CHAT_MODEL, OLLAMA_VISION_MODEL,
    NUM_CTX, NUM_PREDICT, OLLAMA_KEEP_ALIVE,
    RERANK_THRESHOLD, RERANK_SCORE_FLOOR, RERANK_SCORE_CEIL,
    DOCUMENT_PRIORITY
)

from core.ingestion import BGEM3Embedder

# ---------------------------------------------------------------------------
# 1. BGE CROSS-ENCODER RERANKER LAYER
# ---------------------------------------------------------------------------
class BGEReranker:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(BGEReranker, cls).__new__(cls)
            cls._instance.model = None
            cls._instance._load_model()
        return cls._instance

    def _load_model(self):
        if HAS_RERANKER:
            try:
                import torch
                # Enable multi-core CPU parallelism
                num_cores = os.cpu_count() or 8
                torch.set_num_threads(min(num_cores, 8))
                device = "cuda" if torch.cuda.is_available() else "cpu"
                print(f"[Reranker] Loading BGE Reranker model '{BGE_RERANKER_MODEL_NAME}' on {device} (threads={torch.get_num_threads()})...")
                self.model = CrossEncoder(BGE_RERANKER_MODEL_NAME, max_length=256, device=device)
                print(f"[Reranker] Successfully loaded BGE Reranker on {device}.")
            except Exception as e:
                print(f"[Reranker Warning] Could not load BGE Reranker model: {e}")
                self.model = None

    def rerank(self, query: str, candidate_chunks: List[Dict[str, Any]], top_k: int = 10) -> List[Dict[str, Any]]:
        if not candidate_chunks:
            return []
        if not self.model:
            return candidate_chunks[:top_k]

        pairs = [[query, chunk.get("text", "")[:600]] for chunk in candidate_chunks]
        scores = self.model.predict(pairs, batch_size=16, show_progress_bar=False)

        for i, score in enumerate(scores):
            raw = float(score)
            candidate_chunks[i]["rerank_score"] = raw
            # Sigmoid maps raw logit to probability:
            #   raw >  2.0  →  88-97%  (strong match)
            #   raw  0-2.0  →  50-88%  (moderate match)
            #   raw <  0.0  →  filtered out (irrelevant)
            sig = 1.0 / (1.0 + math.exp(-raw))          # sigmoid: 0.0 → 1.0
            match_pct = int(sig * 100)
            match_pct = min(max(match_pct, RERANK_SCORE_FLOOR), RERANK_SCORE_CEIL)
            candidate_chunks[i]["similarity_score"] = match_pct

        # Filter: drop chunks below the raw logit threshold (genuinely irrelevant)
        above_threshold = [c for c in candidate_chunks if c.get("rerank_score", -99) >= RERANK_THRESHOLD]

        # If everything is filtered out, relax and keep the best few anyway
        if not above_threshold:
            above_threshold = sorted(candidate_chunks, key=lambda x: x.get("rerank_score", -99), reverse=True)[:3]

        ranked = sorted(above_threshold, key=lambda x: x.get("rerank_score", 0.0), reverse=True)
        for rank, chunk in enumerate(ranked[:top_k], 1):
            chunk["rank"] = rank

        return ranked[:top_k]

# ---------------------------------------------------------------------------
# 2. HYBRID VECTOR SEARCH KNOWLEDGE RETRIEVER
# ---------------------------------------------------------------------------
class KnowledgeRetriever:
    def __init__(self):
        self.client = chromadb.PersistentClient(path=str(CHROMA_PERSIST_DIR), settings=ChromaSettings(anonymized_telemetry=False))
        self.collection = self.client.get_or_create_collection(name=CHROMA_COLLECTION_NAME)
        self.embedder = BGEM3Embedder()
        self.reranker = BGEReranker()

    def _get_priority_boost(self, file_name: str) -> float:
        """Return additive logit boost for authoritative source documents.
        Checked against substrings of the file_name so partial matches work."""
        for pattern, boost in DOCUMENT_PRIORITY.items():
            if pattern in file_name:
                return boost
        return 0.0

    def retrieve(self, query_text: str, top_k: int = 10) -> List[Dict[str, Any]]:
        if not query_text or not query_text.strip():
            return []

        dense_vecs, _ = self.embedder.embed_texts([query_text])
        query_dense = dense_vecs[0]

        results = self.collection.query(
            query_embeddings=[query_dense],
            n_results=min(top_k * 2, 20),  # over-sample more to let priority re-rank
            include=["documents", "metadatas", "distances"]
        )

        candidates = []
        if results and results.get("documents") and results["documents"][0]:
            docs  = results["documents"][0]
            metas = results["metadatas"][0]
            dists = results["distances"][0]

            for i in range(len(docs)):
                meta = metas[i] if i < len(metas) else {}
                candidates.append({
                    "text":             docs[i],
                    "title":            meta.get("file_name", "وثيقة"),
                    "section_title":    f"جزء {meta.get('chunk_index', 1)}",
                    "similarity_score": 0,
                    "distance":         dists[i],
                    "metadata":         meta
                })

        # Step 1: cross-encoder rerank
        reranked = self.reranker.rerank(query_text, candidates, top_k=top_k)

        # Step 2: apply document priority boost to raw logit, recompute sigmoid score
        for chunk in reranked:
            boost = self._get_priority_boost(chunk.get("title", ""))
            if boost > 0.0:
                new_raw = chunk.get("rerank_score", 0.0) + boost
                chunk["rerank_score"] = new_raw
                sig = 1.0 / (1.0 + math.exp(-new_raw))
                chunk["similarity_score"] = min(max(int(sig * 100), RERANK_SCORE_FLOOR), RERANK_SCORE_CEIL)

        # Step 3: re-sort by boosted score, reassign ranks
        reranked.sort(key=lambda x: x.get("rerank_score", 0.0), reverse=True)
        for rank, chunk in enumerate(reranked, 1):
            chunk["rank"] = rank

        return reranked

def clean_formatting(text: str) -> str:
    """Strips markdown bold/italic asterisks (**text**) and header hashes to deliver clean presentation text."""
    if not text:
        return ""
    # Strip markdown bold/italic asterisks: **text** -> text, *text* -> text
    cleaned = re.sub(r'\*+', '', text)
    # Strip markdown headers like ### -> clean text
    cleaned = re.sub(r'^#+\s*', '', cleaned, flags=re.MULTILINE)
    return cleaned.strip()

# ---------------------------------------------------------------------------
# 3. CONSOLIDATED RAG & DIALECT CHATBOT ENGINE
# ---------------------------------------------------------------------------
class RAGChatbot:
    def __init__(self):
        self.retriever = KnowledgeRetriever()
        self.ollama_url = OLLAMA_URL
        self.ollama_model = OLLAMA_CHAT_MODEL

    def detect_and_normalize_query(self, query: str) -> Tuple[str, str]:
        q = query.strip()

        # 1. English detection & MSA normalization for vector retrieval
        if re.search(r'[a-zA-Z]{3,}', q) and not re.search(r'[\u0600-\u06FF]', q):
            english_msa_map = {
                "registration": "التسجيل في نقابة المهندسين",
                "requirements": "شروط ومتطلبات التسجيل",
                "insurance": "التأمين الصحي والأطباء",
                "pension": "صندوق التقاعد والاستعلام",
                "services": "الخدمات الإلكترونية والمساندة",
                "loans": "الاستعلام عن القروض",
                "fees": "الرسوم والاشتراكات"
            }
            normalized_terms = []
            lower_q = q.lower()
            for eng_kw, msa_trans in english_msa_map.items():
                if eng_kw in lower_q:
                    normalized_terms.append(msa_trans)
            
            normalized_search = " ".join(normalized_terms) if normalized_terms else q
            return normalized_search, "english"

        # 2. Jordanian Dialect detection & MSA normalization
        jordanian_keywords = [
            "شو", "بدي", "عشان", "عشانك", "كيف بقدر", "وين", "قديش", "ايش", 
            "بصير", "يلي", "ليش", "هون", "هناك", "عم بدرس", "حبيت اعرف", "اسجل", "شباب", "الاوراق", "الأوراق"
        ]

        if any(re.search(r'\b' + re.escape(kw) + r'\b', q) for kw in jordanian_keywords):
            normalized_search = q
            replacements = {
                "شو هي": "ما هي", "شو الاوراق": "الوثائق والمستندات", "شو الأوراق": "الوثائق والمستندات",
                "شو": "ما هي", "بدي أسجل": "التسجيل في النقابة", "بدي اسجل": "التسجيل في النقابة",
                "بدي": "أريد", "عشان": "من أجل", "قديش": "ما قيمة", "كيف بقدر": "كيفية", "وين": "مكان",
                "ايش": "ما هي", "حبيت اعرف": "أستفسر عن"
            }
            for k, v in replacements.items():
                normalized_search = normalized_search.replace(k, v)
            return normalized_search, "jordanian"

        return q, "msa"

    def answer_question(self, query: str, history: Optional[List[Dict[str, str]]] = None, top_k: int = 15) -> Dict[str, Any]:
        start_time = time.time()
        if not query or not query.strip():
            return {"query": query, "answer": "الرجاء إدخال سؤال للبحث والإجابة.", "sources": [], "time_taken": 0.0}

        normalized_query, detected_accent = self.detect_and_normalize_query(query)
        sources = self.retriever.retrieve(query_text=normalized_query, top_k=top_k)

        if not sources:
            fallback_msg = {
                "jordanian": "للأسف ما لقيت وثائق أو معلومات مباشرة بتخص سؤالك بقاعدة المعرفة حالياً.",
                "english": "Sorry, no relevant documents or sources were found in the knowledge base.",
                "msa": "لم يتم العثور على وثائق أو مصادر مرتبطة بسؤالك في قاعدة المعرفة."
            }
            return {"query": query, "answer": fallback_msg.get(detected_accent, fallback_msg["msa"]), "sources": [], "time_taken": round(time.time() - start_time, 2)}

        # Dynamic Context Window Exploitation — only include sources above relevance threshold
        # Note: With 300-token chunks, BGE reranker scores cluster around 33-39% for good matches.
        # A threshold of 28% filters truly irrelevant noise (raw BGE score < -0.8) while keeping real results.
        RELEVANCE_THRESHOLD = 28  # Min similarity % to include a source
        MAX_CONTEXT_CHARS = 9000

        context_blocks = []
        accumulated_chars = 0
        included_sources = []

        for src in sources:
            score = src.get("similarity_score", 0)
            if score < RELEVANCE_THRESHOLD:
                continue  # Skip weak/irrelevant chunks

            rank = src.get("rank", len(included_sources) + 1)
            title = src.get("title", "وثيقة")
            section = src.get("section_title", "")
            raw_text = src.get("text", "").strip()

            header = f"[المصدر {rank}] {title}"
            if section:
                header += f" — {section}"
            header += f" (نسبة التطابق: {score}%)"

            block_str = f"{header}\n{raw_text}\n"
            if accumulated_chars + len(block_str) > MAX_CONTEXT_CHARS and context_blocks:
                break

            context_blocks.append(block_str)
            accumulated_chars += len(block_str)
            included_sources.append(src)

        # If ALL sources are below threshold, return honest no-results response
        if not included_sources:
            no_info_msg = {
                "jordanian": "ما عندي معلومات كافية في قاعدة المعرفة تخص هاد السؤال. تواصل مع نقابة المهندسين مباشرة للاستفسار.",
                "english": "I could not find sufficient information in the knowledge base for this question. Please contact the Jordan Engineers Association directly.",
                "msa": "لا تتوفر معلومات كافية في قاعدة المعرفة للإجابة على هذا السؤال بدقة. يُرجى التواصل مع نقابة المهندسين الأردنيين مباشرة."
            }
            return {"query": query, "answer": no_info_msg.get(detected_accent, no_info_msg["msa"]), "sources": [], "llm_connected": False, "time_taken": round(time.time() - start_time, 2)}

        context_str = "\n" + ("=" * 50) + "\n" + "\n".join(context_blocks) + ("=" * 50)

        history_str = ""
        if history and len(history) > 0:
            hist_lines = []
            for h in history[-6:]:
                role = "المستخدم" if h.get("role") == "user" else "المساعد"
                hist_lines.append(f"{role}: {h.get('content', '')}")
            history_str = "\nسياق المحادثة السابقة:\n" + "\n".join(hist_lines) + "\n"

        # --- Dynamic language & tone instruction ---
        if detected_accent == "jordanian":
            lang_instruction = (
                "لغة الإجابة: اللهجة الأردنية الودية والواضحة — خاطب المهندس بأسلوب نقابي حميمي ومتعاطف."
            )
        elif detected_accent == "english":
            lang_instruction = (
                "Response language: Clear, professional English. Use formal tone appropriate for an engineering association."
            )
        else:
            lang_instruction = (
                "لغة الإجابة: العربية الفصحى الرسمية — أسلوب واضح ومحترف يليق بنقابة مهنية."
            )

        # --- Dynamic query-type detection for response shaping ---
        q_lower = query.lower()
        is_procedural = any(kw in q_lower for kw in ["كيف", "خطوات", "إجراءات", "طريقة", "how", "steps", "process", "procedure"])
        is_list_query = any(kw in q_lower for kw in ["ما هي", "شو هي", "what are", "اذكر", "قائمة", "list"])
        is_fee_query  = any(kw in q_lower for kw in ["رسوم", "اشتراك", "قسط", "تكلفة", "كم", "قديش", "fee", "cost", "price"])
        is_eligibility = any(kw in q_lower for kw in ["شروط", "متطلبات", "من يحق", "يستحق", "eligib", "require", "condition"])

        if is_procedural:
            response_shape = "قدّم الإجابة كخطوات مرقمة واضحة ومتسلسلة."
        elif is_list_query:
            response_shape = "قدّم الإجابة كقائمة نقطية منظمة، بند واحد لكل عنصر."
        elif is_fee_query:
            response_shape = "قدّم جميع المبالغ والرسوم في جدول أو قائمة مرقمة واضحة مع ذكر الشرائح إن وجدت."
        elif is_eligibility:
            response_shape = "وضّح الشروط والمتطلبات في قائمة مرقمة، مع تمييز الشروط الإلزامية عن الاختيارية."
        else:
            response_shape = "قدّم الإجابة في فقرات موجزة ومنظمة تجيب مباشرة على السؤال."

        # --- Robust system prompt: reasoning synthesizer ---
        num_sources = len(included_sources)
        system_prompt = (
            f"أنت مساعد ذكي متخصص في شؤون نقابة المهندسين الأردنيين (Jordan Engineers Association — JEA).\n"
            f"مهمتك: قراءة المصادر المرفقة بعناية، استيعابها، والإجابة بطريقة تركيبية منطقية.\n"
            f"لديك {num_sources} مصدر من قاعدة المعرفة لهذا السؤال.\n"
            f"\n"
            f"أسلوب الإجابة المطلوب:\n"
            f"• ابدأ بجملة تمهيدية واحدة موجزة تحدد محور الإجابة.\n"
            f"• اقرأ جميع المصادر واستخلص كل المعلومات ذات الصلة بالسؤال.\n"
            f"• اربط المعلومات من مصادر مختلفة إن تكاملت، واذكر رقم المصدر [المصدر N] عند كل معلومة.\n"
            f"• إذا كانت صياغة المصدر تقنية أو غير واضحة، وضّحها بلغة بسيطة — لكن لا تضف أي فكرة من خارج المصادر.\n"
            f"• اختم بجملة خاتمة تلخّص الفكرة الرئيسية أو توجّه المستخدم للخطوة التالية عند الحاجة.\n"
            f"\n"
            f"قواعد صارمة:\n"
            f"1. لا تخترع أي معلومة. كل فكرة يجب أن تكون موجودة في المصادر المرفقة.\n"
            f"2. إذا لم تجد إجابة في المصادر، قل ذلك صراحةً بدلاً من التكهن.\n"
            f"3. يُحظر استخدام رموز Markdown (* أو ** أو ##). استخدم الأرقام والنقاط النصية فقط.\n"
            f"4. {response_shape}\n"
            f"5. {lang_instruction}\n"
        )

        full_prompt = (
            f"{system_prompt}\n"
            f"{'=' * 60}\n"
            f"المصادر من قاعدة المعرفة:\n{context_str}\n"
            f"{'=' * 60}\n"
            f"{history_str}"
            f"سؤال المستخدم (أجب عنه فقط من المصادر أعلاه): {query}\n"
            f"{'=' * 60}\n"
            f"الإجابة:"
        )

        generated_answer = ""
        llm_success = False

        # Direct Ollama API Call (Primary Default)
        if self.provider == "ollama" or not llm_success:
            models_to_try = [self.ollama_model, "qwen2.5:7b", "ministral-3:3b-instruct-2512-q4_K_M"]
            for model_tag in models_to_try:
                try:
                    ollama_payload = {
                        "model": model_tag,
                        "prompt": full_prompt,
                        "stream": False,
                        "keep_alive": OLLAMA_KEEP_ALIVE,
                        "options": {
                            "temperature": 0.1,
                            "num_predict": NUM_PREDICT,
                            "num_ctx": NUM_CTX
                        }
                    }
                    res = requests.post(self.ollama_url, json=ollama_payload, timeout=(5.0, 300.0))
                    if res.status_code == 200:
                        content = res.json().get("response", "").strip()
                        if content:
                            generated_answer = content
                            llm_success = True
                            print(f"[RAG Chatbot] Successfully generated answer using Ollama model '{model_tag}'.")
                            break
                except Exception as o_err:
                    print(f"[RAG Chatbot Warning] Ollama call ('{model_tag}') failed: {o_err}")


        if not llm_success:
            generated_answer = (
                "تم استخراج أهم المصادر ذات صلة بسؤالك من قاعدة المعرفة. "
                "(ملاحظة: خادم التوليد المحلي Ollama غير متصل حالياً للتوليد المباشر، يمكنك الاطلاع على المصادر أدناه):"
            )

        # Apply post-processing cleaner to guarantee zero markdown star artifacts (**text**)
        clean_answer = clean_formatting(generated_answer)

        return {
            "query": query,
            "normalized_query": normalized_query,
            "detected_accent": detected_accent,
            "answer": clean_answer,
            "sources": included_sources,
            "llm_connected": llm_success,
            "time_taken": round(time.time() - start_time, 2)
        }

    def answer_question_stream(self, query: str, history: Optional[List[Dict[str, str]]] = None, top_k: int = 15):
        """Streaming generator that yields JSON chunks for SSE / real-time token rendering."""
        import json
        start_time = time.time()
        if not query or not query.strip():
            yield f"data: {json.dumps({'type': 'done', 'answer': 'الرجاء إدخال سؤال للبحث والإجابة.', 'sources': []}, ensure_ascii=False)}\n\n"
            return

        normalized_query, detected_accent = self.detect_and_normalize_query(query)
        sources = self.retriever.retrieve(query_text=normalized_query, top_k=top_k)

        if not sources:
            fallback_msg = {
                "jordanian": "للأسف ما لقيت وثائق أو معلومات مباشرة بتخص سؤالك بقاعدة المعرفة حالياً.",
                "english": "Sorry, no relevant documents or sources were found in the knowledge base.",
                "msa": "لم يتم العثور على وثائق أو مصادر مرتبطة بسؤالك في قاعدة المعرفة."
            }
            ans = fallback_msg.get(detected_accent, fallback_msg["msa"])
            yield f"data: {json.dumps({'type': 'done', 'answer': ans, 'sources': [], 'time_taken': round(time.time() - start_time, 2)}, ensure_ascii=False)}\n\n"
            return

        RELEVANCE_THRESHOLD = 28
        MAX_CONTEXT_CHARS = 9000

        context_blocks = []
        accumulated_chars = 0
        included_sources = []

        for src in sources:
            score = src.get("similarity_score", 0)
            if score < RELEVANCE_THRESHOLD:
                continue

            rank = src.get("rank", len(included_sources) + 1)
            title = src.get("title", "وثيقة")
            section = src.get("section_title", "")
            raw_text = src.get("text", "").strip()

            header = f"[المصدر {rank}] {title}"
            if section:
                header += f" — {section}"
            header += f" (نسبة التطابق: {score}%)"

            block_str = f"{header}\n{raw_text}\n"
            if accumulated_chars + len(block_str) > MAX_CONTEXT_CHARS and context_blocks:
                break

            context_blocks.append(block_str)
            accumulated_chars += len(block_str)
            included_sources.append(src)

        if not included_sources:
            no_info_msg = {
                "jordanian": "ما عندي معلومات كافية في قاعدة المعرفة تخص هاد السؤال. تواصل مع نقابة المهندسين مباشرة للاستفسار.",
                "english": "I could not find sufficient information in the knowledge base for this question. Please contact the Jordan Engineers Association directly.",
                "msa": "لا تتوفر معلومات كافية في قاعدة المعرفة للإجابة على هذا السؤال بدقة. يُرجى التواصل مع نقابة المهندسين الأردنيين مباشرة."
            }
            ans = no_info_msg.get(detected_accent, no_info_msg["msa"])
            yield f"data: {json.dumps({'type': 'done', 'answer': ans, 'sources': [], 'time_taken': round(time.time() - start_time, 2)}, ensure_ascii=False)}\n\n"
            return

        # Send metadata event first
        yield f"data: {json.dumps({'type': 'meta', 'sources': included_sources, 'detected_accent': detected_accent}, ensure_ascii=False)}\n\n"

        context_str = "\n" + ("=" * 50) + "\n" + "\n".join(context_blocks) + ("=" * 50)

        history_str = ""
        if history and len(history) > 0:
            hist_lines = []
            for h in history[-6:]:
                role = "المستخدم" if h.get("role") == "user" else "المساعد"
                hist_lines.append(f"{role}: {h.get('content', '')}")
            history_str = "\nسياق المحادثة السابقة:\n" + "\n".join(hist_lines) + "\n"

        if detected_accent == "jordanian":
            lang_instruction = "لغة الإجابة: اللهجة الأردنية الودية والواضحة — خاطب المهندس بأسلوب نقابي حميمي ومتعاطف."
        elif detected_accent == "english":
            lang_instruction = "Response language: Clear, professional English. Use formal tone appropriate for an engineering association."
        else:
            lang_instruction = "لغة الإجابة: العربية الفصحى الرسمية — أسلوب واضح ومحترف يليق بنقابة مهنية."

        system_prompt = (
            f"أنت مساعد ذكي متخصص في شؤون نقابة المهندسين الأردنيين (Jordan Engineers Association — JEA).\n"
            f"مهمتك: قراءة المصادر المرفقة بعناية، استيعابها، والإجابة بطريقة تركيبية منطقية.\n"
            f"لديك {len(included_sources)} مصدر من قاعدة المعرفة لهذا السؤال.\n\n"
            f"أسلوب الإجابة المطلوب:\n"
            f"• ابدأ بجملة تمهيدية واحدة موجزة تحدد محور الإجابة.\n"
            f"• اقرأ جميع المصادر واستخلص كل المعلومات ذات الصلة بالسؤال.\n"
            f"• اربط المعلومات من مصادر مختلفة إن تكاملت، واذكر رقم المصدر [المصدر N] عند كل معلومة.\n"
            f"• اختم بجملة خاتمة تلخّص الفكرة الرئيسية أو توجّه المستخدم للخطوة التالية عند الحاجة.\n\n"
            f"قواعد صارمة:\n"
            f"1. لا تخترع أي معلومة. كل فكرة يجب أن تكون موجودة في المصادر المرفقة.\n"
            f"2. إذا لم تجد إجابة في المصادر، قل ذلك صراحةً بدلاً من التكهن.\n"
            f"3. يُحظر استخدام رموز Markdown (* أو ** أو ##). استخدم الأرقام والنقاط النصية فقط.\n"
            f"4. {lang_instruction}\n"
        )

        full_prompt = (
            f"{system_prompt}\n"
            f"{'=' * 60}\n"
            f"المصادر من قاعدة المعرفة:\n{context_str}\n"
            f"{'=' * 60}\n"
            f"{history_str}"
            f"سؤال المستخدم (أجب عنه فقط من المصادر أعلاه): {query}\n"
            f"{'=' * 60}\n"
            f"الإجابة:"
        )

        # Stream tokens from Ollama
        stream_success = False
        models_to_try = [self.ollama_model, "qwen2.5:7b", "ministral-3:3b-instruct-2512-q4_K_M"]
        for model_tag in models_to_try:
            ollama_payload = {
                "model": model_tag,
                "prompt": full_prompt,
                "stream": True,
                "keep_alive": OLLAMA_KEEP_ALIVE,
                "options": {
                    "temperature": 0.1,
                    "num_predict": NUM_PREDICT,
                    "num_ctx": NUM_CTX
                }
            }

            try:
                res = requests.post(self.ollama_url, json=ollama_payload, stream=True, timeout=(5.0, 300.0))
                if res.status_code == 200:
                    for line in res.iter_lines():
                        if line:
                            chunk = json.loads(line.decode('utf-8'))
                            token = chunk.get("response", "")
                            if token:
                                token_clean = token.replace("*", "").replace("#", "")
                                yield f"data: {json.dumps({'type': 'token', 'token': token_clean}, ensure_ascii=False)}\n\n"
                            if chunk.get("done", False):
                                stream_success = True
                                break
                    if stream_success:
                        break
            except Exception as e:
                print(f"[Streaming Error] model '{model_tag}': {e}")

        total_time = round(time.time() - start_time, 2)
        yield f"data: {json.dumps({'type': 'done', 'time_taken': total_time}, ensure_ascii=False)}\n\n"

