import re
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

from config.settings import (
    CHROMA_PERSIST_DIR, CHROMA_COLLECTION_NAME, BGE_RERANKER_MODEL_NAME,
    LMSTUDIO_BASE_URL, LMSTUDIO_CHAT_MODEL, OLLAMA_URL, OLLAMA_VISION_MODEL, VLM_PROVIDER
)
from ingestion_pipeline import BGEM3Embedder

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
                print(f"[Reranker] Loading BGE Reranker model '{BGE_RERANKER_MODEL_NAME}'...")
                self.model = CrossEncoder(BGE_RERANKER_MODEL_NAME, max_length=512)
                print("[Reranker] Successfully loaded BGE Reranker.")
            except Exception as e:
                print(f"[Reranker Warning] Could not load BGE Reranker model: {e}")
                self.model = None

    def rerank(self, query: str, candidate_chunks: List[Dict[str, Any]], top_k: int = 10) -> List[Dict[str, Any]]:
        if not candidate_chunks:
            return []
        if not self.model:
            return candidate_chunks[:top_k]

        pairs = [[query, chunk.get("text", "")] for chunk in candidate_chunks]
        scores = self.model.predict(pairs)

        for i, score in enumerate(scores):
            candidate_chunks[i]["rerank_score"] = float(score)
            match_pct = min(max(int((float(score) + 4.0) / 8.0 * 100), 50), 99)
            candidate_chunks[i]["similarity_score"] = match_pct

        ranked_chunks = sorted(candidate_chunks, key=lambda x: x.get("rerank_score", 0.0), reverse=True)
        for rank, chunk in enumerate(ranked_chunks[:top_k], 1):
            chunk["rank"] = rank

        return ranked_chunks[:top_k]

# ---------------------------------------------------------------------------
# 2. HYBRID VECTOR SEARCH KNOWLEDGE RETRIEVER
# ---------------------------------------------------------------------------
class KnowledgeRetriever:
    def __init__(self):
        self.client = chromadb.PersistentClient(path=str(CHROMA_PERSIST_DIR), settings=ChromaSettings(anonymized_telemetry=False))
        self.collection = self.client.get_or_create_collection(name=CHROMA_COLLECTION_NAME)
        self.embedder = BGEM3Embedder()
        self.reranker = BGEReranker()

    def retrieve(self, query_text: str, top_k: int = 10) -> List[Dict[str, Any]]:
        if not query_text or not query_text.strip():
            return []

        dense_vecs, _ = self.embedder.embed_texts([query_text])
        query_dense = dense_vecs[0]

        results = self.collection.query(
            query_embeddings=[query_dense],
            n_results=min(top_k * 3, 30), # Over-sample for cross-encoder reranking
            include=["documents", "metadatas", "distances"]
        )

        candidates = []
        if results and results.get("documents") and results["documents"][0]:
            docs = results["documents"][0]
            metas = results["metadatas"][0]
            dists = results["distances"][0]

            for i in range(len(docs)):
                dist = dists[i]
                sim_pct = int(max(0.0, (1.0 - dist)) * 100)
                meta = metas[i] if i < len(metas) else {}

                candidates.append({
                    "text": docs[i],
                    "title": meta.get("file_name", "وثيقة"),
                    "section_title": f"جزء {meta.get('chunk_index', 1)}",
                    "similarity_score": sim_pct,
                    "distance": dist,
                    "metadata": meta
                })

        return self.reranker.rerank(query_text, candidates, top_k=top_k)

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
        self.provider = (VLM_PROVIDER or "ollama").lower()
        self.ollama_url = OLLAMA_URL
        self.ollama_model = OLLAMA_VISION_MODEL
        self.lmstudio_url = f"{LMSTUDIO_BASE_URL.rstrip('/')}/chat/completions"
        self.lmstudio_model = LMSTUDIO_CHAT_MODEL

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

    def answer_question(self, query: str, history: Optional[List[Dict[str, str]]] = None, top_k: int = 10) -> Dict[str, Any]:
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

        context_blocks = []
        for src in sources:
            rank = src.get("rank", 1)
            title = src.get("title", "وثيقة")
            section = src.get("section_title", "")
            score = src.get("similarity_score", 0.0)
            raw_text = src.get("text", "").strip()
            text_snippet = raw_text[:400] + "..." if len(raw_text) > 400 else raw_text

            header = f"[المصدر {rank}] {title}"
            if section:
                header += f" — {section}"
            header += f" (نسبة التطابق: {score}%)"
            context_blocks.append(f"{header}\n{text_snippet}\n")

        context_str = "\n" + ("=" * 50) + "\n" + "\n".join(context_blocks) + ("=" * 50)

        history_str = ""
        if history and len(history) > 0:
            hist_lines = []
            for h in history[-6:]:
                role = "المستخدم" if h.get("role") == "user" else "المساعد"
                hist_lines.append(f"{role}: {h.get('content', '')}")
            history_str = "\nسياق المحادثة السابقة:\n" + "\n".join(hist_lines) + "\n"

        accent_instruction = ""
        if detected_accent == "jordanian":
            accent_instruction = "6. المستخدم سأل باللهجة الأردنية. صغ الإجابة النهائية باللهجة الأردنية النقابية الودية والواضحة."
        elif detected_accent == "english":
            accent_instruction = "6. The user asked in English. Provide the final response in clear professional English."
        else:
            accent_instruction = "6. صغ الإجابة باللغة العربية الفصحى الرسمية السليمة."

        system_prompt = (
            "أنت مساعد ذكي مخصص لنقابة المهندسين الأردنيين (Jordan Engineers Association).\n"
            "مهمتك هي الإجابة عن سؤال المستخدم بدقة وموضوعية اعتماداً حصرياً على المصادر العشرة المرفقة أدناه.\n"
            "تعليمات التنسيق الهامة:\n"
            "1. استخرج المعلومات المباشرة والإحصائيات والأنظمة والتعليمات ذات الصلة بالسؤال.\n"
            "2. اذكر المصادر المستخدمة في إجابتك باستخدام التنسيق [المصدر N: اسم الوثيقة].\n"
            "3. لا تخترع أو تتكهن بأي معلومات غير موجودة في المصادر.\n"
            "4. يمنع منعاً باتاً استخدام رموز النجوم Markdown (مثل **نص** أو *نص*) في الإجابة إطلاقاً. اكتب العناوين والقوائم بنقاط وأرقام نظيفة وواضحة بدون أي نجوم.\n"
            "5. صغ القوائم بشكل نظيف ومقروء مثل: 1. اسم الخدمة: التوضيح الشامل.\n"
            f"{accent_instruction}"
        )

        full_prompt = f"{system_prompt}\n\n{history_str}سؤال المستخدم الحالي: {query}\n\nالمصادر العشرة المتاحة (Top 10 Sources):\n{context_str}"

        generated_answer = ""
        llm_success = False

        # Direct Ollama API Call (Primary Default)
        if self.provider == "ollama" or not llm_success:
            models_to_try = [self.ollama_model, "qwen2.5:7b", "qwen2.5vl:7b", "qwen2.5-vl:7b"]
            for model_tag in models_to_try:
                try:
                    ollama_payload = {
                        "model": model_tag,
                        "prompt": full_prompt,
                        "stream": False,
                        "options": {
                            "temperature": 0.3,
                            "num_predict": 1500,
                        }
                    }
                    res = requests.post(self.ollama_url, json=ollama_payload, timeout=(3.0, 30.0))
                    if res.status_code == 200:
                        content = res.json().get("response", "").strip()
                        if content:
                            generated_answer = content
                            llm_success = True
                            break
                except Exception as o_err:
                    print(f"[RAG Chatbot Warning] Ollama call ('{model_tag}') failed: {o_err}")

        # Secondary fallback if LM Studio is explicitly requested or Ollama call failed
        if not llm_success and self.provider == "lmstudio":
            try:
                payload = {
                    "model": self.lmstudio_model,
                    "messages": [{"role": "user", "content": full_prompt}],
                    "temperature": 0.3,
                    "max_tokens": 1500
                }
                res = requests.post(self.lmstudio_url, json=payload, timeout=(3.0, 30.0))
                if res.status_code == 200:
                    choices = res.json().get("choices", [])
                    if choices:
                        content = choices[0].get("message", {}).get("content", "")
                        if content:
                            generated_answer = content.strip()
                            llm_success = True
            except Exception as err:
                print(f"[RAG Chatbot Warning] LM Studio call failed ({type(err).__name__}).")

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
            "sources": sources,
            "llm_connected": llm_success,
            "time_taken": round(time.time() - start_time, 2)
        }
