import time
import json
from typing import Dict, Any, List, Tuple, Optional, TypedDict

from core.agents.cache_agent import SemanticCacheAgent
from core.agents.rewriter_agent import DialectRewriterAgent
from core.agents.retriever_agent import RetrieverAgent
from core.agents.reranker_agent import RerankerAgent
from core.agents.generator_agent import ResponseGeneratorAgent
from core.agents.verifier_agent import VerifierAgent


class AgentState(TypedDict, total=False):
    query: str
    history: List[Dict[str, str]]
    standalone_query: str
    detected_accent: str
    sources: List[Dict[str, Any]]
    prompt: str
    answer: str
    groundedness_score: float
    confidence_score: float
    reflexion_count: int
    cache_hit: bool
    llm_connected: bool
    metadata: Dict[str, Any]
    time_taken: float


class RAGStateGraph:
    """
    Agentic Multi-Agent StateGraph Orchestrator (Category 4).
    Coordinates 6 specialized subagents and manages dynamic state transitions,
    conditional routing, and CRAG self-correction reflexion loops.
    """

    def __init__(self):
        self.cache_agent = SemanticCacheAgent()
        self.rewriter_agent = DialectRewriterAgent()
        self.retriever_agent = RetrieverAgent()
        self.reranker_agent = RerankerAgent()
        self.generator_agent = ResponseGeneratorAgent()
        self.verifier_agent = VerifierAgent()

    def _detect_accent(self, query: str) -> str:
        q = query.strip()
        import re
        q_lower = q.lower()
        if any(kw in q_lower for kw in ["english", "in english", "answer in english", "reply in english", "translate to english"]):
            return "english"

        latin_chars = len(re.findall(r'[a-zA-Z]', q))
        arabic_chars = len(re.findall(r'[\u0600-\u06FF]', q))

        if latin_chars >= 3 and latin_chars >= arabic_chars:
            return "english"
        elif any(kw in q for kw in ["شو", "بدي", "عشان", "قديش", "وين", "ايش", "كيف بقدر"]):
            return "jordanian"
        return "msa"

    def run(self, query: str, history: Optional[List[Dict[str, str]]] = None, top_k: int = 5) -> Dict[str, Any]:
        """Executes full StateGraph graph workflow synchronously."""
        start_time = time.time()
        q = query.strip() if query else ""
        if not q:
            return {"query": query, "answer": "الرجاء إدخال سؤال للبحث والإجابة.", "sources": [], "time_taken": 0.0}

        hist = history if history else []


        # NODE 1: Semantic Cache Intercept
        if not hist or len(hist) == 0:
            cached_res = self.cache_agent.get(q)
            if cached_res:
                return cached_res

        # NODE 2: Dialect Rewriting & History Synthesis
        standalone_q, rewrite_meta = self.rewriter_agent.rewrite_query(q, history=hist)
        accent = self._detect_accent(q)

        # NODE 3: Hybrid Retrieval (Dense BGE-M3 + Sparse BM25 + RRF k=60)
        candidates = self.retriever_agent.retrieve_hybrid(standalone_q, top_k=min(top_k * 2, 30))

        # NODE 4: Reranking & Strict Cutoff Floor
        reranked_sources = self.reranker_agent.rerank(standalone_q, candidates, top_k=top_k)

        # Apply Parent Context Swap
        for chunk in reranked_sources:
            if "parent_text" in chunk and chunk["parent_text"]:
                chunk["child_text"] = chunk.get("text", "")
                chunk["text"] = chunk["parent_text"]

        # Evaluate CRAG Initial Confidence
        crag_eval = self.verifier_agent.evaluate_retrieval_confidence(reranked_sources)
        reflexion_count = 0

        # REFLEXION LOOP 1: Initial Query Expansion if retrieval confidence is low
        if crag_eval["confidence_low"]:
            expansions = self.verifier_agent.generate_query_expansions(standalone_q)
            existing_ids = {s.get("chunk_id") for s in reranked_sources}
            for exp_q in expansions:
                extra_cands = self.retriever_agent.retrieve_hybrid(exp_q, top_k=5)
                extra_reranked = self.reranker_agent.rerank(exp_q, extra_cands, top_k=5)
                for s in extra_reranked:
                    if s.get("chunk_id") not in existing_ids:
                        reranked_sources.append(s)
                        existing_ids.add(s.get("chunk_id"))
            reranked_sources = sorted(reranked_sources, key=lambda x: x.get("similarity_score", 0), reverse=True)[:top_k]
            reflexion_count += 1

        if not reranked_sources:
            no_sources_msg = {
                "jordanian": "للأسف ما لقيت وثائق أو معلومات مباشرة بتخص سؤالك بقاعدة المعرفة حالياً.",
                "english": "Sorry, no relevant documents were found in the knowledge base.",
                "msa": "لم يتم العثور على وثائق أو مصادر مرتبطة بسؤالك في قاعدة المعرفة."
            }
            return {
                "query": q,
                "standalone_query": standalone_q,
                "answer": no_sources_msg.get(accent, no_sources_msg["msa"]),
                "sources": [],
                "cache_hit": False,
                "rewrite_metadata": rewrite_meta,
                "crag_metadata": crag_eval,
                "time_taken": round(time.time() - start_time, 2)
            }

        # NODE 5: Grounded Response Generation
        prompt_str, included_sources = self.generator_agent.build_prompt(q, standalone_q, reranked_sources, detected_accent=accent)
        if not included_sources:
            return {
                "query": q,
                "standalone_query": standalone_q,
                "answer": "لا تتوفر معلومات كافية في قاعدة المعرفة للإجابة على هذا السؤال بدقة.",
                "sources": [],
                "cache_hit": False,
                "time_taken": round(time.time() - start_time, 2)
            }

        generated_answer, llm_connected = self.generator_agent.generate(prompt_str)
        if not generated_answer:
            generated_answer = f"تم العثور على {len(included_sources)} مصادر مرتبطة بسؤالك، ولكن تعذّر الاتصال بنموذج التوليد المحلي حالياً."

        # NODE 6: Verifier & Groundedness Reflexion
        verification = self.verifier_agent.verify_answer_groundedness(generated_answer, included_sources)

        # REFLEXION LOOP 2: Secondary re-retrieval if groundedness is low (< 65%)
        if not verification.get("is_grounded") and verification.get("score", 0) < 65.0 and reflexion_count < 2:
            expansions = self.verifier_agent.generate_query_expansions(standalone_q)
            existing_ids = {s.get("chunk_id") for s in included_sources}
            for exp_q in expansions:
                extra_cands = self.retriever_agent.retrieve_hybrid(exp_q, top_k=5)
                extra_reranked = self.reranker_agent.rerank(exp_q, extra_cands, top_k=5)
                for s in extra_reranked:
                    if s.get("chunk_id") not in existing_ids:
                        included_sources.append(s)
                        existing_ids.add(s.get("chunk_id"))

            # Re-generate with updated sources
            prompt_str_2, included_sources = self.generator_agent.build_prompt(q, standalone_q, included_sources, detected_accent=accent)
            generated_answer_2, llm_conn_2 = self.generator_agent.generate(prompt_str_2)
            if generated_answer_2:
                generated_answer = generated_answer_2
                llm_connected = llm_conn_2
                verification = self.verifier_agent.verify_answer_groundedness(generated_answer, included_sources)
            reflexion_count += 1

        final_answer = verification.get("cleaned_answer", generated_answer)

        res_payload = {
            "query": q,
            "standalone_query": standalone_q,
            "answer": final_answer,
            "sources": included_sources,
            "llm_connected": llm_connected,
            "cache_hit": False,
            "rewrite_metadata": rewrite_meta,
            "crag_metadata": crag_eval,
            "verification_metadata": verification,
            "reflexion_count": reflexion_count,
            "time_taken": round(time.time() - start_time, 2)
        }

        # Cache payload if single turn
        if not hist or len(hist) == 0:
            self.cache_agent.put(q, res_payload)

        return res_payload

    def run_stream(self, query: str, history: Optional[List[Dict[str, str]]] = None, top_k: int = 5):
        """Executes real-time SSE streaming StateGraph workflow."""

        start_time = time.time()
        q = query.strip() if query else ""
        if not q:
            yield "data: [DONE]\n\n"
            return

        hist = history if history else []

        # NODE 1: Semantic Cache Check
        if not hist or len(hist) == 0:
            cached_res = self.cache_agent.get(q)
            if cached_res:
                meta_event = {
                    "type": "meta",
                    "query": q,
                    "standalone_query": cached_res.get("standalone_query", q),
                    "sources": cached_res.get("sources", []),
                    "rewrite_metadata": cached_res.get("rewrite_metadata"),
                    "crag_metadata": cached_res.get("crag_metadata"),
                    "cache_hit": True,
                    "time_taken_ms": cached_res.get("time_taken_ms", 0.0)
                }
                yield f"data: {json.dumps(meta_event, ensure_ascii=False)}\n\n"
                answer_text = cached_res.get("answer", "")
                token_event = {"type": "token", "token": answer_text}
                yield f"data: {json.dumps(token_event, ensure_ascii=False)}\n\n"
                yield "data: [DONE]\n\n"
                return

        # NODE 2: Dialect Rewriting
        standalone_q, rewrite_meta = self.rewriter_agent.rewrite_query(q, history=hist)
        accent = self._detect_accent(q)

        # NODE 3: Hybrid Retrieval
        candidates = self.retriever_agent.retrieve_hybrid(standalone_q, top_k=min(top_k * 2, 30))

        # NODE 4: Reranking
        reranked_sources = self.reranker_agent.rerank(standalone_q, candidates, top_k=top_k)
        for chunk in reranked_sources:
            if "parent_text" in chunk and chunk["parent_text"]:
                chunk["child_text"] = chunk.get("text", "")
                chunk["text"] = chunk["parent_text"]

        crag_eval = self.verifier_agent.evaluate_retrieval_confidence(reranked_sources)
        if crag_eval["confidence_low"]:
            expansions = self.verifier_agent.generate_query_expansions(standalone_q)
            existing_ids = {s.get("chunk_id") for s in reranked_sources}
            for exp_q in expansions:
                extra_cands = self.retriever_agent.retrieve_hybrid(exp_q, top_k=5)
                extra_reranked = self.reranker_agent.rerank(exp_q, extra_cands, top_k=5)
                for s in extra_reranked:
                    if s.get("chunk_id") not in existing_ids:
                        reranked_sources.append(s)
                        existing_ids.add(s.get("chunk_id"))
            reranked_sources = sorted(reranked_sources, key=lambda x: x.get("similarity_score", 0), reverse=True)[:top_k]

        prompt_str, included_sources = self.generator_agent.build_prompt(q, standalone_q, reranked_sources, detected_accent=accent)

        # Yield Meta Event immediately
        meta_event = {
            "type": "meta",
            "query": q,
            "standalone_query": standalone_q,
            "sources": included_sources,
            "rewrite_metadata": rewrite_meta,
            "crag_metadata": crag_eval,
            "cache_hit": False,
            "time_taken": round(time.time() - start_time, 2)
        }
        yield f"data: {json.dumps(meta_event, ensure_ascii=False)}\n\n"

        if not included_sources:
            no_info = "لا تتوفر معلومات كافية في قاعدة المعرفة للإجابة على هذا السؤال بدقة."
            yield f"data: {json.dumps({'type': 'token', 'token': no_info}, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"
            return

        # Real-time token streaming from ResponseGeneratorAgent
        stream_gen = self.generator_agent.generate_stream(prompt_str)
        streamed_tokens = []
        for token in stream_gen:
            streamed_tokens.append(token)
            yield f"data: {json.dumps({'type': 'token', 'token': token}, ensure_ascii=False)}\n\n"

        yield "data: [DONE]\n\n"
