import json
import re
from typing import Any, Dict, Generator, List

from src.rag_chatbot_engine import RAGChatbotEngine


def clean_formatting(text: str) -> str:
    if not text:
        return ""
    cleaned = re.sub(r'\*+', '', text)
    cleaned = re.sub(r'^#+\s*', '', cleaned, flags=re.MULTILINE)
    return cleaned.strip()


class RAGChatbot:
    """Wrapper around RAGChatbotEngine for API & CLI backward compatibility."""

    def __init__(self, provider: str = "ollama"):
        self.engine = RAGChatbotEngine()

    @property
    def cache_agent(self):
        return self.engine.cache

    def answer_question(self, query: str, history: List[Dict[str, str]] = None, top_k: int = 15, **kwargs) -> Dict[str, Any]:
        res = self.engine.process_query(query)
        ans = res.get("answer", "")
        if isinstance(ans, (tuple, list)):
            ans = ans[0] if ans else ""
        return {
            "answer": ans,
            "sources": res["sources"][:top_k],
            "cache_hit": res.get("cache_hit", False),
            "metadata": res.get("profiling", {})
        }

    def answer_question_stream(self, query: str, history: List[Dict[str, str]] = None, top_k: int = 15, **kwargs) -> Generator[str, None, None]:
        cached_res = self.engine.cache.get(query)
        if cached_res:
            ans = cached_res.get("answer", "")
            if isinstance(ans, (tuple, list)):
                ans = ans[0] if ans else ""
            meta_payload = json.dumps({
                "type": "meta",
                "sources": cached_res.get("sources", [])[:top_k],
                "cache_hit": True,
                "profiling": {}
            }, ensure_ascii=False)
            yield f"data: {meta_payload}\n\n"

            if ans:
                for token in ans:
                    yield f"data: {json.dumps({'type': 'token', 'token': token}, ensure_ascii=False)}\n\n"

            done_payload = json.dumps({
                "type": "done",
                "answer": ans,
                "text_breakdown": "Cache Hit"
            }, ensure_ascii=False)
            yield f"data: {done_payload}\n\n"
            return

        query_obj = self.engine.nlp_pipeline.process(query)
        search_query = query_obj.get("canonical_msa") or query_obj.get("normalized_text")
        fused_cands = self.engine.retriever.retrieve_hybrid(search_query, top_k=15)
        reranked_cands = self.engine.reranker.apply_priority_boost(fused_cands)
        conf_res = self.engine.confidence_eval.evaluate(reranked_cands, query_obj)
        route_res = self.engine.router.route(query_obj, conf_res)
        route = route_res["route"]

        meta_payload = json.dumps({
            "type": "meta",
            "sources": reranked_cands[:top_k],
            "cache_hit": False,
            "profiling": {}
        }, ensure_ascii=False)
        yield f"data: {meta_payload}\n\n"

        full_answer = ""
        if route == "DETERMINISTIC":
            full_answer = "يرجى زيارة بوابة نقابة المهندسين (jea.org.jo) للحصول على التفاصيل والخدمات الرسمية."
            for char in full_answer:
                yield f"data: {json.dumps({'type': 'token', 'token': char}, ensure_ascii=False)}\n\n"

        else:
            prompt_str, sources_used = self.engine.generator.build_prompt(
                query=query,
                standalone_query=query_obj["canonical_msa"],
                sources=reranked_cands,
                detected_accent=query_obj["language"]
            )
            if prompt_str:
                for token in self.engine.generator.generate_stream(prompt_str):
                    full_answer += token
                    yield f"data: {json.dumps({'type': 'token', 'token': token}, ensure_ascii=False)}\n\n"

            if not full_answer:
                gen_out = self.engine.generator.generate(prompt_str) if prompt_str else ""
                full_answer = gen_out[0] if isinstance(gen_out, tuple) else gen_out
                if not full_answer:
                    full_answer = "لم أتمكن من العثور على معلومات دقيقة في موارد النقابة."
                for token in full_answer:
                    yield f"data: {json.dumps({'type': 'token', 'token': token}, ensure_ascii=False)}\n\n"

        if full_answer:
            self.engine.cache.put(query, full_answer, reranked_cands[:3])

        done_payload = json.dumps({
            "type": "done",
            "answer": full_answer,
            "text_breakdown": ""
        }, ensure_ascii=False)
        yield f"data: {done_payload}\n\n"


class KnowledgeRetriever:
    """Knowledge retriever providing fast retrieval-only search without LLM generation overhead."""
    def __init__(self):
        self.engine = RAGChatbotEngine()

    @property
    def collection(self):
        return self.engine.retriever.collection

    def search(self, query: str, top_k: int = 5, **kwargs) -> List[Dict[str, Any]]:
        if not query or not query.strip():
            return []
        query_obj = self.engine.nlp_pipeline.process(query)
        search_query = query_obj.get("canonical_msa") or query_obj.get("normalized_text")
        fused_cands = self.engine.retriever.retrieve_hybrid(search_query, top_k=top_k * 2)
        if hasattr(self.engine.reranker, "rerank"):
            reranked_cands = self.engine.reranker.rerank(search_query, fused_cands)
        else:
            reranked_cands = self.engine.reranker.apply_priority_boost(fused_cands)
        return reranked_cands[:top_k]

    def retrieve(self, query_text: str = "", query: str = "", top_k: int = 5, **kwargs) -> List[Dict[str, Any]]:
        q = query_text or query
        return self.search(q, top_k=top_k)
