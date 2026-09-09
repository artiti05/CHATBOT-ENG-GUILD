import json
import logging
import re
from typing import Any, Dict, Generator, List, Optional

from src.rag_chatbot_engine import RAGChatbotEngine

logger = logging.getLogger("rag_engine")


def clean_formatting(text: str) -> str:
    if not text:
        return ""
    cleaned = re.sub(r'\*+', '', text)
    cleaned = re.sub(r'^#+\s*', '', cleaned, flags=re.MULTILINE)
    return cleaned.strip()


class KnowledgeRetriever:
    """Backward compatibility wrapper around RetrieverAgent."""

    def __init__(self):
        from src.pipeline.stage_02_retrieve.retriever_agent import RetrieverAgent
        self.agent = RetrieverAgent()
        self.collection = getattr(self.agent, "collection", None)

    def retrieve(self, query_text: str, top_k: int = 15) -> List[Dict[str, Any]]:
        results = self.agent.retrieve_hybrid(query_text, top_k=top_k)
        normalized = []
        for r in results:
            if isinstance(r, dict):
                normalized.append({
                    "title": r.get("source_id") or r.get("title") or r.get("document", "وثيقة نقابية"),
                    "text": r.get("chunk_text") or r.get("text") or r.get("content", ""),
                    "score": r.get("score") or r.get("final_score", 0),
                    "final_score": r.get("final_score") or r.get("score", 0),
                })
            else:
                normalized.append({"text": str(r), "title": "وثيقة نقابية"})
        return normalized


class RAGChatbot:
    """Wrapper around RAGChatbotEngine for API & CLI with backend integration."""

    def __init__(self, provider: str = "ollama"):
        self.engine = RAGChatbotEngine()

    @property
    def cache_agent(self):
        return self.engine.cache

    def answer_question(
        self,
        query: str,
        history: List[Dict[str, str]] = None,
        top_k: int = 15,
        session_id: Optional[str] = None,
        user: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        res = self.engine.process_query(
            user_query=query,
            history=history,
            session_id=session_id,
        )
        ans = res.get("answer", "")
        if isinstance(ans, (tuple, list)):
            ans = ans[0] if ans else ""

        # Check for unanswerable / escalation trigger
        route_info = res.get("route") or {}
        route = route_info.get("route") if isinstance(route_info, dict) else str(route_info)
        conf_res = res.get("confidence") or {}

        is_unanswerable = (
            not ans
            or "لم أتمكن من العثور" in ans
            or route == "DETERMINISTIC"
            or (
                isinstance(conf_res, dict)
                and conf_res.get("confidence_level") == "LOW"
                and not res.get("sources")
            )
        )

        ticket_id = None
        escalated = False

        if is_unanswerable and (session_id or user):
            try:
                from src.services.backend_client import BackendTicketingClient

                client = BackendTicketingClient()
                phone = user.get("phone") if user else None
                user_name = user.get("name") if user else "مستفيد"

                ticket_res = client.create_internal_ticket(
                    title=f"استفسار نقابي - {user_name}",
                    content=f"استفسار المستفيد: {query}\n\nسياق الروبوت: لم يتم العثور على معلومات مؤكدة في قاعدة المعرفة.",
                    user_phone=phone,
                    session_id=session_id,
                    priority="MEDIUM",
                )
                if ticket_res.get("success"):
                    escalated = True
                    ticket_id = ticket_res.get("ticket_id")
                    ans = f"{ans}\n\nتم فتح تذكرة دعم فني لمتابعة استفسارك مع موظف النقابة المختص برقم مرجعي #{ticket_id}."
            except Exception as e:
                logger.error(f"[Ticket Escalation Error] {e}")

        # Normalize sources for JEA Backend KbRagProvider
        raw_sources = res.get("sources", [])[:top_k]
        normalized_sources = []
        for s in raw_sources:
            if isinstance(s, dict):
                normalized_sources.append({
                    "title": s.get("source_id") or s.get("title") or s.get("document", "وثيقة نقابية"),
                    "text": s.get("chunk_text") or s.get("text") or s.get("content", ""),
                    "score": s.get("score") or s.get("final_score", 0),
                    "final_score": s.get("final_score") or s.get("score", 0),
                })
            else:
                normalized_sources.append({"text": str(s), "title": "وثيقة نقابية"})

        return {
            "answer": ans,
            "sources": normalized_sources,
            "cache_hit": res.get("cache_hit", False),
            "escalated": escalated,
            "ticket_id": ticket_id,
            "metadata": res.get("profiling", {}),
        }

    def answer_question_stream(
        self,
        query: str,
        history: List[Dict[str, str]] = None,
        top_k: int = 15,
        session_id: Optional[str] = None,
        user: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Generator[str, None, None]:
        cached_res = self.engine.cache.get(query, session_id=session_id)
        if cached_res:
            ans = cached_res.get("answer", "")
            if isinstance(ans, (tuple, list)):
                ans = ans[0] if ans else ""
            meta_payload = json.dumps(
                {
                    "type": "meta",
                    "sources": cached_res.get("sources", [])[:top_k],
                    "cache_hit": True,
                    "profiling": {},
                },
                ensure_ascii=False,
            )
            yield f"data: {meta_payload}\n\n"

            if ans:
                for token in ans:
                    yield f"data: {json.dumps({'type': 'token', 'token': token}, ensure_ascii=False)}\n\n"

            done_payload = json.dumps(
                {
                    "type": "done",
                    "answer": ans,
                    "text_breakdown": "Cache Hit",
                },
                ensure_ascii=False,
            )
            yield f"data: {done_payload}\n\n"
            return

        query_obj = self.engine.nlp_pipeline.process(query, history=history)
        search_query = (
            query_obj.get("canonical_msa") or query_obj.get("normalized_text")
        )
        fused_cands = self.engine.retriever.retrieve_hybrid(
            search_query, top_k=15
        )
        reranked_cands = self.engine.reranker.apply_priority_boost(fused_cands)
        conf_res = self.engine.confidence_eval.evaluate(reranked_cands, query_obj)
        route_res = self.engine.router.route(query_obj, conf_res)
        route = route_res["route"]

        normalized_sources = []
        for s in reranked_cands[:top_k]:
            if isinstance(s, dict):
                normalized_sources.append({
                    "title": s.get("source_id") or s.get("title") or s.get("document", "وثيقة نقابية"),
                    "text": s.get("chunk_text") or s.get("text") or s.get("content", ""),
                    "score": s.get("score") or s.get("final_score", 0),
                    "final_score": s.get("final_score") or s.get("score", 0),
                })
            else:
                normalized_sources.append({"text": str(s), "title": "وثيقة نقابية"})

        meta_payload = json.dumps(
            {
                "type": "meta",
                "sources": normalized_sources,
                "cache_hit": False,
                "profiling": {},
            },
            ensure_ascii=False,
        )
        yield f"data: {meta_payload}\n\n"

        full_answer = ""
        if route == "DETERMINISTIC":
            full_answer = "يرجى زيارة بوابة نقابة المهندسين (jea.org.jo) للحصول على التفاصيل والخدمات الرسمية."
            for char in full_answer:
                yield f"data: {json.dumps({'type': 'token', 'token': char}, ensure_ascii=False)}\n\n"
        else:
            prompt_str, sources_used = self.engine.generator.build_prompt(
                query=query,
                standalone_query=query_obj.get("canonical_msa"),
                sources=reranked_cands,
                detected_accent=query_obj.get("language"),
            )
            if prompt_str:
                for token in self.engine.generator.generate_stream(prompt_str):
                    full_answer += token
                    yield f"data: {json.dumps({'type': 'token', 'token': token}, ensure_ascii=False)}\n\n"

            if not full_answer:
                gen_out = (
                    self.engine.generator.generate(prompt_str)
                    if prompt_str
                    else ""
                )
                full_answer = (
                    gen_out[0] if isinstance(gen_out, tuple) else gen_out
                )

        if not full_answer:
            full_answer = "لم أتمكن من العثور على معلومات دقيقة."

        done_payload = json.dumps(
            {
                "type": "done",
                "answer": full_answer,
                "text_breakdown": "Generated Answer",
            },
            ensure_ascii=False,
        )
        yield f"data: {done_payload}\n\n"
