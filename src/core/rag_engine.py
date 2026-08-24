import os
import re
import json
from typing import List, Dict, Any, Tuple, AsyncGenerator, Optional
from pathlib import Path

import asyncio

from src.rag_chatbot_engine import RAGChatbotEngine
from src.pipeline.stage_05_ticket.ticket_agent import PROMPT_USER_INTENT, PROMPT_LOW_CONFIDENCE


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

    async def answer_question(self, query: str, history: List[Dict[str, str]] = None, top_k: int = 15, **kwargs) -> Dict[str, Any]:
        res = await self.engine.process_query(query, history=history)
        ans = res.get("answer", "")
        if isinstance(ans, (tuple, list)):
            ans = ans[0] if ans else ""
        return {
            "answer": ans,
            "sources": res["sources"][:top_k],
            "cache_hit": res.get("cache_hit", False),
            "metadata": res.get("profiling", {})
        }

    async def answer_question_stream(self, query: str, history: List[Dict[str, str]] = None, top_k: int = 15, **kwargs) -> AsyncGenerator[str, None]:
        hist = history or []

        # NODE 0: Ticket intake intercept (mirrors process_query) -- neither
        # trigger touches the knowledge base, so it runs before the cache.
        pending = self.engine.ticket_agent.is_awaiting_contact(hist)
        if pending:
            reason, issue_text = pending
            ticket_id = await asyncio.to_thread(
                self.engine.ticket_agent.create_ticket, reason, issue_text, query
            )
            answer_text = self.engine.ticket_agent.build_confirmation(ticket_id)
            yield f"data: {json.dumps({'type': 'meta', 'sources': [], 'cache_hit': False, 'profiling': {}}, ensure_ascii=False)}\n\n"
            yield f"data: {json.dumps({'type': 'token', 'token': answer_text}, ensure_ascii=False)}\n\n"
            yield f"data: {json.dumps({'type': 'done', 'answer': answer_text, 'text_breakdown': 'Ticket Filed'}, ensure_ascii=False)}\n\n"
            return

        if await self.engine.ticket_agent.detect_intent(query, hist):
            yield f"data: {json.dumps({'type': 'meta', 'sources': [], 'cache_hit': False, 'profiling': {}}, ensure_ascii=False)}\n\n"
            yield f"data: {json.dumps({'type': 'token', 'token': PROMPT_USER_INTENT}, ensure_ascii=False)}\n\n"
            yield f"data: {json.dumps({'type': 'done', 'answer': PROMPT_USER_INTENT, 'text_breakdown': 'Ticket Intent'}, ensure_ascii=False)}\n\n"
            return

        cached_res = self.engine.cache.get(query) if not hist else None
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
        fused_cands = self.engine.retriever.retrieve_hybrid(query_obj["normalized_text"], top_k=15)
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
            primary_intent = query_obj["intent"][0] if query_obj["intent"] else "MEMBERSHIP"
            tmpl_ans = StructuredAnswerTemplates.get_template_answer(primary_intent)
            full_answer = tmpl_ans or "يرجى زيارة بوابة نقابة المهندسين للحصول على التفاصيل."
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
                async for token in self.engine.generator.generate_stream(prompt_str):
                    full_answer += token
                    yield f"data: {json.dumps({'type': 'token', 'token': token}, ensure_ascii=False)}\n\n"

            if not full_answer:
                gen_out = await self.engine.generator.generate(prompt_str) if prompt_str else ""
                full_answer = gen_out[0] if isinstance(gen_out, tuple) else gen_out
                if not full_answer:
                    full_answer = PROMPT_LOW_CONFIDENCE
                for token in full_answer:
                    yield f"data: {json.dumps({'type': 'token', 'token': token}, ensure_ascii=False)}\n\n"

        if full_answer and not hist and full_answer != PROMPT_LOW_CONFIDENCE:
            self.engine.cache.put(query, full_answer, reranked_cands[:3])

        done_payload = json.dumps({
            "type": "done",
            "answer": full_answer,
            "text_breakdown": ""
        }, ensure_ascii=False)
        yield f"data: {done_payload}\n\n"


class KnowledgeRetriever:
    """Knowledge retriever alias for API dependencies."""
    def __init__(self):
        self.engine = RAGChatbotEngine()

    @property
    def collection(self):
        return self.engine.retriever.collection

    async def search(self, query: str, top_k: int = 5, **kwargs) -> List[Dict[str, Any]]:
        res = await self.engine.process_query(query)
        return res.get("sources", [])[:top_k]

    async def retrieve(self, query_text: str = "", query: str = "", top_k: int = 5, **kwargs) -> List[Dict[str, Any]]:
        q = query_text or query
        res = await self.engine.process_query(q)
        return res.get("sources", [])[:top_k]
