import time
import asyncio
from typing import Dict, Any, List, Optional

from src.monitoring.tracing import RequestTracer
from src.monitoring.profiler import PipelineProfiler
from src.monitoring.diagnostic_report import DiagnosticAnalyzer

from src.pipeline.stage_01_understand.query_preprocessor import QueryPreprocessor
from src.cache_db.semantic_cache import SemanticCache
from src.pipeline.stage_02_retrieve.bm25_retriever import BM25Retriever
from src.pipeline.stage_02_retrieve.rrf_fusion import RRFFusion
from src.pipeline.stage_03_verify_rerank.reranker import PriorityReranker
from src.pipeline.stage_03_verify_rerank.confidence import DeterministicConfidenceEvaluator
from src.pipeline.stage_04_answer.answer_router import AnswerRouter
from src.pipeline.stage_04_answer.generator_agent import ResponseGeneratorAgent
from src.pipeline.stage_02_retrieve.retriever_agent import RetrieverAgent
from src.pipeline.stage_05_ticket.ticket_agent import (
    TicketIntakeAgent, PROMPT_USER_INTENT, PROMPT_LOW_CONFIDENCE
)


class RAGChatbotEngine:
    """
    Central Orchestrator Engine for Eng-Guild Chatbot.
    Implements the complete Normalize → Understand → Retrieve → Verify → Answer architecture
    with first-class 13-stage diagnostic profiling.
    """

    def __init__(self):
        self.nlp_pipeline = QueryPreprocessor()
        self.cache = SemanticCache()
        self.bm25 = BM25Retriever()
        self.rrf = RRFFusion()
        self.reranker = PriorityReranker()
        self.retriever = RetrieverAgent()
        self.confidence_eval = DeterministicConfidenceEvaluator()
        self.router = AnswerRouter()
        self.generator = ResponseGeneratorAgent()
        self.ticket_agent = TicketIntakeAgent()

    async def process_query(self, user_query: str,
                            history: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
        request_id = RequestTracer.generate_request_id()
        profiler = PipelineProfiler(request_id)
        hist = history or []

        # 0. Ticket Intake Stage — runs before cache/retrieval since neither
        # trigger needs the knowledge base. Resolution after intake is a purely
        # human process; the AI's only job here is collecting the details.
        with profiler.time_stage("ticket_intake"):
            pending = self.ticket_agent.is_awaiting_contact(hist)
            if pending:
                reason, issue_text = pending
                ticket_id = await asyncio.to_thread(
                    self.ticket_agent.create_ticket, reason, issue_text, user_query
                )
                answer_text = self.ticket_agent.build_confirmation(ticket_id)
                report = profiler.generate_report()
                profiler.write_request_log(user_query, answer_text, cache_hit=False, route="TICKET_FILED")
                return {
                    "request_id": request_id,
                    "query": {"original": user_query},
                    "answer": answer_text,
                    "sources": [],
                    "cache_hit": False,
                    "route": {"route": "TICKET_FILED", "ticket_id": ticket_id},
                    "profiling": report,
                    "text_breakdown": profiler.format_text_breakdown()
                }

            if await self.ticket_agent.detect_intent(user_query, hist):
                report = profiler.generate_report()
                profiler.write_request_log(user_query, PROMPT_USER_INTENT, cache_hit=False, route="TICKET_INTENT")
                return {
                    "request_id": request_id,
                    "query": {"original": user_query},
                    "answer": PROMPT_USER_INTENT,
                    "sources": [],
                    "cache_hit": False,
                    "route": {"route": "TICKET_INTENT"},
                    "profiling": report,
                    "text_breakdown": profiler.format_text_breakdown()
                }

        # 1. Cache Stage
        with profiler.time_stage("cache"):
            cached_res = self.cache.get(user_query) if not hist else None
            if cached_res:
                report = profiler.generate_report()
                profiler.write_request_log(user_query, cached_res["answer"], cache_hit=True, route="CACHE")
                return {
                    "request_id": request_id,
                    "query": {"original": user_query},
                    "answer": cached_res["answer"],
                    "sources": cached_res.get("sources", []),
                    "cache_hit": True,
                    "profiling": report,
                    "text_breakdown": profiler.format_text_breakdown()
                }

        # 2. Language Detection Stage
        with profiler.time_stage("language_detection"):
            lang_res = self.nlp_pipeline.detect_language(user_query)

        # 3. Normalization Stage
        with profiler.time_stage("normalization"):
            norm_text = self.nlp_pipeline.normalize_arabic(user_query)

        # 4. MSA Conversion & Multi-Representation Stage
        with profiler.time_stage("msa_conversion"):
            query_obj = self.nlp_pipeline.process(user_query)

        # 5. Intent Stage
        with profiler.time_stage("intent"):
            intent_res = {"intents": query_obj["intent"], "confidence": query_obj["intent_confidence"]}

        # 6-9. Dense Vector & Sparse Hybrid Retrieval Stage
        with profiler.time_stage("bm25"):
            fused_cands = self.retriever.retrieve_hybrid(query_obj["normalized_text"], top_k=15)

        with profiler.time_stage("bge_m3"):
            pass

        with profiler.time_stage("rapidfuzz"):
            pass

        with profiler.time_stage("rrf"):
            pass

        # 10. Reranker & Priority Boost Stage
        with profiler.time_stage("priority"):
            reranked_cands = self.reranker.apply_priority_boost(fused_cands)

        # 11. CRAG & Confidence Stage
        with profiler.time_stage("crag"):
            conf_res = self.confidence_eval.evaluate(reranked_cands, query_obj)

        # 12. Answer Routing & Synthesis Stage
        with profiler.time_stage("answer"):
            route_res = self.router.route(query_obj, conf_res)
            route = route_res["route"]

            if route == "DETERMINISTIC":
                primary_intent = query_obj["intent"][0] if query_obj["intent"] else "MEMBERSHIP"
                tmpl_ans = StructuredAnswerTemplates.get_template_answer(primary_intent)
                answer_text = tmpl_ans or "يرجى زيارة بوابة نقابة المهندسين للحصول على التفاصيل."
            else:
                prompt_str, sources_used = self.generator.build_prompt(
                    query=user_query,
                    standalone_query=query_obj["canonical_msa"],
                    sources=reranked_cands,
                    detected_accent=query_obj["language"]
                )
                if prompt_str:
                    gen_out = await self.generator.generate(prompt_str)
                    answer_text = gen_out[0] if isinstance(gen_out, tuple) else gen_out
                    # No grounded answer -> offer human escalation rather than a dead end.
                    if not answer_text:
                        answer_text = PROMPT_LOW_CONFIDENCE
                else:
                    answer_text = PROMPT_LOW_CONFIDENCE

        # Cache final answer if valid (single-turn only, and never the
        # escalation prompt -- it is conversational state, not an answer)
        if answer_text and not hist and answer_text != PROMPT_LOW_CONFIDENCE:
            self.cache.put(user_query, answer_text, reranked_cands[:3])

        profiler.record_stage("answer", profiler.stage_latencies["answer"], {"route": route})
        report = profiler.generate_report()
        diagnostics = DiagnosticAnalyzer.analyze_trace(report)
        profiler.write_request_log(user_query, answer_text, cache_hit=False, route=route)

        return {
            "request_id": request_id,
            "query": query_obj,
            "answer": answer_text,
            "sources": reranked_cands[:5],
            "confidence": conf_res,
            "route": route_res,
            "cache_hit": False,
            "profiling": report,
            "diagnostics": diagnostics,
            "text_breakdown": profiler.format_text_breakdown()
        }
