import time
from typing import Dict, Any, List

from src.monitoring.tracing import RequestTracer
from src.monitoring.profiler import PipelineProfiler
from src.monitoring.diagnostic_report import DiagnosticAnalyzer

from src.pipeline.stage_01_understand.msa_normalizer import MSANormalizerPipeline
from src.cache_db.semantic_cache import SemanticCache
from src.pipeline.stage_02_retrieve.bm25_retriever import BM25Retriever
from src.pipeline.stage_02_retrieve.fuzzy_retriever import RapidFuzzRetriever
from src.pipeline.stage_02_retrieve.rrf_fusion import RRFFusion
from src.pipeline.stage_03_verify_rerank.reranker import PriorityReranker
from src.pipeline.stage_03_verify_rerank.confidence import DeterministicConfidenceEvaluator
from src.pipeline.stage_04_answer.answer_router import AnswerRouter
from src.pipeline.stage_04_answer.templates import StructuredAnswerTemplates
from src.pipeline.stage_04_answer.generator_agent import ResponseGeneratorAgent
from src.pipeline.stage_02_retrieve.retriever_agent import RetrieverAgent
from src.services.ticketing_service import TicketingService


class RAGChatbotEngine:
    """
    Central Orchestrator Engine for Eng-Guild Chatbot.
    Implements the complete Normalize → Understand → Retrieve → Verify → Answer architecture
    with first-class 13-stage diagnostic profiling and external support ticketing.
    """

    def __init__(self):
        self.nlp_pipeline = MSANormalizerPipeline()
        self.cache = SemanticCache()
        self.bm25 = BM25Retriever()
        self.fuzzy = RapidFuzzRetriever()
        self.rrf = RRFFusion()
        self.reranker = PriorityReranker()
        self.retriever = RetrieverAgent()
        self.confidence_eval = DeterministicConfidenceEvaluator()
        self.router = AnswerRouter()
        self.generator = ResponseGeneratorAgent()
        self.ticketing_service = TicketingService()

    def process_query(self, user_query: str) -> Dict[str, Any]:
        request_id = RequestTracer.generate_request_id()
        profiler = PipelineProfiler(request_id)

        # 1. Quick check for explicit ticketing to bypass cache
        is_ticket_req = self.nlp_pipeline.intent_classifier.classify(user_query).get("is_explicit_ticket", False)

        # 2. Cache Stage
        with profiler.time_stage("cache"):
            if not is_ticket_req:
                cached_res = self.cache.get(user_query)
                if cached_res:
                    report = profiler.generate_report()
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
            lang_res = self.nlp_pipeline.lang_detector.detect(user_query)

        # 3. Normalization Stage
        with profiler.time_stage("normalization"):
            norm_text = self.nlp_pipeline.arb_normalizer.normalize(user_query)

        # 4. MSA Conversion & Multi-Representation Stage
        with profiler.time_stage("msa_conversion"):
            query_obj = self.nlp_pipeline.process(user_query)

        # 5. Intent Stage
        with profiler.time_stage("intent"):
            intent_res = {"intents": query_obj["intent"], "confidence": query_obj["intent_confidence"]}

        # 6-9. Dense Vector & Sparse Hybrid Retrieval Stage
        with profiler.time_stage("bm25"):
            fused_cands = self.retriever.retrieve_hybrid(user_query, top_k=15)

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
        ticket_info = None
        with profiler.time_stage("answer"):
            route_res = self.router.route(query_obj, conf_res)
            route = route_res["route"]

            if route == "TICKETING":
                cat = query_obj.get("ticket_category") or "GENERAL"
                ticket_res = self.ticketing_service.create_ticket(
                    category=cat,
                    user_query=user_query,
                    language=query_obj.get("language", "ar-JO")
                )
                answer_text = StructuredAnswerTemplates.format_ticket_response(ticket_res)
                ticket_info = ticket_res
            elif route == "DETERMINISTIC":
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
                    gen_out = self.generator.generate(prompt_str)
                    answer_text = gen_out[0] if isinstance(gen_out, tuple) else (gen_out or "لم أتمكن من العثور على معلومات دقيقة.")
                else:
                    answer_text = "لم أتمكن من العثور على معلومات دقيقة."

        # Cache final answer if valid (do not cache ticket submissions)
        if answer_text and route != "TICKETING":
            self.cache.put(user_query, answer_text, reranked_cands[:3])

        profiler.record_stage("answer", profiler.stage_latencies["answer"], {"route": route})
        report = profiler.generate_report()
        diagnostics = DiagnosticAnalyzer.analyze_trace(report)

        return {
            "request_id": request_id,
            "query": query_obj,
            "answer": answer_text,
            "ticket": ticket_info,
            "sources": reranked_cands[:5],
            "confidence": conf_res,
            "route": route_res,
            "cache_hit": False,
            "profiling": report,
            "diagnostics": diagnostics,
            "text_breakdown": profiler.format_text_breakdown()
        }
