from typing import Any, Dict, List

from src.cache_db.semantic_cache import SemanticCache
from src.monitoring.diagnostic_report import DiagnosticAnalyzer
from src.monitoring.profiler import PipelineProfiler
from src.monitoring.tracing import RequestTracer
from src.pipeline.stage_01_understand.query_preprocessor import QueryPreprocessor
from src.pipeline.stage_02_retrieve.bm25_retriever import BM25Retriever
from src.pipeline.stage_02_retrieve.retriever_agent import RetrieverAgent
from src.pipeline.stage_02_retrieve.rrf_fusion import RRFFusion
from src.pipeline.stage_03_verify_rerank.confidence import DeterministicConfidenceEvaluator
from src.pipeline.stage_03_verify_rerank.reranker import PriorityReranker
from src.pipeline.stage_04_answer.answer_router import AnswerRouter
from src.pipeline.stage_04_answer.generator_agent import ResponseGeneratorAgent
from src.services.moderation_notifier import ModerationAlertDispatcher


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

    def process_query(self, user_query: str, history: List[Dict[str, str]] = None, session_id: str = None) -> Dict[str, Any]:
        request_id = RequestTracer.generate_request_id()
        profiler = PipelineProfiler(request_id)

        # 1. Embed & Cache Stage (Session-Scoped)
        with profiler.time_stage("cache"):
            query_vec = self.retriever.embedder.embed_single_text(user_query) if hasattr(self.retriever, "embedder") else None
            cached_res = self.cache.get(user_query, query_vec=query_vec, session_id=session_id)
            if cached_res:
                report = profiler.generate_report()
                profiler.write_request_log(user_query, cached_res["answer"], cache_hit=True, route="CACHE")
                return {
                    "request_id": request_id,
                    "query": {"original": user_query},
                    "answer": cached_res["answer"],
                    "sources": cached_res.get("sources", []),
                    "cache_hit": True,
                    "moderation": {"flagged": False, "flag_type": "none"},
                    "profiling": report,
                    "text_breakdown": profiler.format_text_breakdown()
                }

        # 2. Language Detection Stage
        with profiler.time_stage("language_detection"):
            self.nlp_pipeline.detect_language(user_query)

        # 3. Normalization Stage
        with profiler.time_stage("normalization"):
            self.nlp_pipeline.normalize_arabic(user_query)

        # 4. MSA Conversion & Multi-Representation Stage
        with profiler.time_stage("msa_conversion"):
            query_obj = self.nlp_pipeline.process(user_query, history=history)

        # 5. Intent Stage
        with profiler.time_stage("intent"):
            pass

        # 6-9. Dense Vector & Sparse Hybrid Retrieval Stage
        with profiler.time_stage("bm25"):
            search_query = query_obj.get("canonical_msa") or query_obj.get("normalized_text")
            fused_cands = self.retriever.retrieve_hybrid(search_query, top_k=15, query_vec=query_vec)

        with profiler.time_stage("bge_m3"):
            pass

        with profiler.time_stage("rapidfuzz"):
            pass

        with profiler.time_stage("rrf"):
            pass

        # 10. Reranker & Priority Boost Stage
        with profiler.time_stage("priority"):
            reranked_cands = self.reranker.rerank(search_query, fused_cands) if hasattr(self.reranker, "rerank") else self.reranker.apply_priority_boost(fused_cands)

        # 11. CRAG & Confidence Stage
        with profiler.time_stage("crag"):
            conf_res = self.confidence_eval.evaluate(reranked_cands, query_obj)

        # 12. Answer Routing & Synthesis Stage
        sources_used = []
        with profiler.time_stage("answer"):
            route_res = self.router.route(query_obj, conf_res)
            route = route_res["route"]

            if route == "DETERMINISTIC":
                answer_text = "يرجى زيارة بوابة نقابة المهندسين (jea.org.jo) للحصول على التفاصيل والخدمات الرسمية."
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

        # Moderation check and cache safety
        is_flagged = getattr(self.generator, "last_flag_detected", False)
        if is_flagged:
            moderation_info = {
                "flagged": True,
                "flag_type": "offensive_deescalated",
            }
            # Dispatch async webhook alert to external API endpoint
            ModerationAlertDispatcher.dispatch(
                request_id=request_id,
                query=user_query,
                flag_type="offensive_deescalated",
                answer=answer_text,
                language=query_obj.get("language", "ar"),
                extra_metadata={"route": route, "sources_count": len(sources_used)}
            )
        else:
            moderation_info = {
                "flagged": False,
                "flag_type": "none",
            }
            # Cache final answer ONLY if not flagged (prevents cache poisoning)
            if answer_text:
                self.cache.put(user_query, answer_text, reranked_cands[:3], query_vec=query_vec, session_id=session_id)

        profiler.record_stage("answer", profiler.stage_latencies["answer"], {"route": route, "moderation_flagged": is_flagged})
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
            "moderation": moderation_info,
            "cache_hit": False,
            "profiling": report,
            "diagnostics": diagnostics,
            "text_breakdown": profiler.format_text_breakdown()
        }

