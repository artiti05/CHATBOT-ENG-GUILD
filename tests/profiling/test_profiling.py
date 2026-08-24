import time

import pytest

from src.monitoring.diagnostic_report import DiagnosticAnalyzer
from src.monitoring.profiler import PipelineProfiler
from src.rag_chatbot_engine import RAGChatbotEngine


@pytest.mark.profiling
class TestPipelineProfiling:
    def test_profiler_initialization(self):
        profiler = PipelineProfiler(request_id="test_req_001")
        assert profiler.request_id == "test_req_001"
        assert len(profiler.stage_latencies) == 13
        assert "cache" in profiler.stage_latencies
        assert "answer" in profiler.stage_latencies

    def test_stage_timer_context_manager(self):
        profiler = PipelineProfiler(request_id="test_req_002")
        with profiler.time_stage("bm25"):
            time.sleep(0.02)  # Simulate 20ms stage work

        latency = profiler.stage_latencies["bm25"]
        assert latency >= 15.0  # At least 15ms recorded
        assert isinstance(latency, float)

    def test_generate_report(self):
        profiler = PipelineProfiler(request_id="test_req_003")
        profiler.record_stage("cache", 1.5)
        profiler.record_stage("bm25", 25.0, {"hits": 15})

        report = profiler.generate_report()
        assert report["request_id"] == "test_req_003"
        assert "total_latency_ms" in report
        assert report["breakdown"]["cache"] == 1.5
        assert report["breakdown"]["bm25"] == 25.0
        assert report["metadata"]["bm25"]["hits"] == 15

    def test_text_breakdown_format(self):
        profiler = PipelineProfiler(request_id="test_req_004")
        profiler.record_stage("normalization", 5.0)
        profiler.record_stage("rrf", 12.0)

        breakdown_str = profiler.format_text_breakdown()
        assert "REQUEST test_req_004" in breakdown_str
        assert "Normalization" in breakdown_str
        assert "RRF" in breakdown_str

    def test_diagnostic_analyzer_bottleneck_detection(self):
        report = {
            "request_id": "test_req_slow",
            "breakdown": {
                "bge_m3": 180.0,
                "reranker": 250.0
            },
            "metadata": {
                "crag": {"llm_used": True},
                "answer": {"llm_used": False}
            }
        }

        diagnostics = DiagnosticAnalyzer.analyze_trace(report)
        assert any("BGE-M3 latency high" in d for d in diagnostics)
        assert any("Reranker latency high" in d for d in diagnostics)
        assert any("CRAG Overuse" in d for d in diagnostics)

    def test_full_pipeline_profiling_benchmark(self, sample_queries):
        engine = RAGChatbotEngine()
        query = sample_queries["registration_msa"]

        start_time = time.perf_counter()
        res = engine.process_query(query)
        end_time = time.perf_counter()

        elapsed_total_ms = (end_time - start_time) * 1000.0

        assert "profiling" in res
        profiling = res["profiling"]
        assert profiling["total_latency_ms"] > 0
        assert profiling["total_latency_ms"] <= elapsed_total_ms + 10.0
