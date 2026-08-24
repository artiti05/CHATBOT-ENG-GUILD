import asyncio
import pytest
from src.rag_chatbot_engine import RAGChatbotEngine
from src.core.rag_engine import RAGChatbot

@pytest.mark.integration
class TestRAGPipelineIntegration:
    @pytest.fixture(autouse=True)
    def setup_engine(self):
        self.engine = RAGChatbotEngine()
        self.wrapper = RAGChatbot()

    def test_process_query_end_to_end(self, sample_queries):
        query = sample_queries["registration_msa"]
        res = asyncio.run(self.engine.process_query(query))
        
        assert "request_id" in res
        assert "query" in res
        assert "answer" in res
        assert "sources" in res
        assert "profiling" in res
        assert "text_breakdown" in res
        
        # Check profiling metrics
        profiling = res["profiling"]
        assert "total_latency_ms" in profiling
        assert "breakdown" in profiling
        assert isinstance(profiling["total_latency_ms"], (int, float))

    def test_cache_hit_pipeline(self):
        query = "اختبار الكاش الشامل للأنظمة"
        answer = "إجابة الكاش النموذجية"
        sources = [{"title": "مصدر الكاش", "similarity_score": 99.0}]

        # Put entry into engine cache
        self.engine.cache.put(query, answer, sources)

        # Process query
        res = asyncio.run(self.engine.process_query(query))
        assert res["cache_hit"] is True
        assert res["answer"] == answer

    def test_wrapper_answer_question(self, sample_queries):
        query = sample_queries["salary_msa"]
        res = asyncio.run(self.wrapper.answer_question(query, top_k=5))
        
        assert "answer" in res
        assert "sources" in res
        assert "cache_hit" in res
        assert "metadata" in res
