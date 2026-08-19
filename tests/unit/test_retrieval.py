import pytest
from src.pipeline.stage_02_retrieve.rrf_fusion import RRFFusion
from src.pipeline.stage_03_verify_rerank.reranker import PriorityReranker
from src.pipeline.stage_02_retrieve.bm25_retriever import BM25Retriever

@pytest.mark.unit
class TestRRFFusion:
    def setup_method(self):
        self.rrf = RRFFusion(k=60)

    def test_combine_two_ranked_lists(self):
        list1 = [
            {"id": "doc1", "title": "شروط الانتساب", "text": "نص 1"},
            {"id": "doc2", "title": "سلم الرواتب", "text": "نص 2"}
        ]
        list2 = [
            {"id": "doc2", "title": "سلم الرواتب", "text": "نص 2"},
            {"id": "doc3", "title": "التأمين الصحي", "text": "نص 3"}
        ]
        results = self.rrf.combine([list1, list2], top_k=10)
        assert len(results) == 3
        # doc2 appears in both lists, so its RRF score should be higher than doc1 and doc3
        assert results[0]["id"] == "doc2"
        assert "rrf_score" in results[0]

    def test_combine_empty_lists(self):
        assert self.rrf.combine([], top_k=5) == []


@pytest.mark.unit
class TestPriorityReranker:
    def setup_method(self):
        self.reranker = PriorityReranker()

    def test_apply_priority_boost(self, sample_documents):
        boosted = self.reranker.apply_priority_boost(sample_documents)
        assert len(boosted) == len(sample_documents)
        for doc in boosted:
            assert "priority_boost" in doc
            assert "final_score" in doc
            assert 50.0 <= doc["similarity_score"] <= 98.0

    def test_sort_by_final_score(self):
        candidates = [
            {"id": "c1", "title": "وثيقة عامة", "rrf_score": 0.01},
            {"id": "c2", "title": "قانون نقابة المهندسين", "rrf_score": 0.03}
        ]
        boosted = self.reranker.apply_priority_boost(candidates)
        assert boosted[0]["id"] == "c2"


@pytest.mark.unit
class TestBM25Retriever:
    def setup_method(self):
        self.bm25 = BM25Retriever()

    def test_search_bm25_returns_list(self):
        results = self.bm25.search("شروط الانتساب", top_k=5)
        assert isinstance(results, list)
