import pytest
from src.pipeline.stage_01_understand.query_preprocessor import QueryPreprocessor


@pytest.mark.unit
class TestQueryPreprocessor:
    def setup_method(self):
        self.preprocessor = QueryPreprocessor()

    def test_normalize_diacritics(self):
        text = "نِقَابَةُ المُنْهَدِسِينَ"
        normalized = self.preprocessor.normalize_arabic(text)
        assert "ِ" not in normalized
        assert "ُ" not in normalized
        assert "ْ" not in normalized

    def test_normalize_alef_variants(self):
        text = "أحمد إبراهيم آمنة"
        normalized = self.preprocessor.normalize_arabic(text)
        assert "أ" not in normalized
        assert "إ" not in normalized
        assert "آ" not in normalized
        assert "احمد ابراهيم امنه" in normalized

    def test_normalize_teh_marbuta_and_maqsura(self):
        text = "نقابة مستشفى"
        normalized = self.preprocessor.normalize_arabic(text)
        assert normalized.endswith("مستشفي")
        assert "نقابه" in normalized

    def test_empty_string(self):
        assert self.preprocessor.normalize_arabic("") == ""
        assert self.preprocessor.normalize_arabic(None) == ""

    def test_detect_arabic(self):
        res = self.preprocessor.detect_language("شو هي الأوراق المطلوب تقديمها للنقابة؟")
        assert res["language"] == "ar"

    def test_detect_english(self):
        res = self.preprocessor.detect_language("What are the engineering guild requirements?")
        assert res["language"] == "en"

    def test_detect_mixed(self):
        res = self.preprocessor.detect_language("ما هي الـ requirements الخاصة بـ JEA؟")
        assert res["language"] == "mixed"

    def test_process_jordanian_query(self, sample_queries):
        res = self.preprocessor.process(sample_queries["registration_jordanian"])
        assert res["language"] == "ar"
        assert "canonical_msa" in res
        assert "normalized_text" in res
        assert isinstance(res["intent"], list)
        assert res["intent_confidence"] > 0

    def test_process_msa_query(self, sample_queries):
        res = self.preprocessor.process(sample_queries["registration_msa"])
        assert res["language"] == "ar"
        assert len(res["canonical_msa"]) > 0


@pytest.mark.unit
class TestAraT5DialectRewriter:
    def test_rewriter_initialization(self):
        from src.pipeline.stage_01_understand.arat5_rewriter import AraT5DialectRewriter
        rewriter = AraT5DialectRewriter(enable=False)
        assert rewriter.enable is False
        assert rewriter.rewrite_to_msa("شو هي الأوراق") == "شو هي الأوراق"
