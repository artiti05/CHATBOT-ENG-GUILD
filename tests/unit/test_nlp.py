import pytest
from src.pipeline.stage_01_understand.arabic_normalizer import ArabicNormalizer
from src.pipeline.stage_01_understand.language_detector import LanguageDetector
from src.pipeline.stage_01_understand.intent_classifier import IntentClassifier
from src.pipeline.stage_01_understand.jordanian_normalizer import JordanianNormalizer
from src.pipeline.stage_01_understand.terminology import TerminologyResolver
from src.pipeline.stage_01_understand.isri_processor import ISRIProcessor
from src.pipeline.stage_01_understand.msa_normalizer import MSANormalizerPipeline

@pytest.mark.unit
class TestArabicNormalizer:
    def setup_method(self):
        self.normalizer = ArabicNormalizer()

    def test_normalize_diacritics(self):
        text = "نِقَابَةُ المُنْهَدِسِينَ"
        normalized = self.normalizer.normalize(text)
        assert "ِ" not in normalized
        assert "ُ" not in normalized
        assert "ْ" not in normalized

    def test_normalize_alef_variants(self):
        text = "أحمد إبراهيم آمنة"
        normalized = self.normalizer.normalize(text)
        assert "أ" not in normalized
        assert "إ" not in normalized
        assert "آ" not in normalized
        assert "اهمد ابراهيم امنه" in normalized or "احمد ابراهيم امنه" in normalized

    def test_normalize_teh_marbuta_and_maqsura(self):
        text = "نقابة مستشفى"
        normalized = self.normalizer.normalize(text)
        assert normalized.endswith("مستصفي") or normalized.endswith("مستشفي")
        assert "نقابه" in normalized

    def test_empty_string(self):
        assert self.normalizer.normalize("") == ""
        assert self.normalizer.normalize(None) == ""


@pytest.mark.unit
class TestLanguageDetector:
    def setup_method(self):
        self.detector = LanguageDetector()

    def test_detect_jordanian(self):
        res = self.detector.detect("شو هي الأوراق المطلوب تقديمها للنقابة؟")
        assert res["language"] == "ar-JO"

    def test_detect_msa(self):
        res = self.detector.detect("ما هي شروط التقاعد والإحالة على المعاش؟")
        assert res["language"] == "ar-MSA"

    def test_detect_english(self):
        res = self.detector.detect("What are the engineering guild requirements?")
        assert res["language"] == "en"

    def test_detect_mixed(self):
        res = self.detector.detect("ما هي الـ requirements الخاصة بـ JEA؟")
        assert res["language"] == "mixed"


@pytest.mark.unit
class TestIntentClassifier:
    def setup_method(self):
        self.classifier = IntentClassifier()

    def test_classify_membership(self):
        res = self.classifier.classify("ما هي شروط تسجيل المنتسبين والوثائق المطلوب إحضارها؟")
        assert "MEMBERSHIP" in res["intents"]
        assert res["confidence"] > 0.6

    def test_classify_fees(self):
        res = self.classifier.classify("كم مبلغ رسم الاشتراك السنوي وتكلفة الانتساب؟")
        assert "FEES" in res["intents"]

    def test_classify_retirement(self):
        res = self.classifier.classify("ماهي شروط راتب صندوق التقاعد الهندسية؟")
        assert "RETIREMENT" in res["intents"]

    def test_classify_general_fallback(self):
        res = self.classifier.classify("مرحباً صباح الخير")
        assert "GENERAL" in res["intents"]
        assert res["confidence"] == 0.50

    def test_explicit_ticket_msa(self):
        res = self.classifier.classify("أرغب برفع تذكرة لمشكلة مالية في الدفع")
        assert res["is_explicit_ticket"] is True
        assert "TICKETING" in res["intents"]
        assert res["ticket_category"] == "FINANCIAL"

    def test_explicit_ticket_jordanian(self):
        res = self.classifier.classify("افتحلي تذكرة عشان عطل بتطبيق التأمين الصحي")
        assert res["is_explicit_ticket"] is True
        assert "TICKETING" in res["intents"]
        assert res["ticket_category"] == "HEALTH"

    def test_explicit_ticket_english(self):
        res = self.classifier.classify("Please open a ticket for portal login bug")
        assert res["is_explicit_ticket"] is True
        assert "TICKETING" in res["intents"]
        assert res["ticket_category"] == "TECHNICAL"

    def test_general_problem_without_explicit_ticket_trigger(self):
        res = self.classifier.classify("عندي مشكلة بالدفع بالفواتير")
        assert res["is_explicit_ticket"] is False
        assert "TICKETING" not in res["intents"]


@pytest.mark.unit
class TestMSANormalizerPipeline:
    def setup_method(self):
        self.pipeline = MSANormalizerPipeline()

    def test_process_jordanian_query(self, sample_queries):
        res = self.pipeline.process(sample_queries["registration_jordanian"])
        assert res["language"] == "ar-JO"
        assert "canonical_msa" in res
        assert "normalized_query" in res
        assert isinstance(res["intent"], list)
        assert res["intent_confidence"] > 0

    def test_process_msa_query(self, sample_queries):
        res = self.pipeline.process(sample_queries["registration_msa"])
        assert res["language"] == "ar-MSA"
        assert len(res["canonical_msa"]) > 0
        assert "MEMBERSHIP" in res["intent"]
