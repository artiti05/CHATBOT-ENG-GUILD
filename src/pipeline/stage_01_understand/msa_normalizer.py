from typing import Dict, Any
from .language_detector import LanguageDetector
from .arabic_normalizer import ArabicNormalizer
from .jordanian_normalizer import JordanianNormalizer
from .terminology import TerminologyResolver
from .isri_processor import ISRIProcessor
from .intent_classifier import IntentClassifier

class MSANormalizerPipeline:
    """Orchestrates query multi-representation construction across language, dialect, terminology, and intent."""

    def __init__(self):
        self.lang_detector = LanguageDetector()
        self.arb_normalizer = ArabicNormalizer()
        self.jo_normalizer = JordanianNormalizer()
        self.term_resolver = TerminologyResolver()
        self.isri_processor = ISRIProcessor()
        self.intent_classifier = IntentClassifier()

    def process(self, query: str) -> Dict[str, Any]:
        original_query = query.strip()

        # 1. Detect language
        lang_res = self.lang_detector.detect(original_query)
        language = lang_res["language"]

        # 2. Normalize original
        normalized_query = self.arb_normalizer.normalize(original_query)

        # 3. Terminology resolution
        term_res = self.term_resolver.resolve(normalized_query)
        term_resolved_text = term_res["resolved_text"]
        entities = term_res["matched_entities"]

        # 4. Dialect & English to Canonical MSA conversion
        if language in ("ar-JO", "mixed"):
            msa_converted, _ = self.jo_normalizer.convert_to_msa(term_resolved_text)
        elif language == "en":
            # Map key English search terms to Canonical MSA for Knowledge Base retrieval
            en_lower = original_query.lower()
            msa_parts = []
            if any(w in en_lower for w in ["requirement", "join", "register", "membership", "document"]):
                msa_parts.append("شروط التسجيل والانتساب والوثائق المطلوبة")
            if any(w in en_lower for w in ["fee", "cost", "subscription", "pay"]):
                msa_parts.append("رسوم الانتساب والاشتراك السنوي")
            if any(w in en_lower for w in ["salary", "wage", "pay scale"]):
                msa_parts.append("سلم الرواتب والحد الادنى للاجور")
            if any(w in en_lower for w in ["retire", "pension"]):
                msa_parts.append("نظام التقاعد والراتب التقاعدي")
            if any(w in en_lower for w in ["health", "insurance", "medical"]):
                msa_parts.append("نظام التامين الصحي")

            msa_converted = " ".join(msa_parts) if msa_parts else original_query
        else:
            msa_converted = term_resolved_text

        # Final cleanup for Canonical MSA
        canonical_msa = self.arb_normalizer.normalize(msa_converted)
        if not canonical_msa:
            canonical_msa = original_query

        # 5. ISRI Representation
        isri_query = self.isri_processor.process(canonical_msa)

        # 6. Intent & Entity Detection
        intent_res = self.intent_classifier.classify(canonical_msa)

        return {
            "original_query": original_query,
            "language": language,
            "normalized_query": normalized_query,
            "canonical_msa": canonical_msa,
            "isri_query": isri_query,
            "intent": intent_res["intents"],
            "entities": entities,
            "intent_confidence": intent_res["confidence"],
            "is_explicit_ticket": intent_res.get("is_explicit_ticket", False),
            "ticket_category": intent_res.get("ticket_category")
        }
