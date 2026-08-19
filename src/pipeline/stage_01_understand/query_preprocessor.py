import re
import unicodedata
from typing import Dict, Any, List, Optional
import nltk
from nltk.stem.isri import ISRIStemmer


from src.pipeline.stage_01_understand.arat5_rewriter import AraT5DialectRewriter


class QueryPreprocessor:
    """
    Unified Stage 1 Query Preprocessor for Eng-Guild Chatbot.
    Provides pure character normalization, language detection, AraT5 GPU neural rewriting, and ISRI root stemming.
    Contains ZERO hardcoded word replacement dictionaries or regex mapping maps.
    """

    def __init__(self):
        try:
            self.stemmer = ISRIStemmer()
        except Exception:
            self.stemmer = None
        self.arat5_rewriter = AraT5DialectRewriter()

    def normalize_arabic(self, text: str) -> str:
        """
        Pure character-level Unicode normalization.
        Strips Tashkeel diacritics and unifies Alef, Yaa, and Teh Marbuta characters.
        Does NOT perform word replacement.
        """
        if not text:
            return ""
        # Strip Arabic Tashkeel (diacritics)
        tashkeel_pattern = re.compile(r'[\u0617-\u061A\u064B-\u0652]')
        text = re.sub(tashkeel_pattern, '', text)
        
        # Unify Alef variants (أ, إ, آ -> ا)
        text = re.sub(r'[أإآ]', 'ا', text)
        # Unify Yaa / Alef Maqsura (ى -> ي)
        text = re.sub(r'ى', 'ي', text)
        # Unify Teh Marbuta (ة -> ه)
        text = re.sub(r'ة', 'ه', text)
        
        return text.strip()

    def detect_language(self, query: str) -> Dict[str, Any]:
        """Detects query language: ar (Arabic), en (English), or mixed based on Unicode bounds."""
        q = (query or "").strip()
        latin_chars = len(re.findall(r'[a-zA-Z]', q))
        arabic_chars = len(re.findall(r'[\u0600-\u06FF]', q))
        
        if arabic_chars > 0 and latin_chars > 0:
            lang = "mixed"
            conf = 0.90
        elif latin_chars > 0 and arabic_chars == 0:
            lang = "en"
            conf = 0.95
        else:
            lang = "ar"
            conf = 0.95
            
        return {"language": lang, "confidence": conf}

    def stem_isri(self, text: str) -> str:
        """Extracts Arabic root stems using ISRIStemmer for sparse BM25 retrieval."""
        if not text or not self.stemmer:
            return text or ""
        tokens = text.split()
        stemmed = [self.stemmer.stem(t) for t in tokens if len(t) > 2]
        return " ".join(stemmed)

    def classify_intent(self, text: str) -> Dict[str, Any]:
        """General Intent Identifier. Returns clean generalized intent metadata."""
        return {
            "intents": ["GENERAL"],
            "confidence": 1.0,
            "matched_keywords": []
        }

    def canonicalize_query_ai(self, query: str, language: str) -> str:
        """
        Passes clean normalized query through GPU-accelerated AraT5 Neural Dialect Rewriter.
        Converts Jordanian dialect into formal MSA without hardcoded rules.
        """
        norm = self.normalize_arabic(query)
        if language in ("ar", "mixed"):
            res = self.arat5_rewriter.rewrite_to_msa(norm)
            return res if (res and len(res.strip()) >= 3) else norm
        return norm

    def process(self, query: str) -> Dict[str, Any]:
        """
        Main Stage 1 Orchestrator: Converts raw query into a Multi-Representation Query Object.
        """
        original_query = (query or "").strip()
        lang_info = self.detect_language(original_query)
        language = lang_info["language"]
        
        normalized_text = self.normalize_arabic(original_query)
        canonical_msa = self.canonicalize_query_ai(original_query, language)
        isri_stemmed = self.stem_isri(canonical_msa)
        intent_info = self.classify_intent(canonical_msa)
        
        return {
            "original_query": original_query,
            "normalized_text": normalized_text,
            "canonical_msa": canonical_msa,
            "isri_stemmed": isri_stemmed,
            "language": language,
            "language_confidence": lang_info["confidence"],
            "intent": intent_info["intents"],
            "intent_confidence": intent_info["confidence"],
            "matched_keywords": intent_info["matched_keywords"]
        }
