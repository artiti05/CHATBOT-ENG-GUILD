import re

class ArabicNormalizer:
    """Deterministic Arabic text normalizer (alef, teh marbuta, tatweel, diacritics)."""

    def normalize(self, text: str) -> str:
        if not text:
            return ""
        # Strip diacritics (tashkeel)
        text = re.sub(r'[\u064B-\u0652]', '', text)
        # Strip tatweel (kashida)
        text = re.sub(r'\u0640', '', text)
        # Normalize Alef variants -> ا
        text = re.sub(r'[أإآ]', 'ا', text)
        # Normalize Teh Marbuta -> ه
        text = re.sub(r'ة', 'ه', text)
        # Normalize Alef Maqsura -> ي
        text = re.sub(r'ى', 'ي', text)
        return text.strip()
