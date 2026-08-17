import re
from typing import Dict, Any

class LanguageDetector:
    """Detects query language: ar-JO (Jordanian Arabic), ar-MSA, en (English), or mixed."""

    JORDANIAN_KEYWORDS = [
        "شو", "بدي", "عشان", "قديش", "وين", "ايش", "كيف بقدر", "بصحة", "أنتسب",
        "اقدر", "يلي", "ليش", "هيك", "شو الأوراق", "كيف اسجل"
    ]

    def detect(self, query: str) -> Dict[str, Any]:
        q = query.strip()
        q_lower = q.lower()
        
        latin_chars = len(re.findall(r'[a-zA-Z]', q))
        arabic_chars = len(re.findall(r'[\u0600-\u06FF]', q))
        
        if arabic_chars > 0 and latin_chars > 0:
            lang = "mixed"
            conf = 0.90
        elif latin_chars > 0 and arabic_chars == 0:
            lang = "en"
            conf = 0.98
        elif any(kw in q for kw in self.JORDANIAN_KEYWORDS):
            lang = "ar-JO"
            conf = 0.95
        else:
            lang = "ar-MSA"
            conf = 0.92

        return {
            "language": lang,
            "confidence": conf,
            "latin_chars": latin_chars,
            "arabic_chars": arabic_chars
        }
