import re

class ISRIProcessor:
    """Extracts ISRI stems and key root terms for lexical matching."""

    def process(self, text: str) -> str:
        if not text:
            return ""
        # Strip common prefixes (الـ) and suffixes
        words = text.split()
        stems = []
        for w in words:
            clean = re.sub(r'^\bال', '', w)
            clean = re.sub(r'[\u064B-\u0652]', '', clean)
            if len(clean) >= 3:
                stems.append(clean)
            else:
                stems.append(w)
        return " ".join(stems)
