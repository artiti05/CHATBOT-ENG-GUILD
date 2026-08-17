from typing import Dict, Any, List
from .arabic_normalizer import ArabicNormalizer

class IntentClassifier:
    """Rule & keyword based intent classifier for Guild services and Explicit Support Ticketing."""

    def __init__(self):
        self.arb_normalizer = ArabicNormalizer()

    INTENT_PATTERNS = {
        "MEMBERSHIP": ["انتساب", "عضوية", "تسجيل", "شروط", "أوراق", "وثائق", "شروط التسجيل"],
        "FEES": ["رسوم", "كم", "قديش", "تكلفة", "اشتراك", "دفعة", "مبلغ"],
        "RETIREMENT": ["تقاعد", "راتب تقاعدي", "صندوق التقاعد", "سن التقاعد"],
        "INSURANCE": ["تأمين", "صحي", "مستشفى", "علاج", "تأمين صحي"],
        "LAWS": ["قانون", "نظام", "تعليمات", "ممارسة مهنة", "مكاتب"]
    }

    TICKET_TRIGGERS = [
        # Modern Standard Arabic (MSA)
        "فتح تذكرة", "رفع تذكرة", "إنشاء تذكرة", "تقديم بلاغ", "تقديم تذكرة", "تسجيل شكوى", "فتح طلب دعم", "تذكرة", "تذكره",
        # Jordanian Dialect
        "بدّي تذكرة", "بدّي أعمل تذكرة", "افتحلي تذكرة", "سجلي تذكرة", "أعمل تذكرة", "بدي تذكره", "افتح تذكرة", "بدي تذكرة",
        # English
        "create a ticket", "open a ticket", "submit a ticket", "file a ticket", "raise a ticket", "report a ticket", "ticket an issue"
    ]

    TICKET_CATEGORIES = {
        "FINANCIAL": ["دفع", "شراء", "خصم", "مالية", "فواتيركم", "رصيد", "مبلغ", "payment", "financial", "purchase", "deduction", "billing"],
        "HEALTH": ["تأمين", "مستشفى", "علاج", "مطالبة", "طبي", "حامل بطاقة", "health", "insurance", "medical", "claim"],
        "TECHNICAL": ["بوابة", "دخول", "تطبيق", "عطل", "تقنية", "خلل", "سيستم", "portal", "login", "app", "technical", "bug", "system"]
    }

    def _norm(self, s: str) -> str:
        if not s:
            return ""
        normed = self.arb_normalizer.normalize(s).lower()
        return normed

    def classify(self, text: str) -> Dict[str, Any]:
        text_raw_lower = (text or "").lower()
        text_norm = self._norm(text)
        detected_intents = []
        matched_words = []

        for intent, keywords in self.INTENT_PATTERNS.items():
            for kw in keywords:
                kw_norm = self._norm(kw)
                if kw in text_raw_lower or kw_norm in text_norm:
                    detected_intents.append(intent)
                    matched_words.append(kw)
                    break

        # Check explicit ticketing trigger
        is_explicit_ticket = False
        ticket_category = None

        for trigger in self.TICKET_TRIGGERS:
            trig_norm = self._norm(trigger)
            if trigger in text_raw_lower or trig_norm in text_norm:
                is_explicit_ticket = True
                matched_words.append(trigger)
                break

        if is_explicit_ticket:
            detected_intents.append("TICKETING")
            # Resolve category
            for cat, cat_kws in self.TICKET_CATEGORIES.items():
                if any(kw in text_raw_lower or self._norm(kw) in text_norm for kw in cat_kws):
                    ticket_category = cat
                    break
            if not ticket_category:
                ticket_category = "GENERAL"

        if not detected_intents:
            detected_intents = ["GENERAL"]
            confidence = 0.50
        else:
            confidence = min(0.95, 0.70 + 0.15 * len(detected_intents))

        return {
            "intents": list(set(detected_intents)),
            "confidence": round(confidence, 2),
            "matched_keywords": matched_words,
            "is_explicit_ticket": is_explicit_ticket,
            "ticket_category": ticket_category
        }
