import re
from typing import Dict, Any, List

class TerminologyResolver:
    """Guild domain terminology dictionary mapping English/bilingual & informal terms to canonical Guild MSA."""

    TERMINOLOGY_MAP = {
        "membership requirements": "متطلبات الانتساب والتسجيل",
        "membership": "الانتساب والعضوية",
        "professional registration": "تسجيل المهندس مهنياً",
        "registration": "التسجيل النقابي",
        "fees": "رسوم الاشتراكات",
        "pension": "نظام التقاعد",
        "health insurance": "التأمين الصحي",
        "electrical engineering": "الهندسة الكهربائية",
        "civil engineering": "الهندسة المدنية",
        "mechanical engineering": "الهندسة الميكانيكية",
        "architectural engineering": "العمارة وهندسة المباني"
    }

    def resolve(self, text: str) -> Dict[str, Any]:
        resolved_text = text
        matched_entities = {}
        for eng_term, msa_term in self.TERMINOLOGY_MAP.items():
            pattern = re.compile(re.escape(eng_term), re.IGNORECASE)
            if pattern.search(resolved_text):
                resolved_text = pattern.sub(msa_term, resolved_text)
                matched_entities[eng_term] = msa_term

        return {
            "resolved_text": resolved_text,
            "matched_entities": matched_entities
        }
