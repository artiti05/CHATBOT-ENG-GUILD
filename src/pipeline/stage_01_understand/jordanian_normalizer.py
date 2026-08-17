import re
from typing import Tuple

class JordanianNormalizer:
    """Rule-based translator mapping Jordanian dialect phrases into canonical MSA expressions."""

    MAPPINGS = {
        r"\bشو\b": "ما هي",
        r"\bبدي\b": "أريد",
        r"\bعشان\b": "من أجل",
        r"\bقديش\b": "كم",
        r"\bوين\b": "أين",
        r"\bايش\b": "ما هي",
        r"\bكيف بقدر\b": "كيف يمكنني",
        r"\bالأوراق\b": "الوثائق",
        r"\bأنتسب\b": "التسجيل والانتساب",
        r"\bاسجل\b": "التسجيل",
        r"\bيلي\b": "التي",
        r"\bليش\b": "لماذا"
    }

    def convert_to_msa(self, text: str) -> Tuple[str, int]:
        result = text
        transformations = 0
        for pattern, replacement in self.MAPPINGS.items():
            new_res, count = re.subn(pattern, replacement, result, flags=re.IGNORECASE)
            if count > 0:
                result = new_res
                transformations += count
        return result.strip(), transformations
