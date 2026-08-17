import os
import re
import json
import requests
from typing import Dict, Any, List, Tuple, Optional

from src.config import OLLAMA_URL, OLLAMA_CHAT_MODEL, OLLAMA_KEEP_ALIVE, REQUEST_TIMEOUT, NUM_CTX, NUM_PREDICT


DIALECT_MSA_MAP = {
    # Levantine / Jordanian
    r"\bشو هي\b": "ما هي",
    r"\bشو هو\b": "ما هو",
    r"\bشو\b": "ما هي",
    r"\bايش\b": "ما هي",
    r"\bإيش\b": "ما هي",
    r"\bبدي أسجل\b": "طريقة التسجيل وشروط العضوية في النقابة",
    r"\bبدي اسجل\b": "طريقة التسجيل وشروط العضوية في النقابة",
    r"\bبدي\b": "أريد معرفة",
    r"\bعشان\b": "من أجل",
    r"\bقديش\b": "ما قيمة رسوم",
    r"\bقديه\b": "ما قيمة رسوم",
    r"\bكيف بقدر\b": "ما هي كيفية إجراءات",
    r"\bوين\b": "أين يقع مقر",
    r"\bحبيت اعرف\b": "أستفسر عن",
    r"\bبصير\b": "هل يحق لي",
    r"\bيلي\b": "الذي",
    r"\bليش\b": "ما هو سبب",
    r"\bالاوراق\b": "الوثائق والمستندات الرسمية المطلوب تقديمها",
    r"\bالأوراق\b": "الوثائق والمستندات الرسمية المطلوب تقديمها",
    r"\bشباب\b": "المهندسين الشباب حديثي التخرج",
    r"\bالمصاري\b": "المبالغ المالية والرسوم والاشتراكات",

    # Egyptian
    r"\bكام\b": "ما قيمة كم تبلغ",
    r"\bازاي\b": "كيفية خطوات إجراءات",
    r"\bإزاي\b": "كيفية خطوات إجراءات",
    r"\bعايز\b": "أريد معرفة",
    r"\bعايزة\b": "أريد معرفة",
    r"\bفين\b": "أين موقع مكان",
    r"\bعلشان\b": "من أجل",
    r"\bليه\b": "ما سبب",
    r"\bدلوقتي\b": "حالياً",

    # Gulf
    r"\bشلون\b": "كيفية إجراءات",
    r"\bوش\b": "ما هي",
    r"\bابي\b": "أريد معرفة",
    r"\bأبي\b": "أريد معرفة",
    r"\bشنو\b": "ما هي",
    r"\bمتى\b": "ما هو موعد تصفية",
}


class DialectRewriterAgent:
    """
    Agentic LLM Query Rewriter & Dialect Translator.
    1. Resolves multi-turn conversational context.
    2. Dynamically translates regional Arabic dialects into formal MSA legal terminology.
    3. Provides a fast regex fallback if LLM is offline or times out.
    """

    def __init__(self, ollama_url: str = OLLAMA_URL, model: str = OLLAMA_CHAT_MODEL):
        self.ollama_url = ollama_url
        self.model = model

    def fallback_rule_rewriter(self, query: str, history: Optional[List[Dict[str, str]]] = None) -> str:
        q = query.strip()
        for pattern, replacement in DIALECT_MSA_MAP.items():
            q = re.sub(pattern, replacement, q, flags=re.IGNORECASE)

        if history and len(history) > 0:
            last_user_msg = next((h.get("content", "") for h in reversed(history) if h.get("role") == "user"), "")
            if last_user_msg:
                entity_patterns = [
                    r'صندوق\s+[\w\d_]+',
                    r'التأمين\s+[\w\d_]+',
                    r'تأمين\s+[\w\d_]+',
                    r'قانون\s+[\w\d_]+',
                    r'نظام\s+[\w\d_]+',
                    r'سلم\s+الرواتب',
                    r'العضوية',
                    r'شروط\s+[\w\d_]+'
                ]
                for pat in entity_patterns:
                    match = re.search(pat, last_user_msg)
                    if match:
                        subject = match.group(0)
                        if not any(kw in q for kw in subject.split()):
                            q = f"{q} في {subject}"
                        break

        return q

    def rewrite_query(self, query: str, history: Optional[List[Dict[str, str]]] = None) -> Tuple[str, Dict[str, Any]]:
        raw_query = query.strip()
        if not raw_query:
            return "", {"rewritten": False, "method": "empty"}

        history_context = ""
        has_history = False
        if history and len(history) > 0:
            recent_turns = history[-6:]
            formatted_turns = []
            for h in recent_turns:
                role = "المستخدم" if h.get("role") == "user" else "المساعد"
                content = h.get("content", "").strip()
                if content:
                    formatted_turns.append(f"{role}: {content}")
            if formatted_turns:
                history_context = "\n".join(formatted_turns)
                has_history = True

        if not has_history:
            fast_query = self.fallback_rule_rewriter(raw_query)
            return fast_query, {
                "raw_query": raw_query,
                "rewritten_query": fast_query,
                "rewritten": (fast_query != raw_query),
                "method": "fast_rule_agent",
                "has_history": False
            }

        prompt = (
            "أنت خبير إعادة صياغة وتحويل الاستعلامات لنظام استرجاع قانوني ونقابي لنقابة المهندسين الأردنيين.\n"
            "مهمتك إعادة صياغة السؤال لإرجاع استعلام بحث مستقل ودقيق باللغة العربية الفصحى الرسمية (Canonical MSA):\n"
            "1. حل جميع الضمائر والسياق التراكمي من المحادثة السابقة ليصبح السؤال مستقلاً بذاته (Standalone Query).\n"
            "2. توحيد أي لغة أو لهجة (إنجليزية، أردنية، شامية، مصرية، خليجية) وتحويلها إلى مصطلحات قانونية ونقابية رسمية بالفصحى تُناسب البحث في وثائق النقابة.\n\n"
            "قواعد صارمة:\n"
            "• اطبع فقط السؤال الجديد المحسن والمستقل بالفصحى الرسمية (Canonical MSA).\n"
            "• لا تضف أي شرح أو تعليق أو مقدمات إطلاقاً.\n"
            "• حافظ على جميع المعلومات التفصيلية المحددة في سؤال المستخدم دون حذف.\n\n"
        )

        if has_history:
            prompt += f"سياق المحادثة السابقة:\n{history_context}\n\n"

        prompt += f"سؤال المستخدم الحالي: {raw_query}\n\nالسؤال المستقل المحسن بالفصحى (Canonical MSA):"

        try:
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "think": False,
                "keep_alive": OLLAMA_KEEP_ALIVE,
                "options": {
                    "temperature": 0.1,
                    "num_predict": 120,
                    "num_ctx": NUM_CTX
                }
            }
            res = requests.post(self.ollama_url, json=payload, timeout=(3.0, 8.0))
            if res.status_code == 200:
                rewritten_text = res.json().get("response", "").strip()
                rewritten_text = re.sub(r'^["\'\s]+|["\'\s]+$', '', rewritten_text).strip()
                if rewritten_text and len(rewritten_text) > 3:
                    return rewritten_text, {
                        "raw_query": raw_query,
                        "rewritten_query": rewritten_text,
                        "rewritten": True,
                        "method": "llm_agent",
                        "has_history": has_history
                    }
        except Exception:
            pass

        fallback_query = self.fallback_rule_rewriter(raw_query, history=history)
        return fallback_query, {
            "raw_query": raw_query,
            "rewritten_query": fallback_query,
            "rewritten": fallback_query != raw_query,
            "method": "rule_fallback",
            "has_history": has_history
        }
