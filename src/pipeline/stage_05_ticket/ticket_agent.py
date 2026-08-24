import re
import logging
import httpx
from typing import Any, Dict, List, Optional, Tuple

from src.config import VLLM_CHAT_URL, VLLM_CHAT_MODEL
from src.cache_db.document_registry import TicketRegistry

logger = logging.getLogger(__name__)

# Unique substring shared by both contact-request prompts below. Its presence
# in the last assistant turn is how we detect -- statelessly, using only the
# history the client already round-trips -- that the current user message is
# the reply to a contact-info request rather than a new question.
_CONTACT_REQUEST_ANCHOR = "لإتمام العملية، فضلاً زوّدنا بالمعلومات التالية في رسالة واحدة"

# Only present in the low-confidence variant; used to recover `reason` once
# _CONTACT_REQUEST_ANCHOR confirms we're mid-flow.
_LOW_CONFIDENCE_MARK = "للأسف ما لقيت إجابة دقيقة كافية"

PROMPT_USER_INTENT = (
    "تمام، رح أحوّل طلبك لأحد موظفي النقابة لمتابعته معك مباشرة.\n"
    f"{_CONTACT_REQUEST_ANCHOR}:\n"
    "١) الاسم الكامل\n"
    "٢) رقم الهاتف\n"
    "٣) الرقم النقابي (إن وجد)\n\n"
    "وبيتم فتح تذكرة ومتابعتها من قبل الموظف المختص."
)

PROMPT_LOW_CONFIDENCE = (
    f"{_LOW_CONFIDENCE_MARK} بخصوص سؤالك ضمن قاعدة المعرفة الحالية.\n"
    "بقدر أحوّل استفسارك لأحد موظفي النقابة ليتابعه معك مباشرة.\n"
    f"{_CONTACT_REQUEST_ANCHOR}:\n"
    "١) الاسم الكامل\n"
    "٢) رقم الهاتف\n"
    "٣) الرقم النقابي (إن وجد)\n\n"
    "وبيتم فتح تذكرة ومتابعتها من قبل الموظف المختص."
)

# Emergency fallback ONLY -- used when the LLM intent classifier is unreachable
# (vLLM down/timeout). Deliberately a blunt keyword net: it will misfire on
# purely informational questions ("what's the complaint procedure?"), which is
# an acceptable trade-off only because it's the degraded path, not the primary one.
_INTENT_FALLBACK_PATTERNS = [
    re.compile(r'شكو[ىي]', re.IGNORECASE),
    re.compile(r'اشتك', re.IGNORECASE),
    re.compile(r'تذكرة', re.IGNORECASE),
    re.compile(r'موظف\s*(بشري|حقيقي|مختص)?', re.IGNORECASE),
    re.compile(r'شخص\s*(حقيقي|فعلي)', re.IGNORECASE),
    re.compile(r'(كلم|حول|حولني|حولوني|وصلني|وصلوني)\w*\s*(موظف|احد|حدا|شخص)', re.IGNORECASE),
    re.compile(r'\bcomplaint\b', re.IGNORECASE),
    re.compile(r'\bticket\b', re.IGNORECASE),
    re.compile(r'\bhuman\s*(support|agent|being)?\b', re.IGNORECASE),
    re.compile(r'\bagent\b', re.IGNORECASE),
    re.compile(r'\brepresentative\b', re.IGNORECASE),
    re.compile(r'talk to (a |an )?(human|agent|person|representative)', re.IGNORECASE),
    re.compile(r'customer service', re.IGNORECASE),
]

_INTENT_CLASSIFIER_PROMPT_TEMPLATE = (
    "أنت مصنّف نوايا لمساعد نقابة المهندسين الأردنيين.\n"
    "مهمتك: حدد فقط إذا كانت رسالة المستخدم الحالية تعبّر عن رغبة فعلية بالتحدث مع موظف بشري، "
    "تقديم شكوى حقيقية عن مشكلة يعيشها، أو طلب متابعة إنسانية مباشرة لقضيته.\n\n"
    "لا تعتبرها نية تصعيد لمجرد ورود كلمة مثل 'شكوى' أو 'موظف' أو 'تذكرة' ضمن سؤال معلوماتي عادي.\n"
    "مثال على سؤال عادي (الجواب: لا): \"ما هي خطوات تقديم شكوى ضد مكتب هندسي؟\" -- هذا استفسار عن "
    "إجراء، وليس شكوى فعلية.\n"
    "مثال على نية تصعيد حقيقية (الجواب: نعم): \"قدمت طلب تسجيل من شهر ولا حد رد علي، بدي احكي مع حدا\" "
    "أو \"هذا رد غير مقبول، بدي اشتكي رسمياً\" أو \"وصلني رد بالخطأ وبدي اتواصل مع موظف مباشرة\".\n\n"
    "{context}"
    "رسالة المستخدم الحالية: \"{query}\"\n\n"
    "أجب بكلمة واحدة فقط بدون أي شرح أو علامات ترقيم: نعم أو لا."
)

_PHONE_RE = re.compile(r'(?:\+?962|00962)?0?7[789]\d{7}')
# NOTE (port fix, not in the Qwen3.6 original): each keyword tolerates an
# optional definite article, so "الرقم النقابي" -- the exact wording the bot
# asks for in PROMPT_USER_INTENT -- parses. Previously only the bare
# "رقم نقابي" form matched and the field was silently dropped.
_ENGINEER_NO_RE = re.compile(
    r'(?:رقم\s*(?:ال)?(?:نقابي|نقابة|عضوية|مهندس)|membership\s*(?:no\.?|number)?|eng(?:ineer)?\s*(?:no\.?|number)?)'
    r'\D{0,6}(\d{2,10})',
    re.IGNORECASE,
)
_NAME_RE = re.compile(r'(?:اسمي|الاسم\s*(?:هو)?[:：]?|اسم[:：]|my name is)\s*([^\n,،؛;]+)', re.IGNORECASE)


class TicketIntakeAgent:
    """Detects the two ticket-filing triggers and runs the (stateless,
    history-driven) contact-info collection exchange. Once a ticket is
    created, resolution is entirely a human process -- this agent never
    generates suggestions or follow-up answers for an existing ticket."""

    def __init__(self, vllm_url: str = VLLM_CHAT_URL, model: str = VLLM_CHAT_MODEL):
        self.registry = TicketRegistry()
        self.vllm_url = vllm_url
        self.model = model

    async def detect_intent(self, query: str, history: Optional[List[Dict[str, str]]] = None) -> bool:
        """Semantic escalation-intent classifier -- judges what the user MEANS
        (wants a human / is filing a real complaint / wants to escalate an
        actual problem) rather than matching fixed trigger phrases, so an
        informational question that merely mentions 'complaint' or 'ticket'
        is not misfired on. Falls back to a keyword heuristic only if the
        classifier LLM itself is unreachable."""
        if not query or not query.strip():
            return False

        context = ""
        if history:
            formatted = []
            for h in history[-2:]:
                role = "المستخدم" if h.get("role") == "user" else "المساعد"
                content = h.get("content", "").strip()
                if content:
                    formatted.append(f"{role}: {content}")
            if formatted:
                context = "سياق آخر رسالتين من المحادثة:\n" + "\n".join(formatted) + "\n\n"

        prompt = _INTENT_CLASSIFIER_PROMPT_TEMPLATE.format(context=context, query=query.strip())

        try:
            payload = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "temperature": 0.0,
                "max_tokens": 5,
                "chat_template_kwargs": {"enable_thinking": False},
            }
            async with httpx.AsyncClient(timeout=httpx.Timeout(8.0, connect=3.0)) as client:
                res = await client.post(self.vllm_url, json=payload)
            if res.status_code == 200:
                verdict = res.json()["choices"][0]["message"]["content"].strip()
                return verdict.startswith("نعم") or verdict.lower().startswith("yes")
            logger.warning("[TicketAgent] Intent classifier returned %s: %s", res.status_code, res.text[:200])
        except Exception:
            logger.exception("[TicketAgent] Intent classifier call failed; falling back to keyword heuristic")

        return self._detect_intent_keywords(query)

    def _detect_intent_keywords(self, query: str) -> bool:
        return any(p.search(query) for p in _INTENT_FALLBACK_PATTERNS)

    def is_awaiting_contact(self, history: Optional[List[Dict[str, str]]]) -> Optional[Tuple[str, str]]:
        """If the last assistant turn was a contact-info request, returns
        (reason, original_issue_text); otherwise None."""
        if not history:
            return None
        last_assistant = next((h for h in reversed(history) if h.get("role") == "assistant"), None)
        if not last_assistant or _CONTACT_REQUEST_ANCHOR not in last_assistant.get("content", ""):
            return None

        reason = "low_confidence" if _LOW_CONFIDENCE_MARK in last_assistant["content"] else "user_intent"

        last_assistant_idx = len(history) - 1 - next(
            i for i, h in enumerate(reversed(history)) if h is last_assistant
        )
        issue_text = ""
        for h in reversed(history[:last_assistant_idx]):
            if h.get("role") == "user":
                issue_text = h.get("content", "")
                break

        return reason, issue_text

    def parse_contact_reply(self, text: str) -> Dict[str, str]:
        contact = {"name": "", "phone": "", "engineer_number": "", "raw_text": text}

        phone_match = _PHONE_RE.search(text)
        if phone_match:
            contact["phone"] = phone_match.group(0)

        eng_match = _ENGINEER_NO_RE.search(text)
        if eng_match:
            contact["engineer_number"] = eng_match.group(1)

        name_match = _NAME_RE.search(text)
        if name_match:
            contact["name"] = name_match.group(1).strip()

        return contact

    def create_ticket(self, reason: str, issue_text: str, query: str) -> int:
        """Synchronous (SQLite) -- call via asyncio.to_thread from async code."""
        contact = self.parse_contact_reply(query)
        return self.registry.create_ticket(reason, issue_text, contact)

    def build_confirmation(self, ticket_id: int) -> str:
        return (
            f"تم فتح تذكرة رقم #{ticket_id} بنجاح، وسيتواصل معك أحد موظفي النقابة قريباً لمتابعة طلبك.\n"
            "شكراً لتواصلك مع النقابة."
        )
