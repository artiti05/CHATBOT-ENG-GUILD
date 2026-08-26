import re
import logging
import httpx
from typing import Any, Dict, List, Optional, Tuple

from src.config import VLLM_CHAT_URL, VLLM_CHAT_MODEL
from src.cache_db.document_registry import TicketRegistry

logger = logging.getLogger(__name__)

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

# Data-correction language bumps an explicit escalation to high priority --
# same signal the ported-from reference project uses.
_PRIORITY_BUMP_RE = re.compile(r'خطأ|خطا|تعديل|تحديث', re.IGNORECASE)

_MSG_LOW_CONFIDENCE_APOLOGY = (
    "للأسف ما لقيت إجابة دقيقة كافية بخصوص سؤالك ضمن قاعدة المعرفة الحالية.\n"
    "بقدر أحوّل استفسارك لأحد موظفي النقابة ليتابعه معك مباشرة."
)


class TicketIntakeAgent:
    """Detects the two ticket-filing triggers (explicit escalation request /
    low RAG confidence) and files a ticket immediately, using the identity
    the caller already supplies -- no follow-up turn is needed to collect
    contact info. Resolution after that point is a purely human process;
    this agent never generates suggestions or follow-up answers for an
    existing ticket."""

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

    def escalate(
        self,
        reason: str,
        query: str,
        history: Optional[List[Dict[str, str]]],
        user: Optional[Dict[str, str]],
    ) -> Tuple[int, str]:
        """Files the ticket and returns (ticket_id, user_facing_message).
        Synchronous (SQLite) -- call via asyncio.to_thread from async code."""
        priority = "high" if reason == "user_intent" and _PRIORITY_BUMP_RE.search(query) else "medium"
        ticket_id = self.registry.create_ticket(
            reason=reason, query=query, history=history or [], user=user or {}, priority=priority
        )
        if reason == "low_confidence":
            message = f"{_MSG_LOW_CONFIDENCE_APOLOGY}\n\nتم فتح تذكرة متابعة رقم #{ticket_id}."
        else:
            message = self._build_confirmation(ticket_id)
        return ticket_id, message

    def _build_confirmation(self, ticket_id: int) -> str:
        return (
            f"تم فتح تذكرة رقم #{ticket_id} بنجاح، وسيتواصل معك أحد موظفي النقابة قريباً لمتابعة طلبك.\n"
            "شكراً لتواصلك مع النقابة."
        )
