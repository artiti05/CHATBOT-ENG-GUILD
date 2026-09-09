import re
import json
import logging
import httpx
from typing import Dict, Any, List, Optional, Tuple

from src.config import (
    ACTIVE_CHAT_URL,
    ACTIVE_CHAT_MODEL,
    ACTIVE_CHAT_HEADERS,
    LLM_PROVIDER,
    ENABLE_BLACK_MESSAGE_MODERATION,
    BLACK_MESSAGE_REFUSAL_MESSAGE,
)
from src.services.black_message_client import BlackMessageClient

logger = logging.getLogger(__name__)

# Heuristic patterns for instant zero-latency detection of blatant abuse/profanity
# Covers common Arabic and English insults, curses, threats, and harassment.
_HEURISTIC_PROFANITY_PATTERNS = [
    # Arabic vulgar insults & curses
    re.compile(r'\b(كلب|حمار|حيوان|سافل|واطي|حقير|نصابين|حرامية|عرص|منيوك|قحبة|شرموط|عرصة)\b', re.IGNORECASE),
    re.compile(r'يلعن\s*(دين|شرف|ام|ابو|عرض|اخت|روح)', re.IGNORECASE),
    re.compile(r'(كس\s*(امك|اختك|عرضك|عينك|النقابة)|كسخت|كسامك)', re.IGNORECASE),
    re.compile(r'(يا\s*ابن\s*(الكلب|الحرام|الشرموطة|القحبة))', re.IGNORECASE),
    re.compile(r'(طز\s*(فيك|فيكم|بالنقابة)?)', re.IGNORECASE),
    # Threats of violence
    re.compile(r'((رح|س|بدي|راح|سأ|سا)?\s*(اذبحك|اقتلك|افجر|اكسر\s*المكاتب|احرق|سأقتلك|ساقتلك))', re.IGNORECASE),
    # English vulgar insults & threats
    re.compile(r'\b(fuck|shit|bitch|bastard|asshole|cunt|dick|pussy|motherfucker|whore|slut)\b', re.IGNORECASE),
    re.compile(r'\b(kill\s*you|burn\s*(you|down)|blow\s*up)\b', re.IGNORECASE),
]

_MODERATION_PROMPT_TEMPLATE = (
    "أنت خبير مراقبة المحتوى وسياسات الاستخدام لمساعد نقابة المهندسين الأردنيين.\n"
    "مهمتك: فحص رسالة المستخدم وتحديد هل هي رسالة سوداء / مخالفة (Black Message) تتضمن شتائم، ألفاظ نابية، "
    "إهانات شخصية، تهديدات، تحرش، محتوى إباحي، أو ازدراء وإساءة صارخة.\n\n"
    "ملاحظة هامة:\n"
    "- الانتقاد المشروع أو الشكوى الشديدة من تأخر معاملة أو التعبير عن الغضب بأدب (\"أنا غير راضٍ عن خدمتكم أبداً\") ليس رسالة سوداء.\n"
    "- السب، الشتم، التهديد، والألفاظ البذيئة (\"أنتو حمير وكلاب\") هي رسالة سوداء قطعاً.\n\n"
    "{context}"
    "رسالة المستخدم المراد فحصها:\n\"{query}\"\n\n"
    "أجب حصراً بصيغة JSON التالية بدون أي نصوص أو شروحات إضافية:\n"
    "{{\"is_black\": true/false, \"reason\": \"سبب التصنيف بإيجاز إن كانت مخالفة وإلا فارغ\", \"category\": \"profanity/threat/harassment/hate_speech/clean\"}}"
)


class BlackMessageAgent:
    """
    Stage 00 Moderation Guardrail Agent.
    Evaluates incoming user messages before RAG retrieval, cache, or ticket intake.
    Integrates directly with BlackMessageClient to tag messages on jea_backend.
    """

    def __init__(
        self,
        client: Optional[BlackMessageClient] = None,
        chat_url: Optional[str] = None,
        model: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
        provider: Optional[str] = None,
        enabled: Optional[bool] = None,
    ):
        self.client = client or BlackMessageClient()
        self.chat_url = chat_url or ACTIVE_CHAT_URL
        self.model = model or ACTIVE_CHAT_MODEL
        self.headers = headers if headers is not None else ACTIVE_CHAT_HEADERS
        self.provider = (provider or LLM_PROVIDER).lower()
        self.enabled = ENABLE_BLACK_MESSAGE_MODERATION if enabled is None else enabled
        self._http_client: Optional[httpx.Client] = None

    def _get_http_client(self) -> httpx.Client:
        if self._http_client is None or getattr(self._http_client, "is_closed", None) is True:
            self._http_client = httpx.Client(
                timeout=httpx.Timeout(4.0, connect=1.5),
                limits=httpx.Limits(max_keepalive_connections=15, max_connections=50),
            )
        return self._http_client

    def _detect_heuristics(self, text: str) -> Optional[str]:
        """Scans for instant explicit regex violations."""
        for pattern in _HEURISTIC_PROFANITY_PATTERNS:
            if pattern.search(text):
                return "Profanity or abusive content detected via security filter"
        return None

    def is_black_message(
        self,
        query: str,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> Tuple[bool, Optional[str], Optional[str]]:
        """
        Synchronous moderation check.
        Returns: (is_black: bool, reason: Optional[str], category: Optional[str])
        """
        if not self.enabled or not query or not query.strip():
            return False, None, "clean"

        cleaned_query = query.strip()

        # 1. Fast heuristic scan
        heuristic_reason = self._detect_heuristics(cleaned_query)
        if heuristic_reason:
            logger.warning("[BlackMessageAgent] Heuristic flagged black message: '%s...'", cleaned_query[:40])
            return True, heuristic_reason, "profanity"

        # 2. Semantic LLM moderation classifier
        context = ""
        if history:
            formatted = []
            for h in history[-2:]:
                role = "المستخدم" if h.get("role") == "user" else "المساعد"
                content = h.get("content", "").strip()
                if content:
                    formatted.append(f"{role}: {content}")
            if formatted:
                context = "سياق المحادثة السابق:\n" + "\n".join(formatted) + "\n\n"

        prompt = _MODERATION_PROMPT_TEMPLATE.format(context=context, query=cleaned_query)

        try:
            payload: Dict[str, Any] = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "temperature": 0.0,
                "max_tokens": 80,
            }
            if self.provider == "vllm":
                payload["chat_template_kwargs"] = {"enable_thinking": False}

            client = self._get_http_client()
            res = client.post(self.chat_url, json=payload, headers=self.headers)
            if res.status_code == 200:
                raw_reply = res.json()["choices"][0]["message"]["content"].strip()
                clean_json_text = re.sub(r'```(?:json)?\s*|\s*```', '', raw_reply).strip()
                data = json.loads(clean_json_text)
                is_black = bool(data.get("is_black", False))
                reason = data.get("reason", "") or None
                category = data.get("category", "clean")
                if is_black:
                    logger.warning("[BlackMessageAgent] LLM classified as black message: %s (%s)", reason, category)
                return is_black, reason, category
        except Exception as e:
            logger.debug("[BlackMessageAgent] Semantic moderation check skipped due to error: %s", e)

        return False, None, "clean"

    async def is_black_message_async(
        self,
        query: str,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> Tuple[bool, Optional[str], Optional[str]]:
        """
        Asynchronous moderation check for non-blocking FastAPI pipeline execution.
        """
        if not self.enabled or not query or not query.strip():
            return False, None, "clean"

        cleaned_query = query.strip()

        # 1. Fast heuristic scan
        heuristic_reason = self._detect_heuristics(cleaned_query)
        if heuristic_reason:
            logger.warning("[BlackMessageAgent] Heuristic flagged black message: '%s...'", cleaned_query[:40])
            return True, heuristic_reason, "profanity"

        # 2. Semantic LLM moderation check
        context = ""
        if history:
            formatted = []
            for h in history[-2:]:
                role = "المستخدم" if h.get("role") == "user" else "المساعد"
                content = h.get("content", "").strip()
                if content:
                    formatted.append(f"{role}: {content}")
            if formatted:
                context = "سياق المحادثة السابق:\n" + "\n".join(formatted) + "\n\n"

        prompt = _MODERATION_PROMPT_TEMPLATE.format(context=context, query=cleaned_query)

        try:
            payload: Dict[str, Any] = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "temperature": 0.0,
                "max_tokens": 80,
            }
            if self.provider == "vllm":
                payload["chat_template_kwargs"] = {"enable_thinking": False}

            async with httpx.AsyncClient(timeout=httpx.Timeout(4.0, connect=1.5)) as client:
                res = await client.post(self.chat_url, json=payload, headers=self.headers)
            if res.status_code == 200:
                raw_reply = res.json()["choices"][0]["message"]["content"].strip()
                clean_json_text = re.sub(r'```(?:json)?\s*|\s*```', '', raw_reply).strip()
                data = json.loads(clean_json_text)
                is_black = bool(data.get("is_black", False))
                reason = data.get("reason", "") or None
                category = data.get("category", "clean")
                if is_black:
                    logger.warning("[BlackMessageAgent] LLM classified as black message: %s (%s)", reason, category)
                return is_black, reason, category
        except Exception as e:
            logger.debug("[BlackMessageAgent] Async semantic moderation check skipped due to error: %s", e)

        return False, None, "clean"

    def handle_black_message(
        self,
        query: str,
        message_id: Optional[str] = None,
        session_id: Optional[str] = None,
        reason: Optional[str] = None,
        category: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Synchronously handles a confirmed black message:
        1. Dispatches flag to jea_backend via BlackMessageClient (if message_id is available).
        2. Formats and returns refusal response payload.
        """
        effective_reason = reason or "Content violates acceptable use policy"
        if message_id:
            try:
                self.client.mark_black_message(
                    message_id=message_id,
                    is_black=True,
                    reason=effective_reason,
                    session_id=session_id,
                )
            except Exception as e:
                logger.warning("[BlackMessageAgent] Failed to flag message %s via client: %s", message_id, e)

        return {
            "answer": BLACK_MESSAGE_REFUSAL_MESSAGE,
            "is_black_message": True,
            "is_black": True,
            "black_reason": effective_reason,
            "black_category": category or "profanity",
            "sources": [],
            "cache_hit": False,
            "escalated": False,
            "is_escalated": False,
            "ticket_id": None,
            "session_id": session_id,
            "message_id": message_id,
            "route": {"route": "BLACK_MESSAGE", "reason": effective_reason},
        }

    async def handle_black_message_async(
        self,
        query: str,
        message_id: Optional[str] = None,
        session_id: Optional[str] = None,
        reason: Optional[str] = None,
        category: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Asynchronously handles a confirmed black message:
        1. Dispatches flag to jea_backend via BlackMessageClient.mark_black_message_async.
        2. Formats and returns refusal response payload.
        """
        effective_reason = reason or "Content violates acceptable use policy"
        if message_id:
            try:
                await self.client.mark_black_message_async(
                    message_id=message_id,
                    is_black=True,
                    reason=effective_reason,
                    session_id=session_id,
                )
            except Exception as e:
                logger.warning("[BlackMessageAgent] Failed to async flag message %s via client: %s", message_id, e)

        return {
            "answer": BLACK_MESSAGE_REFUSAL_MESSAGE,
            "is_black_message": True,
            "is_black": True,
            "black_reason": effective_reason,
            "black_category": category or "profanity",
            "sources": [],
            "cache_hit": False,
            "escalated": False,
            "is_escalated": False,
            "ticket_id": None,
            "session_id": session_id,
            "message_id": message_id,
            "route": {"route": "BLACK_MESSAGE", "reason": effective_reason},
        }
