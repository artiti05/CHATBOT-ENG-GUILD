import os
import time
import logging
import httpx
from typing import Dict, Any, List, Optional

from src.config import (
    JEA_BACKEND_BASE_URL,
    INTERNAL_BYPASS_TOKEN,
)

logger = logging.getLogger(__name__)

# Keywords identifying dedicated AI / Chatbot escalation categories
AI_CATEGORY_KEYWORDS = [
    "ai",
    "chatbot",
    "bot",
    "ذكاء اصطناعي",
    "المساعد الآلي",
    "المساعد الذكي",
    "شات بوت",
    "شاتبوت",
    "تصعيد ذكي",
    "ai escalation",
]

# Mapping NLU intents to department category keywords
INTENT_CATEGORY_KEYWORDS = {
    "registration": ["professional", "مهنية", "تسجيل", "عضوية"],
    "membership": ["professional", "مهنية", "عضوية", "تسجيل"],
    "fees": ["financial", "مالية", "رسوم", "اشتراكات", "صندوق"],
    "loans": ["loan", "financial", "قروض", "تسهيلات", "صندوق"],
    "health_insurance": ["health", "insurance", "تأمين", "صحي"],
    "pension": ["pension", "تقاعد"],
    "bylaws": ["legal", "professional", "قانونية", "أنظمة", "تعليمات"],
    "certificate": ["certificate", "professional", "شهادات", "مزاولة"],
    "training": ["training", "تدريب", "تأهيل", "دورات"],
    "services": ["submissions", "خدمات", "طلبات"],
    "escalation": ["support", "دعم", "فني", "شكاوى", "مساعدة"],
    "complaint": ["complaint", "support", "شكاوى", "دعم"],
    "contact": ["support", "دعم", "استعلامات"],
    "general_rag": ["support", "دعم", "فني"],
    "greeting": ["support", "دعم"],
}


class TicketingClient:
    """
    HTTP client for JEA Backend (NestJS) Internal Ticketing APIs.
    Fetches active categories, detects/swaps designated AI categories,
    and creates escalation tickets.
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        token: Optional[str] = None,
        cache_ttl_seconds: int = 3600,
    ):
        raw_url = base_url or JEA_BACKEND_BASE_URL or "http://localhost:3000"
        # If full /api/v1/tickets/internal was passed as base, strip down to domain root
        if "/api/" in raw_url:
            raw_url = raw_url.split("/api/")[0]
        self.base_url = raw_url.rstrip("/")

        self.token = token or INTERNAL_BYPASS_TOKEN or "jea_rag_token"
        self.headers = {
            "x-internal-token": self.token,
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        self.cache_ttl_seconds = cache_ttl_seconds
        self._categories_cache: List[Dict[str, Any]] = []
        self._last_fetch_time: float = 0.0

        # Optional manual overrides from environment
        self.override_category_id = os.getenv("AI_TICKET_CATEGORY_ID", "").strip()
        self.override_category_name = os.getenv("AI_TICKET_CATEGORY_NAME", "").strip().lower()

    def fetch_categories(self, force_refresh: bool = False) -> List[Dict[str, Any]]:
        """
        Fetch active service categories from jea_backend internal endpoint.
        Caches result in-memory for TTL duration.
        """
        now = time.time()
        if (
            not force_refresh
            and self._categories_cache
            and (now - self._last_fetch_time < self.cache_ttl_seconds)
        ):
            return self._categories_cache

        url = f"{self.base_url}/api/v1/tickets/categories/internal"
        try:
            with httpx.Client(timeout=4.0) as client:
                response = client.get(url, headers=self.headers)
                if response.status_code == 200:
                    envelope = response.json()
                    categories = (
                        envelope.get("data", [])
                        if isinstance(envelope, dict) and "data" in envelope
                        else envelope
                    )
                    if isinstance(categories, list):
                        self._categories_cache = categories
                        self._last_fetch_time = now
                        logger.info(
                            "[TicketingClient] Successfully fetched %d service categories from backend.",
                            len(categories),
                        )
                        return self._categories_cache
                else:
                    logger.warning(
                        "[TicketingClient] Failed to fetch categories: status=%s, body=%s",
                        response.status_code,
                        response.text[:200],
                    )
        except Exception as e:
            logger.warning("[TicketingClient] Error connecting to backend to fetch categories: %s", e)

        return self._categories_cache or []

    def resolve_ai_category(self, categories: List[Dict[str, Any]]) -> Optional[str]:
        """
        Attempts to find a category specifically designated for AI / Chatbot.
        Checks:
          1. AI_TICKET_CATEGORY_ID env override
          2. AI_TICKET_CATEGORY_NAME env override
          3. Keyword scan for AI/chatbot terms across arName and enName.
        Returns category ID if found, otherwise None.
        """
        if not categories:
            return None

        # 1. Environment ID override
        if self.override_category_id:
            for cat in categories:
                if str(cat.get("id")) == self.override_category_id:
                    return str(cat.get("id"))
            # If specified by ID directly, trust it even if not in returned list
            return self.override_category_id

        # 2. Environment Name override
        if self.override_category_name:
            for cat in categories:
                en = str(cat.get("enName", "")).lower()
                ar = str(cat.get("arName", "")).lower()
                if self.override_category_name in en or self.override_category_name in ar:
                    return str(cat.get("id"))

        # 3. AI Keyword Match
        for cat in categories:
            en = str(cat.get("enName", "")).lower()
            ar = str(cat.get("arName", "")).lower()
            for kw in AI_CATEGORY_KEYWORDS:
                if kw.lower() in en or kw in ar:
                    logger.info(
                        "[TicketingClient] Found designated AI category: '%s' / '%s' (ID: %s)",
                        cat.get("enName"),
                        cat.get("arName"),
                        cat.get("id"),
                    )
                    return str(cat.get("id"))

        return None

    def map_intent_to_category(
        self, intent: Optional[str], categories: List[Dict[str, Any]]
    ) -> Optional[str]:
        """
        Maps an NLU intent to a backend ServiceCategory ID based on keyword mapping.
        """
        if not categories:
            return None

        norm_intent = (intent or "escalation").lower().strip()
        keywords = INTENT_CATEGORY_KEYWORDS.get(norm_intent, ["support", "دعم", "فني"])

        for cat in categories:
            en = str(cat.get("enName", "")).lower()
            ar = str(cat.get("arName", ""))
            for kw in keywords:
                if kw.lower() in en or kw in ar:
                    return str(cat.get("id"))

        # Fallback to Support category if exists
        for cat in categories:
            en = str(cat.get("enName", "")).lower()
            ar = str(cat.get("arName", ""))
            if "support" in en or "دعم" in ar:
                return str(cat.get("id"))

        # Ultimate fallback: return first available category ID
        return str(categories[0].get("id")) if categories else None

    def get_target_category_id(
        self,
        intent: Optional[str] = None,
        force_ai_swap: bool = True,
    ) -> Optional[str]:
        """
        Resolves the target category ID for an AI escalation ticket.
        If force_ai_swap is True, it first tries to swap to the designated AI category.
        If no AI category is defined on the backend, it falls back to intent-based mapping.
        """
        categories = self.fetch_categories()
        if not categories:
            return self.override_category_id or None

        if force_ai_swap:
            ai_cat_id = self.resolve_ai_category(categories)
            if ai_cat_id:
                return ai_cat_id

        # Fall back to functional intent mapping
        return self.map_intent_to_category(intent, categories)

    def create_ticket(
        self,
        title: str,
        content: str,
        priority: str = "MEDIUM",
        intent: Optional[str] = None,
        category_id: Optional[str] = None,
        session_id: Optional[str] = None,
        phone: Optional[str] = None,
        reason: Optional[str] = None,
        force_ai_swap: bool = True,
    ) -> Optional[Dict[str, Any]]:
        """
        Submits an escalation ticket to POST /api/v1/tickets/internal on jea_backend.
        Automatically resolves and swaps the category for AI.
        """
        url = f"{self.base_url}/api/v1/tickets/internal"

        # Determine target category
        final_category_id = category_id or self.get_target_category_id(
            intent=intent, force_ai_swap=force_ai_swap
        )

        prio_upper = priority.upper()
        if prio_upper not in ["HIGH", "MEDIUM", "LOW"]:
            prio_upper = "MEDIUM"

        payload: Dict[str, Any] = {
            "title": title[:100],
            "content": content,
            "ticketPriority": prio_upper,
            "skipWorkingHoursCheck": True,
            "source": "AI_CHATBOT",
        }

        if final_category_id:
            payload["serviceCategoryId"] = final_category_id

        if reason:
            payload["reason"] = reason

        if session_id:
            payload["sessionId"] = session_id.strip()
            payload["session_id"] = session_id.strip()

        if phone:
            payload["userPhoneNumber"] = str(phone).strip()

        try:
            with httpx.Client(timeout=5.0) as client:
                response = client.post(url, json=payload, headers=self.headers)
                if response.status_code in [200, 201]:
                    envelope = response.json()
                    ticket_data = (
                        envelope.get("data")
                        if isinstance(envelope, dict) and "data" in envelope
                        else envelope
                    )
                    if ticket_data:
                        ticket_id = ticket_data.get("id") or ticket_data.get("ticket_id")
                        logger.info(
                            "[TicketingClient] Successfully created ticket in backend: ID=%s, category=%s",
                            ticket_id,
                            final_category_id,
                        )
                    return ticket_data
                else:
                    logger.warning(
                        "[TicketingClient] Failed to create ticket: status=%s, body=%s",
                        response.status_code,
                        response.text[:250],
                    )
        except Exception as e:
            logger.warning("[TicketingClient] Error calling backend internal ticket creation endpoint: %s", e)

        return None
