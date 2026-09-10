import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import requests

from src.config import (
    ENABLE_MODERATION_ALERTS,
    MODERATION_WEBHOOK_TIMEOUT,
    MODERATION_WEBHOOK_TOKEN,
    MODERATION_WEBHOOK_URL,
)

logger = logging.getLogger("moderation_notifier")


class ModerationAlertDispatcher:
    """
    Asynchronous, non-blocking webhook alert dispatcher for moderation events.
    Sends standardized JSON payloads to configurable external endpoints when offensive
    or abusive prompts are flagged.
    """

    _executor = ThreadPoolExecutor(max_workers=3, thread_name_prefix="mod_webhook_worker")

    @classmethod
    def build_payload(
        cls,
        request_id: str,
        query: str,
        flag_type: str,
        answer: str = "",
        language: str = "ar",
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        return {
            "event": "moderation_flag",
            "request_id": request_id or "unknown",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "query": query,
            "language": language,
            "flag_type": flag_type,
            "answer": answer,
            "metadata": extra_metadata or {},
        }

    @classmethod
    def _send_http_request(cls, payload: Dict[str, Any], webhook_url: str, token: str, timeout: float) -> bool:
        if not webhook_url:
            return False

        headers = {
            "Content-Type": "application/json",
            "User-Agent": "Eng-Guild-Moderation-Notifier/1.0",
        }
        if token:
            headers["Authorization"] = f"Bearer {token}"
            headers["X-Moderation-Token"] = token

        try:
            res = requests.post(webhook_url, json=payload, headers=headers, timeout=timeout)
            if res.status_code in (200, 201, 202, 204):
                logger.info(f"[Moderation Alert] Dispatched event {payload.get('request_id')} successfully (HTTP {res.status_code}).")
                return True
            else:
                logger.warning(
                    f"[Moderation Alert] Webhook returned non-200 status: {res.status_code} - {res.text[:200]}"
                )
                return False
        except requests.exceptions.Timeout:
            logger.warning(f"[Moderation Alert] Webhook timed out after {timeout}s: {webhook_url}")
            return False
        except Exception as e:
            logger.warning(f"[Moderation Alert] Failed to dispatch webhook to {webhook_url}: {e}")
            return False

    @classmethod
    def dispatch(
        cls,
        request_id: str,
        query: str,
        flag_type: str = "offensive_deescalated",
        answer: str = "",
        language: str = "ar",
        extra_metadata: Optional[Dict[str, Any]] = None,
        sync: bool = False,
    ) -> Optional[bool]:
        """
        Dispatches a moderation alert event.
        Runs asynchronously by default to ensure zero user-facing latency.
        """
        if not ENABLE_MODERATION_ALERTS:
            return False

        url = MODERATION_WEBHOOK_URL.strip()
        if not url:
            logger.debug("[Moderation Alert] No MODERATION_WEBHOOK_URL configured. Skipping dispatch.")
            return False

        payload = cls.build_payload(
            request_id=request_id,
            query=query,
            flag_type=flag_type,
            answer=answer,
            language=language,
            extra_metadata=extra_metadata,
        )

        if sync:
            return cls._send_http_request(payload, url, MODERATION_WEBHOOK_TOKEN, MODERATION_WEBHOOK_TIMEOUT)

        # Asynchronous execution in background thread pool
        cls._executor.submit(
            cls._send_http_request,
            payload,
            url,
            MODERATION_WEBHOOK_TOKEN,
            MODERATION_WEBHOOK_TIMEOUT,
        )
        return True
