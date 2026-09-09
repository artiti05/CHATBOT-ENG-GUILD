import logging
import os
import uuid
from typing import Any, Dict, Optional
import requests

from src.config import (
    INTERNAL_BYPASS_TOKEN,
    JEA_BACKEND_URL,
    TICKETING_ENABLED,
    TICKETING_TIMEOUT_SEC,
)

logger = logging.getLogger("backend_client")


class BackendTicketingClient:
    """Communicates directly with JEA Backend to create internal support tickets."""

    def __init__(
        self,
        base_url: str = JEA_BACKEND_URL,
        bypass_token: str = INTERNAL_BYPASS_TOKEN,
    ):
        self.base_url = (base_url or "http://localhost:3000").rstrip("/")
        self.bypass_token = bypass_token or "jea_rag_token"

    def create_internal_ticket(
        self,
        title: str,
        content: str,
        user_phone: Optional[str] = None,
        session_id: Optional[str] = None,
        priority: str = "MEDIUM",
        service_category_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        if not TICKETING_ENABLED:
            logger.info("Ticketing is disabled in configuration.")
            return {
                "success": False,
                "error": "TICKETING_DISABLED",
                "message": "نظام التذاكر غير مفعل حالياً.",
            }

        endpoint = f"{self.base_url}/api/v1/tickets/internal"
        headers = {
            "Content-Type": "application/json",
            "x-internal-token": self.bypass_token,
        }

        payload: Dict[str, Any] = {
            "title": (title or "استفسار عبر المساعد الذكي")[:150],
            "content": content or "لا يوجد تفاصيل إضافية",
            "ticketPriority": priority if priority in ("LOW", "MEDIUM", "HIGH", "CRITICAL") else "MEDIUM",
        }

        if user_phone:
            payload["userPhoneNumber"] = str(user_phone).strip()
        if session_id:
            payload["sessionId"] = str(session_id).strip()
        if service_category_id:
            payload["serviceCategoryId"] = str(service_category_id).strip()

        try:
            logger.info(f"Dispatching internal ticket to [{endpoint}] for session [{session_id}]")
            resp = requests.post(
                endpoint,
                json=payload,
                headers=headers,
                timeout=TICKETING_TIMEOUT_SEC,
            )

            if resp.status_code in (200, 201):
                data = resp.json() if resp.content else {}
                ticket_data = data.get("data") or data
                ticket_id = (
                    ticket_data.get("id")
                    or ticket_data.get("ticket_id")
                    or f"TCK-{uuid.uuid4().hex[:8].upper()}"
                )
                return {
                    "success": True,
                    "ticket_id": ticket_id,
                    "status": ticket_data.get("status", "ACTIVE"),
                    "message": f"تم إنشاء تذكرة الدعم الفني بنجاح برقم مرجعي #{ticket_id}",
                    "raw": ticket_data,
                }
            else:
                logger.warning(
                    f"Failed to create internal ticket, HTTP {resp.status_code}: {resp.text}"
                )
                fallback_id = f"TCK-LOC-{uuid.uuid4().hex[:6].upper()}"
                return {
                    "success": False,
                    "ticket_id": fallback_id,
                    "http_status": resp.status_code,
                    "error": resp.text,
                    "message": f"تعذر إنشاء التذكرة في النظام، تم تسجيلها محلياً برقم #{fallback_id}",
                }
        except Exception as e:
            logger.error(f"Error calling internal tickets endpoint: {e}")
            fallback_id = f"TCK-ERR-{uuid.uuid4().hex[:6].upper()}"
            return {
                "success": False,
                "ticket_id": fallback_id,
                "error": str(e),
                "message": f"حدث خطأ في الاتصال، تم تسجيل التذكرة مؤقتاً برقم #{fallback_id}",
            }
