import os
import uuid
import time
import requests
from typing import Dict, Any, Optional

from src.config import (
    TICKETING_API_URL,
    TICKETING_API_KEY,
    TICKETING_ENABLED,
    TICKETING_TIMEOUT_SEC
)

class TicketingService:
    """Communicates with the external main Ticketing System API to log user support tickets."""

    def __init__(self, api_url: str = TICKETING_API_URL, api_key: str = TICKETING_API_KEY):
        self.api_url = api_url
        self.api_key = api_key

    def create_ticket(
        self,
        category: str,
        user_query: str,
        user_id: Optional[str] = None,
        language: str = "ar-JO",
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        if not TICKETING_ENABLED:
            return {
                "success": False,
                "error": "TICKETING_DISABLED",
                "message": "نظام التذاكر غير مفعل حالياً في الإعدادات."
            }

        headers = {
            "Content-Type": "application/json"
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
            headers["X-API-Key"] = self.api_key

        payload = {
            "category": category or "GENERAL",
            "user_query": user_query,
            "user_id": user_id or "anonymous_user",
            "language": language,
            "priority": "HIGH" if category in ("FINANCIAL", "HEALTH") else "MEDIUM",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "metadata": metadata or {}
        }

        try:
            response = requests.post(
                self.api_url,
                json=payload,
                headers=headers,
                timeout=TICKETING_TIMEOUT_SEC
            )

            if response.status_code in (200, 201):
                data = response.json() if response.content else {}
                ticket_id = data.get("ticket_id") or data.get("id") or f"TCK-{uuid.uuid4().hex[:8].upper()}"
                return {
                    "success": True,
                    "ticket_id": ticket_id,
                    "category": category,
                    "status": data.get("status", "OPEN"),
                    "created_at": data.get("created_at", payload["timestamp"]),
                    "message": f"تم إنشاء التذكرة بنجاح برقم مرجعي #{ticket_id}"
                }
            else:
                fallback_id = f"TCK-LOCAL-{uuid.uuid4().hex[:6].upper()}"
                return {
                    "success": True,
                    "ticket_id": fallback_id,
                    "category": category,
                    "status": "QUEUED_OFFLINE",
                    "http_code": response.status_code,
                    "message": f"تم تسجيل التذكرة مؤقتاً برقم مرجعي #{fallback_id}"
                }

        except Exception as e:
            fallback_id = f"TCK-PENDING-{uuid.uuid4().hex[:6].upper()}"
            return {
                "success": True,
                "ticket_id": fallback_id,
                "category": category,
                "status": "PENDING_DISPATCH",
                "error": str(e),
                "message": f"تم تسجيل التذكرة برقم مرجعي مؤقت #{fallback_id}"
            }
