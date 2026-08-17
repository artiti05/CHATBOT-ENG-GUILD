import pytest
from unittest.mock import patch, MagicMock
from src.services.ticketing_service import TicketingService

@pytest.mark.unit
class TestTicketingService:
    def setup_method(self):
        self.service = TicketingService(
            api_url="https://api.guild.org/v1/tickets",
            api_key="test_api_key_123"
        )

    @patch("src.services.ticketing_service.requests.post")
    def test_create_ticket_success(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 201
        mock_resp.content = True
        mock_resp.json.return_value = {
            "ticket_id": "TCK-889900",
            "status": "OPEN",
            "created_at": "2026-08-17T12:00:00Z"
        }
        mock_post.return_value = mock_resp

        res = self.service.create_ticket(
            category="FINANCIAL",
            user_query="أريد رفع تذكرة خلل بدفع الرسوم"
        )

        assert res["success"] is True
        assert res["ticket_id"] == "TCK-889900"
        assert res["category"] == "FINANCIAL"
        assert res["status"] == "OPEN"

    @patch("src.services.ticketing_service.requests.post")
    def test_create_ticket_http_error_fallback(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 500
        mock_post.return_value = mock_resp

        res = self.service.create_ticket(
            category="TECHNICAL",
            user_query="افتحلي تذكرة عشان عطل الدخول"
        )

        assert res["success"] is True
        assert res["ticket_id"].startswith("TCK-LOCAL-")
        assert res["status"] == "QUEUED_OFFLINE"

    @patch("src.services.ticketing_service.requests.post", side_effect=Exception("ConnectionRefusedError"))
    def test_create_ticket_exception_fallback(self, mock_post):
        res = self.service.create_ticket(
            category="HEALTH",
            user_query="Open ticket for health insurance"
        )

        assert res["success"] is True
        assert res["ticket_id"].startswith("TCK-PENDING-")
        assert res["status"] == "PENDING_DISPATCH"
