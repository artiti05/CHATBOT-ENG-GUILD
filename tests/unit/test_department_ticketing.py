import json
import pytest
from unittest.mock import MagicMock, patch
from src.pipeline.stage_05_ticket.ticketing_client import (
    TicketingClient,
    DEFAULT_FALLBACK_SECTORS_STRUCTURE,
)
from src.pipeline.stage_05_ticket.ticket_agent import TicketIntakeAgent
from src.cache_db.document_registry import TicketRegistry


class TestDepartmentTicketing:

    def test_fetch_sectors_structure_success(self):
        client = TicketingClient(base_url="http://mock-backend:3000")
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "sectors": ["دائرة التأمين الصحي", "الدائرة المالية"],
            "departments": ["دائرة التأمين الصحي", "الدائرة المالية"],
            "sections": ["قسم التأمين الصحي والمطالبات"],
            "departmentSections": {"دائرة التأمين الصحي": ["قسم التأمين الصحي والمطالبات"]},
        }

        with patch("httpx.Client") as mock_httpx:
            mock_httpx.return_value.__enter__.return_value.get.return_value = mock_response
            res = client.fetch_sectors_structure(force_refresh=True)

        assert "departments" in res
        assert "دائرة التأمين الصحي" in res["departments"]
        assert "الدائرة المالية" in res["departments"]
        assert len(res["sections"]) == 1

        # Check caching works without calling http again
        res_cached = client.fetch_sectors_structure(force_refresh=False)
        assert res_cached == res

    def test_fetch_sectors_structure_fallback(self):
        client = TicketingClient(base_url="http://unreachable-backend:3000")
        with patch("httpx.Client", side_effect=Exception("Connection refused")):
            res = client.fetch_sectors_structure(force_refresh=True)

        assert res is not None
        assert "departments" in res
        assert len(res["departments"]) > 0
        assert "دائرة التأمين الصحي" in res["departments"]

    def test_classify_department_llm_json(self):
        agent = TicketIntakeAgent()
        agent.ticketing_client.fetch_sectors_structure = MagicMock(return_value={
            "departments": ["دائرة التأمين الصحي", "الدائرة المالية"],
            "sectors": ["دائرة التأمين الصحي", "الدائرة المالية"],
            "sections": ["قسم التأمين الصحي والمطالبات"],
            "departmentSections": {"دائرة التأمين الصحي": ["قسم التأمين الصحي والمطالبات"]},
        })
        mock_http = MagicMock()
        mock_http.is_closed = False
        mock_res = MagicMock()
        mock_res.status_code = 200
        mock_res.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": json.dumps({
                            "department": "دائرة التأمين الصحي",
                            "section": "قسم التأمين الصحي والمطالبات"
                        })
                    }
                }
            ]
        }
        mock_http.post.return_value = mock_res
        agent._client = mock_http

        dept, sec = agent.classify_department("عندي مشكلة ببطاقة التأمين الصحي للمهندسين")
        assert dept == "دائرة التأمين الصحي"
        assert sec == "قسم التأمين الصحي والمطالبات"

    def test_classify_department_keyword_fallback(self):
        agent = TicketIntakeAgent()
        agent.ticketing_client.fetch_sectors_structure = MagicMock(return_value={
            "departments": ["دائرة التأمين الصحي", "الدائرة المالية"],
            "sectors": ["دائرة التأمين الصحي", "الدائرة المالية"],
            "sections": [],
            "departmentSections": {},
        })
        mock_http = MagicMock()
        mock_http.is_closed = False
        mock_http.post.side_effect = Exception("LLM timeout")
        agent._client = mock_http

        dept, sec = agent.classify_department("بدي استفسر عن الرواتب والاشتراكات في الدائرة المالية")
        assert dept == "الدائرة المالية"
        assert sec is None

    def test_escalate_stores_department_and_section(self, tmp_path):
        db_file = tmp_path / "test_tickets.db"
        registry = TicketRegistry(db_path=db_file)
        agent = TicketIntakeAgent()
        agent.registry = registry

        # Mock classify_department
        agent.classify_department = MagicMock(return_value=("دائرة التأمين الصحي", "قسم التأمين الصحي والمطالبات"))
        agent._post_to_internal_ticket_api = MagicMock(return_value="ticket-uuid-999")

        ticket_id, msg = agent.escalate(
            reason="user_intent",
            query="بدي اشتكي على التأمين الصحي",
            history=[{"role": "user", "content": "مرحبا"}],
            user={"name": "المهندس أحمد", "phone": "0791234567"},
            session_id="session-123",
        )

        assert ticket_id == "ticket-uuid-999"
        assert "تم فتح تذكرة" in msg

        # Verify internal API call included department & section
        agent._post_to_internal_ticket_api.assert_called_once()
        call_kwargs = agent._post_to_internal_ticket_api.call_args[1]
        assert call_kwargs["department"] == "دائرة التأمين الصحي"
        assert call_kwargs["section"] == "قسم التأمين الصحي والمطالبات"

        # Verify SQLite stored department & section
        saved_tickets = registry.list_tickets(department="دائرة التأمين الصحي")
        assert len(saved_tickets) == 1
        assert saved_tickets[0]["department"] == "دائرة التأمين الصحي"
        assert saved_tickets[0]["section"] == "قسم التأمين الصحي والمطالبات"
