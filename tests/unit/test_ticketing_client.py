from unittest.mock import MagicMock, patch
from src.pipeline.stage_05_ticket.ticketing_client import TicketingClient
from src.pipeline.stage_05_ticket.ticket_agent import TicketIntakeAgent


def test_create_ticket_payload_matches_backend_dto():
    client = TicketingClient()

    mock_resp = MagicMock()
    mock_resp.status_code = 201
    mock_resp.json.return_value = {
        "status": "success",
        "data": {"id": "t-1001", "ticketPriority": "HIGH"},
    }

    with patch("httpx.Client.post", return_value=mock_resp) as mock_post, \
            patch("httpx.Client.get") as mock_get:
        res = client.create_ticket(
            title="Problem with engineer verification",
            content="Context of the conversation",
            priority="high",
            session_id="session-123",
            phone="0791234567",
            department="دائرة التأمين الصحي",
            section="قسم التأمين الصحي والمطالبات",
        )

        assert res is not None
        assert res["id"] == "t-1001"
        mock_post.assert_called_once()
        mock_get.assert_not_called()
        sent_body = mock_post.call_args[1]["json"]
        assert sent_body["ticketPriority"] == "HIGH"
        assert sent_body["sessionId"] == "session-123"
        assert sent_body["userPhoneNumber"] == "0791234567"
        assert sent_body["department"] == "دائرة التأمين الصحي"
        assert sent_body["sector"] == "دائرة التأمين الصحي"
        assert sent_body["section"] == "قسم التأمين الصحي والمطالبات"
        # jea_backend removed ServiceCategory and rejects unknown fields (forbidNonWhitelisted)
        assert "serviceCategoryId" not in sent_body
        assert "source" not in sent_body
        assert "reason" not in sent_body


def test_agent_internal_ticket_payload_has_no_service_category(monkeypatch):
    # Even with the legacy env override still set, the removed field must not be sent.
    monkeypatch.setenv("AI_TICKET_CATEGORY_ID", "legacy-cat-uuid")
    agent = TicketIntakeAgent(
        internal_tickets_url="http://mock-backend/api/v1/tickets/internal",
        internal_tickets_key="mock-key",
    )

    mock_http = MagicMock()
    mock_http.is_closed = False
    mock_res = MagicMock()
    mock_res.status_code = 201
    mock_res.json.return_value = {"data": {"id": "ticket-uuid-1"}}
    mock_http.post.return_value = mock_res
    agent._client = mock_http

    ticket_id = agent._post_to_internal_ticket_api(
        reason="user_intent",
        query="بدي اشتكي على التأمين الصحي",
        history=[],
        user={"phone": "0791234567"},
        priority="medium",
        session_id="session-123",
        department="دائرة التأمين الصحي",
        section="قسم التأمين الصحي والمطالبات",
    )

    assert ticket_id == "ticket-uuid-1"
    mock_http.post.assert_called_once()
    sent_body = mock_http.post.call_args[1]["json"]
    assert "serviceCategoryId" not in sent_body
    assert sent_body["department"] == "دائرة التأمين الصحي"
    assert sent_body["sector"] == "دائرة التأمين الصحي"
    assert sent_body["section"] == "قسم التأمين الصحي والمطالبات"
    assert sent_body["userPhoneNumber"] == "0791234567"
