import sys
from unittest.mock import patch, MagicMock

if 'chromadb' not in sys.modules:
    sys.modules['chromadb'] = MagicMock()
    sys.modules['chromadb.config'] = MagicMock()

import pytest
import sqlite3
from src.cache_db.document_registry import TicketRegistry
from src.pipeline.stage_05_ticket.ticket_agent import TicketIntakeAgent
from src.api.routes_chat import ChatRequest
from src.core.rag_engine import RAGChatbot

@pytest.fixture
def temp_ticket_registry(tmp_path):
    db_file = tmp_path / "test_registry.db"
    registry = TicketRegistry(db_path=db_file)
    return registry

def test_chat_request_schema_supports_session_id():
    payload = {
        "query": "بدي احكي مع موظف",
        "session_id": "test-session-12345",
        "user": {
            "user_id": "usr_99",
            "name": "المهندس فادي",
            "phone": "0790000000"
        }
    }
    req = ChatRequest(**payload)
    assert req.query == "بدي احكي مع موظف"
    assert req.session_id == "test-session-12345"
    assert req.user.name == "المهندس فادي"

def test_ticket_registry_stores_and_retrieves_session_id(temp_ticket_registry):
    ticket_id = temp_ticket_registry.create_ticket(
        reason="user_intent",
        query="عندي مشكلة في التسجيل",
        history=[{"role": "user", "content": "مرحبا"}],
        user={"name": "سامي", "phone": "0780000000", "user_id": "u1"},
        priority="high",
        session_id="session-abc-xyz"
    )
    assert ticket_id > 0

    ticket = temp_ticket_registry.get_ticket(ticket_id)
    assert ticket is not None
    assert ticket["id"] == ticket_id
    assert ticket["session_id"] == "session-abc-xyz"
    assert ticket["reason"] == "user_intent"
    assert ticket["priority"] == "high"

@pytest.mark.asyncio
async def test_ticket_intake_agent_local_escalate(temp_ticket_registry):
    agent = TicketIntakeAgent(internal_tickets_url="")
    agent.registry = temp_ticket_registry

    ticket_id, message = agent.escalate(
        reason="user_intent",
        query="بدي اشتكي",
        history=[],
        user={"name": "رامي"},
        session_id="sess-456"
    )
    assert ticket_id > 0
    assert f"#{ticket_id}" in message

    ticket = temp_ticket_registry.get_ticket(ticket_id)
    assert ticket["session_id"] == "sess-456"

def test_ticket_intake_agent_internal_api_success(temp_ticket_registry):
    agent = TicketIntakeAgent(
        internal_tickets_url="http://mock-internal-api/api/v1/tickets",
        internal_tickets_key="mock-key"
    )
    agent.registry = temp_ticket_registry
    # Department classification makes its own LLM POST; stub it so only the ticket POST is observed.
    agent.classify_department = MagicMock(return_value=(None, None))

    mock_resp = MagicMock()
    mock_resp.status_code = 201
    mock_resp.json.return_value = {"status": "success", "ticket_id": 9999}

    with patch("httpx.Client.post", return_value=mock_resp) as mock_post:
        ticket_id, message = agent.escalate(
            reason="user_intent",
            query="بدي اشتكي",
            history=[],
            user={"name": "رامي"},
            session_id="sess-789"
        )
        assert ticket_id == 9999
        assert "#9999" in message
        mock_post.assert_called_once()
        sent_payload = mock_post.call_args[1]["json"]
        assert sent_payload.get("sessionId") == "sess-789"
        assert "سبب التصعيد: user_intent" in sent_payload["content"]
        assert "serviceCategoryId" not in sent_payload


@pytest.mark.asyncio
async def test_rag_chatbot_answer_question_escalated_metadata():
    chatbot = RAGChatbot.__new__(RAGChatbot)
    mock_engine = MagicMock()
    mock_res = {
        "answer": "تم فتح تذكرة رقم #42 بنجاح",
        "sources": [],
        "cache_hit": False,
        "escalated": True,
        "is_escalated": True,
        "ticket_id": 42,
        "escalation_reason": "user_intent",
        "session_id": "sess-live-99",
        "profiling": {}
    }
    mock_engine.process_query = MagicMock(return_value=mock_res)

    async def mock_async_process_query(*args, **kwargs):
        return mock_res

    mock_engine.process_query = mock_async_process_query
    chatbot.engine = mock_engine

    response = await chatbot.answer_question(
        query="بدي احكي مع موظف",
        session_id="sess-live-99",
        user={"name": "أحمد"}
    )
    assert response["escalated"] is True
    assert response["is_escalated"] is True
    assert response["ticket_id"] == 42
    assert response["escalation_reason"] == "user_intent"
    assert response["session_id"] == "sess-live-99"
    assert "42" in response["answer"]
