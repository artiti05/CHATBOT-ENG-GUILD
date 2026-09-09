import pytest
import json
from unittest.mock import AsyncMock, patch, MagicMock
from src.pipeline.stage_04_answer.generator_agent import ResponseGeneratorAgent
from src.pipeline.stage_05_ticket.ticket_agent import TicketIntakeAgent

@pytest.mark.asyncio
async def test_response_generator_openai_payload():
    agent = ResponseGeneratorAgent(
        chat_url="https://api.openai.com/v1/chat/completions",
        model="gpt-4o-mini",
        headers={"Authorization": "Bearer sk-test-key", "Content-Type": "application/json"},
        provider="openai"
    )
    payload = agent._build_payload("Test Prompt", stream=False)
    assert payload["model"] == "gpt-4o-mini"
    assert payload["messages"] == [{"role": "user", "content": "Test Prompt"}]
    assert payload["stream"] is False
    assert "chat_template_kwargs" not in payload

@pytest.mark.asyncio
async def test_response_generator_openai_generate_success():
    agent = ResponseGeneratorAgent(
        chat_url="https://api.openai.com/v1/chat/completions",
        model="gpt-4o-mini",
        headers={"Authorization": "Bearer sk-test-key", "Content-Type": "application/json"},
        provider="openai"
    )

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "choices": [{"message": {"content": "This is a direct response."}}]
    }

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_resp) as mock_post:
        result, success = await agent.generate("Hello")
        assert success is True
        assert result == "This is a direct response."
        mock_post.assert_called_once()
        args, kwargs = mock_post.call_args
        assert kwargs["headers"]["Authorization"] == "Bearer sk-test-key"

@pytest.mark.asyncio
async def test_ticket_agent_openai_detect_intent():
    ticket_agent = TicketIntakeAgent(
        chat_url="https://api.openai.com/v1/chat/completions",
        model="gpt-4o-mini",
        headers={"Authorization": "Bearer sk-test-key", "Content-Type": "application/json"},
        provider="openai"
    )

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "choices": [{"message": {"content": "نعم"}}]
    }

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_resp):
        is_escalation = await ticket_agent.detect_intent("قدمت معاملة وما حد رد علي بدي موظف")
        assert is_escalation is True
