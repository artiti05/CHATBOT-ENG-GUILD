import pytest
from unittest.mock import MagicMock, patch
from src.services.black_message_client import BlackMessageClient


def test_black_message_client_url_construction():
    client = BlackMessageClient(base_url="http://mock-backend:3000")
    # Standard URL
    url = client._resolve_url("msg-123")
    assert url == "http://mock-backend:3000/api/v1/live-chat/messages/msg-123/black/internal"

    # Session-scoped URL
    url_session = client._resolve_url("msg-123", session_id="sess-abc")
    assert url_session == "http://mock-backend:3000/api/v1/live-chat/sessions/sess-abc/messages/msg-123/black/internal"


def test_mark_black_message_success():
    client = BlackMessageClient(base_url="http://mock-backend:3000")
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "success": True,
        "data": {
            "id": "msg-123",
            "sessionId": "sess-123",
            "isBlackMessage": True,
            "senderType": "USER",
        },
    }

    with patch("httpx.Client.patch", return_value=mock_resp) as mock_patch:
        res = client.mark_black_message(
            message_id="msg-123",
            is_black=True,
            reason="Offensive content violation",
        )

        assert res is not None
        assert res["id"] == "msg-123"
        assert res["isBlackMessage"] is True
        mock_patch.assert_called_once()
        args, kwargs = mock_patch.call_args
        assert kwargs["json"]["isBlackMessage"] is True
        assert kwargs["json"]["reason"] == "Offensive content violation"
        assert kwargs["headers"]["x-internal-token"] == client.token


def test_unflag_message_success():
    client = BlackMessageClient(base_url="http://mock-backend:3000")
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "success": True,
        "data": {
            "id": "msg-123",
            "isBlackMessage": False,
        },
    }

    with patch("httpx.Client.patch", return_value=mock_resp) as mock_patch:
        res = client.unflag_message("msg-123", reason="False positive review")
        assert res is not None
        assert res["isBlackMessage"] is False
        kwargs = mock_patch.call_args[1]
        assert kwargs["json"]["isBlackMessage"] is False


@pytest.mark.asyncio
async def test_mark_black_message_async_success():
    client = BlackMessageClient(base_url="http://mock-backend:3000")
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "success": True,
        "data": {
            "id": "msg-async-456",
            "sessionId": "sess-456",
            "isBlackMessage": True,
        },
    }

    with patch("httpx.AsyncClient.patch", return_value=mock_resp) as mock_patch:
        res = await client.mark_black_message_async(
            message_id="msg-async-456",
            is_black=True,
            reason="Async moderation check",
            session_id="sess-456",
        )

        assert res is not None
        assert res["id"] == "msg-async-456"
        assert res["sessionId"] == "sess-456"
        mock_patch.assert_called_once()


def test_mark_black_message_by_body():
    client = BlackMessageClient(base_url="http://mock-backend:3000")
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "success": True,
        "data": {"id": "msg-body-789", "isBlackMessage": True},
    }

    with patch("httpx.Client.post", return_value=mock_resp) as mock_post:
        res = client.mark_black_message_by_body(
            message_id="msg-body-789",
            is_black=True,
            reason="Body driven moderation",
        )

        assert res is not None
        assert res["id"] == "msg-body-789"
        mock_post.assert_called_once()
        sent_json = mock_post.call_args[1]["json"]
        assert sent_json["messageId"] == "msg-body-789"
        assert sent_json["isBlackMessage"] is True
