# -*- coding: utf-8 -*-
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.pipeline.stage_00_moderation.black_message_agent import BlackMessageAgent
from src.services.black_message_client import BlackMessageClient


class TestBlackMessageAgent:
    @pytest.fixture
    def mock_client(self):
        client = MagicMock(spec=BlackMessageClient)
        client.mark_black_message = MagicMock(return_value={"success": True})
        client.mark_black_message_async = AsyncMock(return_value={"success": True})
        return client

    @pytest.fixture
    def agent(self, mock_client):
        return BlackMessageAgent(client=mock_client)

    def test_heuristic_detection_toxic_arabic(self, agent):
        toxic_queries = [
            "يا كلب يا وقح",
            "كسخت النقابة",
            "سأقتلك لو ما رديت",
            "انت حمار وغبي جدا",
            "عرص",
        ]
        for query in toxic_queries:
            reason = agent._detect_heuristics(query)
            assert reason is not None, f"Failed to detect toxic query: {query}"

    def test_heuristic_detection_toxic_english(self, agent):
        toxic_queries = [
            "fuck you all",
            "you are a stupid bitch",
            "i will kill you",
            "piece of shit",
        ]
        for query in toxic_queries:
            reason = agent._detect_heuristics(query)
            assert reason is not None, f"Failed to detect toxic query: {query}"

    def test_heuristic_detection_benign(self, agent):
        benign_queries = [
            "كيف يمكنني تجديد اشتراك النقابة لعام 2026؟",
            "ما هي شروط التقاعد في نقابة المهندسين الأردنيين؟",
            "Hello, where is the engineering guild office located?",
            "اريد فتح تذكرة لقسم التأمين الصحي",
            "هل يوجد خصم للمهندسين الجدد؟",
        ]
        for query in benign_queries:
            reason = agent._detect_heuristics(query)
            assert reason is None, f"False positive on benign query: {query}"

    @pytest.mark.asyncio
    async def test_llm_classification_black_message(self, agent):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": '{"is_black": true, "reason": "severe insult", "category": "harassment"}'}}]
        }

        with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_resp):
            is_black, reason, category = await agent.is_black_message_async("رسالة غير لائقة جدا وغير مؤدبة")
            assert is_black is True
            assert reason == "severe insult"
            assert category == "harassment"

    @pytest.mark.asyncio
    async def test_llm_classification_clean_message(self, agent):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": '{"is_black": false, "reason": "clean inquiry", "category": "clean"}'}}]
        }

        with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_resp):
            is_black, reason, category = await agent.is_black_message_async("ما هي دورات تدريب المهندسين المتوفرة؟")
            assert is_black is False
            assert category == "clean"

    def test_sync_llm_classification(self, agent):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": '{"is_black": true, "reason": "hate speech", "category": "hate"}'}}]
        }

        with patch("httpx.Client.post", return_value=mock_resp):
            is_black, reason, category = agent.is_black_message("رسالة كراهية")
            assert is_black is True
            assert reason == "hate speech"
            assert category == "hate"

    @pytest.mark.asyncio
    async def test_handle_black_message_async_dispatches_client(self, agent, mock_client):
        res = await agent.handle_black_message_async(
            query="fuck you",
            message_id="msg-12345",
            session_id="sess-abc",
            reason="offensive language",
            category="profanity"
        )
        assert res["is_black_message"] is True
        assert res["message_id"] == "msg-12345"
        assert res["route"]["route"] == "BLACK_MESSAGE"
        mock_client.mark_black_message_async.assert_awaited_once_with(
            message_id="msg-12345",
            is_black=True,
            reason="offensive language",
            session_id="sess-abc"
        )

    def test_handle_black_message_sync_dispatches_client(self, agent, mock_client):
        res = agent.handle_black_message(
            query="fuck you",
            message_id="msg-sync-123",
            session_id="sess-sync",
            reason="offensive language",
            category="profanity"
        )
        assert res["is_black_message"] is True
        assert res["message_id"] == "msg-sync-123"
        mock_client.mark_black_message.assert_called_once_with(
            message_id="msg-sync-123",
            is_black=True,
            reason="offensive language",
            session_id="sess-sync"
        )

    @pytest.mark.asyncio
    async def test_handle_black_message_async_without_message_id(self, agent, mock_client):
        # Should complete gracefully without calling the backend client
        res = await agent.handle_black_message_async(
            query="offensive content",
            message_id=None,
            reason="offensive language",
        )
        assert res["is_black_message"] is True
        assert res["message_id"] is None
        mock_client.mark_black_message_async.assert_not_called()

    @pytest.mark.asyncio
    async def test_handle_black_message_async_swallows_client_exception(self, agent, mock_client):
        mock_client.mark_black_message_async.side_effect = RuntimeError("Backend down")
        # Should not raise exception
        res = await agent.handle_black_message_async(
            query="offensive content",
            message_id="msg-err",
            reason="offensive language",
        )
        assert res["is_black_message"] is True
        mock_client.mark_black_message_async.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_disabled_moderation_passes_everything(self, mock_client):
        with patch("src.pipeline.stage_00_moderation.black_message_agent.ENABLE_BLACK_MESSAGE_MODERATION", False):
            disabled_agent = BlackMessageAgent(client=mock_client)
            is_black, reason, category = await disabled_agent.is_black_message_async("يا كلب يا وقح fuck you")
            assert is_black is False
            assert reason is None
            assert category == "clean"
