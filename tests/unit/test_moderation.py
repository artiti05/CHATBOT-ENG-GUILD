from unittest.mock import MagicMock, patch

import pytest
import requests

from src.pipeline.stage_04_answer.generator_agent import (
    ReasoningStreamFilter,
    clean_formatting,
    detect_moderation_flag,
    strip_reasoning,
)
from src.services.moderation_notifier import ModerationAlertDispatcher


@pytest.mark.unit
class TestModerationFlagging:
    def test_strip_flag_tags(self):
        text = "<flag>offensive_detected</flag>The annual renewal fee is 30 JOD."
        assert strip_reasoning(text) == "The annual renewal fee is 30 JOD."

    def test_strip_flag_and_think_combined(self):
        text = "<think>User is insulting the bot</think><flag>offensive_detected</flag>Direct answer here"
        assert strip_reasoning(text) == "Direct answer here"

    def test_clean_formatting_strips_flag(self):
        raw = "### Details\n<flag>offensive_detected</flag>**The answer is 30 JOD** [المصدر 1]"
        cleaned = clean_formatting(raw)
        assert "<flag>" not in cleaned
        assert "</flag>" not in cleaned
        assert "offensive_detected" not in cleaned
        assert "The answer is 30 JOD" in cleaned
        assert "[المصدر 1]" not in cleaned

    def test_detect_moderation_flag(self):
        assert detect_moderation_flag("<flag>offensive_detected</flag>Answer") is True
        assert detect_moderation_flag("<flag>abusive</flag>") is True
        assert detect_moderation_flag("Normal clean answer") is False

    def test_stream_filter_strips_flag_and_tracks_detection(self):
        filter_agent = ReasoningStreamFilter()
        stream_tokens = [
            "<flag>", "offensive_", "detected", "</flag>",
            "The ", "annual ", "renewal ", "fee ", "is ", "30 JOD."
        ]
        emitted = []
        for token in stream_tokens:
            out = filter_agent.feed(token)
            if out:
                emitted.append(out)
        tail = filter_agent.flush()
        if tail:
            emitted.append(tail)

        full_output = "".join(emitted)
        assert "<flag>" not in full_output
        assert "offensive_detected" not in full_output
        assert "The annual renewal fee is 30 JOD." in full_output
        assert filter_agent.flag_detected is True

    def test_stream_filter_clean_prompt(self):
        filter_agent = ReasoningStreamFilter()
        stream_tokens = ["The ", "fee ", "is ", "30 JOD."]
        emitted = [filter_agent.feed(t) for t in stream_tokens]
        emitted.append(filter_agent.flush())
        full_output = "".join(emitted)

        assert full_output == "The fee is 30 JOD."
        assert filter_agent.flag_detected is False


@pytest.mark.unit
class TestModerationAlertDispatcher:
    def test_build_payload(self):
        payload = ModerationAlertDispatcher.build_payload(
            request_id="req_123",
            query="you thieves",
            flag_type="offensive_deescalated",
            answer="The fee is 30 JOD.",
            language="en",
            extra_metadata={"route": "RAG"}
        )
        assert payload["event"] == "moderation_flag"
        assert payload["request_id"] == "req_123"
        assert payload["query"] == "you thieves"
        assert payload["language"] == "en"
        assert payload["flag_type"] == "offensive_deescalated"
        assert payload["answer"] == "The fee is 30 JOD."
        assert payload["metadata"]["route"] == "RAG"
        assert "timestamp" in payload

    @patch("src.services.moderation_notifier.requests.post")
    def test_dispatch_sync_success(self, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_post.return_value = mock_response

        with patch("src.services.moderation_notifier.MODERATION_WEBHOOK_URL", "https://example.com/mod-webhook"), \
             patch("src.services.moderation_notifier.MODERATION_WEBHOOK_TOKEN", "test-token"), \
             patch("src.services.moderation_notifier.ENABLE_MODERATION_ALERTS", True):

            success = ModerationAlertDispatcher.dispatch(
                request_id="req_999",
                query="test bad query",
                flag_type="offensive_deescalated",
                answer="Calm response",
                sync=True
            )
            assert success is True
            mock_post.assert_called_once()
            args, kwargs = mock_post.call_args
            assert args[0] == "https://example.com/mod-webhook"
            assert kwargs["headers"]["Authorization"] == "Bearer test-token"
            assert kwargs["json"]["query"] == "test bad query"

    @patch("src.services.moderation_notifier.requests.post")
    def test_dispatch_handles_timeout_gracefully(self, mock_post):
        mock_post.side_effect = requests.exceptions.Timeout("Connection timeout")

        with patch("src.services.moderation_notifier.MODERATION_WEBHOOK_URL", "https://example.com/mod-webhook"), \
             patch("src.services.moderation_notifier.ENABLE_MODERATION_ALERTS", True):

            success = ModerationAlertDispatcher.dispatch(
                request_id="req_timeout",
                query="query",
                sync=True
            )
            assert success is False  # Must not crash the app

    def test_dispatch_skipped_when_no_url(self):
        with patch("src.services.moderation_notifier.MODERATION_WEBHOOK_URL", ""):
            res = ModerationAlertDispatcher.dispatch(request_id="req_none", query="test", sync=True)
            assert res is False
