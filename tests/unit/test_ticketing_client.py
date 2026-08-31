from unittest.mock import MagicMock, patch
from src.pipeline.stage_05_ticket.ticketing_client import TicketingClient


def test_fetch_categories():
    client = TicketingClient()
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "statusCode": 200,
        "data": [
            {"id": "cat-1", "enName": "Professional Services", "arName": "الخدمات المهنية"},
            {"id": "cat-2", "enName": "AI Support & Chatbot", "arName": "المساعد الذكي والذكاء الاصطناعي"},
        ],
    }

    with patch("httpx.Client.get", return_value=mock_resp) as mock_get:
        categories = client.fetch_categories()
        assert len(categories) == 2
        assert categories[0]["id"] == "cat-1"
        assert categories[1]["id"] == "cat-2"
        mock_get.assert_called_once()

        # Check in-memory caching (second call shouldn't trigger GET)
        mock_get.reset_mock()
        cached = client.fetch_categories()
        assert len(cached) == 2
        mock_get.assert_not_called()


def test_resolve_ai_category_keywords():
    client = TicketingClient()
    categories = [
        {"id": "cat-reg", "enName": "Registration & Membership", "arName": "التسجيل والعضوية"},
        {"id": "cat-ai", "enName": "AI Assistant Escalation", "arName": "تصعيد المساعد الذكي"},
        {"id": "cat-support", "enName": "Helpdesk", "arName": "الدعم الفني"},
    ]

    ai_cat = client.resolve_ai_category(categories)
    assert ai_cat == "cat-ai"


def test_resolve_ai_category_override(monkeypatch):
    client = TicketingClient()
    categories = [
        {"id": "cat-reg", "enName": "Registration", "arName": "التسجيل"},
        {"id": "cat-custom", "enName": "Custom Queue", "arName": "قسم مخصص"},
    ]

    client.override_category_id = "cat-custom"
    ai_cat = client.resolve_ai_category(categories)
    assert ai_cat == "cat-custom"


def test_map_intent_to_category_fallback():
    client = TicketingClient()
    categories = [
        {"id": "cat-prof", "enName": "Professional Services", "arName": "الخدمات المهنية والتسجيل"},
        {"id": "cat-fin", "enName": "Financial & Loans", "arName": "الشؤون المالية والقروض"},
        {"id": "cat-help", "enName": "General Support", "arName": "الدعم الفني والشكاوى"},
    ]

    assert client.map_intent_to_category("registration", categories) == "cat-prof"
    assert client.map_intent_to_category("loans", categories) == "cat-fin"
    assert client.map_intent_to_category("unknown_xyz", categories) == "cat-help"


def test_ai_swap_priority():
    client = TicketingClient()
    # When both AI category and intent categories exist, AI swap takes precedence
    categories = [
        {"id": "cat-reg", "enName": "Registration", "arName": "التسجيل"},
        {"id": "cat-ai-target", "enName": "Chatbot Inquiries", "arName": "استفسارات الشات بوت"},
    ]

    client._categories_cache = categories
    client._last_fetch_time = 9999999999.0

    target = client.get_target_category_id(intent="registration", force_ai_swap=True)
    assert target == "cat-ai-target"

    # If force_ai_swap is False, it maps to the intent
    target_no_swap = client.get_target_category_id(intent="registration", force_ai_swap=False)
    assert target_no_swap == "cat-reg"


def test_create_ticket_with_ai_swap():
    client = TicketingClient()
    categories = [
        {"id": "ai-cat-uuid", "enName": "AI Support Desk", "arName": "دعم الذكاء الاصطناعي"},
    ]
    client._categories_cache = categories
    client._last_fetch_time = 9999999999.0

    mock_resp = MagicMock()
    mock_resp.status_code = 201
    mock_resp.json.return_value = {
        "status": "success",
        "data": {"id": "t-1001", "ticketPriority": "HIGH"},
    }

    with patch("httpx.Client.post", return_value=mock_resp) as mock_post:
        res = client.create_ticket(
            title="Problem with engineer verification",
            content="Context of the conversation",
            priority="high",
            intent="registration",
            session_id="session-123",
            phone="0791234567",
            force_ai_swap=True,
        )

        assert res is not None
        assert res["id"] == "t-1001"
        mock_post.assert_called_once()
        sent_body = mock_post.call_args[1]["json"]
        assert sent_body["serviceCategoryId"] == "ai-cat-uuid"
        assert sent_body["ticketPriority"] == "HIGH"
        assert sent_body["sessionId"] == "session-123"
        assert sent_body["userPhoneNumber"] == "0791234567"
        assert "source" not in sent_body
        assert "reason" not in sent_body


def test_affairs_not_misidentified_as_ai():
    """Verify that words containing 'ai' like 'Affairs' or 'Training' are NOT misidentified as AI."""
    client = TicketingClient()
    categories = [
        {"id": "cat-affairs", "enName": "Membership & Registration Affairs", "arName": "شؤون العضوية والتسجيل"},
        {"id": "cat-training", "enName": "Training & Qualification", "arName": "التدريب والتأهيل"},
        {"id": "cat-support", "enName": "Technical Support", "arName": "الدعم الفني"},
    ]

    # None of the above are AI categories
    ai_cat = client.resolve_ai_category(categories)
    assert ai_cat is None

