import pytest
from src.rag_chatbot_engine import RAGChatbotEngine

@pytest.mark.integration
class TestTicketingIntegration:
    def setup_method(self):
        self.engine = RAGChatbotEngine()

    def test_explicit_ticket_query_flow(self):
        query = "أرغب برفع تذكرة لمشكلة مالية في خصم الرسوم"
        res = self.engine.process_query(query)

        assert res["route"]["route"] == "TICKETING"
        assert res["ticket"] is not None
        assert res["ticket"]["success"] is True
        assert "ticket_id" in res["ticket"]
        assert "الرقم المرجعي للتذكرة" in res["answer"]
        assert "مالية" in res["answer"]

    def test_general_query_not_routed_to_ticketing(self):
        query = "ما هي شروط تسجيل المهندسين الأردنيين؟"
        res = self.engine.process_query(query)

        assert res.get("route", {}).get("route") != "TICKETING"
        assert res.get("ticket") is None
