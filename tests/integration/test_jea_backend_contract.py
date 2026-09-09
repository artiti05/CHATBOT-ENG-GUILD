import sys
from unittest.mock import MagicMock

if 'chromadb' not in sys.modules:
    sys.modules['chromadb'] = MagicMock()
    sys.modules['chromadb.config'] = MagicMock()

import pytest
from fastapi.testclient import TestClient
from src.api.main import app

@pytest.mark.integration
class TestJeaBackendContract:
    @pytest.fixture(autouse=True)
    def setup_client(self):
        self.client = TestClient(app)
        self.internal_headers = {
            "x-internal-token": "jea_rag_token",
            "Content-Type": "application/json",
            "Accept": "application/json"
        }

    def test_internal_token_can_access_admin_files(self):
        response = self.client.get("/api/admin/files", headers=self.internal_headers)
        assert response.status_code == 200
        data = response.json()
        assert "documents" in data
        assert "count" in data

    def test_internal_token_can_access_admin_tickets(self):
        response = self.client.get("/api/admin/tickets", headers=self.internal_headers)
        assert response.status_code == 200
        data = response.json()
        assert "tickets" in data
        assert "count" in data

    def test_health_matches_jea_backend_contract(self):
        response = self.client.get("/api/health", headers=self.internal_headers)
        assert response.status_code == 200
        data = response.json()
        assert data.get("status") == "ok"
        assert "chunks" in data

    def test_stats_matches_jea_backend_contract(self):
        response = self.client.get("/api/stats", headers=self.internal_headers)
        assert response.status_code == 200
        data = response.json()
        assert "total_documents" in data
        assert "total_chunks" in data

    def test_chat_payload_and_response_contract(self):
        payload = {
            "query": "ما هي شروط الانتساب؟",
            "history": [
                {"role": "user", "content": "مرحبا"},
                {"role": "assistant", "content": "أهلاً بك في نقابة المهندسين"}
            ],
            "top_k": 3,
            "session_id": "test-session-uuid-123",
            "user": {
                "name": "مهندس تجريبي",
                "phone": "+962790000000",
                "engineer_number": "12345"
            }
        }
        response = self.client.post("/api/chat", json=payload, headers=self.internal_headers)
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert "sources" in data
        assert "cache_hit" in data
        assert "escalated" in data
        assert data.get("session_id") == "test-session-uuid-123"
