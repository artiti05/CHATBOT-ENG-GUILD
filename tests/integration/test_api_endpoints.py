import pytest
from fastapi.testclient import TestClient

from src.api.dependencies import ADMIN_API_KEY, USER_API_KEY
from src.api.main import app


@pytest.mark.integration
class TestAPIEndpoints:
    @pytest.fixture(autouse=True)
    def setup_client(self):
        self.client = TestClient(app)
        self.user_headers = {"X-API-Key": USER_API_KEY or "456def"}
        self.admin_headers = {"X-API-Key": ADMIN_API_KEY or "123abcadmin"}

    def test_root_ui_endpoint(self):
        response = self.client.get("/")
        assert response.status_code == 200
        assert "نقابة المهندسين الأردنيين" in response.text
        assert "TEST_QUESTIONS_BANK" in response.text

    def test_health_endpoint(self):
        response = self.client.get("/api/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "chunks" in data

    def test_stats_endpoint(self):
        response = self.client.get("/api/stats")
        assert response.status_code == 200
        data = response.json()
        assert "total_documents" in data
        assert "total_chunks" in data

    def test_admin_unauthorized(self):
        response = self.client.get("/api/admin/files", headers={"X-API-Key": "invalid_key"})
        assert response.status_code in (401, 403, 500)

    def test_chat_empty_query(self):
        response = self.client.post("/api/chat", json={"query": ""}, headers=self.user_headers)
        assert response.status_code == 400

    def test_chat_valid_query(self):
        response = self.client.post(
            "/api/chat",
            json={"query": "ما هو سلم رواتب المهندسين؟", "top_k": 5},
            headers=self.user_headers
        )
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert "sources" in data
        assert "metadata" in data

    def test_chat_get_query(self):
        response = self.client.get(
            "/api/chat",
            params={"query": "ما هو سلم رواتب المهندسين؟", "top_k": 5},
            headers=self.user_headers
        )
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert "sources" in data

    def test_query_alias(self):
        response = self.client.get(
            "/api/query",
            params={"query": "شروط الانتساب"},
            headers=self.user_headers
        )
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data

    def test_search_kb(self):
        response = self.client.post(
            "/api/search",
            json={"query": "شروط الانتساب", "top_k": 3},
            headers=self.user_headers
        )
        assert response.status_code == 200
        data = response.json()
        assert "results" in data

    def test_admin_list_files(self):
        response = self.client.get("/api/admin/files", headers=self.admin_headers)
        assert response.status_code == 200
        data = response.json()
        assert "documents" in data
