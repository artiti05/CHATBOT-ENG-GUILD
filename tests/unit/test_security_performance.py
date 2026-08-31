import sys
from unittest.mock import patch, MagicMock

if 'chromadb' not in sys.modules:
    sys.modules['chromadb'] = MagicMock()
    sys.modules['chromadb.config'] = MagicMock()

import pytest
import sqlite3
import time
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

from src.api.security import (
    safe_compare,
    verify_authenticated_client,
    sanitize_user_input,
    SecurityHeadersMiddleware,
    InMemoryRateLimiterMiddleware,
)
from src.cache_db.document_registry import DocumentRegistry, TicketRegistry

def test_safe_compare_constant_time():
    assert safe_compare("my_secret_token", "my_secret_token") is True
    assert safe_compare("my_secret_token", "wrong_token") is False
    assert safe_compare("", "token") is False
    assert safe_compare(None, "token") is False

def test_sanitize_user_input_cleans_and_truncates():
    # Null bytes and control characters
    malicious = "Hello\x00World\x07! \x1fTesting\nNormal"
    cleaned = sanitize_user_input(malicious)
    assert "\x00" not in cleaned
    assert "\x07" not in cleaned
    assert "\x1f" not in cleaned
    assert "HelloWorld! Testing" in cleaned

    # Length truncation
    long_query = "A" * 5000
    truncated = sanitize_user_input(long_query, max_length=2000)
    assert len(truncated) == 2000

def test_verify_authenticated_client_with_internal_token():
    with patch("src.api.security.INTERNAL_BYPASS_TOKEN", "jea_rag_token"):
        role = verify_authenticated_client(
            x_api_key=None,
            auth_header=None,
            internal_token="jea_rag_token"
        )
        assert role == "internal_service"

def test_verify_authenticated_client_with_bearer_token():
    with patch("src.api.security.USER_API_KEY", "secret_user_key"):
        role = verify_authenticated_client(
            x_api_key=None,
            auth_header="Bearer secret_user_key",
            internal_token=None
        )
        assert role == "user"

def test_security_headers_middleware():
    test_app = FastAPI()
    test_app.add_middleware(SecurityHeadersMiddleware)

    @test_app.get("/ping")
    def ping():
        return {"ping": "pong"}

    client = TestClient(test_app)
    res = client.get("/ping")
    assert res.status_code == 200
    assert res.headers.get("X-Content-Type-Options") == "nosniff"
    assert res.headers.get("X-Frame-Options") == "SAMEORIGIN"
    assert "X-Process-Time" in res.headers
    assert "X-Request-ID" in res.headers

def test_rate_limiter_middleware_enforces_limit_and_allows_bypass():
    test_app = FastAPI()
    test_app.add_middleware(InMemoryRateLimiterMiddleware, max_requests=3, window_seconds=60)

    @test_app.get("/limited")
    def limited_endpoint():
        return {"status": "ok"}

    client = TestClient(test_app)

    # 3 allowed requests
    for _ in range(3):
        res = client.get("/limited")
        assert res.status_code == 200

    # 4th request should be rate-limited
    res_limited = client.get("/limited")
    assert res_limited.status_code == 429
    assert "Rate limit exceeded" in res_limited.text

    # Bypass with internal token
    with patch("src.api.security.INTERNAL_BYPASS_TOKEN", "jea_rag_token"):
        res_bypassed = client.get("/limited", headers={"x-internal-token": "jea_rag_token"})
        assert res_bypassed.status_code == 200

def test_sqlite_wal_mode_and_concurrency_pragmas(tmp_path):
    db_file = tmp_path / "test_perf.db"
    reg = DocumentRegistry(db_path=db_file)
    with reg._get_connection() as conn:
        journal_mode = conn.execute("PRAGMA journal_mode;").fetchone()[0]
        busy_timeout = conn.execute("PRAGMA busy_timeout;").fetchone()[0]
        # In SQLite, WAL mode returns 'wal'
        assert journal_mode.lower() == "wal"
        assert busy_timeout >= 5000
