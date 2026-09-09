import os
import secrets
from typing import Optional
from fastapi import Security, HTTPException
from fastapi.security import APIKeyHeader
from dotenv import load_dotenv

load_dotenv()

ADMIN_API_KEY = os.getenv("ADMIN_API_KEY", "")
USER_API_KEY = os.getenv("USER_API_KEY", "")
INTERNAL_BYPASS_TOKEN = os.getenv("INTERNAL_BYPASS_TOKEN", "jea_rag_token")

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
internal_token_header = APIKeyHeader(name="x-internal-token", auto_error=False)
auth_header = APIKeyHeader(name="Authorization", auto_error=False)

def safe_compare(val1: Optional[str], val2: Optional[str]) -> bool:
    if not val1 or not val2:
        return False
    return secrets.compare_digest(val1.strip(), val2.strip())

def verify_admin_key(
    key: Optional[str] = Security(api_key_header),
    internal_token: Optional[str] = Security(internal_token_header),
    authorization: Optional[str] = Security(auth_header),
):
    # 1. Allow internal microservice token
    if internal_token and safe_compare(internal_token, INTERNAL_BYPASS_TOKEN):
        return "internal_service"

    # 2. Extract Bearer token if provided
    bearer_token = None
    if authorization and authorization.lower().startswith("bearer "):
        bearer_token = authorization[7:].strip()

    candidate = key or bearer_token

    # 3. Allow anonymous dev if no admin key is configured
    if not ADMIN_API_KEY:
        return candidate or "anonymous_dev"

    if candidate and safe_compare(candidate, ADMIN_API_KEY):
        return candidate

    raise HTTPException(status_code=403, detail="Invalid or missing Admin API key or internal authorization token.")

def verify_user_key(
    key: Optional[str] = Security(api_key_header),
    internal_token: Optional[str] = Security(internal_token_header),
    authorization: Optional[str] = Security(auth_header),
):
    if internal_token and safe_compare(internal_token, INTERNAL_BYPASS_TOKEN):
        return "internal_service"

    bearer_token = None
    if authorization and authorization.lower().startswith("bearer "):
        bearer_token = authorization[7:].strip()

    candidate = key or bearer_token

    if not USER_API_KEY and not ADMIN_API_KEY:
        return candidate or "anonymous_dev"

    if candidate and (safe_compare(candidate, USER_API_KEY) or safe_compare(candidate, ADMIN_API_KEY)):
        return candidate

    raise HTTPException(status_code=403, detail="Invalid or missing API key or authorization token.")
