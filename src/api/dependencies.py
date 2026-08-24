import os
import secrets

from dotenv import load_dotenv
from fastapi import HTTPException, Security
from fastapi.security import APIKeyHeader

load_dotenv()

ADMIN_API_KEY = os.getenv("ADMIN_API_KEY", "").strip()
USER_API_KEY = os.getenv("USER_API_KEY", "").strip()

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

def verify_admin_key(key: str = Security(api_key_header)):
    if not ADMIN_API_KEY:
        raise HTTPException(status_code=500, detail="ADMIN_API_KEY not configured on server")
    if not key or not secrets.compare_digest(key, ADMIN_API_KEY):
        raise HTTPException(status_code=403, detail="Invalid or missing Admin API key")
    return key

def verify_user_key(key: str = Security(api_key_header)):
    valid_keys = [k for k in (USER_API_KEY, ADMIN_API_KEY) if k]
    if not valid_keys:
        return key
    if not key or not any(secrets.compare_digest(key, k) for k in valid_keys):
        raise HTTPException(status_code=403, detail="Invalid or missing API key")
    return key

