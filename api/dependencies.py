import os
from fastapi import Security, HTTPException
from fastapi.security import APIKeyHeader
from dotenv import load_dotenv

load_dotenv()

# We define two API keys: one for users (web UI), one for admins (management panel)
ADMIN_API_KEY = os.getenv("ADMIN_API_KEY", "")
USER_API_KEY = os.getenv("USER_API_KEY", "")

# We use the same header for both, but validate against different keys depending on the endpoint
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

def verify_admin_key(key: str = Security(api_key_header)):
    if not ADMIN_API_KEY:
        raise HTTPException(status_code=500, detail="ADMIN_API_KEY not configured on server")
    if key != ADMIN_API_KEY:
        raise HTTPException(status_code=403, detail="Invalid or missing Admin API key")
    return key

def verify_user_key(key: str = Security(api_key_header)):
    if not USER_API_KEY:
        raise HTTPException(status_code=500, detail="USER_API_KEY not configured on server")
    if key != USER_API_KEY:
        raise HTTPException(status_code=403, detail="Invalid or missing User API key")
    return key
