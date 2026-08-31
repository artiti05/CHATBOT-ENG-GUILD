from fastapi import APIRouter, HTTPException, Depends, Query
from pydantic import BaseModel, Field
from typing import Optional, List
from src.api.security import verify_authenticated_client, sanitize_user_input
from src.core.rag_engine import RAGChatbot, KnowledgeRetriever
from src.config import REGISTRY_DB_PATH

router = APIRouter()
_chatbot = None
_retriever = None

def get_chatbot() -> RAGChatbot:
    global _chatbot
    if _chatbot is None:
        _chatbot = RAGChatbot()
    return _chatbot

def get_retriever() -> KnowledgeRetriever:
    global _retriever
    if _retriever is None:
        _retriever = KnowledgeRetriever()
    return _retriever

class ChatMessage(BaseModel):
    role: str = Field(..., max_length=50)
    content: str = Field(..., max_length=4000)

class UserContext(BaseModel):
    user_id: Optional[str] = Field(None, max_length=100)
    name: Optional[str] = Field(None, max_length=150)
    phone: Optional[str] = Field(None, max_length=50)
    engineer_number: Optional[str] = Field(None, max_length=50)

class ChatRequest(BaseModel):
    query: str = Field(..., max_length=4000)
    history: Optional[List[ChatMessage]] = []
    top_k: int = Field(default=15, ge=1, le=50)
    user: Optional[UserContext] = None
    session_id: Optional[str] = Field(None, max_length=128)

class SearchRequest(BaseModel):
    query: str = Field(..., max_length=4000)
    top_k: int = Field(default=15, ge=1, le=50)

def extract_query_and_k(req: Optional[BaseModel], query_param: Optional[str], top_k_param: int):
    q = getattr(req, "query", None) if req else None
    if not q:
        q = query_param
    k = getattr(req, "top_k", top_k_param) if req else top_k_param
    return q, k

@router.api_route("/search", methods=["GET", "POST"], dependencies=[Depends(verify_authenticated_client)])
async def search_kb(req: Optional[SearchRequest] = None, query: Optional[str] = Query(None), top_k: int = Query(15, ge=1, le=50)):
    q, k = extract_query_and_k(req, query, top_k)
    if not q or not q.strip():
        raise HTTPException(status_code=400, detail="Query string cannot be empty.")
    q = sanitize_user_input(q)
    try:
        retriever = get_retriever()
        results = await retriever.retrieve(query_text=q, top_k=k)
        return {"query": q, "results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.api_route("/chat", methods=["GET", "POST"], dependencies=[Depends(verify_authenticated_client)])
async def chat_with_kb(req: Optional[ChatRequest] = None, query: Optional[str] = Query(None), top_k: int = Query(15, ge=1, le=50)):
    q, k = extract_query_and_k(req, query, top_k)
    if not q or not q.strip():
        raise HTTPException(status_code=400, detail="Query string cannot be empty.")
    q = sanitize_user_input(q)

    hist_dicts = []
    if req and req.history:
        hist_dicts = [{"role": h.role, "content": sanitize_user_input(h.content)} for h in req.history]
    user_dict = req.user.model_dump(exclude_none=True) if req and req.user else None
    session_id = req.session_id if req else None

    bot = get_chatbot()
    response = await bot.answer_question(query=q, history=hist_dicts, top_k=k, user=user_dict, session_id=session_id)
    return response

@router.api_route("/query", methods=["GET", "POST"], dependencies=[Depends(verify_authenticated_client)])
async def query_alias(req: Optional[ChatRequest] = None, query: Optional[str] = Query(None), top_k: int = Query(15, ge=1, le=50)):
    return await chat_with_kb(req=req, query=query, top_k=top_k)

@router.api_route("/chat/stream", methods=["GET", "POST"], dependencies=[Depends(verify_authenticated_client)])
async def chat_with_kb_stream(req: Optional[ChatRequest] = None, query: Optional[str] = Query(None), top_k: int = Query(15, ge=1, le=50)):
    q, k = extract_query_and_k(req, query, top_k)
    if not q or not q.strip():
        raise HTTPException(status_code=400, detail="Query string cannot be empty.")
    q = sanitize_user_input(q)

    from fastapi.responses import StreamingResponse
    hist_dicts = []
    if req and req.history:
        hist_dicts = [{"role": h.role, "content": sanitize_user_input(h.content)} for h in req.history]
    user_dict = req.user.model_dump(exclude_none=True) if req and req.user else None
    session_id = req.session_id if req else None

    bot = get_chatbot()
    return StreamingResponse(
        bot.answer_question_stream(query=q, history=hist_dicts, top_k=k, user=user_dict, session_id=session_id),
        media_type="text/event-stream"
    )
