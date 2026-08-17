from fastapi import APIRouter, HTTPException, Depends, Query
from pydantic import BaseModel
from typing import Optional, List
from src.api.dependencies import verify_user_key
from src.core.rag_engine import RAGChatbot, KnowledgeRetriever
from src.config import REGISTRY_DB_PATH

router = APIRouter()
chatbot = RAGChatbot()

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    query: str
    history: Optional[List[ChatMessage]] = []
    top_k: int = 15

class SearchRequest(BaseModel):
    query: str
    top_k: int = 15

def extract_query_and_k(req: Optional[BaseModel], query_param: Optional[str], top_k_param: int):
    q = getattr(req, "query", None) if req else None
    if not q:
        q = query_param
    k = getattr(req, "top_k", top_k_param) if req else top_k_param
    return q, k

@router.api_route("/search", methods=["GET", "POST"], dependencies=[Depends(verify_user_key)])
def search_kb(req: Optional[SearchRequest] = None, query: Optional[str] = Query(None), top_k: int = Query(15)):
    q, k = extract_query_and_k(req, query, top_k)
    if not q or not q.strip():
        raise HTTPException(status_code=400, detail="Query string cannot be empty.")
    try:
        retriever = KnowledgeRetriever()
        results = retriever.retrieve(query_text=q, top_k=k)
        return {"query": q, "results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.api_route("/chat", methods=["GET", "POST"], dependencies=[Depends(verify_user_key)])
def chat_with_kb(req: Optional[ChatRequest] = None, query: Optional[str] = Query(None), top_k: int = Query(15)):
    q, k = extract_query_and_k(req, query, top_k)
    if not q or not q.strip():
        raise HTTPException(status_code=400, detail="Query string cannot be empty.")

    hist_dicts = []
    if req and req.history:
        hist_dicts = [{"role": h.role, "content": h.content} for h in req.history]
    
    response = chatbot.answer_question(query=q, history=hist_dicts, top_k=k)
    return response

@router.api_route("/query", methods=["GET", "POST"], dependencies=[Depends(verify_user_key)])
def query_alias(req: Optional[ChatRequest] = None, query: Optional[str] = Query(None), top_k: int = Query(15)):
    return chat_with_kb(req=req, query=query, top_k=top_k)

@router.api_route("/chat/stream", methods=["GET", "POST"], dependencies=[Depends(verify_user_key)])
def chat_with_kb_stream(req: Optional[ChatRequest] = None, query: Optional[str] = Query(None), top_k: int = Query(15)):
    q, k = extract_query_and_k(req, query, top_k)
    if not q or not q.strip():
        raise HTTPException(status_code=400, detail="Query string cannot be empty.")

    from fastapi.responses import StreamingResponse
    hist_dicts = []
    if req and req.history:
        hist_dicts = [{"role": h.role, "content": h.content} for h in req.history]

    return StreamingResponse(
        chatbot.answer_question_stream(query=q, history=hist_dicts, top_k=k),
        media_type="text/event-stream"
    )
