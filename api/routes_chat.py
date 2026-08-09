from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import Optional, List
from api.dependencies import verify_user_key
from core.rag_engine import RAGChatbot
from core.config import REGISTRY_DB_PATH

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

@router.post("/search", dependencies=[Depends(verify_user_key)])
def search_kb(req: SearchRequest):
    if not req.query or not req.query.strip():
        raise HTTPException(status_code=400, detail="Query string cannot be empty.")
    try:
        from core.rag_engine import KnowledgeRetriever
        retriever = KnowledgeRetriever()
        results = retriever.retrieve(query_text=req.query, top_k=req.top_k)
        return {"query": req.query, "results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/chat", dependencies=[Depends(verify_user_key)])
def chat_with_kb(req: ChatRequest):
    if not req.query or not req.query.strip():
        raise HTTPException(status_code=400, detail="Query string cannot be empty.")

    hist_dicts = [{"role": h.role, "content": h.content} for h in req.history] if req.history else []
    response = chatbot.answer_question(query=req.query, history=hist_dicts, top_k=req.top_k)
    return response

@router.post("/chat/stream", dependencies=[Depends(verify_user_key)])
def chat_with_kb_stream(req: ChatRequest):
    if not req.query or not req.query.strip():
        raise HTTPException(status_code=400, detail="Query string cannot be empty.")

    from fastapi.responses import StreamingResponse
    hist_dicts = [{"role": h.role, "content": h.content} for h in req.history] if req.history else []
    return StreamingResponse(
        chatbot.answer_question_stream(query=req.query, history=hist_dicts, top_k=req.top_k),
        media_type="text/event-stream"
    )


