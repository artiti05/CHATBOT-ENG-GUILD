from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import sys
from pathlib import Path
#extra imports 
import os
from fastapi import FastAPI, HTTPException, Security
from fastapi.security import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from rag_chatbot_engine import KnowledgeRetriever, RAGChatbot
from crawler_admin import DocumentRegistry

app = FastAPI(title="Guild Knowledge Base RAG Chatbot UI", version="3.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "").split(",") if os.getenv("CORS_ORIGINS") else [],
    allow_methods=["POST", "GET"],
    allow_headers=["X-API-Key", "Content-Type"],
)
API_KEY = os.getenv("RAG_API_KEY", "")
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

def verify_key(key: str = Security(api_key_header)):
    if not API_KEY:
        raise HTTPException(status_code=500, detail="RAG_API_KEY not configured on server")
    if key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    return key

retriever = KnowledgeRetriever()
chatbot = RAGChatbot()
registry = DocumentRegistry()

class SearchRequest(BaseModel):
    query: str
    top_k: int = 10

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    query: str
    history: Optional[List[ChatMessage]] = []
    top_k: int = 10

@app.post("/api/search")
async def search_documents(req: SearchRequest, _=Security(verify_key)):
    if not req.query or not req.query.strip():
        raise HTTPException(status_code=400, detail="Query string cannot be empty.")
    
    results = retriever.retrieve(query_text=req.query, top_k=req.top_k)
    return {"query": req.query, "count": len(results), "results": results}

@app.post("/api/chat")
async def chat_with_kb(req: ChatRequest, _=Security(verify_key)):
    if not req.query or not req.query.strip():
        raise HTTPException(status_code=400, detail="Query string cannot be empty.")

    hist_dicts = [{"role": h.role, "content": h.content} for h in req.history] if req.history else []
    response = chatbot.answer_question(query=req.query, history=hist_dicts, top_k=req.top_k)
    return response



@app.get("/api/health")
async def health():
    try:
        chunk_count = retriever.collection.count()
        return {"status": "ok", "chunks": chunk_count}
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))    
    
    hist_dicts = [{"role": h.role, "content": h.content} for h in req.history] if req.history else []
    response = chatbot.answer_question(query=req.query, history=hist_dicts, top_k=req.top_k)
    return response

@app.get("/api/stats")
async def get_stats():
    total_docs = len(registry.list_documents())
    total_chunks = retriever.collection.count()
    return {"total_documents": total_docs, "total_chunks": total_chunks}

@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    return HTML_CONTENT

HTML_CONTENT = """<!DOCTYPE html>
<html lang="ar" dir="rtl">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>نقابة المهندسين الأردنيين — المساعد الذكي (RAG Chatbot)</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Tajawal:wght@400;500;700;800&family=Outfit:wght@400;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-gradient: linear-gradient(135deg, #0b0f19 0%, #111827 50%, #0b0f19 100%);
            --header-bg: rgba(17, 24, 39, 0.95);
            --card-bg: rgba(30, 41, 59, 0.7);
            --card-border: rgba(255, 255, 255, 0.08);
            --user-msg-bg: linear-gradient(135deg, #4f46e5 0%, #6366f1 100%);
            --bot-msg-bg: rgba(30, 41, 59, 0.85);
            --accent-glow: #6366f1;
            --teal-accent: #14b8a6;
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
            --accordion-bg: rgba(15, 23, 42, 0.6);
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: 'Tajawal', 'Outfit', sans-serif;
        }

        body {
            background: var(--bg-gradient);
            color: var(--text-main);
            height: 100vh;
            display: flex;
            flex-direction: column;
            overflow: hidden;
        }

        /* CENTERED HEADER */
        header {
            background: var(--header-bg);
            backdrop-filter: blur(16px);
            border-bottom: 1px solid var(--card-border);
            padding: 1.2rem 2rem;
            display: flex;
            justify-content: center;
            align-items: center;
            text-align: center;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
            z-index: 100;
        }

        .header-brand {
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 0.3rem;
        }

        .header-brand h1 {
            font-size: 1.65rem;
            font-weight: 800;
            background: linear-gradient(135deg, #ffffff 0%, #cbd5e1 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            letter-spacing: -0.5px;
            display: flex;
            align-items: center;
            gap: 0.6rem;
        }

        .header-brand h1::before {
            content: "🇯🇴";
            font-size: 1.5rem;
        }

        .header-brand p {
            font-size: 0.9rem;
            color: var(--text-muted);
            font-weight: 500;
        }

        /* MAIN CHAT WRAPPER */
        main {
            flex: 1;
            display: flex;
            flex-direction: column;
            max-width: 1000px;
            width: 100%;
            margin: 0 auto;
            padding: 1rem 1.5rem;
            position: relative;
            height: calc(100vh - 85px);
        }

        /* CHAT MESSAGES SCROLL CONTAINER */
        #chat-feed {
            flex: 1;
            overflow-y: auto;
            padding: 1rem 0.5rem;
            display: flex;
            flex-direction: column;
            gap: 1.5rem;
            scroll-behavior: smooth;
        }

        #chat-feed::-webkit-scrollbar {
            width: 6px;
        }
        #chat-feed::-webkit-scrollbar-thumb {
            background: rgba(255, 255, 255, 0.15);
            border-radius: 4px;
        }

        /* WELCOME SCREEN */
        .welcome-card {
            background: var(--card-bg);
            border: 1px solid var(--card-border);
            border-radius: 20px;
            padding: 2.5rem 2rem;
            text-align: center;
            margin: auto 0;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
            backdrop-filter: blur(12px);
        }

        .welcome-card h2 {
            font-size: 1.6rem;
            font-weight: 700;
            color: #fff;
            margin-bottom: 0.75rem;
        }

        .welcome-card p {
            color: var(--text-muted);
            font-size: 1rem;
            line-height: 1.6;
            max-width: 650px;
            margin: 0 auto 1.5rem auto;
        }

        .suggestion-chips {
            display: flex;
            flex-wrap: wrap;
            justify-content: center;
            gap: 0.75rem;
        }

        .chip {
            background: rgba(99, 102, 241, 0.15);
            border: 1px solid rgba(99, 102, 241, 0.3);
            color: #e2e8f0;
            padding: 0.6rem 1.25rem;
            border-radius: 50px;
            font-size: 0.9rem;
            cursor: pointer;
            transition: all 0.25s ease;
        }

        .chip:hover {
            background: rgba(99, 102, 241, 0.35);
            border-color: var(--primary-accent);
            transform: translateY(-2px);
            color: #fff;
        }

        /* MESSAGE BUBBLES */
        .message-row {
            display: flex;
            flex-direction: column;
            width: 100%;
            animation: fadeIn 0.3s ease-in-out;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }

        .message-row.user {
            align-items: flex-end;
        }

        .message-row.assistant {
            align-items: flex-start;
        }

        .bubble {
            max-width: 85%;
            padding: 1.15rem 1.5rem;
            border-radius: 20px;
            font-size: 1rem;
            line-height: 1.7;
            position: relative;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
        }

        .message-row.user .bubble {
            background: var(--user-msg-bg);
            color: #ffffff;
            border-bottom-left-radius: 4px;
            font-weight: 500;
        }

        .message-row.assistant .bubble {
            background: var(--bot-msg-bg);
            color: var(--text-main);
            border: 1px solid var(--card-border);
            border-bottom-right-radius: 4px;
            backdrop-filter: blur(10px);
        }

        .assistant-avatar {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            font-size: 0.85rem;
            color: var(--text-muted);
            margin-bottom: 0.4rem;
            font-weight: 600;
        }

        .response-text {
            white-space: pre-wrap;
            word-break: break-word;
        }

        /* COLLAPSIBLE SOURCES ACCORDION (MODERN CHATBOT STYLE) */
        .sources-accordion-container {
            margin-top: 1rem;
            width: 100%;
            border-top: 1px solid rgba(255, 255, 255, 0.08);
            padding-top: 0.75rem;
        }

        .sources-toggle-btn {
            background: rgba(15, 23, 42, 0.7);
            border: 1px solid rgba(255, 255, 255, 0.12);
            color: #cbd5e1;
            padding: 0.65rem 1.1rem;
            border-radius: 12px;
            font-size: 0.9rem;
            font-weight: 600;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: space-between;
            width: 100%;
            transition: all 0.2s ease;
        }

        .sources-toggle-btn:hover {
            background: rgba(99, 102, 241, 0.2);
            border-color: var(--primary-accent);
            color: #fff;
        }

        .sources-toggle-btn .arrow {
            transition: transform 0.3s ease;
            font-size: 0.75rem;
        }

        .sources-toggle-btn.active .arrow {
            transform: rotate(180deg);
        }

        .sources-list {
            display: none;
            flex-direction: column;
            gap: 0.75rem;
            margin-top: 0.75rem;
            max-height: 450px;
            overflow-y: auto;
            padding-left: 0.3rem;
        }

        .sources-list.show {
            display: flex;
        }

        .source-card {
            background: var(--accordion-bg);
            border: 1px solid rgba(255, 255, 255, 0.06);
            border-radius: 10px;
            padding: 0.85rem 1rem;
            font-size: 0.85rem;
            line-height: 1.5;
            transition: border-color 0.2s ease;
        }

        .source-card:hover {
            border-color: rgba(99, 102, 241, 0.4);
        }

        .source-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 0.35rem;
            color: #818cf8;
            font-weight: 700;
        }

        .source-score {
            background: rgba(20, 184, 166, 0.2);
            color: #2dd4bf;
            padding: 0.15rem 0.5rem;
            border-radius: 50px;
            font-size: 0.75rem;
        }

        .source-snippet {
            color: #94a3b8;
            font-size: 0.82rem;
            white-space: pre-wrap;
        }

        /* CHAT INPUT CONTAINER FIXED AT BOTTOM */
        .input-bar-container {
            padding: 1rem 0;
            background: transparent;
            width: 100%;
        }

        .input-form {
            display: flex;
            gap: 0.75rem;
            background: var(--card-bg);
            border: 1px solid var(--card-border);
            padding: 0.6rem 0.8rem;
            border-radius: 16px;
            backdrop-filter: blur(16px);
            box-shadow: 0 8px 30px rgba(0, 0, 0, 0.4);
            transition: border-color 0.25s ease;
        }

        .input-form:focus-within {
            border-color: var(--primary-accent);
            box-shadow: 0 0 20px rgba(99, 102, 241, 0.3);
        }

        .input-form input {
            flex: 1;
            background: transparent;
            border: none;
            outline: none;
            color: #fff;
            font-size: 1rem;
            padding: 0.5rem 0.75rem;
        }

        .input-form input::placeholder {
            color: #64748b;
        }

        .send-btn {
            background: var(--user-msg-bg);
            color: #fff;
            border: none;
            border-radius: 12px;
            padding: 0.75rem 1.6rem;
            font-weight: 700;
            font-size: 0.95rem;
            cursor: pointer;
            transition: all 0.2s ease;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }

        .send-btn:hover {
            opacity: 0.9;
            transform: scale(1.02);
        }

        .send-btn:disabled {
            opacity: 0.5;
            cursor: not-allowed;
        }

        /* TYPING INDICATOR */
        .typing-indicator {
            display: flex;
            align-items: center;
            gap: 0.4rem;
            padding: 0.5rem 0;
        }

        .typing-dot {
            width: 8px;
            height: 8px;
            background: #818cf8;
            border-radius: 50%;
            animation: bounce 1.4s infinite ease-in-out both;
        }

        .typing-dot:nth-child(1) { animation-delay: -0.32s; }
        .typing-dot:nth-child(2) { animation-delay: -0.16s; }

        @keyframes bounce {
            0%, 80%, 100% { transform: scale(0); }
            40% { transform: scale(1.0); }
        }
    </style>
</head>
<body>

    <!-- CENTERED HEADER (NO STATS PILLS, NO UNWANTED TABS) -->
    <header>
        <div class="header-brand">
            <h1>نقابة المهندسين الأردنيين</h1>
            <p>المساعد الذكي للمعرفة والأنظمة الهندسية (RAG Chatbot Engine)</p>
        </div>
    </header>

    <main>
        <!-- SCROLLABLE CHAT FEED -->
        <div id="chat-feed">
            <div class="welcome-card" id="welcome-screen">
                <h2>أهلاً بك في المساعد الذكي لنقابة المهندسين 🤖</h2>
                <p>تم إعداد وتكفيف هذا المساعد المتقدم للإجابة على جميع استفساراتك المتعلقة بأنظمة وقوانين النقابة، التكافل الاجتماعي، صندوق التقاعد، وشروط التسجيل اعتماداً على قاعدة المعرفة الموثوقة.</p>
                <div class="suggestion-chips">
                    <div class="chip" onclick="sendSuggestion('ما هي شروط تسجيل المهندسين الأردنيين في النقابة؟')">ما هي شروط تسجيل المهندسين الأردنيين؟</div>
                    <div class="chip" onclick="sendSuggestion('شو هي خدمات صندوق التقاعد للمهندسين ورسوم الاشتراك؟')">شو هي خدمات صندوق التقاعد للمهندسين؟</div>
                    <div class="chip" onclick="sendSuggestion('ما هو سلم رواتب المهندسين وكيف يحسب الحد الأدنى؟')">ما هو سلم رواتب المهندسين الحد الأدنى؟</div>
                </div>
            </div>
        </div>

        <!-- FIXED BOTTOM CHAT INPUT BAR -->
        <div class="input-bar-container">
            <form class="input-form" id="chat-form" onsubmit="handleSend(event)">
                <input type="text" id="user-input" placeholder="اكتب سؤالك هنا باللغة العربية، اللهجة الأردنية، أو الإنجليزية..." autocomplete="off" />
                <button type="submit" class="send-btn" id="send-button">
                    <span>إرسال</span>
                    <span>✨</span>
                </button>
            </form>
        </div>
    </main>

    <script>
        let chatHistory = [];
        let messageCounter = 0;

        function sendSuggestion(text) {
            document.getElementById('user-input').value = text;
            handleSend(new Event('submit'));
        }

        async function handleSend(e) {
            e.preventDefault();
            const inputEl = document.getElementById('user-input');
            const query = inputEl.value.trim();
            if (!query) return;

            // Hide welcome screen on first message
            const welcomeScreen = document.getElementById('welcome-screen');
            if (welcomeScreen) welcomeScreen.style.display = 'none';

            // Append User Message to UI
            appendUserMessage(query);
            inputEl.value = '';

            // Disable send button while processing
            const sendBtn = document.getElementById('send-button');
            sendBtn.disabled = true;

            // Append Assistant Typing Indicator
            const typingId = appendTypingIndicator();

            try {
                const response = await fetch('/api/chat', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        query: query,
                        history: chatHistory,
                        top_k: 10
                    })
                });

                const data = await response.json();
                removeTypingIndicator(typingId);

                // Update chat history memory
                chatHistory.push({ role: 'user', content: query });
                chatHistory.push({ role: 'assistant', content: data.answer });

                // Append Assistant Response Message with Collapsible Sources
                appendAssistantMessage(data);

            } catch (err) {
                removeTypingIndicator(typingId);
                appendAssistantMessage({
                    answer: 'عذراً، حدث خطأ أثناء الاتصال بالخادم الرئيسي. الرجاء التأكد من تشغيل الخادم والمحاولة مجدداً.',
                    sources: []
                });
            } finally {
                sendBtn.disabled = false;
                scrollToBottom();
            }
        }

        function appendUserMessage(text) {
            const feed = document.getElementById('chat-feed');
            const row = document.createElement('div');
            row.className = 'message-row user';
            row.innerHTML = `
                <div class="bubble">
                    <div class="response-text">${escapeHtml(text)}</div>
                </div>
            `;
            feed.appendChild(row);
            scrollToBottom();
        }

        function appendTypingIndicator() {
            const feed = document.getElementById('chat-feed');
            const id = 'typing-' + Date.now();
            const row = document.createElement('div');
            row.className = 'message-row assistant';
            row.id = id;
            row.innerHTML = `
                <div class="assistant-avatar">🤖 المساعد الذكي يحلل المصادر...</div>
                <div class="bubble">
                    <div class="typing-indicator">
                        <div class="typing-dot"></div>
                        <div class="typing-dot"></div>
                        <div class="typing-dot"></div>
                    </div>
                </div>
            `;
            feed.appendChild(row);
            scrollToBottom();
            return id;
        }

        function removeTypingIndicator(id) {
            const el = document.getElementById(id);
            if (el) el.remove();
        }

        function appendAssistantMessage(data) {
            const feed = document.getElementById('chat-feed');
            messageCounter++;
            const msgId = 'msg-' + messageCounter;
            const row = document.createElement('div');
            row.className = 'message-row assistant';

            let sourcesHtml = '';
            if (data.sources && data.sources.length > 0) {
                const sourceCards = data.sources.map(src => `
                    <div class="source-card">
                        <div class="source-header">
                            <span>[المصدر ${src.rank}] ${escapeHtml(src.title)} ${src.section_title ? '— ' + escapeHtml(src.section_title) : ''}</span>
                            <span class="source-score">${src.similarity_score}% تطابق</span>
                        </div>
                        <div class="source-snippet">${escapeHtml(src.text ? src.text.substring(0, 300) + '...' : '')}</div>
                    </div>
                `).join('');

                sourcesHtml = `
                    <div class="sources-accordion-container">
                        <button class="sources-toggle-btn" onclick="toggleSources('${msgId}')" id="btn-${msgId}">
                            <span>📚 المصادر العشرة المعتمدة (${data.sources.length} مصادر)</span>
                            <span class="arrow">▼</span>
                        </button>
                        <div class="sources-list" id="list-${msgId}">
                            ${sourceCards}
                        </div>
                    </div>
                `;
            }

            row.innerHTML = `
                <div class="assistant-avatar">🤖 المساعد الذكي (RAG Engine)</div>
                <div class="bubble">
                    <div class="response-text">${escapeHtml(data.answer)}</div>
                    ${sourcesHtml}
                </div>
            `;

            feed.appendChild(row);
            scrollToBottom();
        }

        function toggleSources(msgId) {
            const listEl = document.getElementById('list-' + msgId);
            const btnEl = document.getElementById('btn-' + msgId);
            if (listEl && btnEl) {
                listEl.classList.toggle('show');
                btnEl.classList.toggle('active');
            }
        }

        function scrollToBottom() {
            const feed = document.getElementById('chat-feed');
            feed.scrollTop = feed.scrollHeight;
        }

        function escapeHtml(text) {
            if (!text) return '';
            return text
                .replace(/&/g, "&amp;")
                .replace(/</g, "&lt;")
                .replace(/>/g, "&gt;")
                .replace(/"/g, "&quot;")
                .replace(/'/g, "&#039;");
        }
    </script>
</body>
</html>
"""

if __name__ == "__main__":
    import uvicorn
    import socket
    import os

    def is_port_in_use(port: int, host: str = "127.0.0.1") -> bool:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex((host, port)) == 0

    target_port = int(os.getenv("PORT", "8080"))
    if is_port_in_use(target_port):
        for alt_port in [8000, 8501, 5000, 8081]:
            if not is_port_in_use(alt_port):
                print(f"ℹ️ Port {target_port} is currently occupied by another application. Falling back to port {alt_port}.")
                target_port = alt_port
                break

    print(f"\n🚀 Starting Guild Knowledge Base RAG Chatbot Server on http://0.0.0.0:{target_port} ...")
    uvicorn.run("app:app", host="0.0.0.0", port=target_port, reload=True)
