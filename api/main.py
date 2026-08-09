import sys
import os
from pathlib import Path

# Force UTF-8 encoding on Windows console stdout/stderr
if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from fastapi import FastAPI, HTTPException, Security
from fastapi.security import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from core.rag_engine import KnowledgeRetriever
from db.registry import DocumentRegistry
from api.routes_chat import router as chat_router
from api.routes_admin import router as admin_router
from api.routes_admin_ui import router as admin_ui_router
from api.dependencies import USER_API_KEY, ADMIN_API_KEY

app = FastAPI(title="Guild Knowledge Base RAG Chatbot UI", version="3.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "").split(",") if os.getenv("CORS_ORIGINS") else ["*"],
    allow_methods=["POST", "GET", "DELETE"],
    allow_headers=["X-API-Key", "Content-Type"],
)

app.include_router(chat_router, prefix="/api")
app.include_router(admin_router, prefix="/api/admin")
app.include_router(admin_ui_router)

@app.get("/api/health")
async def health():
    try:
        retriever = KnowledgeRetriever()
        chunk_count = retriever.collection.count()
        return {"status": "ok", "chunks": chunk_count}
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))

@app.get("/api/stats")
def get_stats():
    registry = DocumentRegistry()
    retriever = KnowledgeRetriever()
    total_docs = len(registry.list_documents())
    total_chunks = retriever.collection.count()
    return {"total_documents": total_docs, "total_chunks": total_chunks}

@app.get("/", response_class=HTMLResponse)
def serve_ui():
    default_key = USER_API_KEY or ADMIN_API_KEY or ""
    return HTML_CONTENT.replace("{{DEFAULT_USER_KEY}}", default_key)

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
            flex-direction: column;
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
            content: none;
        }

        .header-brand p {
            font-size: 0.9rem;
            color: var(--text-muted);
            font-weight: 500;
        }

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
            justify-content: center;
        }

        .chip {
            background: rgba(15, 23, 42, 0.8);
            border: 1px solid var(--card-border);
            color: #cbd5e1;
            padding: 0.65rem 1.1rem;
            border-radius: 14px;
            font-size: 0.88rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.25s ease;
        }

        .chip:hover {
            background: rgba(99, 102, 241, 0.25);
            border-color: var(--primary-accent);
            color: #fff;
            transform: translateY(-2px);
            box-shadow: 0 4px 15px rgba(99, 102, 241, 0.3);
        }

        /* Chat Message Rows */
        .message-row {
            display: flex;
            flex-direction: column;
            max-width: 85%;
            animation: messageSlide 0.3s ease-out;
        }

        @keyframes messageSlide {
            from { opacity: 0; transform: translateY(8px); }
            to { opacity: 1; transform: translateY(0); }
        }

        .message-row.user {
            align-self: flex-start;
        }

        .message-row.assistant {
            align-self: flex-end;
            width: 100%;
            max-width: 95%;
        }

        .user .bubble {
            background: var(--user-msg-bg);
            color: #fff;
            padding: 0.9rem 1.3rem;
            border-radius: 18px 18px 4px 18px;
            box-shadow: 0 4px 15px rgba(79, 70, 229, 0.3);
            font-size: 1.05rem;
            line-height: 1.6;
        }

        .assistant-avatar {
            font-size: 0.82rem;
            color: var(--teal-accent);
            font-weight: 700;
            margin-bottom: 0.4rem;
            display: flex;
            align-items: center;
            gap: 0.4rem;
        }

        .assistant .bubble {
            background: var(--bot-msg-bg);
            border: 1px solid var(--card-border);
            padding: 1.25rem 1.4rem;
            border-radius: 18px 18px 18px 4px;
            backdrop-filter: blur(12px);
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
            width: 100%;
        }

        .response-text {
            white-space: pre-wrap;
            word-break: break-word;
            line-height: 1.85;
            font-size: 1.02rem;
            color: #f1f5f9;
        }

        /* Sources Section */
        .sources-container {
            margin-top: 1rem;
            border-top: 1px solid rgba(255, 255, 255, 0.08);
            padding-top: 0.75rem;
        }

        .sources-toggle-btn {
            background: var(--sources-bg);
            border: 1px solid rgba(255, 255, 255, 0.1);
            color: #cbd5e1;
            padding: 0.6rem 1rem;
            border-radius: 10px;
            font-size: 0.88rem;
            font-weight: 600;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: space-between;
            width: 100%;
            transition: all 0.2s ease;
        }

        .sources-toggle-btn:hover {
            border-color: var(--primary-accent);
            color: #fff;
        }

        .sources-toggle-btn.active .arrow {
            transform: rotate(180deg);
        }

        .sources-list {
            display: none;
            flex-direction: column;
            gap: 0.75rem;
            margin-top: 0.75rem;
        }

        .sources-list.show {
            display: flex;
        }

        .source-card {
            background: rgba(15, 23, 42, 0.9);
            border: 1px solid rgba(255, 255, 255, 0.06);
            border-radius: 10px;
            padding: 0.85rem 1rem;
        }

        .source-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            font-size: 0.85rem;
            font-weight: 700;
            color: #818cf8;
            margin-bottom: 0.4rem;
        }

        .source-score {
            background: rgba(20, 184, 166, 0.15);
            color: var(--teal-accent);
            padding: 0.2rem 0.5rem;
            border-radius: 6px;
            font-size: 0.75rem;
        }

        .source-snippet {
            font-size: 0.82rem;
            color: var(--text-muted);
            line-height: 1.5;
        }

        /* Typing Dot Indicator */
        .typing-indicator {
            display: flex;
            align-items: center;
            gap: 0.4rem;
            padding: 0.4rem 0;
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
            40% { transform: scale(1); }
        }

        /* Input Bar Container */
        .input-bar-container {
            padding: 1rem 1.5rem 1.5rem;
            background: transparent;
        }

        .input-form {
            display: flex;
            gap: 0.75rem;
            background: rgba(30, 41, 59, 0.85);
            border: 1px solid var(--card-border);
            padding: 0.6rem 0.85rem;
            border-radius: 16px;
            backdrop-filter: blur(16px);
            box-shadow: 0 8px 30px rgba(0, 0, 0, 0.4);
            transition: border-color 0.25s ease, box-shadow 0.25s ease;
        }

        .input-form:focus-within {
            border-color: var(--primary-accent);
            box-shadow: 0 0 25px rgba(99, 102, 241, 0.35);
        }

        .input-form input {
            flex: 1;
            background: transparent;
            border: none;
            outline: none;
            color: #fff;
            font-size: 1rem;
            padding: 0.5rem;
        }

        .send-btn {
            background: var(--user-msg-bg);
            color: #fff;
            border: none;
            border-radius: 12px;
            padding: 0.65rem 1.5rem;
            font-weight: 700;
            font-size: 0.9rem;
            cursor: pointer;
            transition: all 0.2s ease;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }

        .send-btn:hover {
            opacity: 0.95;
            transform: scale(1.02);
            box-shadow: 0 0 15px rgba(99, 102, 241, 0.5);
        }

        .send-btn:disabled {
            opacity: 0.4;
            cursor: not-allowed;
            transform: none;
            box-shadow: none;
        }
    </style>
</head>
<body>

    <header>
        <div class="header-brand">
            <h1>نقابة المهندسين الأردنيين</h1>
            <p>المساعد الذكي للمعرفة والأنظمة الهندسية (RAG Engine)</p>
        </div>
    </header>

    <main>
        <div id="chat-feed">
            <div class="welcome-card" id="welcome-screen">
                <h2>أهلاً بك في المساعد الذكي لنقابة المهندسين</h2>
                <p>تم إعداد هذا المساعد الذكي للإجابة الموثوقة على جميع استفساراتك المتعلقة بأنظمة النقابة، التكافل الاجتماعي، صندوق التقاعد، وشروط التسجيل اعتماداً على قاعدة المعرفة المعتمدة.</p>
                <div class="suggestion-chips">
                    <div class="chip" data-prompt="ما هي شروط تسجيل المهندسين الأردنيين في النقابة؟">ما هي شروط تسجيل المهندسين الأردنيين؟</div>
                    <div class="chip" data-prompt="شو هي خدمات صندوق التقاعد للمهندسين ورسوم الاشتراك؟">شو هي خدمات صندوق التقاعد للمهندسين؟</div>
                    <div class="chip" data-prompt="ما هو سلم رواتب المهندسين وكيف يحسب الحد الأدنى؟">ما هو سلم رواتب المهندسين الحد الأدنى؟</div>
                </div>
            </div>
        </div>

        <div class="input-bar-container">
            <form class="input-form" id="chat-form">
                <input type="text" id="user-input" placeholder="اكتب سؤالك هنا باللغة العربية، اللهجة الأردنية، أو الإنجليزية..." autocomplete="off" />
                <button type="submit" class="send-btn" id="send-button">
                    <span>إرسال</span>
                </button>
            </form>
        </div>
    </main>

    <script>
        let chatHistory = [];
        let messageCounter = 0;
        let isGenerating = false;

        document.addEventListener('DOMContentLoaded', function() {
            const form = document.getElementById('chat-form');
            if (form) {
                form.addEventListener('submit', function(e) {
                    e.preventDefault();
                    submitPrompt();
                });
            }

            document.querySelectorAll('.chip').forEach(function(chip) {
                chip.addEventListener('click', function() {
                    const promptText = this.getAttribute('data-prompt');
                    if (promptText) {
                        const inputEl = document.getElementById('user-input');
                        if (inputEl) inputEl.value = promptText;
                        submitPrompt();
                    }
                });
            });
        });

        function getApiKey() {
            const defaultKey = "{{DEFAULT_USER_KEY}}" || "456def";
            let key = localStorage.getItem('rag_api_key');
            if (!key || key.trim() === "") {
                key = defaultKey;
                localStorage.setItem('rag_api_key', key);
            }
            return key.trim();
        }

        function handleChipClick(text) {
            const inputEl = document.getElementById('user-input');
            if (inputEl) {
                inputEl.value = text;
                submitPrompt();
            }
        }

        function renderSourcesHtml(sourcesData, msgId) {
            if (!sourcesData || sourcesData.length === 0) return '';
            let cards = '';
            for (let i = 0; i < sourcesData.length; i++) {
                let s = sourcesData[i];
                let rank = s.rank || (i + 1);
                let title = escapeHtml(s.title || 'وثيقة');
                let score = s.similarity_score || 0;
                let snippet = escapeHtml(s.text ? s.text.substring(0, 300) + '...' : '');
                cards += '<div class="source-card">' +
                    '<div class="source-header">' +
                    '<span>[المصدر ' + rank + '] ' + title + '</span>' +
                    '<span class="source-score">' + score + '% تطابق</span>' +
                    '</div>' +
                    '<div class="source-snippet">' + snippet + '</div>' +
                    '</div>';
            }
            return '<div class="sources-container">' +
                '<button class="sources-toggle-btn" onclick="toggleSources(&quot;' + msgId + '&quot;)" id="btn-' + msgId + '">' +
                '<span>📚 المصادر المعتمدة (' + sourcesData.length + ' مصادر)</span>' +
                '<span class="arrow">▼</span>' +
                '</button>' +
                '<div class="sources-list" id="list-' + msgId + '">' +
                cards +
                '</div>' +
                '</div>';
        }

        async function submitPrompt() {
            if (isGenerating) return;

            const inputEl = document.getElementById('user-input');
            const query = inputEl ? inputEl.value.trim() : '';
            if (!query) return;

            isGenerating = true;
            const sendBtn = document.getElementById('send-button');
            if (sendBtn) sendBtn.disabled = true;

            const welcomeScreen = document.getElementById('welcome-screen');
            if (welcomeScreen) welcomeScreen.style.display = 'none';

            appendUserMessage(query);
            if (inputEl) inputEl.value = '';

            const typingId = appendTypingIndicator();

            try {
                const response = await fetch('/api/chat/stream', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                        'X-API-Key': getApiKey()
                    },
                    body: JSON.stringify({
                        query: query,
                        history: chatHistory,
                        top_k: 10
                    })
                });

                removeTypingIndicator(typingId);

                if (!response.ok) {
                    appendAssistantMessage({
                        answer: 'عذراً، حدث خطأ في الخادم (رمز الخطأ: ' + response.status + '). يرجى المحاولة لاحقاً.',
                        sources: []
                    });
                    return;
                }

                messageCounter++;
                const msgId = 'msg-' + messageCounter;
                const feed = document.getElementById('chat-feed');
                const row = document.createElement('div');
                row.className = 'message-row assistant';
                row.innerHTML = `
                    <div class="assistant-avatar">المساعد الذكي (RAG Engine)</div>
                    <div class="bubble">
                        <div class="response-text" id="text-${msgId}"></div>
                        <div id="sources-${msgId}"></div>
                    </div>
                `;
                feed.appendChild(row);
                scrollToBottom();

                const textEl = document.getElementById('text-' + msgId);
                const sourcesEl = document.getElementById('sources-' + msgId);

                let fullText = '';
                const reader = response.body.getReader();
                const decoder = new TextDecoder('utf-8');
                let buffer = '';

                while (true) {
                    const { done, value } = await reader.read();
                    if (done) break;

                    buffer += decoder.decode(value, { stream: true });
                    const lines = buffer.split(String.fromCharCode(10));
                    buffer = lines.pop() || '';

                    for (const line of lines) {
                        const trimmed = line.trim();
                        if (trimmed.indexOf('data: ') === 0) {
                            const jsonStr = trimmed.substring(6).trim();
                            if (!jsonStr) continue;
                            try {
                                const payload = JSON.parse(jsonStr);
                                if (payload.type === 'meta') {
                                    const sourcesData = payload.sources || [];
                                    if (sourcesData.length > 0 && sourcesEl) {
                                        sourcesEl.innerHTML = renderSourcesHtml(sourcesData, msgId);
                                    }
                                } else if (payload.type === 'token') {
                                    fullText += (payload.token || '');
                                    if (textEl) textEl.textContent = fullText;
                                    scrollToBottom();
                                }
                            } catch (err) {
                                console.error('SSE JSON Error:', err);
                            }
                        }
                    }
                }

                chatHistory.push({ role: 'user', content: query });
                chatHistory.push({ role: 'assistant', content: fullText });

            } catch (err) {
                console.error('Fetch error:', err);
                removeTypingIndicator(typingId);
                appendAssistantMessage({
                    answer: 'عذراً، حدث خطأ أثناء الاتصال بالخادم الرئيسي. الرجاء المحاولة مجدداً.',
                    sources: []
                });
            } finally {
                isGenerating = false;
                if (sendBtn) sendBtn.disabled = false;
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

        function appendAssistantMessage(data) {
            const feed = document.getElementById('chat-feed');
            messageCounter++;
            const msgId = 'msg-' + messageCounter;
            const row = document.createElement('div');
            row.className = 'message-row assistant';
            const sourcesHtml = renderSourcesHtml(data.sources, msgId);

            row.innerHTML = `
                <div class="assistant-avatar">المساعد الذكي (RAG Engine)</div>
                <div class="bubble">
                    <div class="response-text">${escapeHtml(data.answer)}</div>
                    ${sourcesHtml}
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
                <div class="assistant-avatar">المساعد الذكي يحلل المصادر...</div>
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
            if (feed) feed.scrollTop = feed.scrollHeight;
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

    def is_port_in_use(port: int, host: str = "127.0.0.1") -> bool:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex((host, port)) == 0

    target_port = 8080
    for p in [8080, 8000, 8081, 8501, 5000]:
        if not is_port_in_use(p):
            target_port = p
            break

    print(f"\n[INFO] Starting Guild Knowledge Base Chatbot Web Server on http://127.0.0.1:{target_port} ...")
    uvicorn.run(app, host="127.0.0.1", port=target_port)
