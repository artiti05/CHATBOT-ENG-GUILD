# 🇯🇴 Arabic PDF Parser & Hybrid RAG Chatbot System
### Jordan Engineers Association (نقابة المهندسين الأردنيين)

An end-to-end **Arabic PDF Vision Parser, Hybrid RAG (Retrieval-Augmented Generation) Engine, and AI Chatbot System** built specifically for the Jordan Engineers Association.

It processes complex Arabic documents, handles reversed Bidi character streams, transcribes multi-header RTL tables using **Qwen2.5-VL**, embeds document chunks using **GTE-Qwen / BAAI BGE-M3**, reranks search results via **BGE Cross-Encoder**, normalizes **Jordanian Dialect** queries into formal Arabic, and serves a secure FastAPI Web UI and REST API.

---

## 📌 Table of Contents
- [What the Project is About](#-what-the-project-is-about)
- [System Architecture & Storage](#-system-architecture--storage)
- [Prerequisites & Setup](#-prerequisites--setup)
- [How to Run](#-how-to-run)
- [Knowledge Base Handling & Ingestion](#-knowledge-base-handling--ingestion)
- [Security & API Integration Guide](#-security--api-integration-guide)
  - [Authentication & CORS](#1-authentication--cors)
  - [API Endpoints Reference](#2-api-endpoints-reference)
  - [Integration Code Examples](#3-integration-code-examples)
  - [Error Handling & Best Practices](#4-error-handling--best-practices)
- [Configuration Reference](#-configuration-reference)
- [License](#-license)

---

## 💡 What the Project is About

The Jordan Engineers Association manages an extensive collection of legal bylaws, strategic plans, service guides, university admission criteria, and operational news. This system provides an intelligent knowledge base assistant capable of:

1. **Vision-Based Document Parsing**: Uses OpenCV CLAHE image enhancement and **Qwen2.5-VL** (via Ollama or LM Studio) to transcribe Arabic prose and format complex nested RTL tables as HTML `<table>` elements with accurate `colspan` and `rowspan`.
2. **Jordanian Dialect Normalization**: Translates colloquial Jordanian phrases (e.g., *"شو الأوراق"*, *"بدي أسجل"*, *"قديش الإشتراك"*) into formal Modern Standard Arabic (MSA) search terms for accurate semantic retrieval.
3. **Hybrid Retrieval & Reranking**: Combines dense vector similarity search in **ChromaDB** with a cross-encoder reranker (**BAAI/bge-reranker-v2-m3**) to score top relevance matches.
4. **FastAPI Web UI & Rest API**: Provides an ambient dark-mode web application and secure API endpoints protected by header authentication.

---

## 🏗 System Architecture & Storage

```
CHATBOT-ENG-GUILD/
├── app.py                      # FastAPI Web Application & UI Server
├── main.py                     # Unified Multi-Mode CLI Launcher
├── rag_chatbot_engine.py       # RAG Retriever, Cross-Encoder Reranker & Jordanian Normalizer
├── ingestion_pipeline.py       # Qwen2.5-VL Vision Parser, Text Cleaner, Chunker & Indexer
├── crawler_admin.py            # SQLite Document Registry & Admin Manager
├── batch_ingest.py             # Batch Ingestion Utilities & Database Management
├── config/
│   └── settings.py             # Global Configuration & Model Settings
├── texts/                      # 430+ Categorized Markdown Knowledge Base files
├── pdfs/                       # Source PDF Documents directory
├── output_dir/                 # Pre-parsed Markdown files & output artifacts
└── storage/                    # Persistent Storage
    ├── registry.db             # SQLite document status registry
    └── chroma_db/              # Persistent ChromaDB vector database
```

### Storage Architecture
- **SQLite Registry ([`storage/registry.db`](file:///c:/Users/VICTUS/Desktop/CHATBOT-ENG-GUILD/storage/registry.db))**: Tracks document metadata, hashes, file types, page counts, parsing status (`active`/`excluded`), and execution timestamps.
- **ChromaDB Store ([`storage/chroma_db`](file:///c:/Users/VICTUS/Desktop/CHATBOT-ENG-GUILD/storage/chroma_db))**: Persists dense vector embeddings under collection `guild_knowledge_base`.

---

## 🚀 Prerequisites & Setup

### 1. Python Environment Setup
```powershell
# Clone repository
git clone https://github.com/shatnawiO/arabic-pdf-parser.git
cd CHATBOT-ENG-GUILD

# Create virtual environment
python -m venv venv

# Activate virtual environment (Windows PowerShell)
.\venv\Scripts\Activate.ps1
# Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Environment Configuration
Create a `.env` file from `.env.example`:
```powershell
Copy-Item .env.example .env
```
Ensure security credentials and model providers are configured:
```ini
RAG_API_KEY=your_secure_api_key_here
LLM_PROVIDER=ollama
VLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_CHAT_MODEL=qwen2.5:7b
OLLAMA_VISION_MODEL=qwen2.5vl:7b
CORS_ORIGINS=http://localhost:3000,http://127.0.0.1:8000
```

### 3. Local Ollama Service Setup
Install and start [Ollama](https://ollama.com), then pull the required models:
```powershell
ollama pull qwen2.5:7b
ollama pull qwen2.5vl:7b
ollama serve
```

---

## 💻 How to Run

All execution modes are controlled through the unified entry point [`main.py`](file:///c:/Users/VICTUS/Desktop/CHATBOT-ENG-GUILD/main.py).

### Launch the Web Application & Chatbot UI
```powershell
python main.py
```
*(Or specify a custom port: `python main.py serve --port 8000`)*

Open your browser at **`http://localhost:8000`** *(automatically detects alternative ports if 8000 is occupied)*.

---

## 📚 Knowledge Base Handling & Ingestion

The system ingests three primary types of knowledge sources:

| Source Directory | Format | Content Description | Ingestion Method |
| :--- | :--- | :--- | :--- |
| **`texts/`** | `.md` / `.txt` | 430+ text knowledge files (Insurance, Registration, Laws, Services) | Text Cleaning & Semantic Chunking |
| **`pdfs/`** | `.pdf` | Official PDF publications, complex multi-header tables, legal decrees | Qwen2.5-VL Vision Parsing + OCR |
| **`output_dir/`** | `.md` | Pre-parsed Markdown files extracted from vision pipelines | Markdown Indexer |

### CLI Ingestion Commands

* **Ingest Text Knowledge Base (`texts/`):**
  ```powershell
  python main.py ingest-texts --no-reset-db
  ```
  *(Omit `--no-reset-db` if you want to wipe the vector store and start fresh)*.

* **Ingest PDF Documents (`pdfs/`):**
  ```powershell
  python main.py ingest-pdfs --no-reset-db
  ```

* **Ingest Pre-Parsed Markdown (`output_dir/`):**
  ```powershell
  python main.py ingest-markdown
  ```

* **Full End-to-End Ingestion (All Sources):**
  ```powershell
  python main.py ingest-all
  ```

* **Preview Ingestion Plan (Dry Run):**
  ```powershell
  python main.py ingest-all --dry-run
  ```

> ⚠️ **Important Safety Rule:** Do not run ingestion scripts while a live Web API server is performing heavy write operations on `storage/chroma_db` to avoid lock contention.

---

## 🔑 Security & API Integration Guide

### 1. Authentication & CORS
All modifying/data endpoints (`POST /api/chat` and `POST /api/search`) require header authentication.

- **Header Name**: `X-API-Key`
- **Configuration**: Set via `RAG_API_KEY` in `.env`.
- **CORS Restrictions**: Configured in `.env` under `CORS_ORIGINS`.

```http
X-API-Key: your_configured_rag_api_key
Content-Type: application/json
```

---

### 2. API Endpoints Reference

#### A. `POST /api/chat` — Conversational RAG Endpoint
Executes query translation (Jordanian Dialect -> MSA), hybrid dense vector retrieval in ChromaDB, cross-encoder reranking, and local LLM context-augmented generation.

**Request Headers:**
```http
X-API-Key: <RAG_API_KEY>
Content-Type: application/json
```

**Request Body Example:**
```json
{
  "query": "شو الأوراق المطلوبة للتسجيل بالنقابة؟",
  "history": [
    {
      "role": "user",
      "content": "مرحبا"
    },
    {
      "role": "assistant",
      "content": "أهلاً بك! كيف يمكنني مساعدتك اليوم في خدمات نقابة المهندسين؟"
    }
  ],
  "top_k": 10
}
```

**Response Body Example:**
```json
{
  "query": "شو الأوراق المطلوبة للتسجيل بالنقابة؟",
  "normalized_query": "ما هي الوثائق والمستندات المطلوبة للتسجيل والانتساب لنقابة المهندسين؟",
  "detected_accent": "jordanian",
  "answer": "للتسجيل والانتساب بنقابة المهندسين الأردنيين، يجب تقديم الوثائق التالية:\n1. صورة مصدقة عن شهادة الدراسة الثانوية العامة (التوجيهي).\n2. صورة مصدقة عن مصدقة البكالوريوس والشهادة الجامعية.\n3. صورة عن بطاقة الأحوال المدنية.\n...",
  "sources": [
    {
      "rank": 1,
      "source_id": "text_8f2a1b9c",
      "title": "الانتساب للنقابة",
      "section_title": "الأوراق والوثائق المطلوبة",
      "text": "الوثائق المطلوبة للانتساب: 1- صورة مصدقة عن التوجيهي...",
      "similarity_score": 94,
      "page_number": null
    }
  ],
  "llm_connected": true,
  "time_taken": 6.82
}
```

---

#### B. `POST /api/search` — Raw Hybrid Search Endpoint
Performs vector retrieval and cross-encoder reranking directly without invoking the LLM generation stage. Useful for search UI components and reference verification.

**Request Body Example:**
```json
{
  "query": "شروط التقاعد والتأمين الصحي",
  "top_k": 5
}
```

**Response Body Example:**
```json
{
  "query": "شروط التقاعد والتأمين الصحي",
  "results": [
    {
      "rank": 1,
      "source_id": "text_4a5b6c7d",
      "title": "التأمين الصحي وصندوق التقاعد",
      "text": "يستحق المهندس الراتب التقائدي عند استكمال السن القانوني...",
      "similarity_score": 91
    }
  ]
}
```

---

#### C. `GET /api/stats` — System Health & Knowledge Base Stats
Returns operational metrics and counts for registered documents and indexed vector chunks. No authentication header required.

**Response Body Example:**
```json
{
  "total_documents": 76,
  "chroma_chunks": 89,
  "collection_name": "guild_knowledge_base"
}
```

---

### 3. Integration Code Examples

#### cURL Example
```bash
curl -X POST "http://localhost:8000/api/chat" \
     -H "Content-Type: application/json" \
     -H "X-API-Key: your_configured_rag_api_key" \
     -d '{
           "query": "كيف بقدر اشترك بالتأمين الصحي؟",
           "top_k": 5
         }'
```

#### Python Integration Example
```python
import requests

API_URL = "http://localhost:8000/api/chat"
API_KEY = "your_configured_rag_api_key"

payload = {
    "query": "شو الأوراق المطلوبة للانتساب؟",
    "history": [],
    "top_k": 5
}

headers = {
    "Content-Type": "application/json",
    "X-API-Key": API_KEY
}

try:
    response = requests.post(API_URL, json=payload, headers=headers, timeout=90)
    response.raise_for_status()
    data = response.json()

    print(f"Dialect Detected: {data['detected_accent']}")
    print(f"Normalized Query: {data['normalized_query']}")
    print(f"\nAnswer:\n{data['answer']}")
    print(f"\nTop Source: {data['sources'][0]['title']} ({data['sources'][0]['similarity_score']}% match)")

except requests.exceptions.RequestException as e:
    print(f"API Integration Error: {e}")
```

---

### 4. Error Handling & Best Practices

| Status Code | Reason | Cause | Solution |
| :--- | :--- | :--- | :--- |
| **`401 Unauthorized`** | Invalid/Missing Key | Missing or incorrect `X-API-Key` header | Verify `RAG_API_KEY` in `.env` matches request header |
| **`500 Internal Server Error`** | Server Misconfiguration | `RAG_API_KEY` not set on server environment | Configure `RAG_API_KEY` in `.env` |
| **`503 Service Unavailable`** | LLM Unreachable | Ollama / VLM process is down | Verify `ollama serve` is active |

1. **Stateless Callers**: The client application is responsible for holding user conversation history and providing recent context in `history`.
2. **Client Timeout**: Set HTTP client timeouts to at least **90 seconds** to accommodate reranking and local 7B LLM context inference.
3. **`llm_connected` Flag**: Always check `llm_connected` in the response payload. If `false`, the LLM service was unreachable and a fallback answer was produced.

---

## ⚙️ Configuration Reference (`config/settings.py`)

| Parameter | Default Value | Purpose |
| :--- | :--- | :--- |
| `QWEN_EMBEDDING_MODEL_NAME` | `Alibaba-NLP/gte-Qwen2-1.5B-instruct` | Dense vector embedding model |
| `BGE_RERANKER_MODEL_NAME` | `BAAI/bge-reranker-v2-m3` | Cross-encoder reranking model |
| `OLLAMA_CHAT_MODEL` | `qwen2.5:7b` | Ollama LLM for conversational responses |
| `OLLAMA_VISION_MODEL` | `qwen2.5vl:7b` | Ollama VLM for Arabic PDF visual table parsing |
| `TARGET_CHUNK_WORDS` | `700` | Target word count per document chunk |
| `CHUNK_OVERLAP_TOKENS` | `50` | Token overlap for chunk boundaries |

---

## 📄 License

MIT License. Designed and developed for the Jordan Engineers Association (نقابة المهندسين الأردنيين).
