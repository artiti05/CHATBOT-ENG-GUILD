# 🇯🇴 Arabic PDF Parser & Hybrid RAG Chatbot System
### Jordan Engineers Association (نقابة المهندسين الأردنيين)

An end-to-end **Arabic PDF Vision Parser, Hybrid RAG (Retrieval-Augmented Generation) Engine, and AI Chatbot System** built specifically for the Jordan Engineers Association.

It processes complex Arabic documents, handles reversed Bidi character streams, transcribes multi-header RTL tables using **Qwen2.5-VL**, embeds document chunks using **GTE-Qwen / BAAI BGE-M3**, reranks search results via **BGE Cross-Encoder**, normalizes **Jordanian Dialect** queries into formal Arabic, and serves a secure FastAPI Web UI, Admin Dashboard, and REST API.

---

## Table of Contents
- [What the Project is About](#-what-the-project-is-about)
- [System Architecture & Storage](#-system-architecture--storage)
- [Quick Start Guide](#-quick-start-guide)
  - [Option A: Running with Docker (Recommended)](#option-a-running-with-docker-recommended)
  - [Option B: Running Locally with Python](#option-b-running-locally-with-python)
- [CLI Ingestion Commands Reference](#-cli-ingestion-commands-reference)
- [Security & API Integration Guide](#-security--api-integration-guide)
  - [Authentication & CORS](#1-authentication--cors)
  - [API Endpoints Reference](#2-api-endpoints-reference)
  - [Integration Code Examples](#3-integration-code-examples)
  - [Error Handling & Best Practices](#4-error-handling--best-practices)
- [Configuration Reference](#-configuration-reference)
- [Master Feature Matrix](#-master-feature-comparison-matrix)
- [License](#-license)

---

## 💡 What the Project is About

The Jordan Engineers Association manages an extensive collection of legal bylaws, strategic plans, service guides, university admission criteria, and operational news. This system provides an intelligent knowledge base assistant capable of:

1. **Vision-Based Document Parsing**: Uses OpenCV CLAHE image enhancement and **Qwen2.5-VL** (via Ollama) to transcribe Arabic prose and format complex nested RTL tables as HTML `<table>` elements with accurate `colspan` and `rowspan`.
2. **Jordanian Dialect Normalization**: Translates colloquial Jordanian phrases (e.g., *"شو الأوراق"*, *"بدي أسجل"*, *"قديش الإشتراك"*) into formal Modern Standard Arabic (MSA) search terms for accurate semantic retrieval.
3. **Hybrid Retrieval & Reranking (Dense + Arabic BM25 + RRF $k=60$)**: Combines dense vector similarity search in **ChromaDB (BGE-M3)** with sparse keyword search (**Arabic Normalized BM25**) fused via Reciprocal Rank Fusion ($k=60$), then reranks results using **BAAI/bge-reranker-v2-m3** with a strict 50% quality floor.
4. **Hierarchical Parent-Child Chunking**: Indexes granular **Child Chunks (~150 tokens)** for high-precision retrieval while providing rich **Parent Context Blocks (~800 tokens)** to the LLM for answer synthesis.
5. **FastAPI Web UI, Admin Dashboard & REST API**: Provides an ambient dark-mode web UI (`http://localhost:8000`), an interactive Admin Dashboard (`http://localhost:8000/admin/ui`), and secure API endpoints protected by header authentication (`ADMIN_API_KEY` & `USER_API_KEY`).
6. **Agentic 4-Stage RAG Pipeline & CRAG Reflexion**: Orchestrates intent classification, hybrid retrieval, cross-encoder verification, and response generation with automatic query expansion and self-correction loops when retrieval confidence is low.

---

## 🏗️ System Architecture & Storage

```
CHATBOT-ENG-GUILD/
├── main.py                     # Unified CLI entry point & Web UI launcher
├── Dockerfile                  # Container build file (Python 3.12-slim)
├── docker-compose.yml          # Docker Compose orchestration (FastAPI RAG + Ollama)
├── .env.example                # Environment template
├── requirements.txt            # Python dependencies
│
├── src/                        # Modular RAG Architecture
│   ├── api/                    # FastAPI Server & REST API Endpoints
│   │   ├── main.py             # FastAPI App instance & middleware
│   │   ├── routes_chat.py      # Conversational RAG & Search endpoints
│   │   ├── routes_admin.py     # Admin VDB management routes
│   │   ├── routes_admin_ui.py  # Admin HTML UI dashboard
│   │   └── dependencies.py     # API key authentication & security dependencies
│   ├── cache_db/               # Storage Registries & Caches
│   │   ├── document_registry.py# SQLite Document Registry (metadata, SHA256 hashes)
│   │   └── semantic_cache.py   # In-Memory/SQLite Cosine Semantic Cache (<10ms)
│   ├── core/                   # Global Engine & Configuration
│   │   ├── config.py           # Settings, paths, model names & hyperparameters
│   │   └── rag_engine.py       # Core RAG Coordinator
│   ├── monitoring/             # Diagnostics & Profiling
│   │   ├── diagnostic_report.py# System diagnostic generator
│   │   ├── profiler.py         # Latency & memory profiler
│   │   └── tracing.py          # Request tracing utilities
│   ├── pipeline/               # 4-Stage Agentic RAG Pipeline
│   │   ├── stage_01_understand/# Arabic/Jordanian Normalization & Intent Classification
│   │   ├── stage_02_retrieve/  # Hybrid Search (Dense + Arabic BM25 + RRF + Ingestion)
│   │   ├── stage_03_verify_rerank/ # BGE Cross-Encoder Reranking & Verification
│   │   └── stage_04_answer/    # Answer Synthesis, Dialect Rewriter & Generator Agents
│   ├── services/               # Domain Services
│   │   └── ticketing_service.py# Ticketing integration service
│   └── rag_chatbot_engine.py   # Engine Entrypoint
│
├── data/                       # Knowledge Base & Persistent Storage
│   ├── texts/                  # Raw text/markdown KB files (.md, .txt)
│   ├── pdfs/                   # Source PDF documents
│   ├── markdowns/              # Pre-parsed PDF Markdown files (*_parsed.md)
│   ├── output_dir/             # PDF Vision Parser debug artifacts & images
│   └── storage/                # ChromaDB vector store, BM25 index & SQLite registry.db
│
├── scripts/                    # Ingestion & Benchmark Utilities
│   ├── batch_ingest.py         # Batch Ingestion Manager
│   └── test_pipeline.py        # Multi-Agent Pipeline Test Suite
└── tests/                      # Pytest Unit & Integration Test Suites
```

### Storage Architecture
- **SQLite Registry (`data/storage/registry.db`)**: Tracks document metadata, SHA-256 hashes, file types, page counts, and parsing statuses.
- **ChromaDB Store (`data/storage/chroma_db`)**: Persists dense vector embeddings under collection `guild_knowledge_base`.
- **BM25 Store (`data/storage/bm25_index.pkl`)**: Persists Arabic sparse keyword index.

---

## 🚀 Quick Start Guide

### Prerequisites
- **Docker & Docker Compose** (for Docker setup), **OR** **Python 3.10–3.12** (for local setup).
- **Ollama** installed with models pulled:
  ```bash
  ollama pull qwen2.5:7b
  ollama pull qwen2.5vl:7b
  ```

---

### Option A: Running with Docker (Recommended)

Docker Compose sets up both the **FastAPI RAG Web Server** and **Ollama** in orchestrated containers.

1. **Clone the repository and copy environment configuration:**
   ```bash
   git clone https://github.com/artiti05/CHATBOT-ENG-GUILD.git
   cd CHATBOT-ENG-GUILD
   cp .env.example .env
   ```

2. **Configure API Keys in `.env`:**
   ```ini
   ADMIN_API_KEY=your_admin_secret_key
   USER_API_KEY=your_user_secret_key
   ```

3. **Build and start the container stack:**
   ```bash
   docker-compose up --build
   ```

4. **Access the application:**
   - **Web Chatbot UI**: `http://localhost:8000`
   - **Admin UI Dashboard**: `http://localhost:8000/admin/ui`
   - **API Docs (Swagger)**: `http://localhost:8000/docs`

---

### Option B: Running Locally with Python

1. **Clone repository and set up Virtual Environment:**
   ```powershell
   git clone https://github.com/artiti05/CHATBOT-ENG-GUILD.git
   cd CHATBOT-ENG-GUILD

   # Create virtual environment
   python -m venv venv

   # Activate virtual environment (Windows PowerShell)
   .\venv\Scripts\Activate.ps1
   # Linux/macOS:
   # source venv/bin/activate

   # Install dependencies
   pip install -r requirements.txt
   ```

2. **Set up Environment Variables:**
   ```powershell
   Copy-Item .env.example .env
   ```
   Edit `.env` to set your API keys (`ADMIN_API_KEY`, `USER_API_KEY`) and Ollama settings.

3. **Ensure Ollama is running:**
   ```bash
   ollama serve
   ```

4. **Launch the Web Application & Chatbot UI:**
   ```bash
   python main.py
   ```
   *(Or specify a custom port: `python main.py serve --port 8000`)*

   Open your browser at **`http://localhost:8000`**.

---

## 📥 CLI Ingestion Commands Reference

All ingestion workflows are controlled via the unified `main.py` entry point:

| Ingestion Command | Target Directory | Description & Purpose | Vision GPU Needed? | DB Reset Default | Key Flags |
| :--- | :--- | :--- | :---: | :---: | :--- |
| **`python main.py ingest-texts`** | `data/texts/` | Ingests raw text knowledge base files (`.md`, `.txt`) into ChromaDB + BM25. | ❌ No | Wipes DB | `--no-reset-db`<br>`--dry-run` |
| **`python main.py ingest-markdown`** | `data/output_dir/` | Ingests pre-parsed Markdown documents (`*_parsed.md`) from output directory. | ❌ No | Wipes DB | `--no-reset-db`<br>`--dry-run` |
| **`python main.py ingest-pdfs`** | `data/pdfs/` | Vision parses un-parsed PDFs in `data/pdfs/` using Qwen2.5-VL and embeds chunks into ChromaDB. |  Yes | Wipes DB | `--no-reset-db`<br>`--dry-run` |
| **`python main.py ingest-all`** | `data/texts/`<br>`data/output_dir/`<br>`data/pdfs/` | **Full Build**: Ingests texts, pre-parsed markdowns, and vision-parses un-parsed source PDFs. |  (PDFs only) | Wipes DB | `--no-reset-db`<br>`--dry-run` |

#### Flag Descriptions:
- **`--no-reset-db`** *(Append Mode)*: Appends vectors to the existing ChromaDB collection without wiping previously indexed documents.
- **`--dry-run`** *(Preview Mode)*: Previews file counts and execution plan without modifying database storage.

---

## 🔐 Security & API Integration Guide

### 1. Authentication & CORS
Protected endpoints require header authentication.

- **Header Name**: `X-API-Key`
- **Configuration**: Set via `USER_API_KEY` or `ADMIN_API_KEY` in `.env`.
- **CORS Restrictions**: Configured in `.env` under `CORS_ORIGINS`.

```http
X-API-Key: your_super_secret_user_key_here
Content-Type: application/json
```

---

### 2. API Endpoints Reference

#### A. `POST /api/chat` — Conversational RAG Endpoint
Executes query translation (Jordanian Dialect -> MSA), hybrid dense vector retrieval in ChromaDB, cross-encoder reranking, and local LLM context-augmented generation.

**Request Headers:**
```http
X-API-Key: <USER_API_KEY>
Content-Type: application/json
```

**Request Body Example:**
```json
{
  "query": "شو الأوراق المطلوبة للتسجيل بالنقابة؟",
  "history": [],
  "top_k": 10
}
```

**Response Body Example:**
```json
{
  "query": "شو الأوراق المطلوبة للتسجيل بالنقابة؟",
  "normalized_query": "ما هي الوثائق والمستندات المطلوبة للتسجيل والانتساب لنقابة المهندسين؟",
  "detected_accent": "jordanian",
  "answer": "للتسجيل والانتساب بنقابة المهندسين الأردنيين، يجب تقديم الوثائق التالية:\n1. صورة مصدقة عن شهادة الدراسة الثانوية العامة.\n2. صورة مصدقة عن مصدقة البكالوريوس والشهادة الجامعية.\n...",
  "sources": [
    {
      "rank": 1,
      "source_id": "text_8f2a1b9c",
      "title": "الانتساب للنقابة",
      "text": "الوثائق المطلوبة للانتساب: 1- صورة مصدقة...",
      "similarity_score": 94
    }
  ],
  "llm_connected": true,
  "time_taken": 6.82
}
```

---

#### B. `POST /api/search` — Raw Hybrid Search Endpoint
Performs vector retrieval and cross-encoder reranking directly without invoking the LLM generation stage.

**Request Body Example:**
```json
{
  "query": "شروط التقاعد والتأمين الصحي",
  "top_k": 5
}
```

---

#### C. `GET /api/stats` — System Health & Knowledge Base Stats
Returns operational metrics and counts for registered documents and indexed vector chunks. No authentication header required.

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
     -H "X-API-Key: your_super_secret_user_key_here" \
     -d '{
           "query": "كيف بقدر اشترك بالتأمين الصحي؟",
           "top_k": 5
         }'
```

#### Python Integration Example
```python
import requests

API_URL = "http://localhost:8000/api/chat"
API_KEY = "your_super_secret_user_key_here"

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
| **`401 Unauthorized`** | Invalid/Missing Key | Missing or incorrect `X-API-Key` header | Verify `USER_API_KEY` or `ADMIN_API_KEY` in `.env` |
| **`500 Internal Server Error`** | Server Error | Missing environment setup or internal error | Check server logs for full traceback |
| **`503 Service Unavailable`** | LLM Unreachable | Ollama / VLM process is down | Verify `ollama serve` is active |

---

## ⚙️ Configuration Reference

| Parameter | Default Value | Purpose |
| :--- | :--- | :--- |
| `QWEN_EMBEDDING_MODEL_NAME` | `Alibaba-NLP/gte-Qwen2-1.5B-instruct` | Dense vector embedding model |
| `BGE_RERANKER_MODEL_NAME` | `BAAI/bge-reranker-v2-m3` | Cross-encoder reranking model |
| `OLLAMA_CHAT_MODEL` | `qwen2.5:7b` | Primary Ollama LLM for conversational responses |
| `OLLAMA_VISION_MODEL` | `qwen2.5vl:7b` | Ollama VLM for Arabic PDF visual table parsing |

---

## 📊 Master Feature Comparison Matrix

| Feature Module | Legacy Baseline | Current Upgraded System | Status |
| :--- | :--- | :--- | :---: |
| **Retrieval Strategy** | Pure Dense Search (ChromaDB) | **Hybrid Search (ChromaDB + Arabic BM25 + RRF $k=60$)** |  Resolved |
| **Query Processing** | Hardcoded regex rules | **LLM Dialect Rewriter (Levantine/Jordanian $\rightarrow$ MSA)** |  Resolved |
| **Multi-Turn Context** | Raw text appended | **Standalone Query Synthesis before retrieval** |  Resolved |
| **Chunking Architecture** | Fixed word-count chunking | **Token-accurate Parent-Child Chunking (~800/~150 tokens)** |  Resolved |
| **Orchestration** | Single linear script | **4-Stage Modular Pipeline (`src/pipeline/`)** |  Resolved |
| **Self-Correction** | Static threshold | **CRAG Verifier + Groundedness Check + Re-retrieval** |  Resolved |
| **Performance / Caching** | None | **In-Memory & Persisted Cosine Semantic Cache (<10ms)** |  Resolved |
| **Containerization** | None | **Production Dockerfile & Docker Compose Stack** |  Resolved |

---

## 📜 License

MIT License. Designed and developed for the Jordan Engineers Association (نقابة المهندسين الأردنيين).
