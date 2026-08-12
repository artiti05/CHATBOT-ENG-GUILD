# 🇯🇴 Arabic PDF Parser & Hybrid RAG Chatbot System
### Jordan Engineers Association (نقابة المهندسين الأردنيين)

An end-to-end **Arabic PDF Vision Parser, Hybrid RAG (Retrieval-Augmented Generation) Engine, and AI Chatbot System** built specifically for the Jordan Engineers Association.

It processes complex Arabic documents, handles reversed Bidi character streams, transcribes multi-header RTL tables using **Qwen2.5-VL**, embeds document chunks using **GTE-Qwen / BAAI BGE-M3**, reranks search results via **BGE Cross-Encoder**, normalizes **Jordanian Dialect** queries into formal Arabic, and serves a secure FastAPI Web UI and REST API.

---

## Table of Contents
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

##  What the Project is About

The Jordan Engineers Association manages an extensive collection of legal bylaws, strategic plans, service guides, university admission criteria, and operational news. This system provides an intelligent knowledge base assistant capable of:

1. **Vision-Based Document Parsing**: Uses OpenCV CLAHE image enhancement and **Qwen2.5-VL** (via Ollama or LM Studio) to transcribe Arabic prose and format complex nested RTL tables as HTML `<table>` elements with accurate `colspan` and `rowspan`.
2. **Jordanian Dialect Normalization**: Translates colloquial Jordanian phrases (e.g., *"شو الأوراق"*, *"بدي أسجل"*, *"قديش الإشتراك"*) into formal Modern Standard Arabic (MSA) search terms for accurate semantic retrieval.
3. **Hybrid Retrieval & Reranking (Dense + Arabic BM25 + RRF $k=60$)**: Combines dense vector similarity search in **ChromaDB (BGE-M3)** with sparse keyword search (**Arabic Normalized BM25**) fused via Reciprocal Rank Fusion ($k=60$), then reranks results using **BAAI/bge-reranker-v2-m3** with a strict 50% quality floor.
4. **Hierarchical Parent-Child Chunking**: Indexes granular **Child Chunks (~150 tokens)** for high-precision retrieval while providing rich **Parent Context Blocks (~800 tokens)** to the LLM for answer synthesis.
5. **FastAPI Web UI & REST API**: Provides an ambient dark-mode web application and secure API endpoints protected by header authentication.
6. **Agentic Multi-Agent StateGraph & CRAG Reflexion**: Orchestrates 6 specialized subagents in a state machine (`core/graph.py`) with automatic query expansion and self-correction loops when retrieval confidence is low.


---

## System Architecture & Storage

```
CHATBOT-ENG-GUILD/
├── main.py                     # Entry point & CLI launcher
│
├── api/                        # FastAPI Web Server & REST API endpoints
│
├── core/                       # Core AI & RAG Engine Modules
│   ├── agents/                 # Agentic Multi-Agent System (6 Subagents)
│   │   ├── retriever_agent.py  # Retriever Agent (Hybrid Dense + Arabic BM25 + RRF k=60)
│   │   ├── reranker_agent.py   # Reranker Agent (Cross-Encoder 50% Floor & Priority Boost)
│   │   ├── rewriter_agent.py   # Dialect Rewriter Agent (Multi-Turn Synthesis & Dialect-to-MSA)
│   │   ├── verifier_agent.py   # Verifier Agent (CRAG Reflexion & Groundedness Checker)
│   │   ├── generator_agent.py # Response Generator Agent (Persona & Stream Handler)
│   │   └── cache_agent.py      # Semantic Query Cache Agent (< 10ms Latency)
│   ├── bm25_search.py          # Arabic Sparse Keyword Search Indexer
│   ├── config.py               # Global Settings, Paths & Hyperparameters
│   ├── graph.py                # Agentic Multi-Agent StateGraph Orchestrator
│   ├── ingestion.py            # Vision PDF Parser & Parent-Child Chunker
│   └── rag_engine.py           # RAG Chatbot Coordinator (Delegates to RAGStateGraph)
│
├── data/                       # Unified Storage & Knowledge Directory
│   ├── texts/                  # 430+ Raw Text KB files (.md, .txt)
│   ├── markdowns/              # Clean pre-parsed PDF Markdown files (*_parsed.md)
│   ├── pdfs/                   # Un-parsed source PDF documents
│   ├── output_dir/             # PDF Vision Parser debug artifacts & page images
│   ├── chunking_logs/          # Chunk inspection reports (JSON)
│   ├── crawler/                # Web crawler cache & crawled documents
│   └── storage/                # ChromaDB vector store, BM25 index & SQLite registry
│
└── scripts/                    # Utilities, Benchmarks & Test Suites
    ├── batch_ingest.py         # Batch Ingestion Manager
    ├── consolidate_markdowns.py # Data Directory Consolidation Helper
    ├── inspect_chunking.py     # Document Parent-Child Chunk Inspector
    ├── test_pipeline.py        # Multi-Agent Pipeline Regression Benchmark Suite
    ├── verify_rag_retrieval.py # RAG Retrieval Layer Test Suite
    ├── verify_agentic_context.py # Multi-Turn & Agentic CRAG Test Suite
    ├── verify_performance_cache.py # Semantic Cache & Latency Test Suite
    └── verify_agentic_graph.py # Agentic StateGraph & Reflexion Test Suite
```



### Storage Architecture
- **SQLite Registry (`data/storage/registry.db`)**: Tracks document metadata, SHA-256 hashes, file types, page counts, and parsing statuses.
- **ChromaDB Store (`data/storage/chroma_db`)**: Persists dense vector embeddings under collection `guild_knowledge_base`.
- **BM25 Store (`data/storage/bm25_index.pkl`)**: Persists Arabic sparse keyword index.


---

## Prerequisites & Setup

### 1. Python Environment Setup
```powershell
# Clone repository
git clone https://github.com/artiti05/CHATBOT-ENG-GUILD.git
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

##  How to Run

All execution modes are controlled through the unified entry point `main.py`:


### Launch the Web Application & Chatbot UI
```powershell
python main.py
```
*(Or specify a custom port: `python main.py serve --port 8000`)*

Open your browser at **`http://localhost:8000`** *(automatically detects alternative ports if 8000 is occupied)*.

---

##  Knowledge Base Handling & Ingestion

The system ingests knowledge sources from standardized folders inside `data/`:

| Source Directory | Format | Content Description | Ingestion Method |
| :--- | :--- | :--- | :--- |
| **`data/texts/`** | `.md` / `.txt` | 430+ text knowledge files (Insurance, Laws, Services) | Text Cleaning & Parent-Child Token Chunking |
| **`data/markdowns/`** | `.md` | Pre-parsed Markdown files (`*_parsed.md`) from PDF Vision OCR | Fast Markdown Indexer (No GPU needed) |
| **`data/pdfs/`** | `.pdf` | Official source PDF publications | Qwen2.5-VL Vision Parsing + OCR |
| **`data/output_dir/`** | Artifacts | Visual page images, table debug crops, per-page text & PDF copies | Output Artifact Storage |

---

### 📥 CLI Ingestion Commands Reference Table

| Ingestion Command | Target Directory | Description & Purpose | GPU / VLM Needed? | DB Reset Default | Key Flags |
| :--- | :--- | :--- | :---: | :---: | :--- |
| **`python main.py ingest-kb`** | `data/texts/`<br>`data/markdowns/` | **Recommended Build**: Embeds all raw text files and pre-parsed markdowns into ChromaDB + BM25. Skips PDF parsing. | ❌ No |  Wipes DB | `--no-reset-db`<br>`--dry-run` |
| **`python main.py ingest-texts`** | `data/texts/` | Ingests only raw text knowledge base files (`.md`, `.txt`) from `data/texts/`. | ❌ No |  Wipes DB | `--no-reset-db`<br>`--dry-run` |
| **`python main.py ingest-markdowns`** | `data/markdowns/` | Ingests only pre-parsed Markdown documents (`*_parsed.md`). | ❌ No |  Wipes DB | `--no-reset-db`<br>`--dry-run` |
| **`python main.py ingest-pdfs`** | `data/pdfs/` | Scans `data/pdfs/` for un-parsed source PDFs and runs OpenCV table detection + Qwen2.5-VL Vision parsing. |  Yes |  Wipes DB | `--no-reset-db`<br>`--dry-run` |
| **`python main.py ingest-all`** | `data/texts/`<br>`data/markdowns/`<br>`data/pdfs/` | **Full End-to-End Build**: Ingests texts, pre-parsed markdowns, and vision-parses any un-parsed source PDFs in `data/pdfs/`. |  (PDFs only) |  Wipes DB | `--no-reset-db`<br>`--dry-run` |
| **`python main.py inspect-chunks`** | `data/markdowns/`<br>`data/texts/` | Interactive or file-specific Parent-Child token chunking inspector. Saves JSON report to `data/chunking_logs/`. | ❌ No | N/A (Read-only) | `--file <path>` |

#### Flag Descriptions:
- **`--no-reset-db`** *(Append Mode)*: Appends vectors to the existing ChromaDB collection without wiping previously indexed documents.
- **`--dry-run`** *(Preview Mode)*: Previews file counts, chunk counts, and execution plan without modifying database storage or loading GPU models.


> ⚠️ **Important Safety Rule:** Do not run ingestion scripts while a live Web API server is performing heavy write operations on `storage/chroma_db` to avoid lock contention.

---

##  Security & API Integration Guide

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

##  Configuration Reference (`config/settings.py`)

| Parameter | Default Value | Purpose |
| :--- | :--- | :--- |
| `QWEN_EMBEDDING_MODEL_NAME` | `Alibaba-NLP/gte-Qwen2-1.5B-instruct` | Dense vector embedding model |
| `BGE_RERANKER_MODEL_NAME` | `BAAI/bge-reranker-v2-m3` | Cross-encoder reranking model |
| `OLLAMA_CHAT_MODEL` | `qwen2.5:7b` | Primary Ollama LLM for conversational responses |
| `OLLAMA_VISION_MODEL` | `qwen2.5vl:7b` | Ollama VLM for Arabic PDF visual table parsing |

---

## 📊 Master Feature Comparison Matrix

| Feature Module | Legacy Baseline | SOTA GitHub Baseline (2025/2026) | Current Upgraded System | Status |
| :--- | :--- | :--- | :--- | :---: |
| **Retrieval Strategy** | Pure Dense Search (ChromaDB) | Hybrid Search: Dense + BM25 + RRF | **Hybrid Search (ChromaDB + Arabic BM25 + RRF $k=60$)** |  Resolved |
| **Query Processing** | Hardcoded if/elif regex list (~20 rules) | HyDE & Multi-Query LLM Rewrite | **LLM Dialect Rewriter (Levantine/Gulf/Egyptian $\rightarrow$ MSA)** |  Resolved |
| **Multi-Turn Context** | Raw text appended post-retrieval | Standalone Query Rewriter | **Standalone Query Synthesis before retrieval** |  Resolved |
| **Chunking Architecture** | Fixed word-count chunking (700-950 tokens) | Parent-Child / Hierarchical Chunking | **Token-accurate Parent-Child Chunking (~800/~150 tokens)** |  Resolved |
| **Orchestration** | Single linear script | Agentic State Graph (LangGraph) | **Agentic StateGraph System (6 Subagents in `core/graph.py`)** |  Resolved |
| **Self-Correction** | Static relevance cutoff threshold | CRAG / Self-RAG Reflection Loops | **CRAG Verifier + Groundedness Check + Re-retrieval Loop** |  Resolved |
| **Performance / Caching** | None (full pipeline on every query) | Semantic Vector Cache | **In-Memory & Persisted Cosine Semantic Cache (<10ms for hits)** |  Resolved |
| **Knowledge Graph** | None | GraphRAG / LightRAG | **Vector + BM25 Hybrid (KG optional for future scale)** | 🟡 Optional |
| **Evaluation** | Manual inspection script | Automated RAGAS / DeepEval | **Pipeline Benchmark Suite (`scripts/test_pipeline.py`)** |  Resolved |

| `CHUNK_SIZE_TOKENS` | `300` | Target chunk size (300 tokens ~ 200–220 words) |
| `CHUNK_OVERLAP_TOKENS` | `40` | Token overlap for chunk boundaries |

---

##  License

MIT License. Designed and developed for the Jordan Engineers Association (نقابة المهندسين الأردنيين).
