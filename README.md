# 🇯🇴 Arabic PDF Parser & Hybrid RAG Chatbot (Jordan Engineers Association / نقابة المهندسين الأردنيين)

An end-to-end **Arabic PDF Vision Parser, Hybrid RAG (Retrieval-Augmented Generation) Engine, and AI Chatbot System** built for the Jordan Engineers Association.

It handles complex Arabic document formats, reversed Bidi character streams, and multi-header RTL tables using **Qwen2.5-VL**, embeds document chunks using **BAAI BGE-M3**, reranks search results via **BGE Cross-Encoder**, and serves a FastAPI web UI supporting Modern Standard Arabic (MSA), English, and **Jordanian Dialect** queries.

---

## 🌟 Key Features

* **👁️ Single-Pass Vision PDF Parser**: Uses PyMuPDF at 300 DPI + OpenCV CLAHE contrast enhancement + **Qwen2.5-VL** (via Ollama or LM Studio) to transcribe Arabic prose and format nested RTL tables as HTML `<table>` elements (`colspan`/`rowspan`).
* **🗣️ Jordanian Dialect Normalization**: Intelligent query translation engine that converts colloquial phrases (e.g., *"شو الأوراق"*, *"بدي أسجل"*, *"قديش"*) into formal Arabic search terms.
* **⚡ Hybrid Vector Retrieval & Cross-Encoder Reranking**: Uses **BAAI/bge-m3** for dense vector similarity in ChromaDB, followed by **BAAI/bge-reranker-v2-m3** for cross-encoder reranking.
* **🚀 Multi-Stage Batch Ingestion Utility (`batch_ingest.py`)**: Supports database resets, pre-parsed Markdown indexing, dry-run execution previews, and background GPU VRAM parsing queues.
* **🎨 Modern Web UI**: Responsive FastAPI & HTML5 interface featuring ambient dark mode, suggestion chips, similarity match badges, and collapsible source citations.

---

## 📁 System Architecture

```
arabic-pdf-parser/
├── app.py                      # FastAPI Web Application & UI server
├── rag_chatbot_engine.py       # RAG Retriever, BGE Reranker & Jordanian Chatbot Engine
├── ingestion_pipeline.py       # Qwen2.5-VL Vision Parser, Text Cleaner, Chunker & Chroma Indexer
├── crawler_admin.py            # SQLite Document Registry & Admin Manager
├── batch_ingest.py             # CLI Tool for Batch Ingestion & Database Management
├── config/
│   └── settings.py             # Global Configuration & Model Hyperparameters
├── output_dir/                 # Pre-parsed Markdown files & output artifacts
├── pdfs/                       # Source PDF documents directory
└── storage/                    # Persistent ChromaDB vector store & SQLite registry.db
```

---

## 🛠️ Prerequisites & Setup

### 1. Python Environment
```bash
# Clone the repository
git clone https://github.com/shatnawiO/arabic-pdf-parser.git
cd arabic-pdf-parser

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Ollama & Vision Model Setup
Ensure [Ollama](https://ollama.com) is installed and running locally with the vision model pulled:
```bash
ollama pull qwen2.5vl:7b
ollama serve
```

*(Alternatively, LM Studio can be used by setting `VLM_PROVIDER=lmstudio` in environment variables).*

---

## 🚀 How to Run

### 1. Batch Ingestion & Vector Database Management (`batch_ingest.py`)

The `batch_ingest.py` script manages database initialization, indexing pre-parsed Markdown files, and batch processing remaining PDFs on GPU.

```bash
# Mode A: Preview execution plan without modifying DB or VLM
python batch_ingest.py --dry-run

# Mode B: Reset VDB & index pre-parsed output_dir Markdown files NOW (No VLM calls)
python batch_ingest.py --index-only

# Mode C: Full Pipeline (Reset VDB -> Index pre-parsed files -> GPU parse remaining PDFs)
python batch_ingest.py

# Mode D: Append-only mode (Parse remaining PDFs without resetting existing VDB)
python batch_ingest.py --no-reset-db
```

### 2. Running the RAG Chatbot Web Application (`app.py`)

Launch the web application using Uvicorn:

```bash
python -m uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

Open your browser and navigate to:
👉 **`http://localhost:8000`**

---

## 📡 API Endpoints

| Endpoint | Method | Description |
| :--- | :--- | :--- |
| `/` | `GET` | Serves the interactive Arabic/Jordanian AI Chatbot Web Interface |
| `/api/chat` | `POST` | RAG endpoint. Accepts query & conversation history, returns generated answer + top 10 sources |
| `/api/search` | `POST` | Raw hybrid vector search endpoint returning reranked document chunks |
| `/api/stats` | `GET` | Returns total indexed document count and ChromaDB chunk count |

---

## 🔧 Configuration Options (`config/settings.py`)

Key parameters in [settings.py](file:///c:/Users/VICTUS/Desktop/arabic-pdf-parser/config/settings.py):

| Variable | Default Value | Purpose |
| :--- | :--- | :--- |
| `BGE_M3_MODEL_NAME` | `BAAI/bge-m3` | Dense embedding model |
| `BGE_RERANKER_MODEL_NAME` | `BAAI/bge-reranker-v2-m3` | Cross-encoder reranking model |
| `OLLAMA_VISION_MODEL` | `qwen2.5vl:7b` | Ollama Vision LLM model tag |
| `CHUNK_SIZE_TOKENS` | `400` | Token chunking size |
| `CHUNK_OVERLAP_TOKENS` | `50` | Token chunking overlap |
| `NUM_CTX` | `8192` | VLM context window size |
| `NUM_PREDICT` | `4096` | Max output tokens for VLM transcription |

---

## 📜 License

MIT License. Designed and developed for the Jordan Engineers Association (نقابة المهندسين الأردنيين).
