import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent
TEXTS_DIR = BASE_DIR / "data" / "texts"
PDFS_DIR = BASE_DIR / "data" / "pdfs"
NEW_UPLOADS_DIR = BASE_DIR / "data" / "new_uploads"
FILES_DIR = BASE_DIR / "files"

# Ingestion Exclusion Flags
IGNORE_TEXTS_DIR = False
EXCLUDE_DIRS = ["الإعلانات والأخبار", "large_pdfs_over_30_pages"]

# Storage paths
STORAGE_DIR = BASE_DIR / "data" / "storage"
CHROMA_PERSIST_DIR = STORAGE_DIR / "chroma_db"
REGISTRY_DB_PATH = STORAGE_DIR / "registry.db"

# Ensure directories exist
STORAGE_DIR.mkdir(parents=True, exist_ok=True)
CHROMA_PERSIST_DIR.mkdir(parents=True, exist_ok=True)

# Vector DB settings
CHROMA_COLLECTION_NAME = "guild_knowledge_base"

# Embedding Model Settings
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "qwen")  # "qwen" or "bge-m3"
QWEN_EMBEDDING_MODEL_NAME = os.getenv("QWEN_EMBEDDING_MODEL_NAME", "Alibaba-NLP/gte-Qwen2-1.5B-instruct")
BGE_M3_MODEL_NAME = "BAAI/bge-m3"
BGE_RERANKER_MODEL_NAME = "BAAI/bge-reranker-v2-m3"
USE_FP16 = True

# Chunking settings (550 Token Chunks ~ 380-420 Words for stronger BGE reranker matches)
CHUNK_SIZE_TOKENS = 550
TARGET_CHUNK_WORDS = 400
MIN_CHUNK_WORDS = 250
MAX_CHUNK_WORDS = 650
OVERLAP_SENTENCES = 2
CHUNK_OVERLAP_TOKENS = 60

# Web Crawler settings
CRAWL_CACHE_DIR = STORAGE_DIR / "crawler_cache"
CRAWL_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Multi-Stage Ingestion Pipeline & Ollama Settings
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")  # "ollama" or "lmstudio"
VLM_PROVIDER = os.getenv("VLM_PROVIDER", "ollama")  # "ollama" or "lmstudio" or "mock"
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_URL = os.getenv("OLLAMA_URL", f"{OLLAMA_BASE_URL}/api/generate")
OLLAMA_CHAT_MODEL = os.getenv("OLLAMA_CHAT_MODEL", "ministral-3:8b")
OLLAMA_VISION_MODEL = os.getenv("OLLAMA_VISION_MODEL", "qwen2.5vl:7b")
LMSTUDIO_BASE_URL = os.getenv("LMSTUDIO_BASE_URL", "http://127.0.0.1:1234/v1")
LMSTUDIO_CHAT_MODEL = os.getenv("LMSTUDIO_CHAT_MODEL", "jais-adapted-7b-chat")
LMSTUDIO_VISION_MODEL = os.getenv("LMSTUDIO_VISION_MODEL", "qwen/qwen2.5-vl-7b")

# Vision PDF Parser Tuning
PARSED_OUTPUT_DIR = STORAGE_DIR / "pdf_parsed_results"
PARSED_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RENDER_DPI = 300
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID = (8, 8)
MIN_TABLE_AREA_FRACTION = 0.02
TABLE_UPSCALE_FACTOR = 2.0
TABLE_CROP_PADDING = 10
REQUEST_TIMEOUT = 300
NUM_CTX = 8192
NUM_PREDICT = 4096

VLM_MODEL_NAME = "llava:latest"
USE_CAMELOT_TABLES = True
ENABLE_VLM_VISUAL_REGIONS = True
MAX_VRAM_GB = 8.0

UNIFIED_VISION_PROMPT = """You are an expert document parser. This is a page from an Arabic document
that may contain Arabic text, English text, and tables.

Instructions:
1. Transcribe all text exactly as it appears, preserving reading order (right-to-left for Arabic).
2. If the page contains one or more tables, output them as clean HTML <table> elements
   (use <table>, <tr>, <td>, and colspan/rowspan attributes if cells are merged).
   Do NOT use Markdown tables -- HTML only, since it supports merged cells.
3. Output all non-table text as plain paragraphs, in the correct order relative to
   where they appear on the page (e.g. a heading before a paragraph before a table).
4. Do not translate anything. Keep Arabic text in Arabic and English text in English.
5. Do not add commentary, explanations, or notes of your own. Output ONLY the transcribed
   content (paragraphs + HTML tables), nothing else.
6. If a region of the page is unreadable, mark it as [UNREADABLE] rather than guessing.

IMPORTANT - Arabic tables with merged/nested headers:
Arabic tables are read right-to-left, and often have a top-level header cell that spans
multiple sub-columns below it (colspan). Do not flatten these into a single row of numbers.
Keep the header hierarchy explicit using colspan, and keep column order as it visually
appears on the page (rightmost visual column = first <td> in each row, since this is RTL).
"""
