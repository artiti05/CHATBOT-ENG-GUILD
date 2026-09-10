import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Base paths (src/ -> repo root)
BASE_DIR = Path(__file__).resolve().parent.parent
TEXTS_DIR = BASE_DIR / "data" / "texts"
MARKDOWNS_DIR = BASE_DIR / "data" / "markdowns"
PDFS_DIR = BASE_DIR / "data" / "pdfs"
OUTPUT_DIR = BASE_DIR / "data" / "output_dir"
PARSED_OUTPUT_DIR = OUTPUT_DIR  # Alias for vision parser output directory
CHUNKING_LOGS_DIR = BASE_DIR / "data" / "chunking_logs"
CRAWLER_CACHE_DIR = BASE_DIR / "data" / "crawler"
CRAWL_CACHE_DIR = CRAWLER_CACHE_DIR  # Alias for backward compatibility
NEW_UPLOADS_DIR = BASE_DIR / "data" / "new_uploads"
LOGS_DIR = BASE_DIR / "data" / "logs"
FILES_DIR = BASE_DIR / "files"


# Ensure essential data directories exist
for p in (TEXTS_DIR, MARKDOWNS_DIR, PDFS_DIR, OUTPUT_DIR, CHUNKING_LOGS_DIR, CRAWLER_CACHE_DIR, LOGS_DIR):
    p.mkdir(parents=True, exist_ok=True)

# Ingestion Exclusion Flags
IGNORE_TEXTS_DIR = False
EXCLUDE_DIRS = ["الإعلانات والأخبار", "large_pdfs_over_30_pages"]

# Storage paths
STORAGE_DIR = BASE_DIR / "data" / "storage"
CHROMA_PERSIST_DIR = STORAGE_DIR / "chroma_db"
REGISTRY_DB_PATH = STORAGE_DIR / "registry.db"
BM25_INDEX_PATH = STORAGE_DIR / "bm25_index.pkl"
CACHE_PERSIST_PATH = STORAGE_DIR / "semantic_cache.json"

# Semantic Cache Settings
CACHE_SIMILARITY_THRESHOLD = 0.90
CACHE_MAX_ENTRIES = 1000

# Moderation & Outbound Alert Webhook Settings
ENABLE_MODERATION_ALERTS = os.getenv("ENABLE_MODERATION_ALERTS", "true").lower() in ("1", "true", "yes")
MODERATION_WEBHOOK_URL = os.getenv("MODERATION_WEBHOOK_URL", "")
MODERATION_WEBHOOK_TOKEN = os.getenv("MODERATION_WEBHOOK_TOKEN", "")
MODERATION_WEBHOOK_TIMEOUT = float(os.getenv("MODERATION_WEBHOOK_TIMEOUT", "3.0"))

# Ensure storage directories exist
STORAGE_DIR.mkdir(parents=True, exist_ok=True)
CHROMA_PERSIST_DIR.mkdir(parents=True, exist_ok=True)

# Vector DB settings
CHROMA_COLLECTION_NAME = "guild_knowledge_base"

# Embedding & Neural Rewriter Model Settings
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "qwen")  # "qwen" or "bge-m3"
QWEN_EMBEDDING_MODEL_NAME = os.getenv("QWEN_EMBEDDING_MODEL_NAME", "Alibaba-NLP/gte-Qwen2-1.5B-instruct")
BGE_M3_MODEL_NAME = "BAAI/bge-m3"
BGE_RERANKER_MODEL_NAME = "BAAI/bge-reranker-v2-m3"
ENABLE_ARAT5_REWRITER = os.getenv("ENABLE_ARAT5_REWRITER", "true").lower() in ("1", "true", "yes")
LOCAL_ARAT5_PATH = BASE_DIR / "araT5" / "models" / "arat5-dialect-msa"
DEFAULT_ARAT5_MODEL = str(LOCAL_ARAT5_PATH) if LOCAL_ARAT5_PATH.exists() else "UBC-NLP/AraT5-base"
ARAT5_MODEL_NAME = os.getenv("ARAT5_MODEL_NAME", DEFAULT_ARAT5_MODEL)
USE_FP16 = True

# Device execution controls for hybrid CPU/GPU allocation
# Hybrid Mode (Recommended for 8GB GPU): Reranker on GPU (CUDA), Embedder on CPU, AraT5 on CPU
# All-on-CPU Mode: Set RERANKER_DEVICE="cpu", EMBEDDER_DEVICE="cpu", ARAT5_DEVICE="cpu"
RERANKER_DEVICE = os.getenv("RERANKER_DEVICE", "cuda")
EMBEDDER_DEVICE = os.getenv("EMBEDDER_DEVICE", "cpu")
ARAT5_DEVICE = os.getenv("ARAT5_DEVICE", "cpu")

# Reranker score threshold — chunks below raw logit 0.0 (sigmoid 50%) are strictly discarded
RERANK_THRESHOLD    = 0.0   # raw logit floor (50% match) — anything below is dropped BEFORE boost
RELEVANCE_THRESHOLD = 40    # Minimum similarity percentage (40%) to include chunk in LLM context
RERANK_SCORE_FLOOR  = 50    # minimum % shown (for chunks that pass threshold)
RERANK_SCORE_CEIL   = 97    # maximum % shown (no chunk ever claims 100%)

# Document priority boosts — added to raw reranker logit ONLY AFTER passing threshold (raw_logit >= 0.0)
DOCUMENT_PRIORITY = {
    # ─── Tier 1: Primary official laws (highest authority) ───
    "قانون_نقابة_المهندسين":                0.35,
    "النظام_الداخلي_للنقابة":              0.30,
    "نظام_التقاعد_2023":                   0.30,
    # ─── Tier 2: Core regulations ───
    "نظام_التأمين_الصحي":                  0.25,
    "نظام_ممارسة_مهنة_الهندسة":           0.25,
    "نظام_المكاتب_والشركات":              0.20,
    "نظام_التكافل":                        0.20,
    "نظام_صندوق_التأمين_الاجتماعي":       0.20,
    "نظام_الصندوق_الهندسي_للتدريب":       0.15,
    "نظام_التأهيل_والاعتماد":             0.15,
    # ─── Tier 3: Registration & financial docs ───
    "شروط-تسجيل-الاردنيين":              0.20,
    "شروط_تسجيل":                         0.15,
    "سلم_الرواتب":                         0.20,
    "تعليمات_المنفعة":                     0.15,
    "النشرة_الارشادية":                   0.10,
    "فوائد العضوية":                      0.10,
    # ─── Tier 4: Informational / generated markdown ───
    "عن النقابة":                          0.05,
    "ممارسة المهنة":                       0.05,
    "المهندسين الشباب":                   0.05,
}

# Hierarchical Parent-Child Chunking settings
PARENT_CHUNK_TOKENS = 800       # Large context block for LLM answer generation
PARENT_OVERLAP_TOKENS = 100     # Parent overlap
CHILD_CHUNK_TOKENS = 150        # Small granular chunk for precise vector/BM25 retrieval
CHILD_OVERLAP_TOKENS = 25       # Child overlap
CHUNK_SIZE_TOKENS = CHILD_CHUNK_TOKENS  # Fallback backward-compatibility alias
CHUNK_OVERLAP_TOKENS = CHILD_OVERLAP_TOKENS

# Multi-Stage Ingestion Pipeline & Ollama Settings
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_URL = os.getenv("OLLAMA_URL", f"{OLLAMA_BASE_URL}/api/generate")

EXPOSE_DEBUG_METADATA = os.getenv("EXPOSE_DEBUG_METADATA", "false").lower() in ("1", "true", "yes")
OLLAMA_CHAT_MODEL = os.getenv("OLLAMA_CHAT_MODEL", "ministral-3:8b")
OLLAMA_VISION_MODEL = os.getenv("OLLAMA_VISION_MODEL", "qwen2.5vl:7b")

RENDER_DPI = 300
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID = (8, 8)
MIN_TABLE_AREA_FRACTION = 0.02
TABLE_UPSCALE_FACTOR = 2.0
TABLE_CROP_PADDING = 10
OLLAMA_KEEP_ALIVE = -1 if os.getenv("OLLAMA_KEEP_ALIVE", "-1") == "-1" else os.getenv("OLLAMA_KEEP_ALIVE")

REQUEST_TIMEOUT = 300
NUM_CTX = int(os.getenv("NUM_CTX", "8192"))
NUM_PREDICT = int(os.getenv("NUM_PREDICT", "1280"))

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

# ─── JEA Backend Integration & Ticketing Configuration ───
JEA_BACKEND_URL = os.getenv("JEA_BACKEND_URL", "http://localhost:3000").rstrip("/")
INTERNAL_BYPASS_TOKEN = os.getenv("INTERNAL_BYPASS_TOKEN", "jea_rag_token")
TICKETING_ENABLED = os.getenv("TICKETING_ENABLED", "true").lower() in ("1", "true", "yes")
TICKETING_TIMEOUT_SEC = int(os.getenv("TICKETING_TIMEOUT_SEC", "10"))
