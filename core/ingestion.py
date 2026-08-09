import os
import gc
import re
import sys
import time
import base64
import hashlib
import sqlite3
import unicodedata
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import cv2
import fitz  # PyMuPDF
import numpy as np
import pdfplumber
import requests
import chromadb
from chromadb.config import Settings as ChromaSettings

try:
    from bidi.algorithm import get_display
    HAS_BIDI = True
except ImportError:
    HAS_BIDI = False

try:
    from FlagEmbedding import BGEM3FlagModel
    HAS_BGEM3 = True
except ImportError:
    HAS_BGEM3 = False

from core.config import (
    BASE_DIR, STORAGE_DIR, CHROMA_PERSIST_DIR, REGISTRY_DB_PATH,
    CHROMA_COLLECTION_NAME, BGE_M3_MODEL_NAME, USE_FP16, CHUNK_SIZE_TOKENS,
    CHUNK_OVERLAP_TOKENS, OLLAMA_URL, OLLAMA_VISION_MODEL, PARSED_OUTPUT_DIR,
    RENDER_DPI, CLAHE_CLIP_LIMIT, CLAHE_TILE_GRID, MIN_TABLE_AREA_FRACTION,
    TABLE_UPSCALE_FACTOR, TABLE_CROP_PADDING, REQUEST_TIMEOUT, NUM_CTX, NUM_PREDICT,
    FILES_DIR, TEXTS_DIR, PDFS_DIR, IGNORE_TEXTS_DIR, EXCLUDE_DIRS, UNIFIED_VISION_PROMPT
)


# ---------------------------------------------------------------------------
# 1. TEXT CLEANING & ARABIC BIDI NORMALIZATION LAYER
# ---------------------------------------------------------------------------
def fix_reversed_arabic_text(text: str) -> str:
    """
    Detects and fixes reversed Arabic in PDF text extraction:
    1. Arabic Presentation Forms (\uFE70-\uFEFE, \uFB50-\uFDFF): Un-shapes & reverses Bidi order.
    2. Standard Character LTR Reversed Arabic (e.g. 'ليومتلا ةدم بسح'): Reverses line character stream & swaps brackets.
    """
    if not text:
        return text

    lines = text.splitlines()
    fixed_lines = []

    for line in lines:
        if not line.strip():
            fixed_lines.append(line)
            continue

        # Case 1: Arabic Presentation Forms
        if re.search(r'[\uFE70-\uFEFE\uFB50-\uFDFF]', line):
            if HAS_BIDI:
                try:
                    bidi_fixed = get_display(line)
                    norm_line = unicodedata.normalize('NFKC', bidi_fixed)
                    fixed_lines.append(norm_line)
                    continue
                except Exception:
                    pass

        # Case 2: Standard Arabic letters in reversed LTR order
        if re.search(r'[\u0600-\u06FF]', line):
            if re.search(r'\b[اأإآ]ل[ء-ي]+\b', line) is None and re.search(r'\b[ء-ي]+لا\b', line):
                try:
                    rev = line[::-1]
                    rev = rev.replace('(', 'TEMP_L').replace(')', '(').replace('TEMP_L', ')')
                    rev = rev.replace('[', 'TEMP_BL').replace(']', '[').replace('TEMP_BL', ']')
                    fixed_lines.append(rev)
                    continue
                except Exception:
                    pass

        fixed_lines.append(line)

    return "\n".join(fixed_lines)


def clean_document_text(text: str) -> str:
    """
    Cleans raw text: fixes reversed Arabic, strips PDF font CID artifacts,
    normalizes whitespace and line breaks.
    """
    if not text:
        return ""

    text = fix_reversed_arabic_text(text)
    cleaned = re.sub(r'\(?cid:\d+\)?', '', text)
    cleaned = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', cleaned)
    lines = [line.rstrip() for line in cleaned.splitlines()]
    cleaned = "\n".join(lines)
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
    return cleaned.strip()

# ---------------------------------------------------------------------------
# 2. OPENCV UNICODE HELPERS & IMAGE PREPROCESSING (GRAYSCALE + CLAHE)
# ---------------------------------------------------------------------------
def cv2_imread_unicode(path: Path, flags: int = cv2.IMREAD_COLOR) -> np.ndarray:
    """Reads an image from a path containing Unicode/non-ASCII characters."""
    img_array = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(img_array, flags)
    if img is None:
        raise ValueError(f"Failed to decode image from path: {path}")
    return img


def cv2_imwrite_unicode(path: Path, img: np.ndarray) -> bool:
    """Writes an image to a path containing Unicode/non-ASCII characters."""
    ext = path.suffix if path.suffix else ".png"
    success, buf = cv2.imencode(ext, img)
    if not success:
        return False
    buf.tofile(str(path))
    return True


def preprocess_image_cv2(img_path: Path, out_dir: Path) -> Path:
    """Applies Grayscale conversion & CLAHE adaptive contrast enhancement."""
    img = cv2_imread_unicode(img_path, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    del img

    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILE_GRID)
    enhanced = clahe.apply(gray)
    del gray, clahe

    out_path = out_dir / img_path.name
    cv2_imwrite_unicode(out_path, enhanced)
    del enhanced
    return out_path

# ---------------------------------------------------------------------------
# 3. OPENCV LINE MORPHOLOGY TABLE DETECTION & CROP MASKING
# ---------------------------------------------------------------------------
def detect_table_boxes(gray_img: np.ndarray) -> List[Tuple[int, int, int, int]]:
    """Detect bordered tables in a grayscale image using line morphology."""
    h, w = gray_img.shape

    bin_img = cv2.adaptiveThreshold(
        gray_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 10
    )

    horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(w // 30, 20), 1))
    horiz_lines = cv2.erode(bin_img, horiz_kernel, iterations=1)
    horiz_lines = cv2.dilate(horiz_lines, horiz_kernel, iterations=1)
    del horiz_kernel

    vert_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(h // 30, 20)))
    vert_lines = cv2.erode(bin_img, vert_kernel, iterations=1)
    vert_lines = cv2.dilate(vert_lines, vert_kernel, iterations=1)
    del vert_kernel, bin_img

    grid = cv2.add(horiz_lines, vert_lines)
    del horiz_lines, vert_lines

    grid = cv2.dilate(grid, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)), iterations=2)
    contours, _ = cv2.findContours(grid, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    del grid

    page_area = h * w
    boxes = []
    for cnt in contours:
        x, y, cw, ch = cv2.boundingRect(cnt)
        area_fraction = (cw * ch) / page_area
        if area_fraction < MIN_TABLE_AREA_FRACTION:
            continue
        boxes.append((x, y, cw, ch))

    del contours
    boxes = merge_close_boxes(boxes)
    boxes.sort(key=lambda b: b[1])
    return boxes


def merge_close_boxes(boxes: List[Tuple[int, int, int, int]], gap_thresh: int = 25) -> List[Tuple[int, int, int, int]]:
    if not boxes:
        return []

    def rects_close(a, b, gap):
        ax, ay, aw, ah = a
        bx, by, bw, bh = b
        a_exp = (ax - gap, ay - gap, ax + aw + gap, ay + ah + gap)
        b_rect = (bx, by, bx + bw, by + bh)
        return not (
            a_exp[2] < b_rect[0] or b_rect[2] < a_exp[0] or
            a_exp[3] < b_rect[1] or b_rect[3] < a_exp[1]
        )

    merged = list(boxes)
    changed = True
    while changed:
        changed = False
        result = []
        used = [False] * len(merged)
        for i in range(len(merged)):
            if used[i]:
                continue
            cur = merged[i]
            for j in range(i + 1, len(merged)):
                if used[j]:
                    continue
                if rects_close(cur, merged[j], gap_thresh):
                    x1 = min(cur[0], merged[j][0])
                    y1 = min(cur[1], merged[j][1])
                    x2 = max(cur[0] + cur[2], merged[j][0] + merged[j][2])
                    y2 = max(cur[1] + cur[3], merged[j][1] + merged[j][3])
                    cur = (x1, y1, x2 - x1, y2 - y1)
                    used[j] = True
                    changed = True
            result.append(cur)
        merged = result
    return merged


def mask_tables_on_image(gray_img: np.ndarray, boxes: List[Tuple[int, int, int, int]]) -> np.ndarray:
    masked = gray_img.copy()
    for idx, (x, y, w, h) in enumerate(boxes, start=1):
        cv2.rectangle(masked, (x, y), (x + w, y + h), color=180, thickness=-1)
        cv2.rectangle(masked, (x, y), (x + w, y + h), color=80, thickness=2)

        token = f"[[TABLE_{idx}]]"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = max(min(w, h) / 300, 0.6)
        thickness = 2
        text_size, _ = cv2.getTextSize(token, font, font_scale, thickness)
        text_x = x + max((w - text_size[0]) // 2, 5)
        text_y = y + max((h + text_size[1]) // 2, 20)
        cv2.putText(masked, token, (text_x, text_y), font, font_scale, 20, thickness, cv2.LINE_AA)
    return masked


def crop_table(gray_img: np.ndarray, box: Tuple[int, int, int, int], padding: int = TABLE_CROP_PADDING) -> np.ndarray:
    h, w = gray_img.shape
    x, y, bw, bh = box
    x0 = max(x - padding, 0)
    y0 = max(y - padding, 0)
    x1 = min(x + bw + padding, w)
    y1 = min(y + bh + padding, h)
    crop = gray_img[y0:y1, x0:x1]

    if TABLE_UPSCALE_FACTOR != 1.0:
        crop = cv2.resize(
            crop, None, fx=TABLE_UPSCALE_FACTOR, fy=TABLE_UPSCALE_FACTOR,
            interpolation=cv2.INTER_CUBIC,
        )
    return crop


def array_to_base64_png(img: np.ndarray) -> str:
    success, buf = cv2.imencode(".png", img)
    if not success:
        raise RuntimeError("Failed to encode image to PNG")
    b64_str = base64.b64encode(buf.tobytes()).decode("utf-8")
    del buf
    return b64_str

# ---------------------------------------------------------------------------
# 4. VISION VLM CALL ENGINE & UNIFIED SINGLE-PASS PARSER
# ---------------------------------------------------------------------------
def call_vlm_vision_api(prompt: str, img_b64: str) -> str:
    """Calls Ollama vision endpoint for PDF parsing."""
    # Default to Ollama (qwen2.5vl:7b)
    payload = {
        "model": OLLAMA_VISION_MODEL,
        "prompt": prompt,
        "images": [img_b64],
        "stream": False,
        "options": {
            "temperature": 0.1,
            "num_predict": NUM_PREDICT,
            "num_ctx": NUM_CTX,
        },
    }
    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        return data.get("response", "").strip()
    except Exception as e:
        print(f"[VLM Ollama Error]: {e}")
        return f"[ERROR calling Vision VLM: {e}]"


def process_vision_page(gray_img: np.ndarray, page_num: int, debug_dir: Path) -> str:
    """Single-pass Vision page parser: converts preprocessed image to Base64 and transcribes via Qwen VLM."""
    page_b64 = array_to_base64_png(gray_img)
    parsed_text = call_vlm_vision_api(UNIFIED_VISION_PROMPT, page_b64)
    del page_b64
    return parsed_text



def parse_and_save_pdf_vision(pdf_path: Path) -> Tuple[str, Path, int]:
    """
    Processes PDF page by page using full Vision PDF Parser (OpenCV table detection + Qwen2.5-VL),
    saves all output artifacts under PARSED_OUTPUT_DIR / pdf_stem /, and returns (full_text, parsed_md_path, total_pages).
    """
    pdf_path = Path(pdf_path).resolve()
    pdf_out_dir = PARSED_OUTPUT_DIR / pdf_path.stem

    raw_img_dir = pdf_out_dir / "raw_pages"
    proc_img_dir = pdf_out_dir / "processed_pages"
    debug_dir = pdf_out_dir / "debug_detected_tables"
    text_dir = pdf_out_dir / "page_text"
    for d in (raw_img_dir, proc_img_dir, debug_dir, text_dir):
        d.mkdir(parents=True, exist_ok=True)

    doc = fitz.open(pdf_path)
    total_pages = len(doc)
    print(f"\n[Vision Pipeline] Processing PDF: '{pdf_path.name}' ({total_pages} pages)")
    print(f"[Vision Pipeline] Saving output artifacts to: '{pdf_out_dir}'")

    zoom = RENDER_DPI / 72
    mat = fitz.Matrix(zoom, zoom)

    combined_md = []
    for i in range(total_pages):
        page_num = i + 1
        page = doc[i]

        pix = page.get_pixmap(matrix=mat)
        raw_img_path = raw_img_dir / f"page_{page_num:03d}.png"
        pix.save(str(raw_img_path))
        del pix, page

        proc_img_path = preprocess_image_cv2(raw_img_path, proc_img_dir)
        gray_img = cv2_imread_unicode(proc_img_path, cv2.IMREAD_GRAYSCALE)
        page_text = process_vision_page(gray_img, page_num, debug_dir)
        del gray_img

        page_out_path = text_dir / f"page_{page_num:03d}.md"
        page_out_path.write_text(page_text, encoding="utf-8")

        combined_md.append(f"\n\n<!-- ===== Page {page_num} ===== -->\n\n{page_text}")
        del page_text

        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        gc.collect()

    doc.close()
    del doc

    parsed_md_path = pdf_out_dir / f"{pdf_path.stem}_parsed.md"
    full_text = "".join(combined_md)
    parsed_md_path.write_text(full_text, encoding="utf-8")
    del combined_md

    print(f"[Vision Pipeline] Completed '{pdf_path.name}'. Parsed output saved: {parsed_md_path}")
    return full_text, parsed_md_path, total_pages

# ---------------------------------------------------------------------------
# 5. TOKEN CHUNKER LAYER
# ---------------------------------------------------------------------------
class TextChunker:
    def __init__(self, chunk_size_tokens: int = CHUNK_SIZE_TOKENS, chunk_overlap_tokens: int = CHUNK_OVERLAP_TOKENS):
        self.chunk_size = chunk_size_tokens
        self.chunk_overlap = chunk_overlap_tokens

    def chunk_text(self, document_dict: Dict[str, Any]) -> List[Dict[str, Any]]:
        text = document_dict.get("full_text", "")
        if not text or not text.strip():
            return []

        source_id = document_dict.get("source_id", "doc_unknown")
        file_name = document_dict.get("file_name", "unknown")
        source_type = document_dict.get("source_type", "pdf")
        paragraphs = text.split("\n\n")

        chunks = []
        current_words = []
        current_len = 0
        chunk_idx = 1

        for para in paragraphs:
            para_words = para.strip().split()
            if not para_words:
                continue

            para_len = len(para_words)
            if current_len + para_len > self.chunk_size and current_words:
                chunk_str = " ".join(current_words)
                chunks.append({
                    "chunk_id": f"{source_id}_chunk_{chunk_idx}",
                    "source_id": source_id,
                    "file_name": file_name,
                    "source_type": source_type,
                    "chunk_index": chunk_idx,
                    "text": chunk_str,
                    "word_count": len(current_words)
                })
                chunk_idx += 1
                overlap_words = current_words[-self.chunk_overlap:] if len(current_words) > self.chunk_overlap else []
                current_words = overlap_words + para_words
                current_len = len(current_words)
            else:
                current_words.extend(para_words)
                current_len += para_len

        if current_words:
            chunk_str = " ".join(current_words)
            chunks.append({
                "chunk_id": f"{source_id}_chunk_{chunk_idx}",
                "source_id": source_id,
                "file_name": file_name,
                "source_type": source_type,
                "chunk_index": chunk_idx,
                "text": chunk_str,
                "word_count": len(current_words)
            })

        return chunks

# ---------------------------------------------------------------------------
# 6. BGE-M3 EMBEDDER LAYER
# ---------------------------------------------------------------------------
class BGEM3Embedder:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(BGEM3Embedder, cls).__new__(cls)
            cls._instance.model = None
            cls._instance._load_model()
        return cls._instance

    def _load_model(self):
        if HAS_BGEM3:
            try:
                print(f"[Embedder] Loading BGE-M3 model '{BGE_M3_MODEL_NAME}'...")
                self.model = BGEM3FlagModel(BGE_M3_MODEL_NAME, use_fp16=USE_FP16)
                print("[Embedder] Successfully loaded BGE-M3 model.")
            except Exception as e:
                print(f"[Embedder Warning] Could not load BGE-M3 model: {e}")
                self.model = None

    def embed_texts(self, texts: List[str]) -> Tuple[List[List[float]], List[Dict[str, float]]]:
        if not texts:
            return [], []
        if self.model:
            out = self.model.encode(texts, return_dense=True, return_sparse=True, return_colbert_vecs=False)
            dense_vecs = out['dense_vecs'].tolist()
            sparse_vecs = out['lexical_weights']
            return dense_vecs, sparse_vecs
        else:
            return [[0.0] * 1024 for _ in texts], [{}] * len(texts)

# ---------------------------------------------------------------------------
# 7. CHROMA VECTOR STORE INDEXER LAYER
# ---------------------------------------------------------------------------
class ChromaIndexer:
    def __init__(self):
        self.client = chromadb.PersistentClient(path=str(CHROMA_PERSIST_DIR), settings=ChromaSettings(anonymized_telemetry=False))
        self.collection = self.client.get_or_create_collection(
            name=CHROMA_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine", "description": "Guild Knowledge Base BGE-M3 Embeddings"}
        )
        self.embedder = BGEM3Embedder()

    def index_chunks(self, chunks: List[Dict[str, Any]]):
        if not chunks:
            return 0

        ids = [c["chunk_id"] for c in chunks]
        documents = [c["text"] for c in chunks]
        metadatas = [
            {
                "source_id": c["source_id"],
                "file_name": c["file_name"],
                "source_type": c["source_type"],
                "chunk_index": c["chunk_index"],
                "word_count": c["word_count"]
            }
            for c in chunks
        ]

        dense_embeddings, _ = self.embedder.embed_texts(documents)
        self.collection.upsert(ids=ids, embeddings=dense_embeddings, documents=documents, metadatas=metadatas)
        return len(chunks)

    def delete_document_chunks(self, source_id: str):
        try:
            self.collection.delete(where={"source_id": source_id})
        except Exception:
            pass

# ---------------------------------------------------------------------------
# 8. CONSOLIDATED INGESTION PIPELINE
# ---------------------------------------------------------------------------
class IngestionPipeline:
    def __init__(self):
        self.chunker = TextChunker()
        self.indexer = ChromaIndexer()

    def process_file(self, file_path: Path) -> Dict[str, Any]:
        file_path = Path(file_path).resolve()
        ext = file_path.suffix.lower()

        if ext in [".txt", ".md"]:
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                raw_text = f.read()
            clean_text = clean_document_text(raw_text)
            doc_dict = {
                "source_id": f"text_{file_path.stem}",
                "file_name": file_path.name,
                "source_type": "text",
                "full_text": clean_text,
                "total_pages": 1
            }
        elif ext == ".pdf":
            raw_text, parsed_md_path, total_pages = parse_and_save_pdf_vision(file_path)
            clean_text = clean_document_text(raw_text)

            sha256 = hashlib.sha256()
            with open(file_path, 'rb') as f:
                while chunk := f.read(8192):
                    sha256.update(chunk)

            doc_dict = {
                "source_id": f"pdf_{file_path.stem}",
                "file_name": file_path.name,
                "source_type": "pdf",
                "content_hash": sha256.hexdigest(),
                "total_pages": total_pages,
                "full_text": clean_text,
                "parsed_md_path": str(parsed_md_path)
            }
        else:
            raise ValueError(f"Unsupported file extension: {ext}")

        chunks = self.chunker.chunk_text(doc_dict)
        indexed_cnt = self.indexer.index_chunks(chunks)

        self.flush_memory()

        return {"document": doc_dict, "chunks_count": len(chunks), "indexed_count": indexed_cnt}

    def flush_memory(self):
        """Flushes Python garbage collector and GPU VRAM cache after every file processing step."""
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        gc.collect()
