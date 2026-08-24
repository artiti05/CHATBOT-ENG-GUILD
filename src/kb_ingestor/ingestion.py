import os
import gc
import re
import sys
import time
import base64
import hashlib
import sqlite3
import unicodedata
import shutil
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

from src.config import (
    BASE_DIR, STORAGE_DIR, CHROMA_PERSIST_DIR, REGISTRY_DB_PATH,
    CHROMA_COLLECTION_NAME, BGE_M3_MODEL_NAME, USE_FP16, PARENT_CHUNK_TOKENS,
    PARENT_OVERLAP_TOKENS, CHILD_CHUNK_TOKENS, CHILD_OVERLAP_TOKENS,
    OLLAMA_URL, OLLAMA_VISION_MODEL, PARSED_OUTPUT_DIR, MARKDOWNS_DIR,
    RENDER_DPI, CLAHE_CLIP_LIMIT, CLAHE_TILE_GRID, MIN_TABLE_AREA_FRACTION,
    TABLE_UPSCALE_FACTOR, TABLE_CROP_PADDING, REQUEST_TIMEOUT, NUM_CTX, NUM_PREDICT,
    FILES_DIR, TEXTS_DIR, PDFS_DIR, IGNORE_TEXTS_DIR, EXCLUDE_DIRS, UNIFIED_VISION_PROMPT,
    TABLE_VISION_PROMPT, ENABLE_TABLE_ROUTING, SAVE_TABLE_DEBUG_IMAGES, MAX_TABLES_PER_PAGE,
    MIN_VRAM_GB_EMBEDDER
)
from src.pipeline.stage_02_retrieve.bm25_search import BM25Indexer
from src.core.gpu_utils import pick_device


def fix_reversed_arabic_text(text: str) -> str:
    if not text:
        return text

    lines = text.splitlines()
    fixed_lines = []

    for line in lines:
        if not line.strip():
            fixed_lines.append(line)
            continue

        if re.search(r'[\uFE70-\uFEFE\uFB50-\uFDFF]', line):
            if HAS_BIDI:
                try:
                    bidi_fixed = get_display(line)
                    norm_line = unicodedata.normalize('NFKC', bidi_fixed)
                    fixed_lines.append(norm_line)
                    continue
                except Exception:
                    pass

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
    if not text:
        return ""

    text = fix_reversed_arabic_text(text)
    cleaned = re.sub(r'\(?cid:\d+\)?', '', text)
    cleaned = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', cleaned)
    lines = [line.rstrip() for line in cleaned.splitlines()]
    cleaned = "\n".join(lines)
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
    return cleaned.strip()


def cv2_imread_unicode(path: Path, flags: int = cv2.IMREAD_COLOR) -> np.ndarray:
    img_array = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(img_array, flags)
    if img is None:
        raise ValueError(f"Failed to decode image from path: {path}")
    return img


def cv2_imwrite_unicode(path: Path, img: np.ndarray) -> bool:
    ext = path.suffix if path.suffix else ".png"
    success, buf = cv2.imencode(ext, img)
    if not success:
        return False
    buf.tofile(str(path))
    return True


def preprocess_image_cv2(img_path: Path, out_dir: Path) -> Path:
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


def detect_table_boxes(gray_img: np.ndarray) -> List[Tuple[int, int, int, int]]:
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


def call_vlm_vision_api(prompt: str, img_b64: str) -> str:
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
        from src.pipeline.stage_04_answer.generator_agent import strip_reasoning
        return strip_reasoning(data.get("response", "").strip())
    except Exception as e:
        print(f"[VLM Ollama Error]: {e}")
        return f"[ERROR calling Vision VLM: {e}]"


def _single_pass_page(gray_img: np.ndarray) -> str:
    """Whole-page single-pass VLM transcription (legacy/fallback path)."""
    page_b64 = array_to_base64_png(gray_img)
    parsed_text = call_vlm_vision_api(UNIFIED_VISION_PROMPT, page_b64)
    del page_b64
    return parsed_text


def _save_table_debug_image(gray_img: np.ndarray, boxes: List[Tuple[int, int, int, int]],
                            page_num: int, debug_dir: Path) -> None:
    """Saves an annotated copy of the page with detected table boxes drawn and labelled,
    so detection quality (misses vs. false positives) can be checked visually."""
    try:
        annotated = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)
        for idx, (x, y, w, h) in enumerate(boxes, start=1):
            cv2.rectangle(annotated, (x, y), (x + w, y + h), color=(0, 0, 255), thickness=3)
            cv2.putText(annotated, f"TABLE_{idx}", (x + 5, max(y - 10, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA)
        debug_dir.mkdir(parents=True, exist_ok=True)
        out_path = debug_dir / f"page_{page_num:03d}_tables.png"
        cv2_imwrite_unicode(out_path, annotated)
        del annotated
    except Exception as e:
        print(f"[Vision Warning] Failed to save table debug image for page {page_num}: {e}")


def _stitch_tables(prose_md: str, table_results: List[str], page_num: int) -> Tuple[str, int, int]:
    """Substitutes each [[TABLE_i]] token in prose_md with its extracted table HTML.

    Never silently drops a table:
      - token found, extraction OK      -> token replaced with the table HTML.
      - token found, extraction failed  -> token replaced with a greppable
                                           [[TABLE_i: EXTRACTION_FAILED]] marker.
      - token missing from the prose    -> the content (table HTML or failure marker) is
                                           appended at the end under a
                                           "<!-- TABLE_i (position unresolved) -->" comment.
      - token appears more than once    -> only the first occurrence is replaced; logged.

    Returns (stitched_text, tables_extracted, tokens_matched).
    """
    stitched = prose_md
    tables_extracted = 0
    tokens_matched = 0
    unresolved: List[str] = []

    for idx, table_md in enumerate(table_results, start=1):
        table_html = table_md.strip() if table_md and table_md.strip() else ""
        if table_html:
            tables_extracted += 1
            content = table_html
        else:
            content = f"[[TABLE_{idx}: EXTRACTION_FAILED]]"

        token_pattern = re.compile(r'\[\[\s*TABLE[_\s]*' + str(idx) + r'\s*\]\]')
        occurrences = list(token_pattern.finditer(stitched))

        if not occurrences:
            unresolved.append(f"<!-- TABLE_{idx} (position unresolved) -->\n{content}")
            continue

        if len(occurrences) > 1:
            print(f"[Vision Warning] Page {page_num}: token TABLE_{idx} appears "
                  f"{len(occurrences)} times in prose output; replacing first occurrence only.")

        first = occurrences[0]
        stitched = stitched[:first.start()] + content + stitched[first.end():]
        tokens_matched += 1

    if unresolved:
        stitched = stitched + "\n\n" + "\n\n".join(unresolved)

    return stitched, tables_extracted, tokens_matched


def process_vision_page(gray_img: np.ndarray, page_num: int, debug_dir: Path) -> str:
    """Two-pass table-aware Vision page parser.

    Detects bordered tables with OpenCV, transcribes the page prose with those regions
    masked out (Pass A), transcribes each table separately from a cropped/upscaled image
    (Pass B), then stitches the results back together via the [[TABLE_i]] tokens stamped
    by mask_tables_on_image. Falls back to a single whole-page VLM call when table routing
    is disabled, no tables are detected, or detection finds more than MAX_TABLES_PER_PAGE
    (likely a false-positive explosion).

    Known limitation: detect_table_boxes only finds *bordered* tables (line morphology).
    Borderless/whitespace-aligned tables, diagrams, and rule-only tables fall through to
    the single-pass path below -- this is accepted, not a bug to fix here.
    """
    if not ENABLE_TABLE_ROUTING:
        return _single_pass_page(gray_img)

    boxes = detect_table_boxes(gray_img)

    if SAVE_TABLE_DEBUG_IMAGES and boxes:
        _save_table_debug_image(gray_img, boxes, page_num, debug_dir)

    if not boxes:
        return _single_pass_page(gray_img)

    if len(boxes) > MAX_TABLES_PER_PAGE:
        print(f"[Vision Warning] Page {page_num}: {len(boxes)} table(s) detected, "
              f"exceeds MAX_TABLES_PER_PAGE={MAX_TABLES_PER_PAGE}; falling back to single-pass.")
        return _single_pass_page(gray_img)

    # Pass A: transcribe the page prose with table regions blanked out and stamped
    # with [[TABLE_i]] tokens, so layout and reading order survive around the tables.
    masked = mask_tables_on_image(gray_img, boxes)
    masked_b64 = array_to_base64_png(masked)
    del masked
    prose_md = call_vlm_vision_api(UNIFIED_VISION_PROMPT, masked_b64)
    del masked_b64

    # Pass B: transcribe each table separately from its own cropped/upscaled image.
    # Process and release one crop at a time -- never hold all of them in memory.
    table_results: List[str] = []
    for box in boxes:
        crop = crop_table(gray_img, box)
        crop_b64 = array_to_base64_png(crop)
        del crop
        table_md = call_vlm_vision_api(TABLE_VISION_PROMPT, crop_b64)
        del crop_b64
        table_results.append(table_md)

    stitched, extracted, matched = _stitch_tables(prose_md, table_results, page_num)

    print(f"[Vision] Page {page_num}: {len(boxes)} table(s) detected, "
          f"{extracted} extracted, {matched} token(s) matched, two-pass parse.")
    return stitched


def parse_and_save_pdf_vision(pdf_path: Path) -> Tuple[str, Path, int]:
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

    try:
        MARKDOWNS_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copy2(parsed_md_path, MARKDOWNS_DIR / parsed_md_path.name)
    except Exception as e:
        print(f"[Vision Pipeline Warning] Failed to copy parsed markdown to markdowns/: {e}")

    try:
        target_pdf_copy = pdf_out_dir / pdf_path.name
        if not target_pdf_copy.exists():
            shutil.copy2(pdf_path, target_pdf_copy)
    except Exception as e:
        print(f"[Vision Pipeline Warning] Failed to copy source PDF to output_dir: {e}")

    print(f"[Vision Pipeline] Completed '{pdf_path.name}'. Parsed output saved: {parsed_md_path}")
    return full_text, parsed_md_path, total_pages


class TextChunker:
    def __init__(
        self,
        parent_tokens: int = PARENT_CHUNK_TOKENS,
        parent_overlap_tokens: int = PARENT_OVERLAP_TOKENS,
        child_tokens: int = CHILD_CHUNK_TOKENS,
        child_overlap_tokens: int = CHILD_OVERLAP_TOKENS
    ):
        self.words_per_parent = int(parent_tokens / 1.5)
        self.words_parent_overlap = int(parent_overlap_tokens / 1.5)
        self.words_per_child = int(child_tokens / 1.5)
        self.words_child_overlap = int(child_overlap_tokens / 1.5)

    def _split_into_word_chunks(self, words: List[str], max_words: int, overlap_words: int) -> List[List[str]]:
        if not words:
            return []
        chunks = []
        start = 0
        while start < len(words):
            end = start + max_words
            chunk = words[start:end]
            if chunk:
                chunks.append(chunk)
            start += max_words - overlap_words
            if max_words <= overlap_words:
                break
        return chunks

    def chunk_text(self, document_dict: Dict[str, Any]) -> List[Dict[str, Any]]:
        text = document_dict.get("full_text", "")
        if not text or not text.strip():
            return []

        source_id = document_dict.get("source_id", "doc_unknown")
        file_name = document_dict.get("file_name", "unknown")
        source_type = document_dict.get("source_type", "pdf")

        words = text.split()
        if not words:
            return []

        parent_word_blocks = self._split_into_word_chunks(words, self.words_per_parent, self.words_parent_overlap)
        
        child_chunks = []
        global_child_idx = 1

        for p_idx, p_words in enumerate(parent_word_blocks, start=1):
            parent_id = f"{source_id}_parent_{p_idx}"
            parent_str = " ".join(p_words)

            child_word_blocks = self._split_into_word_chunks(p_words, self.words_per_child, self.words_child_overlap)
            
            for c_idx, c_words in enumerate(child_word_blocks, start=1):
                child_str = " ".join(c_words)
                child_chunks.append({
                    "chunk_id": f"{source_id}_c{global_child_idx}",
                    "source_id": source_id,
                    "file_name": file_name,
                    "source_type": source_type,
                    "chunk_index": global_child_idx,
                    "text": child_str,
                    "parent_id": parent_id,
                    "parent_text": parent_str,
                    "word_count": len(c_words),
                    "estimated_tokens": int(len(c_words) * 1.5)
                })
                global_child_idx += 1

        return child_chunks


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
                device = pick_device(MIN_VRAM_GB_EMBEDDER)
                print(f"[Embedder] Loading BGE-M3 model '{BGE_M3_MODEL_NAME}' on {device}...")
                use_fp16 = USE_FP16 and device != "cpu"
                self.model = BGEM3FlagModel(BGE_M3_MODEL_NAME, use_fp16=use_fp16, devices=device)
                print(f"[Embedder] Device: {device}, fp16: {use_fp16}")
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

    def embed_single_text(self, text: str) -> List[float]:
        if not text or not text.strip():
            return [0.0] * 1024
        dense_vecs, _ = self.embed_texts([text])
        return dense_vecs[0] if dense_vecs else [0.0] * 1024


class ChromaIndexer:
    def __init__(self):
        self.client = chromadb.PersistentClient(path=str(CHROMA_PERSIST_DIR), settings=ChromaSettings(anonymized_telemetry=False))
        self.collection = self.client.get_or_create_collection(
            name=CHROMA_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine", "description": "Guild Knowledge Base BGE-M3 Embeddings"}
        )
        self.embedder = BGEM3Embedder()
        self.bm25_indexer = BM25Indexer()

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
                "word_count": c["word_count"],
                "parent_id": c["parent_id"],
                "parent_text": c["parent_text"]
            }
            for c in chunks
        ]

        dense_embeddings, _ = self.embedder.embed_texts(documents)
        self.collection.upsert(ids=ids, embeddings=dense_embeddings, documents=documents, metadatas=metadatas)
        self.bm25_indexer.add_chunks(chunks)
        return len(chunks)

    def delete_document_chunks(self, source_id: str):
        try:
            self.collection.delete(where={"source_id": source_id})
        except Exception:
            pass


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
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        gc.collect()
