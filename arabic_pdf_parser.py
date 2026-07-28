"""
Arabic PDF -> Images -> Grayscale + CLAHE -> Qwen2.5-VL (via Ollama) parser
============================================================================
v2: table detection + isolation

Instead of asking the model to parse prose AND tables in one pass (which
breaks on Arabic tables with merged/nested RTL headers), this version:

  1. Renders each page to a high-DPI image (PyMuPDF)
  2. Grayscale + CLAHE (adaptive contrast, preserves faint borders/shading)
  3. Detects table regions using OpenCV line-morphology (finds bordered
     tables by their horizontal + vertical grid lines)
  4. Masks each detected table out of the page image, stamping a literal
     placeholder token (e.g. [[TABLE_1]]) in its place
  5. Sends the MASKED page to Qwen asking for prose only, preserving the
     placeholder tokens verbatim wherever a masked region appears
  6. Crops + upscales each detected table region from the UNMASKED page,
     and sends each one to Qwen as its OWN focused request, with a prompt
     specialized for RTL merged-header tables
  7. Splices the table HTML back into the prose output, replacing each
     [[TABLE_N]] token with the corresponding parsed table

Requirements (install once):
    pip install pymupdf opencv-python-headless requests tqdm numpy --break-system-packages

Requires Ollama running locally with the vision model pulled:
    ollama pull qwen2.5vl:7b
    ollama serve   (if not already running as a service)

Usage:
    python arabic_pdf_parser.py /path/to/input.pdf /path/to/output_dir
"""

import base64
import re
import sys
from pathlib import Path

import cv2
import fitz  # PyMuPDF
import numpy as np
import requests
from tqdm import tqdm

# ----------------------------- Config ---------------------------------

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "qwen2.5vl:7b"
RENDER_DPI = 300
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID = (8, 8)
REQUEST_TIMEOUT = 300
NUM_CTX = 8192
NUM_PREDICT = 4096

# Table detection tuning -- adjust if tables are missed or false positives appear
MIN_TABLE_AREA_FRACTION = 0.02   # a detected box smaller than this % of the page is ignored
TABLE_UPSCALE_FACTOR = 2.0       # how much to enlarge a cropped table before sending to Qwen
TABLE_CROP_PADDING = 10          # pixels of padding around a detected table box

PROSE_PROMPT_TEMPLATE = """You are an expert document parser. This is a page from an Arabic
document that may contain Arabic text, English text, and tables.

Some regions of this page have been intentionally blanked out (solid gray boxes) because
they contain tables that are being processed separately. Each blanked box has a placeholder
token printed inside it in square brackets, e.g. [[TABLE_1]].

Instructions:
1. Transcribe all text exactly as it appears, preserving reading order (right-to-left for Arabic).
2. When you reach a blanked/gray box, output ONLY the placeholder token exactly as printed
   inside it (e.g. [[TABLE_1]]) -- do not describe the box, do not say "table here", just the
   literal token, on its own line.
3. Do not translate anything. Keep Arabic text in Arabic and English text in English.
4. Do not add commentary, explanations, or notes of your own. Output ONLY the transcribed
   content and placeholder tokens, nothing else.
5. If a region of the page is unreadable, mark it as [UNREADABLE] rather than guessing.
"""

TABLE_PROMPT = """You are an expert table parser. This image is a CROP of a single table from
an Arabic document. Output ONLY a clean HTML <table> for it -- nothing else, no commentary.

Rules:
- Use <table>, <tr>, <th>/<td>, and colspan/rowspan attributes for merged cells.
- Arabic tables are read right-to-left. Keep column order exactly as it visually appears on
  the page, left side of the image = last column, right side of the image = first column.
- Do not translate anything. Keep Arabic text in Arabic and English/numbers as-is.
- If the table has a merged/spanning header cell above several sub-columns, use colspan on
  that header cell and list the sub-column labels in a second header row, in the same
  right-to-left order they appear visually.

Example of the correct pattern for a merged-header RTL table:

Visual layout on the page (right to left):
    | سقف التغطية (دينار أردني)          | البيان |
    | الدرجة الثانية | الدرجة الأولى | الدرجة الخاصة |        |

Correct HTML output:
<table>
  <tr>
    <th colspan="3">سقف التغطية (دينار أردني)</th>
    <th>البيان</th>
  </tr>
  <tr>
    <th>الدرجة الثانية</th>
    <th>الدرجة الأولى</th>
    <th>الدرجة الخاصة</th>
    <th></th>
  </tr>
  <tr>
    <td>12000</td>
    <td>15000</td>
    <td>15000</td>
    <td>سقف التغطية للفرد الواحد في السنة</td>
  </tr>
</table>

Apply this same pattern to ANY merged/spanning header you see, regardless of the actual
labels or values. If the table has no merged headers, just output a normal HTML table with
one header row.
"""


# ------------------------- Step 1: PDF -> images -------------------------

def pdf_to_images(pdf_path: Path, out_dir: Path, dpi: int = RENDER_DPI) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    doc = fitz.open(pdf_path)
    zoom = dpi / 72
    mat = fitz.Matrix(zoom, zoom)

    image_paths = []
    for i, page in enumerate(doc):
        pix = page.get_pixmap(matrix=mat)
        img_path = out_dir / f"page_{i + 1:03d}.png"
        pix.save(str(img_path))
        image_paths.append(img_path)
    doc.close()
    return image_paths


# ------------------- Step 2: grayscale + CLAHE preprocessing -------------

def preprocess_image(img_path: Path, out_dir: Path) -> Path:
    img = cv2.imread(str(img_path))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILE_GRID)
    enhanced = clahe.apply(gray)

    out_path = out_dir / img_path.name
    cv2.imwrite(str(out_path), enhanced)
    return out_path


# ------------------------- Step 3: table detection ------------------------

def detect_table_boxes(gray_img: np.ndarray) -> list[tuple[int, int, int, int]]:
    """
    Detect bordered tables in a grayscale image using line morphology.
    Returns a list of (x, y, w, h) boxes, sorted top-to-bottom.
    """
    h, w = gray_img.shape

    # Binarize (inverse so lines are white on black for morphology)
    bin_img = cv2.adaptiveThreshold(
        gray_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 10
    )

    # Detect horizontal lines
    horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(w // 30, 20), 1))
    horiz_lines = cv2.erode(bin_img, horiz_kernel, iterations=1)
    horiz_lines = cv2.dilate(horiz_lines, horiz_kernel, iterations=1)

    # Detect vertical lines
    vert_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(h // 30, 20)))
    vert_lines = cv2.erode(bin_img, vert_kernel, iterations=1)
    vert_lines = cv2.dilate(vert_lines, vert_kernel, iterations=1)

    # Combine -- a table's grid is where horizontal and vertical lines both exist
    grid = cv2.add(horiz_lines, vert_lines)
    grid = cv2.dilate(grid, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)), iterations=2)

    contours, _ = cv2.findContours(grid, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    page_area = h * w
    boxes = []
    for cnt in contours:
        x, y, cw, ch = cv2.boundingRect(cnt)
        area_fraction = (cw * ch) / page_area
        if area_fraction < MIN_TABLE_AREA_FRACTION:
            continue
        # A real table region should have some horizontal AND vertical line pixels inside it
        boxes.append((x, y, cw, ch))

    # Merge boxes that overlap or are very close (a table can get split into
    # multiple contours if a row has a gap in its borders)
    boxes = merge_close_boxes(boxes)

    # Sort top-to-bottom (reading order)
    boxes.sort(key=lambda b: b[1])
    return boxes


def merge_close_boxes(
    boxes: list[tuple[int, int, int, int]], gap_thresh: int = 25
) -> list[tuple[int, int, int, int]]:
    """Merge boxes whose bounding rectangles are within gap_thresh px of each other."""
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


# --------------------- Step 4: mask tables + stamp tokens ------------------

def mask_tables_on_image(
    gray_img: np.ndarray, boxes: list[tuple[int, int, int, int]]
) -> np.ndarray:
    """Return a copy of the image with each table box painted over with a solid
    gray box, stamped with a literal [[TABLE_N]] token."""
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


def crop_table(
    gray_img: np.ndarray, box: tuple[int, int, int, int], padding: int = TABLE_CROP_PADDING
) -> np.ndarray:
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


# ------------------------ Step 5: call Qwen via Ollama --------------------

def array_to_base64_png(img: np.ndarray) -> str:
    success, buf = cv2.imencode(".png", img)
    if not success:
        raise RuntimeError("Failed to encode image to PNG")
    return base64.b64encode(buf.tobytes()).decode("utf-8")


def call_qwen(prompt: str, img_b64: str) -> str:
    payload = {
        "model": MODEL_NAME,
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
    except requests.exceptions.RequestException as e:
        body = ""
        try:
            body = resp.text
        except Exception:
            pass
        return f"[ERROR calling Ollama: {e} | response body: {body}]"

    data = resp.json()
    return data.get("response", "").strip()


# ------------------------------- Orchestration -----------------------------

def process_page(gray_img: np.ndarray, page_num: int, debug_dir: Path) -> str:
    boxes = detect_table_boxes(gray_img)

    if not boxes:
        # No tables detected -- just parse the whole page as prose
        page_b64 = array_to_base64_png(gray_img)
        return call_qwen(PROSE_PROMPT_TEMPLATE, page_b64)

    # Save a debug copy showing detected boxes (helps you tune MIN_TABLE_AREA_FRACTION)
    debug_img = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)
    for (x, y, w, h) in boxes:
        cv2.rectangle(debug_img, (x, y), (x + w, y + h), (0, 0, 255), 3)
    debug_dir.mkdir(parents=True, exist_ok=True)  # defensive, same reasoning as text_dir
    cv2.imwrite(str(debug_dir / f"page_{page_num:03d}_detected_tables.png"), debug_img)

    # 1. Mask tables on the page, get prose with placeholder tokens
    masked_img = mask_tables_on_image(gray_img, boxes)
    masked_b64 = array_to_base64_png(masked_img)
    prose_text = call_qwen(PROSE_PROMPT_TEMPLATE, masked_b64)

    # 2. Parse each table crop separately
    table_htmls = []
    for (x, y, w, h) in boxes:
        crop = crop_table(gray_img, (x, y, w, h))
        crop_b64 = array_to_base64_png(crop)
        table_html = call_qwen(TABLE_PROMPT, crop_b64)
        table_htmls.append(table_html)

    # 3. Splice table HTML back into the prose text, replacing each token
    final_text = prose_text
    for idx, table_html in enumerate(table_htmls, start=1):
        token = f"[[TABLE_{idx}]]"
        if token in final_text:
            final_text = final_text.replace(token, f"\n\n{table_html}\n\n")
        else:
            # Model didn't echo the token back -- append at the end so nothing is lost
            final_text += f"\n\n<!-- placeholder {token} not found in prose output, appending -->\n\n{table_html}\n\n"

    return final_text


def run_pipeline(pdf_path: str, output_dir: str):
    pdf_path = Path(pdf_path)
    output_dir = Path(output_dir)

    raw_img_dir = output_dir / "raw_pages"
    proc_img_dir = output_dir / "processed_pages"
    debug_dir = output_dir / "debug_detected_tables"
    text_dir = output_dir / "page_text"
    for d in (raw_img_dir, proc_img_dir, debug_dir, text_dir):
        d.mkdir(parents=True, exist_ok=True)

    print(f"[1/3] Rendering '{pdf_path.name}' to images @ {RENDER_DPI} DPI ...")
    raw_images = pdf_to_images(pdf_path, raw_img_dir)
    print(f"      -> {len(raw_images)} pages rendered")

    print("[2/3] Preprocessing (grayscale + CLAHE) ...")
    processed_images = [preprocess_image(p, proc_img_dir) for p in tqdm(raw_images)]

    print(f"[3/3] Detecting tables + parsing pages with {MODEL_NAME} via Ollama ...")
    combined_md = []
    for i, img_path in enumerate(tqdm(processed_images), start=1):
        gray_img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        page_text = process_page(gray_img, i, debug_dir)

        page_out_path = text_dir / f"page_{i:03d}.md"
        text_dir.mkdir(parents=True, exist_ok=True)  # defensive: OneDrive/AV can transiently evict a fresh folder
        page_out_path.write_text(page_text, encoding="utf-8")

        combined_md.append(f"\n\n<!-- ===== Page {i} ===== -->\n\n{page_text}")

    final_path = output_dir / f"{pdf_path.stem}_parsed.md"
    final_path.write_text("".join(combined_md), encoding="utf-8")

    print(f"\nDone. Combined output written to: {final_path}")
    print(f"Per-page raw text saved under: {text_dir}")
    print(f"Debug images with detected table boxes saved under: {debug_dir}")
    return final_path


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python arabic_pdf_parser.py <input.pdf> <output_dir>")
        sys.exit(1)

    run_pipeline(sys.argv[1], sys.argv[2])
