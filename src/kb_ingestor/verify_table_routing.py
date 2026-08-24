"""
Verify Two-Pass Table-Aware Vision Parsing
============================================
Ad-hoc test harness for process_vision_page()'s table routing. Renders one or a
few pages of a PDF, runs the real vision parser directly (bypassing chunking,
embedding, and the document registry), and writes debug artifacts + parsed
markdown so table routing can be inspected without running full ingestion.

Requires Ollama running with OLLAMA_VISION_MODEL pulled (see .env) -- this
calls the same vision endpoint as real ingestion, just for one page at a time.

Usage:
    python -m src.kb_ingestor.verify_table_routing --pdf data/pdfs/some_file.pdf --page 3
    python -m src.kb_ingestor.verify_table_routing --pdf data/pdfs/some_file.pdf --pages 3,4,7

A/B comparison against the pre-routing behavior (single-pass):
    python -m src.kb_ingestor.verify_table_routing --pdf data/pdfs/some_file.pdf --page 3 --no-routing

Output goes under data/output_dir/<pdf_stem>_TEST/ so it never collides with
real ingestion output for the same PDF:
    debug_detected_tables/page_XXX_tables.png  -- annotated boxes (detection quality)
    page_text/page_XXX.md                      -- parsed output (extraction quality)
"""
import argparse
import os
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))


def main():
    parser = argparse.ArgumentParser(description="Test the two-pass table-aware Vision parser on specific PDF pages.")
    parser.add_argument("--pdf", required=True, help="Path to a PDF (e.g. data/pdfs/some_file.pdf)")
    parser.add_argument("--page", type=int, help="Single 1-indexed page number to test")
    parser.add_argument("--pages", type=str, help="Comma-separated 1-indexed page numbers, e.g. 3,4,7")
    parser.add_argument("--no-routing", action="store_true",
                         help="Force ENABLE_TABLE_ROUTING=0 for this run (baseline for A/B comparison)")
    args = parser.parse_args()

    if args.no_routing:
        os.environ["ENABLE_TABLE_ROUTING"] = "0"

    import fitz
    import cv2
    from src.config import RENDER_DPI, PARSED_OUTPUT_DIR, ENABLE_TABLE_ROUTING
    from src.kb_ingestor.ingestion import preprocess_image_cv2, cv2_imread_unicode, process_vision_page

    pdf_path = Path(args.pdf).resolve()
    if not pdf_path.exists():
        print(f"PDF not found: {pdf_path}")
        sys.exit(1)

    if args.page:
        page_nums = [args.page]
    elif args.pages:
        page_nums = [int(p.strip()) for p in args.pages.split(",")]
    else:
        print("Specify --page N or --pages N,M,...")
        sys.exit(1)

    print(f"ENABLE_TABLE_ROUTING = {ENABLE_TABLE_ROUTING}")

    out_dir = PARSED_OUTPUT_DIR / f"{pdf_path.stem}_TEST"
    raw_dir = out_dir / "raw_pages"
    proc_dir = out_dir / "processed_pages"
    debug_dir = out_dir / "debug_detected_tables"
    text_dir = out_dir / "page_text"
    for d in (raw_dir, proc_dir, debug_dir, text_dir):
        d.mkdir(parents=True, exist_ok=True)

    doc = fitz.open(pdf_path)
    zoom = RENDER_DPI / 72
    mat = fitz.Matrix(zoom, zoom)

    for page_num in page_nums:
        if page_num < 1 or page_num > len(doc):
            print(f"Page {page_num} out of range (PDF has {len(doc)} pages), skipping.")
            continue

        page = doc[page_num - 1]
        pix = page.get_pixmap(matrix=mat)
        raw_path = raw_dir / f"page_{page_num:03d}.png"
        pix.save(str(raw_path))

        proc_path = preprocess_image_cv2(raw_path, proc_dir)
        gray_img = cv2_imread_unicode(proc_path, cv2.IMREAD_GRAYSCALE)

        print(f"\n--- Testing page {page_num} ---")
        result = process_vision_page(gray_img, page_num, debug_dir)

        out_md = text_dir / f"page_{page_num:03d}.md"
        out_md.write_text(result, encoding="utf-8")
        print(f"Parsed output: {out_md}")

        debug_img = debug_dir / f"page_{page_num:03d}_tables.png"
        if debug_img.exists():
            print(f"Detection debug image: {debug_img}")

    doc.close()
    print(f"\nAll test artifacts under: {out_dir}")


if __name__ == "__main__":
    main()
