"""
Arabic PDF Parser & Knowledge Base Batch Ingestion Tool
========================================================

Usage:
  python -m src.kb_ingestor.batch_ingest --index-only  # Resets DB & indexes pre-parsed Markdown files into storage NOW
  python -m src.kb_ingestor.batch_ingest --parse-only  # GPU machine: parses remaining PDFs into storage
  python -m src.kb_ingestor.batch_ingest               # Runs full pipeline (reset -> index pre-parsed -> GPU parse remaining)
  python -m src.kb_ingestor.batch_ingest --dry-run     # Previews files to process without modifying DB or VLM
"""

import argparse
import hashlib
import os
import shutil
import sys
from pathlib import Path

# Force UTF-8 stdout encoding for Windows PowerShell/CMD
if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Add project root to path
BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.cache_db.document_registry import DocumentRegistry  # noqa: E402
from src.config import (  # noqa: E402
    CHROMA_PERSIST_DIR,
    MARKDOWNS_DIR,
    OUTPUT_DIR,
    PARSED_OUTPUT_DIR,
    PDFS_DIR,
    REGISTRY_DB_PATH,
    STORAGE_DIR,
    TEXTS_DIR,
)

from .ingestion import IngestionPipeline, clean_document_text  # noqa: E402


def reset_storage_db():
    """Resets the ChromaDB vector store directory and SQLite document registry."""
    print("=" * 65)
    print("[Phase 0] Resetting Vector DB & Registry DB in storage/ ...")
    print("=" * 65)

    if CHROMA_PERSIST_DIR.exists():
        try:
            shutil.rmtree(CHROMA_PERSIST_DIR)
            print(f"  ✔ Removed ChromaDB directory: '{CHROMA_PERSIST_DIR}'")
        except Exception as e:
            print(f"  ⚠ Warning removing ChromaDB directory: {e}")

    if REGISTRY_DB_PATH.exists():
        try:
            os.remove(REGISTRY_DB_PATH)
            print(f"  ✔ Removed SQLite registry DB: '{REGISTRY_DB_PATH}'")
        except Exception as e:
            print(f"  ⚠ Warning removing Registry DB: {e}")

    CHROMA_PERSIST_DIR.mkdir(parents=True, exist_ok=True)
    STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    print("  ✔ Clean storage initialized.\n")


def index_texts_directory(texts_dir: Path, pipeline: IngestionPipeline, registry: DocumentRegistry, dry_run: bool = False) -> int:
    """Scans texts_dir (and all subdirectories) for .md and .txt files, cleans, chunks, embeds, and indexes them into ChromaDB."""
    print("=" * 65)
    print(f"[Phase 1/3] Scanning & Indexing Knowledge Base Text files in '{texts_dir.name}' ...")
    print("=" * 65)

    if not texts_dir.exists():
        print(f"  ⚠ Texts directory '{texts_dir}' does not exist. Skipping Phase 1.\n")
        return 0

    text_files = list(texts_dir.rglob("*.md")) + list(texts_dir.rglob("*.txt"))
    if not text_files:
        print(f"  ⚠ No text/markdown files found in '{texts_dir}'. Skipping Phase 1.\n")
        return 0

    print(f"  ✔ Found {len(text_files)} text files across subdirectories.")

    total_indexed_chunks = 0
    indexed_files_count = 0

    for idx, text_path in enumerate(text_files, start=1):
        rel_path = text_path.relative_to(texts_dir)

        if dry_run:
            print(f"  [Dry Run {idx}/{len(text_files)}] Would index text file: '{rel_path}'")
            continue

        print(f"  [{idx}/{len(text_files)}] Indexing text file: '{rel_path}' ...")
        try:
            with open(text_path, "r", encoding="utf-8", errors="replace") as f:
                raw_text = f.read()

            if not raw_text.strip():
                print(f"      ⚠ File '{rel_path}' is empty. Skipping.")
                continue

            clean_text = clean_document_text(raw_text)
            source_id = f"text_{hashlib.md5(str(rel_path).encode('utf-8')).hexdigest()[:8]}"
            doc_dict = {
                "source_id": source_id,
                "file_name": str(rel_path),
                "source_type": "text_kb",
                "full_text": clean_text,
                "text_path": str(text_path.resolve())
            }

            chunks = pipeline.chunker.chunk_text(doc_dict)
            if chunks:
                cnt = pipeline.indexer.index_chunks(chunks)
                registry.register_document(doc_dict, text_path, status="active")
                total_indexed_chunks += cnt
                indexed_files_count += 1
                print(f"      -> Successfully indexed {cnt} chunks.")
            else:
                print(f"      ⚠ No chunks generated for '{rel_path}'.")
        except Exception as e:
            print(f"      ❌ Error indexing '{rel_path}': {e}")

    if not dry_run:
        pipeline.flush_memory()
        print(f"  ✔ Phase 1 Complete: {indexed_files_count} text files indexed ({total_indexed_chunks} chunks total).\n")

    return indexed_files_count


def index_preparsed_markdown_files(output_dir: Path, pipeline: IngestionPipeline, registry: DocumentRegistry, dry_run: bool = False):
    """Scans output_dir and pdf_parsed_results for pre-parsed Markdown files and indexes them into ChromaDB."""
    print("=" * 65)
    print(f"[Phase 2/3] Scanning & Indexing pre-parsed Markdown files in '{output_dir.name}' & '{PARSED_OUTPUT_DIR.name}' ...")
    print("=" * 65)

    search_dirs = [output_dir, PARSED_OUTPUT_DIR, MARKDOWNS_DIR]
    parsed_files = []
    seen_paths = set()

    for s_dir in search_dirs:
        if s_dir and s_dir.exists():
            found = list(s_dir.rglob("*_parsed.md"))
            if not found:
                found = [f for f in s_dir.rglob("*.md") if not f.name.startswith("page_")]
            for f in found:
                if f.resolve() not in seen_paths:
                    seen_paths.add(f.resolve())
                    parsed_files.append(f)

    if not parsed_files:
        print("  ⚠ No pre-parsed Markdown files found. Skipping Phase 2.\n")
        return set()

    print(f"  ✔ Found {len(parsed_files)} pre-parsed Markdown files.")

    indexed_stems = set()
    total_indexed_chunks = 0

    for idx, md_path in enumerate(parsed_files, start=1):
        stem = md_path.name.replace("_parsed.md", "").replace(".md", "")
        indexed_stems.add(stem)

        if dry_run:
            print(f"  [Dry Run {idx}/{len(parsed_files)}] Would index pre-parsed: '{md_path.name}' (Stem: {stem})")
            continue

        print(f"  [{idx}/{len(parsed_files)}] Indexing pre-parsed: '{md_path.name}' ...")
        try:
            with open(md_path, "r", encoding="utf-8", errors="replace") as f:
                raw_text = f.read()

            clean_text = clean_document_text(raw_text)
            source_id = f"pdf_{stem}"
            doc_dict = {
                "source_id": source_id,
                "file_name": f"{stem}.pdf",
                "source_type": "pdf_parsed",
                "full_text": clean_text,
                "parsed_md_path": str(md_path.resolve())
            }

            chunks = pipeline.chunker.chunk_text(doc_dict)
            cnt = pipeline.indexer.index_chunks(chunks)
            registry.register_document(doc_dict, md_path, status="active")
            total_indexed_chunks += cnt
            print(f"      -> Successfully indexed {cnt} chunks.")
        except Exception as e:
            print(f"      ❌ Error indexing '{md_path.name}': {e}")

    if not dry_run:
        pipeline.flush_memory()
        print(f"  ✔ Phase 2 Complete: {len(indexed_stems)} files indexed ({total_indexed_chunks} chunks total).\n")

    return indexed_stems


def parse_remaining_pdfs_gpu(pdfs_dir: Path, pipeline: IngestionPipeline, registry: DocumentRegistry, indexed_stems: set, dry_run: bool = False):
    """Scans pdfs_dir for remaining unparsed PDF files and parses them using GPU VLM."""
    print("=" * 65)
    print(f"[Phase 3/3] Scanning PDF files in '{pdfs_dir.name}' for GPU Vision Parsing ...")
    print("=" * 65)

    if not pdfs_dir.exists():
        print(f"  ⚠ PDFs directory '{pdfs_dir}' does not exist. Skipping Phase 3.\n")
        return

    all_pdfs = list(pdfs_dir.rglob("*.pdf"))
    remaining_pdfs = [p for p in all_pdfs if p.stem not in indexed_stems]

    print(f"  ✔ Total PDFs found: {len(all_pdfs)}")
    print(f"  ✔ Already indexed stems: {len(indexed_stems)}")
    print(f"  ✔ Remaining PDFs to parse with GPU VLM: {len(remaining_pdfs)}\n")

    if dry_run:
        print("  [Dry Run Mode] Remaining PDFs that would be parsed:")
        for idx, pdf_path in enumerate(remaining_pdfs, start=1):
            print(f"    {idx}. {pdf_path.name} ({pdf_path.parent.name})")
        print("\n  ✔ Dry run complete. No files were processed.")
        return

    success_count = 0
    fail_count = 0

    for idx, pdf_path in enumerate(remaining_pdfs, start=1):
        print(f"[{idx}/{len(remaining_pdfs)}] Parsing GPU Vision PDF: '{pdf_path.name}' ...")
        try:
            res = pipeline.process_file(pdf_path)
            doc = res["document"]
            registry.register_document(doc, pdf_path, status="active")
            indexed_cnt = res.get("indexed_count", 0)
            print(f"    ✔ Finished '{pdf_path.name}': {indexed_cnt} chunks indexed.")
            success_count += 1
        except Exception as e:
            print(f"    ❌ Error processing '{pdf_path.name}': {e}")
            fail_count += 1

        pipeline.flush_memory()

    print("\n" + "=" * 65)
    print(" 🎉 BATCH INGESTION SUMMARY")
    print("=" * 65)
    print(f"  - Pre-parsed documents indexed: {len(indexed_stems)}")
    print(f"  - Remaining PDFs processed:     {success_count} succeeded, {fail_count} failed")
    print(f"  - Total documents in VDB:       {len(indexed_stems) + success_count}")
    print("=" * 65 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Arabic PDF Parser & Knowledge Base Batch Ingestion Tool")
    parser.add_argument("--texts-dir", type=str, default=str(TEXTS_DIR), help="Path to input text knowledge base directory")
    parser.add_argument("--markdowns-dir", type=str, default=str(MARKDOWNS_DIR), help="Path to pre-parsed Markdown directory")
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR), help="Path to PDF vision parser output directory")
    parser.add_argument("--pdfs-dir", type=str, default=str(PDFS_DIR), help="Path to input PDFs directory")
    parser.add_argument("--index-only", action="store_true", help="Reset DB & index pre-parsed Markdown files from markdowns_dir only")
    parser.add_argument("--parse-only", action="store_true", help="Skip pre-parsed indexing and run GPU PDF parsing only")
    parser.add_argument("--no-reset-db", action="store_true", help="Do not reset the existing vector database before indexing")
    parser.add_argument("--dry-run", action="store_true", help="Scan files and print plan without executing modifications")

    args = parser.parse_args()

    texts_dir_path = Path(args.texts_dir).resolve()
    out_dir_path = Path(args.output_dir).resolve()
    pdfs_dir_path = Path(args.pdfs_dir).resolve()

    if not args.no_reset_db and not args.dry_run and not args.parse_only:
        reset_storage_db()

    pipeline = IngestionPipeline()
    registry = DocumentRegistry()

    index_texts_directory(texts_dir_path, pipeline, registry, dry_run=args.dry_run)

    indexed_stems = set()
    if not args.parse_only:
        indexed_stems = index_preparsed_markdown_files(out_dir_path, pipeline, registry, dry_run=args.dry_run)

    if not args.index_only:
        parse_remaining_pdfs_gpu(pdfs_dir_path, pipeline, registry, indexed_stems, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
