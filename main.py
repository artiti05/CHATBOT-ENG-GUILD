"""
Jordan Engineers Association — Unified Multi-Mode CLI Entry Point
===================================================================

Execution Modes:

1. Web Chatbot UI Server:
   python main.py                              # Default: Launch server on http://localhost:8000
   python main.py serve --port 8000            # Specify custom port

2. Ingest Text Files Only (texts/):
   python main.py ingest-texts                 # Clean, chunk & embed texts/ into ChromaDB
   python main.py ingest-texts --no-reset-db   # Append texts/ without resetting existing DB

3. Ingest PDF Files Only (pdfs/):
   python main.py ingest-pdfs                  # GPU Vision parse & embed pdfs/ into ChromaDB
   python main.py ingest-pdfs --no-reset-db    # Append pdfs/ without resetting existing DB

4. Ingest Pre-parsed Markdown Files Only (output_dir/):
   python main.py ingest-markdown              # Index output_dir/ markdown files

5. Ingest All Sources (texts/ + output_dir/ + pdfs/):
   python main.py ingest-all                   # Full end-to-end ingestion
"""

import sys
import os
import socket
import argparse
from pathlib import Path

# Force UTF-8 encoding on Windows console
if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))


def is_port_in_use(port: int, host: str = "127.0.0.1") -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex((host, port)) == 0


def run_server(port: int = 8000):
    """Launches the FastAPI Web Server and Arabic/Jordanian AI Chatbot UI."""
    import uvicorn

    target_port = port
    if is_port_in_use(target_port):
        for alt in [8080, 8000, 8081, 8501, 5000]:
            if not is_port_in_use(alt):
                print(f"[INFO] Port {target_port} is occupied. Using alternative port {alt}.")
                target_port = alt
                break

    print("\n" + "=" * 65)
    print(" 🚀 GUILD KNOWLEDGE BASE RAG WEB SERVER")
    print(f"  - URL: http://localhost:{target_port}")
    print("=" * 65 + "\n")

    from app import app
    uvicorn.run(app, host="0.0.0.0", port=target_port)


def run_ingest_texts(args):
    """Ingests text files from texts/ directory into vector database."""
    from batch_ingest import reset_storage_db, index_texts_directory
    from ingestion_pipeline import IngestionPipeline
    from crawler_admin import DocumentRegistry
    from config.settings import TEXTS_DIR

    texts_dir_path = (BASE_DIR / args.texts_dir).resolve() if hasattr(args, "texts_dir") and args.texts_dir else TEXTS_DIR

    print("\n" + "=" * 65)
    print(" 📝 INGESTING TEXT KNOWLEDGE BASE (texts/)")
    print(f"  - Source Directory: {texts_dir_path}")
    print(f"  - Reset DB:         {not args.no_reset_db and not args.dry_run}")
    print(f"  - Dry Run:          {args.dry_run}")
    print("=" * 65 + "\n")

    if not args.no_reset_db and not args.dry_run:
        reset_storage_db()

    pipeline = IngestionPipeline()
    registry = DocumentRegistry()
    count = index_texts_directory(texts_dir_path, pipeline, registry, dry_run=args.dry_run)
    print(f"\n✅ Text ingestion finished ({count} files processed).")


def run_ingest_pdfs(args):
    """Vision parses and ingests PDF files from pdfs/ directory into vector database."""
    from batch_ingest import reset_storage_db, parse_remaining_pdfs_gpu
    from ingestion_pipeline import IngestionPipeline
    from crawler_admin import DocumentRegistry
    from config.settings import PDFS_DIR

    pdfs_dir_path = (BASE_DIR / args.pdfs_dir).resolve() if hasattr(args, "pdfs_dir") and args.pdfs_dir else PDFS_DIR

    print("\n" + "=" * 65)
    print(" 📄 INGESTING PDF DOCUMENTS (pdfs/) WITH VISION VLM")
    print(f"  - Source Directory: {pdfs_dir_path}")
    print(f"  - Reset DB:         {not args.no_reset_db and not args.dry_run}")
    print(f"  - Dry Run:          {args.dry_run}")
    print("=" * 65 + "\n")

    if not args.no_reset_db and not args.dry_run:
        reset_storage_db()

    pipeline = IngestionPipeline()
    registry = DocumentRegistry()
    parse_remaining_pdfs_gpu(pdfs_dir_path, pipeline, registry, indexed_stems=set(), dry_run=args.dry_run)
    print("\n✅ PDF Vision ingestion finished.")


def run_ingest_markdown(args):
    """Indexes pre-parsed Markdown files from output_dir/ into vector database."""
    from batch_ingest import reset_storage_db, index_preparsed_markdown_files
    from ingestion_pipeline import IngestionPipeline
    from crawler_admin import DocumentRegistry
    from config.settings import PARSED_OUTPUT_DIR

    out_dir_path = (BASE_DIR / args.output_dir).resolve() if hasattr(args, "output_dir") and args.output_dir else PARSED_OUTPUT_DIR

    print("\n" + "=" * 65)
    print(" 📑 INGESTING PRE-PARSED MARKDOWN FILES (output_dir/)")
    print(f"  - Source Directory: {out_dir_path}")
    print(f"  - Reset DB:         {not args.no_reset_db and not args.dry_run}")
    print(f"  - Dry Run:          {args.dry_run}")
    print("=" * 65 + "\n")

    if not args.no_reset_db and not args.dry_run:
        reset_storage_db()

    pipeline = IngestionPipeline()
    registry = DocumentRegistry()
    stems = index_preparsed_markdown_files(out_dir_path, pipeline, registry, dry_run=args.dry_run)
    print(f"\n✅ Markdown ingestion finished ({len(stems)} files indexed).")


def run_ingest_kb(args):
    """Ingests texts/ and output_dir/ pre-parsed Markdown files into vector database (skips pdfs/)."""
    from batch_ingest import reset_storage_db, index_texts_directory, index_preparsed_markdown_files
    from ingestion_pipeline import IngestionPipeline
    from crawler_admin import DocumentRegistry
    from config.settings import TEXTS_DIR, PARSED_OUTPUT_DIR

    texts_dir_path = (BASE_DIR / args.texts_dir).resolve() if hasattr(args, "texts_dir") and args.texts_dir else TEXTS_DIR
    out_dir_path = (BASE_DIR / args.output_dir).resolve() if hasattr(args, "output_dir") and args.output_dir else PARSED_OUTPUT_DIR

    print("\n" + "=" * 65)
    print(" 📚 INGESTING KNOWLEDGE BASE (texts/ + output_dir/ Markdown)")
    print(f"  - Texts Directory:   {texts_dir_path}")
    print(f"  - Markdown Directory:{out_dir_path}")
    print(f"  - Reset DB:         {not args.no_reset_db and not args.dry_run}")
    print(f"  - Dry Run:          {args.dry_run}")
    print("=" * 65 + "\n")

    if not args.no_reset_db and not args.dry_run:
        reset_storage_db()

    pipeline = IngestionPipeline()
    registry = DocumentRegistry()

    # 1. Texts
    count = index_texts_directory(texts_dir_path, pipeline, registry, dry_run=args.dry_run)
    # 2. Markdown
    stems = index_preparsed_markdown_files(out_dir_path, pipeline, registry, dry_run=args.dry_run)

    print(f"\n✅ Knowledge base ingestion finished ({count} text files + {len(stems)} pre-parsed markdown files indexed).")


def run_ingest_all(args):
    """Runs complete ingestion (texts/ + output_dir/ + pdfs/)."""
    from batch_ingest import reset_storage_db, index_texts_directory, index_preparsed_markdown_files, parse_remaining_pdfs_gpu
    from ingestion_pipeline import IngestionPipeline
    from crawler_admin import DocumentRegistry
    from config.settings import TEXTS_DIR, PDFS_DIR, PARSED_OUTPUT_DIR

    texts_dir_path = (BASE_DIR / args.texts_dir).resolve() if hasattr(args, "texts_dir") and args.texts_dir else TEXTS_DIR
    out_dir_path = (BASE_DIR / args.output_dir).resolve() if hasattr(args, "output_dir") and args.output_dir else PARSED_OUTPUT_DIR
    pdfs_dir_path = (BASE_DIR / args.pdfs_dir).resolve() if hasattr(args, "pdfs_dir") and args.pdfs_dir else PDFS_DIR

    print("\n" + "=" * 65)
    print(" ⚡ FULL END-TO-END INGESTION PIPELINE (texts + markdown + pdfs)")
    print("=" * 65 + "\n")

    if not args.no_reset_db and not args.dry_run:
        reset_storage_db()

    pipeline = IngestionPipeline()
    registry = DocumentRegistry()

    # 1. Texts
    index_texts_directory(texts_dir_path, pipeline, registry, dry_run=args.dry_run)

    # 2. Markdown
    indexed_stems = index_preparsed_markdown_files(out_dir_path, pipeline, registry, dry_run=args.dry_run)

    # 3. PDFs
    parse_remaining_pdfs_gpu(pdfs_dir_path, pipeline, registry, indexed_stems=indexed_stems, dry_run=args.dry_run)
    print("\n✅ Full end-to-end ingestion completed.")


def main():
    parser = argparse.ArgumentParser(
        description="Jordan Engineers Association — Arabic PDF Parser & RAG Chatbot CLI",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # 1. Serve command
    serve_parser = subparsers.add_parser("serve", help="Launch FastAPI Web Chatbot UI (Default)")
    serve_parser.add_argument("--port", type=int, default=8000, help="Server port number (default: 8000)")

    # 2. Ingest Knowledge Base (texts/ + output_dir/)
    ingest_kb_parser = subparsers.add_parser("ingest-kb", help="Ingest knowledge base from texts/ and pre-parsed output_dir/ (skips pdfs/)")
    ingest_kb_parser.add_argument("--texts-dir", type=str, default="texts", help="Path to texts directory")
    ingest_kb_parser.add_argument("--output-dir", type=str, default="output_dir", help="Path to output_dir directory")
    ingest_kb_parser.add_argument("--no-reset-db", action="store_true", help="Do not reset vector database before indexing (Append mode)")
    ingest_kb_parser.add_argument("--dry-run", action="store_true", help="Preview file plan without modifying storage")

    # 3. Ingest Texts Only
    ingest_texts_parser = subparsers.add_parser("ingest-texts", help="Ingest text knowledge base files from texts/ directory only")
    ingest_texts_parser.add_argument("--texts-dir", type=str, default="texts", help="Path to texts directory")
    ingest_texts_parser.add_argument("--no-reset-db", action="store_true", help="Do not reset vector database before indexing (Append mode)")
    ingest_texts_parser.add_argument("--dry-run", action="store_true", help="Preview file plan without modifying storage")

    # 4. Ingest PDFs Only
    ingest_pdfs_parser = subparsers.add_parser("ingest-pdfs", help="Vision parse and ingest PDF files from pdfs/ directory only")
    ingest_pdfs_parser.add_argument("--pdfs-dir", type=str, default="pdfs", help="Path to pdfs directory")
    ingest_pdfs_parser.add_argument("--no-reset-db", action="store_true", help="Do not reset vector database before indexing (Append mode)")
    ingest_pdfs_parser.add_argument("--dry-run", action="store_true", help="Preview file plan without modifying storage")

    # 5. Ingest Markdown Only
    ingest_md_parser = subparsers.add_parser("ingest-markdown", help="Index pre-parsed Markdown files from output_dir/ directory only")
    ingest_md_parser.add_argument("--output-dir", type=str, default="output_dir", help="Path to output_dir directory")
    ingest_md_parser.add_argument("--no-reset-db", action="store_true", help="Do not reset vector database before indexing (Append mode)")
    ingest_md_parser.add_argument("--dry-run", action="store_true", help="Preview file plan without modifying storage")

    # 6. Ingest All
    ingest_all_parser = subparsers.add_parser("ingest-all", help="Run full end-to-end ingestion (texts/ + output_dir/ + pdfs/)")
    ingest_all_parser.add_argument("--texts-dir", type=str, default="texts", help="Path to texts directory")
    ingest_all_parser.add_argument("--output-dir", type=str, default="output_dir", help="Path to output_dir directory")
    ingest_all_parser.add_argument("--pdfs-dir", type=str, default="pdfs", help="Path to pdfs directory")
    ingest_all_parser.add_argument("--no-reset-db", action="store_true", help="Do not reset vector database before indexing (Append mode)")
    ingest_all_parser.add_argument("--dry-run", action="store_true", help="Preview file plan without modifying storage")

    # Legacy fallback for `ingest` -> maps to `ingest-all`
    ingest_parser = subparsers.add_parser("ingest", help="Alias for ingest-all")
    ingest_parser.add_argument("--texts-dir", type=str, default="texts")
    ingest_parser.add_argument("--output-dir", type=str, default="output_dir")
    ingest_parser.add_argument("--pdfs-dir", type=str, default="pdfs")
    ingest_parser.add_argument("--no-reset-db", action="store_true")
    ingest_parser.add_argument("--dry-run", action="store_true")

    # Default if no arguments specified: `python main.py` -> `run_server()`
    if len(sys.argv) == 1:
        run_server(port=8000)
        return

    args = parser.parse_args()

    if args.command == "serve":
        run_server(port=args.port)
    elif args.command == "ingest-kb":
        run_ingest_kb(args)
    elif args.command == "ingest-texts":
        run_ingest_texts(args)
    elif args.command == "ingest-pdfs":
        run_ingest_pdfs(args)
    elif args.command == "ingest-markdown":
        run_ingest_markdown(args)
    elif args.command in ["ingest-all", "ingest"]:
        run_ingest_all(args)
    else:
        run_server(port=8000)


if __name__ == "__main__":
    main()
