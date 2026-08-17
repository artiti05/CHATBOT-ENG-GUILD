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


def kill_port_owner(port: int = 8000):
    """Frees target port if occupied by a stale process on Windows/Linux."""
    if is_port_in_use(port):
        print(f"[INFO] Freeing port {port} from stale listener process...")
        try:
            import subprocess, time
            if sys.platform == "win32":
                out = subprocess.check_output(f"netstat -ano | findstr :{port}", shell=True).decode()
                for line in out.strip().splitlines():
                    parts = line.strip().split()
                    if len(parts) >= 5 and "LISTENING" in parts:
                        pid = parts[-1]
                        subprocess.run(f"taskkill /F /PID {pid}", shell=True, capture_output=True)
                time.sleep(0.5)
        except Exception as err:
            print(f"[Warning] Could not auto-free port {port}: {err}")


def run_server(port: int = 8000):
    """Launches the FastAPI Web Server and Arabic/Jordanian AI Chatbot UI on Port 8000."""
    import uvicorn

    target_port = port
    if is_port_in_use(target_port):
        kill_port_owner(target_port)

    print("\n" + "=" * 65)
    print(" 🚀 GUILD KNOWLEDGE BASE RAG WEB SERVER")
    print(f"  - URL: http://localhost:{target_port}")
    print("=" * 65 + "\n")

    from src.api.main import app
    uvicorn.run(app, host="0.0.0.0", port=target_port)


def resolve_dir(dir_arg, config_default: Path) -> Path:
    if not dir_arg:
        return config_default
    p = Path(dir_arg)
    if p.is_absolute():
        return p
    # Check relative to workspace root vs config_default
    if (BASE_DIR / p).exists():
        return (BASE_DIR / p).resolve()
    return config_default.resolve()


def run_ingest_texts(args):
    """Ingests text files from data/texts/ directory into vector database."""
    from scripts.batch_ingest import reset_storage_db, index_texts_directory
    from src.pipeline.stage_02_retrieve.ingestion import IngestionPipeline
    from src.cache_db.document_registry import DocumentRegistry
    from src.config import TEXTS_DIR

    texts_dir_path = resolve_dir(getattr(args, "texts_dir", None), TEXTS_DIR)

    print("\n" + "=" * 65)
    print(" 📝 INGESTING TEXT KNOWLEDGE BASE (data/texts/)")
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
    """Vision parses and ingests PDF files from data/pdfs/ directory into vector database."""
    from scripts.batch_ingest import reset_storage_db, parse_remaining_pdfs_gpu
    from src.pipeline.stage_02_retrieve.ingestion import IngestionPipeline
    from src.cache_db.document_registry import DocumentRegistry
    from src.config import PDFS_DIR

    pdfs_dir_path = resolve_dir(getattr(args, "pdfs_dir", None), PDFS_DIR)

    print("\n" + "=" * 65)
    print(" 📄 INGESTING PDF DOCUMENTS (data/pdfs/) WITH VISION VLM")
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
    """Indexes pre-parsed Markdown files from data/output_dir/ & data/storage/pdf_parsed_results/ into vector database."""
    from scripts.batch_ingest import reset_storage_db, index_preparsed_markdown_files
    from src.pipeline.stage_02_retrieve.ingestion import IngestionPipeline
    from src.cache_db.document_registry import DocumentRegistry
    from src.config import OUTPUT_DIR

    out_dir_path = resolve_dir(getattr(args, "output_dir", None), OUTPUT_DIR)

    print("\n" + "=" * 65)
    print(" 📑 INGESTING PRE-PARSED MARKDOWN FILES (data/output_dir/)")
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
    """Ingests texts/ and pre-parsed Markdown files into vector database (skips pdfs/)."""
    from scripts.batch_ingest import reset_storage_db, index_texts_directory, index_preparsed_markdown_files
    from src.pipeline.stage_02_retrieve.ingestion import IngestionPipeline
    from src.cache_db.document_registry import DocumentRegistry
    from src.config import TEXTS_DIR, OUTPUT_DIR

    texts_dir_path = resolve_dir(getattr(args, "texts_dir", None), TEXTS_DIR)
    out_dir_path = resolve_dir(getattr(args, "output_dir", None), OUTPUT_DIR)

    print("\n" + "=" * 65)
    print(" 📚 INGESTING KNOWLEDGE BASE (data/texts/ + Markdown)")
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
    """Runs complete ingestion (data/texts/ + data/output_dir/ + data/pdfs/)."""
    from scripts.batch_ingest import reset_storage_db, index_texts_directory, index_preparsed_markdown_files, parse_remaining_pdfs_gpu
    from src.pipeline.stage_02_retrieve.ingestion import IngestionPipeline
    from src.cache_db.document_registry import DocumentRegistry
    from src.config import TEXTS_DIR, PDFS_DIR, OUTPUT_DIR

    texts_dir_path = resolve_dir(getattr(args, "texts_dir", None), TEXTS_DIR)
    out_dir_path = resolve_dir(getattr(args, "output_dir", None), OUTPUT_DIR)
    pdfs_dir_path = resolve_dir(getattr(args, "pdfs_dir", None), PDFS_DIR)


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


def run_terminal_chat(args=None):
    """Runs interactive terminal CLI chat directly in the console using RAGChatbotEngine."""
    from src.rag_chatbot_engine import RAGChatbotEngine
    print("\n" + "=" * 65)
    print(" 🤖 ENG-GUILD CHATBOT — INTERACTIVE TERMINAL ENGINE")
    print(" Type your question and press Enter. Type 'exit' or 'q' to quit.")
    print("=" * 65 + "\n")

    engine = RAGChatbotEngine()

    while True:
        try:
            user_input = input("\n👤 سؤالك (User) > ").strip()
            if not user_input:
                continue
            if user_input.lower() in ["exit", "quit", "q", "خروج"]:
                print("\n👋 Goodbye!")
                break

            res = engine.process_query(user_input)
            print("\n🤖 [Answer]:\n" + res["answer"])
            print("\n📊 [Performance & 13-Stage Latency Breakdown]:")
            print(res["text_breakdown"])

        except (KeyboardInterrupt, EOFError):
            print("\n\n👋 Interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error processing query: {e}")


def run_inspect_chunks(args):
    """Runs Parent-Child Chunking Inspector and saves report to data/chunking_logs/."""
    from scripts.inspect_chunking import inspect_file_chunking, interactive_menu
    file_arg = getattr(args, "file", None)
    if file_arg:
        inspect_file_chunking(Path(file_arg))
    else:
        interactive_menu()



def main():
    parser = argparse.ArgumentParser(
        description="Jordan Engineers Association — Arabic PDF Parser & RAG Chatbot CLI",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Interactive Terminal Chat
    subparsers.add_parser("chat", help="Launch interactive Terminal CLI chat directly in the console")

    # 1. Serve command
    serve_parser = subparsers.add_parser("serve", help="Launch FastAPI Web Chatbot UI (Default)")
    serve_parser.add_argument("--port", type=int, default=8000, help="Server port number (default: 8000)")

    # 2. Ingest Knowledge Base (data/texts/ + data/markdowns/)
    ingest_kb_parser = subparsers.add_parser("ingest-kb", help="Ingest knowledge base from data/texts/ and data/markdowns/ (skips un-parsed pdfs/)")
    ingest_kb_parser.add_argument("--texts-dir", type=str, help="Path to texts directory")
    ingest_kb_parser.add_argument("--markdowns-dir", type=str, help="Path to markdowns directory")
    ingest_kb_parser.add_argument("--no-reset-db", action="store_true", help="Do not reset vector database before indexing (Append mode)")
    ingest_kb_parser.add_argument("--dry-run", action="store_true", help="Preview file plan without modifying storage")

    # 3. Ingest Texts Only
    ingest_texts_parser = subparsers.add_parser("ingest-texts", help="Ingest text knowledge base files from data/texts/ directory only")
    ingest_texts_parser.add_argument("--texts-dir", type=str, help="Path to texts directory")
    ingest_texts_parser.add_argument("--no-reset-db", action="store_true", help="Do not reset vector database before indexing (Append mode)")
    ingest_texts_parser.add_argument("--dry-run", action="store_true", help="Preview file plan without modifying storage")

    # 4. Ingest Markdowns Only
    ingest_md_parser = subparsers.add_parser("ingest-markdowns", aliases=["ingest-markdown"], help="Index pre-parsed Markdown files from data/markdowns/ directory only")
    ingest_md_parser.add_argument("--markdowns-dir", type=str, help="Path to markdowns directory")
    ingest_md_parser.add_argument("--no-reset-db", action="store_true", help="Do not reset vector database before indexing (Append mode)")
    ingest_md_parser.add_argument("--dry-run", action="store_true", help="Preview file plan without modifying storage")

    # 5. Ingest PDFs Only
    ingest_pdfs_parser = subparsers.add_parser("ingest-pdfs", help="Vision parse and ingest PDF files from data/pdfs/ directory only")
    ingest_pdfs_parser.add_argument("--pdfs-dir", type=str, help="Path to pdfs directory")
    ingest_pdfs_parser.add_argument("--no-reset-db", action="store_true", help="Do not reset vector database before indexing (Append mode)")
    ingest_pdfs_parser.add_argument("--dry-run", action="store_true", help="Preview file plan without modifying storage")

    # 6. Ingest All
    ingest_all_parser = subparsers.add_parser("ingest-all", aliases=["ingest"], help="Run full end-to-end ingestion (data/texts/ + data/markdowns/ + data/pdfs/)")
    ingest_all_parser.add_argument("--texts-dir", type=str, help="Path to texts directory")
    ingest_all_parser.add_argument("--markdowns-dir", type=str, help="Path to markdowns directory")
    ingest_all_parser.add_argument("--pdfs-dir", type=str, help="Path to pdfs directory")
    ingest_all_parser.add_argument("--no-reset-db", action="store_true", help="Do not reset vector database before indexing (Append mode)")
    ingest_all_parser.add_argument("--dry-run", action="store_true", help="Preview file plan without modifying storage")

    # 7. Inspect Chunking
    inspect_parser = subparsers.add_parser("inspect-chunks", aliases=["inspect-chunking"], help="Inspect document Parent-Child chunking and save report to data/chunking_logs/")
    inspect_parser.add_argument("--file", "-f", type=str, help="Path to text or markdown file to inspect")

    # Default if no arguments specified: `python main.py` -> `run_server()`
    if len(sys.argv) == 1:
        run_server(port=8000)
        return

    args = parser.parse_args()

    if args.command == "chat":
        run_terminal_chat(args)
    elif args.command == "serve":
        run_server(port=args.port)
    elif args.command == "ingest-kb":
        run_ingest_kb(args)
    elif args.command == "ingest-texts":
        run_ingest_texts(args)
    elif args.command in ["ingest-markdowns", "ingest-markdown"]:
        run_ingest_markdown(args)
    elif args.command == "ingest-pdfs":
        run_ingest_pdfs(args)
    elif args.command in ["ingest-all", "ingest"]:
        run_ingest_all(args)
    elif args.command in ["inspect-chunks", "inspect-chunking"]:
        run_inspect_chunks(args)
    else:
        run_server(port=8000)


if __name__ == "__main__":
    main()

