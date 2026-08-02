"""
Guild Knowledge Base — Main Multi-Mode CLI & Service Router
===========================================================

Generic dual-device entry point:
  1. Ingestion Mode (Ubuntu parsing machine):
     python main.py ingest
     python main.py ingest --dry-run
     python main.py ingest --index-only
     python main.py ingest --parse-only

  2. Interactive Testing Mode (Local / Testing device):
     python main.py test

  3. Single Query Testing:
     python main.py query "ما هي شروط تسجيل المهندسين الأردنيين في النقابة؟"

  4. Web Chatbot Server Mode:
     python main.py serve --port 8080
"""

import sys
import io
import os
import argparse
from pathlib import Path

# Force UTF-8 stdout encoding for cross-platform support (Windows PowerShell / Linux Terminal)
if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))


def run_ingest(args):
    """Executes the batch ingestion pipeline (parse PDFs -> chunk -> BGE-M3 embed -> store in ChromaDB)."""
    from batch_ingest import reset_storage_db, index_preparsed_markdown_files, parse_remaining_pdfs_gpu
    from ingestion_pipeline import IngestionPipeline
    from crawler_admin import DocumentRegistry

    out_dir_path = (BASE_DIR / args.output_dir).resolve()
    pdfs_dir_path = (BASE_DIR / args.pdfs_dir).resolve()

    print("\n" + "=" * 65)
    print(" 🚀 GUILD KNOWLEDGE BASE BATCH INGESTION PIPELINE")
    print(f"  - PDFs Directory:     {pdfs_dir_path}")
    print(f"  - Pre-parsed Output:  {out_dir_path}")
    print(f"  - Dry Run Mode:       {args.dry_run}")
    print("=" * 65 + "\n")

    if not args.no_reset_db and not args.dry_run and not args.parse_only:
        reset_storage_db()

    pipeline = IngestionPipeline()
    registry = DocumentRegistry()

    indexed_stems = set()
    if not args.parse_only:
        indexed_stems = index_preparsed_markdown_files(out_dir_path, pipeline, registry, dry_run=args.dry_run)

    if not args.index_only:
        parse_remaining_pdfs_gpu(pdfs_dir_path, pipeline, registry, indexed_stems, dry_run=args.dry_run)


def run_single_query(query_text: str, top_k: int = 10):
    """Executes a single test query against the stored Vector DB and prints output."""
    from rag_chatbot_engine import RAGChatbot

    print(f"\n🔍 Querying Stored VDB with: '{query_text}' (top_k={top_k})...\n")
    chatbot = RAGChatbot()
    res = chatbot.answer_question(query=query_text, top_k=top_k)

    print("=" * 65)
    print(f"📌 Query:             {res.get('query')}")
    print(f"🌐 Detected Accent:   {res.get('detected_accent')}")
    print(f"⚡ Time Taken:        {res.get('time_taken')} seconds")
    print(f"🔌 Ollama Connected:  {res.get('llm_connected')}")
    print("=" * 65)
    print(f"\n🤖 Answer:\n{res.get('answer')}\n")
    
    sources = res.get("sources", [])
    print("=" * 65)
    print(f"📚 Retrieved Sources ({len(sources)}):")
    print("=" * 65)
    for src in sources:
        print(f"  [{src.get('rank')}] {src.get('title')} ({src.get('similarity_score')}% match)")
        text_snippet = src.get('text', '').replace('\n', ' ')[:150]
        print(f"      Snippet: {text_snippet}...\n")


def run_interactive_test():
    """Launches an interactive CLI terminal session to test chatbot queries against stored VDB."""
    from rag_chatbot_engine import RAGChatbot

    print("\n" + "=" * 65)
    print(" 🤖 GUILD KNOWLEDGE BASE — INTERACTIVE CLI TESTER")
    print("  Type your questions below to test the RAG engine against stored VDB.")
    print("  Type 'exit' or 'quit' to stop.")
    print("=" * 65 + "\n")

    chatbot = RAGChatbot()
    history = []

    while True:
        try:
            user_input = input("\n💬 Enter question (or 'exit'): ").strip()
            if not user_input:
                continue
            if user_input.lower() in ["exit", "quit", "q"]:
                print("Exiting interactive testing mode. Goodbye!")
                break

            res = chatbot.answer_question(query=user_input, history=history, top_k=10)
            
            history.append({"role": "user", "content": user_input})
            history.append({"role": "assistant", "content": res.get("answer", "")})

            print("\n" + "-" * 60)
            print(f"⚡ Response Time: {res.get('time_taken')}s | Ollama Status: {'Connected' if res.get('llm_connected') else 'Not Connected'}")
            print("-" * 60)
            print(f"🤖 Answer:\n{res.get('answer')}\n")

            sources = res.get("sources", [])
            print(f"📚 Top Sources ({len(sources)}):")
            for src in sources[:5]:
                print(f"  - [{src.get('rank')}] {src.get('title')} ({src.get('similarity_score')}% similarity)")
            print("-" * 60)

        except (KeyboardInterrupt, EOFError):
            print("\nExiting interactive mode.")
            break


def run_serve(args):
    """Launches the FastAPI web chatbot server."""
    import uvicorn
    import socket

    def is_port_in_use(port: int, host: str = "127.0.0.1") -> bool:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex((host, port)) == 0

    target_port = args.port
    if is_port_in_use(target_port):
        for alt_port in [8080, 8000, 8501, 5000, 8081]:
            if not is_port_in_use(alt_port):
                print(f"ℹ️ Port {target_port} is occupied. Using alternative port {alt_port}.")
                target_port = alt_port
                break

    print(f"\n🚀 Starting Guild Knowledge Base RAG Web Server on http://0.0.0.0:{target_port} ...")
    uvicorn.run("app:app", host="0.0.0.0", port=target_port, reload=args.reload)


def main():
    parser = argparse.ArgumentParser(description="Guild Knowledge Base Unified Multi-Mode CLI Router")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Ingest subcommand
    ingest_parser = subparsers.add_parser("ingest", help="Run batch ingestion (parse PDFs -> chunk -> embed -> store VDB)")
    ingest_parser.add_argument("--output-dir", type=str, default="output_dir", help="Path to pre-parsed Markdown output directory")
    ingest_parser.add_argument("--pdfs-dir", type=str, default="pdfs", help="Path to input PDFs directory")
    ingest_parser.add_argument("--index-only", action="store_true", help="Index pre-parsed Markdown files only (no GPU VLM PDF parsing)")
    ingest_parser.add_argument("--parse-only", action="store_true", help="Run GPU PDF parsing only")
    ingest_parser.add_argument("--no-reset-db", action="store_true", help="Do not reset vector database before indexing")
    ingest_parser.add_argument("--dry-run", action="store_true", help="Scan files and print plan without modifying VDB")

    # Test subcommand
    test_parser = subparsers.add_parser("test", help="Launch interactive CLI chatbot test session against stored VDB")

    # Query subcommand
    query_parser = subparsers.add_parser("query", help="Run a single question query against stored VDB")
    query_parser.add_argument("text", type=str, help="Question text to query")
    query_parser.add_argument("--top-k", type=int, default=10, help="Number of sources to retrieve")

    # Serve subcommand
    serve_parser = subparsers.add_parser("serve", help="Launch FastAPI web chatbot UI and API server")
    serve_parser.add_argument("--port", type=int, default=8080, help="Server port number")
    serve_parser.add_argument("--reload", action="store_true", help="Enable uvicorn live reload")

    args = parser.parse_args()

    if args.command == "ingest":
        run_ingest(args)
    elif args.command == "test":
        run_interactive_test()
    elif args.command == "query":
        run_single_query(args.text, top_k=args.top_k)
    elif args.command == "serve":
        run_serve(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
