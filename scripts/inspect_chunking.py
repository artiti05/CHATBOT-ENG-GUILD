import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any

# Configure UTF-8 encoding for Windows terminals
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Ensure project root is in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from core.ingestion import TextChunker, clean_document_text
from core.config import (
    MARKDOWNS_DIR, TEXTS_DIR, OUTPUT_DIR, CHUNKING_LOGS_DIR,
    PARENT_CHUNK_TOKENS, CHILD_CHUNK_TOKENS
)


def list_available_parsed_files() -> List[Path]:
    """Finds all parsed text and markdown files in markdowns, texts, and output directories."""
    files = []
    for d in (MARKDOWNS_DIR, TEXTS_DIR, OUTPUT_DIR):
        if d.exists():
            files.extend(list(d.rglob("*.txt")) + list(d.rglob("*.md")))
    # Filter out individual page markdown files
    files = [f for f in files if not f.name.startswith("page_")]
    # Unique by resolve
    unique_map = {f.resolve(): f for f in files}
    return sorted(list(unique_map.values()), key=lambda p: p.name)


def inspect_file_chunking(file_path: Path):
    """
    Reads a parsed text file, runs Parent-Child token chunking,
    and displays a step-by-step breakdown of how the file was chunked.
    """
    file_path = Path(file_path).resolve()
    if not file_path.exists():
        print(f"❌ Error: File not found: {file_path}")
        return

    print("=" * 80)
    print(f"📄 CHUNKING INSPECTOR — FILE: {file_path.name}")
    print(f"📍 Full Path: {file_path}")
    print("=" * 80)

    try:
        raw_text = file_path.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        print(f"❌ Error reading file: {e}")
        return

    cleaned_text = clean_document_text(raw_text)
    total_words = len(cleaned_text.split())
    total_chars = len(cleaned_text)

    print(f"\n📊 Document Overview:")
    print(f"  • Total Words: {total_words:,}")
    print(f"  • Total Characters: {total_chars:,}")
    print(f"  • Settings: Parent Block = {PARENT_CHUNK_TOKENS} tokens | Child Chunk = {CHILD_CHUNK_TOKENS} tokens")

    chunker = TextChunker()

    doc_dict = {
        "source_id": file_path.stem,
        "file_name": file_path.name,
        "source_type": file_path.suffix.lstrip("."),
        "full_text": cleaned_text
    }

    chunks = chunker.chunk_text(doc_dict)
    print(f"\n📦 Generated Child Chunks ({len(chunks)} chunks total):")
    print("-" * 80)

    for i, c in enumerate(chunks, 1):
        cid = c.get("chunk_id", f"chunk_{i}")
        pid = c.get("parent_id", "N/A")
        words = c.get("word_count", 0)
        tokens = c.get("estimated_tokens", 0)
        text = c.get("text", "")
        p_text = c.get("parent_text", "")

        print(f"\n🔹 [CHILD CHUNK #{i}] ID: {cid}")
        print(f"   • Linked Parent Block ID: {pid}")
        print(f"   • Metrics: {words} words | ~{tokens} tokens | {len(text)} characters")
        print(f"   • Child Text Preview:")
        child_preview = text[:200].replace("\n", " ")
        print(f"     \"{child_preview}...\"")
        print(f"   • Parent Block Context Preview:")
        parent_preview = p_text[:250].replace("\n", " ")
        print(f"     \"{parent_preview}...\"")
        print("-" * 80)

    # Save Inspection Report Artifact to data/chunking_logs/
    CHUNKING_LOGS_DIR.mkdir(parents=True, exist_ok=True)
    report_file = CHUNKING_LOGS_DIR / f"{file_path.stem}_chunk_report.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump({
            "file_name": file_path.name,
            "total_words": total_words,
            "total_child_chunks": len(chunks),
            "parent_tokens_target": PARENT_CHUNK_TOKENS,
            "child_tokens_target": CHILD_CHUNK_TOKENS,
            "chunks": chunks
        }, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Inspection report saved to: {report_file}\n")


def interactive_menu():
    """Interactive mode to pick specific files for chunking inspection."""
    files = list_available_parsed_files()
    if not files:
        print("ℹ️ No parsed text files found in 'data/markdowns/', 'data/texts/', or 'data/output_dir/'.")
        path_in = input("Enter custom file path to inspect: ").strip()
        if path_in:
            inspect_file_chunking(Path(path_in))
        return

    print("\n📚 Available Files for Chunking Inspection:")
    for idx, f in enumerate(files, 1):
        print(f"  [{idx}] {f.name} ({f.parent.name})")

    choice = input(f"\nSelect file number (1-{len(files)}) or enter custom path: ").strip()
    if choice.isdigit() and 1 <= int(choice) <= len(files):
        inspect_file_chunking(files[int(choice) - 1])
    elif choice:
        inspect_file_chunking(Path(choice))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visual Document Chunking Inspector Tool")
    parser.add_argument("--file", "-f", type=str, help="Path to text or markdown file to inspect")

    args = parser.parse_args()

    if args.file:
        inspect_file_chunking(Path(args.file))
    else:
        interactive_menu()
