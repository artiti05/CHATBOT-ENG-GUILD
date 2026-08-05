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
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from ingestion_pipeline import TextChunker, clean_document_text
from config.settings import (
    PARSED_OUTPUT_DIR, TEXTS_DIR, TARGET_CHUNK_WORDS, MIN_CHUNK_WORDS, MAX_CHUNK_WORDS, OVERLAP_SENTENCES
)


def list_available_parsed_files() -> List[Path]:
    """Finds all parsed text and markdown files in the storage and texts directories."""
    files = []
    if PARSED_OUTPUT_DIR.exists():
        files.extend(list(PARSED_OUTPUT_DIR.glob("*.txt")) + list(PARSED_OUTPUT_DIR.glob("*.md")))
    if TEXTS_DIR.exists():
        files.extend(list(TEXTS_DIR.glob("*.txt")) + list(TEXTS_DIR.glob("*.md")))
    return sorted(files)


def inspect_file_chunking(file_path: Path, target_words: int = TARGET_CHUNK_WORDS, min_words: int = MIN_CHUNK_WORDS, max_words: int = MAX_CHUNK_WORDS):
    """
    Reads a parsed text file, parses document blocks, runs semantic chunking,
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
        raw_text = file_path.read_text(encoding="utf-8")
    except Exception as e:
        print(f"❌ Error reading file: {e}")
        return

    cleaned_text = clean_document_text(raw_text)
    total_words = len(cleaned_text.split())
    total_chars = len(cleaned_text)

    print(f"\n📊 Document Overview:")
    print(f"  • Total Words: {total_words:,}")
    print(f"  • Total Characters: {total_chars:,}")
    print(f"  • Settings: Target={target_words} words | Min={min_words} words | Max={max_words} words | Overlap={OVERLAP_SENTENCES} sentences")

    chunker = TextChunker(
        target_chunk_words=target_words,
        min_chunk_words=min_words,
        max_chunk_words=max_words,
        overlap_sentences=OVERLAP_SENTENCES
    )

    # Step 1: Parse Document Blocks
    blocks = chunker.parse_document_blocks(cleaned_text)
    print(f"\n🧩 Document Block Breakdown ({len(blocks)} structural blocks identified):")
    block_counts = {}
    for b in blocks:
        b_type = b["type"]
        block_counts[b_type] = block_counts.get(b_type, 0) + 1

    for b_type, count in block_counts.items():
        print(f"  • {b_type.capitalize()} Blocks: {count}")

    # Step 2: Generate Semantic Chunks
    doc_dict = {
        "source_id": file_path.stem,
        "file_name": file_path.name,
        "source_type": file_path.suffix.lstrip("."),
        "full_text": cleaned_text
    }

    chunks = chunker.chunk_text(doc_dict)
    print(f"\n📦 Generated Chunks ({len(chunks)} chunks total):")
    print("-" * 80)

    for i, c in enumerate(chunks, 1):
        cid = c.get("chunk_id", f"chunk_{i}")
        sec = c.get("section", "General")
        subsec = c.get("subsection", "")
        p_start = c.get("page_start", 1)
        p_end = c.get("page_end", 1)
        words = c.get("word_count", 0)
        sentences = c.get("sentence_count", 0)
        prev_id = c.get("previous_chunk", "None") or "None"
        next_id = c.get("next_chunk", "None") or "None"
        text = c.get("text", "")

        sec_str = f"{sec}" + (f" -> {subsec}" if subsec else "")

        print(f"\n🔹 [CHUNK #{i}] ID: {cid}")
        print(f"   • Section: {sec_str}")
        print(f"   • Page Range: Page {p_start} to Page {p_end}")
        print(f"   • Metrics: {words} words | {sentences} sentences | {len(text)} characters")
        print(f"   • Linked Neighbors: Previous = '{prev_id}' | Next = '{next_id}'")
        print(f"   • Text Preview (Start):")
        start_preview = text[:250].replace("\n", " ")
        print(f"     \"{start_preview}...\"")
        print(f"   • Text Preview (End / Overlap Context):")
        end_preview = text[-200:].replace("\n", " ")
        print(f"     \"...{end_preview}\"")
        print("-" * 80)

    # Save Inspection Artifact
    reports_dir = BASE_DIR / "storage" / "chunk_inspection_reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_file = reports_dir / f"{file_path.stem}_chunk_report.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump({
            "file_name": file_path.name,
            "total_words": total_words,
            "total_blocks": len(blocks),
            "total_chunks": len(chunks),
            "chunks": chunks
        }, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Inspection report saved to: {report_file}\n")


def interactive_menu():
    """Interactive mode to pick specific files for chunking inspection."""
    files = list_available_parsed_files()
    if not files:
        print("ℹ️ No parsed text files found in 'storage/pdf_parsed_results/' or 'texts/'.")
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
    parser.add_argument("--file", "-f", type=str, help="Path to parsed text or markdown file to inspect")
    parser.add_argument("--target-words", type=int, default=TARGET_CHUNK_WORDS, help="Target chunk word count")
    parser.add_argument("--min-words", type=int, default=MIN_CHUNK_WORDS, help="Minimum chunk word count")
    parser.add_argument("--max-words", type=int, default=MAX_CHUNK_WORDS, help="Maximum chunk word count")

    args = parser.parse_args()

    if args.file:
        inspect_file_chunking(Path(args.file), target_words=args.target_words, min_words=args.min_words, max_words=args.max_words)
    else:
        interactive_menu()
