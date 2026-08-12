import os
import sys
import shutil
from pathlib import Path

# Force UTF-8 encoding on Windows console
if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from core.config import MARKDOWNS_DIR, OUTPUT_DIR, PDFS_DIR, STORAGE_DIR

def consolidate():
    print("=" * 65)
    print(" 📦 CONSOLIDATING DATA FOLDER ARCHITECTURE")
    print("=" * 65)

    MARKDOWNS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PDFS_DIR.mkdir(parents=True, exist_ok=True)

    legacy_parsed_dir = STORAGE_DIR / "pdf_parsed_results"
    
    # 1. Consolidate folders from legacy storage/pdf_parsed_results to data/output_dir
    if legacy_parsed_dir.exists():
        print(f"\n1. Migrating legacy parsed results from '{legacy_parsed_dir}' to '{OUTPUT_DIR}'...")
        migrated_folders = 0
        for item in legacy_parsed_dir.iterdir():
            if item.is_dir():
                dest_dir = OUTPUT_DIR / item.name
                if not dest_dir.exists():
                    shutil.copytree(item, dest_dir)
                    migrated_folders += 1
                else:
                    # Merge files inside
                    for sub in item.rglob("*"):
                        if sub.is_file():
                            rel = sub.relative_to(item)
                            target_file = dest_dir / rel
                            target_file.parent.mkdir(parents=True, exist_ok=True)
                            if not target_file.exists():
                                shutil.copy2(sub, target_file)
        print(f"   ✔ Migrated/merged {migrated_folders} output folders.")

    # 2. Consolidate main parsed markdown files into data/markdowns/
    print(f"\n2. Copying main parsed Markdown files into '{MARKDOWNS_DIR}'...")
    markdown_files = list(OUTPUT_DIR.rglob("*_parsed.md"))
    if not markdown_files:
        markdown_files = [f for f in OUTPUT_DIR.rglob("*.md") if not f.name.startswith("page_")]

    copied_md_count = 0
    for md_file in markdown_files:
        dest_md = MARKDOWNS_DIR / md_file.name
        shutil.copy2(md_file, dest_md)
        copied_md_count += 1

    print(f"   ✔ Copied {copied_md_count} main parsed Markdown files into '{MARKDOWNS_DIR}'.")

    # 3. Ensure each processed folder in data/output_dir has a copy of its source PDF if available
    print(f"\n3. Linking source PDFs into self-contained output folders...")
    pdf_copies = 0
    for pdf_file in PDFS_DIR.glob("*.pdf"):
        pdf_folder = OUTPUT_DIR / pdf_file.stem
        if pdf_folder.exists() and pdf_folder.is_dir():
            target_pdf = pdf_folder / pdf_file.name
            if not target_pdf.exists():
                shutil.copy2(pdf_file, target_pdf)
                pdf_copies += 1

    print(f"   ✔ Copied {pdf_copies} source PDFs into corresponding '{OUTPUT_DIR}' folders.")
    
    print("\n" + "=" * 65)
    print(" 🎉 DATA CONSOLIDATION COMPLETE")
    print("=" * 65 + "\n")

if __name__ == "__main__":
    consolidate()
