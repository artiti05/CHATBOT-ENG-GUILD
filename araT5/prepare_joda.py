"""
prepare_joda.py
===============

Filters the JODA dataset (Abandah et al., 2025) into training files for a
dialect-to-MSA rewriter.

Input  : diacritized_train_set.xlsx / _valid_set.xlsx / _test_set.xlsx
Output : train.jsonl / valid.jsonl / test.jsonl
         each line: {"input": "<prefix><dialect>", "target": "<msa>"}

JODA columns:
  Source          where the sentence came from
  Text            the original sentence (dialect, or erroneous MSA)
  Type            0 = Jordanian dialect,  1 = erroneous MSA
  Corrected Text  the MSA version, NO diacritics   <-- we use this as target
  Diacritized Text the MSA version WITH diacritics <-- we IGNORE this

Why we ignore the diacritized column:
  Your documents in ChromaDB are not diacritized, and your query pipeline
  strips diacritics anyway. Producing diacritized text would put the query
  in a different character space than the index and make retrieval worse,
  while costing extra decoding time.

Usage:
    pip install pandas openpyxl
    python prepare_joda.py --input-dir ./joda --output-dir ./data_msa

    # keep the MSA-error-correction rows too (default is dialect only):
    python prepare_joda.py --input-dir ./joda --output-dir ./data_msa --keep-type1
"""

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd

# The task prefix. T5 models need a prefix telling them which task to do.
# CRITICAL: you must use this EXACT same string at inference time.
# A mismatch between training and serving is a common cause of a good
# fine-tune producing bad output in production.
PREFIX = "حول إلى الفصحى: "

MIN_WORDS = 2      # 1-word rows teach nothing
MAX_WORDS = 40     # long rows bias the model toward long, slow outputs

FILES = {
    "train": "diacritized_train_set.xlsx",
    "valid": "diacritized_valid_set.xlsx",
    "test": "diacritized_test_set.xlsx",
}


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

TASHKEEL = re.compile(r"[\u0617-\u061A\u064B-\u0652]")
TATWEEL = re.compile(r"\u0640")


def clean(text) -> str:
    """Light cleanup only. We do NOT fold Arabic letters here.

    Folding (ة->ه, ى->ي) would put the text in a spelling convention the
    model was never pretrained on. Keep the original orthography; apply any
    folding AFTER the model, not before it.
    """
    if not isinstance(text, str):
        return ""
    text = text.replace("\u200f", "").replace("\u200e", "")  # RTL/LTR marks
    text = TATWEEL.sub("", text)                              # kashida
    text = re.sub(r"\s+", " ", text)                          # collapse spaces
    return text.strip()


def dedup_key(text: str) -> str:
    """Aggressive normalization used ONLY to spot near-duplicates.

    This value is never written to the output files.
    """
    t = TASHKEEL.sub("", text)
    t = re.sub(r"[أإآ]", "ا", t)
    t = t.replace("ى", "ي").replace("ة", "ه")
    t = re.sub(r"[^\w\s]", "", t)
    return re.sub(r"\s+", " ", t).strip().lower()


def word_count(text: str) -> int:
    return len(text.split())


# --------------------------------------------------------------------------
# main filtering
# --------------------------------------------------------------------------

def filter_split(df: pd.DataFrame, name: str, keep_type1: bool) -> pd.DataFrame:
    """Apply every filter in order, printing how many rows each one drops."""
    report = []

    def note(step, frame):
        report.append((step, len(frame)))
        return frame

    note("loaded", df)

    # --- required columns -------------------------------------------------
    needed = {"Text", "Corrected Text", "Type"}
    missing = needed - set(df.columns)
    if missing:
        sys.exit(
            f"\nERROR in {name}: missing column(s) {sorted(missing)}.\n"
            f"Found columns: {list(df.columns)}\n"
            f"Open the xlsx and check the header row spelling."
        )

    df = df[["Text", "Corrected Text", "Type"]].copy()

    # --- 1. Type filter ---------------------------------------------------
    # Type 0 = Jordanian dialect  -> what we want to learn
    # Type 1 = erroneous MSA      -> spelling/grammar correction
    #
    # Default: dialect only, because that is the task. Pass --keep-type1 to
    # include the correction rows as well (defensible: your users make typos
    # too, and it acts as extra regularization). Whichever you pick, say so
    # in your writeup.
    df["Type"] = pd.to_numeric(df["Type"], errors="coerce")
    if keep_type1:
        df = df[df["Type"].isin([0, 1])]
    else:
        df = df[df["Type"] == 0]
    note("after Type filter", df)

    # --- 2. clean text ----------------------------------------------------
    df["input_text"] = df["Text"].map(clean)
    df["target_text"] = df["Corrected Text"].map(clean)

    # --- 3. drop empties --------------------------------------------------
    df = df[(df["input_text"] != "") & (df["target_text"] != "")]
    note("after dropping empty", df)

    # --- 4. drop copy examples -------------------------------------------
    # Rows where input == output teach the model to echo its input. That is
    # exactly the failure mode you least want from a rewriter.
    same = df["input_text"].map(dedup_key) == df["target_text"].map(dedup_key)
    df = df[~same]
    note("after dropping input==output", df)

    # --- 5. length filter -------------------------------------------------
    # Your real inputs are short questions. Training on long sentences biases
    # the model toward long outputs, which costs decoding time per request.
    in_wc = df["input_text"].map(word_count)
    tgt_wc = df["target_text"].map(word_count)
    df = df[
        in_wc.between(MIN_WORDS, MAX_WORDS)
        & tgt_wc.between(MIN_WORDS, MAX_WORDS)
    ]
    note(f"after length filter ({MIN_WORDS}-{MAX_WORDS} words)", df)

    # --- 6. drop wild length ratios --------------------------------------
    # If the MSA version is 4x longer or shorter than the dialect version,
    # the pair is probably misaligned rather than a translation.
    ratio = df["target_text"].map(word_count) / df["input_text"].map(word_count)
    df = df[ratio.between(0.4, 2.5)]
    note("after length-ratio filter", df)

    # --- 7. deduplicate ---------------------------------------------------
    # Social media corpora always contain near-duplicates. Leaving them in
    # over-weights whatever phrase happened to go viral.
    df["_key"] = df["input_text"].map(dedup_key)
    df = df.drop_duplicates(subset="_key", keep="first").drop(columns="_key")
    note("after dedup", df)

    # --- print the report -------------------------------------------------
    print(f"\n  {name}")
    print(f"  {'-' * 50}")
    prev = None
    for step, count in report:
        if prev is None:
            print(f"  {step:<42} {count:>7,}")
        else:
            print(f"  {step:<42} {count:>7,}  (-{prev - count:,})")
        prev = count

    return df


def write_jsonl(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            rec = {
                "input": PREFIX + row["input_text"],
                "target": row["target_text"],
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser(description="Filter JODA for dialect->MSA training")
    ap.add_argument("--input-dir", required=True,
                    help="folder containing the three JODA .xlsx files")
    ap.add_argument("--output-dir", required=True,
                    help="where to write train/valid/test .jsonl")
    ap.add_argument("--keep-type1", action="store_true",
                    help="also keep erroneous-MSA correction rows (Type=1)")
    args = ap.parse_args()

    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)

    print("=" * 60)
    print("JODA preparation")
    print("=" * 60)
    print("  target column : 'Corrected Text' (NOT diacritized)")
    print(f"  Type kept     : {'0 and 1' if args.keep_type1 else '0 only (dialect)'}")
    print(f"  task prefix   : {PREFIX!r}")

    totals = {}
    for split, filename in FILES.items():
        path = in_dir / filename
        if not path.exists():
            sys.exit(f"\nERROR: {path} not found. Files in {in_dir}: "
                     f"{[p.name for p in in_dir.glob('*')]}")
        df = pd.read_excel(path)
        kept = filter_split(df, split, args.keep_type1)
        write_jsonl(kept, out_dir / f"{split}.jsonl")
        totals[split] = len(kept)

    print("\n" + "=" * 60)
    print("Written to", out_dir.resolve())
    for split, n in totals.items():
        print(f"  {split + '.jsonl':<16} {n:>7,} pairs")
    print("=" * 60)
    print("""
NEXT:
  1. Open train.jsonl and read 20 lines. If the pairs do not look like
     dialect -> MSA, stop and fix this before spending GPU time.
  2. Keep your own 300 hand-written JEA pairs in a SEPARATE file. You will
     train on JODA first, then on those, in that order.
  3. Use the prefix above at inference time, character for character.
""")


if __name__ == "__main__":
    main()
