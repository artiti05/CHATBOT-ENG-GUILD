"""
train_arat5.py
==============

Fine-tunes AraT5 (UBC-NLP/AraT5-base) on filtered JODA dataset (data_msa/train.jsonl).
Target column is MSA text WITHOUT diacritics / tashkeel.

Optimized for RTX 4060 GPU using bfloat16 to avoid float16 T5 NaN issues.
"""

import json
import os
import sys
from pathlib import Path

import torch
from datasets import Dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

# Prevent Intel OMP library duplication crash on Windows
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_NAME = "UBC-NLP/AraT5-base"
TRAIN_FILE = SCRIPT_DIR / "data_msa" / "train.jsonl"
VALID_FILE = SCRIPT_DIR / "data_msa" / "valid.jsonl"
OUTPUT_DIR = SCRIPT_DIR / "models" / "arat5-dialect-msa"

MAX_INPUT_LENGTH = 64
MAX_TARGET_LENGTH = 64


def load_jsonl(path: Path) -> list:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def main():
    print("=" * 60)
    print("AraT5 Fine-Tuning Script")
    print("=" * 60)
    print(f"Base model    : {MODEL_NAME}")
    print(f"Train data    : {TRAIN_FILE}")
    print(f"Valid data    : {VALID_FILE}")
    print(f"Output folder : {OUTPUT_DIR.resolve()}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Device Name   : {torch.cuda.get_device_name(0)}")

    if not TRAIN_FILE.exists() or not VALID_FILE.exists():
        sys.exit("ERROR: data_msa files not found. Run prepare_joda.py first.")

    train_data = load_jsonl(TRAIN_FILE)
    valid_data = load_jsonl(VALID_FILE)

    print(f"\nLoaded {len(train_data):,} training samples and {len(valid_data):,} validation samples.")

    # Convert to Hugging Face Dataset
    train_ds = Dataset.from_list(train_data)
    valid_ds = Dataset.from_list(valid_data)

    print(f"Loading Tokenizer ({MODEL_NAME})...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)

    print(f"Loading Model ({MODEL_NAME})...")
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

    def preprocess_function(examples):
        inputs = examples["input"]
        targets = examples["target"]

        model_inputs = tokenizer(
            inputs,
            max_length=MAX_INPUT_LENGTH,
            truncation=True,
            padding=False,
        )
        labels = tokenizer(
            text_target=targets,
            max_length=MAX_TARGET_LENGTH,
            truncation=True,
            padding=False,
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    print("Tokenizing datasets...")
    train_tokenized = train_ds.map(
        preprocess_function,
        batched=True,
        remove_columns=["input", "target"],
    )
    valid_tokenized = valid_ds.map(
        preprocess_function,
        batched=True,
        remove_columns=["input", "target"],
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
    )

    # Detect bfloat16 capability (supported on RTX 4060 Ampere/Ada Lovelace)
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(OUTPUT_DIR),
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=3e-4,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        weight_decay=0.01,
        save_total_limit=2,
        num_train_epochs=3,
        predict_with_generate=True,
        fp16=False,
        bf16=use_bf16,
        logging_steps=100,
        report_to="none",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=valid_tokenized,
        processing_class=tokenizer,
        data_collator=data_collator,
    )

    print("\nStarting training...")
    trainer.train()

    print(f"\nSaving fine-tuned model and tokenizer to {OUTPUT_DIR}...")
    trainer.save_model(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))

    print("\n" + "=" * 60)
    print(" AraT5 Fine-Tuning Complete!")
    print(f" Saved model at: {OUTPUT_DIR.resolve()}")
    print("=" * 60)


if __name__ == "__main__":
    main()
