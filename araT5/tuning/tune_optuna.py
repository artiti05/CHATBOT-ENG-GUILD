"""
tune_optuna.py
==============

Automated Hyperparameter Optimization (HPO) for AraT5 dialect-to-MSA fine-tuning
using Optuna and Hugging Face Seq2SeqTrainer.

Features:
- Dedicated self-contained tuning folder with trial logs, evaluation traces, and reports.
- Anti-overfitting objective: balances minimal validation loss with minimal generalization gap (eval_loss - train_loss).
- Label smoothing & weight decay regularization search.
- Early stopping to abort diverging/overfitting trials early.
- Unified CSV/JSON logging of all training and evaluation metrics in the same directory.
- Automated export and deployment of the best model with the least overfitting.
"""

import argparse
import csv
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# Prevent Intel OMP library duplication crash on Windows
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import optuna
from optuna.trial import Trial
import torch
from datasets import Dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    TrainerCallback,
)

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DEFAULT_DATA_DIR = BASE_DIR / "araT5" / "data_msa"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent
DEFAULT_BASE_MODEL = "UBC-NLP/AraT5-base"
PIPELINE_MODEL_DIR = BASE_DIR / "araT5" / "models" / "arat5-dialect-msa"

MAX_INPUT_LENGTH = 64
MAX_TARGET_LENGTH = 64


def load_jsonl(path: Path) -> List[Dict[str, str]]:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


class EpochMetricsLoggerCallback(TrainerCallback):
    """
    Logs epoch-level train_loss, eval_loss, and the overfitting gap (eval_loss - train_loss)
    to a centralized CSV file inside the tuning directory.
    """

    def __init__(self, trial_id: int, log_csv_path: Path):
        self.trial_id = trial_id
        self.log_csv_path = log_csv_path
        self.current_train_loss: Optional[float] = None
        self.epoch_history: List[Dict[str, Any]] = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            self.current_train_loss = logs["loss"]

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics and "eval_loss" in metrics:
            eval_loss = metrics["eval_loss"]
            epoch = metrics.get("epoch", state.epoch if state else 0.0)
            train_loss = self.current_train_loss if self.current_train_loss is not None else float("nan")
            
            gap = (eval_loss - train_loss) if (train_loss is not None and not (train_loss != train_loss)) else 0.0
            ratio = (eval_loss / train_loss) if (train_loss and train_loss > 0) else 1.0

            entry = {
                "trial_id": self.trial_id,
                "epoch": round(float(epoch), 2),
                "step": state.global_step,
                "train_loss": round(float(train_loss), 5) if train_loss == train_loss else "N/A",
                "eval_loss": round(float(eval_loss), 5),
                "overfitting_gap": round(float(gap), 5),
                "overfitting_ratio": round(float(ratio), 4),
            }
            self.epoch_history.append(entry)

            file_exists = self.log_csv_path.exists()
            with open(self.log_csv_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "trial_id",
                        "epoch",
                        "step",
                        "train_loss",
                        "eval_loss",
                        "overfitting_gap",
                        "overfitting_ratio",
                    ],
                )
                if not file_exists:
                    writer.writeheader()
                writer.writerow(entry)


def tokenize_datasets(train_data: List[Dict[str, str]], valid_data: List[Dict[str, str]], tokenizer):
    train_ds = Dataset.from_list(train_data)
    valid_ds = Dataset.from_list(valid_data)

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
    return train_tokenized, valid_tokenized


def run_tuning(
    n_trials: int,
    base_model_name: str,
    data_dir: Path,
    output_dir: Path,
    overfitting_penalty: float = 0.3,
    max_train_samples: Optional[int] = None,
    max_valid_samples: Optional[int] = None,
    deploy_to_pipeline: bool = False,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = output_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    trials_dir = output_dir / "trial_checkpoints"
    trials_dir.mkdir(parents=True, exist_ok=True)
    best_model_dir = output_dir / "best_model"

    log_csv_path = logs_dir / "training_eval_log.csv"
    study_db_path = f"sqlite:///{logs_dir / 'optuna_study.db'}"
    trials_json_path = logs_dir / "trials_summary.json"

    train_file = data_dir / "train.jsonl"
    valid_file = data_dir / "valid.jsonl"

    if not train_file.exists() or not valid_file.exists():
        sys.exit(f"ERROR: Dataset files not found in {data_dir}. Run prepare_joda.py first.")

    print("=" * 70)
    print("  AraT5 Automated Hyperparameter Optimization (Optuna)")
    print("=" * 70)
    print(f"Base Model          : {base_model_name}")
    print(f"Data Directory      : {data_dir.resolve()}")
    print(f"Output Directory    : {output_dir.resolve()}")
    print(f"Logs Directory      : {logs_dir.resolve()}")
    print(f"Trials to Execute   : {n_trials}")
    print(f"Overfit Penalty (α) : {overfitting_penalty}")
    print(f"CUDA Available      : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU Device Name     : {torch.cuda.get_device_name(0)}")

    train_data = load_jsonl(train_file)
    valid_data = load_jsonl(valid_file)

    if max_train_samples and len(train_data) > max_train_samples:
        train_data = train_data[:max_train_samples]
    if max_valid_samples and len(valid_data) > max_valid_samples:
        valid_data = valid_data[:max_valid_samples]

    print(f"\nActive Samples -> Train: {len(train_data):,}, Validation: {len(valid_data):,}")

    tokenizer = AutoTokenizer.from_pretrained(base_model_name, use_fast=False)
    train_tokenized, valid_tokenized = tokenize_datasets(train_data, valid_data, tokenizer)

    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()

    # Track trials metadata for final summary
    all_trials_records: List[Dict[str, Any]] = []

    def objective(trial: Trial) -> float:
        trial_id = trial.number
        print(f"\n>>> [Trial {trial_id + 1}/{n_trials}] Sampled Hyperparameters:")

        # User-customized discrete search space (96 total combinations)
        lr = trial.suggest_categorical("learning_rate", [5e-5, 1e-4, 2e-4])
        batch_size = trial.suggest_categorical("per_device_train_batch_size", [8, 16])
        grad_accum = trial.suggest_categorical("gradient_accumulation_steps", [1, 2])
        epochs = 4  # Fixed to 4 epochs per requirement
        weight_decay = trial.suggest_categorical("weight_decay", [0.001, 0.01])
        warmup_ratio = trial.suggest_categorical("warmup_ratio", [0.05, 0.10])
        label_smoothing = trial.suggest_categorical("label_smoothing_factor", [0.0, 0.05])

        print(f"    - learning_rate             : {lr:.2e}")
        print(f"    - batch_size                : {batch_size}")
        print(f"    - gradient_accumulation     : {grad_accum} (effective batch: {batch_size * grad_accum})")
        print(f"    - num_train_epochs          : {epochs}")
        print(f"    - weight_decay              : {weight_decay}")
        print(f"    - warmup_ratio              : {warmup_ratio}")
        print(f"    - label_smoothing_factor    : {label_smoothing}")

        trial_output_dir = trials_dir / f"trial_{trial_id}"
        trial_output_dir.mkdir(parents=True, exist_ok=True)

        model = AutoModelForSeq2SeqLM.from_pretrained(base_model_name)

        training_args = Seq2SeqTrainingArguments(
            output_dir=str(trial_output_dir),
            eval_strategy="epoch",
            save_strategy="epoch",
            learning_rate=lr,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=16,
            gradient_accumulation_steps=grad_accum,
            weight_decay=weight_decay,
            warmup_ratio=warmup_ratio,
            label_smoothing_factor=label_smoothing,
            num_train_epochs=epochs,
            predict_with_generate=False,
            fp16=False,
            bf16=use_bf16,
            logging_steps=50,
            save_total_limit=1,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            report_to="none",
            disable_tqdm=False,
        )

        epoch_logger = EpochMetricsLoggerCallback(trial_id=trial_id, log_csv_path=log_csv_path)
        early_stop = EarlyStoppingCallback(early_stopping_patience=2, early_stopping_threshold=0.001)

        data_collator = DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            model=model,
            padding=True,
        )

        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=train_tokenized,
            eval_dataset=valid_tokenized,
            processing_class=tokenizer,
            data_collator=data_collator,
            callbacks=[epoch_logger, early_stop],
        )

        start_time = time.time()
        train_result = trainer.train()
        train_duration = time.time() - start_time

        eval_metrics = trainer.evaluate()
        final_eval_loss = float(eval_metrics.get("eval_loss", 999.0))
        final_train_loss = float(train_result.training_loss)

        # Generalization gap and anti-overfitting penalization
        overfitting_gap = max(0.0, final_eval_loss - final_train_loss)
        
        # Penalized objective: Min(eval_loss + alpha * max(0, eval_loss - train_loss))
        penalized_score = final_eval_loss + (overfitting_penalty * overfitting_gap)

        print(f"\n[Trial {trial_id + 1} Finished in {train_duration:.1f}s]")
        print(f"  -> Final Train Loss    : {final_train_loss:.5f}")
        print(f"  -> Final Eval Loss     : {final_eval_loss:.5f}")
        print(f"  -> Overfitting Gap (Δ) : {overfitting_gap:.5f}")
        print(f"  -> Penalized Objective : {penalized_score:.5f}")

        trial_record = {
            "trial_id": trial_id,
            "params": {
                "learning_rate": lr,
                "batch_size": batch_size,
                "weight_decay": weight_decay,
                "warmup_ratio": warmup_ratio,
                "epochs": epochs,
                "label_smoothing": label_smoothing,
                "grad_accum": grad_accum,
            },
            "train_loss": round(final_train_loss, 5),
            "eval_loss": round(final_eval_loss, 5),
            "overfitting_gap": round(overfitting_gap, 5),
            "penalized_score": round(penalized_score, 5),
            "duration_sec": round(train_duration, 2),
            "checkpoint_dir": str(trial_output_dir),
        }
        all_trials_records.append(trial_record)

        # Save trial summary incrementally
        with open(trials_json_path, "w", encoding="utf-8") as f:
            json.dump(all_trials_records, f, indent=2, ensure_ascii=False)

        # Report to Optuna
        trial.set_user_attr("train_loss", final_train_loss)
        trial.set_user_attr("eval_loss", final_eval_loss)
        trial.set_user_attr("overfitting_gap", overfitting_gap)
        trial.set_user_attr("penalized_score", penalized_score)
        trial.set_user_attr("checkpoint_dir", str(trial_output_dir))

        return penalized_score

    # Setup Optuna study with TPE (Tree-structured Parzen Estimator) sampler
    study = optuna.create_study(
        study_name="arat5_dialect_hp_tuning",
        storage=study_db_path,
        load_if_exists=True,
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42),
    )

    study.optimize(objective, n_trials=n_trials)

    # Sort all trials by penalized objective
    sorted_trials = sorted(all_trials_records, key=lambda x: x["penalized_score"])
    best_trial_info = sorted_trials[0]

    print("\n" + "=" * 70)
    print("  OPTIMIZATION COMPLETE — BEST MODEL SELECTION")
    print("=" * 70)
    print(f"🏆 Best Trial ID        : Trial {best_trial_info['trial_id'] + 1}")
    print(f"📊 Penalized Score     : {best_trial_info['penalized_score']}")
    print(f"📉 Validation Loss      : {best_trial_info['eval_loss']}")
    print(f"📈 Training Loss        : {best_trial_info['train_loss']}")
    print(f"⚖️ Overfitting Gap (Δ) : {best_trial_info['overfitting_gap']}")
    print(f"⚙️ Hyperparameters      :")
    for k, v in best_trial_info["params"].items():
        print(f"   - {k:<20}: {v}")

    # Export best model
    best_trial_ckpt = Path(best_trial_info["checkpoint_dir"])
    if best_model_dir.exists():
        shutil.rmtree(best_model_dir)
    best_model_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nExporting best model to: {best_model_dir.resolve()}...")
    best_model = AutoModelForSeq2SeqLM.from_pretrained(best_trial_ckpt)
    best_model.save_pretrained(str(best_model_dir))
    tokenizer.save_pretrained(str(best_model_dir))

    # Save best hyperparameters JSON
    best_hp_file = output_dir / "best_hyperparameters.json"
    with open(best_hp_file, "w", encoding="utf-8") as f:
        json.dump(best_trial_info, f, indent=2, ensure_ascii=False)

    # Generate Markdown Summary Report
    report_file = output_dir / "tuning_report.md"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write("# AraT5 Hyperparameter Optimization & Model Selection Report\n\n")
        f.write(f"- **Execution Date / Time**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"- **Base Model**: `{base_model_name}`\n")
        f.write(f"- **Total Trials**: `{n_trials}`\n")
        f.write(f"- **Overfitting Penalty Factor (α)**: `{overfitting_penalty}`\n")
        f.write(f"- **Objective Formula**: `Score = eval_loss + {overfitting_penalty} * max(0, eval_loss - train_loss)`\n\n")
        f.write("## 🏆 Selected Best Model\n\n")
        f.write(f"- **Trial ID**: Trial {best_trial_info['trial_id'] + 1}\n")
        f.write(f"- **Penalized Score**: `{best_trial_info['penalized_score']}`\n")
        f.write(f"- **Validation Loss**: `{best_trial_info['eval_loss']}`\n")
        f.write(f"- **Training Loss**: `{best_trial_info['train_loss']}`\n")
        f.write(f"- **Overfitting Gap (Δ)**: `{best_trial_info['overfitting_gap']}`\n\n")
        f.write("### Optimal Hyperparameters\n\n")
        f.write("| Hyperparameter | Value |\n")
        f.write("| :--- | :--- |\n")
        for k, v in best_trial_info["params"].items():
            f.write(f"| `{k}` | `{v}` |\n")
        f.write("\n## 📊 Leaderboard Across All Trials\n\n")
        f.write("| Rank | Trial | Penalized Score | Eval Loss | Train Loss | Gap (Δ) | Learning Rate | Batch Size | Epochs |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n")
        for rank, r in enumerate(sorted_trials, 1):
            f.write(
                f"| {rank} | Trial {r['trial_id'] + 1} | {r['penalized_score']} | {r['eval_loss']} | {r['train_loss']} | {r['overfitting_gap']} | {r['params']['learning_rate']:.2e} | {r['params']['batch_size']} | {r['params']['epochs']} |\n"
            )

    print(f"Saved best hyperparameters to: {best_hp_file.resolve()}")
    print(f"Saved tuning markdown report to: {report_file.resolve()}")

    # Deploy to pipeline if requested
    if deploy_to_pipeline:
        print(f"\nDeploying best model into main chatbot pipeline at: {PIPELINE_MODEL_DIR.resolve()}...")
        PIPELINE_MODEL_DIR.parent.mkdir(parents=True, exist_ok=True)
        if PIPELINE_MODEL_DIR.exists():
            shutil.rmtree(PIPELINE_MODEL_DIR)
        shutil.copytree(best_model_dir, PIPELINE_MODEL_DIR)
        print("✅ Pipeline model successfully updated!")

    print("\n" + "=" * 70)
    print(f" Tuning session finished successfully! All assets saved in: {output_dir.resolve()}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Automated Hyperparameter Optimization for AraT5")
    parser.add_argument("--n-trials", type=int, default=10, help="Number of trials to run (default: 10)")
    parser.add_argument("--base-model", type=str, default=DEFAULT_BASE_MODEL, help="Base HF model (default: UBC-NLP/AraT5-base)")
    parser.add_argument("--data-dir", type=str, default=str(DEFAULT_DATA_DIR), help="Dataset directory")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR), help="Output directory")
    parser.add_argument("--overfitting-penalty", type=float, default=0.3, help="Penalty weight on gap (default: 0.3)")
    parser.add_argument("--max-train-samples", type=int, default=None, help="Optional subset of training samples")
    parser.add_argument("--max-valid-samples", type=int, default=None, help="Optional subset of validation samples")
    parser.add_argument("--no-deploy", dest="deploy_to_pipeline", action="store_false", default=True, help="Disable copying best model to araT5/models/arat5-dialect-msa")

    args = parser.parse_args()

    run_tuning(
        n_trials=args.n_trials,
        base_model_name=args.base_model,
        data_dir=Path(args.data_dir),
        output_dir=Path(args.output_dir),
        overfitting_penalty=args.overfitting_penalty,
        max_train_samples=args.max_train_samples,
        max_valid_samples=args.max_valid_samples,
        deploy_to_pipeline=args.deploy_to_pipeline,
    )


if __name__ == "__main__":
    main()
