# AraT5 Hyperparameter Optimization & Tuning Engine

This directory contains the self-contained Optuna hyperparameter optimization suite for fine-tuning **AraT5** (`UBC-NLP/AraT5-base`) on Jordanian Dialect-to-MSA translation.

---

## 📁 Directory Structure

```text
araT5/tuning/
├── tune_optuna.py              # Main automated tuning & model export script
├── README.md                   # Documentation & execution guide
├── best_hyperparameters.json   # Exported best hyperparameters & metrics
├── tuning_report.md            # Markdown leaderboard & optimization summary
├── best_model/                 # Checkpoint of the selected best model & tokenizer
│   ├── config.json
│   ├── model.safetensors
│   ├── spiece.model
│   └── tokenizer_config.json
├── logs/                       # Centralized training & evaluation logs
│   ├── training_eval_log.csv   # Epoch-by-epoch (train_loss, eval_loss, gap, ratio)
│   ├── trials_summary.json     # Full JSON breakdown of every trial
│   └── optuna_study.db         # SQLite Optuna database
└── trial_checkpoints/          # Temporary trial checkpoints (trial_0, trial_1, ...)
```

---

## 🎯 Anti-Overfitting Objective

To guarantee the selected model generalizes cleanly without memorizing or overfitting the training set, the objective function combines the validation loss with a penalty on the generalization gap:

$$\text{Objective Score} = \text{eval\_loss} + \alpha \cdot \max(0, \text{eval\_loss} - \text{train\_loss})$$

Where:
- $\text{eval\_loss}$: Standard cross-entropy loss on the validation dataset.
- $\text{train\_loss}$: Training cross-entropy loss at the end of training.
- $\alpha$: Generalization gap penalty factor (default: `0.3`).
- **Early Stopping**: Includes `EarlyStoppingCallback(patience=2)` to terminate runs whose validation loss stops improving.
- **Label Smoothing**: Explores label smoothing factors ($0.0 \rightarrow 0.1$) to prevent overconfident token predictions.

---

## 🚀 How to Run (Zero Arguments Needed)

Simply run:
```bash
python araT5/tuning/tune_optuna.py
```
This automatically:
1. Tests the optimal combinations with GPU acceleration (`bfloat16`).
2. Evaluates train/eval generalization gap to prevent overfitting.
3. Automatically saves logs (`training_eval_log.csv`, `trials_summary.json`) in `araT5/tuning/logs/`.
4. Deploys the winning best model directly to `araT5/models/arat5-dialect-msa` for immediate chatbot use.

---

## ⚙️ Configured Hyperparameter Search Space (96 Combinations)

| Hyperparameter | Configured Values | Choices | Purpose |
| :--- | :--- | :--- | :--- |
| `learning_rate` | `[5e-5, 1e-4, 2e-4]` | 3 | Smallest 3 learning rates for stable Seq2Seq fine-tuning |
| `per_device_train_batch_size` | `[8, 16]` | 2 | VRAM optimization (RTX 4060 8GB) |
| `gradient_accumulation_steps` | `[1, 2]` | 2 | Effective batch size (8, 16, 32) |
| `num_train_epochs` | `4` | 1 | Fixed 4 epochs with early stopping |
| `weight_decay` | `[0.001, 0.01]` | 2 | L2 regularization against overfitting |
| `warmup_ratio` | `[0.05, 0.10]` | 2 | Linear learning rate warmup |
| `label_smoothing_factor` | `[0.0, 0.05]` | 2 | Prevents overconfident dialect token predictions |

**Total Grid Combinations**: $3 \times 2 \times 2 \times 1 \times 2 \times 2 \times 2 = \mathbf{96 \text{ combinations}}$.

---

## ⚙️ CLI Arguments

| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--n-trials` | `int` | `10` | Number of Optuna optimization trials to execute. |
| `--base-model` | `str` | `"UBC-NLP/AraT5-base"` | Hugging Face base model identifier. |
| `--data-dir` | `str` | `"araT5/data_msa"` | Folder containing `train.jsonl` and `valid.jsonl`. |
| `--output-dir` | `str` | `"araT5/tuning"` | Directory where all logs, checkpoints, and reports are saved. |
| `--overfitting-penalty`| `float`| `0.3` | Weight $\alpha$ applied to the generalization gap $(\text{eval} - \text{train})$. |
| `--max-train-samples` | `int` | `None` | Cap training data size for faster sweeps. |
| `--max-valid-samples` | `int` | `None` | Cap validation data size for faster sweeps. |
| `--deploy-to-pipeline`| `flag`| `False` | Copies the winning best model to `araT5/models/arat5-dialect-msa`. |
