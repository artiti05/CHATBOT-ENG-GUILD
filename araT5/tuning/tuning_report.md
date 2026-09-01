# AraT5 Hyperparameter Optimization & Model Selection Report

- **Winning Model**: **Trial 11 (ID 11)**
- **Best Validation Loss (`eval_loss`)**: **`1.58398`**
- **Overfitting Gap ($\Delta$)**: **`0.00000`**
- **Model Checkpoint**: `araT5/models/arat5-dialect-msa`

---

## 🏆 Optimal Production Hyperparameters

| Hyperparameter | Optimal Value | Description |
| :--- | :--- | :--- |
| **`learning_rate`** | `2.00e-04` | Optimal step size for Seq2Seq fine-tuning |
| **`per_device_train_batch_size`** | `8` | Balanced batch size for short dialect queries |
| **`gradient_accumulation_steps`** | `2` | Effective batch size of 16 |
| **`num_train_epochs`** | `4` | 4 full passes over JODA dataset |
| **`weight_decay`** | `0.001` | L2 weight regularization |
| **`warmup_ratio`** | `0.10` | 10% linear learning rate warmup |
| **`label_smoothing_factor`** | `0.0` | Exact target token cross-entropy |

---

## 📊 Complete Optimization Leaderboard

| Rank | Trial | Penalized Score | Eval Loss | Train Loss | Gap ($\Delta$) | Learning Rate | Batch Size | Grad Accum |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 🥇 **1** | **Trial 11** | **`1.58398`** | `1.58398` | `7.8818` | `0.000` | `2.00e-04` | `8` | `2` (Batch 16) |
| 🥈 **2** | **Trial 12** | **`1.58398`** | `1.58398` | `7.8818` | `0.000` | `2.00e-04` | `8` | `2` (Batch 16) |
| 🥉 **3** | **Trial 7** | **`1.60446`** | `1.60446` | `7.8825` | `0.000` | `2.00e-04` | `8` | `2` (Batch 16) |
| **4** | **Trial 6** | **`1.60700`** | `1.60700` | `3.9800` | `0.000` | `2.00e-04` | `16` | `1` (Batch 16) |
| **5** | **Trial 4** | **`1.74892`** | `1.74892` | `8.4571` | `0.000` | `2.00e-04` | `16` | `2` (Batch 32) |
| **6** | **Trial 1** | **`1.74977`** | `1.74977` | `3.8679` | `0.000` | `1.00e-04` | `8` | `1` (Batch 8) |
| **7** | **Trial 5** | **`2.17581`** | `2.17581` | `5.6182` | `0.000` | `5.00e-05` | `8` | `1` (Batch 8) |
| **8** | **Trial 3** | **`2.48975`** | `2.48975` | `4.4654` | `0.000` | `2.00e-04` | `16` | `1` (Batch 16) |
| **9** | **Trial 9** | **`2.49421`** | `2.49421` | `12.0417` | `0.000` | `5.00e-05` | `8` | `2` (Batch 16) |
| **10** | **Trial 8** | **`2.62546`** | `2.62546` | `5.0378` | `0.000` | `1.00e-04` | `8` | `1` (Batch 8) |
| **11** | **Trial 10** | **`2.77591`** | `2.77591` | `5.7945` | `0.000` | `1.00e-04` | `16` | `1` (Batch 16) |
| **12** | **Trial 2** | **`3.32204`** | `3.32204` | `7.2065` | `0.000` | `5.00e-05` | `16` | `1` (Batch 16) |
