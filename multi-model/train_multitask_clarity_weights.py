from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score, classification_report
from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)

# ---------------- CONFIG ---------------- #

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

TRAIN_PATH = DATA_DIR / "clarity_evasion_train.csv"
VAL_PATH = DATA_DIR / "clarity_evasion_val.csv"

MODEL_NAME = "microsoft/deberta-v3-base"
OUTPUT_DIR = ARTIFACTS_DIR / "multitask_deberta_clarity_weights"

# NEW: where your clarity-only model was saved
CLARITY_CKPT_DIR = ARTIFACTS_DIR / "clarity_deberta"

MAX_LENGTH = 256
LAMBDA_CLARITY = 1.0
LAMBDA_EVASION = 1.0


# ---------------- DATA ---------------- #

def load_splits():
    """
    Load the multi-task CSVs and return:
      - train_df, val_df (pandas)
      - train_ds, val_ds (HF Dataset with tokenized fields + labels)
      - label_info (for pretty reports)
    """
    if not TRAIN_PATH.is_file():
        raise FileNotFoundError(f"Train CSV not found: {TRAIN_PATH}")
    if not VAL_PATH.is_file():
        raise FileNotFoundError(f"Val CSV not found: {VAL_PATH}")

    train_df = pd.read_csv(TRAIN_PATH)
    val_df = pd.read_csv(VAL_PATH)

    # Only keep what we need
    train_df = train_df[["question", "interview_answer", "label_id", "evasion_id"]].dropna()
    val_df = val_df[["question", "interview_answer", "label_id", "evasion_id"]].dropna()

    print(f"Train rows: {len(train_df)}")
    print(f"Val rows:   {len(val_df)}")

    # Simple id->name maps (you can replace with real names later if you want)
    clarity_id2label = {
        int(i): f"clarity_{int(i)}"
        for i in sorted(train_df["label_id"].unique())
    }
    evasion_id2label = {
        int(i): f"evasion_{int(i)}"
        for i in sorted(train_df["evasion_id"].unique())
    }

    label_info = {
        "clarity_id2label": clarity_id2label,
        "evasion_id2label": evasion_id2label,
    }

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # 1) HF datasets from pandas
    train_ds = Dataset.from_pandas(train_df, preserve_index=False)
    val_ds = Dataset.from_pandas(val_df, preserve_index=False)

    # 2) tokenize ONLY (like your clarity script)
    def tokenize_batch(batch):
        texts = [
            "[QUESTION] " + q + " [ANSWER] " + a
            for q, a in zip(batch["question"], batch["interview_answer"])
        ]
        return tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=MAX_LENGTH,
        )

    train_ds = train_ds.map(tokenize_batch, batched=True)
    val_ds = val_ds.map(tokenize_batch, batched=True)

    # 3) add a single 'labels' field = [clarity_id, evasion_id] (shape [2])
    def add_labels(batch):
        batch["labels"] = list(zip(batch["label_id"], batch["evasion_id"]))
        return batch

    train_ds = train_ds.map(add_labels, batched=True)
    val_ds = val_ds.map(add_labels, batched=True)

    # 4) format for Trainer – keep only tensors we need
    cols = ["input_ids", "attention_mask", "labels"]
    if "token_type_ids" in train_ds.column_names:
        cols.append("token_type_ids")

    train_ds.set_format(type="torch", columns=cols)
    val_ds.set_format(type="torch", columns=cols)

    # debug
    print("Train dataset columns:", train_ds.column_names)
    print("Example labels entry:", train_ds[0]["labels"])

    return train_df, val_df, train_ds, val_ds, tokenizer, label_info


# ---------------- MODEL ---------------- #

class DebertaForClarityEvasion(nn.Module):
    """
    Multi-task model: shared DeBERTa encoder + 2 classification heads.

    - Head 1: clarity (3 classes)
    - Head 2: evasion (K classes)

    Expects 'labels' tensor of shape [B, 2] -> [clarity_id, evasion_id].
    Returns (loss, (logits_clarity, logits_evasion)).
    """

    def __init__(
        self,
        base_model_name: str,
        clarity_num_labels: int,
        evasion_num_labels: int,
        loss_weights: Tuple[float, float] = (1.0, 1.0),
    ):
        super().__init__()

        self.config = AutoConfig.from_pretrained(base_model_name)
        self.encoder = AutoModel.from_pretrained(base_model_name, config=self.config)

        hidden_size = self.config.hidden_size
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)

        self.clarity_head = nn.Linear(hidden_size, clarity_num_labels)
        self.evasion_head = nn.Linear(hidden_size, evasion_num_labels)

        self.loss_fct = nn.CrossEntropyLoss()
        self.lambda_clarity = loss_weights[0]
        self.lambda_evasion = loss_weights[1]

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ):
        # Drop trainer-specific extras that DeBERTa doesn't know about
        kwargs.pop("num_items_in_batch", None)

        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            **kwargs,
        )

        if hasattr(encoder_outputs, "pooler_output") and encoder_outputs.pooler_output is not None:
            pooled = encoder_outputs.pooler_output
        else:
            pooled = encoder_outputs.last_hidden_state[:, 0]  # CLS token

        pooled = self.dropout(pooled)

        logits_clarity = self.clarity_head(pooled)
        logits_evasion = self.evasion_head(pooled)

        loss = None
        if labels is not None:
            labels_clarity = labels[:, 0]
            labels_evasion = labels[:, 1]

            loss_clarity = self.loss_fct(
                logits_clarity.view(-1, logits_clarity.size(-1)),
                labels_clarity.view(-1),
            )
            loss_evasion = self.loss_fct(
                logits_evasion.view(-1, logits_evasion.size(-1)),
                labels_evasion.view(-1),
            )
            loss = self.lambda_clarity * loss_clarity + self.lambda_evasion * loss_evasion

        return (loss, (logits_clarity, logits_evasion))


# ---------------- METRICS ---------------- #

def compute_metrics(eval_pred):
    """
    eval_pred.predictions: (logits_clarity, logits_evasion)
    eval_pred.label_ids: array [N, 2] -> [:,0]=clarity, [:,1]=evasion
    """
    # handle both tuple and EvalPrediction
    if hasattr(eval_pred, "predictions"):
        logits = eval_pred.predictions
        labels = eval_pred.label_ids
    else:
        logits, labels = eval_pred

    logits_clarity, logits_evasion = logits

    labels_clarity = labels[:, 0]
    labels_evasion = labels[:, 1]

    preds_clarity = np.argmax(logits_clarity, axis=-1)
    preds_evasion = np.argmax(logits_evasion, axis=-1)

    metrics: Dict[str, float] = {}

    # per-head metrics with sklearn
    metrics["clarity_acc"] = accuracy_score(labels_clarity, preds_clarity)
    metrics["clarity_macro_f1"] = f1_score(
        labels_clarity, preds_clarity, average="macro"
    )

    metrics["evasion_acc"] = accuracy_score(labels_evasion, preds_evasion)
    metrics["evasion_macro_f1"] = f1_score(
        labels_evasion, preds_evasion, average="macro"
    )

    # joint accuracy (both heads correct) – compute manually
    joint_correct = (preds_clarity == labels_clarity) & (preds_evasion == labels_evasion)
    metrics["joint_acc"] = float(joint_correct.mean())

    return metrics



# ---------------- MAIN ---------------- #

def main():
    ARTIFACTS_DIR.mkdir(exist_ok=True, parents=True)
    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

    # 1. Data
    train_df, val_df, train_ds, val_ds, tokenizer, label_info = load_splits()

    num_clarity_labels = len(sorted(train_df["label_id"].unique()))
    num_evasion_labels = len(sorted(train_df["evasion_id"].unique()))

    print("Num clarity labels:", num_clarity_labels)
    print("Num evasion labels:", num_evasion_labels)

    # 2. Model
    if CLARITY_CKPT_DIR.is_dir():
        base_init = str(CLARITY_CKPT_DIR)
        print(f"Warm-starting encoder from clarity checkpoint: {base_init}")
    else:
        base_init = MODEL_NAME
        print(f"Clarity checkpoint not found at {CLARITY_CKPT_DIR}, falling back to base model: {MODEL_NAME}")

    model = DebertaForClarityEvasion(
        base_model_name=base_init,
        clarity_num_labels=num_clarity_labels,
        evasion_num_labels=num_evasion_labels,
        # evasion_class_weights=evasion_class_weights,
        loss_weights=(LAMBDA_CLARITY, LAMBDA_EVASION),
    )

    # 3. Training Arguments (matching your working clarity script style)
    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=50,
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        num_train_epochs=5,
        weight_decay=0.01,
        warmup_ratio=0.06,
        load_best_model_at_end=True,
        metric_for_best_model="clarity_macro_f1",
        greater_is_better=True,
        report_to=[],
        seed=42,
    )

    # 4. Plain Trainer, NO custom subclass
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=2,
                early_stopping_threshold=0.001,
            )
        ],
    )

    # 5. Train
    trainer.train()

    # 6. Final eval
    eval_results = trainer.evaluate()
    print("\nFinal eval results:", eval_results)

    # 7. Detailed per-label report (clarity head only, for now)
    preds_output = trainer.predict(val_ds)
    logits_clarity, logits_evasion = preds_output.predictions
    preds_clarity = logits_clarity.argmax(axis=-1)
    y_true_clarity = preds_output.label_ids[:, 0]

    print("\nValidation classification report (clarity head):\n")
    print(
        classification_report(
            y_true_clarity,
            preds_clarity,
            target_names=[
                label_info["clarity_id2label"].get(i, str(i))
                for i in sorted(np.unique(y_true_clarity))
            ],
        )
    )

    # 8. Save best model + tokenizer
    # 9. Save best model weights + tokenizer + metadata
    state_path = OUTPUT_DIR / "multitask_deberta_state.pt"
    meta_path = OUTPUT_DIR / "multitask_deberta_meta.json"

    import json, torch

    torch.save(model.state_dict(), state_path)

    meta = {
        "base_model_name": str(CLARITY_CKPT_DIR if CLARITY_CKPT_DIR.is_dir() else MODEL_NAME),
        "max_length": MAX_LENGTH,
        "num_clarity_labels": num_clarity_labels,
        "num_evasion_labels": num_evasion_labels,
        "clarity_id2label": label_info["clarity_id2label"],
        "evasion_id2label": label_info["evasion_id2label"],
        "lambda_clarity": LAMBDA_CLARITY,
        "lambda_evasion": LAMBDA_EVASION,
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    tokenizer.save_pretrained(OUTPUT_DIR)

    print(f"\nSaved multi-task model weights to: {state_path}")
    print(f"Saved metadata to: {meta_path}")
    print(f"Saved tokenizer to: {OUTPUT_DIR}")


if __name__ == "__main__":
    if torch.cuda.is_available():
        print("CUDA is available, training will use GPU.")
    else:
        print("CUDA not available, training will run on CPU (slower).")
    main()
