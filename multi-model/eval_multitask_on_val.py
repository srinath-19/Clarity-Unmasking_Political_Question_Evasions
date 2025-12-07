from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import json
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
)

# ---------------- CONFIG ---------------- #

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

VAL_PATH = DATA_DIR / "clarity_evasion_val.csv"

# Where you saved the multitask artifacts
OUTPUT_DIR = ARTIFACTS_DIR / "multitask_deberta_clarity_weights"
STATE_PATH = OUTPUT_DIR / "multitask_deberta_state.pt"
META_PATH = OUTPUT_DIR / "multitask_deberta_meta.json"

MAX_LENGTH_FALLBACK = 256


# ---------------- MODEL ---------------- #

class DebertaForClarityEvasion(nn.Module):
    """
    Multi-task model: shared DeBERTa encoder + 2 classification heads.

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
    eval_pred.label_ids: array [N, 2]
    """
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

    metrics["clarity_acc"] = accuracy_score(labels_clarity, preds_clarity)
    metrics["clarity_macro_f1"] = f1_score(labels_clarity, preds_clarity, average="macro")

    metrics["evasion_acc"] = accuracy_score(labels_evasion, preds_evasion)
    metrics["evasion_macro_f1"] = f1_score(labels_evasion, preds_evasion, average="macro")

    joint_correct = (preds_clarity == labels_clarity) & (preds_evasion == labels_evasion)
    metrics["joint_acc"] = float(joint_correct.mean())

    return metrics


# ---------------- DATA ---------------- #

def load_val_dataset(tokenizer, max_length: int):
    if not VAL_PATH.is_file():
        raise FileNotFoundError(f"Val CSV not found: {VAL_PATH}")

    val_df = pd.read_csv(VAL_PATH)
    val_df = val_df[["question", "interview_answer", "label_id", "evasion_id"]].dropna()

    val_ds = Dataset.from_pandas(val_df, preserve_index=False)

    def tokenize_batch(batch):
        texts = [
            "[QUESTION] " + q + " [ANSWER] " + a
            for q, a in zip(batch["question"], batch["interview_answer"])
        ]
        return tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )

    val_ds = val_ds.map(tokenize_batch, batched=True)

    def add_labels(batch):
        batch["labels"] = list(zip(batch["label_id"], batch["evasion_id"]))
        return batch

    val_ds = val_ds.map(add_labels, batched=True)

    cols = ["input_ids", "attention_mask", "labels"]
    if "token_type_ids" in val_ds.column_names:
        cols.append("token_type_ids")

    val_ds.set_format(type="torch", columns=cols)

    return val_df, val_ds


# ---------------- MAIN ---------------- #

def main():
    if not META_PATH.is_file():
        raise FileNotFoundError(f"Meta JSON not found: {META_PATH}")
    if not STATE_PATH.is_file():
        raise FileNotFoundError(f"State dict not found: {STATE_PATH}")

    with open(META_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)

    # Prefer loading tokenizer from OUTPUT_DIR (your saved one)
    tokenizer = AutoTokenizer.from_pretrained(str(OUTPUT_DIR))

    max_length = int(meta.get("max_length", MAX_LENGTH_FALLBACK))
    num_clarity_labels = int(meta["num_clarity_labels"])
    num_evasion_labels = int(meta["num_evasion_labels"])

    clarity_id2label = {int(k): v for k, v in meta.get("clarity_id2label", {}).items()}
    evasion_id2label = {int(k): v for k, v in meta.get("evasion_id2label", {}).items()}

    # Resolve base model name/path
    base_from_meta = meta.get("base_model_name", "microsoft/deberta-v3-base")
    base_path = Path(str(base_from_meta))
    base_init = str(base_path) if base_path.is_dir() else str(base_from_meta)

    model = DebertaForClarityEvasion(
        base_model_name=base_init,
        clarity_num_labels=num_clarity_labels,
        evasion_num_labels=num_evasion_labels,
        loss_weights=(
            float(meta.get("lambda_clarity", 1.0)),
            float(meta.get("lambda_evasion", 1.0)),
        ),
    )

    state = torch.load(STATE_PATH, map_location="cpu")
    model.load_state_dict(state, strict=True)

    val_df, val_ds = load_val_dataset(tokenizer, max_length=max_length)

    # Minimal eval args
    eval_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR / "_eval_tmp"),
        per_device_eval_batch_size=16,
        report_to=[],
        dataloader_drop_last=False,
    )

    trainer = Trainer(
        model=model,
        args=eval_args,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    # 1) Metrics
    eval_results = trainer.evaluate()
    print("\nEval results:", eval_results)

    # 2) Predictions
    preds_output = trainer.predict(val_ds)
    logits_clarity, logits_evasion = preds_output.predictions

    y_true_clarity = preds_output.label_ids[:, 0]
    y_true_evasion = preds_output.label_ids[:, 1]

    preds_clarity = logits_clarity.argmax(axis=-1)
    preds_evasion = logits_evasion.argmax(axis=-1)

    # 3) Full classification reports
    print("\nValidation classification report (clarity head):\n")
    print(
        classification_report(
            y_true_clarity,
            preds_clarity,
            target_names=[clarity_id2label.get(i, str(i)) for i in sorted(np.unique(y_true_clarity))],
        )
    )

    print("\nValidation classification report (evasion head):\n")
    print(
        classification_report(
            y_true_evasion,
            preds_evasion,
            target_names=[evasion_id2label.get(i, str(i)) for i in sorted(np.unique(y_true_evasion))],
        )
    )

    # 4) Joint accuracy (explicit print)
    joint_correct = (preds_clarity == y_true_clarity) & (preds_evasion == y_true_evasion)
    print(f"\nJoint accuracy: {joint_correct.mean():.4f}")

    # 5) Build export dataframe
    clarity_probs = torch.softmax(torch.tensor(logits_clarity), dim=-1).numpy()
    evasion_probs = torch.softmax(torch.tensor(logits_evasion), dim=-1).numpy()

    out_df = val_df.copy()

    out_df["true_clarity_id"] = y_true_clarity
    out_df["pred_clarity_id"] = preds_clarity
    out_df["true_clarity_label"] = [clarity_id2label.get(int(i), str(i)) for i in y_true_clarity]
    out_df["pred_clarity_label"] = [clarity_id2label.get(int(i), str(i)) for i in preds_clarity]

    out_df["true_evasion_id"] = y_true_evasion
    out_df["pred_evasion_id"] = preds_evasion
    out_df["true_evasion_label"] = [evasion_id2label.get(int(i), str(i)) for i in y_true_evasion]
    out_df["pred_evasion_label"] = [evasion_id2label.get(int(i), str(i)) for i in preds_evasion]

    # Optional: add max prob confidence
    out_df["clarity_conf"] = clarity_probs.max(axis=-1)
    out_df["evasion_conf"] = evasion_probs.max(axis=-1)

    # Optional: add per-class prob columns (comment out if too wide)
    for i in range(clarity_probs.shape[1]):
        col = clarity_id2label.get(i, f"clarity_{i}")
        out_df[f"prob_{col}"] = clarity_probs[:, i]

    for i in range(evasion_probs.shape[1]):
        col = evasion_id2label.get(i, f"evasion_{i}")
        out_df[f"prob_{col}"] = evasion_probs[:, i]

    # 6) Save CSV
    pred_csv_path = OUTPUT_DIR / "val_predictions.csv"
    out_df.to_csv(pred_csv_path, index=False)

    print(f"\nSaved val predictions CSV to: {pred_csv_path}")


if __name__ == "__main__":
    if torch.cuda.is_available():
        print("CUDA is available, eval can use GPU.")
    else:
        print("CUDA not available, eval will run on CPU.")
    main()
