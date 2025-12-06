from pathlib import Path
import json

import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import classification_report, f1_score, confusion_matrix
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)

# ---------------- CONFIG ---------------- #

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

# Eval split
EVAL_PATH = DATA_DIR / "clarity_evasion_val.csv"

# Trained model directory
MODEL_DIR = ARTIFACTS_DIR / "deberta_evasion_256_model_text"

# Label mapping
EVASION_LABELS_PATH = ARTIFACTS_DIR / "evasion_label_mapping.json"

MAX_LENGTH = 256
EVAL_BATCH_SIZE = 16


# ---------------- HELPERS ---------------- #

def load_evasion_mapping():
    with open(EVASION_LABELS_PATH, "r", encoding="utf-8") as f:
        mapping = json.load(f)
    label2id = {str(k): int(v) for k, v in mapping["label2id"].items()}
    id2label = {int(k): v for k, v in mapping["id2label"].items()}
    return label2id, id2label


def load_eval_dataset():
    df = pd.read_csv(EVAL_PATH)

    required_cols = ["model_text", "evasion_id"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Eval CSV missing column: {col}")

    eval_small = df[["model_text", "evasion_id"]].rename(
        columns={"evasion_id": "labels"}
    )

    eval_small = eval_small.dropna(subset=["model_text", "labels"])

    print(f"Eval rows: {len(eval_small)}")

    eval_ds = Dataset.from_pandas(eval_small, preserve_index=False)
    return eval_ds, df


# ---------------- MAIN ---------------- #

def main():
    if not MODEL_DIR.exists():
        raise FileNotFoundError(f"Model dir not found: {MODEL_DIR}")

    print(f"Loading model from: {MODEL_DIR}")
    print(f"Loading eval data from: {EVAL_PATH}")

    # 1. Load label mapping
    evasion_label2id, id2evasion_label = load_evasion_mapping()
    num_labels = len(evasion_label2id)

    # 2. Load eval dataset
    eval_ds, full_eval_df = load_eval_dataset()

    # 3. Load tokenizer + model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_DIR,
        num_labels=num_labels,
        label2id=evasion_label2id,
        id2label=id2evasion_label,
    )

    # 4. Tokenize
    def tokenize_batch(batch):
        return tokenizer(
            batch["model_text"],
            padding="max_length",
            truncation=True,
            max_length=MAX_LENGTH,
        )

    eval_ds = eval_ds.map(tokenize_batch, batched=True)
    eval_ds.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "labels"],
    )

    # 5. Use Trainer for prediction
    eval_args = TrainingArguments(
        output_dir=str(ARTIFACTS_DIR / "tmp_evasion_eval"),
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        dataloader_drop_last=False,
        do_train=False,
        do_eval=True,
        do_predict=True,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=eval_args,
        tokenizer=tokenizer,
    )

    preds_output = trainer.predict(eval_ds)
    logits = preds_output.predictions
    true_ids = preds_output.label_ids
    pred_ids = logits.argmax(axis=-1)

    target_names = [id2evasion_label[i] for i in range(num_labels)]

    # 6. Classification report (evasion)
    print("\nEvaluation classification report (Evasion labels):\n")
    print(
        classification_report(
            true_ids,
            pred_ids,
            target_names=target_names,
        )
    )

    macro_f1 = f1_score(true_ids, pred_ids, average="macro")
    print(f"\nMacro F1 (evasion): {macro_f1:.4f}")

    # 7. Confusion matrix
    cm = confusion_matrix(true_ids, pred_ids, labels=list(range(num_labels)))

    print("\nConfusion matrix (rows = true, cols = predicted):")
    print("Labels order:", target_names)
    print(cm)

    # Optional: pretty-print as DataFrame
    try:
        import numpy as np  # should already be installed
        import pandas as pd

        cm_df = pd.DataFrame(cm, index=target_names, columns=target_names)
        print("\nConfusion matrix as table:\n")
        print(cm_df)
    except Exception as e:
        print("\nCould not pretty-print confusion matrix as DataFrame:", e)


if __name__ == "__main__":
    if torch.cuda.is_available():
        print("CUDA is available, eval will use GPU.")
    else:
        print("CUDA not available, eval will run on CPU.")
    main()
