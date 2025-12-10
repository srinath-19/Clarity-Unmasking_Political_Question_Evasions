from pathlib import Path
import json

import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import f1_score, classification_report
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EvalPrediction,
    EarlyStoppingCallback,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

TRAIN_PATH = DATA_DIR / "clarity_train_for_model.csv"
VAL_PATH   = DATA_DIR / "clarity_validation_for_model.csv"
LABELS_PATH = ARTIFACTS_DIR / "clarity_label_mapping.json"

MODEL_NAME = "roberta-base"
OUTPUT_DIR = ARTIFACTS_DIR / "clarity_roberta_base_256"

MAX_LENGTH = 128
NUM_EPOCHS = 10

if torch.cuda.is_available():
    TRAIN_BATCH_SIZE = 8
    EVAL_BATCH_SIZE = 16
else:
    TRAIN_BATCH_SIZE = 1
    EVAL_BATCH_SIZE = 2
LEARNING_RATE = 2e-5

def load_label_mapping():
    """Load label2id / id2label mapping from JSON."""
    with open(LABELS_PATH, "r", encoding="utf-8") as f:
        mapping = json.load(f)

    label2id = {str(k): int(v) for k, v in mapping["label2id"].items()}
    id2label = {int(k): v for k, v in mapping["id2label"].items()}
    return label2id, id2label


def load_splits():
    """
    Load train/val CSVs and return:
      - HF datasets (using model_text + label_id)
      - original dataframes (for reporting)
    Expecting columns:
      - model_text
      - label_id
    """
    train_df = pd.read_csv(TRAIN_PATH)
    val_df   = pd.read_csv(VAL_PATH)

    required_cols = ["model_text", "label_id"]
    for col in required_cols:
        if col not in train_df.columns:
            raise ValueError(f"Train CSV missing column: {col}")
        if col not in val_df.columns:
            raise ValueError(f"Val CSV missing column: {col}")

    train_small = train_df[["model_text", "label_id"]].rename(
        columns={"label_id": "labels"}
    )
    val_small = val_df[["model_text", "label_id"]].rename(
        columns={"label_id": "labels"}
    )

    train_small = train_small.dropna(subset=["model_text", "labels"])
    val_small   = val_small.dropna(subset=["model_text", "labels"])

    print(f"Train rows: {len(train_small)}")
    print(f"Val rows:   {len(val_small)}")

    train_ds = Dataset.from_pandas(train_small, preserve_index=False)
    val_ds   = Dataset.from_pandas(val_small,   preserve_index=False)

    return train_ds, val_ds, train_df, val_df

def main():
    ARTIFACTS_DIR.mkdir(exist_ok=True, parents=True)
    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

    label2id, id2label = load_label_mapping()
    num_labels = len(label2id)
    print("label2id:", label2id)
    print("id2label:", id2label)

    train_ds, val_ds, train_df, val_df = load_splits()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tokenize_batch(batch):
        return tokenizer(
            batch["model_text"],
            padding="max_length",
            truncation=True,
            max_length=MAX_LENGTH,
        )

    train_ds = train_ds.map(tokenize_batch, batched=True)
    val_ds   = val_ds.map(tokenize_batch,   batched=True)

    # Drop raw text once tokenized to reduce memory footprint
    train_ds = train_ds.remove_columns(["model_text"])
    val_ds   = val_ds.remove_columns(["model_text"])

    train_ds.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "labels"],
    )
    val_ds.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "labels"],
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=num_labels,
        label2id=label2id,
        id2label=id2label,
    )

    if not torch.cuda.is_available():
        print("CPU detected: freezing encoder weights (train classifier head only).")
        for name, param in model.named_parameters():
            if "classifier" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

    def compute_metrics(eval_pred):
        if isinstance(eval_pred, EvalPrediction):
            logits, labels = eval_pred.predictions, eval_pred.label_ids
        else:
            logits, labels = eval_pred
        preds = logits.argmax(axis=-1)
        macro_f1 = f1_score(labels, preds, average="macro")
        return {"macro_f1": macro_f1}

    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=50,
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        num_train_epochs=NUM_EPOCHS,
        weight_decay=0.01,
        warmup_ratio=0.06,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        report_to=[],
        fp16=torch.cuda.is_available(),
        seed=42,
        optim="adafactor",
        dataloader_pin_memory=False,

    )

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

    trainer.train()

    eval_results = trainer.evaluate()
    print("\nFinal eval results (clarity, RoBERTa-base):", eval_results)

    preds_output = trainer.predict(val_ds)
    preds = preds_output.predictions.argmax(axis=-1)
    y_true = preds_output.label_ids

    print("\nValidation classification report (Clarity, RoBERTa-base):\n")
    print(
        classification_report(
            y_true,
            preds,
            target_names=[id2label[i] for i in range(num_labels)],
        )
    )

    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"\nSaved RoBERTa-base clarity model to: {OUTPUT_DIR}")


if __name__ == "__main__":
    if torch.cuda.is_available():
        print("CUDA is available, training will use GPU.")
    else:
        print("CUDA not available, training will run on CPU (slower).")
    main()
