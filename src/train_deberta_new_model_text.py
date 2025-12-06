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
    EarlyStoppingCallback,
    Trainer,
    EvalPrediction,
)

# ---------------- CONFIG ---------------- #

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

# These should be the outputs from your preprocessing script
TRAIN_PATH = DATA_DIR / "clarity_train_for_model_v2.csv"
VAL_PATH = DATA_DIR / "clarity_validation_for_model_v2.csv"
LABELS_PATH = ARTIFACTS_DIR / "clarity_label_mapping.json"

MODEL_NAME = "microsoft/deberta-v3-base"
OUTPUT_DIR = ARTIFACTS_DIR / "clarity_deberta_256_new_model_text"

# token length & training setup
MAX_LENGTH = 384
NUM_EPOCHS = 5
TRAIN_BATCH_SIZE = 8
EVAL_BATCH_SIZE = 16
LEARNING_RATE = 2e-5  # chosen default for DeBERTa-base


# ---------------- HELPERS ---------------- #

def load_label_mapping():
    """Load label2id / id2label mapping from JSON."""
    with open(LABELS_PATH, "r", encoding="utf-8") as f:
        mapping = json.load(f)
    # ensure proper typing
    label2id = {k: int(v) for k, v in mapping["label2id"].items()}
    id2label = {int(k): v for k, v in mapping["id2label"].items()}
    return label2id, id2label


def load_splits():
    """
    Read train/val CSVs and return HF datasets and train_df.
    Uses 'new_model_text' and 'label_id'.
    """
    train_df = pd.read_csv(TRAIN_PATH)
    val_df = pd.read_csv(VAL_PATH)

    # sanity check
    for col in ["new_model_text", "label_id"]:
        if col not in train_df.columns:
            raise ValueError(f"Train file missing column: {col}")
        if col not in val_df.columns:
            raise ValueError(f"Validation file missing column: {col}")

    # keep only needed columns and drop NAs
    train_df = train_df[["new_model_text", "label_id"]].dropna()
    val_df = val_df[["new_model_text", "label_id"]].dropna()

    # rename for consistency inside the model pipeline
    train_df = train_df.rename(columns={"new_model_text": "model_text"})
    val_df = val_df.rename(columns={"new_model_text": "model_text"})

    print(f"Train rows: {len(train_df)}")
    print(f"Val rows:   {len(val_df)}")

    train_ds = Dataset.from_pandas(train_df, preserve_index=False)
    val_ds = Dataset.from_pandas(val_df, preserve_index=False)
    return train_ds, val_ds, train_df


# ---------------- MAIN ---------------- #

def main():
    ARTIFACTS_DIR.mkdir(exist_ok=True, parents=True)
    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

    # 1. Labels
    label2id, id2label = load_label_mapping()
    num_labels = len(label2id)
    print("label2id:", label2id)
    print("id2label:", id2label)

    # 2. Data
    train_ds, val_ds, train_df = load_splits()

    # 3. Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tokenize_batch(batch):
        return tokenizer(
            batch["model_text"],
            padding="max_length",
            truncation=True,
            max_length=MAX_LENGTH,
        )

    train_ds = train_ds.map(tokenize_batch, batched=True)
    val_ds = val_ds.map(tokenize_batch, batched=True)

    # HF expects column "labels"
    train_ds = train_ds.rename_column("label_id", "labels")
    val_ds = val_ds.rename_column("label_id", "labels")

    train_ds.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "labels"],
    )
    val_ds.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "labels"],
    )

    # 4. Model
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=num_labels,
        label2id=label2id,
        id2label=id2label,
    )

    # 5. Metrics (macro F1)
    def compute_metrics(eval_pred):
        # works with both EvalPrediction and (preds, labels) tuple
        if isinstance(eval_pred, EvalPrediction):
            logits, labels = eval_pred.predictions, eval_pred.label_ids
        else:
            logits, labels = eval_pred
        preds = logits.argmax(axis=-1)
        macro_f1 = f1_score(labels, preds, average="macro")
        return {"macro_f1": macro_f1}

    # 6. Training arguments
    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        eval_strategy="epoch",             # matches your env's working arg
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
        report_to=[],                      # no wandb
        seed=42,
        fp16=True,                         # mixed precision if GPU supports it
    )

    # 7. Trainer with early stopping (no class weights)
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

    # 8. Train
    trainer.train()

    # 9. Final eval + human-readable report
    eval_results = trainer.evaluate()
    print("\nFinal eval results:", eval_results)

    preds_output = trainer.predict(val_ds)
    preds = preds_output.predictions.argmax(axis=-1)
    y_true = preds_output.label_ids

    print("\nValidation classification report:\n")
    print(
        classification_report(
            y_true,
            preds,
            target_names=[id2label[i] for i in range(num_labels)],
        )
    )

    # 10. Save best model + tokenizer
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"\nSaved fine-tuned model to: {OUTPUT_DIR}")


if __name__ == "__main__":
    if torch.cuda.is_available():
        print("CUDA is available, training will use GPU.")
    else:
        print("CUDA not available, training will run on CPU (slower).")
    main()
