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

# These are the files you created with evasion_id + model_text
TRAIN_PATH = DATA_DIR / "clarity_evasion_train.csv"
VAL_PATH   = DATA_DIR / "clarity_evasion_val.csv"

EVASION_LABELS_PATH = ARTIFACTS_DIR / "evasion_label_mapping.json"

MODEL_NAME = "microsoft/deberta-v3-base"
OUTPUT_DIR = ARTIFACTS_DIR / "deberta_evasion_256_model_text_10"

MAX_LENGTH = 256
NUM_EPOCHS = 10
TRAIN_BATCH_SIZE = 8
EVAL_BATCH_SIZE = 16
LEARNING_RATE = 2e-5


# ---------------- HELPERS ---------------- #

def load_evasion_mapping():
    """Load evasion label2id / id2label from JSON."""
    with open(EVASION_LABELS_PATH, "r", encoding="utf-8") as f:
        mapping = json.load(f)
    # ensure proper typing
    label2id = {str(k): int(v) for k, v in mapping["label2id"].items()}
    id2label = {int(k): v for k, v in mapping["id2label"].items()}
    return label2id, id2label


def load_splits():
    train_df = pd.read_csv(TRAIN_PATH)
    val_df   = pd.read_csv(VAL_PATH)

    required_cols = ["model_text", "evasion_id"]
    for col in required_cols:
        if col not in train_df.columns:
            raise ValueError(f"Train CSV missing column: {col}")
        if col not in val_df.columns:
            raise ValueError(f"Val CSV missing column: {col}")

    # Keep minimal columns for training
    train_small = train_df[["model_text", "evasion_id"]].rename(
        columns={
            "model_text": "model_text",  # <-- add this
            "evasion_id": "labels",
        }
    )
    val_small = val_df[["model_text", "evasion_id"]].rename(
        columns={
            "model_text": "model_text",  # <-- add this
            "evasion_id": "labels",
        }
    )

    # Drop NA rows
    train_small = train_small.dropna(subset=["model_text", "labels"])
    val_small   = val_small.dropna(subset=["model_text", "labels"])

    print(f"Train rows: {len(train_small)}")
    print(f"Val rows:   {len(val_small)}")

    train_ds = Dataset.from_pandas(train_small, preserve_index=False)
    val_ds   = Dataset.from_pandas(val_small,   preserve_index=False)

    return train_ds, val_ds, train_df, val_df



# ---------------- MAIN ---------------- #

def main():
    ARTIFACTS_DIR.mkdir(exist_ok=True, parents=True)
    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

    # 1. Load label mapping
    evasion_label2id, id2evasion_label = load_evasion_mapping()
    num_labels = len(evasion_label2id)
    print("Evasion label2id:", evasion_label2id)

    # 2. Load splits
    train_ds, val_ds, train_df, val_df = load_splits()

    # 3. Build evasion -> clarity mapping from TRAIN (for derived clarity eval)
    if "evasion_label" in train_df.columns and "clarity_label" in train_df.columns:
        evasion_to_clarity = {}
        for ev_label, group in train_df.groupby("evasion_label"):
            cl_set = group["clarity_label"].dropna().unique()
            if len(cl_set) != 1:
                print(
                    f"WARNING: evasion label '{ev_label}' maps to multiple clarity labels: {cl_set}"
                )
            evasion_to_clarity[ev_label] = cl_set[0]
        print("Evasion → Clarity mapping:", evasion_to_clarity)
    else:
        evasion_to_clarity = None
        print("No evasion_label/clarity_label in CSV; skipping derived clarity eval.")

    # 4. Tokenizer
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

    train_ds.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "labels"],
    )
    val_ds.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "labels"],
    )

    # 5. Model
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=num_labels,
        label2id=evasion_label2id,
        id2label=id2evasion_label,
    )

    # 6. Metrics (macro F1 on evasion labels)
    def compute_metrics(eval_pred):
        if isinstance(eval_pred, EvalPrediction):
            logits, labels = eval_pred.predictions, eval_pred.label_ids
        else:
            logits, labels = eval_pred
        preds = logits.argmax(axis=-1)
        macro_f1 = f1_score(labels, preds, average="macro")
        return {"macro_f1": macro_f1}

    # 7. Training arguments
    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        eval_strategy="epoch",      # keep this to match your working env
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
        report_to=[],               # no wandb
        seed=42,
        fp16=True,                  # mixed precision if GPU supports it
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

    # 8. Train
    trainer.train()

    # 9. Final eval on evasion labels
    eval_results = trainer.evaluate()
    print("\nFinal eval results (evasion):", eval_results)

    preds_output = trainer.predict(val_ds)
    evasion_pred_ids = preds_output.predictions.argmax(axis=-1)
    evasion_true_ids = preds_output.label_ids

    evasion_target_names = [id2evasion_label[i] for i in range(num_labels)]

    print("\nValidation classification report (Evasion labels):\n")
    print(
        classification_report(
            evasion_true_ids,
            evasion_pred_ids,
            target_names=evasion_target_names,
        )
    )

    # 10. Derived clarity evaluation (if mapping available)
    if evasion_to_clarity is not None:
        # Map ids -> evasion label strings
        evasion_pred_labels = [id2evasion_label[int(i)] for i in evasion_pred_ids]
        evasion_true_labels = [id2evasion_label[int(i)] for i in evasion_true_ids]

        def map_to_clarity(ev_labels):
            return [evasion_to_clarity.get(ev, "UNKNOWN") for ev in ev_labels]

        clarity_pred_labels = map_to_clarity(evasion_pred_labels)
        clarity_true_labels = map_to_clarity(evasion_true_labels)

        print("\nValidation classification report (Derived Clarity from evasion):\n")
        print(
            classification_report(
                clarity_true_labels,
                clarity_pred_labels,
            )
        )

    # 11. Save model
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"\nSaved evasion model to: {OUTPUT_DIR}")


if __name__ == "__main__":
    if torch.cuda.is_available():
        print("CUDA is available, training will use GPU.")
    else:
        print("CUDA not available, training will run on CPU (slower).")
    main()
