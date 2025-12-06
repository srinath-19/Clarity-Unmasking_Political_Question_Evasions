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

TRAIN_PATH = DATA_DIR / "clarity_evasion_train.csv"
VAL_PATH   = DATA_DIR / "clarity_evasion_val.csv"

EVASION_LABELS_PATH = ARTIFACTS_DIR / "evasion_label_mapping.json"

MODEL_NAME = "microsoft/deberta-v3-base"
OUTPUT_DIR = ARTIFACTS_DIR / "deberta_evasion_256_model_text_weighted"

MAX_LENGTH = 256
NUM_EPOCHS = 5
TRAIN_BATCH_SIZE = 8
EVAL_BATCH_SIZE = 16
LEARNING_RATE = 2e-5


# ---------------- HELPERS ---------------- #

def load_evasion_mapping():
    """Load evasion label2id / id2label from JSON."""
    with open(EVASION_LABELS_PATH, "r", encoding="utf-8") as f:
        mapping = json.load(f)
    label2id = {str(k): int(v) for k, v in mapping["label2id"].items()}
    id2label = {int(k): v for k, v in mapping["id2label"].items()}
    return label2id, id2label


def load_splits():
    """
    Load train/val CSVs and return:
      - HF datasets using model_text + evasion_id
      - full dfs for later analysis
    """
    train_df = pd.read_csv(TRAIN_PATH)
    val_df   = pd.read_csv(VAL_PATH)

    required_cols = ["model_text", "evasion_id"]
    for col in required_cols:
        if col not in train_df.columns:
            raise ValueError(f"Train CSV missing column: {col}")
        if col not in val_df.columns:
            raise ValueError(f"Val CSV missing column: {col}")

    train_small = train_df[["model_text", "evasion_id"]].rename(
        columns={"evasion_id": "labels"}
    )
    val_small = val_df[["model_text", "evasion_id"]].rename(
        columns={"evasion_id": "labels"}
    )

    train_small = train_small.dropna(subset=["model_text", "labels"])
    val_small   = val_small.dropna(subset=["model_text", "labels"])

    print(f"Train rows: {len(train_small)}")
    print(f"Val rows:   {len(val_small)}")

    train_ds = Dataset.from_pandas(train_small, preserve_index=False)
    val_ds   = Dataset.from_pandas(val_small,   preserve_index=False)

    return train_ds, val_ds, train_df, val_df


def compute_class_weights(train_df: pd.DataFrame) -> torch.Tensor:
    """
    Compute softer class weights from evasion_id frequencies.
    We'll use inverse-sqrt frequency and normalize to mean 1.0
    so it's not as aggressive as full inverse frequency.
    """
    counts = train_df["evasion_id"].value_counts().sort_index()
    print("\nTrain label counts (evasion_id):")
    print(counts)

    # inverse sqrt frequency (so it's softer than 1/freq)
    inv_sqrt = 1.0 / torch.sqrt(torch.tensor(counts.values, dtype=torch.float))
    # normalize so mean weight ~1.0
    weights = inv_sqrt / inv_sqrt.mean()

    weight_dict = {int(i): float(w) for i, w in zip(counts.index, weights)}
    print("\nClass weights (before tensor):")
    for k, v in weight_dict.items():
        print(f"  id {k}: {v:.4f}")

    return weights  # tensor in index order (0..num_classes-1)


# ---------------- CUSTOM TRAINER ---------------- #

class WeightedTrainer(Trainer):
    """Trainer that applies class weights in cross-entropy loss."""

    def __init__(self, class_weights: torch.Tensor, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Move weights to correct device
        self.class_weights = class_weights.to(self.args.device)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # Hugging Face sometimes passes extra kwarg keys; ignore them.
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        loss_fct = torch.nn.CrossEntropyLoss(weight=self.class_weights)
        loss = loss_fct(
            logits.view(-1, self.model.config.num_labels),
            labels.view(-1),
        )

        if return_outputs:
            return loss, outputs
        return loss


# ---------------- MAIN ---------------- #

def main():
    ARTIFACTS_DIR.mkdir(exist_ok=True, parents=True)
    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

    # 1. Label mapping
    evasion_label2id, id2evasion_label = load_evasion_mapping()
    num_labels = len(evasion_label2id)
    print("Evasion label2id:", evasion_label2id)

    # 2. Data
    train_ds, val_ds, train_df, val_df = load_splits()

    # 3. Evasion -> clarity mapping (for reporting if you want later)
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

    # 6. Metrics (macro F1 on evasion)
    def compute_metrics(eval_pred):
        if isinstance(eval_pred, EvalPrediction):
            logits, labels = eval_pred.predictions, eval_pred.label_ids
        else:
            logits, labels = eval_pred
        preds = logits.argmax(axis=-1)
        macro_f1 = f1_score(labels, preds, average="macro")
        return {"macro_f1": macro_f1}

    # 7. Class weights
    class_weights = compute_class_weights(train_df)

    # 8. Training args
    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        eval_strategy="epoch",
        save_strategy="no",  # <- do NOT save checkpoints every epoch
        load_best_model_at_end=False,  # <- no need for best checkpoint logic
        logging_strategy="steps",
        logging_steps=50,
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        num_train_epochs=NUM_EPOCHS,
        weight_decay=0.01,
        warmup_ratio=0.06,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        report_to=[],
        seed=42,
        fp16=True,
    )

    trainer = WeightedTrainer(
        class_weights=class_weights,
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

    # 9. Train
    trainer.train()

    # 10. Final eval
    eval_results = trainer.evaluate()
    print("\nFinal eval results (evasion, weighted):", eval_results)

    preds_output = trainer.predict(val_ds)
    evasion_pred_ids = preds_output.predictions.argmax(axis=-1)
    evasion_true_ids = preds_output.label_ids

    evasion_target_names = [id2evasion_label[i] for i in range(num_labels)]

    print("\nValidation classification report (Evasion labels, weighted):\n")
    print(
        classification_report(
            evasion_true_ids,
            evasion_pred_ids,
            target_names=evasion_target_names,
        )
    )

    # 11. Optional: derived clarity, like before
    if evasion_to_clarity is not None:
        evasion_pred_labels = [id2evasion_label[int(i)] for i in evasion_pred_ids]
        evasion_true_labels = [id2evasion_label[int(i)] for i in evasion_true_ids]

        def map_to_clarity(ev_labels):
            return [evasion_to_clarity.get(ev, "UNKNOWN") for ev in ev_labels]

        clarity_pred_labels = map_to_clarity(evasion_pred_labels)
        clarity_true_labels = map_to_clarity(evasion_true_labels)

        print("\nValidation classification report (Derived Clarity from weighted evasion):\n")
        print(
            classification_report(
                clarity_true_labels,
                clarity_pred_labels,
            )
        )

    # 12. Save model
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"\nSaved weighted evasion model to: {OUTPUT_DIR}")


if __name__ == "__main__":
    if torch.cuda.is_available():
        print("CUDA is available, training will use GPU.")
    else:
        print("CUDA not available, training will run on CPU (slower).")
    main()
