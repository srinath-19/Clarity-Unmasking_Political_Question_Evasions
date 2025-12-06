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
)

# ---------------- CONFIG ---------------- #

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

TRAIN_PATH = DATA_DIR / "clarity_train_for_model.csv"
VAL_PATH = DATA_DIR / "clarity_validation_for_model.csv"
LABELS_PATH = ARTIFACTS_DIR / "clarity_label_mapping.json"

MODEL_NAME = "microsoft/deberta-v3-base"
OUTPUT_DIR = ARTIFACTS_DIR / "clarity_deberta_10"

MAX_LENGTH = 256


# ---------------- HELPERS ---------------- #

def load_label_mapping():
    """Load label2id / id2label mapping from JSON."""
    with open(LABELS_PATH, "r", encoding="utf-8") as f:
        mapping = json.load(f)
    label2id = {k: int(v) for k, v in mapping["label2id"].items()}
    id2label = {int(k): v for k, v in mapping["id2label"].items()}
    return label2id, id2label


def load_splits():
    """Read CSVs and return HF datasets + pandas train_df."""
    train_df = pd.read_csv(TRAIN_PATH)
    val_df = pd.read_csv(VAL_PATH)

    # Keep only text + label cols for training
    train_df = train_df[["model_text", "label_id"]].dropna()
    val_df = val_df[["model_text", "label_id"]].dropna()

    print(f"Train rows: {len(train_df)}")
    print(f"Val rows:   {len(val_df)}")

    train_ds = Dataset.from_pandas(train_df, preserve_index=False)
    val_ds = Dataset.from_pandas(val_df, preserve_index=False)
    return train_ds, val_ds, train_df


def compute_class_weights(train_df: pd.DataFrame) -> torch.Tensor:
    """
    Compute inverse-frequency class weights as a tensor of shape [num_labels],
    ordered by label_id 0..num_labels-1.
    """
    counts = train_df["label_id"].value_counts().sort_index()
    print("\nTrain label counts:")
    print(counts)

    num_classes = counts.shape[0]
    total = counts.sum()

    # inverse-frequency style: total / (num_classes * count_k)
    weights = total / (num_classes * counts)
    weight_dict = {int(i): float(w) for i, w in weights.items()}
    print("\nClass weights (before tensor):")
    print(weight_dict)

    # Make sure order is 0..num_classes-1
    w_tensor = torch.tensor(
        [weights[i] for i in range(num_classes)],
        dtype=torch.float,
    )
    return w_tensor


# ---------------- CUSTOM TRAINER ---------------- #

class WeightedTrainer(Trainer):
    """
    Trainer that uses class weights in the cross-entropy loss.
    """

    def __init__(self, class_weights: torch.Tensor, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Move weights to correct device (GPU/CPU) after TrainingArguments are set
        self.class_weights = class_weights.to(self.args.device)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """
        Override to apply weighted CrossEntropyLoss.
        HF will still pass num_items_in_batch etc via **kwargs in some versions,
        so we just ignore them.
        """
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
        logits, labels = eval_pred
        preds = logits.argmax(axis=-1)
        macro_f1 = f1_score(labels, preds, average="macro")
        return {"macro_f1": macro_f1}

    # 6. Class weights (on CPU for now; Trainer will move to device)
    class_weights = compute_class_weights(train_df)

    # 7. Training arguments
    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        eval_strategy="epoch",          # use eval_strategy for your HF version
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=50,
        learning_rate=1e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        num_train_epochs=10,            # upper bound; early stopping will cut earlier
        weight_decay=0.01,
        warmup_ratio=0.06,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        report_to=[],                   # disable wandb/tensorboard
        seed=42,
    )

    # 8. Trainer with class weights + early stopping
    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=2,      # stop if 2 evals with no improvement
                early_stopping_threshold=0.001, # ignore tiny wiggles
            )
        ],
        class_weights=class_weights,
    )

    # 9. Train
    trainer.train()

    # 10. Final eval + human-readable report
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

    # 11. Save best model + tokenizer
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"\nSaved fine-tuned model to: {OUTPUT_DIR}")


if __name__ == "__main__":
    if torch.cuda.is_available():
        print("CUDA is available, training will use GPU.")
    else:
        print("CUDA not available, training will run on CPU (slower).")
    main()
