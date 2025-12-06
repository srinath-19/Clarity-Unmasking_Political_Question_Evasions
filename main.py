from pathlib import Path
import json

import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import classification_report
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)

# ---------------- CONFIG ---------------- #

# Project layout: main file lives in C:\projects\Clarity_NLP_project\src or root.
# We'll treat the folder containing this file as PROJECT_ROOT.
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

# Input data: must have at least question + interview_answer
# You can point this to any file you want to run inference on:
#   - clarity_evasion_val.csv
#   - clarity_train_for_model.csv
#   - clarity_test.csv, etc.
DATA_PATH = DATA_DIR / "clarity_test.csv"

# Clarity model (direct clarity classifier)
CLARITY_MODEL_DIR = ARTIFACTS_DIR / "clarity_deberta"  # <-- change if needed
CLARITY_LABELS_PATH = ARTIFACTS_DIR / "clarity_label_mapping.json"

# Evasion model (fine-grained evasion classifier)
EVASION_MODEL_DIR = ARTIFACTS_DIR / "deberta_evasion_256_model_text"
EVASION_LABELS_PATH = ARTIFACTS_DIR / "evasion_label_mapping.json"

# Output file with all predictions
OUTPUT_CSV = DATA_DIR / "clarity_evasion_val_with_clarity_and_evasion_preds_v2.csv"

# Tokenization settings (should match training reasonably well)
CLARITY_MAX_LENGTH = 256
EVASION_MAX_LENGTH = 256
BATCH_SIZE = 16


# ---------------- HELPERS ---------------- #

def build_model_text(row) -> str:
    """
    Build basic model_text from question + answer.
    Used if the CSV does not already have a 'model_text' column.
    """
    q = str(row.get("question", "") or "").strip()
    a = str(row.get("interview_answer", "") or "").strip()
    return f"Question: {q}\nAnswer: {a}"


def ensure_model_text(df: pd.DataFrame) -> pd.DataFrame:
    """
    If df has 'model_text', keep it.
    Otherwise, create it from question + interview_answer.
    """
    if "model_text" not in df.columns:
        print("No 'model_text' column found. Building from question + interview_answer.")
        df = df.copy()
        df["model_text"] = df.apply(build_model_text, axis=1)
    else:
        print("'model_text' column found; using as-is.")
    return df


def load_label_mapping(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        mapping = json.load(f)
    label2id = {str(k): int(v) for k, v in mapping["label2id"].items()}
    id2label = {int(k): v for k, v in mapping["id2label"].items()}
    return label2id, id2label


def build_evasion_to_clarity_mapping() -> dict:
    """
    Use the deterministic mapping you observed earlier.
    """
    return {
        "Claims ignorance": "Clear Non-Reply",
        "Clarification": "Clear Non-Reply",
        "Declining to answer": "Clear Non-Reply",
        "Deflection": "Ambivalent",
        "Dodging": "Ambivalent",
        "General": "Ambivalent",
        "Implicit": "Ambivalent",
        "Partial/half-answer": "Ambivalent",
        "Explicit": "Clear Reply",
    }


def run_model_on_texts(
    texts,
    model_dir: Path,
    label2id: dict,
    id2label: dict,
    max_length: int,
    batch_size: int,
):
    """
    Generic helper:
      - loads tokenizer + model from a directory
      - runs predictions on a list of texts
      - returns list of predicted label strings
    """
    num_labels = len(label2id)

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_dir,
        num_labels=num_labels,
        label2id=label2id,
        id2label=id2label,
    )

    df_temp = pd.DataFrame({"model_text": texts})
    df_temp["labels"] = 0  # dummy

    ds = Dataset.from_pandas(df_temp, preserve_index=False)

    def tokenize_batch(batch):
        return tokenizer(
            batch["model_text"],
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )

    ds = ds.map(tokenize_batch, batched=True)
    ds.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "labels"],
    )

    eval_args = TrainingArguments(
        output_dir=str(ARTIFACTS_DIR / "tmp_inference"),
        per_device_eval_batch_size=batch_size,
        dataloader_drop_last=False,
        do_train=False,
        do_eval=False,
        do_predict=True,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=eval_args,
        tokenizer=tokenizer,
    )

    preds_output = trainer.predict(ds)
    logits = preds_output.predictions
    pred_ids = logits.argmax(axis=-1)
    pred_labels = [id2label[int(i)] for i in pred_ids]
    return pred_labels


# ---------------- MAIN ---------------- #

def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Input data file not found: {DATA_PATH}")

    if not CLARITY_MODEL_DIR.exists():
        raise FileNotFoundError(f"Clarity model dir not found: {CLARITY_MODEL_DIR}")

    if not EVASION_MODEL_DIR.exists():
        raise FileNotFoundError(f"Evasion model dir not found: {EVASION_MODEL_DIR}")

    print(f"Loading data from: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)

    # Make sure we have question / answer
    for col in ["question", "interview_answer"]:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in input CSV.")

    # Ensure model_text exists (used for both models)
    df = ensure_model_text(df)

    texts = df["model_text"].astype(str).tolist()

    # 1) Run clarity model
    clarity_label2id, clarity_id2label = load_label_mapping(CLARITY_LABELS_PATH)
    print("Clarity label2id:", clarity_label2id)

    print("\nRunning clarity model (direct clarity prediction)...")
    pred_clarity_direct = run_model_on_texts(
        texts,
        model_dir=CLARITY_MODEL_DIR,
        label2id=clarity_label2id,
        id2label=clarity_id2label,
        max_length=CLARITY_MAX_LENGTH,
        batch_size=BATCH_SIZE,
    )

    df["pred_clarity_direct"] = pred_clarity_direct

    # 2) Run evasion model
    evasion_label2id, evasion_id2label = load_label_mapping(EVASION_LABELS_PATH)
    print("\nEvasion label2id:", evasion_label2id)

    print("\nRunning evasion model (fine-grained evasion prediction)...")
    pred_evasion = run_model_on_texts(
        texts,
        model_dir=EVASION_MODEL_DIR,
        label2id=evasion_label2id,
        id2label=evasion_id2label,
        max_length=EVASION_MAX_LENGTH,
        batch_size=BATCH_SIZE,
    )

    df["pred_evasion_label"] = pred_evasion

    # 3) Derive clarity from evasion
    evasion_to_clarity = build_evasion_to_clarity_mapping()
    df["pred_clarity_from_evasion"] = [
        evasion_to_clarity.get(ev, "UNKNOWN") for ev in pred_evasion
    ]

    # 4) Print small preview
    cols_preview = [
        "question",
        "interview_answer",
        "pred_clarity_direct",
        "pred_evasion_label",
        "pred_clarity_from_evasion",
    ]
    # Add true labels into preview if present
    if "clarity_label" in df.columns:
        cols_preview.insert(2, "clarity_label")
    if "evasion_label" in df.columns:
        cols_preview.insert(3, "evasion_label")

    print("\nSample rows (transposed):")
    print(df[cols_preview].head(5).T)

    # 5) Save with predictions
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved predictions to: {OUTPUT_CSV}")

    # 6) Optional: quick classification reports if labels exist
    if "clarity_label" in df.columns:
        print("\nClassification report: clarity model vs true clarity_label\n")
        print(
            classification_report(
                df["clarity_label"].astype(str),
                df["pred_clarity_direct"].astype(str),
                zero_division=0,
            )
        )

        print("\nClassification report: clarity_from_evasion vs true clarity_label\n")
        print(
            classification_report(
                df["clarity_label"].astype(str),
                df["pred_clarity_from_evasion"].astype(str),
                zero_division=0,
            )
        )

    if "evasion_label" in df.columns:
        # Align label order to evasion_label2id.keys()
        target_names = list(evasion_label2id.keys())
        print("\nClassification report: evasion model vs true evasion_label\n")
        print(
            classification_report(
                df["evasion_label"].astype(str),
                df["pred_evasion_label"].astype(str),
                labels=target_names,
                zero_division=0,
            )
        )


if __name__ == "__main__":
    if torch.cuda.is_available():
        print("CUDA is available, inference will use GPU.")
    else:
        print("CUDA not available, running on CPU.")
    main()
