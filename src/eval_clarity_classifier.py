from pathlib import Path
import json

import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix
from transformers import AutoTokenizer, AutoModelForSequenceClassification

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

VAL_PATH = DATA_DIR / "clarity_validation_for_model.csv"
MODEL_DIR = ARTIFACTS_DIR / "clarity_deberta_256_new_model_text"
LABELS_PATH = ARTIFACTS_DIR / "clarity_label_mapping.json"


def load_label_mapping():
    with open(LABELS_PATH, "r", encoding="utf-8") as f:
        mapping = json.load(f)
    label2id = {k: int(v) for k, v in mapping["label2id"].items()}
    id2label = {int(k): v for k, v in mapping["id2label"].items()}
    return label2id, id2label

def main():
    label2id, id2label = load_label_mapping()

    df = pd.read_csv(VAL_PATH).dropna(subset=["model_text", "label_id"])
    texts = df["model_text"].tolist()
    true_labels = df["label_id"].tolist()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    model.eval()

    all_preds = []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    with torch.no_grad():
        for i in range(0, len(texts), 16):
            batch_texts = texts[i:i+16]
            enc = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors="pt",
            ).to(device)

            logits = model(**enc).logits
            preds = logits.argmax(dim=-1).cpu().tolist()
            all_preds.extend(preds)

    # Metrics
    target_names = [id2label[i] for i in sorted(id2label.keys())]
    print("Classification report:\n")
    print(classification_report(true_labels, all_preds, target_names=target_names, digits=3))

    print("Confusion matrix (rows=true, cols=pred):")
    print(confusion_matrix(true_labels, all_preds))

    # Optional: print a few mistakes
    df["pred_id"] = all_preds
    df["pred_label"] = df["pred_id"].map(id2label)
    df["true_label"] = df["label_id"].map(id2label)

    mistakes = df[df["pred_label"] != df["true_label"]]
    print("\nSample errors:\n")
    for _, row in mistakes.head(10).iterrows():
        print("Q:", row["question"])
        print("A:", row["interview_answer"][:200].replace("\n", " "), "...")
        print("TRUE:", row["true_label"], "| PRED:", row["pred_label"])
        print("-" * 80)



if __name__ == "__main__":
    main()
