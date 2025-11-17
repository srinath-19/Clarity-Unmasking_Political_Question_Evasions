from pathlib import Path
import json
import pandas as pd


# ---- CONFIG ----
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

TRAIN_PATH = DATA_DIR / "clarity_train.csv"
LABELS_PATH = ARTIFACTS_DIR / "clarity_label_mapping.json"


def main():
    # Make sure artifacts dir exists
    ARTIFACTS_DIR.mkdir(exist_ok=True)

    print(f"Loading train data from: {TRAIN_PATH}")
    df = pd.read_csv(TRAIN_PATH)

    if "clarity_label" not in df.columns:
        raise ValueError("Column 'clarity_label' not found in train CSV!")

    # Get unique labels (drop NaNs just in case)
    labels = sorted(df["clarity_label"].dropna().unique().tolist())
    print("\nFound clarity labels:")
    for lbl in labels:
        print("  -", lbl)

    # Build mappings
    label2id = {label: i for i, label in enumerate(labels)}
    id2label = {i: label for label, i in label2id.items()}

    print("\nLabel → id mapping:")
    for lbl, idx in label2id.items():
        print(f"  {lbl!r} → {idx}")

    # Save to JSON so we can reuse in training
    mapping = {
        "label2id": label2id,
        "id2label": id2label,
    }

    with open(LABELS_PATH, "w", encoding="utf-8") as f:
        json.dump(mapping, f, indent=2, ensure_ascii=False)

    print(f"\nSaved label mapping to: {LABELS_PATH}")

    # Optional: show a few examples with their mapped id
    df["label_id"] = df["clarity_label"].map(label2id)
    print("\nSample rows with mapped label_id:")
    print(df[["question", "clarity_label", "label_id"]].head(5))


if __name__ == "__main__":
    main()
