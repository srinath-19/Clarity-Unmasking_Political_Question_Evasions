from pathlib import Path
import json
import pandas as pd


# ---- CONFIG ----
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

TRAIN_IN  = DATA_DIR / "clarity_train.csv"
VAL_IN    = DATA_DIR / "clarity_test.csv"

TRAIN_OUT = DATA_DIR / "clarity_train_for_model.csv"
VAL_OUT   = DATA_DIR / "clarity_validation_for_model.csv"

LABELS_PATH = ARTIFACTS_DIR / "clarity_label_mapping.json"


def load_label_mapping():
    with open(LABELS_PATH, "r", encoding="utf-8") as f:
        mapping = json.load(f)
    label2id = mapping["label2id"]
    # keys in JSON might be strings; ensure keys are str and values are int
    label2id = {str(k): int(v) for k, v in label2id.items()}
    return label2id


def process_split(in_path: Path, out_path: Path, label2id: dict, split_name: str):
    if not in_path.exists():
        print(f"[{split_name}] File not found, skipping: {in_path}")
        return

    print(f"\n[{split_name}] Loading: {in_path}")
    df = pd.read_csv(in_path)

    # Basic sanity
    needed_cols = ["question", "interview_answer"]
    for col in needed_cols:
        if col not in df.columns:
            raise ValueError(f"[{split_name}] Required column '{col}' not found!")

    # Build the text the model will see
    # You can tweak this later if you want
    df["model_text"] = df.apply(
        lambda row: f"Question: {row['question']}\nAnswer: {row['interview_answer']}",
        axis=1,
    )

    # Map clarity_label -> label_id if labels exist in this split
    if "clarity_label" in df.columns:
        df["label_id"] = df["clarity_label"].map(label2id)

        # Drop rows with missing label or text
        before = len(df)
        df = df.dropna(subset=["model_text", "label_id"])
        after = len(df)
        print(f"[{split_name}] Dropped {before - after} rows with missing text/labels.")
    else:
        print(f"[{split_name}] No 'clarity_label' column, keeping all rows without label_id.")

    # Show a few examples
    print(f"\n[{split_name}] Sample rows:")
    cols_to_show = ["question", "interview_answer", "model_text"]
    if "clarity_label" in df.columns:
        cols_to_show += ["clarity_label", "label_id"]
    print(df[cols_to_show].head(3).T)

    # Save processed CSV
    df.to_csv(out_path, index=False)
    print(f"[{split_name}] Saved processed split to: {out_path}")


def main():
    ARTIFACTS_DIR.mkdir(exist_ok=True)

    # 1. Load label mapping from Step 2
    label2id = load_label_mapping()
    print("Loaded clarity label mapping:", label2id)

    # 2. Process each split
    process_split(TRAIN_IN, TRAIN_OUT, label2id, split_name="train")
    process_split(VAL_IN,   VAL_OUT,   label2id, split_name="validation")


    print("\nDone step 3: built model_text + label_id for each split.")


if __name__ == "__main__":
    main()
