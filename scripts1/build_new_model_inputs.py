from pathlib import Path
import json
import pandas as pd

# ---- CONFIG ----
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

TRAIN_IN  = DATA_DIR / "clarity_train.csv"
VAL_IN    = DATA_DIR / "clarity_test.csv"

# NEW output files so we don't overwrite older ones
TRAIN_OUT = DATA_DIR / "clarity_train_for_model_v2.csv"
VAL_OUT   = DATA_DIR / "clarity_validation_for_model_v2.csv"

LABELS_PATH = ARTIFACTS_DIR / "clarity_label_mapping.json"


def load_label_mapping():
    """Load label2id mapping from JSON (if available)."""
    try:
        with open(LABELS_PATH, "r", encoding="utf-8") as f:
            mapping = json.load(f)
        label2id = mapping["label2id"]
        label2id = {str(k): int(v) for k, v in label2id.items()}
        print("Loaded label2id mapping:", label2id)
        return label2id
    except FileNotFoundError:
        print("WARNING: label mapping JSON not found, label_id will not be added.")
        return None


def build_model_text(row) -> str:
    """
    Old-style model_text:
      Question: ...
      Answer:   ...
    """
    q = str(row.get("question", "") or "").strip()
    a = str(row.get("interview_answer", "") or "").strip()
    return f"Question: {q}\nAnswer: {a}"


def build_new_model_text(row) -> str:
    """
    New structured text with:
      - Metadata flags: multiple_questions, affirmative_questions, inaudible
      - Question: 'question' or 'interview_question'
      - Answer: 'interview_answer'
      - Summary: 'gpt3.5_summary'
    """
    parts = []

    # --- Metadata flags ---
    flags = []

    def is_one(col_name):
        if col_name not in row:
            return False
        val = row[col_name]
        if pd.isna(val):
            return False
        try:
            return int(val) == 1
        except Exception:
            return False

    if is_one("multiple_questions"):
        flags.append("multiple questions")
    if is_one("affirmative_questions"):
        flags.append("affirmative form")
    if is_one("inaudible"):
        flags.append("partly inaudible")

    if flags:
        parts.append("Metadata: " + "; ".join(flags) + ".")

    # --- Question ---
    q_text = None
    if "question" in row and pd.notna(row["question"]) and str(row["question"]).strip():
        q_text = str(row["question"]).strip()
    elif "interview_question" in row and pd.notna(row["interview_question"]) and str(row["interview_question"]).strip():
        q_text = str(row["interview_question"]).strip()

    if q_text:
        parts.append(f"[QUESTION] {q_text}")

    # --- Answer ---
    a_text = None
    if "interview_answer" in row and pd.notna(row["interview_answer"]) and str(row["interview_answer"]).strip():
        a_text = str(row["interview_answer"]).strip()

    if a_text:
        parts.append(f"[ANSWER] {a_text}")

    # --- GPT summary ---
    s_text = None
    if "gpt3.5_summary" in row and pd.notna(row["gpt3.5_summary"]):
        s_text = str(row["gpt3.5_summary"]).strip()

    if s_text:
        parts.append(f"[SUMMARY] {s_text}")

    return " ".join(parts)


def process_split(in_path: Path, out_path: Path, split_name: str, label2id: dict | None):
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

    # 1) Old-style model_text
    df["model_text"] = df.apply(build_model_text, axis=1)

    # 2) New richer text
    df["new_model_text"] = df.apply(build_new_model_text, axis=1)

    # 3) Map clarity_label -> label_id if mapping is available
    if label2id is not None and "clarity_label" in df.columns:
        if "label_id" not in df.columns:
            df["label_id"] = df["clarity_label"].astype(str).map(label2id)
        else:
            print(f"[{split_name}] 'label_id' already exists, not overwriting.")
    else:
        print(f"[{split_name}] Skipping label_id mapping (no mapping or no clarity_label).")

    # 4) Show sample rows
    print(f"\n[{split_name}] Sample rows (transposed):")
    cols_to_show = [c for c in ["question", "interview_answer", "model_text", "new_model_text", "clarity_label", "label_id"] if c in df.columns]
    print(df[cols_to_show].head(3).T)

    # 5) Save full CSV with all columns
    df.to_csv(out_path, index=False)
    print(f"[{split_name}] Saved processed split to: {out_path}")


def main():
    ARTIFACTS_DIR.mkdir(exist_ok=True)

    label2id = load_label_mapping()

    process_split(TRAIN_IN, TRAIN_OUT, split_name="train",      label2id=label2id)
    process_split(VAL_IN,   VAL_OUT,   split_name="validation", label2id=label2id)

    print("\nDone: created full CSVs with model_text, new_model_text, and label_id.")


if __name__ == "__main__":
    main()
