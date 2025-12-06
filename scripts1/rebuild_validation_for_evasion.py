from pathlib import Path
import json
import pandas as pd

# ---- CONFIG ----
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

VAL_SRC   = DATA_DIR / "clarity_test.csv"              # original test file
VAL_OUT   = DATA_DIR / "clarity_validation_for_evasion.csv"  # file used by training
EVASION_LABELS_PATH = ARTIFACTS_DIR / "evasion_label_mapping.json"


def load_evasion_mapping():
    with open(EVASION_LABELS_PATH, "r", encoding="utf-8") as f:
        mapping = json.load(f)
    label2id = {str(k): int(v) for k, v in mapping["label2id"].items()}
    return label2id


def build_model_text(row) -> str:
    """Simple Question + Answer text."""
    q = str(row.get("question", "") or "").strip()
    a = str(row.get("interview_answer", "") or "").strip()
    return f"Question: {q}\nAnswer: {a}"


def main():
    if not VAL_SRC.exists():
        raise FileNotFoundError(f"Validation source file not found: {VAL_SRC}")

    print(f"Loading original validation source: {VAL_SRC}")
    df = pd.read_csv(VAL_SRC)

    # Basic sanity
    needed_cols = ["question", "interview_answer", "evasion_label"]
    for col in needed_cols:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in {VAL_SRC}")

    # 1) Build model_text (if you overwrote it earlier, this recreates it)
    df["model_text"] = df.apply(build_model_text, axis=1)

    # 2) Add evasion_id from mapping
    label2id = load_evasion_mapping()
    df["evasion_id"] = df["evasion_label"].astype(str).map(label2id)

    missing = df["evasion_id"].isna().sum()
    if missing > 0:
        print(f"WARNING: {missing} rows have evasion_label not found in mapping.")

    # 3) Show a sample
    cols_to_show = [c for c in ["question", "interview_answer",
                                "model_text", "evasion_label", "evasion_id",
                                "clarity_label"] if c in df.columns]
    print("\n[validation] Sample rows (transposed):")
    print(df[cols_to_show].head(3).T)

    # 4) Save full CSV (all columns, plus model_text + evasion_id)
    df.to_csv(VAL_OUT, index=False)
    print(f"\nSaved rebuilt validation split with evasion_id to: {VAL_OUT}")


if __name__ == "__main__":
    main()
