from pathlib import Path
import json
import pandas as pd

# ---- CONFIG ----
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

# Input: the v2 files that already have new_model_text, clarity_label, evasion_label
TRAIN_IN = DATA_DIR / "clarity_train_for_model_v2.csv"
VAL_IN   = DATA_DIR / "clarity_validation_for_model_v2.csv"

# Output: same structure but with evasion_id added
TRAIN_OUT = DATA_DIR / "clarity_train_for_evasion.csv"
VAL_OUT   = DATA_DIR / "clarity_validation_for_evasion.csv"

EVASION_LABELS_PATH = ARTIFACTS_DIR / "evasion_label_mapping.json"


def build_evasion_mapping(train_df: pd.DataFrame) -> dict:
    """
    Build evasion_label2id and id2evasion_label from the TRAIN split only.
    """
    if "evasion_label" not in train_df.columns:
        raise ValueError("Train CSV missing 'evasion_label' column.")

    unique_evasion = sorted(train_df["evasion_label"].dropna().unique().tolist())
    evasion_label2id = {label: i for i, label in enumerate(unique_evasion)}
    id2evasion_label = {i: label for label, i in evasion_label2id.items()}

    mapping = {
        "label2id": evasion_label2id,
        "id2label": id2evasion_label,
    }
    return mapping


def add_evasion_ids(df: pd.DataFrame, label2id: dict, split_name: str) -> pd.DataFrame:
    """
    Add an 'evasion_id' column based on evasion_label and label2id mapping.
    Keeps all existing columns.
    """
    if "evasion_label" not in df.columns:
        raise ValueError(f"[{split_name}] 'evasion_label' column not found.")

    df = df.copy()
    df["evasion_id"] = df["evasion_label"].map(label2id)

    missing = df["evasion_id"].isna().sum()
    if missing > 0:
        print(f"[{split_name}] WARNING: {missing} rows have evasion_label not in mapping.")

    return df


def main():
    ARTIFACTS_DIR.mkdir(exist_ok=True, parents=True)

    # 1. Load train/val
    print(f"Loading train: {TRAIN_IN}")
    train_df = pd.read_csv(TRAIN_IN)

    print(f"Loading val:   {VAL_IN}")
    val_df = pd.read_csv(VAL_IN)

    # 2. Build mapping from TRAIN ONLY
    mapping = build_evasion_mapping(train_df)
    evasion_label2id = mapping["label2id"]

    print("Evasion label2id mapping:")
    for k, v in evasion_label2id.items():
        print(f"  {k!r} -> {v}")

    # 3. Save mapping JSON
    with open(EVASION_LABELS_PATH, "w", encoding="utf-8") as f:
        json.dump(mapping, f, indent=2, ensure_ascii=False)
    print(f"\nSaved evasion label mapping to: {EVASION_LABELS_PATH}")

    # 4. Add evasion_id to both splits (without dropping anything)
    train_out_df = add_evasion_ids(train_df, evasion_label2id, "train")
    val_out_df   = add_evasion_ids(val_df,   evasion_label2id, "validation")

    # 5. Show a few rows
    print("\n[train] Sample rows (transposed):")
    cols_to_show = [c for c in ["evasion_label", "evasion_id", "clarity_label", "new_model_text"] if c in train_out_df.columns]
    print(train_out_df[cols_to_show].head(3).T)

    # 6. Save full CSVs with all columns + evasion_id
    train_out_df.to_csv(TRAIN_OUT, index=False)
    val_out_df.to_csv(VAL_OUT, index=False)

    print(f"\n[train] Saved with evasion_id to: {TRAIN_OUT}")
    print(f"[validation] Saved with evasion_id to: {VAL_OUT}")
    print("\nDone: built evasion label mapping + added evasion_id columns.")


if __name__ == "__main__":
    main()
