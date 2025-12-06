from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

# ---- CONFIG ----
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"

SOURCE_PATH = DATA_DIR / "clarity_train_for_evasion.csv"  # full labeled set for evasion
TRAIN_OUT   = DATA_DIR / "clarity_evasion_train.csv"
VAL_OUT     = DATA_DIR / "clarity_evasion_val.csv"

VAL_SIZE = 0.10  # 15% for validation; change to 0.2 if you want 80/20


def main():
    if not SOURCE_PATH.exists():
        raise FileNotFoundError(f"Source file not found: {SOURCE_PATH}")

    print(f"Loading: {SOURCE_PATH}")
    df = pd.read_csv(SOURCE_PATH)

    # Need evasion_id for stratified split
    if "evasion_id" not in df.columns:
        raise ValueError("Column 'evasion_id' not found in source CSV.")

    # Drop rows with missing labels
    before = len(df)
    df = df.dropna(subset=["evasion_id"])
    after = len(df)
    if after < before:
        print(f"Dropped {before - after} rows with missing evasion_id.")

    # Stratified split
    train_df, val_df = train_test_split(
        df,
        test_size=VAL_SIZE,
        random_state=42,
        stratify=df["evasion_id"],
    )

    print(f"Train rows: {len(train_df)}")
    print(f"Val rows:   {len(val_df)}")

    # Save full rows (all columns preserved)
    train_df.to_csv(TRAIN_OUT, index=False)
    val_df.to_csv(VAL_OUT, index=False)

    print(f"\nSaved train split to: {TRAIN_OUT}")
    print(f"Saved val split to:   {VAL_OUT}")


if __name__ == "__main__":
    main()
