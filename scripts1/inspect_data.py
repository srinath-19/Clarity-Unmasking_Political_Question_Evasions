from pathlib import Path
import pandas as pd


# ---- CONFIG ----
DATA_DIR = Path(__file__).resolve().parents[1] / "data"

TRAIN_PATH = DATA_DIR / "clarity_train.csv"        # change if your file name differs
VAL_PATH   = DATA_DIR / "clarity_test.csv"   # optional for now


def main():
    print(f"Loading train data from: {TRAIN_PATH}\n")

    df = pd.read_csv(TRAIN_PATH)

    # 1) Basic shape + columns
    print("Shape:", df.shape)
    print("\nColumns:")
    print(df.columns.tolist())

    # 2) Peek at a few rows
    print("\nSample rows:")
    print(df.head(3).T)  # transpose so you see full text nicely

    # 3) Clarity label distribution
    if "clarity_label" in df.columns:
        print("\nClarity label distribution:")
        print(df["clarity_label"].value_counts(normalize=True))

    # 4) Evasion label distribution (optional, for later task)
    if "evasion_label" in df.columns:
        print("\nEvasion label distribution (top 10):")
        print(df["evasion_label"].value_counts().head(10))

    # 5) Check for missing key fields
    for col in ["question", "interview_answer", "clarity_label"]:
        if col in df.columns:
            n_missing = df[col].isna().sum()
            print(f"\nMissing values in '{col}': {n_missing}")

    print("\nDone step 1: basic inspection.")


if __name__ == "__main__":
    main()
