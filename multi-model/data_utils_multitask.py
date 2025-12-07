# multitask_data.py
"""
Multi-task dataset prep for clarity + evasion classification.

Assumes CSV has at least:
    question,
    interview_answer,
    clarity_label,
    evasion_label,
    label_id,      # numeric clarity label (int)
    evasion_id     # numeric evasion label (int)

This file DOES NOT modify or overwrite the CSVs on disk.
It only reads them and returns pandas DataFrames + label mappings.
"""

from pathlib import Path
from typing import Dict, Tuple

import pandas as pd


REQUIRED_COLS = {
    "question",
    "interview_answer",
    "clarity_label",
    "evasion_label",
    "label_id",
    "evasion_id",
}


def _load_csv(path: Path) -> pd.DataFrame:
    """
    Load a CSV and sanity-check that the required columns are present.

    NOTE: This does NOT save anything back to disk.
    """
    if not path.is_file():
        raise FileNotFoundError(f"CSV not found: {path}")

    df = pd.read_csv(path)

    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"CSV {path} is missing required columns: {missing}")

    # Drop rows with missing text/labels in-memory ONLY
    df = df.dropna(
        subset=[
            "question",
            "interview_answer",
            "clarity_label",
            "evasion_label",
            "label_id",
            "evasion_id",
        ]
    )

    # Ensure numeric types for ids (again, in-memory only)
    df["label_id"] = df["label_id"].astype(int)
    df["evasion_id"] = df["evasion_id"].astype(int)

    return df


def _build_label_mappings(train_df: pd.DataFrame):
    """
    Build mapping dicts from the TRAIN split.

    Uses existing columns:
      clarity_label <-> label_id
      evasion_label <-> evasion_id
    """
    # clarity: string -> int
    clarity_label2id: Dict[str, int] = (
        train_df.groupby("clarity_label")["label_id"].first().to_dict()
    )
    clarity_id2label: Dict[int, str] = {v: k for k, v in clarity_label2id.items()}

    # evasion: string -> int
    evasion_label2id: Dict[str, int] = (
        train_df.groupby("evasion_label")["evasion_id"].first().to_dict()
    )
    evasion_id2label: Dict[int, str] = {v: k for k, v in evasion_label2id.items()}

    label_info = {
        "clarity_label2id": clarity_label2id,
        "clarity_id2label": clarity_id2label,
        "evasion_label2id": evasion_label2id,
        "evasion_id2label": evasion_id2label,
    }
    return label_info


def prepare_multitask_dataframes(
    train_path: Path,
    val_path: Path,
):
    """
    High-level helper:

      - Reads train & val CSVs
      - DOES NOT alter or write back to those files
      - Returns:
          train_df, val_df, label_info

    DataFrames keep all your original columns, e.g.:

      question, interview_answer, clarity_label, evasion_label,
      label_id, evasion_id, model_text, new_model_text, ...
    """
    train_df = _load_csv(train_path)
    val_df = _load_csv(val_path)

    label_info = _build_label_mappings(train_df)

    return train_df, val_df, label_info


# Optional: quick smoke test
if __name__ == "__main__":
    # Adjust these paths to your actual files
    train_csv = Path(r"C:\projects\Clarity_NLP_project\data\clarity_evasion_train.csv")
    val_csv = Path(r"C:\projects\Clarity_NLP_project\data\clarity_evasion_val.csv")

    train_df, val_df, label_info = prepare_multitask_dataframes(
        train_path=train_csv,
        val_path=val_csv,
    )

    print("Train size:", len(train_df))
    print("Val size:", len(val_df))
    print("Clarity mapping:", label_info["clarity_label2id"])
    print("Evasion mapping:", label_info["evasion_label2id"])
    print(
        train_df.head()[
            [
                "question",
                "interview_answer",
                "clarity_label",
                "label_id",
                "evasion_label",
                "evasion_id",
            ]
        ]
    )
