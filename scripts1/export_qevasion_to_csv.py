from datasets import load_dataset
import pandas as pd

# columns in the exact order you showed
COL_ORDER = [
    "title",
    "date",
    "president",
    "url",
    "question_order",
    "interview_question",
    "interview_answer",
    "gpt3.5_summary",
    "gpt3.5_prediction",
    "question",
    "annotator_id",
    "annotator1",
    "annotator2",
    "annotator3",
    "inaudible",
    "multiple_questions",
    "affirmative_questions",
    "index",
    "clarity_label",
    "evasion_label",
]

def export_split(hf_dataset, split_name, out_path):
    ds = hf_dataset[split_name]
    df = ds.to_pandas()

    # keep only columns we listed and in that order
    # (some may be missing in a split, so intersect safely)
    cols = [c for c in COL_ORDER if c in df.columns]
    df = df[cols]

    df.to_csv(out_path, index=False)
    print(f"Saved {split_name} to {out_path} with columns:\n{df.columns.tolist()}\n")

def main():
    print("Downloading QEvasion from Hugging Face...")
    ds = load_dataset("ailsntua/QEvasion")

    # optional: also save HF format
    ds.save_to_disk("clarity_qevasion_hf")

    # export each split as CSV
    for split in ["train", "validation", "test"]:
        if split in ds:
            export_split(ds, split, f"clarity_{split}.csv")

if __name__ == "__main__":
    main()
