import pandas as pd
from transformers import AutoTokenizer
from pathlib import Path
import torch

MODEL_NAME = "bert-base-uncased"
MAX_LENGTH = 256

DATA_DIR = Path("data")
SPLITS = {
    "train": DATA_DIR / "clarity_train.csv",
    "test": DATA_DIR / "clarity_test.csv",
}
OUT_DIR = DATA_DIR / "tokenized"


def load_texts(path: Path):

    df = pd.read_csv(path)

    if "question" not in df.columns:
        raise ValueError(
            f"'question' column not found in {path}. Columns = {list(df.columns)}"
        )

    texts = df["question"].fillna("").tolist()
    return df, texts


def tokenize_texts(tokenizer, texts):
    encodings = tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",  # 回傳 PyTorch tensor
    )
    return encodings


def process_split(split_name: str, csv_path: Path, tokenizer):
    print(f"\n=== Processing {split_name} split ===")
    print("🔹 Loading data from:", csv_path)

    df, texts = load_texts(csv_path)
    print(f"✅ Loaded {len(df)} samples")

    encodings = tokenize_texts(tokenizer, texts)

    print("✅ Tokenization done")
    print("   input_ids shape:", encodings["input_ids"].shape)
    print("   attention_mask shape:", encodings["attention_mask"].shape)

    if split_name == "train":
        for col in ["clarity_label", "evasion_label"]:
            if col in df.columns:
                print(f"\n📊 {col} value counts:")
                print(df[col].value_counts(dropna=False))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_file = OUT_DIR / f"{split_name}_tokenized.pt"

    save_dict = {
        "model_name": MODEL_NAME,
        "max_length": MAX_LENGTH,
        "input_ids": encodings["input_ids"],
        "attention_mask": encodings["attention_mask"],
    }

    for col in ["clarity_label", "evasion_label"]:
        if col in df.columns:
            save_dict[col] = df[col].tolist()

    torch.save(save_dict, out_file)
    print(f"Saved tokenized tensors to: {out_file}")


def main():
    print("🔹 Initializing tokenizer:", MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    for split_name, csv_path in SPLITS.items():
        if not csv_path.exists():
            print(f"⚠️  Skip {split_name}: file not found -> {csv_path}")
            continue
        process_split(split_name, csv_path, tokenizer)


if __name__ == "__main__":
    main()
