from datasets import load_dataset

def main():
    # 1. Download from Hugging Face and cache locally
    dataset = load_dataset("ailsntua/QEvasion")

    # 2. Save to HF disk format (optional but nice to have)
    dataset.save_to_disk("clarity_qevasion_hf")
    print("Saved HF dataset to ./clarity_qevasion_hf")

    # 3. Also save each split as CSV files
    split_names = ["train", "validation", "test"]
    for split in split_names:
        if split in dataset:
            df = dataset[split].to_pandas()
            out_path = f"clarity_{split}.csv"
            df.to_csv(out_path, index=False)
            print(f"Saved {split} split to {out_path}")

if __name__ == "__main__":
    main()
