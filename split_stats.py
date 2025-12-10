import pandas as pd

train_path = "data/clarity_evasion_train.csv"
dev_path   = "data/clarity_evasion_val.csv"

def summarize_split(df: pd.DataFrame, split_name: str):
    n = len(df)
    print(f"\n=== {split_name} split ===")
    print(f"# QA pairs: {n}")

    print("\nClarity (by clarity_label):")
    clarity_counts = df["clarity_label"].value_counts(dropna=False).sort_index()
    for label, count in clarity_counts.items():
        pct = 100.0 * count / n
        
        name = label
        print(f"  {name}: {count} ({pct:.2f}%)")

    print("\nEvasion (by evasion_id / evasion_label):")

    if "evasion_id" in df.columns:
        ev_counts = df["evasion_id"].value_counts(dropna=False).sort_index()
        for ev_id, count in ev_counts.items():
            pct = 100.0 * count / n
            
            tech_name = f"Technique {int(ev_id)}" if pd.notna(ev_id) else "NaN"
            print(f"  {tech_name}: {count} ({pct:.2f}%)")

    if "evasion_label" in df.columns:
        ev_label_counts = df["evasion_label"].value_counts(dropna=False)
        for label, count in ev_label_counts.items():
            pct = 100.0 * count / n
            print(f"  {label}: {count} ({pct:.2f}%)")

train_df = pd.read_csv(train_path)
dev_df   = pd.read_csv(dev_path)

summarize_split(train_df, "Train")
summarize_split(dev_df, "Dev")
