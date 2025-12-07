from pathlib import Path
import json
from typing import Dict, Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import Dataset
from sklearn.metrics import f1_score, accuracy_score, classification_report
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModel,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    default_data_collator,
    TrainerCallback,
)

# ---------------- CONFIG ---------------- #

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

# CSVs with: question, interview_answer, label_id, evasion_id
TRAIN_PATH = DATA_DIR / "clarity_evasion_train.csv"
VAL_PATH = DATA_DIR / "clarity_evasion_val.csv"

MODEL_NAME = "microsoft/deberta-v3-base"
# clarity-only checkpoint for warm-start (HF-style folder)
CLARITY_CKPT_DIR = Path("/content/drive/MyDrive/clarity_deberta")  # adjust if needed

OUTPUT_DIR = ARTIFACTS_DIR / "multitask_deberta_v2"

MAX_LENGTH = 384
LAMBDA_CLARITY = 1.0
LAMBDA_EVASION = 1.0
FREEZE_ENCODER_EPOCHS = 1.0  # freeze encoder for first 1 epoch
SEED = 42


# ---------------- DATA LOADING ---------------- #

def load_splits():
    """
    Read train/val CSVs and return HF datasets + pandas DataFrames.
    We only keep: question, interview_answer, label_id, evasion_id
    """
    train_df = pd.read_csv(TRAIN_PATH)
    val_df = pd.read_csv(VAL_PATH)

    train_df = train_df[["question", "interview_answer", "label_id", "evasion_id"]].dropna()
    val_df = val_df[["question", "interview_answer", "label_id", "evasion_id"]].dropna()

    print(f"Train rows: {len(train_df)}")
    print(f"Val rows:   {len(val_df)}")

    train_ds = Dataset.from_pandas(train_df, preserve_index=False)
    val_ds = Dataset.from_pandas(val_df, preserve_index=False)

    return train_ds, val_ds, train_df, val_df


def build_label_info(train_df: pd.DataFrame) -> Dict[str, Any]:
    clarity_ids = sorted(train_df["label_id"].unique().tolist())
    evasion_ids = sorted(train_df["evasion_id"].unique().tolist())

    clarity_id2label = {i: f"clarity_{i}" for i in clarity_ids}
    clarity_label2id = {v: k for k, v in clarity_id2label.items()}

    evasion_id2label = {i: f"evasion_{i}" for i in evasion_ids}
    evasion_label2id = {v: k for k, v in evasion_id2label.items()}

    return {
        "clarity_id2label": clarity_id2label,
        "clarity_label2id": clarity_label2id,
        "evasion_id2label": evasion_id2label,
        "evasion_label2id": evasion_label2id,
    }


def compute_evasion_class_weights(train_df: pd.DataFrame) -> torch.Tensor:
    counts = train_df["evasion_id"].value_counts().sort_index()
    print("\nEvasion label counts:")
    print(counts)

    num_classes = counts.shape[0]
    total = counts.sum()

    # inverse-frequency style
    weights = total / (num_classes * counts)
    w_tensor = torch.tensor([weights[i] for i in range(num_classes)], dtype=torch.float)
    print("\nEvasion class weights:", w_tensor)
    return w_tensor


# ---------------- MODEL ---------------- #

class DebertaForClarityEvasionDualInput(nn.Module):
    """
    Shared DeBERTa encoder, two-head classifier, dual-input:
    - QA pair: question + [SEP] + answer
    - Answer-only: answer text
    The [CLS]-like pooled outputs from both are concatenated for classification.
    """

    def __init__(
        self,
        base_model_name: str,
        clarity_num_labels: int,
        evasion_num_labels: int,
        evasion_class_weights: torch.Tensor,
        lambda_clarity: float,
        lambda_evasion: float,
    ):
        super().__init__()

        self.config = AutoConfig.from_pretrained(base_model_name)
        self.encoder = AutoModel.from_pretrained(base_model_name, config=self.config)

        hidden_size = self.config.hidden_size
        combined_size = hidden_size * 2  # QA CLS + answer CLS

        self.clarity_head = nn.Linear(combined_size, clarity_num_labels)
        self.evasion_head = nn.Linear(combined_size, evasion_num_labels)

        # loss stuff
        self.lambda_clarity = lambda_clarity
        self.lambda_evasion = lambda_evasion
        # register as buffer so it moves with .to(device)
        self.register_buffer("evasion_class_weights", evasion_class_weights)

    def forward(
        self,
        input_ids_qa=None,
        attention_mask_qa=None,
        input_ids_ans=None,
        attention_mask_ans=None,
        labels=None,  # [B, 2] -> [clarity_id, evasion_id]
    ):
        # Encode QA pair
        out_qa = self.encoder(
            input_ids=input_ids_qa,
            attention_mask=attention_mask_qa,
        )
        # Encode answer-only
        out_ans = self.encoder(
            input_ids=input_ids_ans,
            attention_mask=attention_mask_ans,
        )

        # CLS token representation (position 0) from each
        qa_cls = out_qa.last_hidden_state[:, 0, :]   # [B, H]
        ans_cls = out_ans.last_hidden_state[:, 0, :] # [B, H]

        # Concatenate
        h = torch.cat([qa_cls, ans_cls], dim=-1)  # [B, 2H]

        clarity_logits = self.clarity_head(h)
        evasion_logits = self.evasion_head(h)

        loss = None
        if labels is not None:
            clarity_labels = labels[:, 0]
            evasion_labels = labels[:, 1]

            loss_clarity = F.cross_entropy(clarity_logits, clarity_labels)
            loss_evasion = focal_loss(
                evasion_logits,
                evasion_labels,
                class_weights=self.evasion_class_weights,
                gamma=2.0,
            )
            loss = self.lambda_clarity * loss_clarity + self.lambda_evasion * loss_evasion

        # Trainer uses "loss" and "logits"
        return {
            "loss": loss,
            "logits": (clarity_logits, evasion_logits),
            "clarity_logits": clarity_logits,
            "evasion_logits": evasion_logits,
        }


# ---------------- LOSS: FOCAL + CLASS-WEIGHTED ---------------- #

def focal_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    class_weights: torch.Tensor,
    gamma: float = 2.0,
) -> torch.Tensor:
    """
    Multiclass focal loss with per-class weights (alpha).
    logits: [B, C], labels: [B], class_weights: [C]
    """
    ce = F.cross_entropy(logits, labels, weight=class_weights, reduction="none")  # [B]
    pt = torch.exp(-ce)
    loss = ((1 - pt) ** gamma) * ce
    return loss.mean()


# ---------------- CALLBACK: FREEZE ENCODER ---------------- #

class FreezeEncoderCallback(TrainerCallback):
    def __init__(self, freeze_epochs: float):
        self.freeze_epochs = freeze_epochs

    def on_epoch_begin(self, args, state, control, model=None, **kwargs):
        if model is None or self.freeze_epochs <= 0 or state.epoch is None:
            return

        if state.epoch < self.freeze_epochs:
            requires = False
        else:
            requires = True

        for p in model.encoder.parameters():
            p.requires_grad = requires


# ---------------- METRICS ---------------- #

# ---------------- METRICS ---------------- #

def build_compute_metrics(num_clarity_labels: int, num_evasion_labels: int):
    def _unwrap_logits(x):
        """
        Recursively unwrap tuples/lists until we hit something
        that looks like a tensor/ndarray with shape/argmax.
        """
        while isinstance(x, (tuple, list)):
            if len(x) == 0:
                raise ValueError("Empty logits container encountered in predictions.")
            x = x[0]
        return x

    def compute_metrics(eval_pred):
        # Handle both (logits, labels) and EvalPrediction object
        if hasattr(eval_pred, "predictions"):
            preds = eval_pred.predictions
            labels = eval_pred.label_ids
        else:
            preds, labels = eval_pred

        # preds is coming from our custom model; it might be nested tuples
        if isinstance(preds, (tuple, list)):
            if len(preds) < 2:
                raise ValueError(f"Expected at least 2 prediction tensors, got len={len(preds)}")
            clarity_logits = _unwrap_logits(preds[0])
            evasion_logits = _unwrap_logits(preds[1])
        else:
            raise ValueError(
                f"Expected tuple/list of predictions, got type={type(preds)}"
            )

        labels = np.array(labels)

        clarity_labels = labels[:, 0]
        evasion_labels = labels[:, 1]

        # use numpy-style argmax for both torch.Tensor and np.ndarray
        clarity_preds = np.argmax(clarity_logits, axis=-1)
        evasion_preds = np.argmax(evasion_logits, axis=-1)

        clarity_macro_f1 = f1_score(clarity_labels, clarity_preds, average="macro")
        evasion_macro_f1 = f1_score(evasion_labels, evasion_preds, average="macro")

        clarity_acc = accuracy_score(clarity_labels, clarity_preds)
        evasion_acc = accuracy_score(evasion_labels, evasion_preds)

        # joint correctness (both heads right)
        joint_true = [f"{c}_{e}" for c, e in zip(clarity_labels, evasion_labels)]
        joint_pred = [f"{c}_{e}" for c, e in zip(clarity_preds, evasion_preds)]
        joint_acc = accuracy_score(joint_true, joint_pred)

        return {
            "clarity_acc": clarity_acc,
            "clarity_macro_f1": clarity_macro_f1,
            "evasion_acc": evasion_acc,
            "evasion_macro_f1": evasion_macro_f1,
            "joint_acc": joint_acc,
        }

    return compute_metrics



# ---------------- MAIN ---------------- #

def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Data
    train_ds, val_ds, train_df, val_df = load_splits()
    label_info = build_label_info(train_df)

    num_clarity_labels = len(label_info["clarity_id2label"])
    num_evasion_labels = len(label_info["evasion_id2label"])

    print(f"Num clarity labels: {num_clarity_labels}")
    print(f"Num evasion labels: {num_evasion_labels}")

    # 2. Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # 3. Tokenization: dual input + labels=[clarity, evasion]
    def tokenize_dual(batch):
        questions = batch["question"]
        answers = batch["interview_answer"]

        qa_texts = [
            (q or "") + tokenizer.sep_token + (a or "")
            for q, a in zip(questions, answers)
        ]

        enc_qa = tokenizer(
            qa_texts,
            padding="max_length",
            truncation=True,
            max_length=MAX_LENGTH,
        )

        enc_ans = tokenizer(
            answers,
            padding="max_length",
            truncation=True,
            max_length=MAX_LENGTH,
        )

        labels = list(zip(batch["label_id"], batch["evasion_id"]))

        return {
            "input_ids_qa": enc_qa["input_ids"],
            "attention_mask_qa": enc_qa["attention_mask"],
            "input_ids_ans": enc_ans["input_ids"],
            "attention_mask_ans": enc_ans["attention_mask"],
            "labels": labels,
        }

    # Apply tokenization and drop original text columns
    train_ds = train_ds.map(
        tokenize_dual,
        batched=True,
        remove_columns=train_ds.column_names,
    )
    val_ds = val_ds.map(
        tokenize_dual,
        batched=True,
        remove_columns=val_ds.column_names,
    )

    print("Train dataset columns:", train_ds.column_names)
    print("First train example:", {k: type(v) for k, v in train_ds[0].items()})
    print("First train labels value:", train_ds[0]["labels"])

    # Ensure PyTorch tensors (including labels)
    train_ds.set_format(type="torch")
    val_ds.set_format(type="torch")

    # 4. Evasion class weights
    evasion_class_weights = compute_evasion_class_weights(train_df)

    # 5. Model (warm-start encoder)
    if CLARITY_CKPT_DIR.is_dir():
        base_init = str(CLARITY_CKPT_DIR)
        print(f"Warm-starting encoder from clarity checkpoint: {base_init}")
    else:
        base_init = MODEL_NAME
        print(
            f"Clarity checkpoint not found at {CLARITY_CKPT_DIR}, "
            f"falling back to base model: {MODEL_NAME}"
        )

    model = DebertaForClarityEvasionDualInput(
        base_model_name=base_init,
        clarity_num_labels=num_clarity_labels,
        evasion_num_labels=num_evasion_labels,
        evasion_class_weights=evasion_class_weights,
        lambda_clarity=LAMBDA_CLARITY,
        lambda_evasion=LAMBDA_EVASION,
    )

    # 6. Training args
    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        eval_strategy="epoch",   # your env wants eval_strategy, not evaluation_strategy
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="joint_acc",
        greater_is_better=True,
        save_total_limit=2,
        learning_rate=1e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        num_train_epochs=8,
        weight_decay=0.01,
        warmup_ratio=0.06,
        logging_strategy="steps",
        logging_steps=50,
        report_to=[],  # no wandb/tensorboard
        seed=SEED,
    )

    compute_metrics = build_compute_metrics(num_clarity_labels, num_evasion_labels)

    # 7. Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=default_data_collator,
        compute_metrics=compute_metrics,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=3,
                early_stopping_threshold=0.001,
            ),
            FreezeEncoderCallback(FREEZE_ENCODER_EPOCHS),
        ],
    )

    # 8. Train
    trainer.train()

    # 9. Evaluate best model
    eval_results = trainer.evaluate()
    print("\nFinal eval results:", eval_results)

    preds_output = trainer.predict(val_ds)
    preds = preds_output.predictions
    labels = preds_output.label_ids

    def _unwrap_logits(x):
        while isinstance(x, (tuple, list)):
            if len(x) == 0:
                raise ValueError("Empty logits container encountered in predictions.")
            x = x[0]
        return x

    if isinstance(preds, (tuple, list)):
        if len(preds) < 2:
            raise ValueError(f"Expected at least 2 prediction tensors, got len={len(preds)}")
        clarity_logits = _unwrap_logits(preds[0])
        # evasion_logits = _unwrap_logits(preds[1])  # available if you want a report
    else:
        raise ValueError(f"Expected tuple/list of predictions, got type={type(preds)}")

    clarity_preds = np.argmax(clarity_logits, axis=-1)
    clarity_true = labels[:, 0]


    print("\nValidation classification report (clarity head):\n")
    print(
        classification_report(
            clarity_true,
            clarity_preds,
            target_names=[f"clarity_{i}" for i in range(num_clarity_labels)],
        )
    )

    # 10. Save state + meta + tokenizer
    state_path = OUTPUT_DIR / "multitask_deberta_state.pt"
    meta_path = OUTPUT_DIR / "multitask_deberta_meta.json"

    torch.save(model.state_dict(), state_path)

    meta = {
        "base_model_name": base_init,
        "max_length": MAX_LENGTH,
        "num_clarity_labels": num_clarity_labels,
        "num_evasion_labels": num_evasion_labels,
        "clarity_id2label": label_info["clarity_id2label"],
        "evasion_id2label": label_info["evasion_id2label"],
        "lambda_clarity": LAMBDA_CLARITY,
        "lambda_evasion": LAMBDA_EVASION,
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    tokenizer.save_pretrained(OUTPUT_DIR)

    print(f"\nSaved multi-task model weights to: {state_path}")
    print(f"Saved metadata to: {meta_path}")
    print(f"Saved tokenizer to: {OUTPUT_DIR}")


if __name__ == "__main__":
    if torch.cuda.is_available():
        print("CUDA is available, training will use GPU.")
    else:
        print("CUDA not available, training will run on CPU (slower).")
    main()
