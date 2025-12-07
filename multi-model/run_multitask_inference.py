import json
from pathlib import Path
from typing import Optional, Tuple, Any, Dict

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoConfig, AutoModel

import pandas as pd


# =========================
# CONFIG: set your model dir
# =========================

# Change this to wherever you put the folder
MODEL_DIR = Path(r"C:\projects\Clarity_NLP_project\artifacts\best_multimodel")  # <-- EDIT THIS


# =========================
# MODEL DEFINITION
# (same as training, minus weights & evasion loss stuff)
# =========================

class DebertaForClarityEvasion(nn.Module):
    """
    Multi-task model: shared DeBERTa encoder + 2 classification heads.

    - Head 1: clarity (num_clarity_labels classes)
    - Head 2: evasion (num_evasion_labels classes)

    Expects 'input_ids', 'attention_mask', (optional 'token_type_ids'),
    and optionally 'labels' of shape [B, 2] -> [clarity_id, evasion_id].
    """

    def __init__(
        self,
        base_model_name: str,
        clarity_num_labels: int,
        evasion_num_labels: int,
        loss_weights: Tuple[float, float] = (1.0, 1.0),
    ):
        super().__init__()

        self.config = AutoConfig.from_pretrained(base_model_name)
        self.encoder = AutoModel.from_pretrained(base_model_name, config=self.config)

        hidden_size = self.config.hidden_size
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)

        self.clarity_head = nn.Linear(hidden_size, clarity_num_labels)
        self.evasion_head = nn.Linear(hidden_size, evasion_num_labels)

        self.loss_clarity = nn.CrossEntropyLoss()
        self.loss_evasion = nn.CrossEntropyLoss()

        self.lambda_clarity = loss_weights[0]
        self.lambda_evasion = loss_weights[1]

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ):
        # In training, Trainer may pass num_items_in_batch; ignore it
        kwargs.pop("num_items_in_batch", None)

        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            **kwargs,
        )

        if hasattr(encoder_outputs, "pooler_output") and encoder_outputs.pooler_output is not None:
            pooled = encoder_outputs.pooler_output
        else:
            pooled = encoder_outputs.last_hidden_state[:, 0]  # CLS token

        pooled = self.dropout(pooled)

        logits_clarity = self.clarity_head(pooled)
        logits_evasion = self.evasion_head(pooled)

        loss = None
        if labels is not None:
            labels_clarity = labels[:, 0]
            labels_evasion = labels[:, 1]

            loss_clarity = self.loss_clarity(
                logits_clarity.view(-1, logits_clarity.size(-1)),
                labels_clarity.view(-1),
            )
            loss_evasion = self.loss_evasion(
                logits_evasion.view(-1, logits_evasion.size(-1)),
                labels_evasion.view(-1),
            )
            loss = self.lambda_clarity * loss_clarity + self.lambda_evasion * loss_evasion

        return (loss, (logits_clarity, logits_evasion))


# =========================
# LOADING
# =========================

def load_model_and_tokenizer():
    meta_path = MODEL_DIR / "multitask_deberta_meta.json"
    state_path = MODEL_DIR / "multitask_deberta_state.pt"

    if not meta_path.is_file():
        raise FileNotFoundError(f"Metadata JSON not found: {meta_path}")
    if not state_path.is_file():
        raise FileNotFoundError(f"Model state file not found: {state_path}")

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    base_model_name = meta["base_model_name"]
    max_length = meta["max_length"]
    num_clarity_labels = meta["num_clarity_labels"]
    num_evasion_labels = meta["num_evasion_labels"]
    clarity_id2label = {int(k): v for k, v in meta["clarity_id2label"].items()}
    evasion_id2label = {int(k): v for k, v in meta["evasion_id2label"].items()}

    # 🔴 OLD (causing error)
    # tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

    # ✅ NEW: always load tokenizer from the base model on HF
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)

    model = DebertaForClarityEvasion(
        base_model_name=base_model_name,
        clarity_num_labels=num_clarity_labels,
        evasion_num_labels=num_evasion_labels,
        loss_weights=(1.0, 1.0),  # irrelevant in inference
    )

    state_dict = torch.load(state_path, map_location="cpu")

    # Ignore extra training-only keys like evasion_class_weights, loss_evasion.weight
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    print("Missing keys:", missing_keys)
    print("Unexpected keys:", unexpected_keys)

    model.eval()

    if torch.cuda.is_available():
        model.to("cuda")
        device = "cuda"
    else:
        device = "cpu"

    return model, tokenizer, max_length, clarity_id2label, evasion_id2label, device



# =========================
# SINGLE-EXAMPLE PREDICTION
# =========================

@torch.no_grad()
def predict_single(model, tokenizer, max_length, clarity_id2label, evasion_id2label, device, question, answer):
    text = f"[QUESTION] {question} [ANSWER] {answer}"
    enc = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    enc = {k: v.to(device) for k, v in enc.items()}

    outputs = model(**enc)
    _, (logits_clarity, logits_evasion) = outputs

    probs_clarity = torch.softmax(logits_clarity, dim=-1)[0].cpu().numpy()
    probs_evasion = torch.softmax(logits_evasion, dim=-1)[0].cpu().numpy()

    pred_clarity = int(probs_clarity.argmax())
    pred_evasion = int(probs_evasion.argmax())

    return {
        "clarity_id": pred_clarity,
        "clarity_label": clarity_id2label.get(pred_clarity, str(pred_clarity)),
        "clarity_probs": probs_clarity.tolist(),
        "evasion_id": pred_evasion,
        "evasion_label": evasion_id2label.get(pred_evasion, str(pred_evasion)),
        "evasion_probs": probs_evasion.tolist(),
    }


# =========================
# BATCH PREDICTION ON CSV
# =========================

@torch.no_grad()
def predict_csv(
    model,
    tokenizer,
    max_length,
    clarity_id2label,
    evasion_id2label,
    device,
    csv_path: Path,
    out_path: Path,
):
    df = pd.read_csv(csv_path)
    if not {"question", "interview_answer"}.issubset(df.columns):
        raise ValueError("CSV must contain 'question' and 'interview_answer' columns")

    texts = [
        f"[QUESTION] {q} [ANSWER] {a}"
        for q, a in zip(df["question"], df["interview_answer"])
    ]

    enc = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    enc = {k: v.to(device) for k, v in enc.items()}

    outputs = model(**enc)
    _, (logits_clarity, logits_evasion) = outputs

    probs_clarity = torch.softmax(logits_clarity, dim=-1).cpu().numpy()
    probs_evasion = torch.softmax(logits_evasion, dim=-1).cpu().numpy()

    preds_clarity = probs_clarity.argmax(axis=-1)
    preds_evasion = probs_evasion.argmax(axis=-1)

    df["pred_clarity_id"] = preds_clarity
    df["pred_clarity_label"] = [clarity_id2label.get(int(i), str(i)) for i in preds_clarity]
    df["pred_evasion_id"] = preds_evasion
    df["pred_evasion_label"] = [evasion_id2label.get(int(i), str(i)) for i in preds_evasion]

    df.to_csv(out_path, index=False)
    print(f"Saved predictions to: {out_path}")


# =========================
# MAIN (simple demo)
# =========================

if __name__ == "__main__":
    model, tokenizer, max_length, clarity_id2label, evasion_id2label, device = load_model_and_tokenizer()

    # # ---- 1) Quick single-example test ----
    # q = "Is your message to the Americans who are currently in Ukraine—should they leave the country?"
    # a = "I think it'd be wise to leave the country. Not—I don't mean our—I don't mean—I'm not talking about our diplomatic corps; I'm talking about Americans who are there. I'd hate to see them get caught in a crossfire if in fact they did invade. And there's no need for that. And I—if I were they—if I had anyone there, I'd say leave."
    #
    # out = predict_single(
    #     model=model,
    #     tokenizer=tokenizer,
    #     max_length=max_length,
    #     clarity_id2label=clarity_id2label,
    #     evasion_id2label=evasion_id2label,
    #     device=device,
    #     question=q,
    #     answer=a,
    # )
    #
    # print("\n=== Single example prediction ===")
    # print(f"Question: {q}")
    # print(f"Answer: {a}")
    # print(f"Clarity: {out['clarity_label']} (id={out['clarity_id']})")
    # print(f"Evasion: {out['evasion_label']} (id={out['evasion_id']})")

    # ---- 2) Uncomment this to run on a CSV ----
    csv_in = Path(r"C:\projects\Clarity_NLP_project\data\clarity_evasion_val.csv")   # must have 'question' and 'interview_answer'
    csv_out = Path(r"C:\projects\Clarity_NLP_project\best_multi-model_predictions.csv")
    predict_csv(
        model=model,
        tokenizer=tokenizer,
        max_length=max_length,
        clarity_id2label=clarity_id2label,
        evasion_id2label=evasion_id2label,
        device=device,
        csv_path=csv_in,
        out_path=csv_out,
    )
