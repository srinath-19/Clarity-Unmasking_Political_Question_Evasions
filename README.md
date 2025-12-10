# Multi-Task DeBERTa for Clarity + Evasion (Training + Eval)

This project trains and evaluates a **multi-task** Transformer model to predict two labels from political Q&A responses:

1. **Clarity** (3 classes)
2. **Evasion type** (9 classes)

It uses a **shared DeBERTa-v3-base encoder** with **two classification heads** and supports **warm-starting** from a clarity-only checkpoint.

---

## Why multi-task?

Instead of training two separate models, we train one model that learns a shared representation of the Q&A context and then specializes through two heads:
- **Clarity head**: predicts whether the answer is direct / partially evasive / unclear
- **Evasion head**: predicts the specific evasion strategy

This typically improves sample efficiency and keeps inference simple (one forward pass).

## Setup

### Recommended environment
Python 3.9+ (3.10/3.11 ideal)

### Install dependencies

If you maintain a `requirements.txt`, include:
- torch
- transformers
- datasets
- scikit-learn
- pandas
- numpy

Example install:
```bash
pip install torch transformers datasets scikit-learn pandas numpy
```
---

## High-level approach


---

## Data format

Training/validation CSVs must include:

| column | description |
|-------|-------------|
| question | question text |
| interview_answer | answer text |
| label_id | clarity label id |
| evasion_id | evasion label id |

Rows with missing values in these fields are dropped.


## Input representation

Each example is turned into one sequence:


### Input construction
Each example is concatenated into a single sequence:

[QUESTION] question text [ANSWER] answer text

### Labels
We store both task labels in a single tensor:

labels = [clarity_id, evasion_id]

### Model

**Base encoder:** `microsoft/deberta-v3-base`

**Heads:**
- Linear head for clarity: `hidden_size -> num_clarity_labels`
- Linear head for evasion: `hidden_size -> num_evasion_labels`

**Pooling:**
- Uses `pooler_output` if present, else CLS token.


## Warm start

If this directory exists:
artifacts/clarity_deberta/


the multitask model initializes the encoder from it:
- This helps preserve strong clarity performance and speeds convergence.

If not found, it falls back to:
- `microsoft/deberta-v3-base`

## Training configuration (defaults)

- epochs: `5`
- learning rate: `2e-5`
- batch size:
  - train: `8`
  - eval: `16`
- weight decay: `0.01`
- warmup ratio: `0.06`
- early stopping:
  - patience: `2`
  - threshold: `0.001`
- best model metric:
  - `clarity_macro_f1`



### Loss
Joint loss is a weighted sum:

loss = λ_clarity * CE(clarity) + λ_evasion * CE(evasion)

Defaults:
- `λ_clarity = 1.0`
- `λ_evasion = 1.0`

---


Artifacts:
- `multitask_deberta_state.pt`  
  Model `state_dict`.
- `multitask_deberta_meta.json`  
  Metadata including:
  - base model used
  - label counts
  - id2label maps
  - lambdas
  - max length
- Tokenizer files.

---

## Results 

From a recent run:

- clarity_acc ≈ **0.64**
- clarity_macro_f1 ≈ **0.70**
- evasion_acc ≈ **0.46**
- evasion_macro_f1 ≈ **0.40**
- joint_acc ≈ **0.42**

Clarity per-class on val:

- Ambivalent: F1 ≈ 0.75  
- Clear non-reply: F1 ≈ 0.61  
- Clear reply: F1 ≈ 0.58  

---

## 🚀 How to Run the Project

### 1️⃣ Train the Clarity-Only Model

```bash
python src/train_clarity_classifier_deberta.py
```

This will:

Load the clarity training/validation data

Train a DeBERTa-v3-base clarity classifier

#### Save the checkpoint to:

artifacts/clarity_deberta/

#### Artifacts created here:

- model.safetensors

- config.json

- tokenizer files

These are required for the multitask warm-start.

### 2️⃣ Train the Multitask (Clarity + Evasion) Model

```bash
python src/train_multitask.py
```

#### This script will:

- Load clarity + evasion datasets

Warm-start the encoder from:

- artifacts/clarity_deberta/


Train the multitask model (shared encoder + two classifier heads)

#### Save outputs to:

artifacts/multitask_deberta_clarity_weights/


#### Files generated include:

- multitask_deberta_state.pt

- multitask_deberta_meta.json

- tokenizer snapshot

- training logs

### 3️⃣ Evaluate the Multitask Model

```bash
python src/eval_multitask_on_val.py
```

This will:

Load:

- multitask_deberta_state.pt

- multitask_deberta_meta.json

- Rebuild the multitask model

#### Print:

- clarity classification report

- evasion classification report

- joint accuracy

#### Export predictions to:

artifacts/multitask_deberta_clarity_weights/val_predictions.csv


The CSV contains:

- question + answer text

- true clarity/evasion labels

- predicted labels


Please contact srinath.m1902@gmail.com if you want the model, I can send you the .zip.
- confidence scores

# 

