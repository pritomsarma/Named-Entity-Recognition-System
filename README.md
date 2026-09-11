# 🏷️ Named-Entity-Recognition-System

**A BERT-based NER engine for extracting people, organizations, locations & miscellaneous entities from text**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/🤗%20Transformers-4.30+-yellow.svg)](https://huggingface.co/docs/transformers)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-ff4b4b.svg?logo=streamlit&logoColor=white)](https://streamlit.io/)

*Fine-tunes a pretrained BERT encoder for token-level Named Entity Recognition, decoded with an optional CRF layer, evaluated with precision / recall / F1.*

---

## 📌 Overview

This project fine-tunes **`bert-base-uncased`** on the **CoNLL-2003** dataset to identify four entity types in raw text:

| Tag | Meaning |
|-----|---------|
| **PER** | Person |
| **ORG** | Organization |
| **LOC** | Location |
| **MISC** | Miscellaneous (events, nationalities, products, etc.) |

Entities are recovered from BIO-tagged token predictions, and a **CRF (Conditional Random Field)** layer sits on top of the linear classifier to enforce valid tag transitions (e.g. an `I-PER` tag can only follow `B-PER`/`I-PER`), which noticeably cleans up the output compared to raw argmax decoding.

## 🚀 Features

- **🧠 Fine-tuned BERT encoder** — `bert-base-uncased` + dropout + linear classifier, with an optional CRF head for structured decoding.
- **📈 Discriminative two-phase fine-tuning** — freeze the encoder and warm up the classifier head first, then unfreeze with a lower encoder learning rate (inspired by ULMFiT) to avoid catastrophic forgetting.
- **🛡️ Built-in overfitting controls** — dropout, label smoothing, weight decay, gradient clipping, and early stopping, all configurable from the CLI.
- **⚙️ Fully typed configuration** — every hyperparameter, path, and label mapping lives in `config.py` as dataclasses, so nothing is hard-coded elsewhere.
- **📊 Evaluation with `seqeval`** — entity-level precision, recall, and F1, not just token accuracy.
- **💻 CLI training & inference** — `train.py` and `predict.py` support single sentences, batch files, and a built-in broadcast-news demo set.
- **🖥️ Streamlit dashboard** (`app.py`) — paste text, get color-coded entity highlighting, confidence scores, and a results table, right in the browser.
- **📉 TensorBoard logging** and JSON training-history export for every run.

## 🏗️ Architecture

```
graph TD
    A["📝 Raw Text"] -->|"BERT Tokenizer"| B("1. Sub-word Tokenization")
    B -->|"Token IDs"| C("2. BERT Encoder")
    C -->|"Contextual Embeddings (768-dim)"| D("3. Dropout")
    D -->|"Regularised Features"| E("4. Linear Classifier")
    E -->|"Per-token Label Logits"| F{"CRF Enabled?"}
    F -->|"Yes"| G("5a. CRF Decoding")
    F -->|"No"| H("5b. Argmax Decoding")
    G --> I["🏷️ BIO Tags → Entities"]
    H --> I

    style A fill:#e2e2e2,stroke:#333,stroke-width:2px,color:#000
    style B fill:#457B9D,stroke:#333,stroke-width:2px,color:#fff
    style C fill:#E07A5F,stroke:#333,stroke-width:2px,color:#fff
    style D fill:#F2CC8F,stroke:#333,stroke-width:2px,color:#000
    style E fill:#8B7E74,stroke:#333,stroke-width:2px,color:#fff
    style F fill:#B5838D,stroke:#333,stroke-width:2px,color:#fff
    style G fill:#81B29A,stroke:#333,stroke-width:2px,color:#000
    style H fill:#81B29A,stroke:#333,stroke-width:2px,color:#000
    style I fill:#3D2C2E,stroke:#333,stroke-width:2px,color:#fff
```

### 📂 Codebase mapping

| File | Purpose |
|------|---------|
| [`config.py`](./config.py) | Central typed configuration — model, training, data, and path settings (`NERConfig`) |
| [`train.py`](./train.py) | Training entry point — standard or two-phase discriminative fine-tuning, with full CLI control over LR, dropout, label smoothing, patience, etc. |
| [`resume_train.py`](./resume_train.py) | Resume training from a saved checkpoint |
| [`predict.py`](./predict.py) | Inference entry point — run on a single string, a file of sentences, or a built-in demo set; optional analytics report |
| [`eval_test.py`](./eval_test.py) | Standalone evaluation script — precision / recall / F1 on the test split via `seqeval` |
| [`app.py`](./app.py) | Streamlit dashboard for interactive, browser-based entity extraction |
| `src/data/` | Dataset loading & `DataLoader` construction (`ner_dataset.py`) |
| `src/model/` | The `BroadcastNERModel` architecture (BERT + classifier + CRF) |
| `src/training/` | The `NERTrainer` training/evaluation loop |
| `src/inference/` | `BroadcastNERPredictor` and analytics utilities |
| [`requirements.txt`](./requirements.txt) | Project dependencies |

## 🛠️ Setup & Installation

**Prerequisites:** Python 3.9+, pip, and (optionally) a CUDA-capable GPU for faster training.

**1. Clone the repository**
```bash
git clone https://github.com/pritomsarma/Named-Entity-Recognition-System.git
cd Named-Entity-Recognition-System
```

**2. Create a virtual environment**
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

## 🏋️ Training

Train with sensible defaults (CoNLL-2003, 4 epochs, quick dataset subset):
```bash
python train.py
```

Recommended: two-phase discriminative fine-tuning:
```bash
python train.py --fine_tune --freeze_epochs 2 --epochs 6
```

Stronger regularisation, to fight overfitting on small subsets:
```bash
python train.py --label_smoothing 0.15 --dropout 0.4 --patience 4
```

GPU training with mixed precision:
```bash
python train.py --device cuda --amp --fine_tune
```

Full CoNLL-2003 dataset (all ~14k training examples — several hours on CPU):
```bash
python train.py --full_data --epochs 10 --fine_tune
```

Training saves the best checkpoint to `checkpoints/best_model.pt`, logs to TensorBoard under `runs/`, and writes a JSON training history to `output/training_history.json`.

## 🔮 Inference

Run the built-in broadcast-news demo:
```bash
python predict.py --demo
```

Predict on custom text:
```bash
python predict.py --model_path checkpoints/best_model.pt --text "Anderson Cooper reported from New York on CNN."
```

Predict on a file (one sentence per line), with analytics and JSON export:
```bash
python predict.py --model_path checkpoints/best_model.pt --file transcripts.txt --analytics --output results.json
```

## 🖥️ Interactive Dashboard

Launch the Streamlit UI for point-and-click entity extraction:
```bash
streamlit run app.py
```

Paste or type any text and click **Analyze Entities** to see color-coded, confidence-scored entities highlighted inline, plus a summary table.

## 📊 Evaluation

Run standalone evaluation (precision, recall, F1 via `seqeval`) on the held-out test split:
```bash
python eval_test.py
```

## ⚙️ Key Configuration Defaults

| Setting | Default |
|---------|---------|
| Base model | `bert-base-uncased` |
| Max sequence length | 128 tokens |
| Dropout | 0.3 |
| CRF decoding | Enabled |
| Encoder LR / Classifier LR | 2e-5 / 5e-4 |
| Label smoothing | 0.1 |
| Weight decay | 0.01 |
| Early-stopping patience | 3 epochs |
| Batch size | 16 |

All of these are overridable via CLI flags on `train.py` or by editing `config.py` directly.

## 🤝 Contributing

Contributions, issues, and feature requests are welcome — feel free to open an issue or a pull request.

## 📝 License

No license has been specified yet for this repository. Consider adding one (e.g. MIT) if you plan to share or accept contributions.
