# RAG Query Router

Classifies a query as `simple`, `moderate` or `difficult` so a RAG system can pick a retrieval strategy.

| Label | Meaning | Typical routing |
|---|---|---|
| `simple` | single fact or definition | direct answer / single-chunk retrieval |
| `moderate` | explanation, how-to, single comparison | standard RAG, a few chunks |
| `difficult` | multi-concept reasoning, trade-offs, design | multi-hop / agentic RAG |

## Model options

A model is a **feature extractor** plus a **classifier head**, chosen independently in the config:

| Setting | Values | Notes |
|---|---|---|
| `model.type` | `tfidf` | word 1-2 grams + char 2-5 grams, no extra dependencies |
| | `qwen` | frozen Qwen embeddings (mean-pooled, L2-normalised); needs `torch` and `transformers` |
| `model.backend` | `sklearn` | class-balanced Logistic Regression |
| | `jax` | softmax regression (`hidden_dim: 0`) or 1-hidden-layer MLP, trained with JAX + Adam; needs `jax` |

Qwen is never fine-tuned. Only the classifier head is trained.

| Config | Features | Classifier | Output |
|---|---|---|---|
| `configs/config.yaml` | TF-IDF | sklearn | `artifacts/router.joblib` |
| `configs/qwen.yaml` | Qwen | sklearn | `artifacts/router_qwen.joblib` |
| `configs/jax.yaml` | TF-IDF | JAX | `artifacts/router_jax.joblib` |
| `configs/qwen_jax.yaml` | Qwen | JAX | `artifacts/router_qwen_jax.joblib` |

## Structure

```
configs/                  YAML configs (see table above)
data/
  queries.csv             your labeled data (columns: query,label)
  synthetic_queries.csv   1000 synthetic queries generated from the fine-tuning guide
scripts/
  generate_synthetic_queries.py   build the synthetic CSV
  make_sample_data.py             tiny placeholder dataset
src/query_router/
  config.py  data.py      config loading, dataset loading and splitting
  model.py                builds the pipeline (features + classifier)
  qwen.py                 frozen Qwen embedder
  jax_classifier.py       JAX classifier head
  train.py  predict.py    training CLI and inference
tests/                    pytest
artifacts/                saved models and metrics (generated)
```

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH=src        # or: pip install -e .
```

## Train

```bash
python -m query_router.train                              # TF-IDF + sklearn
python -m query_router.train --config configs/qwen.yaml
python -m query_router.train --config configs/jax.yaml
python -m query_router.train --config configs/qwen_jax.yaml
```

Training reports 5-fold cross-validation and test macro-F1, then saves the model to `output.model_path` and metrics, including a confusion matrix, to `output.metrics_path`.
The first Qwen run downloads the weights (about 1.2 GB).

## Predict

```bash
python -m query_router.predict "Compare LoRA and QLoRA for a low-memory GPU" --model artifacts/router_jax.joblib
```

```python
from query_router import QueryRouter

router = QueryRouter("artifacts/router_qwen.joblib")
router.predict("What is RAG?")
# {'label': 'simple', 'confidence': 0.9, 'probabilities': {...}}
```

Route on `label`, and fall back to `difficult` when `confidence` is low.

## Data

Put your labeled queries in a CSV with the columns `query,label`, where the labels are `simple`, `moderate` or `difficult`, and set `data.path` in the config. Different column names can be mapped with `data.query_column` and `data.label_column`.

Regenerate the synthetic set (seeded, reproducible):

```bash
python scripts/generate_synthetic_queries.py --n 1000 --seed 42
```

> The synthetic queries are templated, so scores on them (near 1.00 F1) overstate real-world accuracy.
> Validate on real or human-labeled queries before relying on the router.

## Tests

```bash
pytest
```
