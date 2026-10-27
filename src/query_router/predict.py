from __future__ import annotations

import argparse
import json

import joblib

from .config import load_config


class QueryRouter:
    """Loads the trained model and returns a complexity label per query."""

    def __init__(self, model_path: str | None = None):
        self.pipe = joblib.load(model_path or load_config()["output"]["model_path"])

    def predict(self, query: str) -> dict:
        proba = self.pipe.predict_proba([query])[0]
        classes = list(self.pipe.classes_)
        best = int(proba.argmax())
        return {
            "query": query,
            "label": classes[best],
            "confidence": float(proba[best]),
            "probabilities": dict(zip(classes, map(float, proba))),
        }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("query")
    ap.add_argument("--model", default=None)
    args = ap.parse_args()
    print(json.dumps(QueryRouter(args.model).predict(args.query), indent=2))
