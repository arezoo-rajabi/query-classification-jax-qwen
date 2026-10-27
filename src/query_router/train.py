from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import joblib
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import cross_val_score

from .config import load_config
from .data import load_dataset, split
from .model import build_pipeline

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("train")


def main(config_path: str | None = None) -> dict:
    cfg = load_config(config_path) if config_path else load_config()
    df = load_dataset(cfg)
    log.info("Loaded %d queries\n%s", len(df), df["label"].value_counts().to_string())

    X_train, X_test, y_train, y_test = split(df, cfg)
    pipe = build_pipeline(cfg)

    cv = cross_val_score(pipe, X_train, y_train, cv=5, scoring="f1_macro")
    log.info("CV macro-F1: %.3f ± %.3f", cv.mean(), cv.std())

    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    labels = cfg["labels"]
    metrics = {
        "cv_macro_f1_mean": float(cv.mean()),
        "cv_macro_f1_std": float(cv.std()),
        "test_macro_f1": float(f1_score(y_test, pred, average="macro")),
        "report": classification_report(y_test, pred, labels=labels, output_dict=True),
        "confusion_matrix": confusion_matrix(y_test, pred, labels=labels).tolist(),
        "labels": labels,
    }
    log.info("\n%s", classification_report(y_test, pred, labels=labels))

    out = cfg["output"]
    for key in ("model_path", "metrics_path"):
        Path(out[key]).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, out["model_path"])
    Path(out["metrics_path"]).write_text(json.dumps(metrics, indent=2))
    log.info("Saved model to %s", out["model_path"])
    return metrics


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    main(ap.parse_args().config)
