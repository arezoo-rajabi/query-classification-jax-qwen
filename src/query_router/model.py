from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline


def _tfidf_features(m: dict):
    return FeatureUnion([
        ("word", TfidfVectorizer(
            ngram_range=tuple(m["word_ngram_range"]), min_df=m["min_df"],
            lowercase=True, sublinear_tf=True)),
        ("char", TfidfVectorizer(
            analyzer="char_wb", ngram_range=tuple(m["char_ngram_range"]),
            min_df=m["min_df"], sublinear_tf=True)),
    ])


def _qwen_features(m: dict):
    from .qwen import QwenEmbedder  # lazy: torch/transformers only needed here
    q = m.get("qwen", {})
    return QwenEmbedder(
        model_name=q.get("model_name", "Qwen/Qwen3-0.6B"),
        batch_size=q.get("batch_size", 16),
        max_length=q.get("max_length", 128),
        device=q.get("device"),
    )


FEATURES = {"tfidf": _tfidf_features, "qwen": _qwen_features}


def _sklearn_classifier(m: dict):
    return LogisticRegression(
        C=m["C"], class_weight=m["class_weight"], max_iter=m["max_iter"])


def _jax_classifier(m: dict):
    from .jax_classifier import JaxClassifier  # lazy: jax only needed here
    j = m.get("jax", {})
    return JaxClassifier(
        hidden_dim=j.get("hidden_dim", 0),
        epochs=j.get("epochs", 200),
        lr=j.get("lr", 1e-2),
        weight_decay=j.get("weight_decay", 1e-4),
        batch_size=j.get("batch_size", 64),
        class_weight=m.get("class_weight"),
        seed=j.get("seed", 0),
    )


CLASSIFIERS = {"sklearn": _sklearn_classifier, "jax": _jax_classifier}


def build_pipeline(cfg: dict) -> Pipeline:
    m = cfg["model"]
    kind = m.get("type", "tfidf")
    backend = m.get("backend", "sklearn")
    if kind not in FEATURES:
        raise ValueError(f"Unknown model.type '{kind}'. Choose from {list(FEATURES)}")
    if backend not in CLASSIFIERS:
        raise ValueError(f"Unknown model.backend '{backend}'. Choose from {list(CLASSIFIERS)}")
    return Pipeline([("features", FEATURES[kind](m)), ("clf", CLASSIFIERS[backend](m))])
