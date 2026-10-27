from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class QwenEmbedder(BaseEstimator, TransformerMixin):
    """Frozen Qwen encoder -> mean-pooled sentence embeddings (sklearn transformer)."""

    def __init__(self, model_name: str = "Qwen/Qwen3-0.6B", batch_size: int = 16,
                 max_length: int = 128, device: str | None = None):
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length
        self.device = device

    def _load(self):
        if getattr(self, "_model", None) is not None:
            return
        import torch
        from transformers import AutoModel, AutoTokenizer

        self._torch = torch
        dev = self.device or ("cuda" if torch.cuda.is_available()
                              else "mps" if torch.backends.mps.is_available() else "cpu")
        self._dev = dev
        self._tok = AutoTokenizer.from_pretrained(self.model_name)
        if self._tok.pad_token is None:
            self._tok.pad_token = self._tok.eos_token
        self._model = AutoModel.from_pretrained(self.model_name).to(dev).eval()

    def __getstate__(self):  # keep saved artifacts small: reload weights lazily
        state = self.__dict__.copy()
        for k in ("_model", "_tok", "_torch", "_dev"):
            state.pop(k, None)
        return state

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        self._load()
        torch = self._torch
        texts = list(X)
        out = []
        with torch.no_grad():
            for i in range(0, len(texts), self.batch_size):
                enc = self._tok(texts[i:i + self.batch_size], padding=True, truncation=True,
                                max_length=self.max_length, return_tensors="pt").to(self._dev)
                hidden = self._model(**enc).last_hidden_state
                mask = enc["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
                out.append(pooled.float().cpu().numpy())
        emb = np.vstack(out)
        return emb / np.clip(np.linalg.norm(emb, axis=1, keepdims=True), 1e-9, None)
