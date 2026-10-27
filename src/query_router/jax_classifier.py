from __future__ import annotations

from functools import partial

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin


def _to_dense(X) -> np.ndarray:
    if hasattr(X, "toarray"):
        X = X.toarray()
    return np.asarray(X, dtype=np.float32)


def _forward(params, x):
    import jax
    import jax.numpy as jnp

    for W, b in params[:-1]:
        x = jax.nn.gelu(x @ W + b)
    W, b = params[-1]
    return x @ W + b  # logits


def _make_fns():
    """Build jitted functions once (module-level cache keeps pickling clean)."""
    import jax
    import jax.numpy as jnp

    def loss_fn(params, x, y, w, wd):
        logits = _forward(params, x)
        logp = jax.nn.log_softmax(logits)
        nll = -jnp.take_along_axis(logp, y[:, None], axis=1)[:, 0]
        l2 = sum(jnp.sum(W ** 2) for W, _ in params)
        return jnp.sum(nll * w) / jnp.sum(w) + wd * l2

    @partial(jax.jit, static_argnames=("lr",))
    def step(params, m, v, t, x, y, w, wd, lr):
        grads = jax.grad(loss_fn)(params, x, y, w, wd)
        b1, b2, eps = 0.9, 0.999, 1e-8
        m = jax.tree_util.tree_map(lambda a, g: b1 * a + (1 - b1) * g, m, grads)
        v = jax.tree_util.tree_map(lambda a, g: b2 * a + (1 - b2) * g * g, v, grads)
        mh = jax.tree_util.tree_map(lambda a: a / (1 - b1 ** t), m)
        vh = jax.tree_util.tree_map(lambda a: a / (1 - b2 ** t), v)
        params = jax.tree_util.tree_map(
            lambda p, a, s: p - lr * a / (jnp.sqrt(s) + eps), params, mh, vh)
        return params, m, v

    @jax.jit
    def proba(params, x):
        return jax.nn.softmax(_forward(params, x))

    return step, proba


_FNS = None


def _fns():
    global _FNS
    if _FNS is None:
        _FNS = _make_fns()
    return _FNS


class JaxClassifier(BaseEstimator, ClassifierMixin):
    """Softmax regression (hidden_dim=0) or 1-hidden-layer MLP trained with JAX + Adam."""

    def __init__(self, hidden_dim: int = 0, epochs: int = 200, lr: float = 1e-2,
                 weight_decay: float = 1e-4, batch_size: int = 64,
                 class_weight: str | None = "balanced", seed: int = 0):
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.class_weight = class_weight
        self.seed = seed

    def fit(self, X, y):
        import jax
        import jax.numpy as jnp

        X = _to_dense(X)
        self.classes_, y_idx = np.unique(np.asarray(y), return_inverse=True)
        n, d = X.shape
        k = len(self.classes_)

        counts = np.bincount(y_idx, minlength=k).astype(np.float32)
        cw = n / (k * np.maximum(counts, 1)) if self.class_weight == "balanced" else np.ones(k)
        sw = cw[y_idx].astype(np.float32)

        dims = [d] + ([self.hidden_dim] if self.hidden_dim else []) + [k]
        key = jax.random.PRNGKey(self.seed)
        params = []
        for din, dout in zip(dims[:-1], dims[1:]):
            key, sub = jax.random.split(key)
            W = jax.random.normal(sub, (din, dout)) * np.sqrt(2.0 / (din + dout))
            params.append((W, jnp.zeros(dout)))
        params = tuple(params)
        m = jax.tree_util.tree_map(jnp.zeros_like, params)
        v = jax.tree_util.tree_map(jnp.zeros_like, params)

        step, _ = _fns()
        bs = min(self.batch_size, n)
        n_batches = int(np.ceil(n / bs))
        rng = np.random.default_rng(self.seed)
        t = 0
        for _ in range(self.epochs):
            idx = np.resize(rng.permutation(n), n_batches * bs).reshape(n_batches, bs)
            for b in idx:
                t += 1
                params, m, v = step(params, m, v, float(t), X[b], y_idx[b], sw[b],
                                    self.weight_decay, self.lr)
        # store as numpy so the estimator pickles cleanly
        self.params_ = [(np.asarray(W), np.asarray(b)) for W, b in params]
        return self

    def predict_proba(self, X):
        _, proba = _fns()
        return np.asarray(proba(tuple(self.params_), _to_dense(X)))

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]
