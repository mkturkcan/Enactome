"""Array-backend abstraction: run any neuron model on CPU (numpy) or GPU (torch).

The neuron models below are written against a tiny `Backend` shim so the *same* dynamics
code runs on numpy (always available, CPU) or torch (optional, CPU **or** CUDA GPU). Selecting
a backend is the single switch the UI exposes for "run this on CPU vs GPU".
"""
from __future__ import annotations
import numpy as np


class Backend:
    """Minimal array API wrapping numpy or torch, with a device."""

    def __init__(self, kind="numpy", device="cpu", dtype="float32"):
        self.kind = kind
        self.device = device
        if kind == "numpy":
            self.xp = np
            self._dtype = np.float32 if dtype == "float32" else np.float64
        elif kind == "torch":
            import torch
            self.torch = torch
            self.xp = torch
            self._dtype = torch.float32 if dtype == "float32" else torch.float64
            if device == "cuda" and not torch.cuda.is_available():
                raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
        else:
            raise ValueError(f"unknown backend {kind!r}")

    # --- array construction ---
    def array(self, x):
        if self.kind == "numpy":
            return np.asarray(x, dtype=self._dtype)
        return self.torch.as_tensor(np.asarray(x, dtype=np.float32),
                                    dtype=self._dtype, device=self.device)

    def zeros(self, shape):
        if self.kind == "numpy":
            return np.zeros(shape, dtype=self._dtype)
        return self.torch.zeros(shape, dtype=self._dtype, device=self.device)

    def zeros_like(self, x):
        return self.xp.zeros_like(x)

    def randn(self, shape, rng=None):
        if self.kind == "numpy":
            rng = rng or np.random.default_rng()
            return rng.standard_normal(shape).astype(self._dtype)
        return self.torch.randn(shape, dtype=self._dtype, device=self.device)

    # --- elementwise / reductions used by the models ---
    def maximum(self, a, b):
        if self.kind == "numpy":
            return np.maximum(a, b)
        return self.torch.clamp(a, min=b) if np.isscalar(b) else self.torch.maximum(a, self.array(b))

    def clip(self, x, lo, hi):
        return (np.clip(x, lo, hi) if self.kind == "numpy"
                else self.torch.clamp(x, lo, hi))

    def where(self, cond, a, b):
        return self.xp.where(cond, a, b)

    def exp(self, x):
        return self.xp.exp(x)

    def tanh(self, x):
        return self.xp.tanh(x)

    def matmul(self, W, r):
        return self.xp.matmul(W, r) if self.kind == "numpy" else self.torch.matmul(W, r)

    def to_numpy(self, x):
        if self.kind == "numpy":
            return np.asarray(x)
        return x.detach().cpu().numpy()


def get_backend(prefer_gpu=False, dtype="float32"):
    """Return the best available backend. prefer_gpu=True picks torch+cuda if present,
    else torch+cpu, else numpy."""
    if prefer_gpu:
        try:
            import torch
            dev = "cuda" if torch.cuda.is_available() else "cpu"
            return Backend("torch", dev, dtype)
        except Exception:
            pass
    return Backend("numpy", "cpu", dtype)
