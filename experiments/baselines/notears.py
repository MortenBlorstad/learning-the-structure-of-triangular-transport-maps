"""Wrappers around the NOTEARS baselines (installed as the `notears` package,
pinned in pyproject.toml to github.com/xunzheng/notears@4a9ab19).

Exposes a ``.fit(X)`` / ``.get_adjacency_matrix()`` interface mirroring
``triangularmap.sstm.SSTM``, so the experiments can treat NOTEARS and SSTM
interchangeably.

Convention (shared with the NOTEARS ``utils``): ``W[i, j] != 0`` means ``i -> j``
(parent = row). Each wrapper fits at its own ``w_threshold``, so ``W_`` holds the
thresholded weights and ``get_adjacency_matrix`` is just ``W_ != 0``. To vary the
sparsity, refit with a different ``w_threshold`` (or read ``get_weighted_adjacency``
from a ``w_threshold=0`` fit and threshold yourself).

NOTEARS-linear is pure numpy/scipy and dtype-agnostic.
NOTEARS-MLP is the only torch component and runs CPU-only via scipy L-BFGS-B,
which is float64-native; we therefore fit it in float64 and restore the previous
torch default dtype afterwards, so the rest of the harness (SSTM on
float32 / MPS) is unaffected.

Note: this module is imported as ``experiments.baselines.notears``; its ``from
notears import ...`` statements resolve to the *installed* top-level package. Do
not run scripts with ``experiments/baselines/`` on ``sys.path[0]`` (e.g.
``python experiments/baselines/notears.py``) or this file would shadow it.
"""

from __future__ import annotations

import numpy as np
import torch

from notears.nonlinear import NotearsMLP as NotearsMLP_original, notears_nonlinear

import warnings

import igraph as ig

def enforce_dag(W: np.ndarray) -> np.ndarray:
    """Remove a minimum-weight feedback arc set so W becomes a DAG.

    Returns a new DAG-enforced array; the input W is not modified in place.
    """
    dim = W.shape[0]
    if dim <= 20:
        method = "ip"
    else:
        method = "eades"
    W = W.copy()
    G = ig.Graph.Weighted_Adjacency(np.abs(W).tolist())
    if G.is_dag():
        return W
    fas = G.feedback_arc_set(weights="weight", method=method)
    for e in fas:
        i, j = G.es[e].tuple
        W[i, j] = 0.0

    if len(fas) > 2:
        warnings.warn(f"enforce_dag removed {len(fas)} edges; NOTEARS likely did not converge")
    return W


class _NotearsBase:
    """Shared adjacency/thresholding logic for the NOTEARS wrappers."""

    def __init__(self, w_threshold: float = 0.3):
        self.w_threshold = w_threshold
        self.W_: np.ndarray | None = None  # raw continuous weights [d, d]
        self.d_: int | None = None
        self.is_fitted_: bool = False

    def _check_fitted(self):
        if not self.is_fitted_ or self.W_ is None:
            raise RuntimeError("Wrapper is not fitted yet; call .fit(X) first.")

    def get_weighted_adjacency(self) -> np.ndarray:
        """Continuous weighted adjacency ``W[i, j]`` (parent = row), as returned by
        the fit (entries below ``w_threshold`` already zeroed)."""
        self._check_fitted()
        return self.W_.copy()

    def get_adjacency_matrix(self) -> np.ndarray:
        """Binary adjacency ``B[i, j] in {0, 1}``.
        """
        self._check_fitted()
        return (self.W_ != 0).astype(int)


class NotearsMLP(_NotearsBase):
    """Nonlinear NOTEARS-MLP (`NotearsMLP` + `notears_nonlinear`).

    Fit runs in float64: scipy L-BFGS-B is float64-native and `LBFGSBScipy` casts params via `torch.get_default_dtype()`
    on every step, so the *global* default must be double for the duration of the
    fit (`model.double()` alone is overwritten by the optimizer). The previous
    default is restored afterwards, so other models stay float32.
    """

    def __init__(self, hidden: int = 10, bias: bool = True,
                 lambda1: float = 0.01, lambda2: float = 0.01,
                 w_threshold: float = 0.3, max_iter: int = 100):
        super().__init__(w_threshold=w_threshold)
        self.hidden = hidden
        self.bias = bias
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.max_iter = max_iter

    def fit(self, X: np.ndarray) -> "NotearsMLP":
        X = np.asarray(X, dtype=np.float64)
        d = X.shape[1]
        prev_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.double)  # float64 for the L-BFGS-B fit
        
        try:
            model = NotearsMLP_original(dims=[d, self.hidden, 1], bias=self.bias)
            self.W_ = notears_nonlinear(
                model, X, lambda1=self.lambda1, lambda2=self.lambda2,
                max_iter=self.max_iter, w_threshold=self.w_threshold,
            )
        finally:
            torch.set_default_dtype(prev_dtype)  # swap back (float32 for other models)
        self.W_ = enforce_dag(self.W_)
        self.d_ = d
        self.is_fitted_ = True
        return self
