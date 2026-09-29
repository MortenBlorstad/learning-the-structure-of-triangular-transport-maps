"""TrainingLogger — per-epoch introspection of SSTM training dynamics.

SSTM jointly learns a variable ordering, within-row sparsity, and a
monotone triangular map on annealed schedules. This logger records a per-epoch
history of the internal mechanics so the *sequencing* of learning is visible
(e.g. did the order lock before sparsity settled; is the map saturating before it
shows in the loss).

Usage:
    logger = TrainingLogger(log_every=10)
    model.fit(X, epochs=500, ..., logger=logger)
    logger.save()            # <out_dir>/<run_name>.json

Design: `fit` hands over the cheap scalars it already has each epoch (mean
NLL/L0/entropy + schedule values incl. the l1/l2 coefficients); the logger derives the structural and
map-health metrics itself from the model and a fixed probe batch. The separated
loss terms are always logged — never only the total.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

_DEFAULT_OUT = Path(__file__).resolve().parents[2] / "results" / "logs"


class TrainingLogger:
    """Records the per-epoch history described above; `save()` writes it as json."""

    def __init__(self, log_every: int = 10, out_dir=None, run_name: str = "run",
                 probe_size: int = 512):
        self.log_every = log_every
        self.out_dir = Path(out_dir) if out_dir else _DEFAULT_OUT
        self.run_name = run_name
        self.probe_size = probe_size
        self.warmup = None
        self.history: list = []          # one dict per logged epoch
        self._model = None
        self._probe = None               # fixed probe batch, tensor [probe_size, d]
        self._val_probe = None           # fixed val probe (optional), for val NLL
        self._prev_order = None          # for order-lock detection

    # -- lifecycle ---------------------------------------------------------- #
    @torch.no_grad()
    def start(self, model, X, warmup: int = 50, X_val=None):
        """Attach the model and freeze a train (and optional val) probe batch."""
        self._model = model
        self.warmup = warmup
        n = min(self.probe_size, len(X))
        self._probe = torch.as_tensor(np.asarray(X[:n], dtype=np.float32),
                                      device=model.device)
        self._val_probe = None
        if X_val is not None and len(X_val):
            nv = min(self.probe_size, len(X_val))
            self._val_probe = torch.as_tensor(np.asarray(X_val[:nv], dtype=np.float32),
                                              device=model.device)

    def should_log(self, epoch: int) -> bool:
        return epoch % self.log_every == 0

    @torch.no_grad()
    def log_epoch(self, epoch: int, loss_terms: dict, sched: dict):
        """Append one epoch record: fit-provided scalars + derived metrics."""
        rec = {"epoch": epoch}
        rec.update(loss_terms)
        rec.update(sched)
        rec["sparsity_loss"] = loss_terms["l0"] * sched["sparsity_strength"]
        rec["total"] = loss_terms["nll"] + rec["sparsity_loss"]   # l1 is decoupled in the optimizer, not in the loss
        rec.update(self._ordering_metrics())
        rec.update(self._sparsity_metrics())
        rec.update(self._map_health_metrics())
        rec.update(self._val_metrics())
        self.history.append(rec)

    # -- metric groups (each independent + readable) ------------------------ #
    def _ordering_metrics(self) -> dict:
        o = self._model.ordering
        order = self._model._hard_order().cpu().numpy()
        prev = self._prev_order
        hamming = None if prev is None else int((order != prev).sum())
        self._prev_order = order

        margin = None
        if o.perm_method == "softsort":
            s = np.sort(o.logits.detach().cpu().numpy().ravel())[::-1]
            if s.size > 1:
                margin = float(np.diff(s).min() * -1)   # min gap between adjacent ranks (>=0)
        return {
            "hard_order": order.tolist(),
            "order_changed": bool(hamming != 0) if hamming is not None else True,
            "order_hamming": hamming,
            "logit_margin_min": margin,
        }

    def _sparsity_metrics(self) -> dict:
        o = self._model.ordering
        if not o.use_sparsity or o.use_fixed_sparsity:
            return {"keep_prob_mean": None, "gate_active_frac": None}
        q = o.keep_prob(o.gate_affinity).detach().cpu().numpy().ravel()
        return {"keep_prob_mean": float(q.mean()),
                "gate_active_frac": float((q > 0.5).mean())}

    def _map_health_metrics(self) -> dict:
        model = self._model
        was_training = model.training
        model.eval()
        try:
            z, log_jac = model.forward_with_jac(self._probe)      # committed (hard) map
        finally:
            model.train(was_training)

        diag = log_jac.exp()                        # |dT_i/dx_i| per dim
        logdet = log_jac.sum(dim=1)                 # log|det J| per sample
        z_np = z.detach().cpu().numpy()
        z_mean, z_std = z_np.mean(axis=0), z_np.std(axis=0)
        n_nan = int(torch.isnan(z).sum() + torch.isnan(log_jac).sum())
        n_inf = int(torch.isinf(z).sum() + torch.isinf(log_jac).sum())
        return {
            "train_nll_eval": float(model.triangular_transport_loss_fast(z, log_jac)),  # hard map, comparable to val_nll
            "mono_margin_min": float(diag.min()),
            "mono_margin_mean": float(diag.mean()),
            "logdetJ_mean": float(logdet.mean()),
            "logdetJ_std": float(logdet.std()),
            "logdetJ_min": float(logdet.min()),
            "logdetJ_max": float(logdet.max()),
            "n_nan": n_nan,
            "n_inf": n_inf,
            "z_mean": z_mean.tolist(),
            "z_std": z_std.tolist(),
            "z_mean_absmax": float(np.abs(z_mean).max()),
            "z_std_dev_max": float(np.abs(z_std - 1.0).max()),
        }

    def _val_metrics(self) -> dict:
        """Held-out val NLL under the committed (hard) map — comparable to train_nll_eval."""
        if self._val_probe is None:
            return {"val_nll": None}
        model = self._model
        was_training = model.training
        model.eval()
        try:
            z, log_jac = model.forward_with_jac(self._val_probe)
            val_nll = float(model.triangular_transport_loss_fast(z, log_jac))
        finally:
            model.train(was_training)
        return {"val_nll": val_nll}

    # -- output ------------------------------------------------------------- #

    def save(self, path=None) -> Path:
        path = Path(path) if path else self.out_dir / f"{self.run_name}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump({"log_every": self.log_every, "warmup": self.warmup,
                       "history": self.history}, f, indent=2)
        return path
