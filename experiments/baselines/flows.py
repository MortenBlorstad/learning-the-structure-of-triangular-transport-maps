"""MAF / RQ-NSF (AR) baselines via nflows — the NSF authors' own library.

Used by the density (UCI) and synthetic (synthetic context models) experiments.
`FlowBase` owns the fit/log_pdf/sample surface they use.
As baselines against our method, the safest implementation is the authors'
released code with their published hyperparameters — no reimplementation choice of ours (tail bound, scale floor,
missing dropout) can be suspected of weakening them.

One deliberate restoration: the pip release of nflows changed the affine scale from
the paper repo's `sigmoid(a+2)+1e-3` (bounded <= 1.001) to `softplus(a)+1e-3`
(unbounded; the sigmoid line is commented out in nflows/transforms/autoregressive.py).
On POWER the unbounded scale compounds through the 10 layers on ~12-sigma tail points
and overflows float32 within the first few steps for some seeds, where the paper's
bounded scale does not. `AffineAutoregressive` below
restores the paper's lines by default and keeps softplus / zuko's bounded log-scale
selectable (`flows.maf_scale`) for ablation.

Composition mirrors the NSF repo's `experiments/uci.py` (Durkan et al. 2019):
per flow step a linear transform (random permutation + LU-linear, identity-init)
followed by the autoregressive base transform; one extra linear transform at the end.
`MAFBaseline` uses the affine autoregressive base transform (= their Table 1 MAF
baseline), `NSFBaseline` the rational-quadratic spline one (tails="linear",
tail_bound 3). Hyperparameters (hidden width, residual blocks, bins, dropout) come
from the paper's Table 5 via the config blocks, not from defaults here.

Smoke test:  uv run python -m experiments.baselines.flows
"""
from __future__ import annotations

import math

import numpy as np
import torch
from torch.nn import functional as F
from nflows.distributions.normal import StandardNormal
from nflows.flows.base import Flow
from nflows.transforms.autoregressive import (
    MaskedAffineAutoregressiveTransform,
    MaskedPiecewiseRationalQuadraticAutoregressiveTransform,
)
from nflows.transforms.base import CompositeTransform
from nflows.transforms.lu import LULinear
from nflows.transforms.permutations import RandomPermutation
from nflows.utils import torchutils


class FlowBase:
    """Wraps an nflows Flow to match the SSTM fit/log_pdf interface.
    Density-experiment surface only: fit, log_pdf, (optional) sample.
    Standardization is done by the caller, not here.
    """

    def __init__(self, transforms: int, hidden_features: int, num_blocks: int = 2,
                 dropout: float = 0.0, device=None):
        self.transforms = int(transforms)
        self.hidden_features = int(hidden_features)
        self.num_blocks = int(num_blocks)
        self.dropout = float(dropout)
        self.flow = None
        self.d_ = None
        self.is_fitted_ = False
        self.device = device or (
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )

    def _base_transform(self, d: int):
        raise NotImplementedError  # subclass returns the per-step autoregressive transform

    def _build_flow(self, d: int) -> Flow:
        def linear():
            return CompositeTransform([RandomPermutation(features=d),
                                       LULinear(d, identity_init=True)])
        steps = [CompositeTransform([linear(), self._base_transform(d)])
                 for _ in range(self.transforms)]
        flow = Flow(CompositeTransform(steps + [linear()]), StandardNormal([d]))
        # StandardNormal registers its log-normalizer buffer as float64, which MPS
        # cannot hold; everything is float32 in this repo anyway.
        return flow.to(torch.float32)

    def fit(self, X: np.ndarray, epochs: int = 200, batch_size: int = 256,
            lr: float = 1e-3, weight_decay: float = 0.0, verbose: bool = False,
            X_val: np.ndarray = None, eval_every: int = 0):
        """NSF-paper training protocol (their uci.py, steps -> epochs): Adam with
        cosine LR annealing to 0 over `epochs`, grad-norm clip 5, and — when X_val is
        given — per-epoch val NLL with best-val checkpointing: the best-val epoch's
        weights are restored at the end, as in their test evaluation.
        eval_every > 0 records train/val NLL every that many epochs into
        self.history_ (diagnostics only; adds eval cost, off by default)."""
        X = np.asarray(X, dtype=np.float32)
        self.d_ = X.shape[1]
        self.flow = self._build_flow(self.d_).to(self.device)

        X_t = torch.tensor(X, device=self.device)
        ds = torch.utils.data.TensorDataset(X_t)
        loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True,
                                             drop_last=True)   # as the NSF paper's loader

        opt = torch.optim.Adam(self.flow.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs, 0)
        self.history_ = {"epoch": [], "nll_train": [], "nll_val": [], "nll_val_median": [],
                         "nll_val_max": []}
        self.is_fitted_ = True          # log_pdf is callable from here on
        self.best_val_nll_ = float("inf")   # NaN never beats inf -> a diverged epoch is
        self.best_epoch_ = None             # never checkpointed (the NSF paper's comparison)
        best_state = None
        # fixed train subset for the history diagnostic (full X is 1.6M rows on POWER)
        idx = np.random.default_rng(0).choice(len(X), size=min(10_000, len(X)), replace=False)
        X_probe = X[idx]
        self.flow.train()
        for epoch in range(epochs):
            total = 0.0
            for (xb,) in loader:
                loss = -self.flow.log_prob(xb).mean()
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(self.flow.parameters(), max_norm=5.0)  # NSF-paper stabilizer
                opt.step()
                total += loss.item()
            scheduler.step()
            lp_val = self._log_pdf_chunked(X_val) if X_val is not None else None  # one val pass, shared
            if lp_val is not None:       # best-val checkpoint (NSF-paper model selection)
                val_nll = float(-np.mean(lp_val))
                if val_nll < self.best_val_nll_:
                    self.best_val_nll_ = val_nll
                    self.best_epoch_ = epoch + 1
                    best_state = {k: v.detach().to("cpu", copy=True) for k, v in self.flow.state_dict().items()}
            if eval_every and ((epoch + 1) % eval_every == 0 or epoch == 0):
                self.history_["epoch"].append(epoch + 1)
                self.history_["nll_train"].append(float(-np.mean(self._log_pdf_chunked(X_probe))))
                if lp_val is not None:
                    self.history_["nll_val"].append(float(-np.mean(lp_val)))
                    self.history_["nll_val_median"].append(float(np.median(-lp_val)))
                    self.history_["nll_val_max"].append(float(np.max(-lp_val)))   # worst tail point
            self.flow.train()           # log_pdf switched to eval mode
            if verbose and (epoch + 1) % 20 == 0:
                print(f"epoch {epoch+1}: nll={total/len(loader):.4f}")

        self.last_state_ = {k: v.detach().to("cpu", copy=True) for k, v in self.flow.state_dict().items()}
        self.best_state_ = best_state
        if best_state is not None:       # evaluate at the best-val epoch, as they do
            self.flow.load_state_dict(best_state)
            self.flow.eval()
        return self

    def _log_pdf_chunked(self, X: np.ndarray, batch: int = 8192) -> np.ndarray:
        """log_pdf in chunks (the UCI val splits run to 149k rows)."""
        return np.concatenate([self.log_pdf(X[i:i + batch]) for i in range(0, len(X), batch)])

    @torch.no_grad()
    def log_pdf(self, X: np.ndarray) -> np.ndarray:
        assert self.is_fitted_, "call fit first"
        self.flow.eval()
        X_t = torch.tensor(np.asarray(X, dtype=np.float32), device=self.device)
        return self.flow.log_prob(X_t).cpu().numpy()

    @torch.no_grad()
    def sample(self, n: int) -> np.ndarray:
        assert self.is_fitted_, "call fit first"
        self.flow.eval()
        return self.flow.sample(n).cpu().numpy()


def _scale_sigmoid(a):                      # NSF paper repo: bounded (1e-3, 1.001]
    return torch.sigmoid(a + 2.0) + 1e-3


def _scale_softplus(a):                     # pip-nflows: unbounded
    return F.softplus(a) + 1e-3


def _scale_zuko(a):                         # zuko MonotonicAffineTransform(slope=1e-3):
    return torch.exp(a / (1 + (a / math.log(1e-3)).abs()))   # [1e-3, 1e3]


def _scale_clamp_e(a):                      # zuko MonotonicAffineTransform(slope=e^-1): the old
    return torch.exp(a / (1 + a.abs()))     # synthetic MAF setting -- scale in (1/e, e),
                                            # init 1; maf_slope study: stable on hub targets


SCALES = {"sigmoid": _scale_sigmoid, "softplus": _scale_softplus, "zuko": _scale_zuko,
          "clamp_e": _scale_clamp_e}


class AffineAutoregressive(MaskedAffineAutoregressiveTransform):
    """nflows' affine AR transform with a selectable scale parameterization (SCALES).
    Only the two `scale = ...` lines differ from the stock class; the default "sigmoid"
    is the NSF paper repo's bounded scale (1e-3, 1.001], which is safer than the pip-nflows unbounded softplus"""

    def __init__(self, *args, scale: str = "sigmoid", **kw):
        super().__init__(*args, **kw)
        self.scale_name = scale
        self._scale_fn = SCALES[scale]

    def _elementwise_forward(self, inputs, autoregressive_params):
        a, shift = self._unconstrained_scale_and_shift(autoregressive_params)
        scale = self._scale_fn(a)
        logabsdet = torchutils.sum_except_batch(torch.log(scale), num_batch_dims=1)
        return scale * inputs + shift, logabsdet

    def _elementwise_inverse(self, inputs, autoregressive_params):
        a, shift = self._unconstrained_scale_and_shift(autoregressive_params)
        scale = self._scale_fn(a)
        logabsdet = -torchutils.sum_except_batch(torch.log(scale), num_batch_dims=1)
        return (inputs - shift) / scale, logabsdet


class MAFBaseline(FlowBase):
    """Affine autoregressive flow — the NSF paper's Table 1 MAF baseline. `scale` picks
    the affine-scale parameterization (SCALES; default = the paper's bounded sigmoid).
    Config knob: uci.flows.maf_scale."""

    def __init__(self, transforms: int, hidden_features: int, num_blocks: int = 2,
                 dropout: float = 0.0, scale: str = "sigmoid", **kw):
        super().__init__(transforms, hidden_features, num_blocks, dropout, **kw)
        self.scale = scale

    def _base_transform(self, d: int):
        return AffineAutoregressive(
            features=d, hidden_features=self.hidden_features,
            num_blocks=self.num_blocks, use_residual_blocks=True,
            dropout_probability=self.dropout, scale=self.scale)


class NSFBaseline(FlowBase):
    """RQ-NSF (AR): rational-quadratic spline autoregressive transform,
    linear tails outside [-tail_bound, tail_bound]."""

    def __init__(self, transforms: int, hidden_features: int, num_blocks: int = 2,
                 dropout: float = 0.0, bins: int = 8, tail_bound: float = 3.0, **kw):
        super().__init__(transforms, hidden_features, num_blocks, dropout, **kw)
        self.bins = int(bins)
        self.tail_bound = float(tail_bound)

    def _base_transform(self, d: int):
        return MaskedPiecewiseRationalQuadraticAutoregressiveTransform(
            features=d, hidden_features=self.hidden_features,
            num_bins=self.bins, tails="linear", tail_bound=self.tail_bound,
            num_blocks=self.num_blocks, use_residual_blocks=True,
            dropout_probability=self.dropout)


if __name__ == "__main__":
    # Smoke test: fit both flows on the 2-D spiral, check nll + sampling.
    # Run: uv run python -m experiments.baselines.flows
    import os
    import matplotlib.pyplot as plt
    from distributions import sample_spiral_distribution
    from experiments.common import compute_nll

    np.random.seed(0)
    torch.manual_seed(0)

    X = sample_spiral_distribution(3000)
    X_train, X_val, X_test = X[:1000], X[1000:2000], X[2000:]
    mu, sd = X_train.mean(axis=0), X_train.std(axis=0)
    X_train, X_val, X_test = ((A - mu) / sd for A in (X_train, X_val, X_test))

    d = X_train.shape[1]
    gauss_ref = 0.5 * d * np.log(2 * np.pi) + 0.5 * np.mean(np.sum(X_test**2, axis=1))
    print(f"standard-Gaussian reference: test nll {gauss_ref:.3f}")

    samples = {}
    for name, model in [("MAF", MAFBaseline(5, 128)), ("NSF", NSFBaseline(5, 128))]:
        model.fit(X_train, X_val=X_val, eval_every=5, verbose=True)
        s = model.sample(1000)
        assert s.shape == (1000, d), s.shape
        samples[name] = s
        print(f"{name}: train nll {compute_nll(model, X_train):.3f}, "
              f"test nll {compute_nll(model, X_test):.3f}")

    os.makedirs("results/uci", exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharex=True, sharey=True)
    panels = [("data", X_test), ("MAF", samples["MAF"]), ("NSF", samples["NSF"])]
    for ax, (title, pts) in zip(axes, panels):
        ax.scatter(pts[:, 0], pts[:, 1], s=4, alpha=0.5, edgecolor="none")
        ax.set_title(title)
    fig.savefig("results/uci/smoke_spiral.png", dpi=130, bbox_inches="tight")
    print("wrote results/uci/smoke_spiral.png")
