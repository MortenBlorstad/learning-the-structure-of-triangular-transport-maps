"""Paper figures for the synthetic experiments.

- **headline_gap:** gap to the TRUE density (nll_test - nll_true, nats; lower = better) as a
  K (rows) x dataset (cols) grid — learned (ours) against the two-stage baseline and the random
  bound, with the oracle as the dashed reference. Same figure as the gap-to-oracle version; the
  only change is the reference is the true density (nll_true), not the oracle.
  The oracle (dashed) is normally the lowest curve; when it rises *above* random the oracle is
  overfitting and is no longer closest to the truth.

TEST NLL is read straight from the per-run JSON. Styled via `experiments.paper_style`,
saved with `save_paper` (pdf + png) into `results/figures/`.

    python -m experiments.synthetic.plot_results [--results-dir DIR] [--out-dir DIR]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from experiments.common import results_dir
from experiments import paper_style as ps

FIG_DIR = Path("results/figures")

# Column order + display label. SEM datasets map to sampler key f"{key}_e{e}" in the
# JSON (e.g. ER_gauss_e2); funnel has no edge factor.
DATASETS = [
    ("ER_gauss", "ER-Gauss"),
    ("ER_mlp",   "ER-MLP"),
    ("SF_gauss", "SF-Gauss"),
    ("SF_mlp",   "SF-MLP"),
    ("funnel",   "Funnel"),
    ("hierarch",     "Hierarch"),
]
NO_E = ("funnel", "hierarch")   # fixed generative model: no edge factor, own y-scale (not shared with SEM cols)
# Per-panel size (inches). The grid renders at width=\linewidth (\textwidth=6.5in) in the paper,
# so total width PANEL_W*ncol=11.4in downscales ~0.57x. Fonts are set (FIG_RC) so that after that
# downscale the labels land near the ~8-9pt body text -- with 6 columns a plain default is illegible.
PANEL_W, PANEL_H = 2.45, 2.65   # sized so the axes box takes a larger share
                                # of each cell (labels are fixed in points, so they shrink as the
                                # figure downscales), and the taller aspect buys real page height
TITLE_FS = 12
FIG_RC = {"font.size": 13, "axes.labelsize": 14, "xtick.labelsize": 12, "ytick.labelsize": 12,
          "legend.fontsize": 13}
CORE_MODELS = ["oracle", "random", "two_stage", "learned"]   # required: headline hard-errors if any cell is missing
FLOW_MODELS = ["maf", "nsf", "gnf"]                          # optional context: skipped when absent, plotted when complete
STRUCTURE_PLOT_MODELS = ["random", "two_stage", "learned", "gnf"]   # maf/nsf have no structure metrics; gnf does


def load_runs(d: Path) -> list:
    return [json.load(open(f)) for f in sorted(Path(d).glob("*.json"))]


def _sampler_key(dataset: str, e: int) -> str:
    return dataset if dataset in NO_E else f"{dataset}_e{int(e)}"


def line_styles() -> dict:
    """Per-model line kwargs for the saturated models (oracle/true use BOUND_STYLES)."""
    return {
        "learned":   dict(color=ps.C_OURS, ls="-", lw=2.0, marker="o", ms=4, zorder=5),
        "two_stage": dict(color=ps.baseline_color("two_stage"), ls="--", lw=1.4, marker="s", ms=3, zorder=4),
        "random":    dict(**ps.BOUND_STYLES["random"], zorder=3),
        # flow baselines: registry keys are uppercase (avoids baseline_color's hash fallback)
        "maf":       dict(color=ps.baseline_color("MAF"), ls=(0, (3, 1, 1, 1)), lw=1.4, marker="^", ms=3, zorder=2),
        "nsf":       dict(color=ps.baseline_color("NSF"), ls=(0, (1, 1)), lw=1.4, marker="v", ms=3, zorder=2),
        "gnf":       dict(color=ps.baseline_color("GNF"), ls="-.", lw=1.4, marker="<", ms=3, zorder=2),
    }


def _metric_agg(df: pd.DataFrame, ykey: str) -> pd.DataFrame:
    """mean/std of a read-out metric per (sampler, d, n, model)."""
    if ykey not in df.columns:
        raise KeyError(f"results lack '{ykey}'")
    g = (df.groupby(["sampler", "d", "n", "model"])[ykey].agg(["mean", "std"]).reset_index())
    g["std"] = g["std"].fillna(0.0)     # single seed -> no band, not NaN
    return g


def _missing_cells(g: pd.DataFrame, ds, ns, e, models) -> list:
    """Every (dataset, d, n, model) that should be present but has no run — the hard-error list."""
    have = set(zip(g["sampler"], g["d"], g["n"], g["model"]))
    return [f"{_sampler_key(dataset, e)} d={d} n={n} {model}"
            for dataset, _ in DATASETS for d in ds for n in ns for model in models
            if (_sampler_key(dataset, e), d, n, model) not in have]


def _headline_legend(fig, styles, flow_models=()) -> None:
    handles = [Line2D([], [], **{**styles["learned"], "zorder": 1}),
               Line2D([], [], **{**styles["two_stage"], "zorder": 1}),
               Line2D([], [], **ps.BOUND_STYLES["random"]),
               Line2D([], [], **ps.BOUND_STYLES["oracle"])]
    labels = ["Learned (ours)", "Two-stage", "Random", "Oracle"]
    for model in flow_models:
        handles.append(Line2D([], [], **{**styles[model], "zorder": 1}))
        labels.append(model.upper())
    fig.legend(handles, labels, loc="lower center", ncol=len(handles),
               frameon=False, bbox_to_anchor=(0.5, -0.01))


def _overflow_mag(v: float) -> str:
    """Annotation for an off-scale mean: exact below 1e3, order of magnitude above."""
    return f"{v:.0f}" if v < 1e3 else rf"$10^{{{int(np.floor(np.log10(v)))}}}$"


# Models whose overflow is annotated at the top frame instead of scaling the panel:
# panels are scaled to the CORE models so the main comparison stays readable, and the
# flow baselines exit the frame with a marker + magnitude where they overflow.
CAP_EXCLUDE = ("maf", "nsf", "gnf")
CAP_FACTOR = 1.2          # cap ~ the worst core-model mean (the 0.2 gives its band headroom)
# A flow mean is either inside the frame (drawn as a normal curve) or pinned above it
# (marker on the frame + up-arrow magnitude; its in-frame points are still drawn).
ANN_ROW_PT = 9            # row pitch of the stacked overflow annotations (larger value on top)
ANN_PAD_PT = 4            # slack between the last annotation row and the data below it.
                          # Both are kept tight on purpose: the strip is what the frame has
                          # to clear, so a fat stack can make a pin infeasible (its value ends
                          # up BELOW the frame it would need) and the panel grows to contain
                          # the curve instead. Overlap is prevented by the frame rule, not by
                          # this pitch.
ANN_STRIP_MAX = 0.42      # hard ceiling on the strip, so a panel with many overflows still
                          # gives the core models most of its height
ANN_EXTRA = 0.12          # extra frame added AFTER the pin set is fixed, purely to open a gap
                          # between the number stack and the data. Bounded by the lowest
                          # pinned value, so it can never swallow a pin.


def headline_gap(df: pd.DataFrame, ds=(10, 20), e=2, n_grid=(200, 500, 1000), out=FIG_DIR) -> Path:
    """Gap-to-true-density grid: rows = K (=d), columns = dataset. Hard-errors on any
    missing core cell. Each panel's y-axis is capped at 2x the worst mean over the
    non-excluded models; excluded-model values above the cap are marked with that model's
    marker on the top frame plus an up-arrow magnitude annotation."""
    ds, ns = list(ds), sorted(n_grid)
    g = _metric_agg(df, "gap_to_true")
    missing = _missing_cells(g, ds, ns, e, CORE_MODELS)
    if missing:
        raise ValueError("cannot render headline — missing cells:\n  " + "\n  ".join(missing))
    # Flow baselines are optional context: absent -> skipped, complete -> plotted,
    # partial -> hard error (a half-run flow grid must not plot silently).
    flow_models = []
    n_flow_cells = len(DATASETS) * len(ds) * len(ns)
    for model in FLOW_MODELS:
        m = _missing_cells(g, ds, ns, e, [model])
        if not m:
            flow_models.append(model)
        elif len(m) < n_flow_cells:
            raise ValueError(f"flow baseline '{model}' is partially present ({n_flow_cells - len(m)}/{n_flow_cells} cells) — "
                             "finish its grid or clear its cells:\n  " + "\n  ".join(m))
    plot_models = flow_models + ["random", "two_stage", "learned"]
    cap_models = [a for a in CORE_MODELS if a not in CAP_EXCLUDE]

    styles = line_styles()
    nrow, ncol = len(ds), len(DATASETS)
    fig, axes = plt.subplots(nrow, ncol, figsize=(PANEL_W * ncol, PANEL_H * nrow),
                             sharex="col", squeeze=False)

    letters = iter("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    panels = []                       # (ax, cap, overflow): strips are sized after tight_layout
    for r, d in enumerate(ds):
        for c, (dataset, label) in enumerate(DATASETS):
            ax = axes[r][c]
            cell = g[(g["sampler"] == _sampler_key(dataset, e)) & (g["d"] == d)]
            curves = {model: cell[cell["model"] == model].sort_values("n") for model in CORE_MODELS + flow_models}
            ylim0 = CAP_FACTOR * max(curves[a]["mean"].max() for a in cap_models)
            orc = curves["oracle"]
            ax.plot(orc["n"], orc["mean"], **ps.BOUND_STYLES["oracle"], zorder=1)   # oracle reference
            for model in plot_models:
                cv = curves[model]
                ax.plot(cv["n"], cv["mean"], **styles[model])   # never cut: a curve is drawn
                                                              # whole or it is pinned
            panels.append((ax, curves, ylim0))
            ax.set_title(rf"$\mathbf{{{next(letters)}}}$: {label}", loc="left", fontsize=TITLE_FS)
            ax.set_xticks(ns)
            ax.grid(True, alpha=0.25, lw=0.5)
            if r == nrow - 1:
                ax.set_xlabel(r"$N$")
            if c == 0:
                ax.annotate(rf"$K={d}$", xy=(-0.38, 0.5), xycoords="axes fraction",
                            rotation=90, va="center", ha="center", fontsize=ps.LABELSIZE)
    fig.supylabel("gap", x=0.01, fontsize=FIG_RC["axes.labelsize"])
    _headline_legend(fig, styles, flow_models)
    fig.tight_layout(rect=(-0.01, 0.06, 1, 1.0))
    # pass 2: draw the deferred bands and the overflow annotations.
    # The frame IS the cap: 1.2x the worst core-model mean, nothing else. An excluded model
    # above it exits the top and is reported by a marker on the frame plus its magnitude,
    # stacked downward INSIDE the panel, where the top sixth is empty by construction
    # (the cap's 0.2 headroom). Annotating inside is what keeps the cap honest -- the
    # earlier design reserved a strip by raising the frame, which then swallowed the next
    # value up, and each absorption raised it again (funnel K=20 settled at 45 against a
    # cap of 18, squashing the four core models into the bottom quarter).
    renderer = fig.canvas.get_renderer()
    for ax, curves, ylim0 in panels:
        flows = [(n_, v, model) for model in flow_models if model in CAP_EXCLUDE
                 for n_, v in zip(curves[model]["n"], curves[model]["mean"])]
        # The annotation stack is text, so its height is measured in POINTS against the
        # laid-out axes and converted to a fraction of the frame. Rows are counted once,
        # against the BASE cap, and the overflow set is recomputed against the lifted
        # frame exactly once -- so a value that ends up inside is simply drawn, and the
        # frame never grows a second time (the old loop chained until it had swallowed
        # the flows: funnel K=20 settled at 45 against a cap of 18).
        h_pt = ax.get_window_extent(renderer).height / fig.dpi * 72
        core_max = ylim0 / CAP_FACTOR
        # Choose how many of the largest flow means to pin. For each choice the strip is
        # sized from the resulting annotation rows, and the frame must clear BOTH the cap
        # and every curve that stays in view, so the strip is empty of data by
        # construction. Among the valid choices take the smallest frame -- that is the one
        # that leaves the core models the most height.
        ranked = sorted(flows, key=lambda t: -t[1])
        best = None
        for k in range(len(ranked) + 1):
            pins, keep = ranked[:k], ranked[k:]
            rows = max((sum(1 for n2, _, _ in pins if n2 == n_) for n_, _, _ in pins), default=0)
            strip = min((ANN_ROW_PT * rows + ANN_PAD_PT) / h_pt if rows else 0.0, ANN_STRIP_MAX)
            drawn = max([v for _, v, _ in keep] + [core_max])
            frame_k = max(ylim0, drawn / (1 - strip))
            # the second test is an equality by construction when the strip binds,
            # so it needs a tolerance or it fails on rounding and the panel falls
            # back to pinning fewer values and a taller frame
            if all(v > frame_k for _, v, _ in pins) and drawn <= frame_k * (1 - strip) * (1 + 1e-9):
                if best is None or frame_k < best[0]:
                    best = (frame_k, pins)
        frame, over = best
        if over:   # lift the stack clear of the data, but never past the lowest pinned
            frame = min(frame * (1 + ANN_EXTRA),      # value, which must stay CLEARLY above the frame
                        min(v for _, v, _ in over) * 0.95)
        pinned = {model for _, _, model in over}
        for model in plot_models:
            cv = curves[model]
            if model in pinned:
                continue                                       # annotated instead of banded
            ax.fill_between(cv["n"], cv["mean"] - cv["std"],
                            np.minimum(cv["mean"] + cv["std"], frame),
                            color=styles[model]["color"], alpha=0.12, lw=0, zorder=2)
        ax.set_ylim(0, frame)
        # per-N stacking of overflow annotations (maf + nsf can both exit), drawn in
        # value order so at each N the larger overflow sits higher up
        over_rows = {}
        for n_, v, model in sorted(over, key=lambda t: -t[1]):
            row = over_rows.setdefault(n_, 0)
            over_rows[n_] = row + 1
            ax.plot([n_], [frame], marker=styles[model].get("marker", "^"),
                    color=styles[model]["color"], ms=5, ls="none", clip_on=False, zorder=6)
            ax.annotate(rf"$\uparrow${_overflow_mag(v)}", (n_, frame),
                        xytext=(-5 if n_ == ns[0] else 1, -6 - ANN_ROW_PT * row),
                        textcoords="offset points",
                        ha=("left" if n_ == ns[0] else "right"), va="top", fontsize=9,
                        color=styles[model]["color"], clip_on=False)
    out.mkdir(parents=True, exist_ok=True)
    ps.save_paper(fig, str(out / "headline_gap"))
    plt.close(fig)
    return out / "headline_gap.pdf"


def _structure_legend(fig, styles, extra=()) -> None:
    """extra: (style_key, label) pairs appended after Two-stage (e.g. [("gnf", "GNF")])."""
    handles = [Line2D([], [], **{**styles["learned"], "zorder": 1}),
               Line2D([], [], **{**styles["two_stage"], "zorder": 1}),
               *[Line2D([], [], **{**styles[k], "zorder": 1}) for k, _ in extra],
               Line2D([], [], **ps.BOUND_STYLES["random"]),
               Line2D([], [], **ps.BOUND_STYLES["oracle"])]
    labels = ["Learned (ours)", "Two-stage", *[lab for _, lab in extra], "Random", "Oracle"]
    fig.legend(handles, labels, loc="lower center", ncol=len(handles),
               frameon=False, bbox_to_anchor=(0.5, -0.01))


def directed_f1(df: pd.DataFrame) -> pd.Series:
    """Directed F1 from precision = 1 - fdr and recall = tpr, so a reversed edge counts against
    you (unlike skeleton F1) and an empty prediction scores 0 (unlike SHD on a sparse graph)."""
    p, r = 1.0 - df["fdr"], df["tpr"]
    f1 = 2 * p * r / (p + r)
    # empty prediction (p + r == 0) scores 0; rows without structure metrics (flows) stay NaN
    return f1.mask((p + r) == 0, 0.0)




# Radar spokes: (df column, spoke label), starting at 0 deg (right), counter-clockwise.
# Labels are short; the caption defines them (recall = TPR, precision = 1-FDR,
# SHD = inverted normalized SHD, 1 - SHD/(K choose 2)). Short "SHD" sits on the
# right spoke so the side labels stay narrow.
RADAR_AXES = [("shd_score", "SHD"), ("precision", "precision"),
              ("directed_f1", "$F_1$"), ("tpr", "recall")]


def radar_structure(df, ds=(10, 20), e=2, n_show=1000, out=FIG_DIR, fname="radar_structure") -> Path:
    """Radar of the structure metrics standardized to [0,1] (1 = best): recall (TPR),
    precision (1 - FDR), directed F1, and SHD score = 1 - SHD/(d(d-1)/2) (share of
    possible edge errors avoided). Line = mean, band = min-max over seeds; the oracle
    is exactly 1 on every axis (dashed diamond). Panels: K (rows) x dataset (cols),
    fixed N = n_show. NOTE: an empty predicted graph has FDR = 0 by convention, so
    its precision spoke reads 1 (two-stage on the funnel) -- stated in the caption."""
    g = df[df.n == n_show].copy()
    g["precision"] = 1.0 - g["fdr"]
    g["shd_score"] = 1.0 - g["shd"] / (g["d"] * (g["d"] - 1) / 2)
    styles = line_styles()
    ds = list(ds)
    theta = np.linspace(0, 2 * np.pi, len(RADAR_AXES), endpoint=False)
    tc = np.r_[theta, theta[:1]]
    nrow, ncol = len(ds), len(DATASETS)
    fig, axes = plt.subplots(nrow, ncol, figsize=(PANEL_W * ncol, (PANEL_H + 0.5) * nrow),
                             subplot_kw=dict(projection="polar"), squeeze=False)
    letters = iter("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    cols = [c for c, _ in RADAR_AXES]
    for r, d in enumerate(ds):
        for c, (dataset, label) in enumerate(DATASETS):
            ax = axes[r][c]
            cell = g[(g.sampler == _sampler_key(dataset, e)) & (g.d == d)]
            for model in STRUCTURE_PLOT_MODELS:
                gv = cell[cell.model == model][cols]
                if not len(gv):
                    continue
                mean, lo, hi = (np.r_[v, v[:1]] for v in
                                (gv.mean().to_numpy(), gv.min().to_numpy(), gv.max().to_numpy()))
                st = styles[model]
                ax.plot(tc, mean, color=st["color"], ls=st.get("ls", "-"),
                        lw=st.get("lw", 1.5), zorder=st.get("zorder", 3))
                ax.fill(np.r_[tc, tc[::-1]], np.r_[hi, lo[::-1]],
                        color=st["color"], alpha=0.15, lw=0, zorder=1)
            ax.plot(tc, np.ones_like(tc), **ps.BOUND_STYLES["oracle"], zorder=2)   # oracle = 1 on every axis
            ax.set_ylim(0, 1.02)   # slightly negative floor: exact zeros stay visible off-center
            ax.set_xticks(theta)
            ax.set_yticks([0.25, 0.5,0.75, 1.0])
            ax.set_rlabel_position(45)          # radial ticks between the SHD and precision spokes
            ax.set_xticklabels([])              # spoke labels placed manually: per-spoke radius + alignment
            # (angle, radius, ha, va): SHD gets extra clearance; the others hug the ring.
            for ang, (_, lab), rr, ha, va in zip(theta, RADAR_AXES,
                                                 [1.10, 1.1, 1.10, 1.1],
                                                 ["left", "center", "right", "center"],
                                                 ["center", "bottom", "center", "top"]):
                ax.text(ang, rr, lab, ha=ha, va=va, fontsize=10)
            ax.set_yticklabels(["", "", "", "1"], fontsize=8)   # 0.5 ring unlabeled (grazes the oracle diamond)
            ax.tick_params(pad=2)
            ax.set_title(rf"$\mathbf{{{next(letters)}}}$: {label}", loc="left",
                         fontsize=TITLE_FS, y=1.16)   # fixed y: uniform title height across panels
            if c == 0:
                ax.text(-0.38, 0.5, rf"$K={d}$", transform=ax.transAxes, rotation=90,
                        va="center", ha="center", fontsize=FIG_RC["axes.labelsize"])
    _structure_legend(fig, styles, extra=[("gnf", "GNF")] if "gnf" in styles else ())
    fig.subplots_adjust(left=0.005, right=0.99, top=0.86, bottom=0.13, wspace=0.62, hspace=0.25)
    out.mkdir(parents=True, exist_ok=True)
    ps.save_paper(fig, str(out / fname))
    plt.close(fig)
    return out / f"{fname}.pdf"


def main(results_path=None, out_dir=None):
    ps.use_paper_style()
    plt.rcParams.update(FIG_RC)                    # bump fonts for the 6-col grid (see PANEL_W note)
    src = Path(results_path) if results_path else results_dir("synthetic")
    rows = load_runs(src)
    if not rows:
        print(f"no result json found in {src}", flush=True)
        return
    out = Path(out_dir) if out_dir else FIG_DIR
    df = pd.DataFrame(rows)
    df["directed_f1"] = directed_f1(df)
    ds, ng = sorted(df.d.unique()), sorted(df.n.unique())
    # Render whatever n/K the sweep has produced; missing-cell errors flag runs that died.
    print(f"wrote radar    → {radar_structure(df, ds=ds, out=out)} (+ .png)", flush=True)
    print(f"wrote radar200 → {radar_structure(df, ds=ds, out=out, n_show=200, fname='radar_structure_n200')} (+ .png)", flush=True)
    print(f"wrote headline → {headline_gap(df, ds=ds, n_grid=ng, out=out)} (+ .png)", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", default=None)
    p.add_argument("--out-dir", default=None)
    args = p.parse_args()
    main(args.results_dir, args.out_dir)
