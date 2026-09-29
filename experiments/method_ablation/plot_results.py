"""The ablation figure for the paper: every variant against the default, one row each.

Each variant's cells are divided by the default's on the same (dataset, K, N), so the
x-axis reads as "how much worse than the default". Panel A is the relative gap to the
true density (log scale, 1 = the default), panel B the change in directed F1 (0 = the
default). Points are seed means, coloured by dataset, with the median line and IQR band
across all cells for that variant.

    uv run python -m experiments.method_ablation.plot_results

Output: results/figures/ablation_dotplots.{pdf,png}
"""

from __future__ import annotations

import collections
import glob
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiments import paper_style as ps

RESULTS = Path("results/method_ablation")
OUT = Path("results/figures")


def load_all():
    """sampler-family -> (d, n) -> variant -> {seed: row}."""
    out = collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(dict)))
    for fp in glob.glob(str(RESULTS / "*.json")):
        r = json.load(open(fp))
        fam = r["graph"] if r["graph"] in ("funnel", "hierarch") else f"{r['graph']}_{r['sem']}"
        out[fam][(r["d"], r["n"])][r["model"]][r["seed"]] = r
    return out


DOTPLOT_LABEL = {"no_sharing": "w/o sharing", "plain": "w/o sharing, w/o norm",
                 "row_norm_off": "w/o row norm", "mono_frac_1.0": r"mono $\rho{=}1$",
                 "mono_frac_0.75": r"mono $\rho{=}0.75$", "mono_frac_0.25": r"mono $\rho{=}0.25$",
                 "sinkhorn": "Sinkhorn", "hard_concrete": "hard-concrete"}
DOTPLOT_ORDER = ["w/o sharing, w/o norm", "w/o sharing", "w/o row norm",
                 r"mono $\rho{=}1$", r"mono $\rho{=}0.75$", r"mono $\rho{=}0.25$",
                 "Sinkhorn", "hard-concrete"]
DOTPLOT_FAM = {"ER-Gauss": "#4878CF", "ER-MLP": "#6ACC65", "SF-Gauss": "#D65F5F",
               "SF-MLP": "#B47CC7", "Funnel": "#C4AD66", "Hierarch": "#77BEDB"}
_FAM_KEY = {"ER_gauss": "ER-Gauss", "ER_mlp": "ER-MLP", "SF_gauss": "SF-Gauss",
            "SF_mlp": "SF-MLP", "funnel": "Funnel", "hierarch": "Hierarch"}


def _dotplot_df(data):
    """Long dataframe: one row per (variant, dataset, K, N) = seed-mean gap and dir F1."""
    import pandas as pd
    recs = []
    for fam, dnv in data.items():
        for (d, n), cell in dnv.items():
            for variant, rs in cell.items():
                label = "Default" if variant == "default" else DOTPLOT_LABEL.get(variant)
                if label is None:
                    continue
                recs.append(dict(variant=label, dataset=_FAM_KEY[fam], K=d, N=n,
                                 gap=np.mean([r["gap_to_true"] for r in rs.values()]),   # TEST gap (paper figures report test)
                                 f1=np.mean([r["dir_f1"] for r in rs.values()])))
    return pd.DataFrame(recs)


def _dotplot_draw(ax, df, xcol):
    kmark = {10: "o", 20: "^"}
    ys = {v: len(DOTPLOT_ORDER) - 1 - i for i, v in enumerate(DOTPLOT_ORDER)}
    for v in DOTPLOT_ORDER:
        sub = df[df.variant == v].dropna(subset=[xcol])
        if not len(sub):
            continue
        q1, med, q3 = sub[xcol].quantile([0.25, 0.5, 0.75])
        ax.fill_betweenx([ys[v] - 0.36, ys[v] + 0.36], q1, q3, color="0.55", alpha=0.14,
                         zorder=0, linewidth=0)
        ax.plot([med, med], [ys[v] - 0.36, ys[v] + 0.36], color="black", lw=1.6, zorder=2)
        for di, (dsname, c) in enumerate(DOTPLOT_FAM.items()):
            for K, m in kmark.items():
                for N, off, filled in [(200, +0.16, False), (1000, -0.16, True)]:
                    sl = sub[(sub.dataset == dsname) & (sub.K == K) & (sub.N == N)]
                    rng = np.random.RandomState(di * 1000 + K + N)   # stable jitter
                    j = (rng.rand(len(sl)) - 0.5) * 0.14
                    ax.scatter(sl[xcol], np.full(len(sl), ys[v]) + off + j, s=26, marker=m,
                               facecolors=c if filled else "none", edgecolors=c,
                               linewidths=1.0, alpha=0.9, zorder=3)
    for y in np.arange(0.5, len(DOTPLOT_ORDER) - 0.5):
        ax.axhline(y, color="gray", lw=0.3, alpha=0.35)
    ax.set_yticks([ys[v] for v in DOTPLOT_ORDER])
    ax.set_yticklabels(DOTPLOT_ORDER)
    ax.set_ylim(-0.55, len(DOTPLOT_ORDER) - 0.45)


def _dotplot_legend(fig, anchor=(0.54, 0.02), loc="lower center"):
    from matplotlib.lines import Line2D
    from matplotlib.patches import Rectangle
    h = [Line2D([], [], marker="s", color=c, ls="", ms=6, label=d) for d, c in DOTPLOT_FAM.items()]
    h += [Line2D([], [], marker="o", color="black", ls="", ms=6, label="$K{=}10$"),
          Line2D([], [], marker="^", color="black", ls="", ms=6, label="$K{=}20$"),
          Line2D([], [], marker="s", mfc="none", mec="black", ls="", ms=6, label="$N{=}200$"),
          Line2D([], [], marker="s", color="black", ls="", ms=6, label="$N{=}1000$"),
          Line2D([], [], color="black", lw=1.6, label="median"),
          Rectangle((0, 0), 1, 1, color="0.55", alpha=0.2, label="IQR")]
    fig.legend(handles=h, loc=loc, ncol=6, fontsize=plt.rcParams["legend.fontsize"],
               frameon=False, bbox_to_anchor=anchor, columnspacing=1.0, handletextpad=0.4)


def _rc_for_width(width_in):
    """FIG_RC (tuned for the 6-panel 11.4in canvas printed at \\linewidth) rescaled so a
    canvas of `width_in` inches prints with the SAME font sizes as the other paper figures."""
    from experiments.synthetic.plot_results import FIG_RC, PANEL_W, DATASETS
    ref_w = PANEL_W * len(DATASETS)
    k = width_in / ref_w
    return {key: val * k for key, val in FIG_RC.items()}


def dotplots(data):
    import matplotlib as mpl
    from experiments.synthetic.plot_results import PANEL_W, DATASETS
    W_SINGLE, W_COMBINED = 6.6, PANEL_W * len(DATASETS)   # combined = headline canvas width
    with plt.rc_context(_rc_for_width(W_SINGLE)):
        df = _dotplot_df(data)
        d0 = df[df.variant == "Default"].rename(columns={"gap": "gap_def", "f1": "f1_def"})[
            ["K", "dataset", "N", "gap_def", "f1_def"]]
        dd = df[df.variant != "Default"].merge(d0, on=["K", "dataset", "N"])
        dd["ratio"] = dd.gap / dd.gap_def
        dd["dF1"] = dd.f1 - dd.f1_def

    # --- combined two-panel figure (side by side), headline canvas width ---
    from experiments.synthetic.plot_results import TITLE_FS
    with plt.rc_context(_rc_for_width(W_COMBINED)):
        fig, (axA, axB) = plt.subplots(1, 2, figsize=(W_COMBINED, 3.6), sharey=True,
                                       gridspec_kw={"wspace": 0.06})
        top = len(DOTPLOT_ORDER) - 0.05                 # room inside the frame for the label
        _dotplot_draw(axA, dd, "ratio")
        axA.set_xscale("log")
        axA.axvline(1.0, color="black", ls="--", lw=1.0)
        axA.set_ylim(-0.55, top)
        axA.text(1.0, top - 0.22, "default", ha="center", va="center",
                 fontsize=plt.rcParams["font.size"] * 0.85,
                 bbox=dict(facecolor="white", edgecolor="none", pad=0.5))
        axA.set_xticks([0.5, 1, 2, 5, 10, 20, 50])
        axA.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
        axA.set_xlabel("relative gap")
        axA.set_title(r"$\mathbf{A}$: density", loc="left", fontsize=TITLE_FS)
        _dotplot_draw(axB, dd, "dF1")
        axB.axvline(0.0, color="black", ls="--", lw=1.0)
        axB.set_ylim(-0.55, top)
        axB.text(0.0, top - 0.22, "default", ha="center", va="center",
                 fontsize=plt.rcParams["font.size"] * 0.85,
                 bbox=dict(facecolor="white", edgecolor="none", pad=0.5))
        axB.set_xlabel(r"relative directed $F_1$")
        axB.set_title(r"$\mathbf{B}$: structure", loc="left", fontsize=TITLE_FS)
        axB.tick_params(labelleft=False)
        fig.tight_layout()
        _dotplot_legend(fig, anchor=(0.5, -0.02), loc="upper center")   # below the canvas
        ps.save_paper(fig, str(OUT / "ablation_dotplots"))
        plt.close(fig)


def main():
    """Render the ablation figure used in the paper."""
    ps.use_paper_style()
    dotplots(load_all())
    print(OUT / "ablation_dotplots.pdf")


if __name__ == "__main__":
    main()
