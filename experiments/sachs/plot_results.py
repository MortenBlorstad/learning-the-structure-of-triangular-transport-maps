"""Sachs figures: NLL scatter + structure radar, two panels each (pooled | observational).

- ``sachs_nll``:     test NLL. Models on the x-axis (house colours/markers), per-seed
                     points behind a mean +- one-std errorbar. A diverged flow seed is
                     excluded from the mean and pinned to the top frame with the house
                     "$\\uparrow$mag" annotation.
- ``sachs_radar``:   the house structure radar (radar_structure): spokes = SHD score
                     (1 - SHD/(d(d-1)/2)), precision (1 - FDR), directed F1, recall (TPR),
                     all in [0,1] with 1 best; line = mean, band = min-max over seeds;
                     the oracle is the dashed unit ring.

Chrome mirrors experiments/synthetic/plot_results.py (FIG_RC fonts, framed grid
panels, bold-letter titles, house legends). NLL panels keep per-dataset y-axes; levels are
not comparable across datasets (different target distribution and standardization).

Writes to results/figures/.

    uv run python -m experiments.sachs.plot_results [--out-dir DIR]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments import paper_style as ps
from experiments.synthetic.plot_results import (
    FIG_RC, TITLE_FS, RADAR_AXES, line_styles, _overflow_mag, _structure_legend)

MODELS = ["learned", "two_stage", "random", "oracle", "gnf", "maf", "nsf"]
LABELS = {"learned": "Learned", "two_stage": "Two-stage", "random": "Random",
          "oracle": "Oracle", "gnf": "GNF", "maf": "MAF", "nsf": "NSF"}
DATASETS = [("pooled", "pooled"), ("obs853", "observational")]
STRUCTURE_MODELS = ["random", "two_stage", "gnf", "learned"]
D = 11


def styles() -> dict:
    st = line_styles()
    st["oracle"] = dict(**ps.BOUND_STYLES["oracle"], marker="D", ms=4, zorder=3)
    st["random"] = dict(st["random"], marker="P", ms=4)   # marker identity for the point panel
    return st


def load(results_dir):
    rows = {}
    for f in sorted(Path(results_dir).glob("sachs_*.json")):
        r = json.load(f.open())
        rows.setdefault((r["subset"], r["model"]), []).append(r)
    return rows


def fig_nll(rows, cap=22.5):
    st = styles()
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.6))   # 12in wide -> \linewidth shrink ~0.54, house font scale
    for ax, (subset, label), letter in zip(axes, DATASETS, "AB"):
        for i, m in enumerate(MODELS):
            v = np.array([r["nll_test"] for r in rows[(subset, m)]], dtype=float)
            keep = v[v <= cap]
            marker = st[m].get("marker", "o")
            if len(keep):
                jitter = (np.arange(len(keep)) - (len(keep) - 1) / 2) * 0.07
                ax.plot(i + jitter, keep, ls="none", marker=marker, ms=3.5,
                        color=st[m]["color"], alpha=0.45, zorder=2)
                ax.errorbar(i, keep.mean(), yerr=keep.std(ddof=1) if len(keep) > 1 else 0,
                            fmt=marker, ms=6, color=st[m]["color"], capsize=3,
                            elinewidth=1.4, zorder=st[m].get("zorder", 3))
            over = sorted(v[v > cap])
            if len(over) >= 3:                  # several overflows: one pin + the range
                                                # (individual values carry no information)
                ax.plot([i], [1.0], marker=marker, color=st[m]["color"], ms=5, ls="none",
                        transform=ax.get_xaxis_transform(), clip_on=False, zorder=6)
                lab = (_overflow_mag(over[0]) if len(over) == 1 else
                       f"{_overflow_mag(over[0])}--{_overflow_mag(over[-1])}")
                ax.annotate(rf"$\uparrow${lab}", (i, 1.0),
                            xycoords=ax.get_xaxis_transform(), xytext=(0, -6),
                            textcoords="offset points", ha="center", va="top",
                            fontsize=9, color=st[m]["color"], clip_on=False)
            else:
                for k, c in enumerate(over):    # a few overflows: pin each, staggered
                    x = i + (k - (len(over) - 1) / 2) * 0.24
                    ax.plot([x], [1.0], marker=marker, color=st[m]["color"], ms=5, ls="none",
                            transform=ax.get_xaxis_transform(), clip_on=False, zorder=6)
                    ax.annotate(rf"$\uparrow${_overflow_mag(c)}", (x, 1.0),
                                xycoords=ax.get_xaxis_transform(),
                                xytext=(0, -6 - 11 * (k % 2)),
                                textcoords="offset points", ha="center", va="top",
                                fontsize=9, color=st[m]["color"], clip_on=False)
        lo_, hi_ = ax.get_ylim()                # headroom: keep dots clear of the frame-pinned
        ax.set_ylim(lo_, hi_ + 0.12 * (hi_ - lo_))   # overflow labels hanging below the top
        ax.set_xticks(range(len(MODELS)))
        ax.set_xticklabels([LABELS[m] for m in MODELS], fontsize=FIG_RC["xtick.labelsize"] - 1,
                           rotation=25, ha="right")
        ax.set_xlim(-0.6, len(MODELS) - 0.4)
        ax.grid(True, alpha=0.25, lw=0.5)
        ax.set_title(rf"$\mathbf{{{letter}}}$: {label}", fontsize=TITLE_FS)
    fig.supylabel("NLL", x=0.01, fontsize=FIG_RC["axes.labelsize"])
    fig.tight_layout(rect=(0.03, 0, 1, 1))
    return fig


def fig_radar(rows):
    st = styles()
    theta = np.linspace(0, 2 * np.pi, len(RADAR_AXES), endpoint=False)
    tc = np.r_[theta, theta[:1]]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), subplot_kw=dict(projection="polar"))
    key_map = {"shd_score": lambda r: 1.0 - r["shd"] / (D * (D - 1) / 2),
               "precision": lambda r: 1.0 - r["fdr"],
               "directed_f1": lambda r: r["dir_f1"],
               "tpr": lambda r: r["tpr"]}
    for ax, (subset, label), letter in zip(axes, DATASETS, "AB"):
        for m in STRUCTURE_MODELS:
            vals = np.array([[key_map[c](r) for c, _ in RADAR_AXES]
                             for r in rows[(subset, m)]])
            mean, lo, hi = (np.r_[v, v[:1]] for v in
                            (vals.mean(0), vals.min(0), vals.max(0)))
            s = st[m]
            ax.plot(tc, mean, color=s["color"], ls=s.get("ls", "-"),
                    lw=s.get("lw", 1.5), zorder=s.get("zorder", 3))
            ax.fill(np.r_[tc, tc[::-1]], np.r_[hi, lo[::-1]],
                    color=s["color"], alpha=0.15, lw=0, zorder=1)
        ax.plot(tc, np.ones_like(tc), **ps.BOUND_STYLES["oracle"], zorder=2)   # oracle = 1 on every axis
        ax.set_ylim(0, 1.02)
        ax.set_xticks(theta)
        ax.set_yticks([0.25, 0.5, 0.75, 1.0])
        ax.set_rlabel_position(45)
        ax.set_xticklabels([])
        for ang, (_, lab), rr, ha, va in zip(theta, RADAR_AXES,
                                             [1.10, 1.1, 1.10, 1.1],
                                             ["left", "center", "right", "center"],
                                             ["center", "bottom", "center", "top"]):
            ax.text(ang, rr, lab, ha=ha, va=va, fontsize=10)
        ax.set_yticklabels(["", "", "", "1"], fontsize=8)
        ax.tick_params(pad=2)
        ax.set_title(rf"$\mathbf{{{letter}}}$: {label}", loc="left", fontsize=TITLE_FS, y=1.16)
    _structure_legend(fig, st, extra=[("gnf", "GNF")])
    fig.subplots_adjust(left=0.24, right=0.76, top=0.80, bottom=0.14, wspace=0.6)
    # pin the full canvas width against save_paper's tight bbox (else the side margins are
    # cropped and the \linewidth shrink factor -- hence the printed font size -- changes)
    fig.add_artist(plt.Line2D([0.001, 0.999], [0.5, 0.5], transform=fig.transFigure,
                              color="none", alpha=0.0))
    return fig


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", default="results/sachs")
    p.add_argument("--out-dir", default="results/figures")
    a = p.parse_args()
    rows = load(a.results_dir)
    ps.use_paper_style()
    plt.rcParams.update(FIG_RC)
    dest = Path(a.out_dir)
    dest.mkdir(parents=True, exist_ok=True)
    ps.save_paper(fig_nll(rows), str(dest / "sachs_nll"))
    ps.save_paper(fig_radar(rows), str(dest / "sachs_radar"))
    print(f"wrote {dest}/sachs_nll.pdf + {dest}/sachs_radar.pdf (+ .png)")


if __name__ == "__main__":
    main()
