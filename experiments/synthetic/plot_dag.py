"""DAG visualisation for the synthetic datasets.

- `plot_dag(B, ax, ...)`: publication-style plot of one DAG from its adjacency matrix.
- `dag_gallery(...)`: a row of example DAGs, one per graph type (ER, SF, funnel, hier),
  saved to `results/figures/example_dags.{pdf,png}`.

The *graph* depends only on graph_type; the SEM (gauss/mlp) affects the data, not the
adjacency -- so the gallery is one panel per graph type.

    python -m experiments.synthetic.plot_dag [--d 20 --seed 0 --out-dir DIR]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import igraph as ig
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

from experiments.sem import make_data
from experiments import paper_style as ps

FIG_DIR = Path("results/figures")

# One representative panel per graph type (label, graph_type, sem). SEM is only needed by
# make_data's signature for ER/SF; it does not change the adjacency.
GRAPHS = [
    ("Erdos-Renyi", "ER", "mlp"),
    ("Scale-free",  "SF", "mlp"),
    ("Funnel",      "funnel", None),
    ("Hierarch",    "hierarch", None),
]


def plot_dag(B, ax=None, node_names=None, layout="sugiyama",
             highlight_edges=None, node_size=28, show_labels=True):
    """Publication-style plot of a DAG given its adjacency matrix.

    layout: "sugiyama" (layered, best for DAGs), "kk" (Kamada-Kawai), "circle".
    highlight_edges: optional set of (i, j) tuples drawn in red (e.g. false positives).
    show_labels: draw $x_i$ node labels (True; useful for a single labelled DAG). For a
        topology-showcase set False -> small filled nodes, no labels (avoids overlap).
    """
    B = np.asarray(B)
    g = ig.Graph.Adjacency(B.astype(int).tolist())
    if ax is None:
        _, ax = plt.subplots(figsize=(3.3, 3.3))   # single-column width

    # Everything drawn with matplotlib (igraph only for the layout): igraph's GraphArtist
    # sizes vertices/arrows in fixed device pixels, so they vanish at high dpi (raster PNG)
    # while the vector PDF looks fine. matplotlib scatter/FancyArrowPatch scale with dpi.
    lay = g.layout(layout)
    coords = np.asarray(lay.coords, dtype=float)
    node_s = node_size * (6 if show_labels else 4)       # scatter marker area (points^2)
    shrink = float(np.sqrt(node_s / np.pi)) + 1.0        # pull arrow ends back to the node rim (+gap)

    # DIRECTED edges: i -> j for B[i, j] == 1, arrowhead stopping at the target node's rim.
    for i in range(B.shape[0]):
        for j in np.flatnonzero(B[i]):
            col = "#d62728" if highlight_edges and (i, j) in highlight_edges else "#555555"
            ax.add_patch(FancyArrowPatch(
                coords[i], coords[j], arrowstyle="-|>", mutation_scale=9,
                color=col, lw=1.0, shrinkA=shrink, shrinkB=shrink, zorder=2))

    if show_labels:
        if node_names is None:
            node_names = [f"$x_{{{i}}}$" for i in range(B.shape[0])]
        ax.scatter(coords[:, 0], coords[:, 1], s=node_s,
                   facecolor="white", edgecolor="#333333", linewidth=1.0, zorder=5)
        for (x, y), name in zip(coords, node_names):
            ax.annotate(name, (x, y), ha="center", va="center", fontsize=9, zorder=6)
    else:
        ax.scatter(coords[:, 0], coords[:, 1], s=node_s,
                   facecolor="#333333", edgecolor="none", zorder=5)   # filled dots

    ax.autoscale_view()
    ax.margins(0.10)
    ax.axis("off")
    return ax


def dag_gallery(d=10, e=2, seed=0, out=FIG_DIR) -> Path:
    """One example DAG per graph type, in a single row. Returns the .pdf path."""
    ps.use_paper_style()
    n = 200  # data size irrelevant here; only the adjacency is drawn
    fig, axes = plt.subplots(1, len(GRAPHS), figsize=(3.0 * len(GRAPHS), 3.0), squeeze=False)
    letters = iter("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    for ax, (label, graph, sem) in zip(axes[0], GRAPHS):
        s0 = int(e) * d                                   # #edges for ER/SF; ignored by funnel/hier
        B, _X, _lp, _order = make_data(d, s0, n, sem, graph, seed)
        plot_dag(B, ax=ax, show_labels=False)         # topology showcase -> clean, no node labels
        ax.set_title(rf"$\mathbf{{{next(letters)}}}$: {label} ($|E|={int(np.asarray(B).sum())}$)",
                     loc="left", fontsize=ps.LABELSIZE)
    fig.tight_layout()
    out.mkdir(parents=True, exist_ok=True)
    ps.save_paper(fig, str(out / "example_dags"))
    plt.close(fig)
    return out / "example_dags.pdf"


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--d", type=int, default=10)
    p.add_argument("--e", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", default=None)
    args = p.parse_args()
    outp = dag_gallery(args.d, args.e, args.seed, Path(args.out_dir) if args.out_dir else FIG_DIR)
    print(f"wrote example DAGs -> {outp} (+ .png)")
