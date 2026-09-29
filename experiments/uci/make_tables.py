"""UCI density tables for the paper, in the paper's own layout.

Generates the two ``tab:density_*`` tables exactly as they appear in main.tex, with every
number derived rather than hand-typed:

- ``tab_density_results.tex`` -- test NLL (nats), mean +- one std over seeds, one column per
  dataset, rows SSTM (= the SSTM cells) / MAF / NSF / GNF. Read from the flat result
  cells in results/uci/ (ablation studies are diagnostics and are never included). A missing
  (dataset, model) cell renders ``---`` (never silently dropped); an incomplete seed set is
  flagged in a comment above the table.
- ``tab_density_hparams.tex`` -- the appendix hyperparameter table: split sizes (constants
  below, cross-checked against the rows' train counts), the Table-5 per-dataset budget and
  flow settings from experiments/config.yaml, epochs from the rows, SSTM sizes from
  config (``hidden_per_dataset`` resolved per dataset).

Row set and labels track main.tex: the model is SSTM in the paper and SSTM in the
code, and the hyperparameter table carries no seeds / hidden-layers / SSTM-learning-rate
rows. Keep the two in step -- a regeneration overwrites whatever the paper has.

Writes ``\\input``-able files to results/tables/ and echoes them to stdout.

    uv run python -m experiments.uci.make_tables [--results-dir DIR] [--out DIR]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from experiments.uci.experiment import parse_cell_id

CAPTION = (
    r"Hyperparameters of the density experiment on the UCI benchmarks, per dataset. Data: the standard splits of \citet{papamakarios2017maf}. Common: the per-dataset batch size and step budget of \citet[Table 5]{durkan2019nsf}, converted to whole epochs and shared by all models. Flows: their published per-dataset settings. \textsc{SSTM}: per-dataset size, chosen on validation.")

DATASETS = [("POWER", 6), ("GAS", 8), ("HEPMASS", 21), ("MINIBOONE", 43)]
MODELS = [("SSTM", r"\textsc{SSTM} (ours)"), ("MAF", "MAF"), ("NSF", "NSF"),
          ("GNF", "GNF")]
N_SEEDS = 5
# Standard splits of Papamakarios et al. (2017): (train, val, test) row counts.
# Train counts are cross-checked against the rows' own n at generation time.
SPLITS = {"POWER": (1_659_917, 184_435, 204_928), "GAS": (852_174, 94_685, 105_206),
          "HEPMASS": (315_123, 35_013, 174_987), "MINIBOONE": (29_556, 3_284, 3_648)}


def collect(results_path: Path) -> dict:
    """{(dataset, model): [nll_test per seed]} from the flat result cells."""
    cells: dict = {}
    for f in sorted(results_path.glob("*.json")):
        c = parse_cell_id(f.stem)
        cells.setdefault((c["dataset"], c["model"]), []).append(
            json.load(f.open())["nll_test"])
    return cells


def make_table(results_path: Path) -> str:
    cells = collect(results_path)
    lines = []
    for ds, _ in DATASETS:
        for m, _ in MODELS:
            v = cells.get((ds, m), [])
            if v and len(v) < N_SEEDS:
                lines.append(f"% INCOMPLETE: {ds} {m} has {len(v)}/{N_SEEDS} seeds")

    def cell(ds, m):
        v = cells.get((ds, m), [])
        if len(v) < N_SEEDS:
            return "---"
        return f"${np.mean(v):.3f} \\pm {np.std(v, ddof=1):.3f}$"

    head = " & ".join(rf"\textsc{{{ds.capitalize()}}}" for ds, _ in DATASETS)
    dims = " & ".join(rf"$K={k}$" for _, k in DATASETS)
    body = "\n".join(
        f"    {label:28s} & " + " & ".join(cell(ds, m) for ds, _ in DATASETS) + r" \\"
        for m, label in MODELS)
    return "\n".join(lines + [
        r"\begin{table}[t]",
        r"  \centering",
        r"  \caption{Real-data density estimation on the UCI benchmarks, Mean NLL $\pm$ one standard deviation over five seeds.}",
        r"  \label{tab:density_results}",
        r"  \small",
        r"  \begin{tabular}{@{}lrrrr@{}}",
        r"    \toprule",
        f"    & {head} \\\\",
        f"    & {dims} \\\\",
        r"    \midrule",
        body,
        r"    \bottomrule",
        r"  \end{tabular}",
        r"\end{table}",
    ])


def make_hparams_table(results_path: Path, cfg_path: Path = Path("experiments/config.yaml")) -> str:
    """The appendix hyperparameter table, from config.yaml + the result rows."""
    cfg = yaml.safe_load(cfg_path.open())["uci"]
    be, flows = cfg["SSTM"], cfg["flows"]

    # read off the rows: epochs (shared by all models -- asserted) and the train-count check
    epochs: dict = {}
    for f in sorted(results_path.glob("*.json")):
        c = parse_cell_id(f.stem)
        r = json.load(f.open())
        epochs.setdefault(c["dataset"], set()).add(r["epochs"])
        assert r["n"] == SPLITS[c["dataset"]][0], f"{c['dataset']}: row n != SPLITS train count"
    for ds, es in epochs.items():
        assert len(es) == 1, f"{ds}: models disagree on epochs {es}"

    def row(label, vals, fmt=str):
        cells = " & ".join(fmt(v) if v is not None else "---" for v in vals)
        return f"    {label:26s} & {cells} \\\\"

    ds_names = [ds for ds, _ in DATASETS]
    pd = [flows["per_dataset"][ds] for ds in ds_names]
    hid = [list(be.get("hidden_per_dataset", {}).get(ds, be["hidden"])) for ds in ds_names]
    for h in hid:
        assert len(set(h)) == 1, f"non-uniform widths {h} need a new format"

    return "\n".join([
        r"\begin{table}[h]",
        r"  \centering",
        (r"  \caption{" + CAPTION + "}"),
        r"  \label{tab:density_hparams}",
        r"  \small",
        r"  \begin{tabular}{@{}lrrrr@{}}",
        r"    \toprule",
        "    & " + " & ".join(rf"\textsc{{{ds.capitalize()}}}" for ds in ds_names) + r" \\",
        r"    \midrule",
        r"    \multicolumn{5}{@{}l}{\textit{Data}} \\",
        row("Dimension $K$", [k for _, k in DATASETS]),
        row("Training points", [SPLITS[ds][0] for ds in ds_names], "{:,}".format),
        row("Validation points", [SPLITS[ds][1] for ds in ds_names], "{:,}".format),
        row("Test points", [SPLITS[ds][2] for ds in ds_names], "{:,}".format),
        r"    \midrule",
        r"    \multicolumn{5}{@{}l}{\textit{Common (all models)}} \\",
        row("Batch size", [c["batch_size"] for c in pd]),
        row("Step budget", [c["steps"] for c in pd], "{:,}".format),
        row("Epochs", [next(iter(epochs[ds])) if ds in epochs else None for ds in ds_names]),
        r"    \midrule",
        r"    \multicolumn{5}{@{}l}{\textit{Flows (MAF, NSF)}} \\",
        row("Learning rate", [c["lr"] for c in pd]),
        row("Transforms", [flows["transforms"]] * 4),
        row("Residual blocks", [c["num_blocks"] for c in pd]),
        row("Hidden features", [c["hidden_features"] for c in pd]),
        row("Spline bins (NSF)", [c["bins"] for c in pd]),
        row("Dropout", [c["dropout"] for c in pd]),
        r"    \midrule",
        r"    \multicolumn{5}{@{}l}{\textit{\textsc{SSTM}}} \\",
        row("Hidden features", [h[0] for h in hid], lambda w: f"${w}$"),
        r"    \bottomrule",
        r"  \end{tabular}",
        r"\end{table}",
    ])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", default="results/uci")
    p.add_argument("--out", default="results/tables")
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    header = "% Generated by experiments.uci.make_tables -- do not hand-edit numbers.\n"
    for name, tex in [
        ("tab_density_results.tex",
         header + f"% Source: {a.results_dir}/*.json (test NLL over {N_SEEDS} seeds).\n"
         + make_table(Path(a.results_dir)) + "\n"),
        ("tab_density_hparams.tex",
         header + f"% Source: experiments/config.yaml (uci) + {a.results_dir}/*.json.\n"
         + make_hparams_table(Path(a.results_dir)) + "\n"),
    ]:
        dest = out / name
        dest.write_text(tex)
        print(tex)
        print(f"wrote {dest}\n")


if __name__ == "__main__":
    main()
