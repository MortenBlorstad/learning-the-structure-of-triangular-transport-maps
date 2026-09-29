"""Cheap insurance for the paper figure generators (synthetic.plot_results).

Builds a synthetic results frame (5 datasets x 2 K x 3 n x 4 models x 2 seeds) and checks the
two grid figures render end to end and that a missing cell is a hard error for the headline.
"""
from __future__ import annotations

import pandas as pd
import pytest

from experiments.synthetic import plot_results as pr

E = 2
DS = (10, 20)
NS = (200, 500, 1000)
SEEDS = (0, 1)
# per-model (gap_to_true nats, skeleton_f1, shd, directed_f1) for the synthetic frame
MODELS = {"oracle": (0.5, 1.0, 0.0, 1.0), "random": (3.0, 0.6, 20.0, 0.4),
          "two_stage": (4.0, 0.4, 15.0, 0.2), "learned": (1.0, 0.8, 8.0, 0.7)}


def _synthetic_df() -> pd.DataFrame:
    rows = []
    for dataset, _ in pr.DATASETS:
        s = pr._sampler_key(dataset, E)
        for d in DS:
            for n in NS:
                for seed in SEEDS:
                    for model, (gap, f1, shd, dirf1) in MODELS.items():
                        rows.append(dict(sampler=s, d=d, n=n, seed=seed, model=model,
                                         gap_to_true=gap, skeleton_f1=f1, shd=shd, directed_f1=dirf1))
    return pd.DataFrame(rows)


def test_directed_f1():
    df = pd.DataFrame([{"fdr": 0.0, "tpr": 0.0}, {"fdr": 0.2, "tpr": 0.6}, {"fdr": 1.0, "tpr": 0.0}])
    f1 = pr.directed_f1(df).tolist()
    assert f1[0] == 0.0                                   # nothing recovered -> 0
    assert abs(f1[1] - 2 * 0.8 * 0.6 / (0.8 + 0.6)) < 1e-9
    assert f1[2] == 0.0                                   # precision 0 -> 0 (no NaN)


def test_headline_figure_runs(tmp_path):
    out = pr.headline_gap(_synthetic_df(), ds=DS, e=E, n_grid=NS, out=tmp_path)
    assert out.exists() and out.suffix == ".pdf"
    assert (tmp_path / "headline_gap.png").exists()




def test_missing_cell_is_a_hard_error(tmp_path):
    df = _synthetic_df()
    df = df[~((df["sampler"] == "funnel") & (df["n"] == 500))]   # drop a cell
    with pytest.raises(ValueError, match="missing cells"):
        pr.headline_gap(df, ds=DS, e=E, n_grid=NS, out=tmp_path)
