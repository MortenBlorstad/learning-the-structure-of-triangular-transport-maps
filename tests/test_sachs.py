"""Sachs experiment: loader provenance (both subsets), cell-id grammar, 2-epoch smoke."""
import copy

import numpy as np
import pytest
import yaml

from experiments.sachs import experiment as sx


def test_loader_pooled():
    X, B = sx.load_sachs("pooled")
    assert X.shape == (7466, 11) and X.dtype == np.float32
    assert B.shape == (11, 11) and B.sum() == 20
    assert np.all(np.diag(B) == 0)


def test_loader_obs853():
    X, B = sx.load_sachs("obs853")
    assert X.shape == (853, 11)
    assert B.sum() == 20                      # same 20-edge reference for both subsets
    Xp, _ = sx.load_sachs("pooled")
    assert np.array_equal(X, Xp[:853])        # the observational block IS the file's head


def test_cell_id_roundtrip():
    cfg = {"subsets": sx.SUBSETS, "models": sx.MODELS, "seeds": [0, 1, 2, 3, 4]}
    for c in sx.cells(cfg):
        assert sx.parse_cell_id(sx.cell_id(c)) == c
    assert len(sx.cells(cfg)) == 70


@pytest.mark.parametrize("model,subset", [("learned", "obs853"), ("maf", "obs853"),
                                          ("learned", "pooled"), ("gnf", "obs853")])
def test_smoke_run_one(model, subset):
    cfg = copy.deepcopy(yaml.safe_load(open("experiments/config.yaml"))["sachs"])
    cfg["SSTM"]["epochs"] = 2
    cfg["MAF"]["epochs"] = 2
    cfg["GNF"]["epochs"] = {"pooled": 2, "obs853": 2}   # per-subset dict since the schedule-matched protocol
    row = sx.run_one({"subset": subset, "model": model, "seed": 0}, cfg)
    assert "nll_test" in row and np.isfinite(row["nll_test"])
    assert "gap_to_true" not in row and "nll_true_test" not in row
    assert row["subset"] == subset and row["sampler"] == f"sachs_{subset}"
    assert row["n"] == (682 if subset == "obs853" else 5972)
    if model in ("learned", "gnf"):
        assert "shd" in row and "dir_f1" in row and row["nnz_true"] == 20
    else:
        assert "shd" not in row
