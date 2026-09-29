"""method_ablation experiment: cell-id grammar and variant-config validity.

The variant-validity test constructs a tiny model from every variant's overrides,
so a typo'd kwarg in config.yaml fails here instead of part-way through a sweep.
"""

import pytest

from experiments.method_ablation import experiment as ma
from experiments.runner.registry import load_section, DEFAULT_CONFIG
from triangularmap.sstm import SSTM


@pytest.fixture(scope="module")
def cfg():
    return load_section(str(DEFAULT_CONFIG), "method_ablation")


def test_cell_id_round_trip(cfg):
    cs = ma.cells(cfg)
    # full grid: (1 default + variants) x (ER,SF x sems x e + funnel,hierarch) x d x n x seeds
    n_variants = 1 + len(cfg["variants"])
    n_targets = 2 * len(cfg["sem"]) * len(cfg["e"]) + 2
    expected = n_variants * n_targets * len(cfg["d"]) * len(cfg["n"]) * len(cfg["seeds"])
    assert len(cs) == expected
    ids = {ma.cell_id(c) for c in cs}
    assert len(ids) == len(cs)                        # ids unique
    for c in cs:
        assert ma.parse_cell_id(ma.cell_id(c)) == c, ma.cell_id(c)


def test_variant_overrides_construct(cfg):
    # every variant's overrides must be valid SSTM kwargs (tiny model, d=4)
    be = cfg["SSTM"]
    base = dict(input_size=4, hidden=[8, 8], output_dim=1, disconnect_first_n=2,
                random_state=0, gate_method=be["gate_method"],
                perm_method=be.get("perm_method", "softsort"))
    for name, overrides in {"default": {}, **cfg["variants"]}.items():
        kw = {**base, **overrides}
        m = SSTM(**kw)
        assert sum(p.numel() for p in m.parameters()) > 0, name


def test_run_one_row_contract():
    # A real (tiny) run: the row must carry the shared identity keys, the cost
    # metrics, and the learned-structure objects, and be JSON-serializable.
    import json
    tiny = {"n_test": 200, "n_val": 200,
            "variants": {"no_sharing": {"parameter_sharing": False}},
            "SSTM": {"epochs": 2, "hidden": [8, 8], "disconnect_first_n": 2,
                             "batch_size": 32, "lr_map": 0.01, "lr_ordering": 0.01,
                             "lr_sparsity": 0.01, "l2": 0, "l1": None, "optimizer": "adam",
                             "gate_method": "ste", "perm_method": "softsort",
                             "sparsity_strength": 1e-3, "n_samples": 2}}
    cell = {"variant": "no_sharing", "graph": "funnel", "sem": None,
            "d": 4, "e": None, "n": 100, "seed": 0}
    row = ma.run_one(cell, tiny)
    json.dumps(row)                                    # serializable
    for key in ("model", "sampler", "d", "n",                          # identity keys shared with the synthetic experiment
                "n_params", "n_params_maps", "fit_seconds", "seconds_per_epoch",
                "epochs", "n_samples", "peak_cuda_mem_mb",             # cost metrics
                "hard_order", "B_est",                                 # structure objects
                "gap_to_true_val", "tpr", "shd", "nnz"):
        assert key in row, key
    assert row["model"] == "no_sharing" and row["epochs"] == 2 and row["n_samples"] == 2
    assert len(row["hard_order"]) == 4 and len(row["B_est"]) == 4


def test_variants_change_exactly_one_knob(cfg):
    # star design: each variant overrides one design choice (plain = no_sharing's
    # bundle partner is the documented exception with two keys).
    for name, overrides in cfg["variants"].items():
        limit = 2 if name == "plain" else 1
        assert 1 <= len(overrides) <= limit, (name, overrides)
