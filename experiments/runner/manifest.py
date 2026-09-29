"""Register all experiments. Import this module to populate the registry.

Each experiment is registered by ADAPTING its existing pieces to the cell-dict
contract — no behaviour change, so result jsons stay identical. Add a new
experiment by writing its cells/cell_id/run_one and registering it here.
"""
from experiments.runner.registry import Experiment, register

# --- synthetic (density + structure on generated data) --------------------- #
from experiments.synthetic.experiment import (
    build_cells as _syn_build_cells, cell_id as _syn_cell_id, run_one as _syn_run_one)

_SYN_AXES = ("model", "graph", "sem", "d", "e", "n", "seed")


def _syn_cells(cfg):
    # build_cells yields (model, graph, sem, d, e, n, seed) tuples -> dicts
    return [dict(zip(_SYN_AXES, tup)) for tup in _syn_build_cells(cfg)]


def _syn_id(c):
    return _syn_cell_id(c["model"], c["graph"], c["sem"], c["d"], c["e"], c["n"], c["seed"])


def _syn_runone(cell, cfg, *, logger=None, checkpoint_dir=None, n_checkpoints=0):
    return _syn_run_one(cell["model"], cell["graph"], cell["sem"], cell["d"], cell["e"],
                       cell["n"], cell["seed"], cfg,
                       logger=logger, checkpoint_dir=checkpoint_dir, n_checkpoints=n_checkpoints)


register(Experiment(
    name="synthetic",
    config_key="synthetic",
    cells=_syn_cells,
    cell_id=_syn_id,
    run_one=_syn_runone,
    logs_cell=lambda c: c["model"] in ("learned", "oracle", "maf", "nsf"),   # method + oracle (overfit diagnosis) + flow baselines
))


# --- uci (real-data density estimation) ------------------------------------ #
from experiments.uci import experiment as _uci

register(Experiment(
    name="uci",
    config_key="uci",
    cells=_uci.cells,
    cell_id=_uci.cell_id,
    run_one=_uci.run_one,
    logs_cell=_uci.logs_cell,
))


# --- method_ablation (one-variant-at-a-time design ablation) ---------------- #
from experiments.method_ablation import experiment as _ma

register(Experiment(
    name="method_ablation",
    config_key="method_ablation",
    cells=_ma.cells,
    cell_id=_ma.cell_id,
    run_one=_ma.run_one,
    logs_cell=_ma.logs_cell,
))


# --- sachs (real data with a known sparse graph) ---------------------------- #
from experiments.sachs import experiment as _sx

register(Experiment(
    name="sachs",
    config_key="sachs",
    cells=_sx.cells,
    cell_id=_sx.cell_id,
    run_one=_sx.run_one,
    logs_cell=_sx.logs_cell,
))
