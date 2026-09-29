"""The Experiment contract + a name->Experiment registry.

An experiment is three functions over its own config section:
  * cells(cfg)   -> list of cell param-dicts (the grid to sweep)
  * cell_id(cell)-> str filename stem
  * run_one(cell, cfg, *, logger, checkpoint_dir, n_checkpoints) -> metrics dict
plus two optional hooks:
  * logs_cell(cell) -> bool   which cells get a TrainingLogger / checkpoints

The runner (launch/run_cell) is experiment-agnostic and shared.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import yaml

from experiments.common import results_dir

REPO = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO / "experiments" / "config.yaml"


@dataclass(frozen=True)
class Experiment:
    """One experiment, as the runner sees it.

    `name` is what `--experiment` takes and names the results directory.
    `config_key` is the section of config.yaml the experiment reads. The three
    callables are the contract described at the top of this module.
    """

    name: str
    config_key: str
    cells: Callable[[dict], list]
    cell_id: Callable[[dict], str]
    run_one: Callable[..., dict]
    logs_cell: Callable[[dict], bool] = field(default=lambda cell: True)


_REGISTRY: dict[str, Experiment] = {}


def register(exp: Experiment) -> Experiment:
    """Add an experiment to the registry. Raises if the name is already taken."""
    if exp.name in _REGISTRY:
        raise ValueError(f"experiment '{exp.name}' already registered")
    _REGISTRY[exp.name] = exp
    return exp


def get(name: str) -> Experiment:
    """The registered experiment called `name`. Raises KeyError if there is none."""
    if name not in _REGISTRY:
        raise KeyError(f"unknown experiment '{name}'; registered: {names()}")
    return _REGISTRY[name]


def names() -> list:
    """Every registered experiment name, sorted."""
    return sorted(_REGISTRY)


def load_section(config_path, config_key: str) -> dict:
    """The experiment's config.yaml section (a mutable copy so axes can be filtered)."""
    with open(config_path) as f:
        raw = yaml.safe_load(f)
    return dict(raw.get(config_key, raw))


def cell_id_for(exp: Experiment, cell: dict) -> str:
    """Cell-id stem. For an ablation cell (carries '_ablate'), the swept value is encoded
    into the id (<base>__<tag>), so cells at different hyperparameter values never collide."""
    ab = cell.get("_ablate")
    base = {k: v for k, v in cell.items() if k != "_ablate"} if ab else cell
    cid = exp.cell_id(base)
    return f"{cid}__{ab['tag']}" if ab else cid


def out_path_for(exp: Experiment, cell: dict) -> Path:
    """results/<experiment>/<cell_id>.json, or, for an ablation cell,
    results/<experiment>/ablation/<study>/<cell_id>__<tag>.json."""
    ab = cell.get("_ablate")
    if ab:
        return results_dir(ab["sub"]) / f"{cell_id_for(exp, cell)}.json"
    return results_dir(exp.name) / f"{exp.cell_id(cell)}.json"
