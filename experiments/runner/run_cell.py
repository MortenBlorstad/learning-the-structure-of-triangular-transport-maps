"""Run ONE cell of a registered experiment and write its result json.

Looks the experiment up in the registry, runs the single cell given as a JSON dict,
and writes the row atomically (tmp + rename), so many cells can run in parallel
without ever touching the same file. Normally driven by launch.py, not by hand.

    python -m experiments.runner.run_cell --experiment synthetic \
        --cell '{"model":"learned","graph":"ER","sem":"gauss","d":10,"e":2,"n":1000,"seed":0}'
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch
import yaml

from triangularmap.utils import TrainingLogger
from experiments.common import git_provenance, results_dir
from experiments.runner import manifest  # noqa: F401  (registers experiments on import)
from experiments.runner.registry import get, load_section, out_path_for, DEFAULT_CONFIG


def _parse_val(s: str):
    """Best-effort parse for a --set value: int, then float, then bool, then
    YAML list/dict (e.g. hidden_features=[32, 32]), else str."""
    for cast in (int, float):
        try:
            return cast(s)
        except ValueError:
            pass
    if s.lstrip().startswith(("[", "{")):
        try:
            return yaml.safe_load(s)
        except yaml.YAMLError:
            pass
    return {"true": True, "false": False}.get(s.lower(), s)


def _set_nested(cfg: dict, dotted: str, value) -> None:
    """cfg['a']['b'] = value for dotted='a.b' (keys must already exist)."""
    d = cfg
    keys = dotted.split(".")
    for k in keys[:-1]:
        d = d[k]
    d[keys[-1]] = value


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--experiment", required=True)
    p.add_argument("--cell", required=True, help="JSON dict of cell params")
    p.add_argument("--config", default=str(DEFAULT_CONFIG))
    p.add_argument("--override", action="store_true")
    p.add_argument("--log-every", type=int, default=None, help="override cfg log_every (0 disables)")
    p.add_argument("--n-checkpoints", type=int, default=None, help="override cfg n_checkpoints (0 disables)")
    p.add_argument("--out-sub", default=None, help="write under results/<out-sub>/ instead of results/<experiment>/ (ablation)")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VAL",
                   help="dotted config override, e.g. SSTM.l2=0.001 (repeatable)")
    p.add_argument("--tag", default=None, help="suffix appended to the cell id (encodes the ablated value)")
    args = p.parse_args()

    exp = get(args.experiment)
    cell = json.loads(args.cell)

    cid = exp.cell_id(cell)
    if args.tag:
        cid = f"{cid}__{args.tag}"
    out_path = (results_dir(args.out_sub) / f"{cid}.json") if args.out_sub else out_path_for(exp, cell)
    if out_path.exists() and not args.override:
        print(f"SKIP {out_path.name} (exists)", flush=True)
        return

    cfg = load_section(args.config, exp.config_key)
    for kv in args.set:                              # dotted config overrides (ablation)
        key, _, val = kv.partition("=")
        _set_nested(cfg, key, _parse_val(val))

    # seed everything from the cell's seed (also seeds NOTEARS-MLP init; SSTM self-seeds)
    seed = int(cell["seed"])
    np.random.seed(seed)
    torch.manual_seed(seed)

    want_log = exp.logs_cell(cell) or bool(args.out_sub)   # ablation cells always log
    log_every = args.log_every if args.log_every is not None else int(cfg.get("log_every", 0))
    logger = (TrainingLogger(log_every=log_every, out_dir=out_path.parent / "logs", run_name=cid)
              if log_every > 0 and want_log else None)
    n_checkpoints = args.n_checkpoints if args.n_checkpoints is not None else int(cfg.get("n_checkpoints", 0))
    ckpt_dir = (out_path.parent / "checkpoints" / cid) if n_checkpoints > 0 and want_log else None

    row = exp.run_one(cell, cfg, logger=logger, checkpoint_dir=ckpt_dir, n_checkpoints=n_checkpoints)

    row = {**git_provenance(), **row}   # stamp the commit that produced this run
    tmp = out_path.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump(row, f, indent=2)
    os.replace(tmp, out_path)  # atomic
    if logger is not None and logger.history:
        logger.save()
    print(f"wrote {out_path.name}", flush=True)


if __name__ == "__main__":
    main()
