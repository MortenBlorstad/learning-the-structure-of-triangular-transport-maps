"""Schedule a registered experiment's cells across GPUs (cluster) or jobs (Mac/MPS).

Expands the experiment's grid into cells and dispatches `run_cell.py` subprocesses,
one per cell.

    # laptop (MPS/CPU): 8 cells in parallel, SSTM only
    python -m experiments.runner.launch --experiment synthetic --jobs 8 --select model=learned
    # cluster: 4 cells per GPU on 3 GPUs
    python -m experiments.runner.launch --experiment synthetic --gpus 0,2,5 --per-gpu 4
    # preview
    python -m experiments.runner.launch --experiment synthetic --jobs 8 --dry-run
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from collections import deque

from experiments.common import git_provenance, results_dir
from experiments.runner import manifest  # noqa: F401  (registers experiments on import)
from experiments.runner.registry import get, load_section, out_path_for, names, DEFAULT_CONFIG


def cmd_for(exp, cell, args) -> list:
    ab = cell.get("_ablate")
    base = {k: v for k, v in cell.items() if k != "_ablate"} if ab else cell
    c = [sys.executable, "-m", "experiments.runner.run_cell",
         "--experiment", exp.name,
         "--cell", json.dumps(base, separators=(",", ":")),
         "--config", args.config]
    if args.override:
        c.append("--override")
    if ab:                                           # ablation: namespaced output + config override
        c += ["--out-sub", ab["sub"], "--set", ab["set"], "--tag", ab["tag"]]
    for kv in args.set_const:                        # constant overrides held fixed across the sweep
        c += ["--set", kv]
    return c


def _split_values(vals: str) -> list:
    """Split 'a,b,c' on commas outside brackets, so list values survive:
    'hidden=[64],[64,64],[128,128]' -> ['[64]', '[64,64]', '[128,128]']."""
    out, cur, depth = [], "", 0
    for ch in vals:
        depth += (ch in "[{") - (ch in "]}")
        if ch == "," and depth == 0:
            out.append(cur); cur = ""
        else:
            cur += ch
    out.append(cur)
    return [v.strip() for v in out if v.strip()]


def _ablation_cells(exp, base_cells, args) -> list:
    """Expand base cells x the swept values into ablation cells (each carries '_ablate' with the
    output sub-dir, the --set config override, and the id tag) and write settings.json once."""
    param, _, vals = args.ablate.partition("=")
    values = _split_values(vals)
    study = args.study or param.replace(".", "_")
    sub, short = f"{exp.name}/ablation/{study}", param.split(".")[-1]
    cells = [{**bc, "_ablate": {"sub": sub, "set": f"{param}={v}", "tag": f"{short}={v}"}}
             for v in values for bc in base_cells]
    settings = {"experiment": exp.name, "param": param, "values": values, "short": short,
                "study": study, "select": args.select, "set_const": args.set_const,
                "base_cells": len(base_cells), "config": args.config, **git_provenance()}
    (results_dir(sub) / "settings.json").write_text(json.dumps(settings, indent=2) + "\n")
    print(f"[{exp.name}] ablation '{study}': {param} over {values} x {len(base_cells)} base cells "
          f"-> results/{sub}/", flush=True)
    return cells


def apply_selects(cells, selects):
    """--select key=v1,v2 keeps cells whose cell[key] is in {v1,v2}. Replaces the old
    synthetic-specific --models/--graphs with a generic, per-axis filter."""
    for sel in selects:
        key, _, vals = sel.partition("=")
        allowed = set(vals.split(","))
        cells = [c for c in cells if str(c.get(key)) in allowed]
    return cells


def pending_cells(exp, cells, args) -> list:
    if args.override:
        return list(cells)
    return [c for c in cells if not out_path_for(exp, c).exists()]


def _progress(done, total, t0):
    el = time.time() - t0
    eta = el / done * (total - done) if done else 0.0
    def _m(s):
        return f"{s / 60:.1f}m" if s >= 60 else f"{s:.0f}s"
    print(f"  progress [{done}/{total}] {100 * done // total}%  elapsed {_m(el)}  eta ~{_m(eta)}", flush=True)


def run_gpu(exp, cells, gpus, per_gpu, args):
    pending = deque(cells)
    running = []
    free = {g: per_gpu for g in gpus}
    total, done, t0 = len(cells), 0, time.time()
    while pending or running:
        for g in gpus:
            while free[g] > 0 and pending:
                cell = pending.popleft()
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(g), OMP_NUM_THREADS="1",
                           MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1")
                running.append((subprocess.Popen(cmd_for(exp, cell, args), env=env), g))
                free[g] -= 1
        time.sleep(1.0)
        for proc, g in running[:]:
            if proc.poll() is not None:
                running.remove((proc, g)); free[g] += 1; done += 1
                _progress(done, total, t0)


def run_pool(exp, cells, jobs, args):
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
               OPENBLAS_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1")
    pending = deque(cells)
    running = []
    total, done, t0 = len(cells), 0, time.time()
    while pending or running:
        while len(running) < jobs and pending:
            running.append(subprocess.Popen(cmd_for(exp, pending.popleft(), args), env=env))
        time.sleep(1.0)
        still = [p for p in running if p.poll() is None]
        if len(still) < len(running):
            done += len(running) - len(still)
            _progress(done, total, t0)
        running = still


def run_serial(exp, cells, args):
    total, t0 = len(cells), time.time()
    for i, cell in enumerate(cells, 1):
        subprocess.run(cmd_for(exp, cell, args), check=False)
        _progress(i, total, t0)


def dry_run(exp, cells, gpus, per_gpu, jobs, args):
    for i, cell in enumerate(cells):
        prefix = f"CUDA_VISIBLE_DEVICES={gpus[i % len(gpus)]} " if gpus else ""
        print(prefix + " ".join(cmd_for(exp, cell, args)))
    if gpus:
        conc = f"{len(gpus)} gpus x {per_gpu} slots = {len(gpus) * per_gpu} concurrent"
    elif jobs > 1:
        conc = f"{jobs} concurrent (MPS/CPU pool)"
    else:
        conc = "serial"
    print(f"# {len(cells)} cells, {conc}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--experiment", required=True, help=f"one of: {names() or '(none registered)'}")
    p.add_argument("--gpus", default="", help="comma list, e.g. 0,2,5 (empty = serial/pool)")
    p.add_argument("--per-gpu", type=int, default=4)
    p.add_argument("--jobs", type=int, default=1, help="Mac/MPS/CPU: run N cells concurrently")
    p.add_argument("--serial", action="store_true", help="force serial (overrides --jobs)")
    p.add_argument("--config", default=str(DEFAULT_CONFIG))
    p.add_argument("--override", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--select", action="append", default=[],
                   help="filter cells by axis, e.g. --select model=learned --select graph=ER,funnel")
    p.add_argument("--ablate", default=None, metavar="PARAM=v1,v2,..",
                   help="ablate one config knob over values, e.g. SSTM.l2=1e-4,1e-3,1e-2")
    p.add_argument("--study", default=None, help="ablation study name -> results/<exp>/ablation/<study>/")
    p.add_argument("--set", action="append", default=[], dest="set_const", metavar="KEY=VAL",
                   help="config override held fixed across every cell, e.g. SSTM.l2=0.0 (repeatable)")
    args = p.parse_args()

    exp = get(args.experiment)
    cfg = load_section(args.config, exp.config_key)
    cells = apply_selects(exp.cells(cfg), args.select)
    if args.ablate:
        cells = _ablation_cells(exp, cells, args)     # base x values, each a namespaced ablation cell
    todo = pending_cells(exp, cells, args)

    gpus = [int(x) for x in args.gpus.split(",") if x.strip() != ""]
    use_gpus = bool(gpus) and not args.serial
    use_pool = not use_gpus and args.jobs > 1 and not args.serial
    mode = (f"gpus {gpus} x {args.per_gpu}/gpu" if use_gpus
            else f"{args.jobs} concurrent (MPS/CPU pool)" if use_pool else "serial")
    print(f"[{exp.name}] {len(cells)} cells total, {len(todo)} to run ({mode})", flush=True)

    if args.dry_run:
        dry_run(exp, todo, gpus if use_gpus else [], args.per_gpu, args.jobs if use_pool else 1, args)
    elif use_gpus:
        run_gpu(exp, todo, gpus, args.per_gpu, args)
    elif use_pool:
        run_pool(exp, todo, args.jobs, args)
    else:
        run_serial(exp, todo, args)
    print("done", flush=True)


if __name__ == "__main__":
    main()
