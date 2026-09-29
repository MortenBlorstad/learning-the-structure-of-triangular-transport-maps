"""Method-design ablation: one variant at a time around the paper-default SSTM.

Star design: the `default` variant is the centre and every other variant overrides
exactly one design choice, so each comparison isolates that choice. Variants are
defined in config as `variants: {name: {SSTM ctor overrides}}`; the default
variant ({} overrides) is emitted automatically by cells() and never listed there.

Run (any machine, like the other experiments):
    uv run python -m experiments.runner.launch --experiment method_ablation --jobs 8      # Mac pool
    uv run python -m experiments.runner.launch --experiment method_ablation --gpus 0,1    # GPU node
    ... --select variant=default,no_sharing --select d=10 --dry-run                       # staging

Cell grammar (variant sits in synthetic's model slot; names contain no '='):
    funnel/hierarch:  {graph}_{variant}_d{d}_n{n}_seed{seed}
    ER/SF:            {graph}_{sem}_{variant}_d{d}_e{e}_n{n}_seed{seed}

Result rows carry the synthetic identity keys (model=variant, sampler, d, n, ...)
so the same readers work on both, plus n_params / n_params_maps / fit_seconds /
seconds_per_epoch (the cost half of the sharing and ordering-relaxation questions).
"""

import time

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

from triangularmap.sstm import SSTM
from experiments.sem import make_data
from experiments.synthetic.experiment import eval_metrics


def _variant_names(cfg):
    return ["default"] + list(cfg["variants"].keys())


def cells(cfg) -> list:
    """Grid: variant x target x d x n x seed. funnel/hierarch have no sem/e."""
    out = []
    for variant in _variant_names(cfg):
        for graph in cfg["graph_type"]:
            sems = cfg["sem"] if graph in ("ER", "SF") else [None]
            es = cfg["e"] if graph in ("ER", "SF") else [None]
            for sem in sems:
                for d in cfg["d"]:
                    for e in es:
                        for n in cfg["n"]:
                            for seed in cfg["seeds"]:
                                out.append({"variant": variant, "graph": graph, "sem": sem,
                                            "d": int(d), "e": (int(e) if e is not None else None),
                                            "n": int(n), "seed": int(seed)})
    return out


def cell_id(c) -> str:
    if c["graph"] in ("funnel", "hierarch"):
        return f"{c['graph']}_{c['variant']}_d{c['d']}_n{c['n']}_seed{c['seed']}"
    return (f"{c['graph']}_{c['sem']}_{c['variant']}_d{c['d']}_e{c['e']}"
            f"_n{c['n']}_seed{c['seed']}")


def parse_cell_id(stem: str) -> dict:
    """Inverse of cell_id (round-trip tested). Accepts a filename stem."""
    m_tail = stem.split("_d")[-1]                      # "{d}_e{e}_n{n}_seed{seed}" or "{d}_n{n}_seed{seed}"
    head = stem[: len(stem) - len(m_tail) - 2]         # strip "_d{tail}"
    parts = m_tail.split("_")
    d = int(parts[0])
    e = None
    rest = parts[1:]
    if rest[0].startswith("e"):
        e = int(rest[0][1:]); rest = rest[1:]
    n = int(rest[0][1:])                               # "n{n}"
    seed = int(rest[1][4:])                            # "seed{seed}"
    first = head.split("_", 1)[0]
    if first in ("funnel", "hierarch"):
        graph, sem, variant = first, None, head.split("_", 1)[1]
    else:
        graph, sem_variant = head.split("_", 1)
        sem, variant = sem_variant.split("_", 1)
    return {"variant": variant, "graph": graph, "sem": sem, "d": d, "e": e, "n": n, "seed": seed}


def logs_cell(c) -> bool:
    return True                                        # every variant has SSTM dynamics


def run_one(cell, cfg, *, logger=None, checkpoint_dir=None, n_checkpoints=0) -> dict:
    """Fit one (variant, target, d, n, seed) cell; same data path as synthetic."""
    variant, graph, sem = cell["variant"], cell["graph"], cell["sem"]
    d, e, n, seed = cell["d"], cell["e"], cell["n"], cell["seed"]
    be = cfg["SSTM"]
    n_test, n_val = int(cfg["n_test"]), int(cfg["n_val"])
    s0 = int(e) * d if e is not None else 0            # ignored for funnel/hierarch

    # one draw, three held-out splits: train | test | val (synthetic convention)
    B, X, log_p, _ = make_data(d, s0, n + n_test + n_val, sem, graph, seed)
    X_train = X[:n]
    X_test,  log_p_test = X[n:n + n_test],                  log_p[n:n + n_test]
    X_val,   log_p_val  = X[n + n_test:n + n_test + n_val], log_p[n + n_test:n + n_test + n_val]

    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train).astype(np.float32)
    X_test  = scaler.transform(X_test).astype(np.float32)
    X_val   = scaler.transform(X_val).astype(np.float32)

    sampler = graph if graph in ("funnel", "hierarch") else f"{graph}_{sem}_e{e}"
    log_scale = np.log(scaler.scale_).sum()
    nll_true_test = float(-(log_p_test + log_scale).mean())
    nll_true_val  = float(-(log_p_val  + log_scale).mean())

    overrides = {} if variant == "default" else dict(cfg["variants"][variant])
    model_kw = dict(input_size=d, hidden=list(be["hidden"]), output_dim=1,
                    disconnect_first_n=be["disconnect_first_n"], random_state=seed,
                    gate_method=be["gate_method"],
                    perm_method=be.get("perm_method", "softsort"))
    model_kw.update(overrides)                         # exactly one design choice changes
    model = SSTM(**model_kw)

    fit_kw = dict(epochs=be["epochs"], batch_size=be["batch_size"], lr_map=be["lr_map"],
                  lr_decay=be.get("lr_decay"), compile=bool(be.get("compile", False)),
                  lr_ordering=be["lr_ordering"], lr_sparsity=be["lr_sparsity"],
                  l2=be["l2"], l1=be.get("l1", None), sparsity_strength=be["sparsity_strength"],
                  optimizer=be.get("optimizer", "adam"),
                  map_refine_tail=be.get("map_refine_tail", False),
                  n_samples=be.get("n_samples", 10), verbose=False)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    model.fit(X_train, **fit_kw, logger=logger, X_val=X_val,
              checkpoint_dir=checkpoint_dir, n_checkpoints=n_checkpoints)
    fit_seconds = time.perf_counter() - t0
    peak_mem = int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else None

    return {"model": variant, "variant": variant, "overrides": overrides,
            "sampler": sampler, "graph": graph, "sem": sem,
            "d": int(d), "e": (int(e) if e is not None else None), "n": int(n), "seed": int(seed),
            "nnz_true": int(B.sum()),
            "n_params": int(sum(p.numel() for p in model.parameters())),
            "n_params_maps": int(sum(p.numel() for p in model.maps.parameters())),
            "fit_seconds": round(fit_seconds, 2),
            "seconds_per_epoch": round(fit_seconds / be["epochs"], 4),
            "epochs": int(be["epochs"]), "n_samples": int(be.get("n_samples", 10)),
            "peak_cuda_mem_mb": (round(peak_mem / 2**20, 1) if peak_mem is not None else None),
            # the structure each variant learned: the variable order and its adjacency
            "hard_order": [int(v) for v in model._hard_order().cpu().numpy()],
            "B_est": np.asarray(model.get_adjacency_matrix(), dtype=int).tolist(),
            **eval_metrics(model, X_train, X_test, X_val, nll_true_test, nll_true_val, B, d)}
