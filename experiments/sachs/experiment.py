"""Sachs 2005 — real data with a known sparse graph.

The synthetic model set (learned / random / oracle / two_stage + MAF / NSF) on the
pooled 7,466 x 11 continuous Sachs matrix, with the 20-edge consensus network as the
reference graph (fetched once per machine into experiments/sachs/data/; see
experiments/sachs/README.md for the curl commands -- the files are deliberately not
committed). The oracle
fixes the reference graph; two_stage estimates one with NOTEARS-MLP (reproduction anchor:
SHD ~ 16, ~13 edges, arXiv:1909.13189).

Two subsets, both scored against the same 20-edge reference: ``pooled`` (all 7,466
rows, the 14 conditions incl. interventions — the NOTEARS-comparable setting) and
``obs853`` (the first 853 rows = the observational cd3/cd28 condition, verified
byte-identical to bnlearn's observational file, guarded by a checksum).
Unlike UCI (fixed literature splits), each seed draws its own 80/10/10 split
(pooled 5,972/747/747; obs853 682/85/86) AND its own model initialization — mirroring
the synthetic per-seed data draw. No true density exists, so rows carry no nll_true/gap fields (eval_metrics
omits them when nll_true is None); structure metrics are measured against the reference.

Run:  uv run python -m experiments.runner.launch --experiment sachs [--select subset=obs853]
"""
from __future__ import annotations

from pathlib import Path

import igraph as ig
import numpy as np
from sklearn.preprocessing import StandardScaler

from experiments.baselines.gnf.wrapper import GNFBaseline
from experiments.synthetic.experiment import build_model, eval_both

DATA_DIR = Path(__file__).parent / "data"
COLUMNS = ["raf", "mek", "plc", "pip2", "pip3", "erk", "akt", "pka", "pkc", "p38", "jnk"]
MODELS = ["learned", "random", "oracle", "two_stage", "maf", "nsf", "gnf"]
SUBSETS = ["pooled", "obs853"]
N_ROWS, D = 7466, 11
N_OBS = 853                      # rows of condition 1 (cd3/cd28, observational) at the top of the file
N_EDGES_TRUE = 20
# sha256 of X[:853].tobytes() (float32), fixed after verifying the block is
# byte-identical to bnlearn's 853-row observational file -- guards the condition boundary.
OBS853_SHA256 = "be97bf60b1e3aa532e0f7ce1cca44d3764ad512af4c281d703575ff50cff3060"


def load_sachs(subset: str = "pooled"):
    """(X float32, B_true [11, 11] int with 20 edges); subset = "pooled" (7,466 rows) or
    "obs853" (the first 853 = the observational condition). Asserts provenance."""
    data_path = DATA_DIR / "sachs.2005.continuous.txt"
    if not data_path.exists():
        raise FileNotFoundError(
            f"Sachs data not found in {DATA_DIR} -- it is not committed. "
            "Run the curl commands in experiments/sachs/README.md.")
    header = data_path.open().readline().split()
    assert header == COLUMNS, f"column drift: {header}"
    X = np.loadtxt(data_path, skiprows=1, dtype=np.float32)
    assert X.shape == (N_ROWS, D), f"shape drift: {X.shape}"

    idx = {c: i for i, c in enumerate(COLUMNS)}
    B = np.zeros((D, D), dtype=int)
    for line in (DATA_DIR / "sachs.2005.ground.truth.graph.txt").open():
        if "-->" not in line:
            continue
        a, _, b = line.strip().split(". ", 1)[1].split()    # "1. erk --> akt"
        B[idx[a], idx[b]] = 1
    assert int(B.sum()) == N_EDGES_TRUE, f"edge-count drift: {B.sum()}"
    assert ig.Graph.Adjacency(B.tolist()).is_dag(), "reference graph is not a DAG"
    if subset == "obs853":
        import hashlib
        X = np.ascontiguousarray(X[:N_OBS])
        assert hashlib.sha256(X.tobytes()).hexdigest() == OBS853_SHA256, \
            "observational block drift -- upstream file changed; re-verify vs bnlearn"
    else:
        assert subset == "pooled", f"unknown subset {subset!r} (expected {SUBSETS})"
    return X, B


def cells(cfg) -> list:
    return [{"subset": sub, "model": m, "seed": s}
            for sub in cfg["subsets"] for m in cfg["models"] for s in cfg["seeds"]]


def cell_id(c) -> str:
    return f"sachs_{c['subset']}_{c['model']}_seed{c['seed']}"


def parse_cell_id(stem: str) -> dict:
    assert stem.startswith("sachs_"), stem
    subset, rest = stem[len("sachs_"):].split("_", 1)     # subset tokens carry no underscore
    model, seed = rest.rsplit("_seed", 1)
    return {"subset": subset, "model": model, "seed": int(seed)}


def logs_cell(c) -> bool:
    return c["model"] in ("learned", "oracle", "maf", "nsf", "gnf")


def run_one(cell, cfg, *, logger=None, checkpoint_dir=None, n_checkpoints=0) -> dict:
    """Fit one (model, seed) cell. Global np/torch seeding is done by the runner."""
    model_name, seed, subset = cell["model"], int(cell["seed"]), cell["subset"]
    be, notears = cfg["SSTM"], cfg["NotearsMLP"]

    X, B = load_sachs(subset)
    n_total, d = X.shape
    n_train = int(0.8 * n_total)                       # pooled 5,972 / obs853 682
    n_val = (n_total - n_train) // 2                   # pooled 747 / obs853 85 (test = rest)
    perm = np.random.default_rng(seed).permutation(n_total)
    X_train = X[perm[:n_train]]
    X_val = X[perm[n_train:n_train + n_val]]
    X_test = X[perm[n_train + n_val:]]

    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train).astype(np.float32)
    X_val = scaler.transform(X_val).astype(np.float32)
    X_test = scaler.transform(X_test).astype(np.float32)

    true_order = ig.Graph.Adjacency(B.tolist()).topological_sorting()
    base = {"model": model_name, "subset": subset, "sampler": f"sachs_{subset}",
            "d": int(d), "n": int(n_train), "seed": seed, "nnz_true": int(B.sum())}

    if model_name == "gnf":
        # Graphical Normalizing Flow (their protocol: DAGness-gated best-val, two-phase
        # training with in-run post-processing; adjacency scored via get_adjacency_matrix)
        g = cfg["GNF"]
        
        model = GNFBaseline(emb_net=g["emb_net"], int_net=g["int_net"], nb_steps=g["nb_steps"],
                            solver=g["solver"], l1=g["l1"], nb_steps_dual=g["nb_steps_dual"])
        print(g["epochs"][subset])
        model.fit(X_train, epochs=g["epochs"][subset], batch_size=g["batch_size"], lr=g["lr"],
                  weight_decay=g["weight_decay"], X_val=X_val,
                  eval_every=(logger.log_every if logger is not None else 0))
        if logger is not None:
            h = model.history_
            logger.history = [
                {"epoch": e, "nll": tr, "train_nll_eval": tr, "val_nll": v,
                 "val_nll_median": vmed, "val_nll_max": vmax, "dagness": dg}
                for e, tr, v, vmed, vmax, dg in zip(h["epoch"], h["nll_train"], h["nll_val"],
                                                    h["nll_val_median"], h["nll_val_max"],
                                                    h["dagness"])
            ]
        row = {**base, **eval_both(model, X_train, X_test, X_val, None, None, B, d)}
        row["dagness_final"] = model.dagness_final_       # acyclicity measure at the end of training
        row["gnf_events"] = model.step_log_[:20]          # first 20 augmented-Lagrangian / post-processing steps
        return row

    if model_name in ("maf", "nsf"):
        flows = {"MAF": cfg["MAF"], "NSF": cfg["NSF"]}
        c = flows[model_name.upper()]
        model = build_model(model_name, d, seed, B, true_order, X_train, be, notears, flows=flows)
        model.fit(X_train, epochs=c["epochs"], batch_size=c["batch_size"],
                  lr=c["lr"], weight_decay=c["weight_decay"],
                  X_val=X_val,   # best-val restore, the flows' papers' protocol
                  eval_every=(logger.log_every if logger is not None else 0))
        if logger is not None:
            h = model.history_
            logger.history = [
                {"epoch": e, "nll": tr, "train_nll_eval": tr, "val_nll": v,
                 "val_nll_median": vmed, "val_nll_max": vmax}
                for e, tr, v, vmed, vmax in zip(h["epoch"], h["nll_train"], h["nll_val"],
                                                h["nll_val_median"], h["nll_val_max"])
            ]
        return {**base, **eval_both(model, X_train, X_test, X_val, None, None, B, d)}

    fit_kw = dict(epochs=be["epochs"], batch_size=be["batch_size"], lr_map=be["lr_map"],
                  lr_ordering=be["lr_ordering"], lr_sparsity=be["lr_sparsity"],
                  l2=be["l2"], l1=be.get("l1", None), sparsity_strength=be["sparsity_strength"],
                  optimizer=be.get("optimizer", "adam"),
                  map_refine_tail=be.get("map_refine_tail", False),
                  n_samples=be.get("n_samples", 10), verbose=False)
    model = build_model(model_name, d, seed, B, true_order, X_train, be, notears)
    model.fit(X_train, **fit_kw, X_val=X_val,
              logger=(logger if model_name != "random" else None),
              checkpoint_dir=checkpoint_dir, n_checkpoints=n_checkpoints)
    return {**base, **eval_both(model, X_train, X_test, X_val, None, None, B, d)}
