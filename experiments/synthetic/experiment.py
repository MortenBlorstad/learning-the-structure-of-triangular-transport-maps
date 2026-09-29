"""Synthetic density + structure experiment: does learning the order and the sparsity
pattern pay off, and how close does it get to the true density?

Cells span (model, graph, sem, d, e, n, seed) from the config grid. `learned` is the
method; `oracle` and `random` fix the structure to the true parents and to a random
order, `two_stage` takes NOTEARS-MLP's graph, and maf/nsf/gnf are external flow
baselines. Each cell reports test NLL, the gap to the true density, and (where the
model has an adjacency) the structure metrics.

    uv run python -m experiments.runner.launch --experiment synthetic --jobs 8
"""

import numpy as np
import igraph as ig
from sklearn.preprocessing import StandardScaler
from notears import utils
from triangularmap.sstm import SSTM
from experiments.baselines.notears import NotearsMLP
from experiments.baselines.gnf.wrapper import GNFBaseline
from experiments.baselines.flows import MAFBaseline, NSFBaseline
from experiments.sem import make_data
from experiments.common import compute_nll, skeleton_f1


# --------------------------------------------------------------------------- #
# Structure sources: same SSTM, only the order/sparsity source differs.
# maf/nsf are external flow baselines (density only, no structure metrics).
# --------------------------------------------------------------------------- #
MODELS = ["oracle", "random", "two_stage", "learned", "maf", "nsf", "gnf"]


def build_cells(cfg) -> list:
    """Grid of (model, graph, sem, d, e, n, seed) cells; funnel/hier have no sem/e."""
    NO_SEM = ("funnel", "hierarch")                        # fixed generative model, no SEM/e axes
    cells = []
    for m in cfg["models"]:
        for graph in cfg["graph_type"]:
            sems = [None] if graph in NO_SEM else cfg["sem"]
            es = [None] if graph in NO_SEM else cfg["e"]
            for sem in sems:
                for e in es:
                    for d in cfg["d"]:
                        for n in cfg["n"]:
                            for seed in cfg["seeds"]:
                                cells.append((m, graph, sem, int(d),
                                              (int(e) if e is not None else None),
                                              int(n), int(seed)))
    return cells


def cell_id(model, graph, sem, d, e, n, seed) -> str:
    if graph in ("funnel", "hierarch"):                    # fixed generative model: no sem/e in the id
        return f"{graph}_{model}_d{d}_n{n}_seed{seed}"
    return f"{graph}_{sem}_{model}_d{d}_e{e}_n{n}_seed{seed}"


def parent_mask(B, order):
    """[d, d] (rank, parent-feature) mask: m[r, p] = B[p, order[r]] — the true
    parents of the feature introduced at rank r. Same convention (i->j = B[i,j])
    works on a NOTEARS binary adjacency."""
    d = B.shape[0]
    m = np.zeros((d, d), dtype=np.float32)
    for r, feat in enumerate(order):
        m[r, :] = B[:, feat]
    return m


def build_model(model_name, d, seed, B, true_order, X_train, be, notears, random_order=None, flows=None):
    """Construct the model for one cell. The SSTM variants differ only in the structure
    source; maf/nsf are external flow baselines (no structure)."""
    if model_name == "maf":
        c = flows["MAF"]
        return MAFBaseline(c["transforms"], c["hidden_features"], num_blocks=c["num_blocks"],
                           dropout=c["dropout"], scale=c.get("maf_scale", "sigmoid"))
    if model_name == "nsf":
        c = flows["NSF"]
        return NSFBaseline(c["transforms"], c["hidden_features"], num_blocks=c["num_blocks"],
                           dropout=c["dropout"], bins=c["bins"], tail_bound=c["tail_bound"])
    common = dict(input_size=d, hidden=list(be["hidden"]), output_dim=1,
                  disconnect_first_n=be["disconnect_first_n"], random_state=seed,
                  gate_method=be["gate_method"])
    if model_name == "learned":                        # order + sparsity learned
        return SSTM(**common)
    if model_name == "oracle":                         # true order + true sparsity, fixed
        return SSTM(**common, fixed_order=true_order,
                            fixed_sparsity=parent_mask(B, true_order))
    if model_name == "random":                         # random order fixed, sparsity learned
        rand_order = random_order if random_order is not None else list(np.random.default_rng(seed).permutation(d))
        return SSTM(**common, fixed_order=rand_order)
    if model_name == "two_stage":                      # order + sparsity from NOTEARS, fixed
        ns = NotearsMLP(hidden=notears["hidden"], lambda1=notears["lambda1"],
                        lambda2=notears["lambda2"], w_threshold=notears["w_threshold"],
                        max_iter=notears["max_iter"]).fit(X_train)
        A = ns.get_adjacency_matrix()
        order_ns = ig.Graph.Adjacency(A.tolist()).topological_sorting()
        return SSTM(**common, fixed_order=order_ns,
                            fixed_sparsity=parent_mask(A, order_ns))
    raise ValueError(f"unknown model '{model_name}' (expected {MODELS})")


def eval_metrics(model, X_train, X_test, X_val, nll_true_test, nll_true_val, B, d) -> dict:
    """Density + structure metrics for one fitted model (shared with method_ablation)."""
    nll_test = compute_nll(model, X_test)
    nll_val  = compute_nll(model, X_val)
    nll_val_pt = -model.log_pdf(X_val)                   # per-point NLL
    out = {"nll_train": compute_nll(model, X_train),
           "nll_test": nll_test, "nll_val": nll_val,
           "nll_val_median": float(np.median(nll_val_pt))}   # robust to heavy-tail blowups
    if nll_true_test is not None:      # real data (sachs) has no true density -> no gap fields
        out.update({"nll_true_test": nll_true_test, "gap_to_true": nll_test - nll_true_test,
                    "nll_true_val": nll_true_val, "gap_to_true_val": nll_val - nll_true_val})
    # Val tail diagnostics: on heavy-tailed targets (funnel) the MEAN val NLL is dominated by
    # a handful of extreme-x points the map extrapolates to huge |z| (0.5|z|^2 blows up). Record
    # the extrapolation size so the tail can be told apart from the bulk. Flow baselines have
    # no pushforward/adjacency, so they emit density metrics only (structure keys absent).
    if hasattr(model, "pushforward"):
        z_val = model.pushforward(X_val)                     # [n_val, d]
        z_absmax = np.abs(z_val).max(axis=1)                 # worst |z_i| per val point
        out["val_z_absmax"] = float(z_absmax.max())          # single worst extrapolation
        out["val_n_blowup20"] = int((z_absmax > 20).sum())   # # points mapped past |z|=20
    if hasattr(model, "get_adjacency_matrix"):
        B_est = np.asarray(model.get_adjacency_matrix(), dtype=int)
        acc = utils.count_accuracy(B, B_est)     # shd (to DAG), tpr, fdr, fpr, nnz
        prec = 1.0 - acc["fdr"]                   # directed precision
        dir_f1 = (2 * prec * acc["tpr"] / (prec + acc["tpr"])) if (prec + acc["tpr"]) > 0 else 0.0
        out.update({"dir_f1": dir_f1,                       # directed F1 = 2*prec*tpr/(prec+tpr), prec=1-fdr
                    "skeleton_f1": skeleton_f1(B, B_est),   # read-out only (never in loss/forward/keep-rule)
                    "edge_density": float(B_est.sum()) / (d * (d - 1) / 2) if d > 1 else 0.0,  # [0,1]: 0=independent, 1=complete DAG
                    **acc})
    return out


def eval_both(model, X_train, X_test, X_val, nll_true_test, nll_true_val, B, d) -> dict:
    """eval_metrics under the model's headline weights (unsuffixed keys) plus, when fit kept
    both checkpoints (best_state_/last_state_ with X_val), the full metric set at the last and
    at the best-validation epoch as *_last / *_best. Convention: unsuffixed = the model's
    reporting convention (SSTM: last epoch; flows: best-validation, their papers');
    the suffixed copies make the other convention inspectable without rerunning."""
    out = eval_metrics(model, X_train, X_test, X_val, nll_true_test, nll_true_val, B, d)
    best, last = getattr(model, "best_state_", None), getattr(model, "last_state_", None)
    if best is None or last is None:
        return out
    net = getattr(model, "flow", model)                     # flows wrap the torch module
    current = {k: v.detach().to("cpu", copy=True) for k, v in net.state_dict().items()}
    for tag, state in (("last", last), ("best", best)):
        net.load_state_dict(state)
        m = eval_metrics(model, X_train, X_test, X_val, nll_true_test, nll_true_val, B, d)
        out.update({f"{k}_{tag}": v for k, v in m.items()})
    net.load_state_dict(current)
    if getattr(model, "best_epoch_", None) is not None:
        out["best_epoch"] = int(model.best_epoch_)
        out["best_val_nll"] = float(model.best_val_nll_)
    return out


def run_one(model_name, graph, sem, d, e, n, seed, cfg, logger=None,
            checkpoint_dir=None, n_checkpoints=0) -> dict:
    """Fit one (model, graph, sem, d, e, n, seed) cell; return a flat metrics dict (nats).

    Standardize on TRAIN (shared scaler); held-out test = a fresh slice of the same
    generative process. gap_to_true = NLL_model - NLL_true, with the true log-density
    corrected to standardized space by + sum(log scaler.scale_). Funnel has no sem/e.
    An optional TrainingLogger records the SSTM training dynamics.
    """
    be, notears = cfg["SSTM"], cfg["NotearsMLP"]
    n_test, n_val = int(cfg["n_test"]), int(cfg["n_val"])
    s0 = int(e) * d if e is not None else 0            # ignored for funnel

    # one draw, three held-out splits of the same generative process: train | test | val
    B, X, log_p, true_order = make_data(d, s0, n + n_test + n_val, sem, graph, seed)
    X_train = X[:n]
    X_test,  log_p_test = X[n:n + n_test],          log_p[n:n + n_test]
    X_val,   log_p_val  = X[n + n_test:n + n_test + n_val], log_p[n + n_test:n + n_test + n_val]

    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train).astype(np.float32)
    X_test  = scaler.transform(X_test).astype(np.float32)
    X_val   = scaler.transform(X_val).astype(np.float32)

    sampler = graph if graph in ("funnel", "hierarch") else f"{graph}_{sem}_e{e}"   # plot-grouping id
    log_scale = np.log(scaler.scale_).sum()
    nll_true_test = float(-(log_p_test + log_scale).mean())
    nll_true_val  = float(-(log_p_val  + log_scale).mean())
    fit_kw = dict(epochs=be["epochs"], batch_size=be["batch_size"], lr_map=be["lr_map"], lr_ordering=be["lr_ordering"], lr_sparsity=be["lr_sparsity"],
                  l2=be["l2"], l1=be.get("l1", None), sparsity_strength=be["sparsity_strength"],
                  optimizer=be.get("optimizer", "adam"), map_refine_tail=be.get("map_refine_tail", False),
                  n_samples=be.get("n_samples", 10), verbose=False)

    base = {"model": model_name, "sampler": sampler, "graph": graph, "sem": sem,
            "d": int(d), "e": (int(e) if e is not None else None), "n": int(n), "seed": int(seed),
            "nnz_true": int(B.sum())}

    if model_name in ("maf", "nsf"):
        # external flow baseline: own fit signature (no SSTM kwargs, no checkpoints);
        # density metrics only. Seeded globally by run_cell (FlowBase does not self-seed;
        # nflows' RandomPermutation and init draw from the global torch RNG).
        # Training dynamics go through the same TrainingLogger destination as SSTM:
        # FlowBase records train/val NLL via eval_every, adapted here into logger.history
        flows = {"MAF": cfg.get("MAF"), "NSF": cfg.get("NSF")}
        c = flows[model_name.upper()]
        model = build_model(model_name, d, seed, B, true_order, X_train, be, notears, flows=flows)
        model.fit(X_train, epochs=c["epochs"], batch_size=c["batch_size"],
                  lr=c["lr"], weight_decay=c["weight_decay"],
                  X_val=X_val,   # always passed: best-val restore (the flows' papers' protocol);
                  eval_every=(logger.log_every if logger is not None else 0))
        if logger is not None:
            h = model.history_
            logger.history = [
                {"epoch": e, "nll": tr, "train_nll_eval": tr, "val_nll": v,
                 "val_nll_median": vmed, "val_nll_max": vmax}
                for e, tr, v, vmed, vmax in zip(h["epoch"], h["nll_train"], h["nll_val"],
                                                h["nll_val_median"], h["nll_val_max"])
            ]
        return {**base, **eval_both(model, X_train, X_test, X_val, nll_true_test, nll_true_val, B, d), "n_random_orders": 1}

    if model_name == "gnf":
        # Graphical Normalizing Flow (Wehenkel & Louppe 2021): the joint graph+density
        # baseline, via the sachs wrapper (committed-gated best-val, in-run post-processing);
        # adjacency scored like SSTM via get_adjacency_matrix in eval_metrics.
        g = cfg["GNF"]
        # epochs / cadence / gamma may be given per training size n (dict) or as a plain
        # value (e.g. from an --ablate override), which is used as-is.
        per_n = lambda v: v[n] if isinstance(v, dict) else v
        model = GNFBaseline(emb_net=g["emb_net"], int_net=g["int_net"], nb_steps=g["nb_steps"],
                            solver=g["solver"], l1=g["l1"],
                            nb_steps_dual=per_n(g["nb_steps_dual"]),
                            gamma=per_n(g.get("gamma")))   # None = their released 0.9
        ep = per_n(g["epochs"])
        model.fit(X_train, epochs=ep, batch_size=g["batch_size"], lr=g["lr"],
                  weight_decay=g["weight_decay"], X_val=X_val,
                  verbose=g.get("verbose", False),   # --set GNF.verbose=true -> per-100-epoch progress
                  eval_every=(logger.log_every if logger is not None else 0))
        if logger is not None:
            h = model.history_
            logger.history = [
                {"epoch": ep, "nll": tr, "train_nll_eval": tr, "val_nll": v,
                 "val_nll_median": vmed, "val_nll_max": vmax, "dagness": dg}
                for ep, tr, v, vmed, vmax, dg in zip(h["epoch"], h["nll_train"], h["nll_val"],
                                                     h["nll_val_median"], h["nll_val_max"],
                                                     h["dagness"])
            ]
        return {**base, **eval_both(model, X_train, X_test, X_val, nll_true_test, nll_true_val, B, d),
                "n_random_orders": 1, "dagness_final": model.dagness_final_,
                "gnf_events": model.step_log_[:20]}

    if model_name == "random":
        # baseline: ONE random order per seed (seed-derived in build_model), single fit, no logging --
        # the seed axis provides the spread, like every other model (no inner K-order averaging).
        model = build_model("random", d, seed, B, true_order, X_train, be, notears)
        model.fit(X_train, **fit_kw, logger=None, X_val=X_val)   # report-only trace -> best/last kept
        return {**base, **eval_both(model, X_train, X_test, X_val, nll_true_test, nll_true_val, B, d), "n_random_orders": 1}

    model = build_model(model_name, d, seed, B, true_order, X_train, be, notears)
    model.fit(X_train, **fit_kw, logger=logger, X_val=X_val,
              checkpoint_dir=checkpoint_dir, n_checkpoints=n_checkpoints)
    return {**base, **eval_both(model, X_train, X_test, X_val, nll_true_test, nll_true_val, B, d), "n_random_orders": 1}
