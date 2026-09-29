"""Real-data density estimation: SSTM vs MAF/NSF/GNF by test NLL on the UCI suite.

The flow baselines are MAF and RQ-NSF(AR) as implemented in the NSF authors' nflows
library (Table 5 setup), plus GNF. Datasets POWER, GAS, HEPMASS, MINIBOONE (MAF-paper
splits, fixed by the loaders, so experiment seeds vary only init and batch order). There
is no true density and no true graph here, so the metric is test NLL against the flow
baselines. The learned order and sparsity are qualitative readouts.

Protocol: one StandardScaler fit on TRAIN shared by all models. ONE training budget for
every model — the NSF paper's Table 5 per-dataset batch size and step count, converted
to whole epochs — so all models get the same passes over the data. SSTM, MAF and
NSF also share the batch size, hence the update count; GNF keeps the batch size of its
authors' own per-dataset configuration, so it takes the same passes at a different step
count (steps_actual records each model's own). The flow baselines follow their papers'
protocol (cosine annealing, per-epoch best-val checkpointing; GNF's best-val is gated on
its conditioner having committed to a DAG); SSTM trains its fixed budget with no
selection — its VAL numbers are inspection only.

Run (any machine, like the other experiments; data must be downloaded once, see
experiments/uci/README.md):
    uv run python -m experiments.runner.launch --experiment uci --jobs 4      # Mac pool
    uv run python -m experiments.runner.launch --experiment uci --gpus 0,1    # GPU node
    ... --select model=MAF,NSF --dry-run                                          # preview

Cell grammar: {dataset}_{model}_seed{seed}. Result rows carry the identity keys
(sampler=dataset, model, d, n=train size) plus nll_train/val/test, n_params,
fit_seconds, and (SSTM only) nnz / hard_order / B_est.
"""
from __future__ import annotations

import math
import time

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

from triangularmap.sstm import SSTM
from experiments.baselines.gnf.wrapper import GNFBaseline
from experiments.baselines.flows import MAFBaseline, NSFBaseline
from experiments.uci.data import load_uci


def cells(cfg) -> list:
    return [{"dataset": ds, "model": m, "seed": int(s)}
            for ds in cfg["datasets"] for m in cfg["models"] for s in cfg["seeds"]]


def cell_id(c) -> str:
    return f"{c['dataset']}_{c['model']}_seed{c['seed']}"


def parse_cell_id(stem: str) -> dict:
    """Inverse of cell_id (round-trip tested). Accepts a filename stem.
    Dataset names and the seed token never contain underscores, so split from the
    ends (model names may)."""
    parts = stem.split("_")
    return {"dataset": parts[0], "model": "_".join(parts[1:-1]), "seed": int(parts[-1][4:])}


def logs_cell(c) -> bool:
    return c["model"] == "SSTM"          # only the learned map has training dynamics


def batched_nll(model, X, batch: int) -> float:
    """Mean NLL via model.log_pdf in chunks — log_pdf is a single full-array forward
    pass in both model families, and the UCI eval splits run to 205k rows."""
    lps = [model.log_pdf(X[i:i + batch]) for i in range(0, len(X), batch)]
    return float(-np.mean(np.concatenate(lps)))


def run_one(cell, cfg, *, logger=None, checkpoint_dir=None, n_checkpoints=0) -> dict:
    """Fit one (dataset, model, seed) cell; return a flat metrics dict (nats)."""
    dataset, model_name, seed = cell["dataset"], cell["model"], int(cell["seed"])
    eval_batch = int(cfg.get("eval_batch", 8192))

    X_train, X_val, X_test = load_uci(dataset)         # un-normalized; splits fixed by loader
    n_train = cfg.get("n_train")                        # optional train subsample (the loader has
    if n_train:                                         # already shuffled the split with a fixed
        X_train = X_train[:int(n_train)]                # RNG); the row records the resulting n
    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train).astype(np.float32)
    X_val   = scaler.transform(X_val).astype(np.float32)
    X_test  = scaler.transform(X_test).astype(np.float32)
    n, d = X_train.shape

    # ONE training budget for every model: the NSF paper's Table 5 per-dataset batch size
    # and step count, converted to whole epochs (drop_last convention: n // batch steps
    # per epoch). Same passes over the data and same update count for all three models;
    # SSTM's fit keeps the final partial batch, so its steps run one partial
    # batch per epoch over the flows' (steps_actual reports each model's own count).
    c_ds = cfg["flows"]["per_dataset"][dataset]
    batch_size, steps = int(c_ds["batch_size"]), int(c_ds["steps"])
    steps_per_epoch = n // batch_size
    epochs = max(1, math.ceil(steps / steps_per_epoch))

    if model_name == "SSTM":
        be = cfg["SSTM"]
        # per-dataset size override (like the flows' Table-5 per-dataset columns): the small
        # MINIBOONE overfits the uniform width -- hidden_per_dataset wins over hidden.
        hidden = list(be.get("hidden_per_dataset", {}).get(dataset, be["hidden"]))
        # Diagnostic structure knobs (default: both learned). fixed_order "identity" pins the
        # data's column order; dense_gates fixes every earlier variable as a parent (no
        # pruning). Both fixed = the map alone (fit then uses its fixed-structure regime).
        order = np.arange(d) if be.get("fixed_order") == "identity" else None
        # (rank, feature) mask; the ordering's cumulative permutation already limits each
        # rank to earlier features, so all-ones = every earlier variable is a parent.
        sparsity = np.ones((d, d), dtype=np.float32) if be.get("dense_gates") else None
        model = SSTM(input_size=d, hidden=hidden, output_dim=1,
                             disconnect_first_n=be["disconnect_first_n"], random_state=seed,
                             gate_method=be["gate_method"],
                             perm_method=be.get("perm_method", "softsort"),
                             fixed_order=order, fixed_sparsity=sparsity)
        # full X_val: per-epoch val NLL trace in fit (report-only, no selection) + the
        # logger's 512-row probe; the reported nll_val below uses the full split too.
        fit_kw = dict(epochs=epochs, batch_size=batch_size, lr_map=be["lr_map"],
                      lr_ordering=be["lr_ordering"], lr_sparsity=be["lr_sparsity"],
                      l2=be["l2"], l1=be.get("l1", None),
                      sparsity_strength=be["sparsity_strength"],
                      optimizer=be.get("optimizer", "adam"),
                      n_samples=be.get("n_samples", 10), warmup=be.get("warmup", 50),
                      lr_decay=be.get("lr_decay"),
                      verbose=False,
                      logger=logger, X_val=X_val,
                      checkpoint_dir=checkpoint_dir, n_checkpoints=n_checkpoints)
    elif model_name in ("MAF", "NSF"):
        # nflows (the NSF authors' library), NSF-paper Table 5 architecture per dataset
        # (budget resolved above)
        fl = cfg["flows"]
        c = {**{k: v for k, v in fl.items() if k != "per_dataset"},
             **fl["per_dataset"][dataset]}
        h, nb = int(c["hidden_features"]), int(c["num_blocks"])
        if model_name == "MAF":
            model = MAFBaseline(c["transforms"], h, num_blocks=nb, dropout=c["dropout"],
                              scale=c.get("maf_scale", "sigmoid"))
        else:
            model = NSFBaseline(c["transforms"], h, num_blocks=nb, dropout=c["dropout"],
                              bins=c["bins"], tail_bound=c["tail_bound"])
        # X_val drives the papers' best-val checkpointing (fit restores the best epoch)
        fit_kw = dict(epochs=epochs, batch_size=batch_size, lr=c["lr"],
                      weight_decay=c.get("weight_decay", 0.0), verbose=False, X_val=X_val)
    elif model_name == "GNF":
        # the authors' own per-dataset configuration (config block in experiments/config.yaml),
        # at the shared epoch budget resolved above. Their batch size, not the flows', so the
        # step count differs -- steps_actual below records GNF's own.
        g = cfg["GNF"]
        c = {**{k: v for k, v in g.items() if k != "per_dataset"}, **g["per_dataset"][dataset]}
        model = GNFBaseline(emb_net=c["emb_net"], int_net=c["int_net"], nb_steps=c["nb_steps"],
                            solver=c["solver"], l1=c["l1"], nb_steps_dual=c["nb_steps_dual"],
                            gumble_T=c["gumble_T"])
        batch_size = int(c["batch_size"])
        steps_per_epoch = n // batch_size            # their fit drops the last partial batch
        # eval_every=0: no history logging, so the val pass runs only once the conditioner has
        # committed to a binary DAG, which is what best-val selection can use (wrapper docstring)
        fit_kw = dict(epochs=epochs, batch_size=batch_size, lr=c["lr"],
                      weight_decay=c["weight_decay"], X_val=X_val, eval_every=0,
                      verbose=c.get("verbose", False))   # --set GNF.verbose=true -> per-100-epoch
                                                         # loss / dagness / c / lambda progress
    else:
        raise ValueError(f"unknown model {model_name!r} (expected: SSTM, MAF, NSF, GNF)")

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    model.fit(X_train, **fit_kw)
    fit_seconds = time.perf_counter() - t0
    peak_mem = int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else None

    params = model.parameters() if model_name == "SSTM" else model.flow.parameters()
    row = {"model": model_name, "dataset": dataset, "seed": seed,
           "sampler": dataset, "d": int(d), "n": int(n),   # shared identity keys
           "nll_train": batched_nll(model, X_train, eval_batch),
           "nll_val": batched_nll(model, X_val, eval_batch),      # flows: selection val (papers'
                                                                  # protocol); SSTM: inspection only
           "nll_test": batched_nll(model, X_test, eval_batch),
           "n_params": int(sum(p.numel() for p in params)),
           "fit_seconds": round(fit_seconds, 2),
           "seconds_per_epoch": round(fit_seconds / epochs, 4),
           "epochs": int(epochs), "steps_budget": steps,
           "steps_actual": int(epochs * (math.ceil(n / batch_size) if model_name == "SSTM"
                                         else steps_per_epoch)),
           "batch_size": batch_size,
           "peak_cuda_mem_mb": (round(peak_mem / 2**20, 1) if peak_mem is not None else None)}
    if model_name == "MAF":
        row["maf_scale"] = model.scale                     # ablation provenance
    if model_name == "GNF":
        row["dagness_final"] = model.dagness_final_        # visible non-certification
        row["nan_epoch"] = model.nan_epoch_                # not None => training aborted, NLL untrained
        row["gnf_events"] = model.step_log_[:20]           # post-processing narrative
        A = np.asarray(model.get_adjacency_matrix(), dtype=int)   # read-out: l1 = 0, so expect
        row.update({"nnz": int(A.sum()),                          # a near-complete DAG
                    "edge_density": float(A.sum()) / (d * (d - 1) / 2)})
    # Both checkpoints evaluated, so the table can be built either way. nll_test above is
    # the protocol number: flows = best-val weights (their paper), SSTM = last epoch.
    if getattr(model, "best_epoch_", None) is not None:
        row.update({"best_epoch": int(model.best_epoch_), "best_val_nll": float(model.best_val_nll_)})
        net = model if model_name == "SSTM" else model.flow
        current = {k: v.detach().to("cpu", copy=True) for k, v in net.state_dict().items()}
        for tag, state in (("last", model.last_state_), ("best", model.best_state_)):
            net.load_state_dict(state)
            row[f"nll_val_{tag}"] = batched_nll(model, X_val, eval_batch)
            row[f"nll_test_{tag}"] = batched_nll(model, X_test, eval_batch)
        net.load_state_dict(current)
    if getattr(model, "val_history_", None):                 # SSTM per-epoch trace
        row["val_history"] = [round(float(v), 6) for v in model.val_history_]
    if model_name == "SSTM":
        B_est = np.asarray(model.get_adjacency_matrix(), dtype=int)
        row.update({"n_params_maps": int(sum(p.numel() for p in model.maps.parameters())),
                    "nnz": int(B_est.sum()),
                    "edge_density": float(B_est.sum()) / (d * (d - 1) / 2),
                    "hard_order": [int(v) for v in model._hard_order().cpu().numpy()],
                    "B_est": B_est.tolist()})
        if getattr(model, "best_state_", None) is not None:   # structure at the best-val epoch,
            current = model._cpu_state()                      # same as the NLL quartet above
            model.load_state_dict(model.best_state_)
            Bb = np.asarray(model.get_adjacency_matrix(), dtype=int)
            row.update({"nnz_best": int(Bb.sum()),
                        "hard_order_best": [int(v) for v in model._hard_order().cpu().numpy()],
                        "B_est_best": Bb.tolist()})
            model.load_state_dict(current)
    return row
