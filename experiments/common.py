"""Small helpers shared by the experiments: results paths, git provenance, and the
two metrics that several experiments compute the same way (mean NLL, skeleton F1)."""
import subprocess
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]


def results_dir(name):
    """Repo-root/results/<name> directory for an experiment; created if missing."""
    d = REPO_ROOT / "results" / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def git_provenance() -> dict:
    """Commit + tracked-dirty flag, stamped into every run's output for reproducibility.

    `git diff --quiet HEAD` is tracked-only, so gitignored `results/` never marks a run
    dirty; `git_dirty: true` then honestly means the code was uncommitted at run time.
    Returns Nones (never raises) if git is unavailable — a run must not die over provenance.
    """
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
        dirty = subprocess.call(["git", "diff", "--quiet", "HEAD"], cwd=REPO_ROOT) != 0
        return {"git_commit": sha, "git_dirty": dirty}
    except Exception:
        return {"git_commit": None, "git_dirty": None}


def compute_nll(model, X):
    """Mean negative log-likelihood via model.log_pdf (no exp/clip roundtrip)."""
    return float(-np.mean(model.log_pdf(X)))


def skeleton_f1(B_true, B_est) -> float:
    """F1 of the UNDIRECTED skeleton (adjacency, direction ignored). Read-out only —
    isolates 'found the right adjacency' from 'oriented it' (orientation is partly
    unidentifiable from observational data, so a share of SHD-to-DAG is coin-flip)."""
    skeleton_true = (np.asarray(B_true) + np.asarray(B_true).T) != 0 
    skeleton_estimated = (np.asarray(B_est) + np.asarray(B_est).T) != 0
    indices_upper = np.triu_indices_from(skeleton_true, k=1)          # each undirected pair once
    t, e = skeleton_true[indices_upper], skeleton_estimated[indices_upper]
    tp = int((t & e).sum()); fp = int((~t & e).sum()); fn = int((t & ~e).sum()) # tp=true positive, fp=false positive, fn=false negative
    denom = 2 * tp + fp + fn
    return float(2 * tp / denom) if denom > 0 else 1.0 
