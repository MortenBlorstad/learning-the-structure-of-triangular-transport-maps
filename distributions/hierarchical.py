"""Bayesian hierarchical normal model over a two-level DAG.

Builds the DAG, `sample_hierarchical_distribution(n, B, roles)` samples from it
(optionally returning log_pdf). Uses the global numpy RNG, so callers control
the seed (np.random.seed / notears.utils.set_random_seed).

Structure:

    mu       ~ N(0, mu_sd^2)                  location root
    log_tau  ~ N(0, tau_sd^2)                 scale root
    theta_j  ~ N(mu, exp(log_tau)^2)          j = 1..J, child of BOTH roots
    y_j      ~ N(theta_j, y_sd^2)             j = 1..J, child of theta_j

d = 2 + 2J, so d must be even and >= 6.  d=10 -> J=4 (12 edges),
d=20 -> J=9 (27 edges).

Two roots of different KINDS: mu shifts the mean of every theta, log_tau
modulates their variance.  Marginalising log_tau induces strong nonlinear
coupling among the thetas; marginalising mu induces only weak coupling.  So the
two edge families (mu->theta vs log_tau->theta) can be scored separately in the
same run.

`tau_sd` is the difficulty knob: it varies the density geometry with the DAG
held fixed.  tau_sd ~ 1-2 gives funnel-like heavy tails; tau_sd ~ 0.05 is
effectively homoscedastic (do not pass exactly 0 -- log_tau would be
degenerate and log_pdf would be -inf).
"""
import numpy as np

MU_SD = 3.0
TAU_SD = 1.0
Y_SD = 1.0


def hierarchical_adjacency(d):
    """Two-level hierarchical DAG on d = 2 + 2J nodes, roles placed at random.

    Returns (B, roles) where B[i, j] = 1 means i -> j and roles is a dict with
    keys 'mu', 'log_tau' (ints) and 'theta', 'y' (arrays, paired by position).
    """
    if d < 6 or d % 2:
        raise ValueError(f"d must be even and >= 6 (d = 2 + 2J); got {d}")
    J = (d - 2) // 2
    perm = np.random.permutation(d)
    mu, log_tau = int(perm[0]), int(perm[1])
    theta, y = perm[2:2 + J], perm[2 + J:]

    B = np.zeros((d, d), dtype=int)
    for t, yy in zip(theta, y):
        B[mu, t] = 1
        B[log_tau, t] = 1
        B[t, yy] = 1

    roles = {"mu": mu, "log_tau": log_tau, "theta": theta, "y": y}
    return B, roles


def recover_roles(B):
    """Recover roles from B alone.

    mu and log_tau are graph-symmetric (identical in/out degree), so which root
    is which is NOT recoverable from B.  Convention here: the lower-indexed root
    is mu.  Pass the `roles` dict from `hierarchical_adjacency` instead whenever
    you need the true assignment -- e.g. when scoring mu->theta separately from
    log_tau->theta.
    """
    indeg, outdeg = B.sum(axis=0), B.sum(axis=1)
    roots = np.flatnonzero((indeg == 0) & (outdeg > 0))
    if len(roots) != 2:
        raise ValueError(f"expected 2 roots, found {len(roots)}")
    mu, log_tau = int(roots.min()), int(roots.max())
    theta = np.flatnonzero(indeg == 2)
    y = np.array([int(np.flatnonzero(B[t])[0]) for t in theta])
    return {"mu": mu, "log_tau": log_tau, "theta": theta, "y": y}


def sample_hierarchical_distribution(n, adjacency_matrix, roles=None,
                                     mu_sd=MU_SD, tau_sd=TAU_SD, y_sd=Y_SD,
                                     log_pdf=False):
    """Sample the hierarchical normal model laid out by `adjacency_matrix`."""
    d = adjacency_matrix.shape[0]
    if roles is None:
        roles = recover_roles(adjacency_matrix)
    i_mu, i_tau = roles["mu"], roles["log_tau"]
    theta_idx, y_idx = np.asarray(roles["theta"]), np.asarray(roles["y"])
    J = len(theta_idx)

    mu = np.random.normal(0.0, mu_sd, n)
    log_tau = np.random.normal(0.0, tau_sd, n)
    tau = np.exp(log_tau)

    theta = mu[:, None] + tau[:, None] * np.random.normal(0.0, 1.0, (n, J))
    y = theta + np.random.normal(0.0, y_sd, (n, J))

    X = np.empty((n, d), dtype=np.float32)
    X[:, i_mu] = mu
    X[:, i_tau] = log_tau
    X[:, theta_idx] = theta
    X[:, y_idx] = y

    if not log_pdf:
        return X

    c = 0.5 * np.log(2 * np.pi)
    lp = -0.5 * (mu / mu_sd) ** 2 - np.log(mu_sd) - c
    lp += -0.5 * (log_tau / tau_sd) ** 2 - np.log(tau_sd) - c
    lp += np.sum(-0.5 * ((theta - mu[:, None]) / tau[:, None]) ** 2
                 - log_tau[:, None] - c, axis=1)
    lp += np.sum(-0.5 * ((y - theta) / y_sd) ** 2 - np.log(y_sd) - c, axis=1)
    return X, lp


def edge_families(B, roles):
    """Split the true edges into the three families, for per-family scoring.

    Returns a dict of {name: set of (i, j)} -- 'mu->theta' (weak coupling when
    marginalised), 'tau->theta' (strong coupling), 'theta->y'.
    """
    i_mu, i_tau = roles["mu"], roles["log_tau"]
    theta, y = np.asarray(roles["theta"]), np.asarray(roles["y"])
    return {
        "mu->theta": {(i_mu, int(t)) for t in theta},
        "tau->theta": {(i_tau, int(t)) for t in theta},
        "theta->y": {(int(t), int(yy)) for t, yy in zip(theta, y)},
    }