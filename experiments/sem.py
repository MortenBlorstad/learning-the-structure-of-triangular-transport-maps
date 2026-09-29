"""SEM data generation: the NOTEARS-standard samplers plus `make_data`, which draws a
random DAG and the data it implies. Shared by the synthetic and ablation experiments so
both see exactly the same generative process."""
import numpy as np
from scipy.special import expit as sigmoid
import igraph as ig

from notears import utils
from distributions.neils_funnel import funnel_adjacency, sample_funnel_distribution

from distributions.hierarchical import (
    hierarchical_adjacency, sample_hierarchical_distribution)


def sample_linear_sem(W, n, noise_scale=None, log_pdf=False):
    """Linear Gaussian SEM matching NOTEARS simulate_linear_sem (gauss).
    W: [d,d], W[i,j]=i->j. Returns X or (X, log_p)."""
    X = utils.simulate_linear_sem(W, n, "gauss", noise_scale=noise_scale)  
    if not log_pdf:
        return X
    d = W.shape[0]
    s = np.ones(d) if noise_scale is None else (
        noise_scale*np.ones(d) if np.isscalar(noise_scale) else np.asarray(noise_scale))
    mu = X @ W
    r = X - mu
    log_p = (-0.5*(r/s)**2 - np.log(s) - 0.5*np.log(2*np.pi)).sum(axis=1)
    return X, log_p
    

def sample_nonlinear_sem(B, n, sem_type='mlp', noise_scale=1.0, log_pdf=False):
    """Matches NOTEARS simulate_nonlinear_sem (mlp) but freezes mechanisms
    so true log_p is computable. Uses np.random (seed via utils.set_random_seed)."""
    d = B.shape[0]
    scale_vec = noise_scale if not np.isscalar(noise_scale) else noise_scale * np.ones(d)
    G = ig.Graph.Adjacency(B.tolist())
    order = G.topological_sorting()
    print("Topological order:", order)
    assert len(order) == d

    # freeze mechanism per node, drawn EXACTLY as NOTEARS does
    mech = {}
    for j in order:
        parents = G.neighbors(j, mode=ig.IN)
        pa = len(parents)
        if pa == 0:
            mech[j] = (parents, None)
        elif sem_type == 'mlp':
            hidden = 100
            W1 = np.random.uniform(low=0.5, high=2.0, size=[pa, hidden])
            W1[np.random.rand(*W1.shape) < 0.5] *= -1
            W2 = np.random.uniform(low=0.5, high=2.0, size=hidden)
            W2[np.random.rand(hidden) < 0.5] *= -1
            mech[j] = (parents, (W1, W2))
        else:
            raise ValueError(sem_type)

    def f(j, X):
        parents, p = mech[j]
        if p is None:
            return np.zeros(X.shape[0])
        W1, W2 = p
        return sigmoid(X[:, parents] @ W1) @ W2

    X  = np.zeros([n, d])
    mu = np.zeros([n, d])
    for j in order:
        mu[:, j] = f(j, X)
        X[:, j]  = mu[:, j] + np.random.normal(scale=scale_vec[j], size=n)

    if not log_pdf:
        return X
    log_p = np.zeros(n)
    for j in range(d):
        r = X[:, j] - mu[:, j]
        s = scale_vec[j]
        log_p += -0.5*(r/s)**2 - np.log(s) - 0.5*np.log(2*np.pi)
    return X, log_p


def make_data(d:int, s0:int, n:int, sem:str, graph:str, seed:int):
    """Simulate data from a random DAG with d nodes and e*d edges.
    Returns (B, X) where B is the true binary adjacency and X is the data."""
    utils.set_random_seed(seed)
    if graph == 'funnel':
        B = funnel_adjacency(d)
        X, log_pdf = sample_funnel_distribution(n, B, log_pdf=True)
    elif graph == 'hierarch':
        B, roles = hierarchical_adjacency(d)
        X, log_pdf = sample_hierarchical_distribution(n, B, roles=roles, log_pdf=True)
    else:
        B = utils.simulate_dag(d, s0, graph)
        if sem == 'gauss':
            X,log_pdf = sample_linear_sem(B, n, log_pdf=True)
        elif sem == 'mlp':
            X,log_pdf = sample_nonlinear_sem(B, n, sem_type=sem,log_pdf=True)
        else:
            raise ValueError(sem)
    order = ig.Graph.Adjacency(B.tolist()).topological_sorting()
    return B, X, log_pdf, order
