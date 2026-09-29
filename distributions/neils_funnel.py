"""Neal's funnel distribution over a star DAG with an arbitrary root.

`funnel_adjacency(d)` builds the star DAG B (a random root -> all others);
`sample_funnel_distribution(n, B)` samples from it (optionally returning log_pdf).
Uses the global numpy RNG, so callers control the seed (np.random.seed /
notears.utils.set_random_seed).
"""
import numpy as np


def funnel_adjacency(d):
    """Neal's funnel DAG: parent_idx -> every other node (a star)."""
    parent_idx = np.random.randint(0, d)
    B = np.zeros((d, d), dtype=int)
    B[parent_idx, :] = 1
    B[parent_idx, parent_idx] = 0
    return B

def sample_funnel_distribution(n, adjacency_matrix, log_pdf=False):
    """
    Neal's funnel distribution with an arbitrary root variable: Xparent ~ N(0, 3^2), X_rest | Xparent ~ N(0, exp(Xparent/2)^2).
    The parent variable has a large variance, and the remaining coordinates have variances that depend on the parent coordinate, creating a "funnel" shape. This distribution is often used to test the ability of sampling algorithms to handle strong dependencies and varying scales.
    """
   
    d = adjacency_matrix.shape[0]
    parent_idx = int(np.argmax(adjacency_matrix.sum(axis=1))) 
    other_cols = [j for j in range(d) if j != parent_idx]


    v = np.random.normal(0, 3, n)
    sigma = np.exp(v / 2)
    X_rest = np.random.normal(0, 1, (n, d - 1)) * sigma[:, None]
    X = np.empty((n, d), dtype=np.float32)
    X[:, parent_idx] = v
    X[:, other_cols] = X_rest
    

    if not log_pdf:
        return X
    
    log_p = -0.5 * (v / 3.0)**2 - np.log(3.0) - 0.5 * np.log(2 * np.pi)
    log_p += np.sum(
        -0.5 * (X_rest / sigma[:, None])**2
        - (v[:, None] / 2.0)
        - 0.5 * np.log(2 * np.pi),
        axis=1,
    )
   
    return X, log_p


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    B = funnel_adjacency(5)
    X = sample_funnel_distribution(1000, B)
    plt.scatter(X[:, 0], X[:, 1], s=2, color="black")
    plt.title("Neal's funnel (first two dims)"); plt.tight_layout(); plt.show()
