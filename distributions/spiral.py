"""Spiral distribution in 2D, used as a smoke-test target.

`sample_spiral_distribution` is adapted from `Figures/Figure 05/create_figure_05.py`
(line 44) of https://github.com/MaxRamgraber/Triangular-Transport-Tutorial at commit
09d58f359823f02ff35315ba6e236ff414c43af7, which is licensed under the Apache License,
Version 2.0. A copy of that licence is at http://www.apache.org/licenses/LICENSE-2.0.

Changes from the original: the sampling body is extracted into a function taking a
`size` argument and returning the samples, instead of running at module level for one
figure. The mathematics is unchanged.
"""
import copy

import numpy as np
import scipy.stats


def sample_spiral_distribution(size):
    """Sample `size` points from the spiral distribution. Returns [size, 2]."""
    # First draw some rotation samples from a beta distribution, then scale 
    # them to the range between -pi and +2pi
    seeds = scipy.stats.beta.rvs(
        a       = 4,
        b       = 3,
        size    = size)*3*np.pi-np.pi
    
    # Create a local copy of the rotations
    seeds_orig = copy.copy(seeds)
    
    # Re-normalize the rotations, then scale them to the range between [-3,+3]
    vals    = (seeds+np.pi)/(3*np.pi)*6-3
    
    # Plot the rotation samples on a straight spiral
    X       = np.column_stack((
        np.cos(seeds)[:,np.newaxis],
        np.sin(seeds)[:,np.newaxis]))*((1+seeds+np.pi)/(3*np.pi)*5)[:,np.newaxis]

    # Offset each sample along the spiral's normal vector by scaled Gaussian 
    # noise
    X   += np.column_stack([
        np.cos(seeds_orig),
        np.sin(seeds_orig)])*(scipy.stats.norm.rvs(size=size)*scipy.stats.norm.pdf(vals))[:,np.newaxis]

    return X/2


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Sample a spiral distribution
    X = sample_spiral_distribution(size=1000)

    # Plot the sampled points
    plt.figure(figsize=(8, 8))
    plt.scatter(X[:, 0], X[:, 1], s=1)
    plt.axis("equal")
    plt.title("Spiral Distribution")
    plt.show()