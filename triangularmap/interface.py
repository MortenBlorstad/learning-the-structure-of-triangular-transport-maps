"""The transport-map interface that SSTM implements.

A map T pushes the target distribution forward to a reference (standard normal by
default) and pulls samples back the other way. Given `pushforward`, `pullback`,
`log_jacobian` and `fit`, this base class supplies the density and sampling methods
by change of variables, so a new map only has to provide those four.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from typing import Optional


import numpy as np


class BaseTransportMap(ABC):
    """
    Base interface for (invertible) transport maps between:
      - target/data space X in R^K
      - reference space Z (often standard normal) in R^K
    """

    def __init__(self, *, random_state: Optional[int] = None):
        self.random_state = random_state
        self._rng = np.random.default_rng(random_state)
        self.is_fitted_: bool = False
        self.K: Optional[int] = None
        

    @abstractmethod
    def log_jacobian(self, x: np.ndarray, **kwargs) -> np.ndarray:
        """
        Return log-Jacobian of T at x: log |J| = dT/dx.
        """
        raise NotImplementedError

    @abstractmethod
    def fit(self, X: np.ndarray, y: None = None, **kwargs) -> "BaseTransportMap":
        """Fit map parameters from data X. Must set self.is_fitted_ and self.K."""
        raise NotImplementedError


    def check_is_fitted(self):
        """Raise if the map has not been fitted yet."""
        if not self.is_fitted_ or self.K is None:
            raise RuntimeError("Map is not fitted. Call fit(X) first.")

        
    @abstractmethod
    def pushforward(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """Map X -> Z using T(x). Shape: (n, K)."""
        raise NotImplementedError

    @abstractmethod
    def pullback(self, Z: np.ndarray, **kwargs) -> np.ndarray:
        """Map Z -> X using T^{-1}(z). Shape: (n, K)."""
        raise NotImplementedError
    
    def log_pdf_reference(self, z: np.ndarray, **kwargs) -> np.ndarray:
        """Log-density of reference distribution p_Z at z. Default: standard normal."""
        return -0.5 * np.sum(z**2, axis=1) - 0.5 * self.K * np.log(2 * np.pi)
    
    def log_pdf(self, x: np.ndarray, **kwargs) -> np.ndarray:
        r"""
        Returns log \hat{pi}(x) via change-of-variables:
            log p_X(x) = log p_Z(S(x)) + log|det J_S(x)|.
        """
        self.check_is_fitted()
        z = self.pushforward(x, **kwargs)
        log_pz = self.log_pdf_reference(z, **kwargs)
        log_det_jacobian = self.log_jacobian(x, **kwargs)
        return log_pz + log_det_jacobian

    def pdf(self, x: np.ndarray, **kwargs) -> np.ndarray:
        """Density at x, i.e. exp(log_pdf(x))."""
        return np.exp(self.log_pdf(x, **kwargs))

    def base_sample(self, n: int) -> np.ndarray:
        """Sample z ~ p_Z. Default: standard normal."""
        self.check_is_fitted()
        return self._rng.standard_normal(size=(n, self.K))
    

    

    def sample(self, n: int = 100, **kwargs) -> np.ndarray:
        """
        Sample x ~ p_X by drawing z~p_Z and mapping back via T^{-1}.
        """
        z = self.base_sample(n)
        return self.pullback(z, **kwargs)