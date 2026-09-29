"""The activation the map components are built from.

A component stays monotone in its own variable by combining a positive weight path
with an activation whose derivative has a known sign. `MonotonicActivation` is that
activation, mixing convex, concave and bounded branches in fixed proportions so the
composition can bend either way while keeping the derivative positive. The layers that
use it are in `sstm.py`.
"""
from typing import List

import torch
import torch.nn as nn


class MonotonicActivation(nn.Module):
    """Activation mixing convex, concave and bounded branches in fixed proportions.

    `s` gives the three weights. A single branch would bound the achievable shapes:
    convex alone cannot bend down, bounded alone saturates. Mixing keeps the
    derivative positive while allowing both curvatures.
    """

    def __init__(self, s:List[float], activation:str='relu'):
        super().__init__()
        self.s = s
        if activation not in ['relu', 'elu', 'selu', 'leaky_relu']:
            raise ValueError("Unsupported activation function. Choose from 'relu', 'elu', 'selu', 'leaky_relu'.")
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'elu':
            self.activation = nn.ELU()
        elif activation == 'selu':
            self.activation = nn.SELU()
        elif activation == 'leaky_relu':
            self.activation = nn.LeakyReLU(negative_slope=0.1)
        
    def concave(self, z):
        """Concave branch: -activation(-z), non-decreasing with decreasing slope."""
        return -self.activation(-z)
    
    def convex(self, z):
        """Convex branch: activation(z), non-decreasing with increasing slope."""
        return self.activation(z)
    
    def bounded(self, z):
        """Bounded branch: convex below 0, concave above, joined at +-1 so the output saturates."""
        one = torch.tensor(1.0, device=z.device)
        return torch.where(
            z < 0,
            self.convex(z + 1) - self.convex(one),
            self.concave(z - 1) + self.convex(one)
        )

    
    def forward(self, z):
        """Split the last dim into convex / concave / bounded parts in the proportions `s`."""
        m = z.shape[-1]
        s = [int(m*frac) for frac in self.s]
        while sum(s) < m:
            for i in range(len(s)):
                if sum(s) < m:
                    s[i] += 1
                else:
                    break

        z0, z1, z2= torch.split(z, split_size_or_sections=s, dim=-1) # (batch_size, s[0]), (batch_size, s[1]), (batch_size, s[2])
        a0 = self.convex(z0)
        a1 = self.concave(z1)
        a2 = self.bounded(z2)
        return torch.cat([a0, a1, a2], dim=-1)
    
    def derivative(self, z):
        """Derivative of forward(z) w.r.t. z, evaluated at z (pre-activation)."""
        m = z.shape[-1]
        s = [int(m * frac) for frac in self.s]
        while sum(s) < m:
            for i in range(len(s)):
                if sum(s) < m:
                    s[i] += 1
                else:
                    break

        z0, z1, z2 = torch.split(z, split_size_or_sections=s, dim=-1)

        # d/dz convex(z) = d/dz activation(z)
        d0 = self._activation_derivative(z0)

        # d/dz concave(z) = d/dz [-activation(-z)] = activation'(-z)
        d1 = self._activation_derivative(-z1)

        # d/dz bounded(z):
        #   z < 0: convex(z+1) - const  => activation'(z+1)
        #   z >= 0: concave(z-1) + const => activation'(-(z-1)) = activation'(1-z)
        d2 = torch.where(
            z2 < 0,
            self._activation_derivative(z2 + 1),
            self._activation_derivative(1 - z2),
        )

        return torch.cat([d0, d1, d2], dim=-1)

    def _activation_derivative(self, z):
        if isinstance(self.activation, nn.ELU):
            alpha = self.activation.alpha
            z_safe = z.clamp(max=20.0)  # prevent exp overflow in dead branch
            return torch.where(z >= 0, torch.ones_like(z), alpha * torch.exp(z_safe))
        elif isinstance(self.activation, nn.ReLU):
            return (z >= 0).float()
        elif isinstance(self.activation, nn.LeakyReLU):
            slope = self.activation.negative_slope
            return torch.where(z >= 0, torch.ones_like(z), torch.full_like(z, slope))
        elif isinstance(self.activation, nn.SELU):
            alpha = 1.6732632423543772
            scale = 1.0507009873554805
            z_safe = z.clamp(max=20.0)
            return scale * torch.where(z >= 0, torch.ones_like(z), alpha * torch.exp(z_safe))
