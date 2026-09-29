from __future__ import annotations

from typing import Optional, List,Sequence
from pathlib import Path
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from triangularmap.interface import BaseTransportMap
from triangularmap.monotonic import MonotonicActivation
from triangularmap.triangular_ordering import TriangularOrdering

EBIC_C = 0.23


class _MonoMasks:
    """Monotone/free mask conventions shared by the map layers.

    Each layer splits its units into a monotone path and a free path. `U_cols`
    [1, q] marks the monotone output columns, a fraction `mono_frac` of them spread
    evenly. `input_mono_mask` marks the monotone inputs, which for the input layer
    depend on the current ordering and so are rebuilt per call. Rank 0 is all
    monotone in both.
    """

    @staticmethod
    def make_U_cols(output_size, all_monotonic, mono_frac):
        """U_cols [1, q] bool: exactly max(1, round(mono_frac*q)) mono columns,
        spread evenly (Bresenham rule). mono_frac=0.5 reproduces the historical
        even-column convention exactly (incl. q=1 -> mono)."""
        if all_monotonic:
            return torch.ones(1, output_size, dtype=torch.bool)
        n_mono = max(1, int(round(mono_frac * output_size)))
        return ((torch.arange(output_size) * n_mono) % output_size < n_mono).view(1, -1)

    def input_mono_mask(self, x, k_map):
        """V: monotone input mask, [.., K, p] float. None if all_monotonic."""
        if self.all_monotonic:
            return None
        if self.is_input_layer:
            # The variable introduced at rank k takes the monotone path,
            # so T_k stays monotone in x_{pi(k)} under any permutation.
            V = k_map.clone()
            V[:, 0, :] = 1.0                      # member 0: all mono
        else:
            # Hidden activations: mono inputs = the previous layer's mono columns.
            V = x.new_zeros(1, self.ensemble_size, self.input_size)
            V[:, 0, :] = 1.0                      # member 0: all mono
            if self.in_mono_cols is not None:
                V[:, 1:, self.in_mono_cols] = 1.0  # prev layer's U_cols
            else:
                V[:, 1:, ::2] = 1.0               # fallback: even = mono (valid at mono_frac=0.5 only)
        return V

    def output_mono_mask(self, x):
        """U: monotone output mask, [1, K, q] float, from U_cols + member-0 row."""
        U = self.U_cols.to(x.dtype).expand(self.ensemble_size, -1).clone()
        U = U.unsqueeze(0)                        # [1, K, q]
        U[:, 0, :] = 1.0                          # member 0: all mono
        return U


class LinearBatchEnsemble(_MonoMasks, nn.Module):
    """BatchEnsemble linear layer with a monotone / free path split.
    Mask conventions: see _MonoMasks."""

    def __init__(self, input_size, ensemble_size, output_size,
                 all_monotonic=False, disconnect=False, is_input_layer=False,
                 row_norm=True, mono_frac=0.5, in_mono_cols=None):
        super().__init__()
        self.input_size = input_size
        self.ensemble_size = ensemble_size
        self.output_size = output_size
        self.is_input_layer = is_input_layer
        self.disconnect = disconnect
        self.all_monotonic = all_monotonic
        self.row_norm = row_norm                  # False: W/R carry magnitude too (no direction/gain split)

        self.R = nn.Parameter(torch.empty(ensemble_size, input_size))
        self.S = nn.Parameter(torch.empty(ensemble_size, output_size))
        self.B = nn.Parameter(torch.randn(ensemble_size, output_size))
        self.W = nn.Parameter(torch.empty(output_size, input_size))

        # --- static column convention: mono output cols spread evenly (even cols at 0.5) ---
        self.register_buffer("U_cols", self.make_U_cols(output_size, all_monotonic, mono_frac),
                             persistent=False)
        # mono columns of the PREVIOUS layer: which of our inputs are mono activations
        # (used by hidden layers; None for the input layer / all-mono layers).
        self.register_buffer("in_mono_cols",
                             None if in_mono_cols is None else in_mono_cols.reshape(-1).clone(),
                             persistent=False)

        self.reset_parameters()

    def reset_parameters(self):
        """Small positive R/S (the rank-1 adapters start near a no-op) and Kaiming W."""
        nn.init.uniform_(self.R, 0.1, 0.11)
        nn.init.uniform_(self.S, 0.1, 0.11)
        nn.init.kaiming_uniform_(self.W, a=math.sqrt(5.0))
        nn.init.zeros_(self.B)

    # ------------------------------------------------------------------
    # Shared core (masks inherited from _MonoMasks)
    # ------------------------------------------------------------------

    def _adapters(self, k_map):
        """Per-rank adapters. k_map mixes rows only; columns keep meaning."""
        R = k_map @ self.R                        # [K, p]
        S = k_map @ self.S                        # [K, q]
        B = k_map @ self.B                        # [K, q]
        return R, S, B

    # ------------------------------------------------------------------
    # Forwards
    # ------------------------------------------------------------------

    def forward(self, x, k_map):
        """x: [batch, K, p], k_map: [K, p] -> [batch, K, q]"""
        W_pos = F.softplus(self.W)
        if self.row_norm:                         
            W_pos_sq = W_pos.square()             
            W_sq = self.W.square()                
                                                  
        R, S, B = self._adapters(k_map)
        R_pos = F.softplus(R)
        if self.row_norm:
            R_pos_sq = R_pos.square()
        S_pos = F.softplus(S)

        if self.all_monotonic:
            z = (x * R_pos) @ W_pos.mT * S_pos
            if self.row_norm:
                norm = torch.sqrt((W_pos_sq @ R_pos_sq.mT).mT + 1e-8)
            else:
                norm = 1.0
            return z / norm + B

        V = self.input_mono_mask(x, k_map)
        U = self.output_mono_mask(x)
        one_minus_V = 1 - V
        one_minus_U = 1 - U

        R_mono = R_pos * V
        R_free = R * one_minus_V
        S_mono = S_pos * U

        z_mono = (x * R_mono) @ W_pos.mT
        z_free = (x * R_free) @ self.W.mT
        z = z_mono * S_mono + z_free * S

        if self.row_norm:
            R_mono_sq = R_pos_sq * V
            R_free_sq = R.square() * one_minus_V

            norm_mono = (W_pos_sq @ R_mono_sq.mT + W_sq @ R_free_sq.mT).mT
            norm_free = (W_sq @ R_free_sq.mT).mT
            norm = torch.sqrt(U * norm_mono + one_minus_U * norm_free + 1e-8)
        else:
            norm = 1.0

        return z / norm + B

    def forward_with_jac(self, x, k_map, jac):
        """Forward + diagonal-Jacobian propagation + l1 diagnostic."""
        W_pos = F.softplus(self.W)
        if self.row_norm:                         
            W_pos_sq = W_pos.square()             
            W_sq = self.W.square()                
                                                  
        R, S, B = self._adapters(k_map)
        R_pos = F.softplus(R)
        if self.row_norm:
            R_pos_sq = R_pos.square()
        S_pos = F.softplus(S)

        if self.all_monotonic:
            z = (x * R_pos) @ W_pos.mT * S_pos
            if self.row_norm:
                norm = torch.sqrt((W_pos_sq @ R_pos_sq.mT).mT + 1e-8)
            else:
                norm = 1.0
            z_out = z / norm + B
            jac_out = (jac * R_pos) @ W_pos.mT * S_pos / norm
            return z_out, jac_out

        V = self.input_mono_mask(x, k_map)
        U = self.output_mono_mask(x)
        one_minus_V = 1 - V
        one_minus_U = 1 - U

        R_mono = R_pos * V
        R_free = R * one_minus_V
        S_mono = S_pos * U

        z_mono = (x * R_mono) @ W_pos.mT
        z_free = (x * R_free) @ self.W.mT
        z = z_mono * S_mono + z_free * S

        if self.row_norm:
            R_mono_sq = R_pos_sq * V
            R_free_sq = R.square() * one_minus_V

            norm_mono = (W_pos_sq @ R_mono_sq.mT + W_sq @ R_free_sq.mT).mT
            norm_free = (W_sq @ R_free_sq.mT).mT
            norm = torch.sqrt(U * norm_mono + one_minus_U * norm_free + 1e-8)
        else:
            norm = 1.0

        z_out = z / norm + B

        jac_mono = (jac * R_mono) @ W_pos.mT * S_mono
        jac_free = (jac * R_free) @ self.W.mT * S
        jac_out = (jac_mono + jac_free) / norm

        return z_out, jac_out

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    @torch.no_grad()
    def s_gain_stats(self):
        """(s_pos_mean_on_mono_cols, s_free_abs_mean). Parameter-space,
        exact snapshot; uses U_cols so it cannot drift from the forward."""
        sp = F.softplus(self.S)
        cols = self.U_cols.to(sp.dtype)
        s_pos = (sp * cols).sum() / (cols.sum() * self.S.shape[0])
        if self.all_monotonic:
            s_free = self.S.new_zeros(())
        else:
            s_free = self.S.abs().mean()
        return s_pos, s_free


class MonotoneLinear(_MonoMasks, nn.Module):
    """Per-component linear layer, without parameter sharing.

    Each of the K components has its own weight matrix and bias, stacked as
    W [K, q, p] and B [K, q] and applied with one batched matmul, so this is K
    independent networks evaluated together. Same interface and mask conventions as
    `LinearBatchEnsemble`, which shares one W instead. Per-component parameters are
    routed by k_map, so they follow the learned permutation.

    With row_norm=False the layer is a standard linear map, z = W x + b. With
    row_norm=True the rows of W are normalized to direction only and a per-row gain
    S [K, q] carries the magnitude; the gain is needed because a normalized W alone
    caps the diagonal derivative at about 1 and so excludes any locally expanding
    map.

    Monotone inputs pass through softplus(W) into monotone output columns only.
    Free columns receive only free-input contributions, which carry no diagonal
    Jacobian, so the monotone chain stays exact.
    """

    def __init__(self, input_size, ensemble_size, output_size,
                 all_monotonic=False, disconnect=False, is_input_layer=False,
                 row_norm=False, mono_frac=0.5, in_mono_cols=None):
        super().__init__()
        self.input_size = input_size
        self.ensemble_size = ensemble_size
        self.output_size = output_size
        self.is_input_layer = is_input_layer
        self.disconnect = disconnect
        self.all_monotonic = all_monotonic
        self.row_norm = row_norm

        self.W = nn.Parameter(torch.empty(ensemble_size, output_size, input_size))
        self.B = nn.Parameter(torch.empty(ensemble_size, output_size))
        if row_norm:                              # per-row gain: S's role, S's regularization
            self.S = nn.Parameter(torch.empty(ensemble_size, output_size))

        # --- static column convention: mono output cols spread evenly (same rule as BE) ---
        self.register_buffer("U_cols", self.make_U_cols(output_size, all_monotonic, mono_frac),
                             persistent=False)
        self.register_buffer("in_mono_cols",
                             None if in_mono_cols is None else in_mono_cols.reshape(-1).clone(),
                             persistent=False)

        self.reset_parameters()

    def reset_parameters(self):
        """Kaiming per member, so each component matches one shared layer's fan-in."""
        for w in self.W:                          # per-member kaiming: same fan as one shared layer
            nn.init.kaiming_uniform_(w, a=math.sqrt(5.0))
        nn.init.zeros_(self.B)
        if self.row_norm:
            nn.init.uniform_(self.S, 0.1, 0.11)   # same init as BE's S

    def _routed(self, k_map):
        """Per-rank weights/bias. k_map mixes members only; q/p keep meaning."""
        W = (k_map @ self.W.flatten(1)).unflatten(-1, (self.output_size, self.input_size))
        B = k_map @ self.B
        return W, B

    @staticmethod
    def _mm(W, a):
        """[.., K, q, p] x [.., K, p] -> [.., K, q] (leading dims broadcast)."""
        return (W @ a.unsqueeze(-1)).squeeze(-1)

    def _forward_core(self, x, k_map, jac):
        """Single core for forward and forward_with_jac so the two cannot drift.
        row_norm=True mirrors LinearBatchEnsemble's norm/gain algebra with R == 1."""
        W, B = self._routed(k_map)
        W_pos = F.softplus(W)

        if not self.row_norm:
            if self.all_monotonic:
                z = self._mm(W_pos, x) + B
                jac_out = self._mm(W_pos, jac) if jac is not None else None
                return z, jac_out
            V = self.input_mono_mask(x, k_map)
            U = self.output_mono_mask(x)
            one_minus_V = 1 - V
            z = self._mm(W_pos, x * V) * U + self._mm(W, x * one_minus_V) + B
            if jac is None:
                return z, None
            jac_out = self._mm(W_pos, jac * V) * U + self._mm(W, jac * one_minus_V)
            return z, jac_out

        S = k_map @ self.S                        # routed gain, [.., K, q]
        S_pos = F.softplus(S)

        if self.all_monotonic:
            norm = torch.sqrt(W_pos.square().sum(-1) + 1e-8)
            z = self._mm(W_pos, x) * S_pos / norm + B
            jac_out = self._mm(W_pos, jac) * S_pos / norm if jac is not None else None
            return z, jac_out

        V = self.input_mono_mask(x, k_map)
        U = self.output_mono_mask(x)
        one_minus_V = 1 - V
        one_minus_U = 1 - U
        S_mono = S_pos * U

        W_pos_sq = W_pos.square()
        W_sq = W.square()
        norm_mono = self._mm(W_pos_sq, V) + self._mm(W_sq, one_minus_V)
        norm_free = self._mm(W_sq, one_minus_V)
        norm = torch.sqrt(U * norm_mono + one_minus_U * norm_free + 1e-8)

        z = (self._mm(W_pos, x * V) * S_mono + self._mm(W, x * one_minus_V) * S) / norm + B
        if jac is None:
            return z, None
        jac_out = (self._mm(W_pos, jac * V) * S_mono + self._mm(W, jac * one_minus_V) * S) / norm
        return z, jac_out

    def forward(self, x, k_map):
        """x: [batch, K, p], k_map: [.., K, K] -> [batch, K, q]"""
        z, _ = self._forward_core(x, k_map, jac=None)
        return z

    def forward_with_jac(self, x, k_map, jac):
        """Forward + diagonal-Jacobian propagation."""
        return self._forward_core(x, k_map, jac)

    @torch.no_grad()
    def s_gain_stats(self):
        """(s_pos_mean_on_mono_cols, s_free_abs_mean); same semantics as
        LinearBatchEnsemble's. Only called when S exists (row_norm=True)."""
        sp = F.softplus(self.S)
        cols = self.U_cols.to(sp.dtype)
        s_pos = (sp * cols).sum() / (cols.sum() * self.S.shape[0])
        if self.all_monotonic:
            s_free = self.S.new_zeros(())
        else:
            s_free = self.S.abs().mean()
        return s_pos, s_free


class AdamSReg(torch.optim.Adam):
    """Adam with decoupled l1/l2 penalties on the per-row gains S.

    The penalties are applied in the step rather than added to the loss, so Adam's
    second-moment estimate stays a clean curvature estimate of the data term. On
    monotone columns they act in gain space (through softplus), on free columns on
    the raw value.
    """

    def __init__(self, params, l1=0.0, l2=0.0, **kw):
        super().__init__(params, **kw)
        for group in self.param_groups:
            group.setdefault('l1', l1)
            group.setdefault('l2', l2)
            group.setdefault('mono_mask', None)

    @torch.no_grad()
    def step(self, closure=None):
        """Adam step, then the decoupled S penalties on the groups that carry a mono mask."""
        for group in self.param_groups:
            l1, l2 = group['l1'], group['l2']
            mono_mask = group['mono_mask']
            if mono_mask is None or (l1 == 0.0 and l2 == 0.0):       # only the S groups carry a mask
                continue
            mono_cols = mono_mask.to(torch.float32)                  # 1 on monotone columns, else 0
            free_cols = (~mono_mask).to(torch.float32)               # 1 on free columns, else 0
            for S in group['params']:                                # each param here is a layer's S
                if S.grad is None:
                    continue
                sigmoid_S = torch.sigmoid(S)
                l1_grad = sigmoid_S * mono_cols + S.sign() * free_cols          # softplus-grad on mono (S_pos), abs-grad on free (|S|)
                l2_grad = F.softplus(S) * sigmoid_S * mono_cols + S * free_cols  # mono gain-space, free raw
                S.grad.add_(l1 * l1_grad + l2 * l2_grad)             # injected pre-step -> Adam preconditions it
        return super().step(closure)
##
## Triangular Map Implementation
##
class SSTM(BaseTransportMap, nn.Module):
    """
    Monotone triangular transport map with BatchEnsemble layers.

    The learned triangular ordering produces a *triangular mask* M in original coordinates:
        M[k,j] ~= 1 if feature j is available to stage k.

    The "introduced feature" selector is:
        Δ[k,:] = M[k,:] - M[k-1,:]    (with M[-1,:]=0)

    Shapes:
        D = input_size
        x: [batch_size,D] (external)
        x_rep: [batch_size,K,D] (internal BE replication)
        M: [D,D] (triangular mask)
    """

    def __init__(self, input_size: int, hidden: List[int], output_dim: int = 1,disconnect_first_n = 2, random_state: Optional[int] = None, fixed_order: Optional[Sequence[int]] = None,
                 gate_method='ste', perm_method='softsort',antithetic=False, fixed_sparsity=None,
                 parameter_sharing=True, row_norm=True, mono_frac=0.5):
        nn.Module.__init__(self)
        BaseTransportMap.__init__(self, random_state=random_state)
        torch.manual_seed(random_state if random_state is not None else 0)
       

        self.K = input_size
        
        self._config = dict(input_size=input_size, hidden=list(hidden), output_dim=output_dim,
                            disconnect_first_n=disconnect_first_n, random_state=random_state,
                            fixed_order=fixed_order, gate_method=gate_method, perm_method=perm_method,
                            parameter_sharing=parameter_sharing, row_norm=row_norm, mono_frac=mono_frac)
        self.maps = nn.ModuleList()

        
        self.activation = MonotonicActivation(s=[1/3,1/3, 1/3], activation='elu')
        self.fixed_order = fixed_order

        droprate_init = 0.33 

        self.ordering = TriangularOrdering(K=input_size, tau=0.5, n_iters=10, droprate_init=droprate_init, fixed_order=fixed_order,
                                            gate_method=gate_method, perm_method=perm_method, antithetic=antithetic, fixed_sparsity=fixed_sparsity)
        

        # parameter_sharing=True: BatchEnsemble (shared W + rank-1 adapters);
        # False: MonotoneLinear (each component its own W and bias, no adapters).
        # row_norm applies to both: BE off = W/R carry magnitude too (S stays);
        # MonotoneLinear on = norm + per-row gain S (the pure no-sharing counterpart
        # of BE), off = the plain baseline.
        layer_cls = LinearBatchEnsemble if parameter_sharing else MonotoneLinear
        layer_kwargs = {"row_norm": row_norm, "mono_frac": mono_frac}
        n_hidden = len(hidden)
        prev_U_cols = None                        # threads each layer's mono cols into the next
        for i in range(n_hidden+ 1):
            in_size = input_size if i == 0 else hidden[i - 1]
            out_size = hidden[i] if i < n_hidden else output_dim
            is_input_layer = (i == 0)
            disconnect = (i <= disconnect_first_n)
            all_monotonic = (not disconnect)

            layer = layer_cls(
                            input_size=in_size,
                        ensemble_size=input_size,
                        output_size=out_size,
                        all_monotonic=all_monotonic,
                        disconnect=disconnect,
                        is_input_layer=is_input_layer,
                        in_mono_cols=None if is_input_layer else prev_U_cols,
                        **layer_kwargs
                    )
            self.maps.append(layer)
            prev_U_cols = layer.U_cols
            
        self.device = (
            torch.device("cuda") if torch.cuda.is_available()
            else torch.device("mps")     if torch.backends.mps.is_available()
            else torch.device("cpu")
        )
        self.to(self.device)


    def forward(self, x, mask=None, k_map=None):
        """Apply the map: x [n, K] -> z [n, K]. Draws the ordering if not given."""
        if mask is None or k_map is None:
            mask, k_map,_,_  = self.ordering()
        if mask.dim() == 2:
            mask = mask.unsqueeze(0)
            k_map = k_map.unsqueeze(0)
        if x.dim() == 2:
            x = x.unsqueeze(1).expand(-1, self.K, -1)
        
        x = mask * x
        for i, layer in enumerate(self.maps):
            x = layer(x, k_map=k_map)
            if i < len(self.maps) - 1:
                x = self.activation(x)
        return x.squeeze(-1)

    def forward_with_jac(self, x, mask=None, k_map=None):
        """As `forward`, but also returns the per-component log-derivative.

        Returns (z, log_jac), both [n, K]. The Jacobian is diagonal because the map
        is triangular, so the log-determinant is log_jac.sum(dim=1).
        """
        if mask is None or k_map is None:
            mask, k_map,_,_  = self.ordering()
        if mask.dim() == 2:
            mask = mask.unsqueeze(0)
            k_map = k_map.unsqueeze(0)
        if x.dim() == 2:
            x = x.unsqueeze(1).expand(-1, self.K, -1)
        x = mask * x
        

        jac_diag = (mask * k_map).expand(x.shape[0], -1, -1).clone()

        
        log_scale_diag = torch.zeros(x.shape[0], self.K, device=x.device)

       
        for i, layer in enumerate(self.maps):
            is_hidden = (i < len(self.maps) - 1)
            jac_diag_max = jac_diag.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
            jac_diag = jac_diag / jac_diag_max
            log_scale_diag = log_scale_diag + jac_diag_max.squeeze(-1).log()

            x, jac_diag = layer.forward_with_jac(x, k_map=k_map, jac=jac_diag)

            if is_hidden:
                phi_prime = self.activation.derivative(x)
                x = self.activation(x)
                jac_diag = jac_diag * phi_prime

        z = x.squeeze(-1)
        diag_jac = jac_diag.squeeze(-1)
        log_jac = diag_jac.abs().clamp(min=1e-8).log() + log_scale_diag
        return z, log_jac

    def _get_monotonic_W(self,k_map: torch.Tensor) -> torch.Tensor:
        # [k, out_dim]
        emsemble_Ws = []
        for module in self.maps:
            if hasattr(module, "_get_monotonic_W"):
                wm = module._get_monotonic_W(k_map)  # (k, out_dim, K)
                emsemble_Ws.append(wm)
        return emsemble_Ws


    def log_jacobian(self, x: torch.Tensor, k_map) -> torch.Tensor:
        """Per-component log |dT_k/dx_k| for x [n, K]. Returns [n, K]."""
        
        mask, k_map,_,_ = self.ordering()  # [K, K]
        z, log_jac = self.forward_with_jac(x, mask=mask, k_map=k_map)
        return log_jac 
        
    
    @torch.no_grad()
    def log_pdf(self, x: np.ndarray) -> np.ndarray:
        """log p_X(x) via a single forward_with_jac pass. No exp/clip roundtrip."""
        self.check_is_fitted()
        self.eval()
        x_tensor = torch.tensor(x, dtype=torch.float32, device=self.device)
        z, log_jac = self.forward_with_jac(x_tensor)               # [batch_size, K]
        log_pz = self.log_pdf_reference(z.detach().cpu().numpy())  # [batch_size]
        log_diag_J = log_jac.sum(dim=1).detach().cpu().numpy()     # [batch_size]
        return log_pz + log_diag_J                                  # [batch_size]

    def pdf(self, x: np.ndarray) -> np.ndarray:
        """Density p_X(x) for x [n, K]. Returns [n]."""
        return np.exp(self.log_pdf(x))
    

    def triangular_transport_loss_fast(self, z, log_jac):
        """Transport objective: mean of ||z||^2 / 2 minus the log-determinant."""

        norm_term = 0.5 * (z ** 2).sum(dim=1)
        log_det = log_jac.sum(dim=1)
        return (norm_term - log_det).mean()

    def build_optimizer(self, lr_map, lr_ordering, lr_sparsity,
                    l2=0.0001, l1=0.0, optimizer="adam"):
        """Group the parameters by role and build the optimizer.

        Map weights, ordering logits and gate affinities get their own learning
        rates, and the per-row gains S carry the l1/l2 penalties.
        """
        
        B_params, no_decay, logits_params, gate_affinity = [], [], [], []

        for name, p in self.named_parameters():         
            if not p.requires_grad:
                continue
            if "logits" in name:
                logits_params.append(p)
            elif "gate_affinity" in name:
                gate_affinity.append(p)
            elif name.endswith('.B'):
                B_params.append(p)
            elif not name.endswith('.S'):                # W, R: row-normalized, no penalty
                no_decay.append(p)
            # .S handled per layer below

        param_groups = [                                  # fixed head: scheduler idx 0..3
            {'params': B_params,      'lr': lr_map,      'weight_decay': l2},   # B: native l2
            {'params': no_decay,      'lr': lr_map,      'weight_decay': 0.0},
            {'params': logits_params, 'lr': lr_ordering, 'weight_decay': 0.0},
            {'params': gate_affinity, 'lr': lr_sparsity, 'weight_decay': 0.0},
        ]
        for layer in self.maps:                           # S: manual penalties, native decay OFF
            if not hasattr(layer, "S"):                   # MonotoneLinear: no S -> no penalty group
                continue                                  
            # AdamSReg applies l1 (sigmoid on mono / sign on free) and l2 (gain-space on mono,
            # raw on free) using mono_mask, in the optimizer step (decoupled from Adam's exp_avg_sq).
            param_groups.append({
                'params': [layer.S], 'lr': lr_map,
                'weight_decay': 0.0,                      # critical: no native decay on S
                'l1': l1, 'l2': l2,
                'mono_mask': layer.U_cols,
            })

        
        if optimizer.lower() == "adam":
            return AdamSReg(param_groups)                      # Adam + decoupled S penalties
        # plain torch.optim.Adam removed: it silently ignored the S penalty keys, leaving S
        # unregularized. For a no-S-penalty run use optimizer="adam" with l1=0, l2=0.
        raise ValueError(f"unknown optimizer {optimizer!r} (expected: adam)")


    def _map_refine_config(self, epochs, map_refine_tail):
        fully_fixed = self.ordering.use_fixed_order and self.ordering.use_fixed_sparsity
        apply_tail = fully_fixed or map_refine_tail
        refine_epochs = int(round(epochs * 0.25)) if apply_tail else 0   # last-25% map-only tail (500 -> 125)
        total_epochs = refine_epochs if (fully_fixed and refine_epochs) else epochs        # fixed-structure -> 125
        freeze_at = (epochs - refine_epochs) if (refine_epochs and not fully_fixed) else None  # learned: freeze at 375
        in_refine = lambda e: refine_epochs > 0 and (fully_fixed or (freeze_at is not None and e >= freeze_at))
        frozen = False
        return in_refine, frozen, freeze_at, total_epochs

    def fit(self, X: np.ndarray, epochs=10, batch_size=64, lr_map=1e-2, lr_ordering=1e-2,lr_sparsity=1e-2, l2=0.0, l1=None,
            sparsity_strength=0.0, verbose=False, logger=None, checkpoint_dir=None, n_checkpoints=0, X_val=None, optimizer="adam",
            map_refine_tail=False, n_samples=10, warmup=50, lr_decay=None,
            compile=False) -> "SSTM":
        """Fit the map, the ordering and the sparsity pattern jointly.

        X: [n, K] training data. Returns self.

        :param epochs: Training epochs.
        :param batch_size: Minibatch size.
        :param lr_map: Learning rate for the map weights.
        :param lr_ordering: Learning rate for the ordering logits.
        :param lr_sparsity: Learning rate for the sparsity gates.
        :param l2: L2 strength on the per-row gains S and the biases B.
        :param l1: L1 strength on S. None (default) uses an EBIC-style prior,
            0.23 * ln(n) / (2n), which is the density-optimal dose and decays with n.
            Structure recovery generally wants a larger value, so pass one explicitly.
            l1=0.0 disables the penalty.
        :param sparsity_strength: Strength of the L0 gate penalty.
        :param warmup: Epochs of map-only training before the ordering and gates start
            learning. Scale with the epoch budget; the default suits a 500-epoch run.
        :param lr_decay: None keeps the ramp-and-hold schedules. "cosine" glides every
            learning rate to zero over the second half of training.
        :param n_samples: Monte Carlo draws of the ordering per input.
        :param optimizer: "adam" uses AdamSReg, which applies the S penalties in the step.
        :param compile: Wrap forward_with_jac in torch.compile. Faster, but it changes
            float rounding, so runs are not bit-comparable with uncompiled ones.
        :param X_val: Optional held-out data. Only recorded, never used for selection.
        :param verbose: Print training progress.
        """
        self.is_fitted_ = True
        self.train()
        self.x_min_ = torch.tensor(X.min(axis=0), dtype=torch.float32, device=self.device)
        self.x_max_ = torch.tensor(X.max(axis=0), dtype=torch.float32, device=self.device)
        X_tensor = torch.tensor(X, dtype=torch.float32, device=self.device)
        dataset = TensorDataset(X_tensor,)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        self.K = X.shape[1]
        n_training_samples = X.shape[0]
        ramp_l1 = (l1 is None)                            # ramp ONLY the EBIC default; explicit l1 is constant
        if l1 is None:
            l1 = EBIC_C * np.log(n_training_samples) / (2.0 * n_training_samples)
        self.l1_ = float(l1)                                  
        sparsity_strength = sparsity_strength* self.K 
        
        # l1/l2 on S are applied decoupled in the optimizer (free cols: raw S; mono cols: gain softplus(S)),
        # NOT in the loss -> keeps Adam's exp_avg_sq a clean curvature estimate of the data term.
        if compile:
            self.forward_with_jac = torch.compile(self.forward_with_jac)

        opt = self.build_optimizer(lr_map=lr_map, lr_ordering=lr_ordering, lr_sparsity=lr_sparsity,
                                   l2=l2, l1=l1, optimizer=optimizer)
        
       
        warmup = int(warmup)   # epochs of map-only training (ordering LR ~0, gates off, tau at max)
        peak = epochs * 0.5
        lock_frac = 0.25                                   # tau anneal: tau_min reached this frac from the end
        # l1 ramp peak = the epoch tau hits tau_min (= warmup + (1-lock_frac)*(epochs-warmup)); slower ramp
        # than epochs/2, so l1 reaches full only once the ordering is fully annealed/locked.
        l1_ramp_peak = warmup + (1.0 - lock_frac) * (epochs - warmup)   # ramp l1 to full by the tau_min epoch


        in_refine, frozen, freeze_at, total_epochs = self._map_refine_config(epochs, map_refine_tail)
        fully_fixed = self.ordering.use_fixed_order and self.ordering.use_fixed_sparsity  # oracle: constant l1

        _base_increase = lambda epoch: (
            0.5 if epoch < warmup
            else min(0.5 + 0.9 * (epoch - warmup) / (epochs - warmup + 1e-8), 1.0)
        )
        increase_fn = lambda epoch: 1.0 if in_refine(epoch) else _base_increase(epoch)


        # low during warmup, then increase to 1.0 at peak, then HOLD at 1.0
        increase_O_fn = lambda epoch: (
            0.01 if epoch <= warmup
            else 0.01 + 0.999 * (epoch - warmup) / (peak - warmup + 1e-8) if epoch <= peak
            else 1.0
            )
        
        
        # sparsity LR: off during warmup, ramp 0.002 -> lr_sparsity up to `peak`, then HOLD at lr_sparsity
        # (ramp-and-hold: gets full LR in the second half so the gates can commit).
        _spars_lo = 0.002 / max(lr_sparsity, 1e-8)
        sparsity_fn = lambda epoch: (
            0.0 if epoch < warmup
            else _spars_lo + (1.0 - _spars_lo) * (epoch - warmup) / (peak - warmup + 1e-8) if epoch <= peak
            else 1.0
        )
        

        # lr_decay="cosine": all group multipliers glide to 0 over the second half of training.
        if lr_decay == "cosine":
            _decay_start = 0.5 * total_epochs
            _decay = lambda e: (1.0 if e <= _decay_start else
                                0.5 * (1.0 + math.cos(math.pi * (e - _decay_start) / max(1.0, total_epochs - 1 - _decay_start))))
        elif lr_decay is None:
            _decay = lambda e: 1.0
        else:
            raise ValueError(f"unknown lr_decay {lr_decay!r} (expected: None, 'cosine')")
        _with_decay = lambda fn: (lambda e: fn(e) * _decay(e))

        scheduler = torch.optim.lr_scheduler.LambdaLR(
            opt,
            lr_lambda=[_with_decay(f) for f in ([increase_fn, increase_fn, increase_O_fn, sparsity_fn]
                                                + [increase_fn] * (len(opt.param_groups) - 4))],
        )
        
        
        def training_schedule(epoch, total_epochs,
                      tau_max=1.5, tau_min=0.1, lock_frac=0.1):
            """tau anneal: tau_max until `warmup`, then linear to tau_min, reached lock_frac from the end."""
            frac = (epoch - warmup) / max(total_epochs - warmup, 1)   # 0 -> 1 after warmup
            
            if frac > (1.0 - lock_frac):
                return tau_min

            t = frac / (1.0 - lock_frac)
            tau = tau_max + (tau_min - tau_max) * t
            if frac < 0.0:
                return tau_max
         
            return tau 

        

        
        tau_max = 0.3*np.sqrt(self.K)
        tau_min = 0.1
        n_samples = n_samples if self.fixed_order is None else 1   # MC draws of the ordering per input (fixed order -> 1)


        # A supplied mask is not learned, so it applies from epoch 0. Learned gates stay
        # off until warmup ends (see the epoch loop below).
        self.ordering.use_sparsity = self.ordering.use_fixed_sparsity

        if logger is not None:
            logger.start(self, X, warmup=warmup, X_val=X_val)

        ckpt_epochs = set()
        if checkpoint_dir is not None and n_checkpoints > 0:
            ckpt_epochs = set(np.linspace(0, total_epochs - 1, n_checkpoints).round().astype(int).tolist())

        prev_we = None                          # cached W_eff directions for the per-log drift term
        
        self.val_history_, self.best_val_nll_, self.best_epoch_ = [], float("inf"), None
        self.best_state_ = self.last_state_ = None       # CPU state_dict copies (X_val given)
        for epoch in range(total_epochs):
            epoch_losses    = torch.zeros((), device=self.device)
            sparsity_losses = torch.zeros((), device=self.device)
            ep_l0 = ep_ent = 0.0   # separated loss terms for the logger
            ep_grad_total = ep_grad_logits = ep_grad_gate = ep_grad_net = 0.0  # pre-clip grad norms
            count = 0
            tau_e =  training_schedule(epoch, epochs, tau_max=tau_max, tau_min=tau_min, lock_frac=lock_frac)
            self.ordering.tau = tau_e

           
            if fully_fixed or not ramp_l1:
                l1_e = l1
            else:
                l1_e = l1 * min(epoch / max(1.0, l1_ramp_peak), 1.0)
            for g in opt.param_groups:
                if g.get('mono_mask') is not None:               # the S penalty groups only
                    g['l1'] = l1_e

            if epoch >= warmup:
                self.ordering.use_sparsity = True

           
            if freeze_at is not None and epoch == freeze_at:
                self.ordering.requires_grad_(False)
                frozen = True
            if frozen:
                self.ordering.eval()

            ns = 1 if (frozen or self.fixed_order is not None) else n_samples  # fixed order -> 1 sample

            for batch in loader:
                xb = batch[0].detach()
                b = xb.shape[0]
                xb = xb.repeat_interleave(ns, dim=0)  # [batch_size, K] to [batch_size*ns, K] for multiple samples per input

                mask, k_map, l0_penalty, entropy  = self.ordering(b*ns)  # [ns, K, K]
                sparsity_loss = l0_penalty * sparsity_strength
                

                z, log_jac = self.forward_with_jac(xb, mask=mask, k_map=k_map)
                
                tri_loss = self.triangular_transport_loss_fast(z, log_jac)
                loss = tri_loss + sparsity_loss   # l1 handled decoupled in the optimizer (clean exp_avg_sq)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                log_batch = logger is not None and logger.should_log(epoch)
                if log_batch:            # read structural grads pre-clip (clip rescales in place)
                    gln = self.ordering.logits.grad.norm().item()        if self.ordering.logits.grad        is not None else 0.0
                    ggn = self.ordering.gate_affinity.grad.norm().item() if self.ordering.gate_affinity.grad is not None else 0.0
                total_norm = torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)  # pre-clip total
                self.ordering.zero_fixed_grads()
                opt.step()

                epoch_losses    += tri_loss.detach()
                sparsity_losses += sparsity_loss.detach()
                if log_batch:
                    ep_l0 += float(l0_penalty.detach()); ep_ent += float(entropy)
                    tn = float(total_norm)
                    ep_grad_total += tn; ep_grad_logits += gln; ep_grad_gate += ggn
                    ep_grad_net += max(0.0, tn * tn - gln * gln - ggn * ggn) ** 0.5
                count += 1

            
        

            if ((epoch + 1) % 10 == 0 or epoch == 0) and verbose:
                lr_net = scheduler.get_last_lr()[0]
                lr_logits = scheduler.get_last_lr()[2]

                ho_ = self._hard_order().cpu().numpy()
                avg_tri      = (epoch_losses    / count).item()
                avg_sparsity = (sparsity_losses / count).item()


                print(f"Epoch {epoch+1}: " 
                      f"Loss = {avg_tri:.4f}, "
                      f"Sparsity Loss = {avg_sparsity:.4f}, "
                      f"tau={self.ordering.tau:.3f}, "
                      f"sparsity_strength={sparsity_strength:.6f}, "
                      f"LR_map={lr_net:.6f}, "
                      f"LR_logits={lr_logits:.6f}, "
                      f"Entropy={float(entropy):.4f}, "
                      f"Order={ho_}, "
                      )
            if logger is not None and logger.should_log(epoch):
                lrs = scheduler.get_last_lr()
                s_pos_dist, s_free_abs = self.s_gain_stats()
                w_stats, prev_we = self.w_eff_stats(prev=prev_we)
                ent_stats = self.ordering.structure_entropy()  
                logger.log_epoch(
                    epoch,
                    loss_terms=dict(nll=float(epoch_losses / count), l0=ep_l0 / count,
                                    entropy=ep_ent / count,
                                    h_gates=ent_stats["h_gates"], h_perm_hard=ent_stats["h_perm_hard"],
                                    grad_total=ep_grad_total / count, grad_logits=ep_grad_logits / count,
                                    grad_gate=ep_grad_gate / count, grad_net=ep_grad_net / count,
                                    s_pos_dist=s_pos_dist, s_free_abs=s_free_abs, **w_stats),
                    sched=dict(tau=self.ordering.tau, sparsity_strength=sparsity_strength,
                               l2=l2, l1=l1_e, lr_net=lrs[0], lr_logits=lrs[2],
                               lr_gate=lrs[3], use_sparsity=bool(self.ordering.use_sparsity),
                               warmup=warmup),
                )
            if epoch in ckpt_epochs:
                self.save(Path(checkpoint_dir) / f"epoch{epoch:04d}.pt", epoch=epoch)
            if X_val is not None:
                self._record_val(X_val, epoch)
            scheduler.step()
        if X_val is not None:                    # final weights stay the LAST epoch (no selection);
            self.last_state_ = self._cpu_state()  # best_state_ is kept so callers can evaluate it too
        return self

    def _cpu_state(self) -> dict:
        return {k: v.detach().to("cpu", copy=True) for k, v in self.state_dict().items()}

    def _record_val(self, X_val, epoch: int, batch: int = 8192) -> None:
        """Append the full-X_val NLL for this epoch to val_history_ (report-only). Eval-mode
        forward under no_grad: draws no random numbers and touches no optimizer state, so the
        training trajectory is unchanged; model and ordering modes are restored exactly."""
        was_training, ord_training = self.training, self.ordering.training
        try:
            lps = [self.log_pdf(X_val[i:i + batch]) for i in range(0, len(X_val), batch)]
        finally:
            self.train(was_training)
            self.ordering.train(ord_training)
        val_nll = float(-np.mean(np.concatenate(lps)))
        self.val_history_.append(val_nll)
        if val_nll < self.best_val_nll_:
            self.best_val_nll_, self.best_epoch_ = val_nll, epoch + 1
            self.best_state_ = self._cpu_state()


    @torch.no_grad()
    def s_gain_stats(self):
        """Aggregate per-layer S diagnostics. s_pos_dist: mean softplus(S) on mono
        output cols minus ln2 (the l2 fixed point). s_free_abs: mean |S|.
        Column masks come from each layer's U_cols -- same source the forward
        and optimizer use. Layers without S (MonotoneLinear) have no gains to
        report; an all-MonotoneLinear model returns (0.0, 0.0)."""
        with_S = [m for m in self.maps if hasattr(m, "S")]
        if not with_S:
            return 0.0, 0.0
        sp, sf = zip(*(m.s_gain_stats() for m in with_S))
        s_pos_dist = torch.stack(sp).mean().item() - math.log(2.0)
        free = [v for v, m in zip(sf, with_S) if not m.all_monotonic]
        s_free_abs = torch.stack(free).mean().item() if free else 0.0
        return s_pos_dist, s_free_abs


    @torch.no_grad()
    def w_eff_stats(self, prev=None):
        """Per-layer effective-weight *direction* stats (parameter-only, exact snapshot).
        Magnitude of W_eff is S (row-normalised), already logged by s_gain_stats, so we
        normalise and track direction only. member_cos: mean off-diagonal cosine between the
        K members' mono effective-weight directions (low = the K conditionals are genuinely
        different functions). drift: 1 - cosine vs the cached prev snapshot (late drift = still
        moving / memorising; ~0 = frozen). Returns (stats, we_list) so fit can cache we_list."""
        stats, we_list = {}, []
        for i, m in enumerate(self.maps):
            if hasattr(m, "R"):                           # BatchEnsemble: S diag(W) R
                Rp, Sp = F.softplus(m.R), F.softplus(m.S)
                We = torch.einsum('kq,qp,kp->kqp', Sp, F.softplus(m.W), Rp).flatten(1)   # [K, q*p]
            elif hasattr(m, "S"):                         # MonotoneLinear + row-norm gain
                We = (F.softplus(m.S).unsqueeze(-1) * F.softplus(m.W)).flatten(1)        # [K, q*p]
            else:                                         # MonotoneLinear plain: W IS per-member
                We = F.softplus(m.W).flatten(1)           # [K, q*p]
            We = F.normalize(We, dim=1)                                              # direction only
            we_list.append(We)
            C = We @ We.T                                                            # [K, K] cosine sim
            off = C[~torch.eye(m.ensemble_size, dtype=bool, device=C.device)]
            stats[f"l{i}_member_cos"] = off.mean().item()
            if prev is not None and i < len(prev):
                stats[f"l{i}_drift"] = 1.0 - (We * prev[i]).sum(1).mean().item()
        return stats, we_list


    def save(self, path, epoch=None):
        """Checkpoint the model: init config + state_dict + fit-time buffers.
        Reload with SSTM.load(path). (Model only; no scaler.)"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "config": self._config,
            "state_dict": self.state_dict(),
            "x_min": self.x_min_.detach().cpu() if hasattr(self, "x_min_") else None,
            "x_max": self.x_max_.detach().cpu() if hasattr(self, "x_max_") else None,
            "K": self.K,
            "is_fitted_": self.is_fitted_,
            "epoch": epoch,
        }, path)
        return str(path)

    @classmethod
    def load(cls, path, map_location="cpu") -> "SSTM":
        """Reconstruct a SSTM saved by save() and set it to eval()."""
        b = torch.load(path, map_location=map_location, weights_only=False)
        model = cls(**b["config"])
        model.load_state_dict(b["state_dict"])
        if b.get("x_min") is not None:
            model.x_min_ = b["x_min"].to(model.device)
            model.x_max_ = b["x_max"].to(model.device)
        model.K = b["K"]
        model.is_fitted_ = b["is_fitted_"]
        model.eval()
        return model

    @torch.no_grad()
    def _hard_order(self) -> torch.Tensor:
        """Learned ordering as a tensor of feature indices, rank 0 first. Shape [K]."""
        return self.ordering.hard_order()

    @torch.no_grad()
    def hard_order(self) -> np.ndarray:
        """The variable ordering the map learned, as feature indices with rank 0 first.

        ``hard_order()[r]`` is the feature introduced at rank r, so component r is
        monotone in that feature and sees only the features at earlier ranks.
        """
        return self._hard_order().cpu().numpy()

    @torch.no_grad()
    def get_adjacency_matrix(self) -> np.ndarray:
        """Binary adjacency ``B[i, j] in {0, 1}`` with ``B[i, j] = 1`` meaning
        ``i -> j`` (parent = row), matching the NOTEARS convention so the same
        scorer (`count_accuracy`) applies to both methods.
        """
        self.check_is_fitted()
        was_training = self.training
        self.eval()
        try:
            mask, k_map, _, _ = self.ordering()
            mask = mask.squeeze(0).detach().cpu().numpy()
            k_map = k_map.squeeze(0).detach().cpu().numpy()
            adj_rank = mask - k_map               # parents available at each rank
            adj_mat = (k_map.T @ adj_rank).T      # [i, j] = 1 iff i -> j
        finally:
            self.train(was_training)
        return (adj_mat != 0).astype(int)

    @torch.no_grad()
    def pushforward(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """Map X -> Z using T(x). Shape: (n, K)."""
        self.check_is_fitted()
        self.eval()
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        Z_tensor = self.forward(X_tensor).squeeze(-1)  # [n, K]
        Z = Z_tensor.cpu().numpy()
        return Z

    @torch.no_grad()
    def condition(self, n:int, x1k: np.array| float, random_state: int = None) -> np.ndarray:
        """
        Condition on x1k=a by sampling in z space and pulling back. Returns samples in original x space.
        n: number of samples to generate
        x1k: values to condition on for the first k features (shape [k], where k is the number of features to condition on). These should be in original x order.
        """

        rng = np.random.default_rng(random_state)
        if isinstance(x1k, (float, int)):
            x1k = np.array([x1k], dtype=np.float32)
        k = len(x1k)
        true_order = self._hard_order()  # [K]
        js = true_order[:k].tolist() # learned order indices
        # 1) map fixed x1=a to its latent coordinate z1*
        x0 = np.full((n, k), x1k, dtype=np.float32)  # [n, k]
        x_input = np.zeros((n, self.K), dtype=np.float32)
    
        x_input[:, js] = x0  # set the known features
 
        z = self.pushforward(x_input)  # [n, K]
        
        z1_star = z[:, :k]  # [n, k] the corresponding z1* coordinates for the fixed x1=a

        # 2) sample z2 freely
        z2 = rng.standard_normal((n, self.K - k))  # [n, K-k] sample the remaining z coordinates freely

        Z = np.column_stack([z1_star, z2])  # [n, K]

        X = self.pullback(Z)  # [n, K]

        return X

    def confidence_ball(self, n = 100, ci=0.95):
        """# Maps ci-ball in reference space to target space via T^{-1}. Returns radius in original x space."""
        from scipy.stats import chi2
        K = self.K
        radius = np.sqrt(chi2.ppf(ci, df=K))

        u = self._rng.uniform(low=-1.0, high=1.0, size=(n, K))  # [n, K] in reference space
        u = u / np.linalg.norm(u, axis=1, keepdims=True) * radius  # scale to ci-ball in reference space
        x = self.pullback(u)  # map to original space

        return x

    
    def dependency_layers(self, mask):
        """Group ranks into layers that can be inverted simultaneously.

        mask: [K, K] learned structure. Row r is the component at rank r, and
        mask[r, f] = 1 means that component sees feature f. The diagonal entry is
        the feature introduced at rank r, which is the component's own variable.

        A rank depends on every earlier rank whose feature it sees, so ranks with
        no unresolved dependency can be solved together. Returns those groups in
        order, which is what `pullback` iterates over.
        """
        K = self.K
        if mask.dim() == 3:
            mask = mask.squeeze()
        
        order = self._hard_order()            
        order = order.to(torch.long)

        
        rank_of_feature = torch.empty(K, dtype=torch.long, device=mask.device)
        rank_of_feature[order] = torch.arange(K, device=mask.device)

        
        own_feature = order                        
        parents_rank = [[] for _ in range(K)]
        for r in range(K):
            seen = (mask[r] > 0).nonzero(as_tuple=True)[0]   
            for f in seen.tolist():
                if f == own_feature[r].item():
                    continue                       
                parents_rank[r].append(rank_of_feature[f].item())

        
        layer = [-1] * K
        
        for _ in range(K):
            for r in range(K):
                ps = parents_rank[r]
                layer[r] = 0 if not ps else max(layer[p] for p in ps) + 1
        n_layers = max(layer) + 1
        groups = [torch.tensor([r for r in range(K) if layer[r] == L], device=mask.device)
                for L in range(n_layers)]
        return groups
    
    @torch.no_grad()
    def pullback(self, Z: np.ndarray, **kwargs) -> np.ndarray:
        """Inverse map T^-1(z), from reference space back to data space.

        Z: [n, K] reference samples. Returns [n, K] in the original feature order.
        Each component is inverted by bisection. Ranks with no unresolved dependency
        are solved together, so the number of sequential rounds is the depth of the
        learned graph rather than K.
        """
        self.check_is_fitted()
        self.eval()

        z = torch.tensor(Z, dtype=torch.float32, device=self.device)
        batch_size, K = z.shape
        if K != self.K:
            raise ValueError(f"Z has dim {K}, expected {self.K}.")

        mask, k_map,_,_ = self.ordering()
        order = self._hard_order()
        inv_order = torch.empty_like(order)
        inv_order[order] = torch.arange(K, device=self.device)

        
        o, _,_,_  = self.ordering()
        groups = self.dependency_layers(o) 

        x_min = self.x_min_[order]
        x_max = self.x_max_[order]
        span = (x_max - x_min).clamp(min=1.0)
        u_lo0 = x_min - self.K * span
        u_hi0 = x_max + self.K * span

        u = torch.zeros((batch_size, K), device=self.device)
        u_try = torch.zeros((batch_size, K), device=self.device)
        max_expand, n_bisect = 50, 50

        
        for idx in groups:                          
            m = idx.numel()
            target = z[:, idx]                      # [batch_size, m]
            lo = u_lo0[idx].unsqueeze(0).expand(batch_size, m).clone()
            hi = u_hi0[idx].unsqueeze(0).expand(batch_size, m).clone()

            def f(vals):                            # vals: [batch_size, m] -> residual [batch_size, m]
                u_try.copy_(u)
                u_try[:, idx] = vals
                z_pred = self.forward(u_try[:, inv_order], mask=mask, k_map=k_map)  # [batch_size, K]
                return z_pred[:, idx] - target

            
            f_lo, f_hi = f(lo), f(hi)
            expand = 0
            while ((f_lo > 0).any() or (f_hi < 0).any()) and expand < max_expand:
                w = hi - lo
                lo = torch.where(f_lo > 0, lo - w, lo)
                hi = torch.where(f_hi < 0, hi + w, hi)
                f_lo, f_hi = f(lo), f(hi)
                expand += 1

            for _ in range(n_bisect):
                mid = 0.5 * (lo + hi)
                f_mid = f(mid)
                left = f_mid <= 0                    
                lo = torch.where(left, mid, lo)
                hi = torch.where(left, hi, mid)
                if (hi - lo).max() < 1e-8:
                    break
            
            u[:, idx] = 0.5 * (lo + hi)

        x = u[:, inv_order]
        return x.cpu().numpy()
