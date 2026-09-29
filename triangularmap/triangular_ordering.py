
"""Learned variable ordering and within-row sparsity for a triangular map.

The ordering is a permutation matrix P, rows indexed by rank and columns by
feature. It is relaxed during training so gradients can reach it, by SoftSort or
by Gumbel-Sinkhorn, and made hard again with a straight-through estimator, so the
forward pass always sees a genuine permutation. Its cumulative sum is the
triangular mask, giving component k access to the first k variables.

Sparsity is a second, independent set of L0 gates that prune conditioning
variables inside that mask, leaving each component's own variable untouched.
"""

import math
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from scipy.optimize import linear_sum_assignment


def sample_gumbel(shape, device, eps=1e-20):
    """Standard Gumbel noise of the given shape."""
    U = torch.rand(shape, device=device)
    return -torch.log(-torch.log(U + eps) + eps)

def sinkhorn_log(logits, n_iters=20):
    """Sinkhorn normalization in log space: alternate row and column normalization
    until the matrix is approximately doubly stochastic."""
    for _ in range(n_iters):
        logits = logits - torch.logsumexp(logits, dim=2, keepdim=True)  # row norm
        logits = logits - torch.logsumexp(logits, dim=1, keepdim=True)  # col norm
    return logits

def gumbel_sinkhorn(logits, tau=1.0, n_iters=20, noise=False, n_samples=1):
    """Relaxed permutation from [1, K, K] rank-by-feature logits.

    Adds Gumbel noise when sampling, then Sinkhorn-normalizes at temperature tau.
    Returns [n_samples, K, K], approximately doubly stochastic and approaching a
    permutation as tau goes to zero.
    """
    if noise:
        K = logits.shape[1]
        shape = (n_samples, K, K)
        logits = logits + sample_gumbel(shape, logits.device)
    log_P = sinkhorn_log(logits / tau, n_iters=n_iters)
    return torch.exp(log_P)  # [n_samples, K, K] approx doubly-stochastic

def softsort(s, tau=1.0):
    """Relaxed permutation from K scores, by softmax over distances to the sorted
    scores. Cheaper than Gumbel-Sinkhorn: one sort instead of Sinkhorn iterations."""
    s_sorted = s.sort(dim=-1, descending=True).values
    pair = -(s_sorted.unsqueeze(-1) - s.unsqueeze(-2)).abs()
    return (pair / tau).softmax(dim=-1)





class TriangularOrdering(nn.Module):
    """Learns the variable ordering and the sparsity gates of a triangular map.

    The ordering is parameterized by K scores (SoftSort) or a K-by-K logit matrix
    (Gumbel-Sinkhorn), annealed by the temperature tau. The gates are parameterized
    by a feature-space affinity matrix, mapped into rank space by the current
    permutation. Either can be pinned with `fixed_order` or `fixed_sparsity`, which
    is how the oracle and two-stage variants are built.

    `forward` returns the triangular mask, the diagonal selector, the L0 penalty and
    the ordering entropy.
    """

    def __init__(self, K: int, tau: float = 1.0, n_iters: int = 10, temperature: float = 2./3., droprate_init = 0.33, fixed_order=None, gate_method: str = 'ste', perm_method: str = 'softsort',antithetic=False, fixed_sparsity=None):
        super().__init__()
        self.K = K
        self.tau = tau
        self.n_iters = n_iters
        self.temperature = temperature
        
        self.droprate_init = droprate_init

        self.perm_method = perm_method
        if perm_method == 'sinkhorn':
            # logits over assignments: rank slot (row) -> feature (col)
            self.logits = nn.Parameter(torch.empty(1, K, K))
        elif perm_method == 'softsort':
            self.logits = nn.Parameter(torch.empty(1, K))
        else:
            raise ValueError(perm_method)
        
        
        self.gate_affinity = nn.Parameter(torch.zeros(1, K, K)) 

        self.use_sparsity = False
        

        self.gate_method = gate_method
        self.use_fixed_order = fixed_order is not None
        self.use_fixed_sparsity = fixed_sparsity is not None
        self.antithetic = antithetic  # sample Gumbel noise in pairs (x, -x) to reduce variance of gradient wrt logits

        self.reset_parameters()
        if self.use_fixed_order:
            self.set_fixed_order(fixed_order)
        if self.use_fixed_sparsity:
            self.set_fixed_sparsity(fixed_sparsity)
        


    def reset_parameters(self):
        """Flat ordering logits (no preferred permutation) and gate affinities set so the
        initial keep probability equals 1 - droprate_init."""
        with torch.no_grad():
            nn.init.zeros_(self.logits)
            target = math.log(1 - self.droprate_init) - math.log(self.droprate_init)
            self.gate_affinity.data.fill_(target)

    

    def set_fixed_order(self, fixed_order: Sequence[int]):
        """fixed_order: Sequence e.g. [2, 0, 1] means rank 0 -> feature 2, rank 1 -> feature 0, rank 2 -> feature 1"""
        assert len(fixed_order) <= self.K
        assert len(set(fixed_order)) == len(fixed_order), "Duplicate features"
        assert all(0 <= f < self.K for f in fixed_order), "Feature out of range"
        
        fixed_mask = torch.zeros_like(self.logits, dtype=torch.bool)
        with torch.no_grad():
            if self.perm_method == 'sinkhorn':
                for stage, feat in enumerate(fixed_order):
                    self.logits[:, stage, :] = -10
                    self.logits[:, :, feat]  = -10
                    self.logits[:, stage, feat] = 10
                    fixed_mask[:, stage, :] = True
                    fixed_mask[:, :, feat] = True
            else:  # softsort: descending scores, fixed features on top in order
                for rank, feat in enumerate(fixed_order):
                    self.logits[0, feat] = float(self.K - rank) + 10.0  # margin above free (~0)
                    fixed_mask[0, feat] = True
            # features not in fixed_order keep init (~0) and stay learnable
        self.register_buffer('fixed_mask', fixed_mask)


    
    def set_fixed_sparsity(self, mask):
        """Fix the within-row (per-rank) parent sparsity instead of learning it.

        mask: [K, K] (or [1, K, K]) 0/1 in (rank, feature) layout — which earlier
        features (parents) each rank may see. Bypasses the learned L0 gates; the
        diagonal (introduced feature) is still forced on in forward. Used by the
        Oracle (true parent mask) and Two-stage (NOTEARS mask) variants.
        """
        m = torch.as_tensor(mask, dtype=torch.float32)
        if m.dim() == 2:
            m = m.unsqueeze(0)
        assert m.shape == (1, self.K, self.K), f"expected [1,{self.K},{self.K}], got {tuple(m.shape)}"
        self.register_buffer('fixed_sparsity', m)
        self.use_fixed_sparsity = True
        self.gate_affinity.requires_grad_(False)  # unused when sparsity is fixed

    def zero_fixed_grads(self):
        """Zero the gradient on logits pinned by set_fixed_order, so a fixed order stays fixed."""
        if hasattr(self, 'fixed_mask') and self.logits.grad is not None:
            self.logits.grad[self.fixed_mask] = 0.0
        

    def get_loga(self, P_ste):
        """
            Compute log-alpha for gates based on current straight-through permutation and gate affinity.
            This transforms the gate affinity from feature space to rank space, so it can be used to compute gate probabilities.
        """
        # Gradients reach the gate probabilities, not the permutation.
        P_ste = P_ste.detach()
        loga = P_ste @ self.gate_affinity

        return loga
    

    def get_sparsity_mask(self, loga, training=True, method='ste'):
        """
        Compute sparsity mask using either STE or hard-concrete.
        
        Args:
            loga: [K, K] log-alpha values
            training: whether in training mode
            method: 'ste' or 'hard_concrete'
        
        Returns:
            mask: [K, K] with exact zeros possible
        """
        if method == 'ste':
            if training:
                unif = torch.rand_like(loga).clamp(1e-8, 1 - 1e-8)
                g_soft = torch.sigmoid((torch.log(unif) - torch.log(1 - unif) + loga) / self.temperature)
            else:
                g_soft = torch.sigmoid(loga)
            g_hard = (g_soft > 0.5).float()
            if not training:
                return g_hard
            return g_hard - g_soft.detach() + g_soft

        elif method == 'hard_concrete':
            gamma, zeta = -0.25, 1.25

            if training:
                unif = torch.rand_like(loga).clamp(1e-8, 1 - 1e-8)
                g_soft = torch.sigmoid((torch.log(unif) - torch.log(1 - unif) + loga) / self.temperature)
            else:
                g_soft = torch.sigmoid(loga)
            g_stretched = g_soft * (zeta - gamma) + gamma
            g_hard = F.hardtanh(g_stretched, min_val=0.0, max_val=1.0)
            return g_hard
        
        else:
            raise ValueError(f"Unknown method: {method}")

    def keep_prob(self, loga):
        """P(g > 0) — different formula per method."""
        if self.gate_method == 'ste':
            return torch.sigmoid(loga)
        elif self.gate_method == 'hard_concrete':
            gamma, zeta = -0.25, 1.25
            return torch.sigmoid(loga - self.temperature * math.log(-gamma / zeta))
        else:
            raise ValueError(f"Unknown method: {self.gate_method}")


    @torch.no_grad()
    def hard_permutation(self, P_soft):
        """Nearest hard permutation to a relaxed one, by linear assignment."""
        if P_soft.dim() ==2:
            P_soft = P_soft.unsqueeze(0)
        P_hard = torch.zeros_like(P_soft)
        for i in range(P_soft.shape[0]):
            cost = (-P_soft[i]).detach().cpu().numpy()
            row_ind, col_ind = linear_sum_assignment(cost)
            P_hard[i, row_ind, col_ind] = 1.0
        return P_hard

    
    def _sample_permutations(self, n_samples):
        """Draw n_samples relaxed permutations and their hard counterparts."""
        tau = self.tau if self.training else 0.01
        if self.perm_method == 'sinkhorn':
            P_soft = gumbel_sinkhorn(self.logits, tau=tau, n_iters=self.n_iters,
                                    noise=self.training, n_samples=n_samples)
            P_hard = self.hard_permutation(P_soft)              # LAP
        else:  # softsort
            s = self.logits
            if self.training and not self.use_fixed_order:
                if self.antithetic:
                    n_half = (n_samples + 1) // 2
                    s_pert = s + sample_gumbel((n_half, self.K), s.device)
                    s = torch.cat([s_pert, -s_pert], dim=0)[:n_samples]  # each draw + its reverse
                else:
                    s = s + sample_gumbel((n_samples, self.K), s.device)
            else:
                # eval, or a fixed order (rigid: no Gumbel jitter across ranks)
                s = s.expand(n_samples, self.K)
            P_soft = softsort(s, tau)
            idx = s.argsort(dim=-1, descending=True)
            P_hard = torch.zeros_like(P_soft).scatter_(2, idx.unsqueeze(-1), 1.0)  # argsort, no LAP
        return P_soft, P_hard



    def forward(self, n_samples=1):
        """Sample the structure for one training step.

        Returns (mask, k_map, l0_penalty, entropy). `mask` [n, K, K] is the
        triangular mask after gating, `k_map` selects each component's own variable,
        `l0_penalty` is the expected number of retained conditioning variables, and
        `entropy` measures how far the ordering is from committed.

        Training uses relaxed draws; evaluation and a fixed order use a single hard
        permutation.
        """
        
        K = self.K
        # Soft near-permutation (use noise during training, deterministic at eval)
        use_hard = (not self.training) or self.use_fixed_order
        n_samples = 1 if use_hard else n_samples
        P_soft, P_hard = self._sample_permutations(n_samples)
        P_ste = P_hard - P_soft.detach() + P_soft
        # detached 0-dim tensor, NOT .item(): the CPU sync ran every training step and cost
        # ~5-10% of a step; callers convert with float() only when they actually log it.
        entropy = (-(P_soft * (P_soft + 1e-8).log()).sum(dim=(-2, -1)) / self.K).mean().detach()
        M_hard = torch.cumsum(P_hard, dim=1)
        
        
        if self.use_sparsity:
            if self.use_fixed_sparsity:
                sparsity_mask = self.fixed_sparsity              # [1,K,K], given (Oracle/Two-stage)
            else:
                sparsity_mask = self.get_sparsity_mask(self.get_loga(P_ste), training=self.training,method=self.gate_method)
            
        

        prev = M_hard.new_zeros((n_samples,1, K))
        
        if use_hard:       
            k_map = M_hard - torch.cat([prev, M_hard[:,:-1]], dim=1) 
            l0_penalty = self.l0_penalty(k_map, P_ste, M_hard)
            if self.use_sparsity:
                sparsity_mask = torch.where(k_map < 0.5, sparsity_mask,
                                     torch.ones_like(sparsity_mask))
                M_hard = M_hard*sparsity_mask 
            
            
            return M_hard, k_map, l0_penalty, entropy

        M_soft = torch.cumsum(P_soft, dim=1)  # [K, K] nested triangular 
        M_ste = M_hard.detach() - M_soft.detach() + M_soft
        
        k_map = M_ste - torch.cat([prev, M_ste[:,:-1]], dim=1)
        
        l0_penalty = self.l0_penalty(k_map, P_ste, M_ste)
        if self.use_sparsity:
            sparsity_mask = torch.where(k_map < 0.5, sparsity_mask,
                                     torch.ones_like(sparsity_mask))
            
            M_ste = M_ste + M_ste.detach() * (sparsity_mask - 1.0) #  same as M_ste * mask, but gradient flows as if unmasked (ordering learns through closed gates)
            
        return M_ste, k_map, l0_penalty, entropy 
        
    
    def l0_penalty(self, k_map, P_ste, mask):
        """
        Compute expected L0 penalty for the current batch of samples.
        k_map: [n_samples, K, K] binary mask indicating the diagonal variables (1 if the gate is active, 0 if inactive)
        P_ste: [n_samples, K, K] straight-through permutation matrix
        mask: [K, K] binary mask indicating which gates can be active (1) or must be inactive (0)
        """
        # Expected number of “on” gates (proxy for L0). Use keep_prob on loga.
        # keep_prob returns P(z>0). That’s the standard L0 expected count term.
        n_samples = k_map.shape[0]
        if not self.use_sparsity or self.use_fixed_sparsity:
            return torch.tensor(0.0, device=k_map.device)
        can_be_sparse = (mask - k_map).detach()  # stop gradients
        loga = self.get_loga(P_ste)
        q = self.keep_prob(loga)       # [K,K]

    
        off_pen = (q * can_be_sparse ).sum() 
        
        return off_pen/n_samples 

    @torch.no_grad()
    def hard_order(self):
        """Current ordering as feature indices, rank 0 first. Shape [K]."""
        if self.perm_method == 'sinkhorn':
        # rank -> feature
            P = gumbel_sinkhorn(self.logits, tau=0.01, n_iters=self.n_iters, noise=False)
            P_hard = self.hard_permutation(P)
            return P_hard.argmax(dim=-1).squeeze(0)  # [K]
        else:  # softsort
            return self.logits.argsort(dim=-1, descending=True).squeeze(0)

    @torch.no_grad()
    def structure_entropy(self, n_draw=128):
        """Factored structural uncertainty for per-epoch logging (nats). Complements the
        tau-dependent relaxed-permutation `entropy` already logged with two signals that say
        *which subsystem is still deciding*:
          h_gates      -- mean binary entropy of the per-edge keep-probabilities over the
                          prunable region (which parents the L0 gates are still unsure about);
                          nan when sparsity is fixed (Oracle/two-stage) or not yet on.
          h_perm_hard  -- empirical entropy of the realised hard orders under Gumbel
                          perturbation; tau-INDEPENDENT, so it -> 0 only when the sampled
                          argsort orders actually agree (the order is locked), not because the
                          anneal shrank tau. Softsort only.
        """
        K, dev, nan = self.K, self.logits.device, float("nan")

        # H_gates: binary entropy of keep-probs on the can_be_sparse edges (as in l0_penalty).
        if self.use_fixed_sparsity or not self.use_sparsity:
            h_gates = nan
        else:
            order = self.hard_order().reshape(-1)                              # [K] rank->feature (method-agnostic:
            P_hard = torch.zeros(1, K, K, device=dev)                          # softsort argsort OR sinkhorn+LAP)
            P_hard[0, torch.arange(K, device=dev), order] = 1.0
            M_hard = torch.cumsum(P_hard, dim=1)                               # [1,K,K] nested triangular
            prev = M_hard.new_zeros((1, 1, K))
            k_map = M_hard - torch.cat([prev, M_hard[:, :-1]], dim=1)          # introduced feature (diagonal)
            can_be_sparse = (M_hard - k_map).clamp(0, 1)                       # prunable parents
            pi = self.keep_prob(self.get_loga(P_hard)).clamp(1e-6, 1 - 1e-6)   # per-edge keep-prob
            Hb = -(pi * pi.log() + (1 - pi) * (1 - pi).log())                  # per-edge binary entropy
            h_gates = float((Hb * can_be_sparse).sum() / can_be_sparse.sum().clamp_min(1.0))

        # H_perm(hard): empirical entropy of Gumbel-perturbed argsort orders (tau-free).
        if self.perm_method != 'softsort':
            h_perm_hard = nan
        elif self.use_fixed_order:
            h_perm_hard = 0.0
        else:
            s = self.logits + sample_gumbel((n_draw, K), dev)                 # [n_draw,K]
            idx = s.argsort(dim=-1, descending=True)                          # [n_draw,K] rank->feature
            onehot = torch.zeros(n_draw, K, K, device=dev).scatter_(2, idx.unsqueeze(-1), 1.0)
            freq = onehot.mean(0).clamp_min(1e-12)                            # [K,K] feature freq per rank
            h_perm_hard = float((-(freq * freq.log()).sum(-1)).mean())       # mean over ranks

        return {"h_gates": h_gates, "h_perm_hard": h_perm_hard}

