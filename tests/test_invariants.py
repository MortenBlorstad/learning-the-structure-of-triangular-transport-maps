"""Structural invariants of SSTM — the model correctness gate.

`pytest tests/test_invariants.py` MUST pass before keeping any change to the model.
These check the properties an edit to the model must never break: bijectivity, the
nested-k-hot ordering structure, per-component triangularity, monotonicity in the
introduced variable, and — the key one — that the analytic `forward_with_jac`
Jacobian carry produces the same loss AND parameter gradients as computing the
diagonal Jacobian via autograd.
"""

import numpy as np
import torch
import pytest

from triangularmap.sstm import SSTM, _MonoMasks
from distributions import funnel_adjacency, sample_funnel_distribution

FIXED = [2, 0, 3, 1]   # non-identity -> distinguishes rank<->feat semantics

# (parameter_sharing, row_norm, mono_frac):
# BE default | BE without norm | pure no-sharing (norm + gain S) | plain baseline
# | BE at the swept mono fractions | no-sharing at a non-default fraction
VARIANTS = [(True, True, 0.5), (True, False, 0.5), (False, True, 0.5), (False, False, 0.5),
            (True, True, 0.25), (True, True, 1.0), (False, True, 0.25)]


def test_pushforward_pullback():
    n = 1000
    K = 5
    np.random.seed(0)
    X = np.random.randn(n, K)
    X_test = np.random.randn(10, K)

    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    X_norm = (X - mean) / std
    X_test_norm = (X_test - mean) / std

    model = SSTM(input_size=K, hidden=[64, 64], output_dim=1, random_state=0, gate_method="ste")
    model.fit(X_norm, epochs=1, batch_size=32, lr_map=1e-3, l2=1e-4, sparsity_strength=1e-4, verbose=False)

    X_pullback = model.pullback(model.pushforward(X_norm))
    X_test_pullback = model.pullback(model.pushforward(X_test_norm))

    assert X_pullback.shape == X_norm.shape
    assert X_test_pullback.shape == X_test_norm.shape
    # pullback(pushforward(x)) == x: the map is a bijection on the standardized data
    assert np.allclose(X_pullback, X_norm, atol=1e-6)
    assert np.allclose(X_test_pullback, X_test_norm, atol=1e-6)


def test_pushforward_pullback_funnel():
    n = 1000
    K = 5
    np.random.seed(0)
    B = funnel_adjacency(K)
    X = sample_funnel_distribution(n, B)
    X_test = sample_funnel_distribution(n, B)

    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    X_norm = (X - mean) / std
    X_test_norm = (X_test - mean) / std

    model = SSTM(input_size=K, hidden=[64, 64], output_dim=1, random_state=0, gate_method="ste")
    model.fit(X_norm, epochs=10, batch_size=32, lr_map=1e-3, l2=1e-4, sparsity_strength=1e-4, verbose=False)

    X_pullback = model.pullback(model.pushforward(X_norm))
    X_test_pullback = model.pullback(model.pushforward(X_test_norm))

    assert X_pullback.shape == X_norm.shape
    assert X_test_pullback.shape == X_test_norm.shape
    # pullback(pushforward(x)) == x: the map is a bijection on the standardized data
    assert np.allclose(X_pullback, X_norm, atol=1e-6)
    assert np.allclose(X_test_pullback, X_test_norm, atol=1e-6)


def test_monotonic_in_k():
    """The nested-k-hot mask is cumulative: once a feature is introduced at a rank
    it stays active for all higher k, and exactly one feature is introduced per stage.
    """
    m = SSTM(input_size=4, hidden=[8, 8], output_dim=1,
                     random_state=0, fixed_order=FIXED)
    m.eval()
    with torch.no_grad():
        mask, k_map, _, _ = m.ordering()
    mask2d = mask.squeeze(0).cpu()
    k_map2d = k_map.squeeze(0).cpu()
    K = m.K
    # once a feature is introduced at some rank, it stays active for all higher k
    for k in range(1, K):
        assert torch.all(mask2d[k] >= mask2d[k - 1]), f"mask not monotone non-decreasing at row {k}"
    # row k has exactly k+1 active features; final row activates everything
    assert torch.equal(mask2d.sum(dim=1), torch.arange(1, K + 1, dtype=mask2d.dtype))
    assert torch.all(mask2d[K - 1] == 1.0)
    # k_map introduces exactly one feature per stage, consistent with the cumulative mask
    assert torch.allclose(k_map2d.sum(dim=1), torch.ones(K, dtype=k_map2d.dtype))
    prev = torch.zeros_like(mask2d[0])
    for k in range(K):
        assert torch.equal(k_map2d[k], mask2d[k] - prev)
        prev = mask2d[k]


@pytest.mark.parametrize("sharing,row_norm,mono_frac", VARIANTS)
def test_pushforward_pullback_roundtrip_nonidentity_order(sharing, row_norm, mono_frac):
    X = np.random.RandomState(0).randn(128, 4).astype(np.float32)
    m = SSTM(input_size=4, hidden=[16, 16], output_dim=1,
                     random_state=0, fixed_order=FIXED, parameter_sharing=sharing,
                     row_norm=row_norm, mono_frac=mono_frac)
    m.fit(X, epochs=1, batch_size=32, verbose=False)
    Z = m.pushforward(X)
    X_back = m.pullback(Z)
    assert np.abs(X_back - X).max() < 1e-4


def test_hard_order_matches_mask():
    # Direct regression: _hard_order()[k] must point to the unique feature
    # in mask[k, :] that has the 1, otherwise pullback and forward disagree.
    m = SSTM(input_size=4, hidden=[8, 8], output_dim=1,
                     random_state=0, fixed_order=FIXED)
    m.eval()
    with torch.no_grad():
        mask, k_map, _, _ = m.ordering()
    mask2d = mask.squeeze(0)        # tolerate [K,K] or [1,K,K]
    k_map2d = k_map.squeeze(0)
    order = m._hard_order()
    for k in range(m.K):
        assert mask2d[k, int(order[k])].item() == 1.0
        assert k_map2d[k, int(order[k])].item() == 1.0


@pytest.mark.parametrize("sharing,row_norm,mono_frac", VARIANTS)
def test_per_member_input_isolation(sharing, row_norm, mono_frac):
    # Member k's output must only depend on features in mask[k, :].
    # Catches mask broadcast / shape bugs.
    m = SSTM(input_size=4, hidden=[8, 8], output_dim=1,
                     random_state=0, fixed_order=FIXED, parameter_sharing=sharing,
                     row_norm=row_norm, mono_frac=mono_frac)
    m.fit(np.random.RandomState(0).randn(64, 4).astype(np.float32),
          epochs=1, batch_size=32, verbose=False)
    m.eval()
    X_full = torch.randn(8, 4)
    with torch.no_grad():
        mask, _, _, _ = m.ordering()
    mask2d = mask.squeeze(0).cpu().numpy()
    z_full = m.pushforward(X_full.numpy())
    for k in range(m.K):
        X_iso = X_full * torch.tensor(mask2d[k])
        z_iso = m.pushforward(X_iso.numpy())
        assert np.allclose(z_full[:, k], z_iso[:, k], atol=1e-5)


@pytest.mark.parametrize("sharing,row_norm,mono_frac", VARIANTS)
def test_monotonic_in_introduced_variable(sharing, row_norm, mono_frac):
    # T_k must be strictly increasing in x_{order[k]} (architecture guarantee).
    m = SSTM(input_size=4, hidden=[8, 8], output_dim=1,
                     random_state=0, fixed_order=FIXED, parameter_sharing=sharing,
                     row_norm=row_norm, mono_frac=mono_frac)
    m.fit(np.random.RandomState(0).randn(64, 4).astype(np.float32),
          epochs=1, batch_size=32, verbose=False)
    order = m._hard_order().cpu().numpy()
    for k, feat_k in enumerate(order):
        xs = np.linspace(-2, 2, 25).astype(np.float32)
        X_probe = np.zeros((25, m.K), dtype=np.float32)
        X_probe[:, feat_k] = xs
        z = m.pushforward(X_probe)
        assert np.all(np.diff(z[:, k]) > 0), f"T_{k} not monotone in x_{feat_k}"


@pytest.mark.parametrize("sharing,row_norm,mono_frac", VARIANTS)
def test_forward_with_jac_grad_matches_autograd(sharing, row_norm, mono_frac):
    # Parameter gradients from the analytic forward_with_jac path must match
    # those obtained by computing the diagonal Jacobian via autograd, with the
    # ordering learned (not fixed) so the check covers the learned-ordering path.
    m = SSTM(input_size=4, hidden=[8, 8], output_dim=1, random_state=0,
                     parameter_sharing=sharing, row_norm=row_norm, mono_frac=mono_frac)
    m.fit(np.random.RandomState(0).randn(64, 4).astype(np.float32),
          epochs=1, batch_size=32, verbose=False)
    m.train()
    B, K = 32, m.K
    x_sample = torch.tensor(
        np.random.RandomState(1).randn(B, K).astype(np.float32),
        device=m.device,
    )
    mask, k_map, _, _ = m.ordering(B)
    mask, k_map = mask.detach(), k_map.detach()

    # Path A: analytic Jacobian carry.
    m.zero_grad()
    z_a, log_jac_a = m.forward_with_jac(x_sample.clone(), mask=mask, k_map=k_map)
    loss_a = m.triangular_transport_loss_fast(z_a, log_jac_a)
    loss_a.backward()
    grads_a = {n: p.grad.detach().clone() for n, p in m.named_parameters() if p.grad is not None}

    # Path B: same forward, log|J_diag| recovered via autograd on z[:, k].
    m.zero_grad()
    x = x_sample.clone().requires_grad_(True)
    z_b = m.forward(x, mask=mask, k_map=k_map)
    log_jac_b = torch.zeros(B, K, device=m.device)
    for k in range(K):
        grad_k = torch.autograd.grad(z_b[:, k].sum(), x, retain_graph=True, create_graph=True)[0]
        diag = (grad_k * k_map[:, k, :]).sum(dim=-1)
        log_jac_b[:, k] = diag.abs().clamp(min=1e-8).log()
    loss_b = m.triangular_transport_loss_fast(z_b, log_jac_b)
    loss_b.backward()
    grads_b = {n: p.grad.detach().clone() for n, p in m.named_parameters() if p.grad is not None}

    assert torch.allclose(loss_a, loss_b, atol=1e-5, rtol=1e-4), \
        f"loss mismatch: analytic={loss_a.item():.6f}, autograd={loss_b.item():.6f}"
    assert set(grads_a) == set(grads_b), "parameter sets differ between the two paths"
    for name in grads_a:
        denom = grads_b[name].abs().max().item() + 1e-8
        rel = (grads_a[name] - grads_b[name]).abs().max().item() / denom
        assert rel < 1e-3, f"grad mismatch for {name}: rel_err={rel:.2e}"


@pytest.mark.parametrize("sharing,row_norm,mono_frac", VARIANTS)
def test_save_load_roundtrip(sharing, row_norm, mono_frac, tmp_path):
    # _config must fully reconstruct the architecture (incl. parameter_sharing,
    # row_norm, mono_frac): a loaded model reproduces the saved pushforward bit-for-bit.
    X = np.random.RandomState(0).randn(64, 4).astype(np.float32)
    m = SSTM(input_size=4, hidden=[8, 8], output_dim=1,
                     random_state=0, fixed_order=FIXED, parameter_sharing=sharing,
                     row_norm=row_norm, mono_frac=mono_frac)
    m.fit(X, epochs=1, batch_size=32, verbose=False)
    m.eval()
    z1 = m.pushforward(X[:8])
    path = m.save(tmp_path / f"m_sharing_{sharing}_{row_norm}_{mono_frac}.pt")
    m2 = SSTM.load(path)
    z2 = m2.pushforward(X[:8])
    assert np.allclose(z1, z2, atol=1e-6)


def test_structure_entropy_both_perm_methods():
    # Regression: h_gates built the hard permutation
    # by argsorting the logits as softsort scores [1,K]; sinkhorn's [1,K,K] logits made
    # the scatter index rank-mismatch and crashed every logged sinkhorn fit. It must
    # use the method-agnostic hard_order(). h_perm_hard stays softsort-only (nan).
    from triangularmap.triangular_ordering import TriangularOrdering
    for method, perm_finite in (("softsort", True), ("sinkhorn", False)):
        o = TriangularOrdering(K=4, tau=0.5, n_iters=10, droprate_init=0.33,
                               perm_method=method)
        o.use_sparsity = True
        out = o.structure_entropy()
        assert np.isfinite(out["h_gates"]), method
        assert np.isfinite(out["h_perm_hard"]) == perm_finite, method


def test_make_u_cols_rule():
    # Exactly max(1, round(frac*q)) mono columns for any (q, frac); 0.5 with even q
    # (and q=1) reproduces the historical even-column convention bit-identically.
    import torch as t
    for q in range(1, 41):
        for frac in (0.1, 0.25, 0.5, 0.75, 0.9, 1.0):
            U = _MonoMasks.make_U_cols(q, False, frac)
            assert U.shape == (1, q)
            assert int(U.sum()) == max(1, round(frac * q)), (q, frac)
    for q in [1] + list(range(2, 41, 2)):
        expected = (t.arange(q) % 2 == 0).view(1, -1) if q > 1 else t.ones(1, 1, dtype=t.bool)
        assert t.equal(_MonoMasks.make_U_cols(q, False, 0.5), expected), q
    assert _MonoMasks.make_U_cols(6, True, 0.5).all()     # all_monotonic overrides


def test_parameter_sharing_param_counts():
    # BatchEnsemble per layer: q*p (shared W) + K*(p + 2q) (R, S, B).
    # MonotoneLinear plain per layer: K*(q*p + q) (own W and bias per component).
    # MonotoneLinear row_norm per layer: + K*q (the per-row gain S).
    K, H = 4, [8, 8]
    sizes = [(K, 8), (8, 8), (8, 1)]                     # (p, q) per layer
    shared = SSTM(input_size=K, hidden=H, output_dim=1, random_state=0)
    plain = SSTM(input_size=K, hidden=H, output_dim=1, random_state=0,
                         parameter_sharing=False, row_norm=False)
    gained = SSTM(input_size=K, hidden=H, output_dim=1, random_state=0,
                          parameter_sharing=False, row_norm=True)
    unnormed = SSTM(input_size=K, hidden=H, output_dim=1, random_state=0,
                            row_norm=False)
    count = lambda m: sum(p.numel() for p in m.maps.parameters())
    assert count(shared) == sum(q * p + K * (p + 2 * q) for p, q in sizes)
    assert count(unnormed) == count(shared)              # the norm is parameter-free
    assert count(plain) == sum(K * (q * p + q) for p, q in sizes)
    assert count(gained) == sum(K * (q * p + 2 * q) for p, q in sizes)


def test_fit_ignores_x_val():
    # Val-leakage guard: X_val is inspection-only. Passing a totally different X_val to
    # fit must NOT change the trained weights (no early stopping / best-val restore /
    # val-based selection). fit must return the end-of-training model. This catches, in
    # CI, any future edit that lets validation data influence training or weight choice.
    X = np.random.RandomState(0).randn(96, 4).astype(np.float32)
    X_val_real = np.random.RandomState(1).randn(48, 4).astype(np.float32)
    X_val_garbage = np.full((48, 4), 7.0, dtype=np.float32)   # nothing like the data

    def _fit(x_val):
        m = SSTM(input_size=4, hidden=[8, 8], output_dim=1, random_state=0)
        m.fit(X, epochs=3, batch_size=32, verbose=False, X_val=x_val)
        return [p.detach().clone() for p in m.parameters()]

    pa, pb = _fit(X_val_real), _fit(X_val_garbage)
    assert len(pa) == len(pb)
    for a, b in zip(pa, pb):
        assert torch.equal(a, b), "fit weights depend on X_val — validation is leaking into training/selection"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
