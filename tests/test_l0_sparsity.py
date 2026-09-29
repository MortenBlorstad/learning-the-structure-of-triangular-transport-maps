"""Invariant tests for the L0 sparsity gates in TriangularOrdering.

Covers the structural / mathematical invariants the sparsity layer must satisfy
regardless of training dynamics: mask never violates triangularity, diagonal is
never sparsified, get_loga projection is correct, gradient is properly detached
through the permutation, initialization matches droprate_init, eval is
deterministic, and the L0 penalty saturates to the clamp-implied bounds.
"""
import pytest
import torch

from triangularmap.triangular_ordering import TriangularOrdering


def _swap_permutation(K: int, perm: list[int]) -> torch.Tensor:
    """One-hot P[r, perm[r]] = 1, shape [1, K, K]."""
    P = torch.zeros(1, K, K)
    for r, f in enumerate(perm):
        P[0, r, f] = 1.0
    return P


def test_sparsity_preserves_triangularity():
    """Sparsity mask can only multiply by 0/1 — positions originally 0 in M_hard must stay 0."""
    torch.manual_seed(0)
    K = 6
    model = TriangularOrdering(K=K, gate_method='ste')
    model.eval()

    model.use_sparsity = False
    R_base, _, _, _ = model(n_samples=1)
    zero_positions = (R_base == 0)

    model.use_sparsity = True
    for fill in (-100.0, 0.0, 100.0):
        with torch.no_grad():
            model.gate_affinity.data.fill_(fill)
        R, _, _, _ = model(n_samples=1)
        assert (R[zero_positions] == 0).all(), f"sparsity introduced nonzero above rank diagonal at fill={fill}"


def test_diagonal_never_sparsified():
    """Even with all gates fully closed, the newly-introduced variable at each rank stays on."""
    torch.manual_seed(0)
    K = 6
    model = TriangularOrdering(K=K, gate_method='ste')
    model.eval()
    model.use_sparsity = True
    with torch.no_grad():
        model.gate_affinity.data.fill_(-100.0)

    R, k_map, _l0, _ = model(n_samples=1)
    diagonal = (k_map > 0.5)
    assert (R[diagonal] == 1.0).all()

    # Off-diagonal lower-triangular positions should be killed
    prev_use = model.use_sparsity
    model.use_sparsity = False
    R_base, _, _, _ = model(n_samples=1)
    model.use_sparsity = prev_use
    off_diag_active = (R_base > 0.5) & (k_map < 0.5)
    assert (R[off_diag_active] == 0).all()


def test_get_loga_identity_permutation():
    """P = I should return clamp(gate_affinity)."""
    torch.manual_seed(0)
    K = 4
    model = TriangularOrdering(K=K, gate_method='ste')
    with torch.no_grad():
        model.gate_affinity.data = torch.randn(1, K, K)

    P_I = torch.eye(K).unsqueeze(0)
    loga = model.get_loga(P_I)
    expected = model.gate_affinity.data            # get_loga applies no clamp
    assert torch.allclose(loga, expected, atol=1e-6)


def test_get_loga_permutation_swap():
    """For one-hot P with perm[r] = σ(r), loga[r, c] == gate_affinity[σ(r), c].
    Row index is permuted (rank → target feature σ(r)); column index stays as the source
    feature, so the result is in [rank, feature] coordinates matching M_hard."""
    torch.manual_seed(0)
    K = 4
    perm = [2, 0, 3, 1]
    model = TriangularOrdering(K=K, gate_method='ste')
    with torch.no_grad():
        model.gate_affinity.data = torch.arange(K * K, dtype=torch.float32).reshape(1, K, K) * 0.1

    P = _swap_permutation(K, perm)
    loga = model.get_loga(P)

    expected_unclamped = torch.zeros(1, K, K)
    for r in range(K):
        for c in range(K):
            expected_unclamped[0, r, c] = model.gate_affinity.data[0, perm[r], c]
    expected = expected_unclamped                  # get_loga applies no clamp
    assert torch.allclose(loga, expected, atol=1e-6)


def test_get_loga_stops_gradient_through_permutation():
    """get_loga must detach P_ste so gate gradients don't leak into the ordering logits.
    Gradient must still flow into gate_affinity."""
    torch.manual_seed(0)
    K = 4
    model = TriangularOrdering(K=K, gate_method='ste')

    P = torch.eye(K).unsqueeze(0).clone().requires_grad_(True)
    loga = model.get_loga(P)
    loga.sum().backward()

    assert P.grad is None, "gradient leaked through P_ste despite .detach()"
    assert model.gate_affinity.grad is not None
    assert (model.gate_affinity.grad.abs() > 0).any()


def test_init_keep_prob_matches_droprate_ste():
    """At init with STE, mean(keep_prob) ≈ 1 - droprate_init."""
    K = 8
    droprate = 0.33
    model = TriangularOrdering(K=K, droprate_init=droprate, gate_method='ste')

    P_I = torch.eye(K).unsqueeze(0)
    q = model.keep_prob(model.get_loga(P_I))
    assert torch.allclose(q.mean(), torch.tensor(1.0 - droprate), atol=1e-6)


def test_eval_mode_mask_deterministic():
    """In eval mode get_sparsity_mask must produce identical output on identical loga."""
    torch.manual_seed(0)
    K = 6
    model = TriangularOrdering(K=K, gate_method='ste')
    model.eval()
    with torch.no_grad():
        model.gate_affinity.data = torch.randn(1, K, K)

    P_I = torch.eye(K).unsqueeze(0)
    loga = model.get_loga(P_I)

    for method in ('ste', 'hard_concrete'):
        m1 = model.get_sparsity_mask(loga, training=False, method=method)
        m2 = model.get_sparsity_mask(loga, training=False, method=method)
        assert torch.equal(m1, m2), f"eval mask not deterministic for method={method}"


def test_sparsity_preserves_triangularity_training():
    """Same invariant as eval but with Gumbel-noised P_soft / mask and n_samples > 1.
    Seeds the RNG before each forward so P_hard matches between the baseline and sparsity calls."""
    K = 6
    n_samples = 4
    model = TriangularOrdering(K=K, gate_method='ste')
    model.train()

    model.use_sparsity = False
    torch.manual_seed(42)
    R_base, _, _, _ = model(n_samples=n_samples)
    # M_ste has tiny FP error from (M_hard - M_soft + M_soft); use tolerance
    zero_positions = (R_base.abs() < 1e-4)

    model.use_sparsity = True
    for fill in (-100.0, 0.0, 100.0):
        with torch.no_grad():
            model.gate_affinity.data.fill_(fill)
        torch.manual_seed(42)  # reproduce the same Gumbel noise → same P_hard
        R, _, _, _ = model(n_samples=n_samples)
        assert R[zero_positions].abs().max().item() < 1e-4, \
            f"training mode: sparsity introduced nonzero in zero positions at fill={fill}"


def test_diagonal_never_sparsified_training():
    """Diagonal stays on under Gumbel noise + n_samples > 1, even with all gates closed."""
    K = 6
    n_samples = 4
    model = TriangularOrdering(K=K, gate_method='ste')
    model.train()
    model.use_sparsity = True
    with torch.no_grad():
        model.gate_affinity.data.fill_(-100.0)

    torch.manual_seed(42)
    R, k_map, _l0, _ = model(n_samples=n_samples)
    diagonal = (k_map > 0.5)
    assert R.shape[0] == n_samples
    # Tolerance: M_ste = M_hard.detach() - M_soft.detach() + M_soft has FP rounding
    assert torch.allclose(R[diagonal], torch.ones_like(R[diagonal]), atol=1e-4)


def test_get_loga_batched_permutations():
    """get_loga handles [n_samples, K, K] P with a different permutation per sample.
    Per-sample formula: loga[s, r, c] = clamp(gate_affinity[σ_s(r), c])."""
    torch.manual_seed(0)
    K = 4
    n_samples = 3
    perms = [[0, 1, 2, 3], [2, 0, 3, 1], [3, 2, 1, 0]]
    model = TriangularOrdering(K=K, gate_method='ste')
    with torch.no_grad():
        model.gate_affinity.data = torch.randn(1, K, K) * 0.5

    P = torch.zeros(n_samples, K, K)
    for s, perm in enumerate(perms):
        for r, f in enumerate(perm):
            P[s, r, f] = 1.0

    loga = model.get_loga(P)
    assert loga.shape == (n_samples, K, K)

    expected = torch.zeros(n_samples, K, K)
    for s, perm in enumerate(perms):
        for r in range(K):
            for c in range(K):
                expected[s, r, c] = model.gate_affinity.data[0, perm[r], c]
    # expected is the raw permuted projection of gate_affinity (no clamp)
    assert torch.allclose(loga, expected, atol=1e-6)


def test_l0_penalty_extremes():
    """Penalty -> 0 with all gates closed and -> n_sparsifiable with all open.
    get_loga applies no clamp, so keep_prob = sigmoid(±100) reaches ~0 / 1 and the
    expected L0 count is the off-diagonal lower-triangular position count."""
    torch.manual_seed(0)
    K = 5
    model = TriangularOrdering(K=K, gate_method='ste')
    model.eval()
    model.use_sparsity = True

    n_sparsifiable = K * (K - 1) / 2  # below-rank-diagonal off-diagonal positions
    q_min = float(torch.sigmoid(torch.tensor(-100.0)))   # unclamped affinity -> ~0
    q_max = float(torch.sigmoid(torch.tensor(100.0)))     # -> 1

    with torch.no_grad():
        model.gate_affinity.data.fill_(-100.0)
    _, k_map, _l0, _ = model(n_samples=1)
    pen_closed = _l0.item()
    assert pen_closed == pytest.approx(n_sparsifiable * q_min, abs=1e-6)

    with torch.no_grad():
        model.gate_affinity.data.fill_(100.0)
    _, k_map, _l0, _ = model(n_samples=1)
    pen_open = _l0.item()
    assert pen_open == pytest.approx(n_sparsifiable * q_max, rel=1e-3)


def test_sparsity_mask_in_correct_feature_coordinates():
    """Closing one feature-pair gate must zero exactly that data edge in M_hard.

    Pins the [target, source] semantic of `gate_affinity` against the [rank, feature]
    coordinate system of M_hard. Setup: σ = [2, 0, 3, 1], g[target=1, source=2] = -100,
    all other gates open. The rank-3 map targets feature 1, so source feature 2 must
    be masked out at R[0, 3, 2]; feature 0 must remain available at R[0, 3, 1].

    Regression note: this test fails if get_loga reverts to the sandwich projection
    P @ g @ P.T (rank-rank coordinates), which under a non-identity permutation
    zeros R[0, 3, 0] instead of R[0, 3, 2].
    """
    K = 4
    perm = [2, 0, 3, 1]  # rank → feature

    # sinkhorn: logits are [1,K,K], so we can force a permutation by setting logit cells
    model = TriangularOrdering(K=K, gate_method='ste', perm_method='sinkhorn')
    model.eval()
    model.use_sparsity = True

    with torch.no_grad():
        # Force the permutation by writing a clean one-hot logit pattern directly,
        # so the test exercises the gates alone and does not depend on set_fixed_order.
        model.logits.data.fill_(-10.0)
        for r, f in enumerate(perm):
            model.logits.data[0, r, f] = 10.0
        # All gates wide open except g[target=1, source=2] which we slam shut.
        model.gate_affinity.data.fill_(100.0)
        model.gate_affinity.data[0, 1, 2] = -100.0

    R, k_map, _l0, _ = model(n_samples=1)

    # Sanity: confirm the forced permutation actually took effect.
    for r, f in enumerate(perm):
        assert k_map[0, r, f].item() == 1.0, f"expected feature {f} introduced at rank {r}"

    # Correct semantics under the fix.
    assert R[0, 3, 2].item() == 0.0, (
        "Closed gate g[target=1, source=2] should mask out feature 2 as input to the "
        f"rank-3 map (target = feature 1), so R[0, 3, 2] should be 0; got {R[0, 3, 2].item()}. "
        "Likely: get_loga still uses the rank-rank sandwich projection."
    )
    assert R[0, 3, 0].item() == 1.0, (
        "g[target=1, source=0] is open (+100), so feature 0 must remain available at the "
        f"rank-3 map; got R[0, 3, 0] = {R[0, 3, 0].item()}. "
        "Likely: the coordinate mismatch zeroed this cell instead of [3, 2]."
    )


def test_l0_penalty_extremes_training_batched():
    """Same penalty bounds in training mode with n_samples > 1.
    The /n_samples divisor in l0_penalty must cancel the n_samples-fold sum, so
    per-call penalty is invariant to n_samples (catches a missing-divisor regression)."""
    K = 5
    n_samples = 4
    model = TriangularOrdering(K=K, gate_method='ste')
    model.train()
    model.use_sparsity = True

    n_sparsifiable = K * (K - 1) / 2
    q_min = float(torch.sigmoid(torch.tensor(-100.0)))   # unclamped affinity -> ~0
    q_max = float(torch.sigmoid(torch.tensor(100.0)))     # -> 1

    with torch.no_grad():
        model.gate_affinity.data.fill_(-100.0)
    torch.manual_seed(42)
    _, k_map, _l0, _ = model(n_samples=n_samples)
    pen_closed = _l0.item()
    assert pen_closed == pytest.approx(n_sparsifiable * q_min, abs=1e-6)

    with torch.no_grad():
        model.gate_affinity.data.fill_(100.0)
    torch.manual_seed(43)
    _, k_map, _l0, _ = model(n_samples=n_samples)
    pen_open = _l0.item()
    assert pen_open == pytest.approx(n_sparsifiable * q_max, rel=1e-3)
