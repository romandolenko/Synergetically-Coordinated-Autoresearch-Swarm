"""Unit tests for the control law, SCT integrator, selection, and coverage metrics.

Run with: uv run pytest scas/law_tests.py scas/validator_tests.py -q
No embedding model needed — embeddings are hand-built unit vectors.
"""

from __future__ import annotations

import numpy as np
import pytest

from scas.coordinator import (
    EmergentSCTCoordinator,
    HypothesisCard,
    SCTCoordinator,
    compute_alpha,
    compute_psi,
)
from scas.sim import _first_visit_steps, _select_embedding, _steps_to_coverage
from scas.synthetic import Landscape


def _unit(v: list[float]) -> np.ndarray:
    a = np.asarray(v, dtype=np.float32)
    return a / np.linalg.norm(a)


def _card(agent: str = "a0", step: int = 0) -> HypothesisCard:
    return HypothesisCard(
        agent_id=agent, step=step, summary="s", category="optimizer", parent_step=None
    )


# ---------------------------------------------------------------- control law


def test_compute_alpha_bounded_monotone_centered():
    psis = np.linspace(-10, 10, 41)
    alphas = [compute_alpha(p, coupling_T=1.0, psi_0=0.5) for p in psis]
    assert all(0.0 <= a <= 1.0 for a in alphas)
    assert all(b >= a for a, b in zip(alphas, alphas[1:]))  # monotone in psi
    assert compute_alpha(0.5, coupling_T=1.0, psi_0=0.5) == pytest.approx(0.5)
    # numerically stable far from the center
    assert compute_alpha(1e6, 1.0, 0.0) == pytest.approx(1.0)
    assert compute_alpha(-1e6, 1.0, 0.0) == pytest.approx(0.0)


def test_compute_psi_distance_and_perf_gap_terms():
    z = _unit([1, 0, 0])
    c = _unit([0, 1, 0])
    d = float(np.linalg.norm(z - c))
    assert compute_psi(z, c, 1.0, 1.0, w_d=1.0, w_p=0.0) == pytest.approx(d)
    # perf gap clamps at zero (being *better* than group best adds nothing)
    assert compute_psi(z, c, 0.5, 1.0, w_d=0.0, w_p=10.0) == pytest.approx(0.0)
    assert compute_psi(z, c, 1.3, 1.0, w_d=0.0, w_p=10.0) == pytest.approx(3.0)


# ------------------------------------------------------- SCT integrator (psi)


@pytest.mark.parametrize("T", [0.1, 0.5, 1.0, 3.0, 10.0])
def test_sct_psi_relaxes_monotonically_without_oscillation(T):
    """Regression for the forward-Euler bug: at T < dt the old update
    psi += (dt/T)(u - psi) oscillated and diverged; at T == dt it snapped to u
    (zero memory). The exponential update must approach the constant drive u
    monotonically from below for every T, never overshooting."""
    coord = SCTCoordinator(["a0"], ["optimizer"], coupling_T=T, w_d=1.0, w_p=0.0,
                           psi_0=0.5, alpha_temp=0.3)
    e1 = _unit([1, 0, 0])
    e2 = _unit([0, 1, 0])
    coord.observe(_card(step=0), 1.0, e1, accepted=True)  # attractor pinned at e1
    u = float(np.linalg.norm(e2 - e1))

    psis = []
    for step in range(1, 60):
        coord.observe(_card(step=step), 2.0, e2, accepted=True)  # never improves
        psis.append(coord.state.psi["a0"])

    assert all(np.isfinite(psis))
    assert all(0.0 <= p <= u + 1e-9 for p in psis), "psi overshot the drive u"
    assert all(b >= a - 1e-12 for a, b in zip(psis, psis[1:])), "psi oscillated"
    # converges to u; small T converges fast, large T retains memory
    assert psis[-1] == pytest.approx(u, rel=1e-3 if T <= 1 else 5e-2)
    if T >= 3.0:
        assert psis[0] < 0.5 * u, "large T should remember the initial psi"
    if T <= 0.5:
        assert psis[0] > 0.8 * u, "small T should track the instantaneous drive"


@pytest.mark.parametrize("T", [0.1, 1.0, 10.0])
def test_sct_attractor_relaxes_toward_new_best(T):
    coord = SCTCoordinator(["a0"], ["optimizer"], coupling_T=T, w_d=1.0, w_p=0.0)
    e1 = _unit([1, 0, 0])
    e2 = _unit([0, 1, 0])
    coord.observe(_card(step=0), 1.0, e1, accepted=True)
    coord.observe(_card(step=1), 0.9, e2, accepted=True)  # new best -> z* = e2

    sims = []
    for step in range(2, 40):
        coord.observe(_card(step=step), 2.0, e1, accepted=True)  # never improves
        c = coord.state.attractors[0]
        assert np.linalg.norm(c) == pytest.approx(1.0, abs=1e-5)  # stays on sphere
        sims.append(float(c @ e2))

    assert all(b >= a - 1e-6 for a, b in zip(sims, sims[1:])), "attractor oscillated"
    if T <= 0.1:
        assert sims[0] > 0.999, "small T should snap to the new best"
    if T >= 10.0:
        assert sims[0] < 0.9, "large T should move gradually"
    assert sims[-1] > 0.99  # eventually enslaved by the new best


# ------------------------------------------------- emergent group membership


def test_emergent_seeds_all_slots_before_assigning_nearest():
    coord = EmergentSCTCoordinator(["a0", "a1"], n_groups=2, coupling_T=1.0,
                                   psi_0=0.5, alpha_temp=0.3)
    assert coord.agent_to_group == {"a0": None, "a1": None}
    e0 = _unit([1, 0, 0])
    e1 = _unit([0, 1, 0])
    coord.observe(_card("a0", 0), 1.0, e0, accepted=True)
    coord.observe(_card("a1", 0), 1.1, e1, accepted=True)
    # each accepted observation claimed one empty slot
    assert coord.agent_to_group["a0"] != coord.agent_to_group["a1"]
    assert set(coord.state.attractors.keys()) == {0, 1}
    assert float(coord.state.attractors[0] @ e0) == pytest.approx(1.0, abs=1e-5)
    assert float(coord.state.attractors[1] @ e1) == pytest.approx(1.0, abs=1e-5)


def test_emergent_reassigns_agent_to_nearest_attractor():
    coord = EmergentSCTCoordinator(["a0", "a1", "a2"], n_groups=2, coupling_T=1.0,
                                   psi_0=0.5, alpha_temp=0.3)
    e0 = _unit([1, 0, 0])
    e1 = _unit([0, 1, 0])
    coord.observe(_card("a0", 0), 1.0, e0, accepted=True)   # seeds slot 0
    coord.observe(_card("a1", 0), 1.0, e1, accepted=True)   # seeds slot 1
    # a2 lands near e1 -> joins slot 1
    coord.observe(_card("a2", 0), 2.0, _unit([0.1, 1, 0]), accepted=True)
    assert coord.agent_to_group["a2"] == 1
    assert coord.state.groups[1] == ["a1", "a2"]
    # a2 then moves near e0 -> churns to slot 0; membership lists stay a partition
    coord.observe(_card("a2", 1), 2.0, _unit([1, 0.1, 0]), accepted=True)
    assert coord.agent_to_group["a2"] == 0
    assert coord.state.groups[0] == ["a0", "a2"]
    assert coord.state.groups[1] == ["a1"]
    all_members = coord.state.groups[0] + coord.state.groups[1]
    assert sorted(all_members) == ["a0", "a1", "a2"]


def test_emergent_rejected_observation_changes_nothing():
    coord = EmergentSCTCoordinator(["a0"], n_groups=2, coupling_T=1.0)
    coord.observe(_card("a0", 0), 1.0, _unit([1, 0, 0]), accepted=False)
    assert coord.agent_to_group["a0"] is None
    assert coord.state.attractors == {}


def test_emergent_null_psi0_huge_pins_alpha_at_zero():
    """The matched coupling-off null (--psi-0 1e9): same machinery, alpha == 0."""
    coord = EmergentSCTCoordinator(["a0", "a1"], n_groups=2, coupling_T=1.0,
                                   psi_0=1e9, alpha_temp=0.3)
    coord.observe(_card("a0", 0), 1.0, _unit([1, 0, 0]), accepted=True)
    coord.observe(_card("a1", 0), 1.0, _unit([0, 1, 0]), accepted=True)
    coord.observe(_card("a0", 1), 2.0, _unit([0, 0, 1]), accepted=True)
    assert all(a == pytest.approx(0.0) for a in coord.state.alpha.values())


# ------------------------------------------------------------ selection logic


def test_select_embedding_explore_avoids_covered():
    cands = np.stack([_unit([1, 0, 0]), _unit([0, 1, 0]),
                      _unit([0, 0, 1]), _unit([1, 1, 0])])
    covered = cands[0:1]  # candidate 0 exactly covered
    rng = np.random.default_rng(0)
    for _ in range(20):
        idx = _select_embedding(cands, covered, exploit=False, attractor_emb=None,
                                rng=rng, top_k=3)
        assert idx != 0, "farthest-point explore picked an exactly-covered candidate"


def test_select_embedding_exploit_nearest_but_not_exact_covered():
    cands = np.stack([_unit([1, 0, 0]), _unit([0.9, 0.1, 0]),
                      _unit([0, 1, 0]), _unit([0, 0, 1])])
    attractor = _unit([1, 0, 0])
    rng = np.random.default_rng(0)
    # nothing covered: exploit picks the candidate identical to the attractor
    none_cov = np.zeros((0, 3), dtype=np.float32)
    assert _select_embedding(cands, none_cov, True, attractor, rng) == 0
    # exact match covered: exploit falls to the next-nearest candidate
    assert _select_embedding(cands, cands[0:1], True, attractor, rng) == 1


def test_select_embedding_uniform_when_nothing_covered():
    cands = np.stack([_unit([1, 0, 0]), _unit([0, 1, 0]), _unit([0, 0, 1])])
    none_cov = np.zeros((0, 3), dtype=np.float32)
    rng = np.random.default_rng(0)
    seen = {_select_embedding(cands, none_cov, False, None, rng) for _ in range(50)}
    assert seen == {0, 1, 2}


# ----------------------------------------------------------- coverage metrics


def _tiny_landscape() -> Landscape:
    mu = np.stack([_unit([1, 0, 0]), _unit([0, 1, 0])])
    return Landscape(
        mu=mu, sigma=np.array([0.5, 0.5], dtype=np.float32),
        depth=np.array([0.05, 0.05], dtype=np.float32),
        base=1.0, noise_std=0.0, minima_strings=("m0", "m1"),
    )


def test_first_visit_steps_and_coverage():
    ls = _tiny_landscape()
    log = [
        (0, _unit([0, 0, 1])),   # step 1: near no minimum
        (0, _unit([1, 0, 0])),   # step 2: visits minimum 0 (dist 0 < 0.25)
        (0, _unit([1, 0, 0])),   # step 3: re-visit, not a new first
        (1, _unit([0, 1, 0])),   # step 4: visits minimum 1
    ]
    first = _first_visit_steps(log, ls)
    assert first == [2, 4]
    cov = _steps_to_coverage(first, K_prime=2)
    assert cov == {"0.5": 2, "0.8": 4, "1.0": 4}


def test_steps_to_coverage_never_reached_is_none():
    ls = _tiny_landscape()
    first = _first_visit_steps([(0, _unit([0, 0, 1]))], ls)
    assert first == [None, None]
    cov = _steps_to_coverage(first, K_prime=2)
    assert cov == {"0.5": None, "0.8": None, "1.0": None}
