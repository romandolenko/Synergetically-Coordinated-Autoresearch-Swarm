"""Coordinator: dataclasses, baseline hashes, safety-envelope validator, control law."""

from __future__ import annotations

import hashlib
import math
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


BASELINE_PYPROJECT_HASH: str = _file_sha256(_REPO_ROOT / "pyproject.toml")
BASELINE_PREPARE_HASH: str = _file_sha256(_REPO_ROOT / "prepare.py")

MIN_PLAUSIBLE_VAL_BPB: float = 0.5
MAX_FAILURES_IN_LAST_5: int = 2

# Control-law defaults (design §5). Tuned values from results-day3.md §13:
#   w_p=0 (perf-gap penalty off — saturates alpha and traps agents at first attractor)
#   psi_0=2.0 (alpha-baseline biased low — gives the mix proposer room to explore)
# Overridable via env vars for further sweeps (Day 4's coupling_T sweep uses
# SCAS_W_P/SCAS_PSI_0=0/2.0 implicitly via the defaults).
W_D: float = float(os.environ.get("SCAS_W_D", "1.0"))
W_P: float = float(os.environ.get("SCAS_W_P", "0.0"))
PSI_0: float = float(os.environ.get("SCAS_PSI_0", "2.0"))
DEFAULT_COUPLING_T: float = 1.0


@dataclass(frozen=True)
class HypothesisCard:
    agent_id: str
    step: int
    summary: str
    category: str
    parent_step: int | None


@dataclass(frozen=True, eq=False)
class StepResult:
    card: HypothesisCard
    val_bpb: float
    embedding: np.ndarray  # 384-d
    accepted: bool
    reject_reason: str | None


@dataclass
class CoordinatorState:
    groups: dict[int, list[str]] = field(default_factory=dict)
    attractors: dict[int, np.ndarray] = field(default_factory=dict)
    group_best_val_bpb: dict[int, float] = field(default_factory=dict)
    psi: dict[str, float] = field(default_factory=dict)
    alpha: dict[str, float] = field(default_factory=dict)


def validate(commit: dict, recent_failures: int = 0) -> tuple[bool, str | None]:
    """Apply the invariant-manifold safety envelope (design §8).

    Returns (ok, reason). On reject, `reason` is a short non-empty string.
    """
    files = list(commit.get("files_changed", []))
    if files != ["train.py"]:
        extras = [f for f in files if f != "train.py"]
        return False, f"files_changed must be exactly ['train.py']; offending: {extras or files}"
    if commit.get("pyproject_hash") != BASELINE_PYPROJECT_HASH:
        return False, "pyproject.toml hash drifted from baseline"
    if commit.get("prepare_hash") != BASELINE_PREPARE_HASH:
        return False, "prepare.py hash drifted from baseline"
    val_bpb = commit.get("val_bpb")
    if val_bpb is None or val_bpb < MIN_PLAUSIBLE_VAL_BPB:
        return False, f"val_bpb {val_bpb} below MIN_PLAUSIBLE_VAL_BPB={MIN_PLAUSIBLE_VAL_BPB}"
    if recent_failures > MAX_FAILURES_IN_LAST_5:
        return False, f"crash budget exceeded: {recent_failures} failures in last 5"
    return True, None


def compute_psi(
    z: np.ndarray,
    attractor: np.ndarray,
    val_bpb: float,
    group_best: float,
    w_d: float = W_D,
    w_p: float = W_P,
) -> float:
    """psi_i = w_d * dist(z, c_g) + w_p * max(0, val_bpb - val_bpb_g_best)."""
    dist = float(np.linalg.norm(z - attractor))
    perf_gap = max(0.0, float(val_bpb) - float(group_best))
    return w_d * dist + w_p * perf_gap


def compute_alpha(psi: float, coupling_T: float, psi_0: float = PSI_0) -> float:
    """alpha_i = sigmoid((psi - psi_0) / coupling_T) in [0, 1]."""
    x = (psi - psi_0) / coupling_T
    # numerically stable sigmoid
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


class Coordinator:
    """Holds per-group attractors, per-agent psi/alpha, and applies the control law."""

    def __init__(
        self,
        agents: list[str],
        categories: list[str],
        coupling_T: float = DEFAULT_COUPLING_T,
        w_d: float = W_D,
        w_p: float = W_P,
        psi_0: float = PSI_0,
    ):
        if not agents or not categories:
            raise ValueError("agents and categories must both be non-empty")
        self.coupling_T = coupling_T
        # Per-instance control-law constants (default to module/env values, but
        # overridable so sim.py can sweep w_p without env-var gymnastics — the
        # committed w_p=0 makes the coordinator performance-blind, see review).
        self.w_d = w_d
        self.w_p = w_p
        self.psi_0 = psi_0
        self.agent_to_group: dict[str, int] = {
            a: i % len(categories) for i, a in enumerate(agents)
        }
        self.group_to_category: dict[int, str] = {
            i: c for i, c in enumerate(categories)
        }
        self.state = CoordinatorState()
        for g in range(len(categories)):
            self.state.groups[g] = [
                a for a, gg in self.agent_to_group.items() if gg == g
            ]
            self.state.group_best_val_bpb[g] = float("inf")

    def observe(
        self,
        card: HypothesisCard,
        val_bpb: float,
        embedding: np.ndarray,
        accepted: bool,
    ) -> bool:
        """Update group state and recompute psi/alpha for this agent.

        Returns True if this observation improved the group best (attractor update).
        """
        if not accepted:
            return False
        g = self.agent_to_group[card.agent_id]
        improved = (
            g not in self.state.attractors
            or val_bpb < self.state.group_best_val_bpb[g]
        )
        if improved:
            self.state.group_best_val_bpb[g] = float(val_bpb)
            self.state.attractors[g] = embedding.copy()
        attr = self.state.attractors[g]
        best = self.state.group_best_val_bpb[g]
        psi = compute_psi(embedding, attr, val_bpb, best, self.w_d, self.w_p)
        self.state.psi[card.agent_id] = psi
        self.state.alpha[card.agent_id] = compute_alpha(psi, self.coupling_T, self.psi_0)
        return improved


class SCTCoordinator:
    """Genuine synergetic dynamics — the dynamical-systems law the steady-state
    ``Coordinator`` only approximates.

    Two first-order relaxations are integrated explicitly (forward Euler, one
    ``dt`` per observation), both sharing the single timescale ``coupling_T``:

      * **Group attractor as a continuous order parameter.** Instead of snapping
        the attractor to the best card's embedding, it relaxes toward the running
        group-best ``z*_g``::

            coupling_T * ċ_g = z*_g - c_g

        renormalised to the unit sphere (MiniLM embeddings are unit-norm). The
        consensus is a slow order parameter, not an instantaneous argmax.

      * **Per-agent coupling order parameter ``psi``.** The synergetic relaxation
        ``coupling_T * ψ̇ + ψ = u`` is actually integrated, so ``psi`` carries
        memory of past mismatch instead of being recomputed from scratch::

            u_i  = w_d * dist(z_i, c_g) + w_p * max(0, val_bpb_i - val_bpb_g_best)
            ψ_i ← ψ_i + (dt / coupling_T) * (u_i - ψ_i)

      ``alpha_i = sigmoid((psi_i - psi_0) / alpha_temp)`` in [0, 1] then gates the
      proposer's exploit-vs-explore mix.

    Small ``coupling_T`` ⇒ fast relaxation, ``psi`` tracks the instantaneous drive
    (≈ the algebraic steady-state law). Large ``coupling_T`` ⇒ a sticky order
    parameter that enslaves the agents gradually (Haken's enslaving principle).
    That transient memory — and the relaxing consensus — is the dynamical content
    the committed ``Coordinator`` omits by construction.

    Contrast also with the engineered ``--embedding-repulsion``/``--anneal``
    heuristic: this coordinator has **no** shared coverage memory, **no** exploit
    cap, and **no** scripted convergence schedule. Any diversity-with-convergence
    must emerge from the dynamics above.
    """

    coupling_T: float

    def __init__(
        self,
        agents: list[str],
        categories: list[str],
        coupling_T: float = DEFAULT_COUPLING_T,
        w_d: float = W_D,
        w_p: float = W_P,
        psi_0: float = PSI_0,
        alpha_temp: float = 1.0,
        dt: float = 1.0,
    ):
        if not agents or not categories:
            raise ValueError("agents and categories must both be non-empty")
        if coupling_T <= 0:
            raise ValueError("coupling_T must be > 0 (it is a relaxation timescale)")
        self.coupling_T = coupling_T
        self.w_d = w_d
        self.w_p = w_p
        self.psi_0 = psi_0
        self.alpha_temp = alpha_temp
        self.dt = dt
        self.agent_to_group: dict[str, int] = {
            a: i % len(categories) for i, a in enumerate(agents)
        }
        self.group_to_category: dict[int, str] = dict(enumerate(categories))
        self.state = CoordinatorState()
        # Continuous group-best embedding the attractor order parameter chases.
        self._best_emb: dict[int, np.ndarray] = {}
        for g in range(len(categories)):
            self.state.groups[g] = [
                a for a, gg in self.agent_to_group.items() if gg == g
            ]
            self.state.group_best_val_bpb[g] = float("inf")

    def observe(
        self,
        card: HypothesisCard,
        val_bpb: float,
        embedding: np.ndarray,
        accepted: bool,
    ) -> bool:
        """Integrate one step of the attractor and psi relaxations for this agent.

        Returns True if this observation improved the group best.
        """
        if not accepted:
            return False
        g = self.agent_to_group[card.agent_id]
        improved = (
            g not in self.state.attractors
            or val_bpb < self.state.group_best_val_bpb[g]
        )
        if improved:
            self.state.group_best_val_bpb[g] = float(val_bpb)
            self._best_emb[g] = embedding.copy()

        z_star = self._best_emb[g]
        r = self.dt / self.coupling_T
        if g not in self.state.attractors:
            c = z_star.copy()
        else:
            c = self.state.attractors[g]
            c = c + r * (z_star - c)
            norm = float(np.linalg.norm(c))
            if norm > 0:
                c = c / norm
        self.state.attractors[g] = c

        u = compute_psi(
            embedding, c, val_bpb, self.state.group_best_val_bpb[g], self.w_d, self.w_p
        )
        psi_prev = self.state.psi.get(card.agent_id)
        psi = u if psi_prev is None else psi_prev + r * (u - psi_prev)
        self.state.psi[card.agent_id] = psi
        self.state.alpha[card.agent_id] = compute_alpha(psi, self.alpha_temp, self.psi_0)
        return improved
