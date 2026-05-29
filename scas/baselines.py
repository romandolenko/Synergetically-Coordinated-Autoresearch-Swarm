"""Baseline coordinators for head-to-head controls.

Two controls, both with zero coupling (alpha=0 always, proposer never steered):

- ``NullCoordinator`` (design §9): the *independent-walkers* baseline. No group
  assignment; in sim.py each agent draws a random category every step.

- ``PartitionedNullCoordinator`` (review-added fair control): same fixed
  category->group partition as the real ``Coordinator`` and tracks group-best
  attractors so inter/intra separation is *measurable* — but never feeds
  target/avoid back to the proposer and keeps alpha pinned at 0. This isolates
  what the synergetic control law adds *on top of* the partition+prefix
  structure, for both AC-B2 (coverage) and AC-B3 (separation). If this arm
  matches SCAS, the headline effect is the partition, not the coupling.
"""

from __future__ import annotations

import numpy as np

from scas.coordinator import CoordinatorState, HypothesisCard


class NullCoordinator:
    """No-op coordinator: zero coupling, no groups, no attractors."""

    coupling_T: float = 0.0

    def __init__(self, agents: list[str]):
        self.agent_to_group: dict[str, None] = {a: None for a in agents}
        self.group_to_category: dict[int, str] = {}
        self.state = CoordinatorState()
        for a in agents:
            self.state.alpha[a] = 0.0
            self.state.psi[a] = 0.0

    def observe(
        self,
        card: HypothesisCard,
        val_bpb: float,
        embedding: np.ndarray,
        accepted: bool,
    ) -> bool:
        return False


class PartitionedNullCoordinator:
    """Fixed partition + attractor tracking, but zero coupling (alpha=0).

    Mirrors ``Coordinator``'s group structure so AC-B2/AC-B3 are measured on an
    identical footing, while the proposer stays uncoupled (sim.py passes no
    target/avoid and alpha=0). The only difference from SCAS is the control law.
    """

    coupling_T: float = 0.0

    def __init__(self, agents: list[str], categories: list[str]):
        if not agents or not categories:
            raise ValueError("agents and categories must both be non-empty")
        self.agent_to_group: dict[str, int] = {
            a: i % len(categories) for i, a in enumerate(agents)
        }
        self.group_to_category: dict[int, str] = dict(enumerate(categories))
        self.state = CoordinatorState()
        for g in range(len(categories)):
            self.state.groups[g] = [
                a for a, gg in self.agent_to_group.items() if gg == g
            ]
            self.state.group_best_val_bpb[g] = float("inf")
        for a in agents:
            self.state.alpha[a] = 0.0
            self.state.psi[a] = 0.0

    def observe(
        self,
        card: HypothesisCard,
        val_bpb: float,
        embedding: np.ndarray,
        accepted: bool,
    ) -> bool:
        """Track group-best attractor for measurement only; alpha stays 0."""
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
        # alpha/psi deliberately left at 0 — no coupling fed to the proposer.
        return improved
