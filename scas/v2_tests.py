"""Tests for the v2 coordinator (scas/v2.py). No model download, no GPU."""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

from scas.v2 import SwarmV2, diff_key, gain_reward, is_constant_edit


class FakeEmbedder:
    """Deterministic unit vectors; texts sharing a first word are near-identical."""

    def embed(self, text: str) -> np.ndarray:
        def vec(s):
            seed = int(hashlib.sha256(s.encode()).hexdigest()[:8], 16)
            v = np.random.default_rng(seed).normal(size=32)
            return v / np.linalg.norm(v)

        v = vec(text.split()[0]) + 0.05 * vec(text)
        return v / np.linalg.norm(v)


BASE = "A = 1\nB = 2\n"


def diff(old: str, new: str) -> str:
    return "--- a/train.py\n+++ b/train.py\n@@\n-%s\n+%s\n" % (old, new)


def hub(**kw) -> SwarmV2:
    h = SwarmV2(BASE, FakeEmbedder(), **kw)
    for box, v in (("b0", 1.0), ("b1", 1.1), ("b2", 1.2)):
        h.record_measurement(box, h.head, v)
    return h


def run(h, agent, box, code_change, val, card, kind="tune", base=None):
    """Claim + report one experiment built on ``base`` (default: current head)."""
    head, code, _ = h.head_state()
    base = base or head
    d = {"kind": kind, "target_id": None}
    c = h.claim(agent, box, base, diff(*code_change), card, d)
    assert "claim" in c, c
    new_code = h.codes[base].replace(*code_change)
    return h.report(c["claim"], val, new_code)


# ------------------------------------------------------------ small helpers

def test_diff_key_ignores_whitespace_and_order():
    a = "+X = 1\n-X = 2\n"
    b = "-X  =  2\n+X = 1\n"
    assert diff_key(a) == diff_key(b)
    assert diff_key(a) != diff_key("+X = 3\n-X = 2\n")


def test_constant_edit_detection():
    assert is_constant_edit("-LR = 0.1\n+LR = 0.2\n")
    assert not is_constant_edit("-LR = 0.1\n+    x = self.gate(x)\n")
    assert not is_constant_edit("")


def test_reward_is_bounded_and_monotone():
    assert gain_reward(-3) == 0.0
    assert 0 < gain_reward(0.2) < gain_reward(1.0) < gain_reward(10.0) <= 1.0


# ---------------------------------------------------------- shared incumbent

def test_keep_on_head_is_promoted_and_estimated_elsewhere():
    h = hub()
    out = run(h, "a0", "b0", ("A = 1", "A = 9"), 0.90, "Raised A")
    assert out["status"] == "keep" and out["promoted"]
    assert h.head_state()[1] == "A = 9\nB = 2\n"
    assert h.reference("b0") == (0.90, 0)
    ref, hops = h.reference("b1")          # never measured there: 1.1 * 0.90/1.0
    assert hops == 1 and ref == pytest.approx(0.99)


def test_discard_and_crash_leave_head_alone():
    h = hub()
    head = h.head
    assert run(h, "a0", "b0", ("A = 1", "A = 5"), 1.02, "Raised A")["status"] == "discard"
    assert run(h, "a1", "b1", ("B = 2", "B = 5"), None, "Raised B")["status"] == "crash"
    assert h.head == head


def test_estimated_reference_needs_the_wider_margin():
    h = hub(margin=0.0005, est_margin=0.0015)
    run(h, "a0", "b0", ("A = 1", "A = 9"), 0.90, "Raised A")
    # b1's reference for the new head is an estimate (0.99): 0.001 better is not enough
    assert run(h, "a1", "b1", ("B = 2", "B = 3"), 0.9890, "Raised B")["status"] == "discard"
    assert run(h, "a1", "b1", ("B = 2", "B = 4"), 0.9880, "Raised B more")["status"] == "keep"


def test_keep_on_stale_base_is_not_promoted():
    h = hub()
    old = h.head
    run(h, "a0", "b0", ("A = 1", "A = 9"), 0.90, "Raised A")
    out = run(h, "a1", "b1", ("B = 2", "B = 7"), 1.0, "Raised B", base=old)
    assert out["status"] == "keep" and not out["promoted"] and out["head_moved"]
    assert "B = 7" not in h.head_state()[1]


def test_merge_experiment_promotes_the_combination():
    h = hub()
    old = h.head
    run(h, "a0", "b0", ("A = 1", "A = 9"), 0.90, "Raised A")
    run(h, "a1", "b1", ("B = 2", "B = 7"), 1.0, "Raised B", base=old)
    out = run(h, "a1", "b1", ("B = 2", "B = 7"), 0.93, "merge: Raised B", kind="merge")
    assert out["promoted"]
    assert h.head_state()[1] == "A = 9\nB = 7\n"


# ------------------------------------------------------------- verification

def promote_three(h):
    run(h, "a0", "b0", ("A = 1", "A = 2"), 0.95, "Raised A")
    run(h, "a0", "b0", ("A = 2", "A = 3"), 0.90, "Raised A again")
    run(h, "a0", "b0", ("B = 2", "B = 3"), 0.85, "Raised B")


def test_verification_due_after_foreign_promotions():
    h = hub(verify_after=3)
    promote_three(h)
    assert not h.needs_verification("b0")       # it measured every step itself
    assert h.needs_verification("b1")
    v = h.verify("b1", h.head, 0.94)            # 1.1 -> 0.94 is a real gain here too
    assert v["ok"] and not h.needs_verification("b1")
    assert h.reference("b1") == (0.94, 0)


def test_failed_verification_retries_then_rolls_back():
    h = hub(verify_after=3)
    base = h.head
    promote_three(h)
    head = h.head
    v1 = h.verify("b1", head, 1.1)              # no gain at all on this box
    assert not v1["ok"] and v1["action"] == "retry" and h.head == head
    assert h.needs_verification("b1")
    v2 = h.verify("b1", head, 1.1)
    assert v2["action"] == "rolled_back" and h.head == base
    assert all(e["status"] == "rejected" for e in h.archive if e.get("new_hash"))
    # a rejected keep is no longer offered as a refine target
    assert h.next_directive("a1")["kind"] != "refine"


def test_crash_during_verification_counts_as_failure():
    h = hub(verify_after=3)
    promote_three(h)
    assert not h.verify("b1", h.head, None)["ok"]


# ------------------------------------------------------------------- dedup

def test_identical_edit_refused_including_one_still_running():
    h = hub()
    d = {"kind": "tune", "target_id": None}
    first = h.claim("a0", "b0", h.head, diff("A = 1", "A = 9"), "Raised A", d)
    assert "claim" in first
    dup = h.claim("a1", "b1", h.head, diff("A = 1", "A = 9"), "Bumped A for speed", d)
    assert dup["duplicate"]["reason"] == "identical edit"
    assert dup["duplicate"]["status"] == "running"
    assert "claim" in h.claim("a1", "b1", h.head, diff("A = 1", "A = 5"), "Raised A", d)


def test_similar_structural_card_refused_but_constant_edits_are_not():
    h = hub()
    d = {"kind": "structural", "target_id": None}
    assert "claim" in h.claim("a0", "b0", h.head, diff("x = f(x)", "x = g(x)"),
                              "Gated the MLP output", d)
    dup = h.claim("a1", "b1", h.head, diff("y = f(y)", "y = gate(y)"),
                  "Gated the attention output", d)
    assert dup["duplicate"]["reason"] == "near-identical idea"
    # same first word, but both are constant edits with different values: allowed
    assert "claim" in h.claim("a2", "b2", h.head, diff("A = 1", "A = 3"), "Gated nothing", d)


def test_old_edit_may_be_retried_after_the_head_moved_on():
    h = hub(dup_window=1)
    assert run(h, "a0", "b0", ("B = 2", "B = 5"), 1.05, "Raised B")["status"] == "discard"
    run(h, "a0", "b0", ("A = 1", "A = 2"), 0.95, "Raised A")
    d = {"kind": "tune", "target_id": None}
    assert "duplicate" in h.claim("a1", "b1", h.head, diff("B = 2", "B = 5"), "Raised B", d)
    run(h, "a0", "b0", ("A = 2", "A = 3"), 0.90, "Raised A again")
    assert "claim" in h.claim("a1", "b1", h.head, diff("B = 2", "B = 5"), "Raised B", d)


def test_merge_bypasses_dedup():
    h = hub()
    run(h, "a0", "b0", ("B = 2", "B = 5"), 1.05, "Raised B")
    d = {"kind": "merge", "target_id": None}
    assert "claim" in h.claim("a1", "b1", h.head, diff("B = 2", "B = 5"), "merge: Raised B", d)


# -------------------------------------------------------- bandit and targets

def test_bandit_spreads_first_pulls_and_then_follows_reward():
    h = hub()
    first = [h.next_directive("a%d" % i)["kind"] for i in range(2)]
    assert sorted(first) == ["structural", "tune"]      # no keep yet: refine unavailable
    for d in first:
        h.release({"kind": d})
    # tune pays, structural does not
    for i in range(6):
        run(h, "a0", "b0", ("A = %d" % (i + 1), "A = %d" % (i + 2)), 1.0 - 0.05 * (i + 1),
            "Raised A %d" % i, kind="tune")
        run(h, "a1", "b1", ("B = 2", "B = %d" % (i + 10)), 9.9, "Blockchange %d" % i,
            kind="structural")
    picks = []
    for _ in range(20):
        d = h.next_directive("a0")
        picks.append(d["kind"])
        h.release(d)
    assert picks.count("structural") < picks.count("tune") + picks.count("refine")


def test_release_returns_the_pending_pull():
    h = hub()
    d = h.next_directive("a0")
    assert h.bandit[d["kind"]]["pending"] == 1
    h.release(d)
    assert h.bandit[d["kind"]]["pending"] == 0


def test_refine_target_is_discounted_by_refinements():
    h = hub(seed=1)
    run(h, "a0", "b0", ("A = 1", "A = 2"), 0.90, "Alpha change")
    run(h, "a0", "b0", ("B = 2", "B = 3"), 0.81, "Beta change")
    for e in h.archive:
        e["gain_pct"] = 10.0                     # equal merit
    h.archive[0]["children"] = 50                # already refined to death
    h.bandit["tune"].update(n=30.0, sum=0.0)
    h.bandit["structural"].update(n=30.0, sum=0.0)
    h.bandit["refine"].update(n=1.0, sum=1.0)
    targets = []
    for _ in range(30):
        d = h.next_directive("a0")
        assert d["kind"] == "refine"
        targets.append(d["target"])
        h.release(d)
        h.archive[1]["children"] = 0             # keep the comparison static
        h.archive[0]["children"] = 50
    assert targets.count("Beta change") > 25


def test_history_and_snapshot_are_serialisable():
    import json

    h = hub()
    run(h, "a0", "b0", ("A = 1", "A = 9"), 0.90, "Raised A")
    run(h, "a1", "b1", ("B = 2", "B = 5"), None, "Raised B")
    text = h.history_text()
    assert "KEPT" in text and "crash" in text and "Raised A" in text
    snap = h.snapshot()
    json.dumps(snap)
    assert len(snap["chain"]) == 2 and snap["by_kind"]["tune"]["kept"] == 1
