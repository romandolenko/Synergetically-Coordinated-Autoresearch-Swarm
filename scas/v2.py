"""SCAS v2 coordinator: the four changes the S2 run of 2026-11-12 asked for.

In that run the v1 coordinator finished far behind three uncoordinated arms.
Each mechanism here answers one measured failure:

1. **Shared incumbent with merge-and-verify.** Agents kept separate branches,
   so a win had to be re-found by every agent (11-14 keeps re-found per arm)
   and wins from different agents were never combined. Here there is one
   shared train.py. A kept experiment built on the current head is promoted
   to it; one built on an older head is re-tested on the new head (merge).
   Boxes differ in speed, so a box that has not measured the head judges
   against an estimate (its last measurement times the promoters' ratios) and
   re-measures the head after ``verify_after`` foreign promotions. A head that
   fails that check twice on a box is rolled back.

2. **Global duplicate check before a run.** 75 of 164 experiments repeated an
   identical edit. A proposal is refused, before any GPU time is spent, if the
   same diff was tried on a recent head, or if it is a structural edit whose
   card is nearly the same as a recent one. Experiments still training count.

3. **Refine targets with a refinement discount.** One agent was sent to the
   same card seven times. A target is drawn with weight
   ``sigmoid(gain - median gain) / (1 + times already refined)`` (the
   ShinkaEvolve parent-sampling rule), and the directive wording no longer
   forbids hyperparameter changes.

4. **A bandit over directive kinds instead of the psi -> alpha gate.** Alpha
   sat at 0.65-0.77 whatever the agents did. A discounted UCB bandit chooses
   between ``tune``, ``structural`` and ``refine`` from the gains each kind
   has recently produced, shared by all agents.

The SCT law (emergent groups, psi) still runs on every report and its output
is logged, but v2 guidance does not depend on it.

Pure logic, thread-safe, no I/O beyond an optional event log. The driver
(tools/vast/swarm.py, arm ``scas2``) owns the files and the training runs.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
import threading
import time
from pathlib import Path

import numpy as np

KINDS = ("tune", "structural", "refine")

DIRECTIVE_TEXT = {
    "tune": (
        "DIRECTIVE: TUNE\n"
        "Change the value of ONE hyperparameter constant or schedule setting: the "
        "one you now expect to matter most, given the results above."
    ),
    "structural": (
        "DIRECTIVE: STRUCTURAL\n"
        "Change code, not just the value of a constant: a model component, the "
        "optimizer logic or the training loop. One idea only."
    ),
    "refine": (
        "DIRECTIVE: REFINE\n"
        "This earlier change improved val_bpb by {gain:.2f}%:\n"
        '    "{card}"\n'
        "Push the same idea further or adjust it: the next step in the same "
        "direction, or a closely related setting of the same mechanism."
    ),
    "combine": (
        "DIRECTIVE: RE-APPLY\n"
        "This change improved an OLDER version of train.py by {gain:.2f}% but could "
        "not be merged automatically into the current version:\n"
        '    "{card}"\n'
        "Re-implement it on the current train.py."
    ),
}


def code_hash(code: str) -> str:
    return hashlib.sha256(code.replace("\r\n", "\n").encode("utf-8")).hexdigest()[:16]


def diff_key(diff: str) -> str:
    """Identity of an edit: its added and removed lines, whitespace-insensitive."""
    lines = sorted(
        ln[0] + " ".join(ln[1:].split())
        for ln in diff.splitlines()
        if ln[:1] in "+-" and not ln.startswith(("+++", "---")) and ln[1:].strip()
    )
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()[:16]


_CONST = re.compile(r"^[+-]\s*[A-Z][A-Z0-9_]*\s*=")


def is_constant_edit(diff: str) -> bool:
    """True when every changed line assigns a module-level constant."""
    changed = [ln for ln in diff.splitlines()
               if ln[:1] in "+-" and not ln.startswith(("+++", "---")) and ln[1:].strip()]
    return bool(changed) and all(_CONST.match(ln) for ln in changed)


def gain_reward(gain_pct: float) -> float:
    """Bounded bandit reward: 0 for no gain, 0.63 for 1%, about 1 for 5%+."""
    return 1.0 - math.exp(-max(0.0, gain_pct))


class SwarmV2:
    def __init__(
        self,
        base_code: str,
        embedder,
        margin: float = 0.0005,
        est_margin: float = 0.0015,
        verify_after: int = 3,
        dup_cos: float = 0.86,
        dup_window: int = 2,
        ucb_c: float = 0.5,
        discount: float = 0.95,
        seed: int = 0,
        event_log: str | Path | None = None,
        sct=None,
    ):
        self.embedder = embedder
        self.margin = margin
        self.est_margin = est_margin
        self.verify_after = verify_after
        self.dup_cos = dup_cos
        self.dup_window = dup_window
        self.ucb_c = ucb_c
        self.discount = discount
        self.rng = random.Random(seed)
        self.lock = threading.RLock()
        self.event_log = Path(event_log) if event_log else None
        self.sct = sct

        h0 = code_hash(base_code)
        self.codes: dict[str, str] = {h0: base_code}
        # hash -> how that version came to be (the base has no parent)
        self.versions: dict[str, dict] = {h0: {"parent": None, "ratio": 1.0, "index": 0,
                                               "desc": "baseline", "by": None}}
        self.head = h0
        self.rejected: set[str] = set()
        self.refs: dict[str, dict[str, float]] = {}      # box -> hash -> measured val
        self.disputed: dict[str, str] = {}               # box -> head it failed once
        self.archive: list[dict] = []                    # finished experiments
        self.inflight: dict[int, dict] = {}              # claim id -> experiment
        self._next_claim = 1
        self.bandit = {k: {"n": 0.0, "sum": 0.0, "pending": 0} for k in KINDS}

    # ------------------------------------------------------------------ util

    def _event(self, _name: str, **payload) -> None:
        if self.event_log is None:
            return
        with self.event_log.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"t": time.time(), "event": _name, **payload}) + "\n")

    def head_state(self) -> tuple[str, str, int]:
        with self.lock:
            return self.head, self.codes[self.head], self.versions[self.head]["index"]

    # ------------------------------------------------------- reference values

    def record_measurement(self, box: str, h: str, val: float) -> None:
        with self.lock:
            self.refs.setdefault(box, {})[h] = float(val)
            self._event("measured", box=box, version=h, val=val)

    def reference(self, box: str, h: str | None = None) -> tuple[float | None, int]:
        """What version ``h`` scores on ``box``: measured (0 hops), or the last
        measurement on that box times the ratios of the promotions since."""
        with self.lock:
            cur = self.head if h is None else h
            mine = self.refs.get(box, {})
            factor, hops = 1.0, 0
            while cur is not None and cur not in mine:
                v = self.versions[cur]
                factor *= v["ratio"]
                cur = v["parent"]
                hops += 1
            if cur is None:
                return None, hops
            return mine[cur] * factor, hops

    def _anchor(self, box: str, h: str) -> str | None:
        """Nearest ancestor of ``h`` (excluding ``h``) measured on ``box``."""
        mine = self.refs.get(box, {})
        cur = self.versions[h]["parent"]
        while cur is not None and cur not in mine:
            cur = self.versions[cur]["parent"]
        return cur

    def needs_verification(self, box: str) -> bool:
        with self.lock:
            if self.disputed.get(box) == self.head:
                return True
            _, hops = self.reference(box)
            return hops >= self.verify_after

    def verify(self, box: str, h: str, val: float | None) -> dict:
        """``box`` trained version ``h`` (None = it crashed there). The chain
        since the box's last measurement must be a real improvement on this
        box too. First failure: disputed, measure again. Second: roll back."""
        with self.lock:
            anchor = self._anchor(box, h)
            expected, _ = self.reference(box, h)
            if val is not None:
                self.refs.setdefault(box, {})[h] = float(val)
            out = {"version": h, "expected": expected, "measured": val, "anchor": anchor}
            if h != self.head or anchor is None:
                out["ok"] = True          # head moved on, or nothing to compare with
                self._event("verify", box=box, **out)
                return out
            ok = val is not None and val < self.refs[box][anchor] - self.margin
            out["ok"] = ok
            if ok:
                self.disputed.pop(box, None)
            elif self.disputed.get(box) != h:
                self.disputed[box] = h
                if val is not None:
                    del self.refs[box][h]     # the retry replaces this reading
                out["action"] = "retry"
            else:
                self.disputed.pop(box, None)
                cur = h
                while cur != anchor:
                    self.rejected.add(cur)
                    cur = self.versions[cur]["parent"]
                self.head = anchor
                for e in self.archive:
                    if e.get("new_hash") in self.rejected:
                        e["status"] = "rejected"
                out["action"] = "rolled_back"
                out["head"] = anchor
            self._event("verify", box=box, **out)
            return out

    # ------------------------------------------------------------ directives

    def next_directive(self, agent: str) -> dict:
        with self.lock:
            targets = [e for e in self.archive if e["status"] == "keep" and e["gain_pct"] > 0]
            arms = [k for k in KINDS if k != "refine" or targets]
            total = sum(self.bandit[k]["n"] + self.bandit[k]["pending"] for k in arms)
            best, best_score = None, -1.0
            for k in arms:
                b = self.bandit[k]
                n = b["n"] + b["pending"]
                if n < 1e-9:
                    best = k
                    break
                mean = b["sum"] / n          # a pending pull counts as a zero so far
                score = mean + self.ucb_c * math.sqrt(math.log(total + 1.0) / n)
                if score > best_score:
                    best, best_score = k, score
            self.bandit[best]["pending"] += 1
            d = {"kind": best, "target": None, "target_id": None}
            if best == "refine":
                med = float(np.median([e["gain_pct"] for e in targets]))
                w = [1.0 / (1.0 + math.exp(-(e["gain_pct"] - med))) / (1.0 + e["children"])
                     for e in targets]
                t = self.rng.choices(targets, weights=w, k=1)[0]
                t["children"] += 1
                d.update(target=t["card"], target_id=t["id"],
                         text=DIRECTIVE_TEXT["refine"].format(gain=t["gain_pct"], card=t["card"]))
            else:
                d["text"] = DIRECTIVE_TEXT[best]
            self._event("directive", agent=agent, kind=best, target_id=d["target_id"])
            return d

    def combine_directive(self, card: str, gain_pct: float) -> dict:
        return {"kind": "combine", "target": card, "target_id": None,
                "text": DIRECTIVE_TEXT["combine"].format(gain=gain_pct, card=card)}

    def release(self, directive: dict) -> None:
        """A directive that never became an experiment (proposal failed)."""
        with self.lock:
            k = directive.get("kind")
            if k in self.bandit and self.bandit[k]["pending"] > 0:
                self.bandit[k]["pending"] -= 1

    # --------------------------------------------------------------- dedup

    def claim(self, agent: str, box: str, base: str, diff: str, card: str,
              directive: dict) -> dict:
        """Register an experiment before it trains. Returns ``{"claim": id}``
        or ``{"duplicate": {...}}`` when the idea was already tried recently."""
        with self.lock:
            kind = directive.get("kind")
            key = diff_key(diff)
            const = is_constant_edit(diff)
            z = self.embedder.embed(card)
            base_idx = self.versions[base]["index"] if base in self.versions else 0
            if kind not in ("merge", "combine"):
                for e in list(self.inflight.values()) + self.archive:
                    recent = base_idx - e["base_index"] <= self.dup_window
                    same_edit = e["key"] == key and (e["base"] == base or recent)
                    similar = (not const and not e["constant"] and recent
                               and float(np.dot(z, e["z"])) >= self.dup_cos)
                    if same_edit or similar:
                        dup = {"card": e["card"], "status": e.get("status", "running"),
                               "reason": "identical edit" if same_edit else "near-identical idea"}
                        self._event("duplicate", agent=agent, proposed=card, **dup)
                        return {"duplicate": dup}
            cid = self._next_claim
            self._next_claim += 1
            self.inflight[cid] = {
                "id": cid, "agent": agent, "box": box, "base": base, "base_index": base_idx,
                "key": key, "constant": const, "card": card, "z": z, "kind": kind,
                "target_id": directive.get("target_id"), "t_claim": time.time(),
            }
            return {"claim": cid}

    # --------------------------------------------------------------- report

    def report(self, claim_id: int, val: float | None, new_code: str | None = None) -> dict:
        """Outcome of a claimed experiment. ``val`` None means it crashed."""
        with self.lock:
            e = self.inflight.pop(claim_id)
            box, base, kind = e["box"], e["base"], e["kind"]
            ref, hops = self.reference(box, base)
            out = {"ref": ref, "ref_estimated_hops": hops, "promoted": False,
                   "head_moved": base != self.head}
            if val is None or ref is None:
                e.update(status="crash", gain_pct=0.0, val=None)
            else:
                thr = self.margin if hops == 0 else self.est_margin
                keep = val < ref - thr
                e.update(status="keep" if keep else "discard", val=float(val),
                         gain_pct=100.0 * (ref - val) / ref)
                if keep and base == self.head and new_code is not None:
                    h = code_hash(new_code)
                    if h not in self.versions:
                        self.codes[h] = new_code
                        self.versions[h] = {"parent": base, "ratio": val / ref,
                                            "index": self.versions[base]["index"] + 1,
                                            "desc": e["card"], "by": box}
                    self.refs.setdefault(box, {})[h] = float(val)
                    self.rejected.discard(h)
                    self.head = h
                    e["new_hash"] = h
                    out["promoted"] = True
            e["children"] = 0
            e["t_done"] = time.time()
            if kind in self.bandit:
                b = self.bandit[kind]
                b["pending"] = max(0, b["pending"] - 1)
                for k in KINDS:                      # forget old evidence everywhere
                    self.bandit[k]["n"] *= self.discount
                    self.bandit[k]["sum"] *= self.discount
                b["n"] += 1.0
                b["sum"] += gain_reward(e["gain_pct"] if e["status"] == "keep" else 0.0)
            if self.sct is not None and e["status"] != "crash":
                try:
                    from scas.coordinator import HypothesisCard

                    self.sct.observe(
                        HypothesisCard(e["agent"], len(self.archive), e["card"], "emergent", None),
                        -e["gain_pct"], e["z"], accepted=True, can_lead=e["status"] == "keep")
                    e["sct_group"] = self.sct.agent_to_group.get(e["agent"])
                    e["sct_alpha"] = self.sct.state.alpha.get(e["agent"])
                except Exception:  # noqa: BLE001 - telemetry must not break the run
                    pass
            self.archive.append(e)
            out.update(status=e["status"], gain_pct=e["gain_pct"], kind=kind,
                       head=self.head, head_index=self.versions[self.head]["index"])
            self._event("report", **{k: v for k, v in e.items() if k != "z"},
                        promoted=out["promoted"], head=self.head)
            return out

    # -------------------------------------------------------------- history

    def history_text(self) -> str:
        """The shared log every agent sees. Gains are relative to the version
        the experiment was built on, so they are comparable across boxes."""
        with self.lock:
            if not self.archive and not self.inflight:
                return "No experiments yet."
            lines = ["Team experiment log (oldest first). Change is measured against the "
                     "version of train.py the experiment was built on; negative is worse.",
                     "  status        change  description"]
            for e in self.archive:
                st = {"keep": "KEPT" if e.get("new_hash") else "kept-not-merged",
                      "discard": "discard", "crash": "crash",
                      "rejected": "withdrawn"}[e["status"]]
                ch = "   -   " if e["status"] == "crash" else "%+6.2f%%" % e["gain_pct"]
                lines.append("  %-13s %s  %s" % (st, ch, e["card"]))
            for e in self.inflight.values():
                lines.append("  %-13s    ?    %s" % ("running", e["card"]))
            lines.append("train.py is the shared team version: it contains every change "
                         "marked KEPT, from all researchers.")
            return "\n".join(lines)

    def snapshot(self) -> dict:
        with self.lock:
            chain, cur = [], self.head
            while cur is not None:
                v = self.versions[cur]
                chain.append({"hash": cur, "index": v["index"], "by": v["by"],
                              "ratio": v["ratio"], "desc": v["desc"]})
                cur = v["parent"]
            return {
                "head": self.head,
                "chain": chain[::-1],
                "rejected": sorted(self.rejected),
                "refs": self.refs,
                "bandit": {k: dict(v) for k, v in self.bandit.items()},
                "experiments": len(self.archive),
                "by_kind": {k: {"n": sum(1 for e in self.archive if e["kind"] == k),
                                "kept": sum(1 for e in self.archive
                                            if e["kind"] == k and e["status"] == "keep")}
                            for k in KINDS + ("merge", "combine")},
                "refine_targets": [{"card": e["card"], "gain_pct": e["gain_pct"],
                                    "children": e["children"]}
                                   for e in self.archive if e.get("children")],
            }
