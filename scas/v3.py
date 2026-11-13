"""SCAS v3 coordinator: the v2 shared version, judged and steered by a model.

Two proof runs (three RTX 4080 Super boxes, 2026-11-12) left v2 level with the
uncoordinated arms. What cost it: estimates of the shared version on another
box were 0.6-3.3% off, so close results were misjudged; 11 of 46 training runs
went to re-measurement; a fixed keep margin of 0.1% promoted noise on steady
boxes and was far too tight on drifting ones.

v3 keeps the v2 skeleton (one shared train.py, claims, duplicate check, refine
targets, bandit over directive kinds) and replaces everything that compared
numbers with the model in scas/surrogate.py:

1. **State is explicit.** An experiment is its constants, the structural
   changes it carries (flags) and the speed of its box. No embedding.

2. **One model instead of reference values.** Every result from every box is
   an observation. What a version scores on a box is the model's prediction
   there: no ratio estimates and no rollback, and a change that only helped on
   one box loses its estimated effect as other boxes report. The shared
   version is re-measured on a box only when the model is unsure of it there
   (predicted sd above ``sd_verify``) - often while the model is new, rarely
   once it has seen the hardware.

3. **Keep on probability, not on a margin.** A result is kept when the model,
   having seen it, gives at least ``p_keep`` that the change is a real
   improvement on that box. Between ``p_retest`` and ``p_keep`` it is run once
   more (on the current shared version) before deciding.

4. **Refuse what the model already knows will not help.** A constant-only
   proposal whose chance of improvement is below ``p_veto`` is sent back
   before it trains (at most ``max_vetoes`` times in a row per agent). Offline,
   9 of the 10 keeps such a rule would have refused were on a GPU type the
   memory had never seen, so the model may refuse only on a box whose speed
   lies inside the remembered range, or once this run has ``veto_min_obs``
   results of its own.

5. **The model proposes numeric steps itself.** A fourth directive kind,
   ``model``, edits one constant of the shared version to the candidate with
   the highest expected improvement - no agent call. The bandit decides how
   often, from the gains each kind produces.

6. **Memory across runs.** The model starts from the experiments of earlier
   runs (scas/data/v3_memory.jsonl); offline that is what made it useful
   (keep/discard AUC 0.91 with memory, 0.72 without).

Structural ideas still come only from the agents: the model treats a new
structural change as an unknown effect and learns it from its results.
"""

from __future__ import annotations

import math
import time

from scipy.stats import norm

from scas.surrogate import Surrogate, _num, complete, extract, featurize, set_constant
from scas.v2 import DIRECTIVE_TEXT, KINDS, SwarmV2, code_hash, gain_reward

V3_KINDS = KINDS + ("model",)

# (constant, how to step it): factors multiply, offsets add
STEPS = {
    "TOTAL_BATCH_SIZE": ("mul", (0.0625, 0.125, 0.25, 0.5, 2.0)),
    "EMBEDDING_LR": ("mul", (0.5, 0.7, 1.4, 2.0)),
    "UNEMBEDDING_LR": ("mul", (0.5, 0.7, 1.4, 2.0)),
    "MATRIX_LR": ("mul", (0.5, 0.7, 0.85, 1.2, 1.4, 2.0)),
    "SCALAR_LR": ("mul", (0.5, 0.7, 1.4, 2.0)),
    "WEIGHT_DECAY": ("mul", (0.25, 0.5, 0.7, 1.4, 2.0)),
    "BETA1": ("one_minus", (0.5, 2.0)),
    "BETA2": ("one_minus", (0.2, 0.5, 2.0)),
    "WARMUP_RATIO": ("add", (0.02, 0.05, -0.02)),
    "WARMDOWN_RATIO": ("add", (-0.2, -0.1, 0.1, 0.2, 0.3)),
    "FINAL_LR_FRAC": ("add", (0.02, 0.05, 0.1, -0.05)),
}
LIMITS = {"TOTAL_BATCH_SIZE": (2 ** 15, 2 ** 21), "WARMUP_RATIO": (0.0, 0.3),
          "WARMDOWN_RATIO": (0.0, 0.95), "FINAL_LR_FRAC": (0.0, 0.5),
          "BETA1": (0.5, 0.98), "BETA2": (0.8, 0.999)}


def _r(v: float) -> float:
    """Four significant digits for tuning constants; integers such as the batch size exactly."""
    return float(v) if abs(v) >= 1000 else float("%.4g" % v)


def _key(consts: dict, flags) -> tuple:
    return tuple(round(v, 6) for v in featurize(consts)) + (tuple(sorted(flags)),)


class SwarmV3(SwarmV2):
    def __init__(self, base_code: str, embedder, surrogate: Surrogate | None = None,
                 memory: list[dict] | None = None, p_keep: float = 0.8, p_retest: float = 0.5,
                 p_veto: float = 0.05, max_vetoes: int = 2, veto_min_obs: int = 20,
                 sd_verify: float = 0.01, **kw):
        kw.setdefault("sct", None)
        super().__init__(base_code, embedder, **kw)
        self.model = surrogate if surrogate is not None else Surrogate()
        if memory:
            self.model.add_memory(memory)
        self.p_keep, self.p_retest, self.p_veto = p_keep, p_retest, p_veto
        self.max_vetoes = max_vetoes
        self.veto_min_obs = veto_min_obs
        self.sd_verify = sd_verify
        self.verified: set[tuple] = set()        # (box, version) measured or given up on
        self.memory_speed = (min(self.model.S), max(self.model.S)) if len(self.model) else None
        self.run_obs = 0
        consts, rest = extract(base_code)
        if not complete(consts):
            raise ValueError("train.py lacks constants the model needs: %s" % sorted(consts))
        self.consts = {self.head: consts}
        self.rests = {self.head: rest}
        self.flags = {self.head: frozenset()}
        self.crashed: set[tuple] = set()
        self.vetoes: dict[str, int] = {}
        self._next_change = 1
        self.bandit = {k: {"n": 0.0, "sum": 0.0, "pending": 0} for k in V3_KINDS}

    # ------------------------------------------ what v2 measured, v3 predicts

    def record_measurement(self, box: str, h: str, val: float, steps: float | None = None) -> None:
        with self.lock:
            self.refs.setdefault(box, {})[h] = float(val)
            self.verified.add((box, h))
            self.model.add(self.consts[h], self.flags[h], box, val, steps)
            self.run_obs += 1
            self._event("measured", box=box, version=h, val=val, steps=steps)

    def may_veto(self, box: str) -> bool:
        """Does the model know this hardware well enough to refuse a proposal?"""
        if self.run_obs >= self.veto_min_obs:
            return True
        s = self.model.box_speed(box)
        return (s is not None and self.memory_speed is not None
                and self.memory_speed[0] - 0.1 <= s <= self.memory_speed[1] + 0.1)

    def reference(self, box: str, h: str | None = None) -> tuple[float | None, int]:
        with self.lock:
            h = self.head if h is None else h
            return self.model.expected(self.consts[h], self.flags[h], box)[0], 0

    def needs_verification(self, box: str) -> bool:
        """Measure the shared version on this box when the model is unsure of it."""
        with self.lock:
            if (box, self.head) in self.verified:
                return False
            return self.model.expected(self.consts[self.head], self.flags[self.head],
                                       box)[1] > self.sd_verify

    def verify(self, box: str, h: str, val: float | None, steps: float | None = None) -> dict:
        with self.lock:
            expected, sd = self.model.expected(self.consts[h], self.flags[h], box)
            self.verified.add((box, h))          # also after a crash: do not loop on it
            if val is not None:
                self.refs.setdefault(box, {})[h] = float(val)
                self.model.add(self.consts[h], self.flags[h], box, val, steps)
                self.run_obs += 1
            out = {"version": h, "expected": expected, "measured": val, "ok": val is not None,
                   "expected_sd_pct": 100.0 * sd}
            if val is not None:
                out["gap_pct"] = 100.0 * (val / expected - 1.0)
            self._event("verify", box=box, **out)
            return out

    def ablation_directive(self, box: str) -> dict | None:
        return None

    # ------------------------------------------------------------ directives

    def next_directive(self, agent: str, box: str | None = None) -> dict:
        with self.lock:
            targets = [e for e in self.archive
                       if e["status"] == "keep" and e["gain_pct"] > 0
                       and e["kind"] in ("tune", "structural", "refine", "model")]
            cand = self._model_candidate(box) if box is not None else None
            arms = [k for k in V3_KINDS
                    if (k != "refine" or targets) and (k != "model" or cand)]
            total = sum(self.bandit[k]["n"] + self.bandit[k]["pending"] for k in arms)
            best, best_score = None, -1.0
            for k in arms:
                b = self.bandit[k]
                n = b["n"] + b["pending"]
                if n < 1e-9:
                    best = k
                    break
                score = b["sum"] / n + self.ucb_c * math.sqrt(math.log(total + 1.0) / n)
                if score > best_score:
                    best, best_score = k, score
            self.bandit[best]["pending"] += 1
            d = {"kind": best, "target": None, "target_id": None}
            if best == "model":
                d.update(cand)
            elif best == "refine":
                import numpy as np

                med = float(np.median([e["gain_pct"] for e in targets]))
                w = [1.0 / (1.0 + math.exp(-(e["gain_pct"] - med))) / (1.0 + e["children"])
                     for e in targets]
                t = self.rng.choices(targets, weights=w, k=1)[0]
                t["children"] += 1
                d.update(target=t["card"], target_id=t["id"],
                         text=DIRECTIVE_TEXT["refine"].format(gain=t["gain_pct"], card=t["card"]))
            else:
                d["text"] = DIRECTIVE_TEXT[best]
            self._event("directive", agent=agent, kind=best, target_id=d["target_id"],
                        description=d.get("description"))
            return d

    def _candidates(self, c0: dict):
        for name, (how, vals) in STEPS.items():
            lo, hi = LIMITS.get(name, (1e-6, 1e6))
            for v in vals:
                if how == "mul":
                    new = c0[name] * v
                elif how == "add":
                    new = c0[name] + v
                else:
                    new = 1.0 - (1.0 - c0[name]) * v
                new = _r(min(max(new, lo), hi))
                if abs(new - c0[name]) > 1e-9:
                    yield name, new

    def _model_candidate(self, box: str) -> dict | None:
        """The single-constant step from the shared version with the highest
        expected improvement on ``box`` that nobody has tried or is running."""
        c0, f0, code = self.consts[self.head], self.flags[self.head], self.codes[self.head]
        idx = self.versions[self.head]["index"]
        done = list(self.inflight.values()) + self.archive
        tried = self.crashed | {_key(e["consts"], e["flags"]) for e in done if e.get("consts")}
        # the same single step on a version just before this one counts as tried too
        recent = {e["delta"] for e in done
                  if e.get("delta") and idx - e["base_index"] <= self.dup_window}
        cands = []
        for name, new in self._candidates(c0):
            c = dict(c0)
            c[name] = new
            if _key(c, f0) not in tried and frozenset({(name, new)}) not in recent:
                cands.append((name, new, c))
        if not cands:
            return None
        s = self.model.box_speed(box)
        s = self.model.p["speed_mean"] if s is None else s
        mu, cov = self.model.predict([(c, f0, s) for _, _, c in cands] + [(c0, f0, s)])
        b = len(cands)
        best, best_ei = None, -1.0
        for i, (name, new, c) in enumerate(cands):
            d = mu[b] - mu[i]
            sd = math.sqrt(max(cov[i, i] + cov[b, b] - 2 * cov[i, b], 1e-12))
            z = d / sd
            ei = sd * (z * norm.cdf(z) + norm.pdf(z))
            if ei > best_ei:
                best, best_ei = (name, new, d, sd), ei
        name, new, d, sd = best
        new_code = set_constant(code, name, new)
        if new_code is None:
            return None
        shown = name if name not in ("BETA1", "BETA2") else "ADAM_BETAS %s" % name.lower()
        desc = ("Model-chosen step: %s from %s to %s (the team's model expects %+.2f%% "
                "+/- %.2f%%)" % (shown, _num(c0[name]), _num(new),
                                 100 * (1 - math.exp(-d)), 100 * sd))
        return {"new_code": new_code, "description": desc, "text": desc,
                "expected_gain_pct": 100 * (1 - math.exp(-d))}

    # ----------------------------------------------------------------- claim

    def claim(self, agent: str, box: str, base: str, diff: str, card: str,
              directive: dict, new_code: str | None = None) -> dict:
        with self.lock:
            r = super().claim(agent, box, base, diff, card, directive)
            if "claim" not in r:
                return r
            e = self.inflight[r["claim"]]
            c0, f0 = self.consts[base], self.flags[base]
            consts, rest = extract(new_code) if new_code is not None else (dict(c0), self.rests[base])
            consts = {**c0, **consts}             # a constant the edit removed keeps its old value
            structural = rest != self.rests[base]
            change = directive.get("change_id")
            if change is None:
                change = "c%d" % self._next_change
                self._next_change += 1
            flags = f0 | {change} if (structural or directive.get("structural")) else f0
            delta = frozenset((k, _r(v)) for k, v in consts.items() if c0.get(k) != v)
            e.update(consts=consts, flags=flags, rest=rest, change_id=change, delta=delta,
                     structural=bool(structural or directive.get("structural")))
            kind = directive.get("kind")
            if kind in ("tune", "refine") and not e["structural"] and self.may_veto(box):
                cmp = self.model.compare((consts, flags), (c0, f0), box)
                if cmp["p_better"] < self.p_veto and self.vetoes.get(agent, 0) < self.max_vetoes:
                    del self.inflight[r["claim"]]
                    self.vetoes[agent] = self.vetoes.get(agent, 0) + 1
                    msg = ("Your previous proposal was refused before running: from all results "
                           "so far, the team's model expects it to change val_bpb by %+.2f%% "
                           "(+/- %.2f%%), i.e. not to help:\n    \"%s\"\nPropose something else."
                           % (-cmp["gain_pct"], cmp["sd_pct"], card))
                    dup = {"card": card, "status": "predicted %+.2f%%" % cmp["gain_pct"],
                           "reason": "model veto", "message": msg}
                    self._event("veto", agent=agent, box=box, proposed=card,
                                p_better=cmp["p_better"], gain_pct=cmp["gain_pct"],
                                sd_pct=cmp["sd_pct"])
                    return {"duplicate": dup}
                e["p_prior"] = cmp["p_better"]
            self.vetoes[agent] = 0
            return r

    # ---------------------------------------------------------------- report

    def report(self, claim_id: int, val: float | None, new_code: str | None = None,
               steps: float | None = None) -> dict:
        with self.lock:
            e = self.inflight.pop(claim_id)
            box, base, kind = e["box"], e["base"], e["kind"]
            new, old = (e["consts"], e["flags"]), (self.consts[base], self.flags[base])
            out = {"ref_estimated_hops": 0, "promoted": False, "head_moved": base != self.head,
                   "rechecked": False, "change_id": e["change_id"], "structural": e["structural"]}
            if val is None:
                self.crashed.add(_key(*new))
                e.update(status="crash", gain_pct=0.0, val=None, p_better=0.0, model_gain_pct=0.0)
                out["ref"] = self.model.expected(old[0], old[1], box)[0]
            else:
                self.model.add(new[0], new[1], box, val, steps)
                self.run_obs += 1
                cmp = self.model.compare(new, old, box)
                ref = cmp["base_val"]
                p = cmp["p_better"]
                if p >= self.p_keep:
                    status = "keep"
                elif p >= self.p_retest and kind not in ("merge", "combine"):
                    status = "uncertain"         # run it once more before deciding
                else:
                    status = "discard"
                e.update(status=status, val=float(val), gain_pct=100.0 * (ref - val) / ref,
                         p_better=p, model_gain_pct=cmp["gain_pct"])
                out["ref"] = ref
                if status == "keep" and base == self.head and new_code is not None:
                    h = code_hash(new_code)
                    if h not in self.versions:
                        self.codes[h] = new_code
                        self.versions[h] = {"parent": base, "ratio": 1.0 - cmp["gain_pct"] / 100.0,
                                            "index": self.versions[base]["index"] + 1,
                                            "desc": e["card"], "by": box}
                        self.consts[h], self.flags[h], self.rests[h] = new[0], new[1], e["rest"]
                    self.head = h
                    e["new_hash"] = h
                    out["promoted"] = True
            e["children"] = 0
            e["t_done"] = time.time()
            if kind in self.bandit:
                b = self.bandit[kind]
                b["pending"] = max(0, b["pending"] - 1)
                for k in V3_KINDS:
                    self.bandit[k]["n"] *= self.discount
                    self.bandit[k]["sum"] *= self.discount
                b["n"] += 1.0
                b["sum"] += gain_reward(e["model_gain_pct"] if e["status"] == "keep" else 0.0)
            self.archive.append(e)
            out.update(status=e["status"], gain_pct=e["gain_pct"], kind=kind, box=box,
                       p_better=e["p_better"], model_gain_pct=e["model_gain_pct"],
                       head=self.head, head_index=self.versions[self.head]["index"])
            self._event("report", **{k: v for k, v in e.items() if k not in ("z", "flags", "delta")},
                        flags=sorted(e["flags"]), promoted=out["promoted"], head=self.head,
                        steps=steps)
            return out

    def adopt(self, claim_ref: dict, expected_head: str, new_code: str, card: str) -> bool:
        with self.lock:
            before = self.head
            if not super().adopt(claim_ref, expected_head, new_code, card):
                return False
            h = self.head
            consts, rest = extract(new_code)
            self.consts[h] = {**self.consts[before], **consts}
            self.rests[h] = rest
            self.flags[h] = (self.flags[before] | {claim_ref["change_id"]}
                             if claim_ref.get("structural") else self.flags[before])
            return True

    # -------------------------------------------------------------- snapshot

    def snapshot(self) -> dict:
        with self.lock:
            s = super().snapshot()
            s["by_kind"]["model"] = {
                "n": sum(1 for e in self.archive if e["kind"] == "model"),
                "kept": sum(1 for e in self.archive
                            if e["kind"] == "model" and e["status"] == "keep")}
            s["model"] = {"observations": len(self.model),
                          "box_speed": {b: self.model.box_speed(b) for b in self.refs},
                          "head_expected": {b: self.reference(b)[0] for b in self.refs}}
            return s
