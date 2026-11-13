"""Tests for the v3 model (scas/surrogate.py) and coordinator (scas/v3.py).
No model download, no GPU: a synthetic val_bpb function stands in for training."""

from __future__ import annotations

import hashlib
import json
import math

import numpy as np

from scas.surrogate import DEFAULT_PARAMS, Surrogate, complete, extract, set_constant
from scas.v3 import SwarmV3

BASE = """import torch
ASPECT_RATIO = 64
HEAD_DIM = 128
TOTAL_BATCH_SIZE = 2 ** 19
EMBEDDING_LR = 0.6
UNEMBEDDING_LR = 0.004
MATRIX_LR = 0.04   # Muon
SCALAR_LR = 0.5
WEIGHT_DECAY = 0.2
ADAM_BETAS = (0.8, 0.95)
WARMUP_RATIO = 0.0
WARMDOWN_RATIO = 0.5
FINAL_LR_FRAC = 0.0
DEPTH = 8
DEVICE_BATCH_SIZE = 16
def mlp(x):
    return relu(x)
"""
SPEED = {"b0": 62.0, "b1": 75.0, "b2": 60.0}


class FakeEmbedder:
    dim = 16

    def embed(self, text: str) -> np.ndarray:
        h = hashlib.sha256(text.encode("utf-8")).digest()
        v = np.frombuffer(h[:16], dtype=np.uint8).astype(np.float64) - 127.5
        return v / np.linalg.norm(v)


def world(code: str, box: str) -> tuple[float, int]:
    """val_bpb and steps of a train.py on a box: more steps help, MATRIX_LR has
    an optimum at 0.06, gelu helps 2%, everything else does nothing."""
    c, _ = extract(code)
    steps = SPEED[box] * (2 ** 19 / c["TOTAL_BATCH_SIZE"]) ** 0.94
    val = 0.70 * (steps / 62.0) ** -0.25
    val *= 1.0 + 0.5 * math.log(c["MATRIX_LR"] / 0.06) ** 2 * 0.1
    if "gelu" in code:
        val *= 0.98
    return val, round(steps)


def hub(**kw) -> SwarmV3:
    h = SwarmV3(BASE, FakeEmbedder(), surrogate=Surrogate(dict(DEFAULT_PARAMS)), **kw)
    for box in SPEED:
        val, steps = world(BASE, box)
        h.record_measurement(box, h.head, val, steps=steps)
    return h


def diff(old: str, new: str) -> str:
    return "--- a/train.py\n+++ b/train.py\n@@\n-%s\n+%s\n" % (old, new)


def run(h, agent, box, change, card, kind="tune", val=None, extra=None):
    head, code, _ = h.head_state()
    if h.needs_verification(box):            # as the driver does before an experiment
        v, steps = world(code, box)
        h.verify(box, head, v, steps=steps)
    new_code = code.replace(*change)
    d = {"kind": kind, "target_id": None, **(extra or {})}
    c = h.claim(agent, box, head, diff(*change), card, d, new_code=new_code)
    if "claim" not in c:
        return c
    v, steps = world(new_code, box)
    return h.report(c["claim"], v if val is None else val, new_code, steps=steps)


# ------------------------------------------------------------------ settings

def test_extract_and_set_constant_round_trip():
    c, rest = extract(BASE)
    assert complete(c) and c["TOTAL_BATCH_SIZE"] == 2 ** 19 and c["BETA2"] == 0.95
    code = set_constant(BASE, "TOTAL_BATCH_SIZE", 2 ** 17)
    assert "TOTAL_BATCH_SIZE = 2 ** 17" in code
    code = set_constant(code, "MATRIX_LR", 0.056)
    assert "MATRIX_LR = 0.056   # Muon" in code          # the comment survives
    code = set_constant(code, "BETA2", 0.99)
    c2, rest2 = extract(code)
    assert (c2["TOTAL_BATCH_SIZE"], c2["MATRIX_LR"], c2["BETA1"], c2["BETA2"]) == (2 ** 17, 0.056, 0.8, 0.99)
    assert rest2 == rest                                   # only constants changed
    assert extract(BASE.replace("relu", "gelu"))[1] != rest


def test_box_speed_follows_the_reported_steps():
    s = Surrogate(dict(DEFAULT_PARAMS))
    c, _ = extract(BASE)
    for steps in (60, 61, 60):
        s.add(c, (), "b", 0.70, steps)
    before = s.box_speed("b")
    # code that makes different steps (halved batch, a structural change) says
    # nothing about the box the first time it runs ...
    c2 = dict(c, TOTAL_BATCH_SIZE=2 ** 18)
    s.add(c2, (), "b", 0.62, 115)
    s.add(c, ("c1",), "b", 0.69, 45)
    assert s.box_speed("b") == before
    # ... but from then on it does: the host got busy
    for steps in (92, 92, 36):
        s.add(c2 if steps > 50 else c, () if steps > 50 else ("c1",), "b", 0.66, steps)
    assert s.box_speed("b") < before - 0.15
    # a second box is placed by its first run, whatever code that is
    s.add(c2, (), "other", 0.60, 230)
    assert s.box_speed("other") > before + 0.5


def test_model_transfers_a_result_to_a_box_that_never_ran_it():
    h = hub()
    out = run(h, "a0", "b0", ("TOTAL_BATCH_SIZE = 2 ** 19", "TOTAL_BATCH_SIZE = 2 ** 18"), "Halved batch")
    assert out["status"] == "keep" and out["promoted"]
    truth = world(h.codes[h.head], "b1")[0]
    predicted, hops = h.reference("b1")
    assert hops == 0 and predicted < world(BASE, "b1")[0]  # it knows the head is better there too
    # from four results the estimate is rough, and the model knows it: it asks for a measurement
    assert abs(predicted / truth - 1) > 0.02 and h.needs_verification("b1")
    v = h.verify("b1", h.head, truth, steps=world(h.codes[h.head], "b1")[1])
    assert v["ok"] and not h.needs_verification("b1")
    assert abs(h.reference("b1")[0] / truth - 1) < 0.01
    assert not h.needs_verification("b0")                  # the box that ran it is sure


# ----------------------------------------------------------------- decisions

def test_clear_gain_kept_clear_loss_discarded():
    h = hub()
    assert run(h, "a0", "b0", ("MATRIX_LR = 0.04", "MATRIX_LR = 0.06"), "Raised LR")["status"] == "keep"
    out = run(h, "a1", "b1", ("MATRIX_LR = 0.06", "MATRIX_LR = 0.2"), "Raised LR a lot")
    assert out["status"] == "discard" and out["p_better"] < 0.2


def test_gain_inside_the_noise_is_run_again_then_decided():
    h = hub()
    base_val = world(BASE, "b0")[0]
    change = ("WEIGHT_DECAY = 0.2", "WEIGHT_DECAY = 0.1")
    out = run(h, "a0", "b0", change, "Halved weight decay", val=base_val * 0.997)
    assert out["status"] == "uncertain" and not out["promoted"]
    assert "unclear" in h.history_text()
    # the driver re-tests it as a merge carrying the same change id
    again = run(h, "a0", "b0", change, "merge: Halved weight decay", kind="merge",
                val=base_val * 1.004, extra={"change_id": out["change_id"]})
    assert again["status"] == "discard"                    # the second run did not confirm it
    h2 = hub()
    out = run(h2, "a0", "b0", change, "Halved weight decay", val=base_val * 0.997)
    again = run(h2, "a0", "b0", change, "merge: Halved weight decay", kind="merge",
                val=base_val * 0.985, extra={"change_id": out["change_id"]})
    assert again["status"] == "keep" and again["promoted"]


def test_crash_is_remembered_and_not_proposed_by_the_model():
    h = hub()
    head, code, _ = h.head_state()
    new_code = set_constant(code, "TOTAL_BATCH_SIZE", 2 ** 18)
    c = h.claim("a0", "b0", head, diff("x", "y"), "Halved batch", {"kind": "tune", "target_id": None},
                new_code=new_code)
    assert h.report(c["claim"], None, new_code)["status"] == "crash"
    for _ in range(30):
        cand = h._model_candidate("b0")
        assert cand is None or "TOTAL_BATCH_SIZE = 2 ** 18" not in cand["new_code"]
        if cand is None:
            break
        h.crashed.add(tuple(h_key(cand["new_code"])))


def h_key(code):
    from scas.v3 import _key

    return _key(extract(code)[0], frozenset())


# ---------------------------------------------------------------------- veto

def learn_lr_is_bad(h):
    for i, (box, lr) in enumerate((("b0", "0.2"), ("b1", "0.16"), ("b2", "0.25"), ("b0", "0.3"))):
        run(h, "a%d" % (i % 3), box, ("MATRIX_LR = 0.04", "MATRIX_LR = " + lr), "Raised LR to " + lr)


def test_model_refuses_what_it_knows_will_not_help_but_not_forever():
    h = hub(veto_min_obs=5, max_vetoes=2)
    learn_lr_is_bad(h)
    change = ("MATRIX_LR = 0.04", "MATRIX_LR = 0.22")
    r = run(h, "a1", "b1", change, "Raised LR to 0.22")
    assert r["duplicate"]["reason"] == "model veto" and "expects" in r["duplicate"]["message"]
    assert not h.inflight                                  # nothing left claimed
    assert "duplicate" in run(h, "a1", "b1", ("MATRIX_LR = 0.04", "MATRIX_LR = 0.24"), "LR 0.24")
    # third refusal in a row is not allowed: the agent gets its run
    assert "status" in run(h, "a1", "b1", ("MATRIX_LR = 0.04", "MATRIX_LR = 0.23"), "LR 0.23")


def test_no_refusal_on_hardware_the_model_does_not_know_and_never_for_structural():
    h = hub(veto_min_obs=1000)                             # no memory, too few results
    learn_lr_is_bad(h)
    assert not h.may_veto("b1")
    assert "status" in run(h, "a1", "b1", ("MATRIX_LR = 0.04", "MATRIX_LR = 0.22"), "LR 0.22")
    h = hub(veto_min_obs=5)
    learn_lr_is_bad(h)
    out = run(h, "a1", "b1", ("relu(x)", "gelu(x)"), "Switched to gelu", kind="structural")
    assert out["status"] == "keep" and out["structural"]


# ------------------------------------------------------------ model directive

def test_model_directive_edits_one_constant_and_goes_through_the_normal_path():
    h = hub()
    seen = set()
    # the bandit offers the model its turn among the first pulls
    first = [h.next_directive("a%d" % i, "b0") for i in range(3)]
    assert [d["kind"] for d in first].count("model") == 1
    for d in first:
        h.release(d)
    for i in range(4):
        d = {"kind": "model", "target_id": None, **h._model_candidate("b0")}
        head, code, _ = h.head_state()
        c0, rest0 = extract(code)
        c1, rest1 = extract(d["new_code"])
        assert rest1 == rest0 and sum(c0[k] != c1[k] for k in c0) == 1
        assert d["new_code"] not in seen
        seen.add(d["new_code"])
        c = h.claim("a0", "b0", head, diff("old%d" % i, "new%d" % i), d["description"], d,
                    new_code=d["new_code"])
        val, steps = world(d["new_code"], "b0")
        out = h.report(c["claim"], val, d["new_code"], steps=steps)
        assert out["kind"] == "model" and out["status"] in ("keep", "discard", "uncertain")
    assert h.snapshot()["by_kind"]["model"]["n"] == 4


def test_model_places_the_learning_rate_optimum_between_results():
    h = hub()
    for box, lr in (("b0", "0.02"), ("b1", "0.03"), ("b2", "0.05"), ("b0", "0.12"), ("b1", "0.09")):
        run(h, "a0", box, ("MATRIX_LR = 0.04", "MATRIX_LR = " + lr), "LR " + lr)
    c0 = extract(BASE)[0]
    guess = {lr: h.model.compare((dict(c0, MATRIX_LR=lr), ()), (c0, ()), "b2")["gain_pct"]
             for lr in (0.02, 0.06, 0.2)}
    # never ran 0.06 or 0.2: it places the optimum between the results on either side
    assert guess[0.06] > 0.2 and guess[0.02] < 0 and guess[0.2] < guess[0.06]


# ------------------------------------------------------- structural changes

def test_structural_change_is_a_flag_shared_by_its_re_test():
    h = hub()
    out = run(h, "a0", "b0", ("relu(x)", "gelu(x)"), "Switched to gelu", kind="structural")
    assert out["promoted"] and out["structural"]
    assert h.flags[h.head] == {out["change_id"]}
    # a later constant edit keeps the flag; a foreign box expects the gain too
    run(h, "a1", "b1", ("MATRIX_LR = 0.04", "MATRIX_LR = 0.06"), "Raised LR")
    assert h.flags[h.head] == {out["change_id"]}
    assert h.reference("b2")[0] < world(BASE, "b2")[0] * 0.99


def test_snapshot_and_events_are_serialisable(tmp_path):
    h = SwarmV3(BASE, FakeEmbedder(), surrogate=Surrogate(dict(DEFAULT_PARAMS)),
                event_log=tmp_path / "ev.jsonl")
    for box in SPEED:
        val, steps = world(BASE, box)
        h.record_measurement(box, h.head, val, steps=steps)
    run(h, "a0", "b0", ("relu(x)", "gelu(x)"), "Switched to gelu", kind="structural")
    run(h, "a1", "b1", ("MATRIX_LR = 0.04", "MATRIX_LR = 0.3"), "LR 0.3")
    s = json.loads(json.dumps(h.snapshot()))
    assert s["model"]["observations"] == 5 and len(s["chain"]) == 2
    kinds = [json.loads(x)["event"] for x in (tmp_path / "ev.jsonl").read_text().splitlines()]
    assert kinds.count("report") == 2 and kinds.count("measured") == 3


def test_memory_from_earlier_runs_is_used():
    c, _ = extract(BASE)
    mem = []
    for i in range(12):
        lr = 0.02 * 1.3 ** i
        code = set_constant(BASE, "MATRIX_LR", lr)
        val, steps = world(code, "b0")
        mem.append({"run": "old", "box": "b0", "t": i, "val": val, "steps": steps,
                    "consts": extract(code)[0], "flags": []})
    h = SwarmV3(BASE, FakeEmbedder(), surrogate=Surrogate(dict(DEFAULT_PARAMS)), memory=mem)
    val, steps = world(BASE, "b0")
    h.record_measurement("b0", h.head, val, steps=steps)
    assert h.may_veto("b0")                                # same speed as the remembered box
    r = run(h, "a0", "b0", ("MATRIX_LR = 0.04", "MATRIX_LR = 0.3"), "LR 0.3")
    assert r.get("duplicate", {}).get("reason") == "model veto"
