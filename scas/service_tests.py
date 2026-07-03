"""Tests for the Phase 2 live coordinator service (scas/service.py).

The embedder is faked (deterministic hash -> unit vector), so no MiniLM
download is needed; the HTTP test uses an ephemeral port on localhost.
"""

from __future__ import annotations

import threading

import numpy as np
import pytest
import requests

from scas.coordinator import BASELINE_PREPARE_HASH, BASELINE_PYPROJECT_HASH
from scas.service import CoordinatorService, make_server


class FakeEmbedder:
    """Deterministic stand-in for MiniLM: hash the string into a unit vector."""

    def embed(self, s: str) -> np.ndarray:
        rng = np.random.default_rng(abs(hash(s)) % (2**32))
        v = rng.normal(size=16).astype(np.float32)
        return v / np.linalg.norm(v)


def _service(**kw) -> CoordinatorService:
    kw.setdefault("agents", ["a0", "a1"])
    kw.setdefault("n_groups", 2)
    kw.setdefault("embedder", FakeEmbedder())
    return CoordinatorService(**kw)


def _report(agent="a0", summary="try muon on hidden weights", val=0.99, **kw):
    return {"agent_id": agent, "summary": summary, "val_bpb": val, **kw}


# ------------------------------------------------------------------ happy path


def test_step_accepts_and_returns_guidance():
    svc = _service()
    code, r0 = svc.step(_report("a0", "muon on the hidden weights", 0.99))
    code1, r1 = svc.step(_report("a1", "wider FFN at 6x", 0.98))
    assert code == code1 == 200
    assert r0["accepted"] and r1["accepted"]
    # emergent seeding: two accepted observations claim the two slots
    assert {r0["group_id"], r1["group_id"]} == {0, 1}
    assert r0["improved_group_best"] and r1["improved_group_best"]
    assert 0.0 <= r0["alpha"] <= 1.0 and r0["psi"] is not None
    # each agent's target is its own group best so far
    assert r0["target_card_text"] == "muon on the hidden weights"
    assert r1["target_card_text"] == "wider FFN at 6x"


def test_worse_report_updates_gate_but_not_target():
    svc = _service(agents=["a0"], n_groups=1)
    svc.step(_report("a0", "best so far", 0.90))
    code, r = svc.step(_report("a0", "a worse attempt", 0.95))
    assert code == 200 and r["accepted"]
    assert not r["improved_group_best"]
    assert r["target_card_text"] == "best so far"
    assert "a worse attempt" in r["avoid_list"]


def test_avoid_list_most_recent_first_capped_deduped_excludes_target():
    svc = _service(agents=["a0"], n_groups=1, avoid_k=2)
    svc.step(_report("a0", "s1 best", 0.90))
    svc.step(_report("a0", "s2", 0.95))
    svc.step(_report("a0", "s3", 0.96))
    svc.step(_report("a0", "s3", 0.96))  # duplicate submission
    _, r = svc.step(_report("a0", "s4", 0.97))
    assert r["target_card_text"] == "s1 best"
    assert r["avoid_list"] == ["s4", "s3"]  # capped at 2, deduped, no target


def test_unknown_agent_auto_registers():
    svc = _service()
    code, r = svc.step(_report("newcomer", "label smoothing 0.05", 0.97))
    assert code == 200 and r["accepted"]
    assert "newcomer" in svc.coord.agent_to_group


# -------------------------------------------------------------- safety envelope


def test_crash_rejected_and_state_untouched():
    svc = _service()
    svc.step(_report("a0", "good first step", 0.99))
    psi_before = dict(svc.coord.state.psi)
    code, r = svc.step(_report("a0", "double width (OOM)", 0.0, status="crash"))
    assert code == 200
    assert not r["accepted"] and "val_bpb" in r["reject_reason"]
    assert svc.coord.state.psi == psi_before


def test_envelope_rejects_extra_files_changed():
    svc = _service()
    _, r = svc.step(_report(files_changed=["train.py", "prepare.py"]))
    assert not r["accepted"] and "files_changed" in r["reject_reason"]


def test_envelope_rejects_drifted_hash():
    svc = _service()
    _, r = svc.step(_report(pyproject_hash="deadbeef"))
    assert not r["accepted"] and "pyproject" in r["reject_reason"]


def test_strict_envelope_requires_hashes():
    svc = _service(strict_envelope=True)
    _, r = svc.step(_report())  # hashes omitted
    assert not r["accepted"]
    _, r = svc.step(_report(
        pyproject_hash=BASELINE_PYPROJECT_HASH,
        prepare_hash=BASELINE_PREPARE_HASH,
    ))
    assert r["accepted"]


def test_crash_budget_quarantines_then_recovers():
    svc = _service(agents=["a0"], n_groups=1)
    for _ in range(3):
        svc.step(_report("a0", "kaboom", 0.0, status="crash"))
    # window holds 3 crashes: clean reports are rejected until they age out
    rejected = [svc.step(_report("a0", f"clean {i}", 0.99))[1] for i in range(3)]
    assert all(not r["accepted"] for r in rejected)
    assert all("crash budget" in r["reject_reason"] for r in rejected)
    # window is now [c,c,k,k,k] -> 2 crashes <= budget: accepted again
    _, r = svc.step(_report("a0", "clean recovery", 0.99))
    assert r["accepted"]


# ------------------------------------------------------------- schema handling


@pytest.mark.parametrize("bad", [
    {"summary": "s", "val_bpb": 1.0},                      # missing agent_id
    {"agent_id": "a0", "val_bpb": 1.0},                    # missing summary
    {"agent_id": "a0", "summary": "s"},                    # missing val_bpb
    {"agent_id": "a0", "summary": "s", "val_bpb": "NaNay"},
    {"agent_id": "a0", "summary": "s", "val_bpb": 1.0, "status": "meh"},
])
def test_malformed_reports_get_400(bad):
    code, r = _service().step(bad)
    assert code == 400 and "error" in r


# ------------------------------------------------------------- categories mode


def test_categories_mode_uses_fixed_partition():
    svc = _service(
        agents=["a0", "a1", "a2"],
        categories=["optimizer", "architecture"],
    )
    assert svc.mode == "categories"
    _, r0 = svc.step(_report("a0", "lr 3e-4", 0.99))
    _, r2 = svc.step(_report("a2", "muon beta 0.9", 0.98))
    assert r0["group_id"] == r2["group_id"] == 0  # round-robin: a0, a2 share
    # a2 improved the shared group best, so a0's next guidance targets it
    assert r2["improved_group_best"]
    assert r2["target_card_text"] == "muon beta 0.9"


# --------------------------------------------------------------- HTTP layer


def test_http_internal_error_returns_json_500():
    """Regression: a crashing embedder must yield a JSON 500, not a dropped
    connection (first live smoke test died exactly this way)."""

    class BoomEmbedder:
        def embed(self, s):
            raise RuntimeError("boom")

    svc = _service(embedder=BoomEmbedder())
    server = make_server(svc, "127.0.0.1", 0)
    port = server.server_address[1]
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    try:
        r = requests.post(f"http://127.0.0.1:{port}/step", json=_report(), timeout=5)
        assert r.status_code == 500
        assert "boom" in r.json()["error"]
    finally:
        server.shutdown()
        t.join(timeout=5)


def test_embedder_cache_read_across_threads(tmp_path):
    """Regression: the service embeds on HTTP worker threads; the SQLite
    cache connection must survive cross-thread use (check_same_thread=False
    plus internal lock)."""
    import io

    from scas.embedder import Embedder, _key

    e = Embedder(cache_path=tmp_path / "emb.sqlite")
    vec = np.ones(4, dtype=np.float32)
    buf = io.BytesIO()
    np.save(buf, vec, allow_pickle=False)
    e._conn.execute(
        "INSERT INTO embeddings (key, vec) VALUES (?, ?)",
        (_key(e._model_name, "hello"), buf.getvalue()),
    )
    e._conn.commit()

    out: dict = {}
    t = threading.Thread(target=lambda: out.update(v=e.embed("hello")))
    t.start()
    t.join(timeout=10)
    assert (out["v"] == vec).all()  # cache hit from another thread, no model load


def test_http_round_trip():
    svc = _service()
    server = make_server(svc, "127.0.0.1", 0)
    port = server.server_address[1]
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    base = f"http://127.0.0.1:{port}"
    try:
        h = requests.get(f"{base}/health", timeout=5).json()
        assert h["ok"] and h["mode"] == "emergent"

        r = requests.post(f"{base}/step", json=_report(), timeout=5)
        assert r.status_code == 200 and r.json()["accepted"]

        bad = requests.post(f"{base}/step", json={"agent_id": "a0"}, timeout=5)
        assert bad.status_code == 400

        st = requests.get(f"{base}/state", timeout=5).json()
        assert st["mode"] == "emergent" and "a0" in st["agents"]
        assert st["group_best_text"]  # the accepted step registered a best

        assert requests.get(f"{base}/nope", timeout=5).status_code == 404
    finally:
        server.shutdown()
        t.join(timeout=5)
