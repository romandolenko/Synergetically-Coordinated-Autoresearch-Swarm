"""Tests for the autoresearch loop wrapper (scas/loop.py).

The coordinator service runs in-process on an ephemeral port with a fake
embedder; the wrapper talks to it over real HTTP.
"""

from __future__ import annotations

import json
import random
import threading
from pathlib import Path

import numpy as np
import pytest

from scas.loop import (
    LoopClient,
    analyze_trace,
    draw_directive,
    read_last_tsv_row,
    render_packet,
)
from scas.service import CoordinatorService, make_server


class FakeEmbedder:
    def embed(self, s: str) -> np.ndarray:
        rng = np.random.default_rng(abs(hash(s)) % (2**32))
        v = rng.normal(size=16).astype(np.float32)
        return v / np.linalg.norm(v)


@pytest.fixture()
def live_server():
    svc = CoordinatorService(agents=["a0", "a1"], n_groups=2,
                             embedder=FakeEmbedder())
    server = make_server(svc, "127.0.0.1", 0)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()
    t.join(timeout=5)


# ------------------------------------------------------------------ directives


def test_draw_directive_follows_alpha():
    rng = random.Random(0)
    g = {"alpha": 1.0, "target_card_text": "best", "avoid_list": ["a"]}
    assert draw_directive(g, rng)["mode"] == "refine"
    g["alpha"] = 0.0
    assert draw_directive(g, rng)["mode"] == "explore"


def test_draw_directive_without_guidance_explores():
    d = draw_directive(None, random.Random(0))
    assert d["mode"] == "explore" and d["alpha"] is None and "note" in d
    # alpha present but no target yet (nothing accepted in group): still explore
    d = draw_directive({"alpha": 0.9, "target_card_text": None}, random.Random(0))
    assert d["mode"] == "explore"


def test_render_packet_refine_and_explore():
    refine = render_packet("a0", {"mode": "refine", "alpha": 0.63,
                                  "target": "muon on hidden", "avoid": ["old idea"]}, 3)
    assert "REFINE" in refine and '"muon on hidden"' in refine
    assert '- "old idea"' in refine and "directive #3" in refine
    explore = render_packet("a0", {"mode": "explore", "alpha": 0.16,
                                   "target": "muon on hidden", "avoid": ["old idea"]}, 4)
    assert "EXPLORE" in explore and "qualitatively different" in explore
    assert '- "old idea"' in explore


# ------------------------------------------------------------------ round trip


def test_report_then_prompt_round_trip(tmp_path):
    # Single group: with more slots, emergent empty-slot seeding would (by
    # design, results-laws §11) move the second report into its own group.
    svc = CoordinatorService(agents=["a0"], n_groups=1, embedder=FakeEmbedder())
    server = make_server(svc, "127.0.0.1", 0)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    live_server = f"http://127.0.0.1:{server.server_address[1]}"
    c = LoopClient("a0", server=live_server, state_dir=tmp_path,
                   rng=random.Random(0))
    # files_changed passed explicitly: in the real loop the git default is the
    # experiment commit; in this test repo HEAD is whatever was last committed.
    r1 = c.report("muon on the hidden weights", 0.99, files_changed=["train.py"])
    assert r1["accepted"] and r1["group_id"] is not None
    assert json.loads(c.state_path.read_text())["accepted"]

    r2 = c.report("muon momentum 0.9", 0.995,  # worse: gate moves, best stays
                  files_changed=["train.py"])
    assert r2["accepted"]
    assert r2["step"] == r1["step"] + 1  # wrapper let the service count steps
    assert r2["target_card_text"] == "muon on the hidden weights"

    try:
        packet = c.prompt()
    finally:
        server.shutdown()
        t.join(timeout=5)
    assert "SCAS COORDINATION PACKET" in packet
    kinds = [json.loads(x)["kind"] for x in c.trace_path.read_text().splitlines()]
    assert kinds == ["report", "report", "directive"]


def test_report_envelope_reject_surfaces(live_server, tmp_path):
    c = LoopClient("a0", server=live_server, state_dir=tmp_path)
    r = c.report("sneaky change", 0.99, files_changed=["train.py", "prepare.py"])
    assert not r["accepted"] and "files_changed" in r["reject_reason"]


# ------------------------------------------------------------------------- tsv


def test_read_last_tsv_row(tmp_path):
    tsv = tmp_path / "results.tsv"
    tsv.write_text(
        "commit\tval_bpb\tmemory_gb\tstatus\tdescription\n"
        "aaa1111\t0.997900\t44.0\tkeep\tbaseline\n"
        "bbb2222\t0.000000\t0.0\tcrash\tdouble width (OOM)\n"
    )
    row = read_last_tsv_row(tsv)
    assert row == {"summary": "double width (OOM)", "val_bpb": 0.0,
                   "status": "crash"}
    empty = tmp_path / "empty.tsv"
    empty.write_text("commit\tval_bpb\tmemory_gb\tstatus\tdescription\n")
    with pytest.raises(ValueError):
        read_last_tsv_row(empty)


# -------------------------------------------------------------------- analysis


def test_analyze_trace_scores_compliance(tmp_path):
    emb = FakeEmbedder()
    trace = tmp_path / "a0_trace.jsonl"
    rows = [
        # scorable refine, obeyed: the new card IS the target
        {"kind": "directive", "mode": "refine", "target": "T", "avoid": ["A"]},
        {"kind": "report", "summary": "T"},
        # scorable explore, obeyed: the new card IS the avoided one
        {"kind": "directive", "mode": "explore", "target": "T", "avoid": ["A"]},
        {"kind": "report", "summary": "A"},
        # unscorable (no target yet): skipped
        {"kind": "directive", "mode": "explore", "target": None, "avoid": []},
        {"kind": "report", "summary": "whatever"},
    ]
    trace.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    out = analyze_trace(trace, emb)
    assert out["n_pairs_total"] == 3
    assert out["refine"] == {"n": 1, "mean_margin": out["refine"]["mean_margin"],
                             "compliance": 1.0}
    assert out["refine"]["mean_margin"] > 0
    assert out["explore"]["compliance"] == 1.0 and out["explore"]["mean_margin"] < 0
    assert out["refine_minus_explore_margin"] > 0
