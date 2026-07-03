"""Driver for the GPU-free half of Phase 2 item 3 (live K=3 paired run).

Surrogate val_bpb: the POC's own synthetic landscape (seed 0, K'=5, noise 0)
evaluated at the MiniLM embedding of each real, LLM-written hypothesis card.
Declared surrogate - measures steering and grouping, NOT training quality.

Subcommands:
    packets                     print the coordination packet for c0,c1,c2
    submit --arm coord|indep --cards <json>   score + report a round of cards
    finish                      compute effective-alpha + separation metrics
"""

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, "C:/Users/dolen/workspace/Synergetically-Coordinated-Autoresearch-Swarm")

import numpy as np

from scas.embedder import Embedder
from scas.loop import LoopClient, analyze_trace
from scas.synthetic import evaluate, make_landscape

ROOT = Path(__file__).parent / "live"
COORD = {"server": "http://127.0.0.1:8905", "agents": ["c0", "c1", "c2"],
         "state": ROOT / "coord"}
INDEP = {"server": "http://127.0.0.1:8906", "agents": ["i0", "i1", "i2"],
         "state": ROOT / "indep"}

_embedder = None
_landscape = None


def _lazy():
    global _embedder, _landscape
    if _embedder is None:
        _embedder = Embedder()
        _landscape = make_landscape(seed=0, k_prime=5, embedder=_embedder,
                                    noise_std=0.0)
    return _embedder, _landscape


def surrogate_val(text: str) -> float:
    emb, ls = _lazy()
    return evaluate(emb.embed(text), ls, np.random.default_rng(0))


def _client(arm: dict, agent: str, round_no: int | None = None) -> LoopClient:
    rng = random.Random(f"{agent}-r{round_no}") if round_no is not None else None
    return LoopClient(agent, server=arm["server"], state_dir=arm["state"], rng=rng)


def _bests_path() -> Path:
    return ROOT / "agent_bests.json"


def _load_bests() -> dict:
    p = _bests_path()
    return json.loads(p.read_text()) if p.exists() else {}


def cmd_packets(round_no: int) -> None:
    for agent in COORD["agents"]:
        c = _client(COORD, agent, round_no)
        print(c.prompt())
        print()


def cmd_submit(arm_name: str, cards_path: str) -> None:
    arm = COORD if arm_name == "coord" else INDEP
    cards = json.loads(Path(cards_path).read_text())
    bests = _load_bests()
    for card in cards:
        agent, summary = card["agent"], card["summary"]
        status = card.get("status")  # explicit "crash" allowed
        if status == "crash":
            val = 0.0
        else:
            val = surrogate_val(summary)
            best = bests.get(agent)
            status = "keep" if best is None or val < best else "discard"
            if status == "keep":
                bests[agent] = val
        c = _client(arm, agent)
        r = c.report(summary, round(val, 6), status=status,
                     files_changed=["train.py"])
        print(f"{agent} {status:7s} val={val:.4f} -> accepted={r['accepted']} "
              f"g={r['group_id']} alpha={None if r['alpha'] is None else round(r['alpha'], 3)} "
              f"reject={r['reject_reason']}")
    _bests_path().write_text(json.dumps(bests, indent=2))


def _arm_cards(arm: dict) -> tuple[list[tuple[str, int, str]], int, int]:
    """(accepted [(agent, group, summary)], n_reports, n_rejected) from traces."""
    out, n, rej = [], 0, 0
    for agent in arm["agents"]:
        tp = Path(arm["state"]) / f"{agent}_trace.jsonl"
        if not tp.exists():
            continue
        for line in tp.read_text().splitlines():
            r = json.loads(line)
            if r["kind"] != "report":
                continue
            n += 1
            resp = r["response"]
            if not resp["accepted"]:
                rej += 1
            elif resp["group_id"] is not None:
                out.append((agent, resp["group_id"], r["summary"]))
    return out, n, rej


def _separation(cards: list[tuple[str, int, str]]) -> dict:
    from sklearn.metrics import silhouette_score

    emb, _ = _lazy()
    groups = sorted({g for _, g, _ in cards})
    X = np.stack([emb.embed(s) for _, _, s in cards])
    y = np.array([g for _, g, _ in cards])
    cents = {g: X[y == g].mean(axis=0) for g in groups}
    intra = float(np.mean([np.linalg.norm(x - cents[g]) for x, g in zip(X, y)]))
    inter_d = [np.linalg.norm(cents[a] - cents[b])
               for i, a in enumerate(groups) for b in groups[i + 1:]]
    inter = float(np.mean(inter_d)) if inter_d else 0.0
    sil = (float(silhouette_score(X, y))
           if len(groups) >= 2 and len(cards) > len(groups) else None)
    return {"n_cards": len(cards), "n_groups": len(groups),
            "group_sizes": {str(g): int((y == g).sum()) for g in groups},
            "inter_centroid_dist": inter, "intra_dist": intra,
            "inter_over_intra": (inter / intra) if intra > 0 else None,
            "silhouette": sil}


def cmd_finish() -> None:
    emb, _ = _lazy()
    result: dict = {"surrogate": "synthetic landscape seed=0 K'=5 noise=0 at "
                                 "MiniLM embeddings of real LLM-written cards"}
    # effective alpha (coordinated arm only - independent agents get no packets)
    agg = {"refine": {"n": 0, "margins": []}, "explore": {"n": 0, "margins": []},
           "compliant": {"refine": 0, "explore": 0}}
    per_agent = {}
    for agent in COORD["agents"]:
        tp = Path(COORD["state"]) / f"{agent}_trace.jsonl"
        a = analyze_trace(tp, emb)
        per_agent[agent] = a
        for mode in ("refine", "explore"):
            n = a[mode]["n"]
            agg[mode]["n"] += n
            if n and a[mode]["mean_margin"] is not None:
                agg[mode]["margins"] += [a[mode]["mean_margin"]] * n
                agg["compliant"][mode] += round(a[mode]["compliance"] * n)
    pooled = {}
    for mode in ("refine", "explore"):
        n = agg[mode]["n"]
        pooled[mode] = {
            "n": n,
            "mean_margin": float(np.mean(agg[mode]["margins"])) if n else None,
            "compliance": agg["compliant"][mode] / n if n else None,
        }
    sep_margin = (pooled["refine"]["mean_margin"] - pooled["explore"]["mean_margin"]
                  if pooled["refine"]["mean_margin"] is not None
                  and pooled["explore"]["mean_margin"] is not None else None)
    result["effective_alpha"] = {"per_agent": per_agent, "pooled": pooled,
                                 "refine_minus_explore_margin": sep_margin}

    for name, arm in (("coordinated", COORD), ("independent_null", INDEP)):
        cards, n, rej = _arm_cards(arm)
        result[name] = {"n_reports": n, "n_rejected": rej,
                        "rejection_rate": rej / n if n else None,
                        "separation": _separation(cards) if cards else None}

    out = Path("C:/Users/dolen/workspace/Synergetically-Coordinated-Autoresearch-Swarm/sim_runs/live_paired_20260703.json")
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    print(f"\nWritten: {out}")


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    pk = sub.add_parser("packets")
    pk.add_argument("--round", type=int, required=True)
    sb = sub.add_parser("submit")
    sb.add_argument("--arm", choices=["coord", "indep"], required=True)
    sb.add_argument("--cards", required=True)
    sub.add_parser("finish")
    args = p.parse_args()
    ROOT.mkdir(parents=True, exist_ok=True)
    if args.cmd == "packets":
        cmd_packets(args.round)
    elif args.cmd == "submit":
        cmd_submit(args.arm, args.cards)
    else:
        cmd_finish()


if __name__ == "__main__":
    main()
