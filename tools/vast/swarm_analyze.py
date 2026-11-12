#!/usr/bin/env python3
"""Compare the arms of a `swarm.py run` directory and write report.md.

    uv run python tools/vast/swarm_analyze.py vast_runs/swarm-<stamp>

Boxes differ in speed, so every agent is scored by its improvement over the
baseline it measured on its own box in its own arm, not by absolute val_bpb.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

SIM_DUP = 0.80   # cosine above which two cards count as the same idea


def load_arm(adir: Path) -> dict:
    meta = json.loads((adir / "arm.json").read_text(encoding="utf-8"))
    agents = {}
    failed = 0
    for a in meta["agents"]:
        rows = []
        tp = adir / a / "trace.jsonl"
        if tp.exists():
            for line in tp.read_text(encoding="utf-8").splitlines():
                r = json.loads(line)
                if "proposal_failed" in r:
                    failed += 1
                else:
                    diff = adir / a / ("%03d.diff" % r["n"])
                    r["plus"] = plus_lines(diff.read_text(encoding="utf-8")) if diff.exists() else frozenset()
                    r["knobs"] = knobs(r["plus"])
                    rows.append(r)
        agents[a] = rows
    return {"meta": meta, "agents": agents, "proposal_failures": failed, "dir": adir}


def plus_lines(diff: str) -> frozenset:
    return frozenset(ln[1:].strip() for ln in diff.splitlines()
                     if ln.startswith("+") and not ln.startswith("+++") and ln[1:].strip())


def knobs(plus: frozenset) -> frozenset:
    """Module-level constants an edit assigns (empty for structural edits)."""
    out = set()
    for ln in plus:
        m = re.match(r"^([A-Z][A-Z0-9_]+)\s*=", ln)
        if m:
            out.add(m.group(1))
    return frozenset(out)


def is_experiment(r: dict) -> bool:
    return r["n"] > 0 and r.get("kind") != "verify"


def rel_improvement(rows: list[dict], until: float | None = None) -> float | None:
    """Fractional drop of best kept val_bpb below this agent's own baseline."""
    base = next((r["val_bpb"] for r in rows if r["n"] == 0 and r["status"] == "keep"), None)
    if base is None:
        return None
    kept = [r["val_bpb"] for r in rows
            if r["status"] in ("keep", "verify") and (until is None or r["t_done"] <= until)]
    return (base - min(kept)) / base if kept else 0.0


def pct(x) -> str:
    return "-" if x is None else "%.2f%%" % (100.0 * x)


def arm_summary(arm: dict) -> dict:
    exps = [r for rows in arm["agents"].values() for r in rows if is_experiment(r)]
    rels = [rel_improvement(rows) for rows in arm["agents"].values()]
    rels = [x for x in rels if x is not None]
    kept_vals = [r["val_bpb"] for rows in arm["agents"].values() for r in rows
                 if r["status"] in ("keep", "verify")]
    costs = [r.get("cost_usd") or 0.0 for r in exps]
    return {
        "experiments": len(exps),
        "keep": sum(r["status"] == "keep" for r in exps),
        "discard": sum(r["status"] == "discard" for r in exps),
        "crash": sum(r["status"] == "crash" for r in exps),
        "best_rel": max(rels) if rels else None,
        "mean_rel": sum(rels) / len(rels) if rels else None,
        "best_abs": min(kept_vals) if kept_vals else None,
        "agent_cost": sum(costs),
        "structural": sum(1 for r in exps if not r["knobs"]),
    }


def curve(arm: dict, fractions=(0.25, 0.5, 0.75, 1.0)) -> list:
    """Best improvement of any agent after a given share of the arm's GPU time."""
    t0, t1 = arm["meta"]["t_start"], arm["meta"]["deadline"]
    out = []
    for f in fractions:
        until = t0 + f * (t1 - t0)
        rels = [rel_improvement(rows, until) for rows in arm["agents"].values()]
        rels = [x for x in rels if x is not None]
        out.append(max(rels) if rels else None)
    return out


def redundancy(arm: dict, embedder) -> dict:
    """How often an experiment repeats what ANOTHER agent of the arm already
    started (for a one-agent arm: what the agent itself already tried)."""
    import numpy as np

    exps = sorted((r | {"agent": a} for a, rows in arm["agents"].items()
                   for r in rows if is_experiment(r)), key=lambda r: r["t_start"])
    solo = len(arm["agents"]) == 1
    for r in exps:
        r["z"] = embedder.embed(r["description"])
    exact = same_knob = semantic = 0
    sims = []
    rekept = 0
    for i, r in enumerate(exps):
        earlier = [e for e in exps[:i] if solo or e["agent"] != r["agent"]]
        if not earlier:
            continue
        if r["plus"] and any(e["plus"] == r["plus"] for e in earlier):
            exact += 1
        if r["knobs"] and any(e["knobs"] == r["knobs"] for e in earlier):
            same_knob += 1
        s = max(float(np.dot(r["z"], e["z"])) for e in earlier)
        sims.append(s)
        if s >= SIM_DUP:
            semantic += 1
        if r["status"] == "keep" and r["knobs"] and any(
                e["status"] == "keep" and e["knobs"] == r["knobs"] for e in earlier):
            rekept += 1
    n = len(exps)
    return {"n": n, "exact": exact, "same_knob": same_knob, "semantic": semantic,
            "mean_max_sim": sum(sims) / len(sims) if sims else None, "rekept": rekept}


def scas_section(arm: dict, embedder) -> list[str]:
    from scas.loop import analyze_trace

    lines = ["## SCAS arm: what the coordinator did", ""]
    exps = [r for rows in arm["agents"].values() for r in rows if r["n"] > 0]
    for mode in ("refine", "explore"):
        sub = [r for r in exps if (r.get("guidance") or {}).get("mode") == mode]
        k = sum(r["status"] == "keep" for r in sub)
        c = sum(r["status"] == "crash" for r in sub)
        lines.append("- **%s** directives: %d experiments, %d kept, %d crashed"
                     % (mode.upper(), len(sub), k, c))
    rej = [r for r in exps if (r.get("coordinator") or {}).get("accepted") is False]
    err = [r for r in exps if (r.get("coordinator") or {}).get("error")]
    crash_rej = sum(1 for r in rej if r["status"] == "crash")
    lines.append("- reports the safety envelope rejected: %d (%d of them crash reports); "
                 "reports that failed to reach the coordinator: %d"
                 % (len(rej), crash_rej, len(err)))
    led = sum(1 for r in exps if (r.get("coordinator") or {}).get("improved_group_best"))
    led_discard = sum(1 for r in exps if (r.get("coordinator") or {}).get("improved_group_best")
                      and r["status"] != "keep")
    lines.append("- cards that became a group target: %d (%d of them were not kept experiments)"
                 % (led, led_discard))
    snap = arm["dir"] / "coordinator_snapshot.json"
    if snap.exists():
        s = json.loads(snap.read_text(encoding="utf-8"))
        lines.append("- final groups: %s" % json.dumps(s.get("groups")))
        for g, t in sorted((s.get("group_best_text") or {}).items()):
            lines.append("  - group %s target: \"%s\"" % (g, (t or "")[:150]))
    lines += ["", "Steering, measured in embedding space (`scas.loop analyze`):", ""]
    lines.append("| agent | scorable | REFINE n | REFINE compliance | EXPLORE n | EXPLORE compliance | refine-minus-explore margin |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    for a in arm["meta"]["agents"]:
        tp = arm["dir"] / "scas_loop" / ("%s_trace.jsonl" % a)
        if not tp.exists():
            continue
        r = analyze_trace(tp, embedder)

        def f(x, p=False):
            return "-" if x is None else ("%.0f%%" % (100 * x) if p else "%+.3f" % x)

        lines.append("| %s | %d | %d | %s | %d | %s | %s |" % (
            a, r["refine"]["n"] + r["explore"]["n"], r["refine"]["n"],
            f(r["refine"]["compliance"], True), r["explore"]["n"],
            f(r["explore"]["compliance"], True), f(r["refine_minus_explore_margin"])))
    lines.append("")
    return lines


def scas2_section(arm: dict) -> list[str]:
    lines = ["## SCAS v2 arm: what the coordinator did", ""]
    rows = [r for rs in arm["agents"].values() for r in rs if r["n"] > 0]
    exps = [r for r in rows if r.get("kind") != "verify"]
    lines += ["| directive | experiments | kept | summed gain |", "| --- | --- | --- | --- |"]
    for kind in ("tune", "structural", "refine", "merge", "combine"):
        sub = [r for r in exps if r.get("kind") == kind]
        if sub:
            kept = [r for r in sub if r["status"] == "keep"]
            lines.append("| %s | %d | %d | %.2f%% |" % (
                kind, len(sub), len(kept),
                sum((r.get("coordinator") or {}).get("gain_pct", 0.0) for r in kept)))
    lines.append("")
    promoted = sum(1 for r in exps if (r.get("coordinator") or {}).get("promoted"))
    adopted = sum(1 for r in exps if (r.get("coordinator") or {}).get("adopted"))
    requeued = sum(1 for r in exps if r["status"] == "keep"
                   and not (r.get("coordinator") or {}).get("promoted")) - adopted
    ver = [r for r in rows if r.get("kind") == "verify"]
    failed = [r for r in ver if r["status"] == "verify_fail"]
    rolled = sum(1 for r in ver if (r.get("coordinator") or {}).get("action") == "rolled_back")
    lines.append("- kept experiments promoted to the shared version: %d; kept on an "
                 "outdated version and queued for a re-test: %d; adopted after a second "
                 "keep without a third run: %d" % (promoted, requeued, adopted))
    lines.append("- re-measurement runs: %d (%d disagreed, %d rollbacks)"
                 % (len(ver), len(failed), rolled))
    ev = arm["dir"] / "v2_events.jsonl"
    if ev.exists():
        events = [json.loads(x) for x in ev.read_text(encoding="utf-8").splitlines()]
        dup = [e for e in events if e["event"] == "duplicate"]
        lines.append("- proposals refused as duplicates before any GPU time: %d (%d identical "
                     "edits, %d near-identical ideas)" % (
                         len(dup), sum(e["reason"] == "identical edit" for e in dup),
                         sum(e["reason"] != "identical edit" for e in dup)))
    snap = arm["dir"] / "v2_snapshot.json"
    if snap.exists():
        s = json.loads(snap.read_text(encoding="utf-8"))
        lines.append("- shared version at the end: %d changes deep" % (len(s["chain"]) - 1))
        for c in s["chain"][1:]:
            lines.append("  - %+.2f%% (%s) %s" % (100 * (1 - c["ratio"]), c["by"], c["desc"][:120]))
        t = sorted(s.get("refine_targets") or [], key=lambda x: -x["children"])[:5]
        if t:
            lines.append("- most-refined targets: " + "; ".join(
                "%dx \"%s\"" % (x["children"], x["card"][:50]) for x in t))
    lines.append("")
    return lines


def main() -> None:
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    run = Path(sys.argv[1])
    meta = json.loads((run / "run.json").read_text(encoding="utf-8"))
    arms = {a: load_arm(run / a) for a in meta["arms"] if (run / a / "arm.json").exists()}
    from scas.embedder import Embedder

    embedder = Embedder()
    L = ["# S2 comparison: %s" % run.name, ""]
    L.append("Agents: `%s` via headless sessions. %d agent(s) per swarm arm, %.2f h per swarm "
             "arm, %.2f h for the single arm.%s"
             % (meta["model"], meta["agents_per_arm"], meta["hours"], meta["single_hours"],
                " **FAKE TRAINER - numbers are meaningless.**" if meta.get("fake") else ""))
    L += ["", "One run per arm. Differences between arms smaller than the spread between "
          "agents inside an arm are not evidence of anything.", ""]

    L += ["## 1. Outcome", "",
          "Improvement is the drop of an agent's best kept val_bpb below the baseline it "
          "measured on its own box.", "",
          "| arm | agents | experiments | kept | discarded | crashed | best agent | mean agent | per-agent | best val_bpb |",
          "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for name, arm in arms.items():
        s = arm_summary(arm)
        per = ", ".join(pct(rel_improvement(rows)) for rows in arm["agents"].values())
        L.append("| %s | %d | %d | %d | %d | %d | %s | %s | %s | %s |" % (
            name, len(arm["agents"]), s["experiments"], s["keep"], s["discard"], s["crash"],
            pct(s["best_rel"]), pct(s["mean_rel"]), per,
            "-" if s["best_abs"] is None else "%.6f" % s["best_abs"]))

    L += ["", "## 2. Best improvement against GPU time spent", "",
          "Each column is a share of the arm's GPU budget (the same total for every arm).", "",
          "| arm | 25% | 50% | 75% | 100% |", "| --- | --- | --- | --- | --- |"]
    for name, arm in arms.items():
        L.append("| %s | %s |" % (name, " | ".join(pct(x) for x in curve(arm))))

    L += ["", "## 3. Redundancy", "",
          "Share of experiments that repeat something another agent of the same arm had "
          "already started (for `single`: something it had itself already tried).", "",
          "| arm | experiments | identical edit | same constant(s) | similar card (cos >= %.2f) | mean nearest-card cos | keeps re-found | structural edits |" % SIM_DUP,
          "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for name, arm in arms.items():
        r = redundancy(arm, embedder)
        s = arm_summary(arm)
        n = max(r["n"], 1)
        L.append("| %s | %d | %d (%.0f%%) | %d (%.0f%%) | %d (%.0f%%) | %s | %d | %d |" % (
            name, r["n"], r["exact"], 100 * r["exact"] / n, r["same_knob"],
            100 * r["same_knob"] / n, r["semantic"], 100 * r["semantic"] / n,
            "-" if r["mean_max_sim"] is None else "%.3f" % r["mean_max_sim"],
            r["rekept"], s["structural"]))
    L += ["", "\"Keeps re-found\" counts kept experiments that touch exactly the constants "
          "another agent had already kept: work a shared branch would have saved.", ""]

    if "scas" in arms:
        L += scas_section(arms["scas"], embedder)
    if "scas2" in arms:
        L += scas2_section(arms["scas2"])

    L += ["## 4. Agent overhead", "",
          "| arm | failed proposals | agent cost (API-equivalent) |", "| --- | --- | --- |"]
    for name, arm in arms.items():
        s = arm_summary(arm)
        L.append("| %s | %d | $%.2f |" % (name, arm["proposal_failures"], s["agent_cost"]))
    L.append("")

    text = "\n".join(L)
    (run / "report.md").write_text(text, encoding="utf-8")
    print(text)
    print("\nwritten to %s" % (run / "report.md"))


if __name__ == "__main__":
    main()
