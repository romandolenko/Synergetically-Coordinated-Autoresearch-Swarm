#!/usr/bin/env python3
"""analyze.py - turn a fetched real results.tsv into the review's S1 answers.

    uv run python tools/vast/analyze.py vast_runs/<run>/results.tsv

Four things, in order of how much they gate the rest of the project:

  1. Run health         - baseline, best, improvement, crash rate, cost per experiment.
  2. Validator floor    - is MIN_PLAUSIBLE_VAL_BPB=0.5 still safe on THIS dataset?
                          It was calibrated against climbmix (~1.0). On TinyStories a
                          legitimate run can dip under it, which would make the SCAS
                          safety envelope reject real work. Blocks S2.
  3. T5.2 replay        - MiniLM silhouette over the real descriptions, raw and with
                          the [CATEGORY] prefix anchor. This is the review's C1 gate.
  4. Router error rate  - keyword router vs the plan's ground-truth categories
                          (needs sweep_state.json alongside results.tsv). The repo has
                          never had labelled data for this before.

Writes report.md next to the tsv and prints it.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

EXPECTED_COLUMNS = ["commit", "val_bpb", "memory_gb", "status", "description"]


def load_rows(tsv_path: Path) -> list[dict]:
    with tsv_path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        if reader.fieldnames != EXPECTED_COLUMNS:
            raise SystemExit(
                "results.tsv header mismatch.\n  expected: %s\n  got:      %s"
                % (EXPECTED_COLUMNS, reader.fieldnames)
            )
        return list(reader)


def section_run_health(rows, state, lines):
    lines.append("## 1. Run health\n")
    ok = [r for r in rows if r["status"] != "crash"]
    crashed = [r for r in rows if r["status"] == "crash"]
    kept = [r for r in rows if r["status"] == "keep"]
    if not ok:
        lines.append("No non-crash rows. Nothing else can be computed.\n")
        return None, None
    vals = [float(r["val_bpb"]) for r in ok]
    baseline = vals[0]
    best = min(vals)
    lines.append("| metric | value |")
    lines.append("| --- | --- |")
    lines.append("| experiments recorded | %d |" % len(rows))
    lines.append("| kept / discarded / crashed | %d / %d / %d |"
                 % (len(kept), len(rows) - len(kept) - len(crashed), len(crashed)))
    lines.append("| baseline val_bpb | %.6f |" % baseline)
    lines.append("| best val_bpb | %.6f |" % best)
    lines.append("| absolute improvement | %.6f (%.2f%%) |"
                 % (baseline - best, 100.0 * (baseline - best) / baseline))
    lines.append("| peak memory | %.1f GB |"
                 % max((float(r["memory_gb"]) for r in ok), default=0.0))
    if state:
        secs = [e.get("wall_seconds", 0) for e in state.get("experiments", []) if e.get("wall_seconds")]
        if secs:
            total_h = sum(secs) / 3600.0
            lines.append("| wall clock | %.2f h (%.1f min/experiment) |"
                         % (total_h, sum(secs) / len(secs) / 60.0))
        mfus = [e["mfu_percent"] for e in state.get("experiments", []) if e.get("mfu_percent")]
        if mfus:
            lines.append("| median MFU | %.1f%% |" % sorted(mfus)[len(mfus) // 2])
    lines.append("")
    if crashed:
        ids = {}
        for e in (state or {}).get("experiments", []):
            ids[e.get("description")] = e.get("id")
        names = [ids.get(r["description"]) or r["description"][:50] for r in crashed]
        lines.append("Crashed: %s (see `runs/<id>.log` for the traceback)\n" % ", ".join(names))
    return baseline, best


def section_validator_floor(rows, lines):
    from scas.coordinator import (MIN_PLAUSIBLE_FRACTION, MIN_PLAUSIBLE_VAL_BPB,
                                  plausibility_floor)

    lines.append("## 2. Safety-envelope floor (blocks S2)\n")
    ok = [float(r["val_bpb"]) for r in rows if r["status"] != "crash"]
    if not ok:
        lines.append("No usable rows.\n")
        return
    # Replay the rule the live service applies: each row is checked against
    # the best value accepted before it.
    running, rejected, tightest = None, [], None
    for r in rows:
        if r["status"] == "crash":
            continue
        v = float(r["val_bpb"])
        floor = plausibility_floor(running)
        if v < floor:
            rejected.append((v, floor, r["description"][:60]))
            continue
        if running is not None:
            ratio = v / running
            tightest = ratio if tightest is None else min(tightest, ratio)
        running = v if running is None else min(running, v)
    lines.append("The floor is relative: a report may not claim less than **%.0f%%** of "
                 "the best value already accepted (absolute backstop %.2f).\n"
                 % (100 * MIN_PLAUSIBLE_FRACTION, MIN_PLAUSIBLE_VAL_BPB))
    lines.append("| | value |")
    lines.append("| --- | --- |")
    lines.append("| honest rows the envelope would reject | %d of %d |" % (len(rejected), len(ok)))
    if tightest is not None:
        lines.append("| largest single-step drop observed | to %.1f%% of the prior best |"
                     % (100 * tightest))
    lines.append("")
    if rejected:
        lines.append("**FAIL - the floor rejects legitimate results:**\n")
        for v, floor, d in rejected:
            lines.append("- %.6f < floor %.4f  `%s`" % (v, floor, d))
        lines.append("")
    elif tightest is not None and tightest < MIN_PLAUSIBLE_FRACTION + 0.15:
        lines.append("**AT RISK** - an honest step came within 15 points of the "
                     "rejection fraction. Consider lowering `MIN_PLAUSIBLE_FRACTION`.\n")
    else:
        lines.append("**OK - no honest result comes near the floor.**\n")


def section_replay(tsv_path, lines):
    from scas.replay import replay

    lines.append("## 3. T5.2 replay on real descriptions (review C1)\n")
    run_dir, summary = replay(tsv_path)
    sil = summary["silhouette_score"]
    sil_p = summary["silhouette_score_prefixed"]
    thr = summary["silhouette_threshold"]
    lines.append("| | silhouette | vs threshold %.3f |" % thr)
    lines.append("| --- | --- | --- |")
    lines.append("| raw MiniLM | %.4f | %s |" % (sil, "PASS" if sil > thr else "FAIL"))
    lines.append("| `[CATEGORY]`-prefixed | %.4f | %s |"
                 % (sil_p, "PASS" if sil_p > thr else "FAIL"))
    lines.append("| prefix lift | %+.4f | |" % (summary.get("prefix_mitigation_lift") or 0.0))
    lines.append("")
    lines.append("Router-assigned category counts: `%s`\n" % json.dumps(summary["category_counts"]))
    lines.append("Trace: `%s`\n" % run_dir)
    if sil <= thr:
        lines.append("Raw embeddings do not separate these descriptions. That is the review's "
                     "C1 risk showing up on real data - the coupling law has little structure "
                     "to organise unless the prefix anchor (or a stronger embedder, review S4) "
                     "carries it.\n")
    return summary


def explain_route(description):
    """Reproduce scas.replay._category_for and report WHICH keyword decided it.

    An unmatched description is labelled "unknown" (it used to be silently
    defaulted to "optimizer").
    """
    from scas.replay import route

    label, kw = route(description)
    if kw is None:
        return label, "NO KEYWORD MATCHED"
    return label, "matched %r" % kw


def section_router(rows, state, lines):
    lines.append("## 4. Keyword router vs ground truth\n")
    if not state:
        lines.append("No `sweep_state.json` next to the tsv, so there are no ground-truth "
                     "labels to score against. Fetch it with `vastctl fetch`.\n")
        return
    truth = {e["description"]: e["category"] for e in state.get("experiments", [])
             if e.get("category") and e["category"] != "baseline"}
    scored = []
    for r in rows:
        if r["description"] in truth:
            pred, why = explain_route(r["description"])
            scored.append((r["description"], truth[r["description"]], pred, why))
    if not scored:
        lines.append("No description overlap between results.tsv and sweep_state.json.\n")
        return
    correct = sum(1 for _, t, p, _ in scored if t == p)
    fallbacks = sum(1 for _, _, _, w in scored if w.startswith("NO KEYWORD"))
    lines.append("Router accuracy: **%d/%d (%.0f%%)** over labelled rows. "
                 "%d row(s) matched no keyword at all and were left as "
                 "`unknown`.\n" % (correct, len(scored), 100.0 * correct / len(scored), fallbacks))
    per_class = {}
    for _, t, p, _ in scored:
        acc = per_class.setdefault(t, [0, 0])
        acc[1] += 1
        if t == p:
            acc[0] += 1
    lines.append("| true category | correct | n | accuracy |")
    lines.append("| --- | --- | --- | --- |")
    for cat in sorted(per_class):
        good, total = per_class[cat]
        lines.append("| %s | %d | %d | %.0f%% |" % (cat, good, total, 100.0 * good / total))
    lines.append("")
    wrong = [x for x in scored if x[1] != x[2]]
    if wrong:
        lines.append("Misroutes (with the deciding keyword):\n")
        for d, t, p, why in wrong[:15]:
            lines.append("- truth `%s` -> routed `%s` (%s)  \n  `%s`" % (t, p, why, d[:110]))
        lines.append("")
        lines.append("Keywords are anchored at a word start, but the router is still "
                     "first-match-wins across categories. Check any match that looks "
                     "accidental before trusting the label.\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tsv", help="path to a fetched results.tsv")
    ap.add_argument("--skip-replay", action="store_true",
                    help="skip the MiniLM section (no model download)")
    args = ap.parse_args()

    tsv_path = Path(args.tsv).resolve()
    if not tsv_path.exists():
        raise SystemExit("no such file: %s" % tsv_path)
    rows = load_rows(tsv_path)

    state = None
    state_path = tsv_path.parent / "sweep_state.json"
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
    meta_path = tsv_path.parent / "instance.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}

    lines = ["# Real-data analysis: %s\n" % tsv_path.parent.name]
    if meta:
        lines.append("GPU `%s` - instance %s - $%.3f/hr - branch `%s` - dataset `%s`\n"
                     % (meta.get("gpu_name"), meta.get("instance_id"),
                        meta.get("dph_total") or 0.0, meta.get("branch"), meta.get("dataset")))
    lines.append("> Descriptions here come from a deterministic plan, not an LLM. "
                 "The val_bpb numbers are real; the card *text* is authored, so section 3 "
                 "is an upper bound on what real agent cards would give.\n")

    section_run_health(rows, state, lines)
    section_validator_floor(rows, lines)
    if not args.skip_replay:
        section_replay(tsv_path, lines)
    else:
        lines.append("## 3. T5.2 replay\n\nSkipped (--skip-replay).\n")
    section_router(rows, state, lines)

    report = "\n".join(lines) + "\n"
    out = tsv_path.parent / "report.md"
    out.write_text(report, encoding="utf-8")
    print(report)
    print("written to %s" % out)


if __name__ == "__main__":
    main()
