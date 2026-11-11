#!/usr/bin/env python3
"""sweep.py - runs ON the vast.ai box. Executes the program.md experiment loop
with a pre-declared hypothesis plan instead of an LLM proposer.

For each experiment it does exactly what program.md step 2-9 prescribes:

    edit train.py -> uv run train.py > runs/<id>.log -> grep val_bpb
    -> append a row to results.tsv -> git commit if improved, else git checkout

Output is a real `results.tsv` in the canonical 5-column schema, so it drops
straight into `python -m scas.replay`. Richer per-experiment metadata (timing,
tok/sec, MFU, the exact edit) lands in sweep_state.json.

    uv run python sweep.py --plan hypotheses.json --repo /workspace/autoresearch --hours 6

Resumable: re-run with --resume to continue from sweep_state.json.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

TSV_HEADER = "commit\tval_bpb\tmemory_gb\tstatus\tdescription"


def log(msg):
    print("[%s] %s" % (datetime.now(timezone.utc).strftime("%H:%M:%S"), msg), flush=True)


def sh(cmd, cwd, check=True, capture=True):
    proc = subprocess.run(cmd, cwd=str(cwd), shell=isinstance(cmd, str),
                          capture_output=capture, text=True)
    if check and proc.returncode != 0:
        raise RuntimeError("command failed: %s\n%s\n%s"
                           % (cmd, proc.stdout or "", proc.stderr or ""))
    return proc


def git(repo, *args, check=True):
    return sh(["git"] + list(args), repo, check=check)


def apply_edits(train_py: Path, edits: dict) -> dict:
    """Rewrite module-level constants in train.py. Returns {name: (old, new)}.

    Raises if a constant is missing or the value is already identical - a silent
    no-op would record a fake experiment.
    """
    text = train_py.read_text(encoding="utf-8")
    applied = {}
    for name, new_value in edits.items():
        pattern = re.compile(r"^(%s\s*=\s*)(.+?)(\s*(?:#.*)?)$" % re.escape(name), re.MULTILINE)
        m = pattern.search(text)
        if not m:
            raise RuntimeError("constant %r not found at module level in train.py" % name)
        old_value = m.group(2).strip()
        if old_value == str(new_value).strip():
            raise RuntimeError("edit %s=%s is a no-op (already that value)" % (name, new_value))
        text = text[:m.start()] + m.group(1) + str(new_value) + m.group(3) + text[m.end():]
        applied[name] = (old_value, str(new_value))
    train_py.write_text(text, encoding="utf-8")
    return applied


def parse_run_log(path: Path) -> dict:
    """Pull the summary block train.py prints at the end."""
    out = {}
    if not path.exists():
        return out
    text = path.read_text(encoding="utf-8", errors="replace")
    fields = {
        "val_bpb": float, "peak_vram_mb": float, "training_seconds": float,
        "total_seconds": float, "mfu_percent": float, "total_tokens_M": float,
        "num_steps": int, "num_params_M": float, "depth": int,
    }
    for key, cast in fields.items():
        m = re.search(r"^%s:\s*([0-9.]+)\s*$" % re.escape(key), text, re.MULTILINE)
        if m:
            try:
                out[key] = cast(float(m.group(1)))
            except ValueError:
                pass
    if "val_bpb" not in out:
        tail = [ln for ln in text.strip().splitlines()[-40:]]
        out["crash_tail"] = "\n".join(tail)
    return out


def tsv_escape(s: str) -> str:
    return " ".join(str(s).replace("\t", " ").split())


def load_plan(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    experiments = data["experiments"] if isinstance(data, dict) else data
    seen = set()
    for e in experiments:
        for key in ("id", "category", "description"):
            if key not in e:
                raise RuntimeError("experiment missing %r: %s" % (key, json.dumps(e)[:200]))
        if e["id"] in seen:
            raise RuntimeError("duplicate experiment id %r" % e["id"])
        seen.add(e["id"])
        if e["category"] != "baseline" and not e.get("edits"):
            raise RuntimeError("experiment %r has no edits" % e["id"])
    return experiments


def add_baseline_repeats(experiments: list[dict], repeats: int) -> list[dict]:
    """Insert extra unmodified runs right after the baseline so the sweep
    measures its own run-to-run noise (the first real sweep had none, and 3 of
    its 12 keeps were improvements under 0.002)."""
    out = []
    for e in experiments:
        out.append(e)
        if e["category"] == "baseline":
            for k in range(2, repeats + 1):
                out.append({
                    "id": "%s-r%d" % (e["id"], k),
                    "category": "baseline",
                    "description": "baseline repeat %d of %d, no changes, measuring "
                                   "run-to-run noise" % (k, repeats),
                })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", required=True)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--hours", type=float, default=6.0,
                    help="hard wall-clock budget; stops before starting a run that would exceed it")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="validate the plan and exit")
    ap.add_argument("--baseline-repeats", type=int, default=1,
                    help="run the unmodified baseline this many times; with >1 the "
                         "incumbent starts at their mean and the spread is recorded")
    ap.add_argument("--min-improvement", default="0",
                    help="a run is kept only if it beats the incumbent by more than "
                         "this; a number, or 'auto' = 2x the baseline std "
                         "(needs --baseline-repeats >= 2)")
    args = ap.parse_args()
    if args.min_improvement != "auto":
        try:
            float(args.min_improvement)
        except ValueError:
            ap.error("--min-improvement must be a number or 'auto'")
    elif args.baseline_repeats < 2:
        ap.error("--min-improvement auto needs --baseline-repeats >= 2")

    repo = Path(args.repo).resolve()
    train_py = repo / "train.py"
    tsv_path = repo / "results.tsv"
    runs_dir = repo / "runs"
    state_path = repo / "sweep_state.json"
    runs_dir.mkdir(exist_ok=True)

    experiments = add_baseline_repeats(load_plan(Path(args.plan)), args.baseline_repeats)
    log("plan: %d experiments" % len(experiments))
    if args.dry_run:
        # exercise the edit machinery against a scratch copy, then bail
        scratch = repo / ".train_py_dryrun"
        for e in experiments:
            if not e.get("edits"):
                continue
            scratch.write_text(train_py.read_text(encoding="utf-8"), encoding="utf-8")
            applied = apply_edits(scratch, e["edits"])
            log("  ok %-24s %s" % (e["id"], ", ".join(
                "%s: %s -> %s" % (k, v[0], v[1]) for k, v in applied.items())))
        scratch.unlink(missing_ok=True)
        log("dry run complete - plan is valid")
        return

    state = {"started_utc": datetime.now(timezone.utc).isoformat(),
             "plan": str(args.plan), "best_val_bpb": None, "experiments": []}
    if args.resume and state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        log("resuming: %d experiments already recorded, best=%s"
            % (len(state["experiments"]), state.get("best_val_bpb")))

    done_ids = {e["id"] for e in state["experiments"]}
    if not tsv_path.exists():
        tsv_path.write_text(TSV_HEADER + "\n", encoding="utf-8")

    def save_state():
        state_path.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")

    t_start = time.time()
    deadline = t_start + args.hours * 3600
    durations = []

    for exp in experiments:
        eid = exp["id"]
        if eid in done_ids:
            log("skip %s (already done)" % eid)
            continue

        remaining = deadline - time.time()
        predicted = (sum(durations) / len(durations)) if durations else 900.0
        if remaining < predicted:
            log("stopping: %.0fs left, next run needs ~%.0fs. %d experiments unrun."
                % (remaining, predicted, len(experiments) - len(state["experiments"])))
            break

        log("=== %s [%s] %s" % (eid, exp["category"], exp["description"]))

        # start from the current best-known state of train.py
        git(repo, "checkout", "--", "train.py")
        applied = {}
        if exp.get("edits"):
            try:
                applied = apply_edits(train_py, exp["edits"])
            except RuntimeError as err:
                log("SKIP %s - bad edit: %s" % (eid, err))
                state["experiments"].append({
                    "id": eid, "category": exp["category"], "status": "skipped",
                    "error": str(err), "description": exp["description"],
                })
                save_state()
                continue
            for k, (old, new) in applied.items():
                log("    %s: %s -> %s" % (k, old, new))

        run_log = runs_dir / ("%s.log" % eid)
        t0 = time.time()
        proc = subprocess.run(
            "export PATH=$HOME/.local/bin:$PATH && uv run train.py > %s 2>&1" % run_log.name,
            cwd=str(repo), shell=True, capture_output=True, text=True,
        )
        # train.py writes into repo root; keep the log under runs/
        produced = repo / run_log.name
        if produced.exists() and produced != run_log:
            produced.replace(run_log)
        elapsed = time.time() - t0
        durations.append(elapsed)

        parsed = parse_run_log(run_log)
        val_bpb = parsed.get("val_bpb")
        vram_gb = (parsed.get("peak_vram_mb", 0.0) or 0.0) / 1024.0

        if val_bpb is None or proc.returncode != 0:
            status = "crash"
            row_val = 0.0
            row_vram = 0.0
            log("    CRASH after %.0fs (exit %d)" % (elapsed, proc.returncode))
            if parsed.get("crash_tail"):
                log("    tail: " + parsed["crash_tail"].splitlines()[-1][:200])
        else:
            best = state.get("best_val_bpb")
            if exp["category"] == "baseline":
                # unmodified runs: nothing to keep or revert, they define the
                # incumbent (their mean) and the noise band (their std)
                improved = False
                status = "keep"
                vals = state.setdefault("baseline_vals", [])
                vals.append(val_bpb)
                state["best_val_bpb"] = sum(vals) / len(vals)
                if len(vals) > 1:
                    mean = state["best_val_bpb"]
                    state["baseline_noise_std"] = (
                        sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)) ** 0.5
                    log("    baseline n=%d mean=%.6f std=%.6f"
                        % (len(vals), mean, state["baseline_noise_std"]))
            else:
                if args.min_improvement == "auto":
                    margin = 2.0 * state.get("baseline_noise_std", 0.0)
                else:
                    margin = float(args.min_improvement)
                state["min_improvement"] = margin
                improved = best is None or val_bpb < best - margin
                status = "keep" if improved else "discard"
            row_val = val_bpb
            row_vram = vram_gb
            log("    val_bpb=%.6f vram=%.1fGB mfu=%.1f%% steps=%s  %.0fs -> %s"
                % (val_bpb, vram_gb, parsed.get("mfu_percent", 0.0),
                   parsed.get("num_steps", "?"), elapsed, status.upper()))

        if status == "keep" and exp["category"] != "baseline":
            git(repo, "add", "train.py")
            # the baseline experiment edits nothing, so there is nothing to commit;
            # committing anyway would abort the whole sweep on run 1
            if git(repo, "diff", "--cached", "--quiet", check=False).returncode != 0:
                git(repo, "commit", "-q", "-m",
                    "%s: %s" % (exp["category"], tsv_escape(exp["description"])))
            state["best_val_bpb"] = row_val
        else:
            git(repo, "checkout", "--", "train.py")

        commit = git(repo, "rev-parse", "--short", "HEAD").stdout.strip()
        with tsv_path.open("a", encoding="utf-8") as f:
            f.write("\t".join([
                commit,
                "%.6f" % row_val,
                "%.1f" % row_vram,
                status,
                tsv_escape(exp["description"]),
            ]) + "\n")

        record = {
            "id": eid,
            "category": exp["category"],
            "description": exp["description"],
            "edits": {k: {"from": v[0], "to": v[1]} for k, v in applied.items()},
            "commit": commit,
            "status": status,
            "val_bpb": row_val,
            "memory_gb": row_vram,
            "wall_seconds": round(elapsed, 1),
            "finished_utc": datetime.now(timezone.utc).isoformat(),
        }
        for key in ("training_seconds", "total_seconds", "mfu_percent",
                    "total_tokens_M", "num_steps", "num_params_M", "depth"):
            if key in parsed:
                record[key] = parsed[key]
        state["experiments"].append(record)
        save_state()

    total = time.time() - t_start
    kept = [e for e in state["experiments"] if e.get("status") == "keep"]
    crashed = [e for e in state["experiments"] if e.get("status") == "crash"]
    state["finished_utc"] = datetime.now(timezone.utc).isoformat()
    state["total_wall_seconds"] = round(total, 1)
    save_state()
    log("=== done: %d experiments in %.2fh, %d kept, %d crashed, best val_bpb=%s"
        % (len(state["experiments"]), total / 3600.0, len(kept), len(crashed),
           state.get("best_val_bpb")))
    log("results.tsv -> %s" % tsv_path)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log("interrupted")
        sys.exit(130)
