"""Phase 2: the autoresearch loop wrapper (presentation.md Part 5, item 2).

Bridges the ``program.md`` experiment loop and the live coordinator service
(``scas/service.py``). Two commands slot into the loop:

    # before hacking train.py (loop step 2): get the coordination directive
    uv run python -m scas.loop prompt --agent a0

    # after logging results.tsv (loop step 7, BEFORE any git reset):
    uv run python -m scas.loop report --agent a0 --from-tsv results.tsv
    # ...or explicitly:
    uv run python -m scas.loop report --agent a0 --summary "..." \
        --val-bpb 0.9891 --status keep

``prompt`` fetches the agent's last guidance, draws ``bernoulli(alpha)`` —
the *same* gate the synthetic proposer uses, so the live/sim correspondence
is exact — and prints either a REFINE packet (toward ``target_card_text``)
or an EXPLORE packet (away from ``avoid_list``). ``report`` computes the
repo hashes, posts to ``POST /step``, persists the returned guidance for the
next ``prompt``, and exits 0 on accept / 2 on envelope reject.

Every directive and report is appended to ``.scas_loop/<agent>_trace.jsonl``.
That trace is the measuring instrument for the key Phase 2 unknown — the
LLM's *effective* alpha:

    uv run python -m scas.loop analyze --agent a0

joins each directive with the experiment that followed it and scores, in
MiniLM space, whether REFINE steps actually landed nearer the target than
the avoid-list and EXPLORE steps landed away (compliance rates + steering
margins). If prompts steer the live agent the way ``bernoulli(alpha)``
steers the synthetic one, refine compliance is high and the refine-explore
margin separation is positive.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

from scas.coordinator import _REPO_ROOT, _file_sha256

DEFAULT_SERVER = "http://127.0.0.1:8765"
DEFAULT_STATE_DIR = Path(".scas_loop")


# ------------------------------------------------------------------ rendering


def draw_directive(guidance: dict | None, rng: random.Random) -> dict:
    """Sample the exploit/explore gate exactly like the synthetic proposer:
    ``refine`` with probability alpha (needs a target), else ``explore``."""
    g = guidance or {}
    alpha = g.get("alpha")
    target = g.get("target_card_text")
    avoid = list(g.get("avoid_list") or [])
    if alpha is None or target is None:
        return {"mode": "explore", "alpha": alpha, "target": None, "avoid": avoid,
                "note": "no guidance yet - free exploration"}
    mode = "refine" if rng.random() < float(alpha) else "explore"
    return {"mode": mode, "alpha": float(alpha), "target": target, "avoid": avoid}


def render_packet(agent_id: str, directive: dict, directive_no: int) -> str:
    d = directive
    alpha_s = "n/a" if d["alpha"] is None else f"{d['alpha']:.3f}"
    lines = [
        f"=== SCAS COORDINATION PACKET (agent {agent_id}, directive #{directive_no}) ===",
    ]
    if d["mode"] == "refine":
        lines += [
            f"DIRECTIVE: REFINE (alpha={alpha_s} -> exploit)",
            "Your group's current best experiment:",
            f'    "{d["target"]}"',
            "Design your next experiment as a refinement or close variation of",
            "that idea: same mechanism, adjusted knob, scope, or combination.",
            "Stay near it in idea space.",
        ]
    else:
        lines += [
            f"DIRECTIVE: EXPLORE (alpha={alpha_s} -> diversify)",
            "Propose something qualitatively different from your group's recent",
            "work: a fresh mechanism, not a knob tweak on an existing direction.",
        ]
        if d.get("note"):
            lines.append(f"({d['note']})")
    if d["avoid"]:
        lines.append("Do not repeat or trivially rephrase these recent group experiments:")
        lines += [f'    - "{s}"' for s in d["avoid"]]
    lines.append("After the run, report with: uv run python -m scas.loop report "
                 f"--agent {agent_id} --from-tsv results.tsv")
    return "\n".join(lines)


# ------------------------------------------------------------------ the client


class LoopClient:
    """Stateful bridge for one agent. State/trace live under ``state_dir`` so
    the loop can be interrupted and resumed (program.md runs indefinitely)."""

    def __init__(
        self,
        agent_id: str,
        server: str = DEFAULT_SERVER,
        state_dir: Path = DEFAULT_STATE_DIR,
        rng: random.Random | None = None,
        token: str | None = None,
    ):
        self.agent_id = agent_id
        self.server = server.rstrip("/")
        self.token = token
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.state_path = self.state_dir / f"{agent_id}.json"
        self.trace_path = self.state_dir / f"{agent_id}_trace.jsonl"
        self.rng = rng if rng is not None else random.Random()

    # ------------------------------------------------------------- persistence

    def _load_state(self) -> dict | None:
        if self.state_path.exists():
            return json.loads(self.state_path.read_text())
        return None

    def _save_state(self, response: dict) -> None:
        self.state_path.write_text(json.dumps(response, indent=2))

    def _trace(self, kind: str, payload: dict) -> None:
        with self.trace_path.open("a") as f:
            f.write(json.dumps({"kind": kind, "t": time.time(), **payload}) + "\n")

    def _trace_rows(self) -> list[dict]:
        if not self.trace_path.exists():
            return []
        return [json.loads(x) for x in self.trace_path.read_text().splitlines()]

    # ------------------------------------------------------------------ prompt

    def prompt(self) -> str:
        guidance = self._load_state()
        directive = draw_directive(guidance, self.rng)
        n = 1 + sum(1 for r in self._trace_rows() if r["kind"] == "directive")
        self._trace("directive", directive)
        return render_packet(self.agent_id, directive, n)

    # ------------------------------------------------------------------ report

    def report(
        self,
        summary: str,
        val_bpb: float,
        status: str = "keep",
        step: int | None = None,
        files_changed: list[str] | None = None,
        incumbent: float | None = None,
    ) -> dict:
        import requests

        if files_changed is None:
            files_changed = _git_files_changed()
        prev = self._load_state()
        payload = {
            "agent_id": self.agent_id,
            "summary": summary,
            "val_bpb": val_bpb,
            "status": status,
            "files_changed": files_changed,
            "pyproject_hash": _file_sha256(_REPO_ROOT / "pyproject.toml"),
            "prepare_hash": _file_sha256(_REPO_ROOT / "prepare.py"),
        }
        if step is not None:
            payload["step"] = step
        if incumbent is not None:
            # best val_bpb this agent had kept BEFORE this experiment; the
            # coordinator ranks the card by its improvement over that
            payload["incumbent_val_bpb"] = incumbent
        if prev is not None and prev.get("step") is not None:
            payload["parent_step"] = prev["step"]
        headers = {"Authorization": f"Bearer {self.token}"} if self.token else {}
        r = requests.post(f"{self.server}/step", json=payload, headers=headers,
                          timeout=120)
        response = r.json()
        if r.status_code != 200:
            raise RuntimeError(f"service returned {r.status_code}: {response}")
        self._save_state(response)
        self._trace("report", {"summary": summary, "val_bpb": val_bpb,
                               "status": status, "response": response})
        return response


def _git_files_changed() -> list[str]:
    """Files changed by the experiment commit (program.md commits before
    running, and ``report`` runs before any reset — so HEAD is the experiment)."""
    try:
        out = subprocess.run(
            ["git", "diff", "--name-only", "HEAD~1", "HEAD"],
            capture_output=True, text=True, check=True, cwd=str(_REPO_ROOT),
        ).stdout
        files = [x.strip() for x in out.splitlines() if x.strip()]
        return files or ["train.py"]
    except (subprocess.CalledProcessError, FileNotFoundError):
        return ["train.py"]


def read_last_tsv_row(path: Path) -> dict:
    with Path(path).open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    if not rows:
        raise ValueError(f"{path} has no data rows")
    r = rows[-1]

    def _val(row) -> float:
        try:
            return float(row.get("val_bpb", "0") or "0")
        except ValueError:
            return 0.0

    kept_before = [_val(x) for x in rows[:-1]
                   if x.get("status") == "keep" and _val(x) > 0.0]
    return {
        "summary": r["description"],
        "val_bpb": _val(r),
        "status": r.get("status", "keep"),
        "incumbent": min(kept_before) if kept_before else None,
    }


# ----------------------------------------------------- effective-alpha analysis


def analyze_trace(trace_path: Path, embedder) -> dict:
    """Join each directive with the report that followed it and measure, in
    embedding space, whether the agent obeyed.

    steering margin = cos(new_card, target) - mean cos(new_card, avoid_list)
    compliance      = margin > 0 for REFINE, margin < 0 for EXPLORE

    Only directives with a target and a non-empty avoid list are scorable
    (early free-exploration steps have nothing to steer against).
    """
    import numpy as np

    rows = [json.loads(x) for x in Path(trace_path).read_text().splitlines()]
    pairs: list[tuple[dict, dict]] = []
    pending: dict | None = None
    for r in rows:
        if r["kind"] == "directive":
            pending = r
        elif r["kind"] == "report" and pending is not None:
            pairs.append((pending, r))
            pending = None

    margins: dict[str, list[float]] = {"refine": [], "explore": []}
    compliant: dict[str, int] = {"refine": 0, "explore": 0}
    for d, rep in pairs:
        if not d.get("target") or not d.get("avoid"):
            continue
        z = embedder.embed(rep["summary"])
        t = embedder.embed(d["target"])
        avoid = np.stack([embedder.embed(s) for s in d["avoid"]])
        margin = float(z @ t) - float(np.mean(avoid @ z))
        mode = d["mode"]
        margins[mode].append(margin)
        if (mode == "refine" and margin > 0) or (mode == "explore" and margin < 0):
            compliant[mode] += 1

    def _stats(mode: str) -> dict:
        m = margins[mode]
        return {
            "n": len(m),
            "mean_margin": float(np.mean(m)) if m else None,
            "compliance": (compliant[mode] / len(m)) if m else None,
        }

    ref, exp = _stats("refine"), _stats("explore")
    separation = (
        ref["mean_margin"] - exp["mean_margin"]
        if ref["mean_margin"] is not None and exp["mean_margin"] is not None
        else None
    )
    return {
        "n_pairs_total": len(pairs),
        "refine": ref,
        "explore": exp,
        "refine_minus_explore_margin": separation,
        "note": "positive separation + high compliance => prompts steer the "
                "live agent the way bernoulli(alpha) steers the synthetic one",
    }


# ------------------------------------------------------------------------ CLI


def main() -> None:
    p = argparse.ArgumentParser(description="SCAS autoresearch loop wrapper")
    p.add_argument("--agent", required=True, dest="agent_id")
    p.add_argument("--server", default=DEFAULT_SERVER)
    p.add_argument("--state-dir", default=str(DEFAULT_STATE_DIR), dest="state_dir")
    p.add_argument("--token", default=os.environ.get("SCAS_TOKEN"),
                   help="bearer token for a remote coordinator (default: $SCAS_TOKEN)")
    p.add_argument("--seed", type=int, default=None,
                   help="seed the bernoulli(alpha) draw (reproducible packets)")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("prompt", help="print the pre-step coordination packet")

    rp = sub.add_parser("report", help="send the post-step report")
    rp.add_argument("--from-tsv", default=None, dest="from_tsv",
                    help="take summary/val_bpb/status from the last row of this results.tsv")
    rp.add_argument("--summary", default=None)
    rp.add_argument("--val-bpb", type=float, default=None, dest="val_bpb")
    rp.add_argument("--status", default="keep", choices=["keep", "discard", "crash"])
    rp.add_argument("--step", type=int, default=None)
    rp.add_argument("--incumbent", type=float, default=None,
                    help="best val_bpb kept before this experiment "
                         "(--from-tsv derives it from the earlier keep rows)")
    rp.add_argument("--files-changed", default=None, dest="files_changed",
                    help="comma-separated override (default: git diff HEAD~1 HEAD)")

    sub.add_parser("analyze", help="effective-alpha analysis of the trace")

    args = p.parse_args()
    rng = random.Random(args.seed) if args.seed is not None else random.Random()
    client = LoopClient(args.agent_id, server=args.server,
                        state_dir=Path(args.state_dir), rng=rng, token=args.token)

    if args.cmd == "prompt":
        print(client.prompt())
        return

    if args.cmd == "report":
        if args.from_tsv:
            fields = read_last_tsv_row(Path(args.from_tsv))
        elif args.summary is not None and args.val_bpb is not None:
            fields = {"summary": args.summary, "val_bpb": args.val_bpb,
                      "status": args.status}
        if args.incumbent is not None:
            fields["incumbent"] = args.incumbent
        else:
            p.error("report needs --from-tsv or both --summary and --val-bpb")
        files = ([f.strip() for f in args.files_changed.split(",") if f.strip()]
                 if args.files_changed else None)
        resp = client.report(step=args.step, files_changed=files, **fields)
        print(json.dumps(resp, indent=2))
        if not resp["accepted"]:
            print(f"REJECTED by safety envelope: {resp['reject_reason']} — "
                  "this result was NOT counted; fix the violation and continue.",
                  file=sys.stderr)
            sys.exit(2)
        return

    if args.cmd == "analyze":
        from scas.embedder import Embedder

        report = analyze_trace(client.trace_path, Embedder())
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
