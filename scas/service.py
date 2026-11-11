"""Phase 2: the coordinator as a small HTTP service (presentation.md Part 5).

Wraps ``scas.coordinator`` for live agents. Carries the SCT *law* arms
(relaxing attractor + integrated psi/alpha gate), not the embedding-repulsion
heuristic, whose farthest-point search needs a pre-embedded finite candidate
menu that doesn't exist live. Per results-laws §11 the default coordinator is
the *emergent* one (no category router — results-review §8 measured ~43%
router mislabels on realistic text); ``--categories`` switches to the fixed
partition. Per results-laws §12 the explore side of the guidance is an
avoid-list of the group's recent accepted cards — the live approximation of
the farthest-point novelty search, which is the proven quality lever.

Protocol (all JSON):

    POST /step      the post-step report from an agent's experiment loop
        in:  {agent_id, summary, val_bpb, status?, step?, files_changed?,
              pyproject_hash?, prepare_hash?}
        out: {accepted, reject_reason, group_id, alpha, psi,
              target_card_text, avoid_list, improved_group_best, step}

    GET /state      coordinator state snapshot (debugging / dashboards)
    GET /health     {"ok": true, "mode": ..., "n_agents": ...}

The loop wrapper (Phase 2 item 2) turns the response into the pre-step
prompt packet: with probability ``alpha`` refine toward ``target_card_text``,
otherwise explore away from ``avoid_list``. ``accepted=false`` means the
safety envelope rejected the report (reason given) and coordinator state is
untouched.

Safety envelope: every report passes through ``scas.coordinator.validate``
(AC-B5'). The crash budget counts ``status=="crash"`` among the agent's five
previous reports, so a crashing agent is quarantined and recovers only after
clean reports push the crashes out of the window. By default a report that
omits the repo hashes inherits the baseline values (local trusted mode);
``--strict-envelope`` makes omission a rejection.

Run:

    uv run python -m scas.service --port 8765 --agents a0,a1,a2
    uv run python -m scas.service --categories optimizer,architecture,regularization
"""

from __future__ import annotations

import argparse
import hmac
import json
import os
import threading
import traceback
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from scas.coordinator import (
    BASELINE_PREPARE_HASH,
    BASELINE_PYPROJECT_HASH,
    DEFAULT_COUPLING_T,
    W_P,
    EmergentSCTCoordinator,
    HypothesisCard,
    SCTCoordinator,
    validate,
)

# SCT-arm defaults (sim.py applies the same when --sct is on): the committed
# psi_0=2.0 / alpha_temp=1.0 were tuned for the string proposer and squash
# alpha under the law-driven embedding proposer (results-laws §1).
SCT_PSI_0 = 0.5
SCT_ALPHA_TEMP = 0.3
CRASH_WINDOW = 5


class CoordinatorService:
    """Pure request->response logic; the HTTP layer below is a thin shell.

    ``embedder`` is injectable so tests run without MiniLM. Not thread-safe by
    itself — the HTTP layer serialises calls with a lock.
    """

    def __init__(
        self,
        agents: list[str],
        n_groups: int = 3,
        categories: list[str] | None = None,
        coupling_T: float = DEFAULT_COUPLING_T,
        w_p: float = W_P,
        psi_0: float = SCT_PSI_0,
        alpha_temp: float = SCT_ALPHA_TEMP,
        avoid_k: int = 8,
        strict_envelope: bool = False,
        embedder=None,
        min_improvement: float = 0.0,
        state_file: str | None = None,
    ):
        if embedder is None:
            from scas.embedder import Embedder

            embedder = Embedder()
        self.embedder = embedder
        self.avoid_k = avoid_k
        self.strict_envelope = strict_envelope
        self.categories = categories
        if categories:
            self.mode = "categories"
            self.coord = SCTCoordinator(
                agents, categories, coupling_T=coupling_T, w_p=w_p,
                psi_0=psi_0, alpha_temp=alpha_temp,
            )
            n_groups = len(categories)
        else:
            self.mode = "emergent"
            self.coord = EmergentSCTCoordinator(
                agents, n_groups, coupling_T=coupling_T, w_p=w_p,
                psi_0=psi_0, alpha_temp=alpha_temp,
            )
        self.n_groups = n_groups
        # Live guidance state the coordinator's embedding-space view can't
        # provide: the *text* of each group's best card (what the attractor
        # chases) and a ring buffer of recent accepted card texts (avoid-list).
        self._group_best_text: dict[int, str] = {}
        self._group_recent: dict[int, deque[str]] = {
            g: deque(maxlen=4 * avoid_k) for g in range(n_groups)
        }
        self._status_history: dict[str, deque[str]] = {
            a: deque(maxlen=CRASH_WINDOW) for a in agents
        }
        self._last_step: dict[str, int] = {}
        # Credit assignment (real-data finding, vast.ai sweep 2026-11-11): an
        # agent's val_bpb reflects everything stacked on its branch, so ranking
        # cards by absolute val_bpb made discarded experiments the group target.
        # Cards are ranked by improvement over the agent's own incumbent (its
        # best kept val_bpb), and only kept experiments may lead a group.
        self.min_improvement = float(min_improvement)
        self._incumbent: dict[str, float] = {}
        self._group_best_abs: dict[int, float] = {}
        # Best val_bpb accepted from any agent: the relative plausibility floor.
        self._best_accepted: float | None = None
        self.state_file = Path(state_file) if state_file else None
        if self.state_file is not None and self.state_file.exists():
            self._load(json.loads(self.state_file.read_text(encoding="utf-8")))

    # ------------------------------------------------------------- persistence

    def dump(self) -> dict:
        """Full restartable state (JSON-safe)."""
        st = self.coord.state

        def vecs(d):
            return {str(g): [float(x) for x in v] for g, v in d.items()}

        return {
            "version": 1,
            "mode": self.mode,
            "n_groups": self.n_groups,
            "agent_to_group": dict(self.coord.agent_to_group),
            "groups": {str(g): list(m) for g, m in st.groups.items()},
            "attractors": vecs(st.attractors),
            "best_emb": vecs(self.coord._best_emb),
            "group_best_score": {
                str(g): (None if v == float("inf") else v)
                for g, v in st.group_best_val_bpb.items()
            },
            "psi": dict(st.psi),
            "alpha": dict(st.alpha),
            "group_best_text": {str(g): t for g, t in self._group_best_text.items()},
            "group_best_abs": {str(g): v for g, v in self._group_best_abs.items()},
            "group_recent": {str(g): list(d) for g, d in self._group_recent.items()},
            "status_history": {a: list(d) for a, d in self._status_history.items()},
            "last_step": dict(self._last_step),
            "incumbent": dict(self._incumbent),
            "best_accepted": self._best_accepted,
        }

    def _load(self, d: dict) -> None:
        import numpy as np

        if d.get("mode") != self.mode or d.get("n_groups") != self.n_groups:
            raise ValueError(
                "state file was written by a %s/%s-group service; this one is %s/%s"
                % (d.get("mode"), d.get("n_groups"), self.mode, self.n_groups)
            )
        st = self.coord.state
        self.coord.agent_to_group = dict(d["agent_to_group"])
        st.groups = {int(g): list(m) for g, m in d["groups"].items()}
        st.attractors = {
            int(g): np.asarray(v, dtype=np.float32) for g, v in d["attractors"].items()
        }
        self.coord._best_emb = {
            int(g): np.asarray(v, dtype=np.float32) for g, v in d["best_emb"].items()
        }
        st.group_best_val_bpb = {
            int(g): (float("inf") if v is None else float(v))
            for g, v in d["group_best_score"].items()
        }
        st.psi = dict(d["psi"])
        st.alpha = dict(d["alpha"])
        self._group_best_text = {int(g): t for g, t in d["group_best_text"].items()}
        self._group_best_abs = {int(g): v for g, v in d["group_best_abs"].items()}
        for g, items in d["group_recent"].items():
            self._group_recent[int(g)] = deque(items, maxlen=4 * self.avoid_k)
        self._status_history = {
            a: deque(items, maxlen=CRASH_WINDOW)
            for a, items in d["status_history"].items()
        }
        self._last_step = dict(d["last_step"])
        self._incumbent = dict(d["incumbent"])
        self._best_accepted = d.get("best_accepted")

    def _persist(self) -> None:
        if self.state_file is None:
            return
        tmp = self.state_file.with_name(self.state_file.name + ".tmp")
        tmp.write_text(json.dumps(self.dump()), encoding="utf-8")
        tmp.replace(self.state_file)

    # ------------------------------------------------------------- registration

    def _register(self, agent_id: str) -> None:
        """Auto-register an unseen agent (live swarms gain/lose members)."""
        if agent_id in self.coord.agent_to_group:
            return
        if self.mode == "emergent":
            self.coord.agent_to_group[agent_id] = None
        else:
            g = len(self.coord.agent_to_group) % self.n_groups
            self.coord.agent_to_group[agent_id] = g
            self.coord.state.groups[g].append(agent_id)
        self._status_history[agent_id] = deque(maxlen=CRASH_WINDOW)

    # -------------------------------------------------------------------- step

    def step(self, report: dict) -> tuple[int, dict]:
        """Handle one POST /step report. Returns (http_status, response)."""
        if not isinstance(report, dict):
            return 400, {"error": "body must be a JSON object"}
        agent_id = report.get("agent_id")
        summary = report.get("summary")
        if not isinstance(agent_id, str) or not agent_id:
            return 400, {"error": "agent_id (non-empty string) is required"}
        if not isinstance(summary, str) or not summary:
            return 400, {"error": "summary (non-empty string) is required"}
        try:
            val_bpb = float(report.get("val_bpb"))
        except (TypeError, ValueError):
            return 400, {"error": "val_bpb (number) is required"}
        status = report.get("status", "keep")
        if status not in ("keep", "discard", "crash"):
            return 400, {"error": f"status must be keep|discard|crash, got {status!r}"}

        self._register(agent_id)
        step = int(report.get("step", self._last_step.get(agent_id, -1) + 1))

        # Safety envelope (AC-B5'). Hash omission is trusted locally unless
        # --strict-envelope; the crash budget looks at the five *previous*
        # reports, so this report's own status can't save or damn it.
        commit = {
            "files_changed": report.get("files_changed", ["train.py"]),
            "pyproject_hash": report.get("pyproject_hash"),
            "prepare_hash": report.get("prepare_hash"),
            "val_bpb": None if status == "crash" else val_bpb,
        }
        if not self.strict_envelope:
            if commit["pyproject_hash"] is None:
                commit["pyproject_hash"] = BASELINE_PYPROJECT_HASH
            if commit["prepare_hash"] is None:
                commit["prepare_hash"] = BASELINE_PREPARE_HASH
        recent_failures = sum(
            1 for s in self._status_history[agent_id] if s == "crash"
        )
        accepted, reject_reason = validate(
            commit, recent_failures, reference_best=self._best_accepted
        )

        self._status_history[agent_id].append(status)
        self._last_step[agent_id] = step

        # Improvement over the agent's own incumbent. The agent may state the
        # incumbent explicitly (joining mid-run, or after a service restart);
        # otherwise it is the best val_bpb this agent has kept so far.
        incumbent = report.get("incumbent_val_bpb")
        if incumbent is not None:
            try:
                incumbent = float(incumbent)
            except (TypeError, ValueError):
                return 400, {"error": "incumbent_val_bpb must be a number"}
        else:
            incumbent = self._incumbent.get(agent_id)
            if incumbent is None:
                # an agent with no history of its own is measured against the
                # best result the swarm has accepted so far
                incumbent = self._best_accepted
        delta = 0.0 if incumbent is None else val_bpb - incumbent
        can_lead = status == "keep" and (
            incumbent is None or delta < -self.min_improvement
        )

        improved = False
        if accepted:
            embedding = self.embedder.embed(summary)
            g_cat = self.coord.group_to_category.get(
                self.coord.agent_to_group.get(agent_id)
            )
            card = HypothesisCard(
                agent_id=agent_id,
                step=step,
                summary=summary,
                category=g_cat if g_cat is not None else "emergent",
                parent_step=report.get("parent_step"),
            )
            improved = self.coord.observe(
                card, delta, embedding, accepted=True, can_lead=can_lead
            )
            g = self.coord.agent_to_group[agent_id]
            if improved:
                self._group_best_text[g] = summary
                self._group_best_abs[g] = val_bpb
            self._group_recent[g].append(summary)
            if status == "keep":
                prev = self._incumbent.get(agent_id)
                self._incumbent[agent_id] = val_bpb if prev is None else min(prev, val_bpb)
            if self._best_accepted is None or val_bpb < self._best_accepted:
                self._best_accepted = val_bpb
        self._persist()

        return 200, {
            "agent_id": agent_id,
            "step": step,
            "accepted": accepted,
            "reject_reason": reject_reason,
            **self._guidance(agent_id),
            "improved_group_best": improved,
            "delta_vs_incumbent": delta if accepted else None,
            "incumbent_val_bpb": self._incumbent.get(agent_id),
        }

    def _guidance(self, agent_id: str) -> dict:
        """Current guidance for an agent: gate value, target, avoid-list."""
        g = self.coord.agent_to_group.get(agent_id)
        target = self._group_best_text.get(g) if g is not None else None
        avoid: list[str] = []
        if g is not None:
            seen = set()
            for s in reversed(self._group_recent[g]):  # most recent first
                if s != target and s not in seen:
                    avoid.append(s)
                    seen.add(s)
                if len(avoid) >= self.avoid_k:
                    break
        return {
            "group_id": g,
            "alpha": self.coord.state.alpha.get(agent_id),
            "psi": self.coord.state.psi.get(agent_id),
            "target_card_text": target,
            "avoid_list": avoid,
        }

    # ------------------------------------------------------------------- state

    def snapshot(self) -> dict:
        st = self.coord.state
        return {
            "mode": self.mode,
            "n_groups": self.n_groups,
            "agents": sorted(self.coord.agent_to_group),
            "agent_to_group": dict(self.coord.agent_to_group),
            "groups": {str(g): m for g, m in st.groups.items()},
            "group_to_category": {
                str(g): c for g, c in self.coord.group_to_category.items()
            },
            # absolute val_bpb of each group's target card ...
            "group_best_val_bpb": {
                str(g): self._group_best_abs.get(g) for g in st.group_best_val_bpb
            },
            # ... and the score it is ranked by: improvement over its agent's incumbent
            "group_best_delta": {
                str(g): (None if v == float("inf") else v)
                for g, v in st.group_best_val_bpb.items()
            },
            "incumbent_val_bpb": dict(self._incumbent),
            "best_accepted_val_bpb": self._best_accepted,
            "min_improvement": self.min_improvement,
            "group_best_text": {str(g): t for g, t in self._group_best_text.items()},
            "psi": dict(st.psi),
            "alpha": dict(st.alpha),
            "coupling_T": self.coord.coupling_T,
            "psi_0": self.coord.psi_0,
            "alpha_temp": self.coord.alpha_temp,
            "w_p": self.coord.w_p,
            "avoid_k": self.avoid_k,
            "strict_envelope": self.strict_envelope,
        }


# ------------------------------------------------------------------ HTTP layer


def make_server(
    service: CoordinatorService, host: str, port: int, token: str | None = None
) -> ThreadingHTTPServer:
    """``token`` (if set) is required as ``Authorization: Bearer <token>`` on
    everything except ``/health``."""
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def _authorized(self) -> bool:
            if not token:
                return True
            got = self.headers.get("Authorization", "")
            if hmac.compare_digest(got.encode(), ("Bearer " + token).encode()):
                return True
            self._send(401, {"error": "missing or wrong bearer token"})
            return False

        def _send(self, code: int, payload: dict) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _guarded(self, fn) -> None:
            """Run a route; an internal error becomes a JSON 500, not a
            dropped connection (the smoke test caught exactly that)."""
            try:
                fn()
            except Exception as e:  # noqa: BLE001 — boundary of the process
                traceback.print_exc()
                try:
                    self._send(500, {"error": f"{type(e).__name__}: {e}"})
                except Exception:
                    pass  # client already gone

        def do_GET(self):  # noqa: N802 (stdlib API)
            self._guarded(self._get)

        def _get(self) -> None:
            if self.path == "/health":
                with lock:
                    n = len(service.coord.agent_to_group)
                self._send(200, {"ok": True, "mode": service.mode, "n_agents": n})
            elif self.path == "/state":
                if not self._authorized():
                    return
                with lock:
                    snap = service.snapshot()
                self._send(200, snap)
            else:
                self._send(404, {"error": f"unknown path {self.path}"})

        def do_POST(self):  # noqa: N802 (stdlib API)
            self._guarded(self._post)

        def _post(self) -> None:
            if self.path != "/step":
                self._send(404, {"error": f"unknown path {self.path}"})
                return
            if not self._authorized():
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                report = json.loads(self.rfile.read(length) or b"null")
            except (ValueError, json.JSONDecodeError):
                self._send(400, {"error": "malformed JSON body"})
                return
            with lock:
                code, payload = service.step(report)
            self._send(code, payload)

        def log_message(self, fmt, *args):  # quiet by default
            pass

    return ThreadingHTTPServer((host, port), Handler)


def main() -> None:
    p = argparse.ArgumentParser(description="SCAS live coordinator service")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument(
        "--agents", default="a0,a1,a2",
        help="comma-separated initial agent ids (unknown ids auto-register)",
    )
    p.add_argument(
        "--n-groups", type=int, default=3, dest="n_groups",
        help="emergent-mode group slots (ignored with --categories)",
    )
    p.add_argument(
        "--categories", default=None,
        help="comma-separated category list; switches from the emergent "
             "coordinator (default, results-laws §11) to the fixed partition",
    )
    p.add_argument("--coupling-T", type=float, default=DEFAULT_COUPLING_T,
                   dest="coupling_T")
    p.add_argument("--w-p", type=float, default=W_P, dest="w_p")
    p.add_argument("--psi-0", type=float, default=SCT_PSI_0, dest="psi_0")
    p.add_argument("--alpha-temp", type=float, default=SCT_ALPHA_TEMP,
                   dest="alpha_temp")
    p.add_argument("--avoid-k", type=int, default=8, dest="avoid_k",
                   help="max avoid-list length returned to agents")
    p.add_argument(
        "--strict-envelope", action="store_true", dest="strict_envelope",
        help="reject reports that omit pyproject/prepare hashes instead of "
             "trusting the local baseline",
    )
    p.add_argument(
        "--min-improvement", type=float, default=0.0, dest="min_improvement",
        help="a kept experiment must beat its agent's incumbent by more than "
             "this to become a group target (set to the measured run-to-run "
             "noise of val_bpb)",
    )
    p.add_argument(
        "--state-file", default=None, dest="state_file",
        help="persist coordinator state here after every report and reload it "
             "on start, so a restart does not lose the swarm's history",
    )
    p.add_argument(
        "--token", default=os.environ.get("SCAS_TOKEN"),
        help="shared secret required as a Bearer token (default: $SCAS_TOKEN). "
             "Mandatory when --host is not loopback",
    )
    args = p.parse_args()
    if args.host not in ("127.0.0.1", "localhost", "::1") and not args.token:
        p.error("--host %s exposes the coordinator beyond this machine; set "
                "--token or $SCAS_TOKEN" % args.host)

    agents = [a.strip() for a in args.agents.split(",") if a.strip()]
    categories = (
        [c.strip() for c in args.categories.split(",") if c.strip()]
        if args.categories else None
    )
    service = CoordinatorService(
        agents=agents,
        n_groups=args.n_groups,
        categories=categories,
        coupling_T=args.coupling_T,
        w_p=args.w_p,
        psi_0=args.psi_0,
        alpha_temp=args.alpha_temp,
        avoid_k=args.avoid_k,
        strict_envelope=args.strict_envelope,
        min_improvement=args.min_improvement,
        state_file=args.state_file,
    )
    server = make_server(service, args.host, args.port, token=args.token)
    print(json.dumps({
        "listening": f"http://{args.host}:{server.server_address[1]}",
        "auth": "bearer token" if args.token else "none (loopback only)",
        "state_file": args.state_file,
        "mode": service.mode,
        "agents": agents,
        "endpoints": ["POST /step", "GET /state", "GET /health"],
    }))
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.shutdown()


if __name__ == "__main__":
    main()
