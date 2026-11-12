#!/usr/bin/env python3
"""S2: coordinated vs independent vs trivial-hub vs one-agent-with-K-times-the-time.

Runs ON YOUR MACHINE. Each agent is a fresh headless `claude -p` session that
edits a private copy of train.py; the edit is shipped to a rented GPU box,
trained for the fixed 5-minute budget, and kept or reverted exactly as
program.md prescribes. The arms differ only in what the agent is told about
the other agents:

    scas         K agents, guidance packets from the live SCAS coordinator
    independent  K agents, nothing shared
    hub          K agents, a shared plain-text list of everything tried so far
    single       1 agent with K times the wall clock (review concern C3)

The three K-agent arms run one after another on the SAME boxes, so box speed
is not confounded with the arm. `single` runs concurrently on one more box.

    uv run python tools/vast/swarm.py up --boxes 4
    uv run python tools/vast/swarm.py run --hours 2
    uv run python tools/vast/swarm.py down          # run does this itself

`run --fake` exercises the whole loop (real agent calls, real coordinator)
against a stub trainer, without any GPU.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import shutil
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO_ROOT))

import vastctl  # noqa: E402

SWARM_STATE = HERE / ".swarm_state.json"
REMOTE_REPO = vastctl.REMOTE_REPO
REMOTE_LOGS = "/workspace/swarm_logs"

# 2x the run-to-run std measured by three baseline repeats on a rented 4090
# (0.00022, sweep of 2026-11-11), rounded up.
KEEP_MARGIN = 0.0005
TRAIN_TIMEOUT_S = 720       # program.md: kill anything past ~10 minutes
POLL_S = 20
CYCLE_RESERVE_S = 9 * 60    # do not start an experiment that cannot finish
ARMS = ("scas", "independent", "hub", "single")

_print_lock = threading.Lock()


def log(msg):
    with _print_lock:
        print("[%s] %s" % (time.strftime("%H:%M:%S"), msg), flush=True)


# --------------------------------------------------------------------------
# boxes
# --------------------------------------------------------------------------

SUMMARY_FIELDS = {
    "val_bpb": float, "peak_vram_mb": float, "training_seconds": float,
    "total_seconds": float, "mfu_percent": float, "num_steps": int,
    "num_params_M": float, "depth": int,
}


def parse_summary(text: str) -> dict:
    out = {}
    for key, cast in SUMMARY_FIELDS.items():
        m = re.search(r"^%s:\s*([0-9.]+)\s*$" % re.escape(key), text, re.MULTILINE)
        if m:
            try:
                out[key] = cast(float(m.group(1)))
            except ValueError:
                pass
    return out


class Box:
    """One rented instance. All calls are plain ssh; nothing is left attached."""

    def __init__(self, st: dict):
        self.st = st
        self.name = "box%s" % st["instance_id"]

    def sh(self, cmd: str, timeout: int = 120):
        try:
            p = subprocess.run(vastctl.ssh_base(self.st) + [cmd], capture_output=True,
                               text=True, timeout=timeout, stdin=subprocess.DEVNULL,
                               encoding="utf-8", errors="replace")
            return p.returncode, (p.stdout or "")
        except subprocess.TimeoutExpired:
            return 255, ""

    def put(self, local: Path, remote: str) -> bool:
        args = ["scp", "-i", str(vastctl.SSH_KEY), "-P", str(self.st["ssh_port"]),
                "-o", "StrictHostKeyChecking=accept-new",
                "-o", "UserKnownHostsFile=" + str(vastctl.KNOWN_HOSTS),
                "-o", "LogLevel=ERROR", str(local),
                "root@%s:%s" % (self.st["ssh_host"], remote)]
        for _ in range(4):
            try:
                if subprocess.run(args, capture_output=True, timeout=120,
                                  stdin=subprocess.DEVNULL).returncode == 0:
                    return True
            except subprocess.TimeoutExpired:
                pass
            time.sleep(10)
        return False

    def fetch_train_py(self, dest: Path) -> None:
        rc, out = self.sh("cd %s && git checkout -q -- train.py && cat train.py" % REMOTE_REPO)
        if rc != 0 or "val_bpb" not in out:
            raise RuntimeError("%s: could not read the pristine train.py" % self.name)
        dest.write_text(out, encoding="utf-8", newline="\n")

    def train(self, train_py: Path, tag: str) -> dict:
        """Ship train.py, run it detached, poll for the exit marker."""
        if not self.put(train_py, REMOTE_REPO + "/train.py"):
            return {"error": "upload failed"}
        logf = "%s/%s.log" % (REMOTE_LOGS, tag)
        launch = (
            "mkdir -p %s && rm -f %s.exit && cd %s && "
            "(setsid nohup bash -c 'export PATH=$HOME/.local/bin:$PATH; "
            "timeout %d uv run train.py > %s 2>&1; echo $? > %s.exit' "
            "> /dev/null 2>&1 < /dev/null &) ; echo started"
            % (REMOTE_LOGS, logf, REMOTE_REPO, TRAIN_TIMEOUT_S, logf, logf)
        )
        for _ in range(4):
            rc, out = self.sh(launch)
            if rc == 0 and "started" in out:
                break
            time.sleep(10)
        else:
            return {"error": "launch failed"}
        hard_stop = time.time() + TRAIN_TIMEOUT_S + 240
        exit_code = None
        while time.time() < hard_stop:
            time.sleep(POLL_S)
            rc, out = self.sh("cat %s.exit 2>/dev/null || echo PENDING" % logf, timeout=60)
            word = out.strip().splitlines()[-1] if out.strip() else ""
            if rc == 0 and word.isdigit():
                exit_code = int(word)
                break
        if exit_code is None:
            return {"error": "no exit marker (box unreachable or run hung)"}
        rc, tail = self.sh("tail -n 60 %s" % logf, timeout=60)
        res = parse_summary(tail)
        res["exit_code"] = exit_code
        res["log_tail"] = tail
        if "val_bpb" not in res:
            res["error"] = "timeout" if exit_code == 124 else "crash"
        return res


class FakeBox:
    """Stub trainer for --fake: no GPU, a few seconds per run."""

    def __init__(self, i: int):
        self.name = "fake%d" % i
        self.st = {"instance_id": "fake%d" % i, "dph_total": 0.0}
        self.rng = random.Random(100 + i)
        self.level = 0.576

    def fetch_train_py(self, dest: Path) -> None:
        dest.write_text((REPO_ROOT / "train.py").read_text(encoding="utf-8"),
                        encoding="utf-8", newline="\n")

    def train(self, train_py: Path, tag: str) -> dict:
        time.sleep(3)
        if tag.endswith("_000"):
            val = self.level
        else:
            val = self.level + self.rng.uniform(-0.012, 0.02)
            if self.rng.random() < 0.1:
                return {"error": "crash", "exit_code": 1,
                        "log_tail": "RuntimeError: fake shape mismatch"}
        self.level = min(self.level, val)
        return {"val_bpb": round(val, 6), "peak_vram_mb": 6300.0, "num_steps": 300,
                "exit_code": 0, "log_tail": "val_bpb: %.6f" % val}


# --------------------------------------------------------------------------
# the agent
# --------------------------------------------------------------------------

RULES = """You are an autonomous ML researcher running one step of an experiment loop.

The file train.py in the current directory trains a small language model for a
FIXED budget of 5 minutes of wall-clock training on one RTX 4090 (24 GB), then
prints val_bpb (validation bits per byte). LOWER val_bpb IS BETTER. That is the
only goal.

Rules:
- train.py is the only file you may edit. Everything in it is fair game: model
  architecture, optimizer, hyperparameters, schedule, training loop, model size.
- You cannot add dependencies and you cannot change the evaluation or the data.
- The run must finish without crashing. VRAM is a soft limit (24 GB card).
- Simpler is better at equal val_bpb.
- On this GPU, DEVICE_BATCH_SIZE and EVAL_BATCH_SIZE are overridden by an
  autotuner: editing those two constants changes nothing. Do not spend an
  experiment on them.
- train.py currently contains every change marked "keep" below. Changes marked
  "discard" or "crash" were reverted.
"""

TASK = """Now do exactly this:
1. Read train.py as needed.
2. Make ONE experimental change to train.py with the Edit tool. One idea per
   experiment, so its effect can be attributed.
3. Do not run anything and do not create other files.
4. End your reply with a single line in this exact form:
DESCRIPTION: <one sentence: what you changed and why you expect it to help>
"""


def history_block(hist: list[dict]) -> str:
    if not hist:
        return "No experiments yet."
    best = min((h["val_bpb"] for h in hist if h["status"] == "keep"), default=None)
    lines = ["Your experiment history so far (oldest first):",
             "  #  status   val_bpb   steps  description"]
    for h in hist:
        v = "%.6f" % h["val_bpb"] if h["status"] != "crash" else "   -    "
        extra = ""
        if h["status"] == "crash" and h.get("error_line"):
            extra = "   [error: %s]" % h["error_line"][:160]
        lines.append("%3d  %-7s  %s  %5s  %s%s"
                     % (h["n"], h["status"], v, h.get("num_steps") or "-",
                        h["description"], extra))
    if best is not None:
        lines.append("Current best (what train.py reproduces): val_bpb %.6f" % best)
    return "\n".join(lines)


def find_claude() -> str:
    exe = shutil.which("claude")
    if not exe:
        raise RuntimeError("`claude` CLI not found on PATH")
    return exe


def propose(workdir: Path, prompt: str, model: str, timeout: int = 900) -> dict:
    """One fresh headless session. It can only read and edit files in workdir."""
    cmd = [find_claude(), "-p", "--model", model,
           "--tools", "Read,Edit,Grep",
           "--permission-mode", "acceptEdits",
           "--output-format", "json",
           "--no-session-persistence"]
    t0 = time.time()
    try:
        p = subprocess.run(cmd, input=prompt, capture_output=True, text=True,
                           cwd=str(workdir), timeout=timeout,
                           encoding="utf-8", errors="replace")
    except subprocess.TimeoutExpired:
        return {"error": "agent timed out", "secs": time.time() - t0}
    out = {"secs": round(time.time() - t0, 1)}
    try:
        data = json.loads(p.stdout)
    except (json.JSONDecodeError, TypeError):
        out["error"] = "agent output not JSON: %s" % ((p.stdout or p.stderr or "")[:300])
        return out
    text = data.get("result") or ""
    out["cost_usd"] = data.get("total_cost_usd")
    out["turns"] = data.get("num_turns")
    if data.get("is_error"):
        out["error"] = "agent error: %s" % text[:300]
        return out
    m = re.findall(r"^\s*DESCRIPTION:\s*(.+?)\s*$", text, re.MULTILINE)
    if m:
        desc = m[-1]
    else:
        lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
        desc = lines[-1] if lines else ""
    out["description"] = " ".join(desc.replace("\t", " ").split())[:400]
    return out


def git(workdir: Path, *args: str) -> str:
    p = subprocess.run(["git", "-c", "core.autocrlf=false", "-c", "user.name=agent",
                        "-c", "user.email=agent@example.invalid"] + list(args),
                       cwd=str(workdir), capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    if p.returncode != 0:
        raise RuntimeError("git %s failed: %s" % (" ".join(args), p.stderr.strip()))
    return p.stdout


def syntax_error(path: Path) -> str | None:
    try:
        compile(path.read_text(encoding="utf-8"), "train.py", "exec")
        return None
    except SyntaxError as e:
        return "SyntaxError: %s (line %s)" % (e.msg, e.lineno)


# --------------------------------------------------------------------------
# arms
# --------------------------------------------------------------------------

class ArmContext:
    """What one arm shares between its agents."""

    def __init__(self, arm: str, agents: list[str], out: Path, seed: int):
        self.arm = arm
        self.agents = agents
        self.out = out
        self.lock = threading.Lock()
        self.tried: list[dict] = []          # hub
        self.clients: dict = {}              # scas
        self.server = None
        self.service = None
        if arm == "scas":
            from scas.loop import LoopClient
            from scas.service import CoordinatorService, make_server

            self.service = CoordinatorService(
                agents=agents, n_groups=3, min_improvement=KEEP_MARGIN,
                state_file=str(out / "coordinator_state.json"),
            )
            self.server = make_server(self.service, "127.0.0.1", 0)
            threading.Thread(target=self.server.serve_forever, daemon=True).start()
            url = "http://127.0.0.1:%d" % self.server.server_address[1]
            for i, a in enumerate(agents):
                self.clients[a] = LoopClient(a, server=url, state_dir=out / "scas_loop",
                                             rng=random.Random(seed * 1000 + i))

    def guidance(self, agent: str) -> tuple[str, dict | None]:
        """The arm-specific block of the prompt, and what to record about it."""
        if self.arm == "scas":
            packet = self.clients[agent].prompt()
            # the last line tells a human-run agent how to report; the driver reports
            packet = "\n".join(ln for ln in packet.splitlines()
                               if "scas.loop report" not in ln)
            mode = "refine" if "DIRECTIVE: REFINE" in packet else "explore"
            return ("You are one of several researchers working in parallel. "
                    "A coordinator sent you this directive; follow it:\n\n" + packet,
                    {"mode": mode})
        if self.arm == "hub":
            with self.lock:
                others = [t for t in self.tried if t["agent"] != agent]
            if not others:
                return ("You are one of several researchers working in parallel. "
                        "Nobody else has reported an experiment yet.", {"shared": 0})
            lines = ["You are one of several researchers working in parallel, each on "
                     "their own copy of train.py. Experiments the others have already "
                     "tried (do not repeat or trivially rephrase them):"]
            lines += ["  - [%s] %s" % (t["status"], t["description"]) for t in others]
            return "\n".join(lines), {"shared": len(others)}
        return "", None

    def report(self, agent: str, n: int, description: str, val_bpb: float,
               status: str, incumbent: float | None) -> dict | None:
        if self.arm == "hub":
            with self.lock:
                self.tried.append({"agent": agent, "description": description,
                                   "status": status})
            return None
        if self.arm == "scas":
            try:
                r = self.clients[agent].report(
                    description, val_bpb, status=status, step=n,
                    files_changed=["train.py"], incumbent=incumbent)
                return {k: r.get(k) for k in ("accepted", "reject_reason", "group_id",
                                              "alpha", "psi", "improved_group_best",
                                              "delta_vs_incumbent")}
            except Exception as e:  # noqa: BLE001 - never lose the experiment over it
                return {"error": "%s: %s" % (type(e).__name__, e)}
        return None

    def close(self) -> None:
        if self.service is not None:
            (self.out / "coordinator_snapshot.json").write_text(
                json.dumps(self.service.snapshot(), indent=2), encoding="utf-8")
        if self.server is not None:
            self.server.shutdown()


def run_agent(ctx: ArmContext, agent: str, box, workroot: Path, deadline: float,
              model: str) -> None:
    arm = ctx.arm
    tagp = "%s_%s" % (arm, agent)
    adir = ctx.out / agent
    adir.mkdir(parents=True, exist_ok=True)
    wd = workroot / arm / agent
    if wd.exists():
        shutil.rmtree(wd, ignore_errors=True)
    wd.mkdir(parents=True)
    tp = wd / "train.py"
    box.fetch_train_py(tp)
    git(wd, "init", "-q")
    git(wd, "add", "train.py")
    git(wd, "commit", "-q", "-m", "baseline")

    tsv = adir / "results.tsv"
    tsv.write_text("commit\tval_bpb\tmemory_gb\tstatus\tdescription\n", encoding="utf-8")
    trace = adir / "trace.jsonl"
    hist: list[dict] = []
    best = None
    t_arm = time.time()

    def record(row: dict) -> None:
        hist.append(row)
        commit = git(wd, "rev-parse", "--short=7", "HEAD").strip()
        with tsv.open("a", encoding="utf-8") as f:
            f.write("%s\t%.6f\t%.1f\t%s\t%s\n" % (
                commit, row["val_bpb"], (row.get("peak_vram_mb") or 0.0) / 1024.0,
                row["status"], row["description"]))
        with trace.open("a", encoding="utf-8") as f:
            f.write(json.dumps({**row, "arm": arm, "agent": agent, "box": box.name,
                                "commit": commit}) + "\n")

    def run_one(n: int, description: str, extra: dict) -> dict:
        nonlocal best
        t0 = time.time()
        res = box.train(tp, "%s_%03d" % (tagp, n))
        (adir / ("%03d.log" % n)).write_text(res.get("log_tail") or res.get("error") or "",
                                             encoding="utf-8")
        row = {"n": n, "description": description, "t_start": extra.pop("t_start", t0),
               "t_train_start": t0, "t_done": time.time(), "best_before": best,
               "num_steps": res.get("num_steps"), "peak_vram_mb": res.get("peak_vram_mb"),
               **extra}
        if "val_bpb" in res:
            val = res["val_bpb"]
            keep = best is None or val < best - KEEP_MARGIN
            row.update(val_bpb=val, status="keep" if keep else "discard")
        else:
            tail = [ln for ln in (res.get("log_tail") or "").strip().splitlines() if ln.strip()]
            row.update(val_bpb=0.0, status="crash", error=res.get("error"),
                       error_line=tail[-1] if tail else res.get("error"))
        return row

    # ---- baseline: the unmodified train.py on this box
    row = run_one(0, "baseline, no changes", {})
    if row["status"] != "keep":
        record(row)
        log("%s: baseline failed on %s (%s) - agent stops" % (tagp, box.name, row.get("error")))
        return
    best = row["val_bpb"]
    record(row)
    log("%s: baseline %.6f on %s" % (tagp, best, box.name))

    n = 0
    failures = 0
    while time.time() + CYCLE_RESERVE_S < deadline:
        t_start = time.time()
        block, ginfo = ctx.guidance(agent)
        prompt = "\n\n".join(x for x in (RULES, history_block(hist), block, TASK) if x)
        prop = propose(wd, prompt, model)
        changed = git(wd, "status", "--porcelain").strip()
        stray = [ln for ln in changed.splitlines() if not ln.endswith("train.py")]
        if stray:
            git(wd, "clean", "-fdq")
        err = prop.get("error")
        if not err and not git(wd, "diff", "--stat").strip():
            err = "agent made no change to train.py"
        if not err:
            err = syntax_error(tp)
        if err:
            git(wd, "checkout", "--", "train.py")
            failures += 1
            log("%s: proposal failed (%s)" % (tagp, err))
            with trace.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"arm": arm, "agent": agent, "proposal_failed": err,
                                    "t_start": t_start, "t_done": time.time(),
                                    "guidance": ginfo}) + "\n")
            if failures >= 6:
                log("%s: six failed proposals in a row - agent stops" % tagp)
                return
            time.sleep(15)
            continue
        failures = 0
        n += 1
        (adir / ("%03d.diff" % n)).write_text(git(wd, "diff"), encoding="utf-8")
        incumbent = best
        row = run_one(n, prop["description"], {
            "t_start": t_start, "propose_secs": prop.get("secs"),
            "cost_usd": prop.get("cost_usd"), "guidance": ginfo})
        if row["status"] == "keep":
            best = row["val_bpb"]
            git(wd, "commit", "-q", "-am", row["description"][:200])
        else:
            git(wd, "checkout", "--", "train.py")
        row["coordinator"] = ctx.report(agent, n, row["description"], row["val_bpb"],
                                        row["status"], incumbent)
        record(row)
        log("%s #%d %-7s %s (best %.6f, %.1f min into arm)  %s"
            % (tagp, n, row["status"],
               "%.6f" % row["val_bpb"] if row["status"] != "crash" else "crash",
               best, (time.time() - t_arm) / 60.0, row["description"][:70]))


def run_arm(arm: str, agents: list[str], boxes: list, out: Path, workroot: Path,
            hours: float, model: str, seed: int) -> None:
    adir = out / arm
    adir.mkdir(parents=True, exist_ok=True)
    ctx = ArmContext(arm, agents, adir, seed)
    t0 = time.time()
    deadline = t0 + hours * 3600.0
    (adir / "arm.json").write_text(json.dumps({
        "arm": arm, "agents": agents, "boxes": [b.name for b in boxes],
        "hours": hours, "t_start": t0, "deadline": deadline, "model": model,
        "keep_margin": KEEP_MARGIN,
        "rates": {b.name: b.st.get("dph_total") for b in boxes},
    }, indent=2), encoding="utf-8")
    log("=== arm %s: %d agent(s), %.2f h ===" % (arm, len(agents), hours))

    def guarded(agent, box):
        try:
            run_agent(ctx, agent, box, workroot, deadline, model)
        except Exception:  # noqa: BLE001
            log("%s_%s died:\n%s" % (arm, agent, traceback.format_exc()))

    threads = [threading.Thread(target=guarded, args=(a, b)) for a, b in zip(agents, boxes)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    ctx.close()
    log("=== arm %s finished after %.2f h ===" % (arm, (time.time() - t0) / 3600.0))


# --------------------------------------------------------------------------
# renting
# --------------------------------------------------------------------------

def load_swarm() -> dict:
    if SWARM_STATE.exists():
        return json.loads(SWARM_STATE.read_text(encoding="utf-8"))
    return {"boxes": []}


def save_swarm(state: dict) -> None:
    SWARM_STATE.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")


def instances() -> list[dict]:
    return vastctl.vast(["show", "instances"]) or []


def destroy(instance_id) -> bool:
    try:
        vastctl.vast(["destroy", "instance", str(instance_id), "-y"])
    except SystemExit:
        pass
    try:
        return not any(str(r.get("id")) == str(instance_id) for r in instances())
    except SystemExit:
        return False


def rent(offer: dict, label: str, args) -> dict | None:
    try:
        res = vastctl.vast([
            "create", "instance", str(offer["id"]), "--image", args.image,
            "--disk", str(args.disk), "--label", label, "--ssh", "--direct",
            "--onstart-cmd",
            "env | grep _ >> /etc/environment; mkdir -p /workspace; touch /workspace/.vast_ready",
        ])
    except SystemExit:
        return None
    iid = (res or {}).get("new_contract") or (res or {}).get("id")
    if not iid:
        return None
    return {"instance_id": int(iid), "offer_id": int(offer["id"]), "label": label,
            "dph_total": offer.get("dph_total"), "where": offer.get("geolocation"),
            "created_utc": vastctl.now_stamp()}


def wait_ssh(st: dict, minutes: int) -> bool:
    deadline = time.time() + minutes * 60
    while time.time() < deadline:
        try:
            me = next((r for r in instances() if str(r.get("id")) == str(st["instance_id"])), None)
        except SystemExit:
            me = None
        if me and (me.get("actual_status") or me.get("cur_state")) == "running":
            st["dph_total"] = me.get("dph_total", st.get("dph_total"))
            try:
                url = vastctl.vast(["ssh-url", str(st["instance_id"])], raw=False)
            except SystemExit:
                url = ""
            m = re.search(r"ssh://(?:[^@]+@)?([^:/]+):(\d+)", url or "")
            if m:
                st["ssh_host"], st["ssh_port"] = m.group(1), int(m.group(2))
                if Box(st).sh("echo ok", timeout=30)[0] == 0:
                    return True
        time.sleep(15)
    return False


def provision(st: dict, args, logdir: Path) -> bool:
    box = Box(st)
    box.sh("mkdir -p " + vastctl.REMOTE_TOOLS)
    if not box.put(vastctl.REMOTE_SRC / "provision.sh", vastctl.REMOTE_TOOLS + "/provision.sh"):
        return False
    cmd = ("set -o pipefail; bash %s/provision.sh %s %s %s 2>&1 | tee /workspace/provision.log"
           % (vastctl.REMOTE_TOOLS, args.repo, args.branch, args.dataset))
    rc, out = box.sh(cmd, timeout=40 * 60)
    (logdir / ("provision_%s.log" % st["instance_id"])).write_text(out, encoding="utf-8")
    return rc == 0


def cmd_up(args) -> None:
    state = load_swarm()
    if state["boxes"] and not args.force:
        vastctl.die("%d box(es) already tracked. Run `swarm.py down` first." % len(state["boxes"]))
    args.limit = max(args.limit, args.boxes * 4)
    offers = vastctl.search_offers(args)
    seen, pool = set(), []
    for o in offers:                       # one box per physical machine
        if o.get("machine_id") not in seen:
            seen.add(o.get("machine_id"))
            pool.append(o)
    if len(pool) < args.boxes:
        vastctl.die("only %d distinct offers matched; need %d" % (len(pool), args.boxes))
    logdir = HERE / ".swarm_logs"
    logdir.mkdir(exist_ok=True)
    stamp = vastctl.now_stamp()
    ready: list[dict] = []
    lock = threading.Lock()

    def bring_up(slot: int) -> None:
        while True:
            with lock:
                if not pool:
                    return
                offer = pool.pop(0)
            st = rent(offer, "scas-swarm-%s-%d" % (stamp, slot), args)
            if not st:
                log("slot %d: offer %s could not be rented, trying the next" % (slot, offer["id"]))
                continue
            with lock:
                state["boxes"].append(st)
                save_swarm(state)
            log("slot %d: instance %s ($%.3f/hr, %s) booting"
                % (slot, st["instance_id"], st.get("dph_total") or 0, st.get("where")))
            ok = wait_ssh(st, args.wait) and provision(st, args, logdir)
            with lock:
                save_swarm(state)
            if ok:
                log("slot %d: instance %s provisioned" % (slot, st["instance_id"]))
                with lock:
                    st["ready"] = True
                    ready.append(st)
                    save_swarm(state)
                return
            log("slot %d: instance %s failed to boot/provision - destroying it"
                % (slot, st["instance_id"]))
            gone = destroy(st["instance_id"])
            with lock:
                if gone:
                    state["boxes"].remove(st)
                save_swarm(state)

    threads = [threading.Thread(target=bring_up, args=(i,)) for i in range(args.boxes)]
    for t in threads:
        t.start()
        time.sleep(2)
    for t in threads:
        t.join()
    log("%d of %d boxes ready: %s" % (len(ready), args.boxes,
                                      ", ".join(str(s["instance_id"]) for s in ready)))
    if len(ready) < args.boxes:
        vastctl.die("not enough boxes came up; run `swarm.py down` or `swarm.py up --force`")


def cmd_down(args) -> None:
    state = load_swarm()
    left = []
    for st in state["boxes"]:
        if destroy(st["instance_id"]):
            log("destroyed instance %s (confirmed gone)" % st["instance_id"])
        else:
            log("instance %s is STILL LISTED - it may still be billing" % st["instance_id"])
            left.append(st)
    state["boxes"] = left
    if left:
        save_swarm(state)
        sys.exit(1)
    if SWARM_STATE.exists():
        SWARM_STATE.unlink()


def cmd_status(args) -> None:
    state = load_swarm()
    live = {str(r.get("id")): r for r in instances()}
    for st in state["boxes"]:
        r = live.get(str(st["instance_id"]))
        print("instance %s  %s  $%.3f/hr  ready=%s  %s" % (
            st["instance_id"], (r or {}).get("actual_status", "GONE"),
            st.get("dph_total") or 0.0, bool(st.get("ready")), st.get("where")))
    extra = [i for i in live if i not in {str(s["instance_id"]) for s in state["boxes"]}]
    if extra:
        print("untracked instances on the account: " + ", ".join(extra))
    if not state["boxes"] and not extra:
        print("no instances")


def keep_awake() -> None:
    """Stop Windows from idling to sleep while boxes are billing."""
    try:
        import ctypes

        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
    except Exception:  # noqa: BLE001 - not Windows, or not permitted
        pass


def cmd_run(args) -> None:
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    for a in arms:
        if a not in ARMS:
            vastctl.die("unknown arm %r (choose from %s)" % (a, ", ".join(ARMS)))
    swarm_arms = [a for a in arms if a != "single"]
    k = args.agents
    need = (k if swarm_arms else 0) + (1 if "single" in arms else 0)
    if args.fake:
        boxes = [FakeBox(i) for i in range(need)]
    else:
        ready = [s for s in load_swarm()["boxes"] if s.get("ready")]
        if len(ready) < need:
            vastctl.die("need %d provisioned boxes, have %d. Run `swarm.py up --boxes %d`."
                        % (need, len(ready), need))
        boxes = [Box(s) for s in ready[:need]]
    find_claude()
    keep_awake()

    stamp = vastctl.now_stamp()
    out = Path(args.out) if args.out else REPO_ROOT / "vast_runs" / ("swarm-" + stamp)
    out.mkdir(parents=True, exist_ok=True)
    # agent work dirs live outside the repository so a session cannot read the
    # project's documents and learn which arm it is in
    workroot = Path.home() / ".scas_swarm_work" / stamp
    workroot.mkdir(parents=True, exist_ok=True)
    agents = ["a%d" % i for i in range(k)]
    single_hours = args.single_hours if args.single_hours else args.hours * k
    (out / "run.json").write_text(json.dumps({
        "started_utc": stamp, "arms": arms, "agents_per_arm": k, "hours": args.hours,
        "single_hours": single_hours, "model": args.model, "fake": args.fake,
        "boxes": [b.st for b in boxes],
    }, indent=2), encoding="utf-8")
    log("results -> %s" % out)

    def swarm_sequence():
        for i, arm in enumerate(swarm_arms):
            run_arm(arm, agents, boxes[:k], out, workroot, args.hours, args.model, args.seed + i)

    threads = []
    if swarm_arms:
        threads.append(threading.Thread(target=swarm_sequence))
    if "single" in arms:
        threads.append(threading.Thread(target=run_arm, args=(
            "single", ["s0"], [boxes[-1]], out, workroot, single_hours, args.model,
            args.seed + 99)))
    try:
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    finally:
        (out / "run.json").write_text(json.dumps({
            **json.loads((out / "run.json").read_text(encoding="utf-8")),
            "finished_utc": vastctl.now_stamp()}, indent=2), encoding="utf-8")
        if not args.fake and not args.keep:
            log("destroying the boxes (everything is already saved locally)")
            cmd_down(args)
    log("done. Next: uv run python tools/vast/swarm_analyze.py \"%s\"" % out)


def main() -> None:
    p = argparse.ArgumentParser(prog="swarm", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("up", help="rent and provision N boxes")
    sp.add_argument("--boxes", type=int, default=4)
    sp.add_argument("--gpu", default="RTX_4090")
    sp.add_argument("--num-gpus", type=int, default=1)
    sp.add_argument("--max-price", type=float, default=0.60)
    sp.add_argument("--disk", type=int, default=60)
    sp.add_argument("--reliability", type=float, default=0.98)
    sp.add_argument("--inet-down", type=int, default=300)
    sp.add_argument("--cuda", type=float, default=12.4)
    sp.add_argument("--cpu-cores", type=float, default=8.0)
    sp.add_argument("--exclude-countries", default="CN")
    sp.add_argument("--limit", type=int, default=20)
    sp.add_argument("--image", default=vastctl.DEFAULT_IMAGE)
    sp.add_argument("--repo", default=vastctl.DEFAULT_REPO)
    sp.add_argument("--branch", default=vastctl.DEFAULT_BRANCH)
    sp.add_argument("--dataset", default="tinystories")
    sp.add_argument("--wait", type=int, default=15)
    sp.add_argument("--force", action="store_true")
    sp.set_defaults(func=cmd_up)

    sp = sub.add_parser("run", help="run the arms, then destroy the boxes")
    sp.add_argument("--arms", default="scas,independent,hub,single")
    sp.add_argument("--agents", type=int, default=3, help="K, agents per swarm arm")
    sp.add_argument("--hours", type=float, default=2.0, help="wall clock per swarm arm")
    sp.add_argument("--single-hours", type=float, default=None, dest="single_hours",
                    help="wall clock for the single arm (default K x --hours)")
    sp.add_argument("--model", default="sonnet", help="model alias for the agents")
    sp.add_argument("--seed", type=int, default=0)
    sp.add_argument("--out", default=None)
    sp.add_argument("--fake", action="store_true", help="stub trainer, no GPU")
    sp.add_argument("--keep", action="store_true", help="do not destroy the boxes")
    sp.set_defaults(func=cmd_run)

    sp = sub.add_parser("down", help="destroy every tracked box")
    sp.set_defaults(func=cmd_down)
    sp = sub.add_parser("status", help="tracked boxes and what the account reports")
    sp.set_defaults(func=cmd_status)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
