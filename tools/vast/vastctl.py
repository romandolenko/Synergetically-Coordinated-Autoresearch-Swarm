#!/usr/bin/env python3
"""vastctl - drive a rented vast.ai GPU box for autoresearch / SCAS experiments.

Stdlib only. Shells out to the `vastai` CLI (pip install vastai) plus ssh/scp.

Typical session:

    python tools/vast/vastctl.py keys                # one-time: keypair + register
    python tools/vast/vastctl.py search              # look at the market
    python tools/vast/vastctl.py up                  # rent the cheapest match
    python tools/vast/vastctl.py provision           # uv sync + prepare.py + smoke test
    python tools/vast/vastctl.py sweep --hours 6     # run the experiment plan, detached
    python tools/vast/vastctl.py logs -f             # watch it
    python tools/vast/vastctl.py finish              # wait, fetch + verify, DESTROY the box
    # ...or manually:
    python tools/vast/vastctl.py fetch               # pull results.tsv + logs down
    python tools/vast/vastctl.py down                # DESTROY the box (stop paying)

Every command that costs money prints the hourly rate first.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
STATE_PATH = HERE / ".state.json"
ENV_PATH = HERE / ".env.vast"
REMOTE_SRC = HERE / "remote"

SSH_KEY = Path.home() / ".ssh" / "id_ed25519_vast"
SSH_PUB = Path.home() / ".ssh" / "id_ed25519_vast.pub"
KNOWN_HOSTS = Path.home() / ".ssh" / "known_hosts_vast"

DEFAULT_IMAGE = "vastai/base-image:cuda-12.8.1-auto"
DEFAULT_REPO = "https://github.com/romandolenko/Synergetically-Coordinated-Autoresearch-Swarm.git"
DEFAULT_BRANCH = "scas-poc"

REMOTE_WORK = "/workspace"
REMOTE_REPO = "/workspace/autoresearch"
REMOTE_TOOLS = "/workspace/scas-remote"
SWEEP_LOG = "/workspace/sweep.log"


# --------------------------------------------------------------------------
# plumbing
# --------------------------------------------------------------------------

def die(msg, code=1):
    print("error: " + msg, file=sys.stderr)
    sys.exit(code)


def info(msg):
    print("[vastctl] " + msg, flush=True)


def load_api_key():
    key = os.environ.get("VAST_API_KEY", "").strip()
    if key:
        return key
    if ENV_PATH.exists():
        for line in ENV_PATH.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[7:]
            name, _, val = line.partition("=")
            if name.strip() == "VAST_API_KEY":
                return val.strip().strip('"').strip("'")
    legacy = Path.home() / ".config" / "vastai" / "vast_api_key"
    if legacy.exists():
        return legacy.read_text(encoding="utf-8").strip()
    die("no API key. Copy .env.vast.example to tools/vast/.env.vast and set VAST_API_KEY.")


def vastai_bin():
    for name in ("vastai", "vastai.exe", "vast", "vast.exe"):
        found = shutil.which(name)
        if found:
            return found
    die("`vastai` CLI not found on PATH. Install it: pip install vastai")


def vast(args, raw=True, check=True):
    """Run a vastai subcommand. Returns parsed JSON when raw=True, else stdout text."""
    cmd = [vastai_bin()] + list(args)
    if raw:
        cmd.append("--raw")
    cmd += ["--api-key", load_api_key()]
    # stdin is closed so a CLI confirmation prompt can never hang us invisibly
    proc = subprocess.run(cmd, capture_output=True, text=True, stdin=subprocess.DEVNULL)
    err = (proc.stderr or "").strip()
    # vastai >= 1.6 reports API errors on stderr and still exits 0
    api_error = any(ln.startswith('{"error": true') or ln.startswith("Failed with error")
                    for ln in err.splitlines())
    if check and (proc.returncode != 0 or api_error):
        die("vastai " + " ".join(args) + " failed:\n" + (proc.stdout or "") + "\n" + err)
    out = (proc.stdout or "").strip()
    if not raw:
        return out
    starts = [i for i in (out.find("["), out.find("{")) if i != -1]
    if not starts:
        return None
    try:
        return json.loads(out[min(starts):])
    except json.JSONDecodeError:
        die("could not parse vastai JSON output:\n" + out[:800])


def load_state():
    if STATE_PATH.exists():
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    return {}


def save_state(state):
    STATE_PATH.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")


def require_instance():
    state = load_state()
    if not state.get("instance_id"):
        die("no active instance in state. Run `vastctl up` first.")
    if not state.get("ssh_host"):
        die("instance has no ssh endpoint yet. Check `vastctl status`.")
    return state


def ssh_base(state, tty=False):
    if not SSH_KEY.exists():
        die("ssh key " + str(SSH_KEY) + " missing. Run `vastctl keys`.")
    cmd = [
        "ssh",
        "-i", str(SSH_KEY),
        "-p", str(state["ssh_port"]),
        "-o", "StrictHostKeyChecking=accept-new",
        "-o", "UserKnownHostsFile=" + str(KNOWN_HOSTS),
        "-o", "ServerAliveInterval=30",
        "-o", "ServerAliveCountMax=6",
        "-o", "LogLevel=ERROR",
    ]
    if tty:
        cmd.append("-t")
    cmd.append("root@" + str(state["ssh_host"]))
    return cmd


def ssh_run(state, remote_cmd, check=True, stream=False, tty=False):
    cmd = ssh_base(state, tty=tty) + [remote_cmd]
    if stream:
        proc = subprocess.run(cmd)
    else:
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.stdout:
            print(proc.stdout, end="")
        if proc.stderr and proc.stderr.strip():
            print(proc.stderr, end="", file=sys.stderr)
    if check and proc.returncode != 0:
        die("remote command failed (exit %d): %s" % (proc.returncode, remote_cmd))
    return proc


def scp(state, src, dst, to_remote, recursive=False):
    args = [
        "scp",
        "-i", str(SSH_KEY),
        "-P", str(state["ssh_port"]),
        "-o", "StrictHostKeyChecking=accept-new",
        "-o", "UserKnownHostsFile=" + str(KNOWN_HOSTS),
        "-o", "LogLevel=ERROR",
    ]
    if recursive:
        args.append("-r")
    host = "root@" + str(state["ssh_host"])
    if to_remote:
        args += [src, host + ":" + dst]
    else:
        args += [host + ":" + src, dst]
    proc = subprocess.run(args)
    if proc.returncode != 0:
        die("scp failed: " + " ".join(args))


def now_stamp():
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


# --------------------------------------------------------------------------
# commands
# --------------------------------------------------------------------------

def cmd_keys(args):
    if not SSH_KEY.exists():
        SSH_KEY.parent.mkdir(parents=True, exist_ok=True)
        info("generating " + str(SSH_KEY))
        subprocess.run(
            ["ssh-keygen", "-t", "ed25519", "-N", "", "-C", "vast-autoresearch",
             "-f", str(SSH_KEY)],
            check=True,
        )
    else:
        info("reusing " + str(SSH_KEY))
    pub = SSH_PUB.read_text(encoding="utf-8").strip()
    parts = pub.split()
    material = parts[1] if len(parts) > 1 else pub
    existing = vast(["show", "ssh-keys"]) or []
    if any(material in json.dumps(k) for k in existing):
        info("public key already registered with vast.ai")
    else:
        info("registering public key with vast.ai")
        vast(["create", "ssh-key", pub], raw=False)
    who = vast(["show", "user"]) or {}
    info("account=%s credit=$%.2f can_pay=%s"
         % (who.get("email"), who.get("credit") or 0.0, who.get("can_pay")))
    if not who.get("can_pay"):
        info("WARNING: account cannot pay yet - add credit or `up` will fail to schedule.")


def build_query(args):
    return " ".join([
        "gpu_name=" + args.gpu,
        "num_gpus=" + str(args.num_gpus),
        "rentable=true",
        "verified=true",
        "disk_space>" + str(args.disk + 10),
        "dph_total<" + str(args.max_price),
        "reliability>" + str(args.reliability),
        "inet_down>" + str(args.inet_down),
        "cuda_max_good>=" + str(args.cuda),
        # tokenisation runs on CPU every step; a starved box throttles the GPU
        "cpu_cores_effective>=" + str(args.cpu_cores),
    ] + (["geolocation notin [%s]" % args.exclude_countries]
         if args.exclude_countries else []))


def search_offers(args):
    # --storage makes dph_total include the disk we will actually allocate
    return vast(["search", "offers", build_query(args), "-o", "dph_total",
                 "--storage", str(args.disk), "--limit", str(args.limit)]) or []


def fmt_offer(o):
    return ("%10s  %-12s x%s  $%.3f/hr  +$%.3f/GB-mo  disk %6.0fGB  cpu %5.1f  "
            "net %6.0fMb  rel %.3f  %s"
            % (o.get("id"), o.get("gpu_name", ""), o.get("num_gpus", 1),
               o.get("dph_total", 0), o.get("storage_cost", 0), o.get("disk_space", 0),
               o.get("cpu_cores_effective", 0), o.get("inet_down", 0),
               o.get("reliability2", 0), o.get("geolocation", "")))


def cmd_search(args):
    offers = search_offers(args)
    if not offers:
        die("no offers matched. Loosen --max-price / --gpu / --cpu-cores.")
    print("%10s  %-12s      $/hr      storage       disk       cpu       net    rel  where"
          % ("offer_id", "gpu"))
    for o in offers:
        print(fmt_offer(o))
    cheap = offers[0]
    rate = cheap.get("dph_total", 0)
    print("\ncheapest: %s at $%.3f/hr -> ~$%.2f for an 8h overnight run, ~$%.2f/day"
          % (cheap.get("id"), rate, rate * 8, rate * 24))


def cmd_up(args):
    state = load_state()
    if state.get("instance_id") and not args.force:
        die("instance %s already tracked. Run `vastctl down` first, or pass --force."
            % state["instance_id"])
    if not SSH_KEY.exists():
        die("no ssh key. Run `vastctl keys` first.")

    rate = None
    if args.offer:
        offer_id = args.offer
    else:
        offers = search_offers(args)
        if not offers:
            die("no offers matched. Loosen --max-price / --gpu / --cpu-cores.")
        chosen = offers[0]
        offer_id = chosen["id"]
        rate = chosen.get("dph_total")
        info("chose offer " + fmt_offer(chosen).strip())

    label = args.label or ("scas-" + now_stamp())
    info("creating instance (image=%s, disk=%dGB, label=%s)" % (args.image, args.disk, label))
    res = vast([
        "create", "instance", str(offer_id),
        "--image", args.image,
        "--disk", str(args.disk),
        "--label", label,
        "--ssh", "--direct",
        "--onstart-cmd", "env | grep _ >> /etc/environment; mkdir -p /workspace; touch /workspace/.vast_ready",
    ])
    if not res or res.get("success") is False:
        die("instance creation failed: " + json.dumps(res))
    instance_id = res.get("new_contract") or res.get("id")
    if not instance_id:
        die("could not read instance id from response: " + json.dumps(res))
    info("instance %s created; waiting for boot (typically 1-5 min)" % instance_id)

    state = {
        "instance_id": int(instance_id),
        "offer_id": int(offer_id),
        "label": label,
        "created_utc": now_stamp(),
        "dph_total": rate,
        "image": args.image,
        "disk_gb": args.disk,
    }
    save_state(state)

    deadline = time.time() + args.wait * 60
    host = None
    port = None
    while time.time() < deadline:
        rows = vast(["show", "instances"]) or []
        me = None
        for r in rows:
            try:
                if int(r.get("id", -1)) == int(instance_id):
                    me = r
                    break
            except (TypeError, ValueError):
                continue
        if me:
            status = me.get("actual_status") or me.get("cur_state")
            state["dph_total"] = me.get("dph_total", state.get("dph_total"))
            state["gpu_name"] = me.get("gpu_name")
            if status == "running":
                url = vast(["ssh-url", str(instance_id)], raw=False)
                m = re.search(r"ssh://(?:[^@]+@)?([^:/]+):(\d+)", url or "")
                if m:
                    host, port = m.group(1), int(m.group(2))
                    break
            info("  status=%s %s" % (status, me.get("status_msg") or ""))
        time.sleep(15)

    if not host:
        die("instance %s did not become ssh-reachable within %d min. "
            "Check `vastctl status`; destroy it with `vastctl down` if stuck."
            % (instance_id, args.wait))

    state["ssh_host"] = host
    state["ssh_port"] = port
    save_state(state)
    info("up: instance %s at %s:%s  $%.3f/hr"
         % (instance_id, host, port, state.get("dph_total") or 0.0))

    info("waiting for sshd to accept connections")
    ok_deadline = time.time() + 300
    while time.time() < ok_deadline:
        if ssh_run(state, "echo ok", check=False).returncode == 0:
            info("ssh ready. Next: `vastctl provision`")
            return
        time.sleep(10)
    die("ssh never accepted a connection. Try `vastctl status` / `vastctl ssh`.")


def cmd_status(args):
    state = load_state()
    rows = vast(["show", "instances"]) or []
    if not state.get("instance_id"):
        print("no tracked instance")
        if rows:
            print("vast.ai reports these instances on the account:")
            for r in rows:
                print("  %s  %s  %s  $%.3f/hr  label=%s"
                      % (r.get("id"), r.get("actual_status"), r.get("gpu_name"),
                         r.get("dph_total") or 0.0, r.get("label")))
        return
    me = None
    for r in rows:
        try:
            if int(r.get("id", -1)) == int(state["instance_id"]):
                me = r
                break
        except (TypeError, ValueError):
            continue
    if not me:
        print("instance %s no longer exists on vast.ai (destroyed?)" % state["instance_id"])
        return
    dph = me.get("dph_total") or 0.0
    burn = ""
    started = me.get("start_date")
    if started:
        try:
            hrs = max(0.0, (time.time() - float(started)) / 3600.0)
            burn = "  uptime %.2fh  spent ~$%.2f" % (hrs, hrs * dph)
        except (TypeError, ValueError):
            pass
    print("instance %s  %s  %s  $%.3f/hr%s"
          % (me.get("id"), me.get("actual_status"), me.get("gpu_name"), dph, burn))
    print("  ssh      root@%s -p %s" % (state.get("ssh_host"), state.get("ssh_port")))
    print("  label    %s" % me.get("label"))
    print("  message  %s" % (me.get("status_msg") or "-"))
    probe = ssh_run(
        state,
        "test -f %s/results.tsv && wc -l < %s/results.tsv || echo 0" % (REMOTE_REPO, REMOTE_REPO),
        check=False,
    )
    if probe.returncode == 0 and probe.stdout:
        print("  results  %s line(s) in results.tsv" % probe.stdout.strip().splitlines()[-1])


def cmd_ssh(args):
    state = require_instance()
    if args.command:
        ssh_run(state, " ".join(args.command), stream=True, check=False)
    else:
        subprocess.run(ssh_base(state, tty=True))


def cmd_provision(args):
    state = require_instance()
    info("uploading remote helpers")
    ssh_run(state, "mkdir -p " + REMOTE_TOOLS)
    for f in sorted(REMOTE_SRC.iterdir()):
        if f.is_file():
            scp(state, str(f), REMOTE_TOOLS + "/" + f.name, to_remote=True)
    info("running provision.sh (clone, uv sync, prepare.py, smoke test) - several minutes")
    smoke = "--skip-smoke" if args.skip_smoke else ""
    ssh_run(
        state,
        # pipefail: without it the pipeline reports tee's exit status and a failed
        # provision is announced as a success
        "set -o pipefail; bash %s/provision.sh %s %s %s %s 2>&1 | tee %s/provision.log"
        % (REMOTE_TOOLS, args.repo, args.branch, args.dataset, smoke, REMOTE_WORK),
        stream=True,
    )
    state["provisioned_utc"] = now_stamp()
    state["repo"] = args.repo
    state["branch"] = args.branch
    state["dataset"] = args.dataset
    save_state(state)
    info("provisioned. Next: `vastctl sweep --hours N`")


def cmd_sweep(args):
    state = require_instance()
    plan_path = Path(args.plan)
    plan_remote = REMOTE_TOOLS + "/" + plan_path.name
    if plan_path.is_file():
        scp(state, str(plan_path.resolve()), plan_remote, to_remote=True)
    else:
        info("plan %s not found locally; assuming it already exists at %s" % (args.plan, plan_remote))

    running = ssh_run(state, "pgrep -f '[s]weep.py' >/dev/null && echo RUNNING || echo IDLE", check=False)
    if "RUNNING" in (running.stdout or "") and not args.force:
        die("a sweep is already running on the box. Use `vastctl logs -f`, or pass --force.")

    dph = state.get("dph_total") or 0.0
    info("starting sweep: max %.1fh wall clock (~$%.2f at $%.3f/hr)"
         % (args.hours, dph * args.hours, dph))
    flags = []
    if args.resume:
        flags.append("--resume")
    if args.dry_run:
        flags.append("--dry-run")
    flags += ["--baseline-repeats", str(args.baseline_repeats),
              "--min-improvement", str(args.min_improvement)]
    remote = (
        "cd %s && export PATH=$HOME/.local/bin:$PATH && "
        # the subshell keeps the backgrounded job off the ssh session's stdout,
        # otherwise ssh never returns
        "(setsid nohup uv run python %s/sweep.py --plan %s --repo %s --hours %s %s "
        "> %s 2>&1 < /dev/null &) ; echo started"
        % (REMOTE_REPO, REMOTE_TOOLS, plan_remote, REMOTE_REPO, args.hours,
           " ".join(flags), SWEEP_LOG)
    )
    ssh_run(state, remote)
    state["sweep_started_utc"] = now_stamp()
    state["sweep_hours"] = args.hours
    save_state(state)
    info("sweep detached. Watch with: python tools/vast/vastctl.py logs -f")
    if args.auto_down and not args.dry_run:
        info("--auto-down: staying attached; will fetch results and destroy the box "
             "when the sweep completes")
        time.sleep(20)  # let the sweep process appear before the first poll
        cmd_finish(argparse.Namespace(poll=args.poll, max_misses=10, out=None, keep=False))


def cmd_logs(args):
    state = require_instance()
    target = SWEEP_LOG if args.which == "sweep" else (REMOTE_WORK + "/provision.log")
    flag = "-f" if args.follow else ""
    ssh_run(state, "touch %s; tail -n %d %s %s" % (target, args.lines, flag, target),
            stream=True, check=False)


def cmd_stop_sweep(args):
    state = require_instance()
    ssh_run(state, "pkill -f '[s]weep.py' && echo killed || echo 'no sweep running'", check=False)


def cmd_fetch(args):
    state = require_instance()
    label = state.get("label", "run")
    out = Path(args.out) if args.out else (REPO_ROOT / "vast_runs" / (label + "-" + now_stamp()))
    out.mkdir(parents=True, exist_ok=True)

    wanted = [
        (REMOTE_REPO + "/results.tsv", "results.tsv"),
        (REMOTE_REPO + "/sweep_state.json", "sweep_state.json"),
        (SWEEP_LOG, "sweep.log"),
        (REMOTE_WORK + "/provision.log", "provision.log"),
    ]
    for remote, local in wanted:
        probe = ssh_run(state, "test -f %s && echo yes || echo no" % remote, check=False)
        if "yes" in (probe.stdout or ""):
            scp(state, remote, str(out / local), to_remote=False)
            info("  got " + local)
        else:
            info("  skip %s (absent)" % local)

    probe = ssh_run(state, "test -d %s/runs && echo yes || echo no" % REMOTE_REPO, check=False)
    if "yes" in (probe.stdout or ""):
        scp(state, REMOTE_REPO + "/runs", str(out), to_remote=False, recursive=True)
        info("  got runs/ (per-experiment train logs)")

    meta = dict(state)
    meta["fetched_utc"] = now_stamp()
    (out / "instance.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    info("artifacts -> " + str(out))
    print("\nNext: python tools/vast/analyze.py \"%s\"" % (out / "results.tsv"))
    return out


def _remote_out(state, remote_cmd):
    """Run a remote command quietly. Returns stdout, or None if ssh itself failed."""
    proc = subprocess.run(ssh_base(state) + [remote_cmd], capture_output=True, text=True)
    if proc.returncode != 0:
        return None
    return proc.stdout or ""


def cmd_finish(args):
    """Wait for the sweep to end, save the results locally, then destroy the box.

    The instance is destroyed ONLY when the sweep ended normally and the local
    copy is verified against the box. Anything else leaves it running (and
    billing) so nothing is lost, and exits non-zero.
    """
    state = require_instance()
    info("waiting for the sweep to finish (poll every %ds)" % args.poll)
    misses = 0
    while True:
        out = _remote_out(
            state,
            "pgrep -f '[s]weep.py' >/dev/null && echo SWEEP_ALIVE || echo SWEEP_GONE; "
            "tail -n 1 %s 2>/dev/null" % SWEEP_LOG,
        )
        if out is None or ("SWEEP_ALIVE" not in out and "SWEEP_GONE" not in out):
            # an unreachable box is not a finished sweep
            misses += 1
            if misses >= args.max_misses:
                die("lost contact with the instance (%d failed polls). It is STILL "
                    "RUNNING and billing; re-run `vastctl finish` or check `vastctl status`."
                    % misses)
            info("  ssh poll failed (%d/%d), retrying" % (misses, args.max_misses))
        else:
            misses = 0
            last = out.strip().splitlines()[-1][:110] if out.strip() else ""
            if "SWEEP_GONE" in out:
                break
            info("  running: " + last)
        time.sleep(args.poll)

    log_tail = _remote_out(state, "tail -n 5 %s 2>/dev/null" % SWEEP_LOG) or ""
    clean = "=== done:" in log_tail
    if not clean:
        info("sweep process is gone but the log has no '=== done' line:")
        print(log_tail)

    out_dir = cmd_fetch(argparse.Namespace(out=args.out))

    # verify the local copy before anything irreversible
    problems = []
    tsv = out_dir / "results.tsv"
    remote_lines = _remote_out(state, "wc -l < %s/results.tsv" % REMOTE_REPO)
    if not tsv.is_file():
        problems.append("results.tsv was not downloaded")
    elif remote_lines is None or not remote_lines.strip().isdigit():
        problems.append("could not read the remote results.tsv to compare")
    else:
        local_lines = tsv.read_bytes().count(b"\n")
        if local_lines != int(remote_lines.strip()) or local_lines < 2:
            problems.append("results.tsv has %d lines locally vs %s on the box"
                            % (local_lines, remote_lines.strip()))
    try:
        # experiments skipped for a bad edit are in the state but never get a tsv row
        n_state = sum(1 for e in json.loads((out_dir / "sweep_state.json").read_text(
            encoding="utf-8"))["experiments"] if e.get("status") != "skipped")
        if tsv.is_file() and n_state != tsv.read_bytes().count(b"\n") - 1:
            problems.append("sweep_state.json has %d experiments but results.tsv has %d rows"
                            % (n_state, tsv.read_bytes().count(b"\n") - 1))
    except (OSError, ValueError, KeyError) as err:
        problems.append("sweep_state.json missing or unreadable (%s)" % err)
    if not (out_dir / "runs").is_dir():
        problems.append("runs/ (per-experiment logs) was not downloaded")
    if not clean:
        problems.append("the sweep did not end normally (crashed or was killed)")

    if problems:
        for p in problems:
            info("NOT destroying: " + p)
        die("instance %s left RUNNING and billing. Results so far are in %s. "
            "Fix or `vastctl sweep --resume`, or destroy it with `vastctl down`."
            % (state["instance_id"], out_dir))

    info("results verified in %s" % out_dir)
    if args.keep:
        info("--keep given: instance left running. Destroy it with `vastctl down`.")
        return
    cmd_down(argparse.Namespace(yes=True, instance=None))


def cmd_down(args):
    state = load_state()
    iid = state.get("instance_id") or args.instance
    if not iid:
        die("nothing to destroy (no tracked instance, no --instance).")
    if not args.yes:
        print("About to DESTROY instance %s (irreversible; all data on the box is lost)." % iid)
        print("Did you run `vastctl fetch` first?")
        if input("type 'destroy' to confirm: ").strip() != "destroy":
            die("aborted")
    # vastai >= 1.6 asks for its own confirmation unless -y is passed; without it
    # the CLI prints "Aborted." and exits 0, leaving the box running and billing.
    # The CLI prints no response for destroy under --raw, so the only trustworthy
    # confirmation is that the instance has left the account listing.
    vast(["destroy", "instance", str(iid), "-y"])
    still = [r for r in (vast(["show", "instances"]) or [])
             if str(r.get("id")) == str(iid)]
    if still:
        die("instance %s is STILL LISTED after destroy (status=%s) and may still be "
            "billing. Re-run `vastctl down`, or check https://cloud.vast.ai/instances/"
            % (iid, still[0].get("actual_status")))
    info("destroyed instance %s (confirmed gone from the account)" % iid)
    if STATE_PATH.exists():
        STATE_PATH.unlink()


# --------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(prog="vastctl", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    def add_market_flags(sp):
        sp.add_argument("--gpu", default="RTX_4090",
                        help="vast gpu_name, spaces as underscores (RTX_4090, RTX_3090, H100_SXM)")
        sp.add_argument("--num-gpus", type=int, default=1)
        sp.add_argument("--max-price", type=float, default=0.60, help="max $/hr")
        sp.add_argument("--disk", type=int, default=60, help="instance disk in GB")
        sp.add_argument("--reliability", type=float, default=0.98)
        sp.add_argument("--inet-down", type=int, default=300, help="min Mb/s down")
        sp.add_argument("--cuda", type=float, default=12.4, help="min cuda_max_good")
        sp.add_argument("--cpu-cores", type=float, default=8.0,
                        help="min effective CPU cores (tokenisation is CPU-bound)")
        sp.add_argument("--exclude-countries", default="CN",
                        help="comma-separated country codes to skip ('' for none). "
                             "Default CN: a box there failed provisioning with "
                             "corrupted PyPI downloads (hash mismatch)")
        sp.add_argument("--limit", type=int, default=10)

    sp = sub.add_parser("keys", help="create/register the ssh keypair")
    sp.set_defaults(func=cmd_keys)

    sp = sub.add_parser("search", help="list matching offers")
    add_market_flags(sp)
    sp.set_defaults(func=cmd_search)

    sp = sub.add_parser("up", help="rent a box and wait until ssh is live")
    add_market_flags(sp)
    sp.add_argument("--offer", type=int, help="rent this specific offer id")
    sp.add_argument("--image", default=DEFAULT_IMAGE)
    sp.add_argument("--label", default=None)
    sp.add_argument("--wait", type=int, default=15, help="minutes to wait for boot")
    sp.add_argument("--force", action="store_true")
    sp.set_defaults(func=cmd_up)

    sp = sub.add_parser("status", help="instance state, uptime and spend")
    sp.set_defaults(func=cmd_status)

    sp = sub.add_parser("ssh", help="interactive shell, or run a command")
    sp.add_argument("command", nargs="*")
    sp.set_defaults(func=cmd_ssh)

    sp = sub.add_parser("provision", help="clone repo, uv sync, prepare data, smoke test")
    sp.add_argument("--repo", default=DEFAULT_REPO)
    sp.add_argument("--branch", default=DEFAULT_BRANCH)
    sp.add_argument("--dataset", default="tinystories")
    sp.add_argument("--skip-smoke", action="store_true")
    sp.set_defaults(func=cmd_provision)

    sp = sub.add_parser("sweep", help="run the experiment plan, detached")
    sp.add_argument("--plan", default=str(REMOTE_SRC / "hypotheses.json"))
    sp.add_argument("--hours", type=float, default=6.0, help="hard wall-clock budget")
    sp.add_argument("--resume", action="store_true", help="continue from sweep_state.json")
    sp.add_argument("--dry-run", action="store_true", help="validate the plan, run nothing")
    sp.add_argument("--baseline-repeats", type=int, default=3,
                    help="unmodified baseline runs, to measure run-to-run noise")
    sp.add_argument("--min-improvement", default="auto",
                    help="keep threshold: a number, or 'auto' = 2x baseline std")
    sp.add_argument("--force", action="store_true")
    sp.add_argument("--auto-down", action="store_true",
                    help="stay attached, then fetch results and DESTROY the box when "
                         "the sweep completes (same as running `finish` afterwards)")
    sp.add_argument("--poll", type=int, default=120, help="--auto-down poll seconds")
    sp.set_defaults(func=cmd_sweep)

    sp = sub.add_parser("finish", help="wait for the sweep, fetch + verify results, "
                                       "then DESTROY the instance")
    sp.add_argument("--poll", type=int, default=120, help="seconds between checks")
    sp.add_argument("--max-misses", type=int, default=10,
                    help="consecutive failed ssh polls before giving up (box stays up)")
    sp.add_argument("--out", default=None, help="local directory for the results")
    sp.add_argument("--keep", action="store_true",
                    help="fetch and verify but do not destroy")
    sp.set_defaults(func=cmd_finish)

    sp = sub.add_parser("logs", help="tail the sweep (or provision) log")
    sp.add_argument("-f", "--follow", action="store_true")
    sp.add_argument("-n", "--lines", type=int, default=80)
    sp.add_argument("--which", choices=["sweep", "provision"], default="sweep")
    sp.set_defaults(func=cmd_logs)

    sp = sub.add_parser("stop-sweep", help="kill the running sweep, keep the box")
    sp.set_defaults(func=cmd_stop_sweep)

    sp = sub.add_parser("fetch", help="download results.tsv, logs and state")
    sp.add_argument("--out", default=None)
    sp.set_defaults(func=cmd_fetch)

    sp = sub.add_parser("down", help="DESTROY the instance")
    sp.add_argument("--yes", action="store_true")
    sp.add_argument("--instance", type=int, default=None)
    sp.set_defaults(func=cmd_down)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
