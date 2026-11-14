"""Rebuild the v3 experiment memory from the swarm runs under vast_runs/.

Every logged experiment is a diff against the train.py it was built on. This
replays the diffs to recover the full train.py of each experiment, extracts
its settings (scas/surrogate.py) and writes one JSON line per scored run to
scas/data/v3_memory.jsonl - the memory the v3 coordinator starts from.

    uv run python tools/vast/memory_build.py            # all runs under vast_runs/
    uv run python tools/vast/memory_build.py --out x.jsonl vast_runs/swarm-...
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from scas.surrogate import MEMORY_FILE, complete, extract  # noqa: E402

SHARED_ARMS = ("scas2", "scas3")        # arms with one shared train.py and version hashes


def code_hash(code: str) -> str:
    return hashlib.sha256(code.replace("\r\n", "\n").encode("utf-8")).hexdigest()[:16]


class Replayer:
    def __init__(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="scas_mem_"))
        subprocess.run(["git", "init", "-q"], cwd=self.tmp, check=True)

    def apply(self, code: str, diff_path: Path) -> str | None:
        (self.tmp / "train.py").write_text(code, encoding="utf-8", newline="\n")
        (self.tmp / "x.diff").write_bytes(diff_path.read_bytes().replace(b"\r\n", b"\n"))
        p = subprocess.run(["git", "-c", "core.autocrlf=false", "apply", "--whitespace=nowarn",
                            "x.diff"], cwd=self.tmp, capture_output=True, text=True)
        if p.returncode != 0:
            return None
        return (self.tmp / "train.py").read_text(encoding="utf-8")

    def merge(self, current: str, base: str, other: str) -> str | None:
        for name, text in (("cur.py", current), ("b.py", base), ("o.py", other)):
            (self.tmp / name).write_text(text, encoding="utf-8", newline="\n")
        p = subprocess.run(["git", "merge-file", "cur.py", "b.py", "o.py"],
                           cwd=self.tmp, capture_output=True)
        if p.returncode != 0:
            return None
        return (self.tmp / "cur.py").read_text(encoding="utf-8")


def rows_of(adir: Path) -> list[dict]:
    out = []
    for line in (adir / "trace.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        if "n" in r:
            out.append(r)
    return out


def replay_run(rd: Path, base: str, rp: Replayer, with_code: bool = False) -> tuple[list[dict], int]:
    data, fails = [], 0
    rest0 = extract(base)[1]
    for armdir in sorted(p for p in rd.iterdir() if p.is_dir()):
        arm = armdir.name
        agents = sorted(p for p in armdir.iterdir()
                        if p.is_dir() and (p / "trace.jsonl").exists())
        recs = []
        if arm not in SHARED_ARMS:
            for ad in agents:
                cur = base
                for r in rows_of(ad):
                    code = base
                    if r["n"] > 0:
                        dp = ad / ("%03d.diff" % r["n"])
                        code = rp.apply(cur, dp) if dp.exists() else None
                        if code is None:
                            fails += 1
                            continue
                        if r["status"] == "keep":
                            cur = code
                    recs.append((r, ad.name, code))
        else:
            codes = {code_hash(base): base}
            adopted = set()
            for ev in armdir.glob("v*_events.jsonl"):
                for line in ev.read_text(encoding="utf-8").splitlines():
                    e = json.loads(line)
                    if e["event"] == "adopted":
                        adopted.add(e["version"])
            allrows = sorted(((r, ad) for ad in agents for r in rows_of(ad)),
                             key=lambda x: x[0]["t_done"])
            for r, ad in allrows:
                co = r.get("coordinator") or {}
                if r["n"] == 0:
                    code = base
                elif r.get("kind") == "verify":
                    code = codes.get(r["version"])
                else:
                    b = codes.get(r["version"])
                    dp = ad / ("%03d.diff" % r["n"])
                    code = rp.apply(b, dp) if (b is not None and dp.exists()) else None
                    if code is not None:
                        if co.get("promoted"):
                            codes[code_hash(code)] = code
                        if co.get("adopted"):
                            for k in list(codes.values()):
                                c2 = rp.merge(k, b, code)
                                if c2 is not None and code_hash(c2) in adopted:
                                    codes[code_hash(c2)] = c2
                if code is None:
                    fails += 1
                    continue
                recs.append((r, ad.name, code))
        for r, agent, code in recs:
            if not r.get("val_bpb") or not r.get("num_steps"):
                continue
            consts, rest = extract(code)
            if not complete(consts):
                fails += 1
                continue
            co = r.get("coordinator") or {}
            data.append({
                "run": rd.name, "arm": arm, "agent": agent, "box": r["box"], "n": r["n"],
                "t_start": r["t_start"], "t": r["t_done"], "val": r["val_bpb"],
                "steps": r["num_steps"], "status": r["status"], "kind": r.get("kind"),
                "ref": r.get("best_before") or co.get("ref"),
                "v2_expected": co.get("expected") if r.get("kind") == "verify" else None,
                "consts": consts, "flags": [] if rest == rest0 else ["rest:" + rest],
                "code_hash": code_hash(code), "desc": r["description"][:120],
            })
            if with_code:
                data[-1]["code"] = code
    return data, fails


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("runs", nargs="*", help="run directories (default: vast_runs/swarm-*)")
    ap.add_argument("--out", default=str(MEMORY_FILE))
    ap.add_argument("--base", default=None, help="the pristine train.py the runs started from")
    args = ap.parse_args()
    runs = [Path(r) for r in args.runs] or sorted((REPO_ROOT / "vast_runs").glob("swarm-*"))
    base_path = Path(args.base) if args.base else next(
        (p for rd in runs for p in rd.glob("*/base_train.py")), None)
    if base_path is None:
        raise SystemExit("no base_train.py found in the runs; pass --base")
    base = base_path.read_text(encoding="utf-8").replace("\r\n", "\n")
    rp, data = Replayer(), []
    for rd in runs:
        d, fails = replay_run(rd, base, rp)
        print("%s: %d scored runs, %d not reconstructed" % (rd.name, len(d), fails))
        data += d
    data.sort(key=lambda r: r["t"])
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(r) + "\n" for r in data), encoding="utf-8", newline="\n")
    print("wrote %d rows to %s" % (len(data), out))
    print("by arm:", dict(Counter(r["arm"] for r in data)))
    print("with structural flags:", sum(bool(r["flags"]) for r in data))


if __name__ == "__main__":
    main()
