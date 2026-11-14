"""Does the training trajectory say anything about the result beyond step count?

Re-runs train.py versions from an earlier swarm run with per-step telemetry
(weight, update and gradient norms per parameter group) and tests whether any
trajectory statistic predicts val_bpb once box and optimizer steps are
accounted for. A statistic that does is a candidate target manifold for a
closed-loop controller inside training; if none does, that idea is closed for
this benchmark.

    uv run python tools/vast/telemetry.py plan vast_runs/swarm-<stamp> --n 45
    uv run python tools/vast/swarm.py up --boxes 3 --gpu RTX_4080S
    uv run python tools/vast/telemetry.py run vast_runs/telemetry-<stamp>
    uv run python tools/vast/swarm.py down
    uv run python tools/vast/telemetry.py analyze vast_runs/telemetry-<stamp>
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import threading
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(HERE))

import memory_build  # noqa: E402

TELEMETRY_FILE = "telemetry.json"

# Appended to train.py. Everything is looked up at call time, so the position
# of the block in the file does not matter.
HELPERS = '''

# ---- telemetry (tools/vast/telemetry.py) ----------------------------------
_TM = {"rows": [], "every": 4}


def _tm_label(optimizer):
    names = ("lm_head", "wte", "value_embeds", "resid", "x0")
    labels, i = [], 0
    for g in optimizer.param_groups:
        if g.get("kind") == "muon":
            labels.append("matrix")
        else:
            labels.append(names[i] if i < len(names) else "other")
            i += 1
    return labels


def _tm_before(optimizer, step):
    if step % _TM["every"]:
        return None
    return [[p.detach().clone() for p in g["params"]] for g in optimizer.param_groups]


def _tm_after(optimizer, step, pre, loss, lrm, progress):
    if pre is None:
        return
    import torch as _t
    acc = {}
    with _t.no_grad():
        for label, g, old in zip(_tm_label(optimizer), optimizer.param_groups, pre):
            a = acc.setdefault(label, [0.0, 0.0, 0.0, 0.0])
            for p, q in zip(g["params"], old):
                pf = p.detach().float()
                d = pf - q.float()
                a[0] += float(pf.pow(2).sum())
                a[1] += float(d.pow(2).sum())
                if p.grad is not None:
                    gr = p.grad.detach().float()
                    a[2] += float(gr.pow(2).sum())
                    a[3] += float((d * gr).sum())
    row = {"step": int(step), "progress": float(progress), "lrm": float(lrm),
           "loss": float(loss)}
    for label, (w2, u2, g2, ug) in acc.items():
        row[label] = [w2 ** 0.5, u2 ** 0.5, g2 ** 0.5, ug]
    _TM["rows"].append(row)


def _tm_dump():
    import json as _j
    try:
        with open("TELEMETRY_FILE", "w") as f:
            _j.dump(_TM["rows"], f)
    except Exception:
        pass


import atexit as _atexit
_atexit.register(_tm_dump)
'''.replace("TELEMETRY_FILE", TELEMETRY_FILE)

STEP_LINE = "        optimizer.step()\n"


def instrument(code: str) -> str | None:
    """train.py with telemetry around the training loop's optimizer step."""
    start = code.find("def get_lr_multiplier")
    i = code.find(STEP_LINE, start if start >= 0 else 0)
    if start < 0 or i < 0 or code.find(STEP_LINE, i + 1) >= 0:
        return None
    new = ("        _tm_pre = _tm_before(optimizer, step)\n"
           "        optimizer.step()\n"
           "        _tm_after(optimizer, step, _tm_pre, train_loss, lrm, progress)\n")
    out = code[:i] + new + code[i + len(STEP_LINE):]
    j = out.find('\nif __name__ == "__main__":')
    if j < 0:
        return None
    out = out[:j] + HELPERS + out[j:]
    try:
        compile(out, "train.py", "exec")
    except SyntaxError:
        return None
    return out


# --------------------------------------------------------------------------
# plan
# --------------------------------------------------------------------------

def cmd_plan(args) -> None:
    rd = Path(args.run)
    base_path = next(rd.glob("*/base_train.py"))
    base = base_path.read_text(encoding="utf-8").replace("\r\n", "\n")
    rows, fails = memory_build.replay_run(rd, base, memory_build.Replayer(), with_code=True)
    print("%s: %d scored runs, %d not reconstructed" % (rd.name, len(rows), fails))
    batch = 2 ** args.log2_batch
    seen, pool = set(), []
    for r in rows:
        if r["consts"]["TOTAL_BATCH_SIZE"] != batch or r["code_hash"] in seen:
            continue
        code = instrument(r["code"])
        if code is None:
            continue
        seen.add(r["code_hash"])
        pool.append(dict(r, code=code))
    print("%d distinct versions at batch 2**%d can be instrumented" % (len(pool), args.log2_batch))
    # the reference: the pristine file with only the batch size changed
    from scas.surrogate import set_constant
    ref = instrument(set_constant(base, "TOTAL_BATCH_SIZE", batch))
    if ref is None:
        raise SystemExit("could not instrument the reference version")
    rng = random.Random(args.seed)
    # spread over the range of results: sort by the original value, take every k-th
    pool.sort(key=lambda r: r["val"] / (r["ref"] or r["val"]))
    if len(pool) > args.n:
        idx = sorted({round(i * (len(pool) - 1) / (args.n - 1)) for i in range(args.n)})
        pool = [pool[i] for i in idx]
    rng.shuffle(pool)
    out = Path(args.out) if args.out else REPO_ROOT / "vast_runs" / (
        "telemetry-" + time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()))
    (out / "code").mkdir(parents=True, exist_ok=True)
    (out / "code" / "ref.py").write_text(ref, encoding="utf-8", newline="\n")
    plan = []
    for k, r in enumerate(pool):
        vid = "v%03d" % k
        (out / "code" / (vid + ".py")).write_text(r["code"], encoding="utf-8", newline="\n")
        plan.append({"id": vid, "slot": k % args.boxes,
                     **{f: r[f] for f in ("arm", "agent", "box", "n", "val", "steps", "status",
                                          "kind", "consts", "flags", "code_hash", "desc")}})
    (out / "plan.json").write_text(json.dumps({
        "source": rd.name, "boxes": args.boxes, "ref_repeats": args.ref_repeats,
        "versions": plan}, indent=1), encoding="utf-8")
    print("planned %d versions + %d reference runs per box -> %s"
          % (len(plan), args.ref_repeats, out))
    print("kept: %d, structural: %d" % (sum(p["status"] == "keep" for p in plan),
                                        sum(bool(p["flags"]) for p in plan)))


# --------------------------------------------------------------------------
# run
# --------------------------------------------------------------------------

def cmd_run(args) -> None:
    import swarm
    out = Path(args.dir)
    plan = json.loads((out / "plan.json").read_text(encoding="utf-8"))
    ready = [s for s in swarm.load_swarm()["boxes"] if s.get("ready")]
    if len(ready) < plan["boxes"]:
        raise SystemExit("need %d provisioned boxes, have %d" % (plan["boxes"], len(ready)))
    boxes = [swarm.Box(s) for s in ready[:plan["boxes"]]]
    swarm.keep_awake()
    res_path = out / "results.jsonl"
    done = set()
    if res_path.exists():
        for line in res_path.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            if r.get("val_bpb"):
                done.add(r["tag"])
    lock = threading.Lock()
    (out / "telemetry").mkdir(exist_ok=True)

    def one(box, vid: str, tag: str) -> None:
        if tag in done:
            return
        box.sh("rm -f %s/%s" % (swarm.REMOTE_REPO, TELEMETRY_FILE))
        t0 = time.time()
        res = box.train(out / "code" / (vid + ".py"), tag)
        tele = None
        if "val_bpb" in res:
            for _ in range(3):
                rc, txt = box.sh("cat %s/%s" % (swarm.REMOTE_REPO, TELEMETRY_FILE), timeout=120)
                try:
                    tele = json.loads(txt) if rc == 0 else None
                except ValueError:
                    tele = None
                if tele:
                    break
                time.sleep(5)
        if tele:
            (out / "telemetry" / (tag + ".json")).write_text(json.dumps(tele), encoding="utf-8")
        row = {"tag": tag, "id": vid, "box": box.name, "seconds": round(time.time() - t0),
               "telemetry_rows": len(tele or []),
               **{k: v for k, v in res.items() if k != "log_tail"}}
        if "error" in res:
            row["log_tail"] = res.get("log_tail", "")[-1500:]
        with lock:
            with res_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
        swarm.log("%s %s on %s: val %s, steps %s, %d telemetry rows%s"
                  % (tag, vid, box.name, res.get("val_bpb"), res.get("num_steps"),
                     len(tele or []), (" [%s]" % res["error"]) if "error" in res else ""))

    def work(slot: int) -> None:
        box = boxes[slot]
        mine = [v["id"] for v in plan["versions"] if v["slot"] == slot]
        reps = plan["ref_repeats"]
        # reference runs are spread through the queue: they give the box's level and its drift
        marks = {round(i * len(mine) / max(reps - 1, 1)) for i in range(reps)}
        k = 0
        for i in range(len(mine) + 1):
            if i in marks:
                one(box, "ref", "ref_s%d_%d" % (slot, k))
                k += 1
            if i < len(mine):
                one(box, mine[i], mine[i])

    threads = [threading.Thread(target=work, args=(s,)) for s in range(plan["boxes"])]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    swarm.log("telemetry run finished -> %s" % out)


# --------------------------------------------------------------------------
# analyze
# --------------------------------------------------------------------------

PERMS = 3000
GROUPS = ("matrix", "wte", "lm_head", "value_embeds")


def features(tele: list[dict]) -> dict:
    """A small, fixed set of trajectory statistics (chosen before seeing results)."""
    f = {}
    rows = [r for r in tele if r["step"] > 2]
    if len(rows) < 8:
        return f
    flat = [r for r in rows if 0.1 <= r["progress"] <= 0.5] or rows
    for g in GROUPS:
        if g not in rows[0]:
            continue
        ratio = [math.log(r[g][1] / r[g][0]) for r in flat if r[g][0] > 0 and r[g][1] > 0]
        if ratio:
            # size of one update relative to the weights, constant-rate phase
            f["update_ratio:" + g] = float(np.mean(ratio))
        w0, w1 = rows[0][g][0], rows[-1][g][0]
        if w0 > 0 and w1 > 0:
            f["weight_growth:" + g] = math.log(w1 / w0)
        gn = [math.log(r[g][2]) for r in flat if r[g][2] > 0]
        if gn:
            f["grad_norm:" + g] = float(np.mean(gn))
        # how much of the update points down the current gradient
        al = [-r[g][3] / (r[g][1] * r[g][2]) for r in flat if r[g][1] > 0 and r[g][2] > 0]
        if al:
            f["alignment:" + g] = float(np.mean(al))
    # loss-curve statistics: reported separately, an early loss predicting the
    # final loss is not a process variable one could hold
    for q in (0.25, 0.5):
        near = sorted(rows, key=lambda r: abs(r["progress"] - q))[:5]
        f["loss_at_%d%%" % (q * 100)] = math.log(float(np.mean([r["loss"] for r in near])))
    return f


def _resid(y: np.ndarray, X: np.ndarray) -> np.ndarray:
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return y - X @ beta


def _loo_r2(y: np.ndarray, X: np.ndarray) -> float:
    """Leave-one-out R2 of a least-squares fit of y on X (X carries its own intercept)."""
    H = X @ np.linalg.pinv(X)                       # leave-one-out error = e / (1 - h)
    e = (y - H @ y) / np.clip(1.0 - np.diag(H), 1e-9, None)
    return 1.0 - float((e ** 2).sum() / ((y - y.mean()) ** 2).sum())


def _rank(a: np.ndarray) -> np.ndarray:
    r = np.empty(len(a))
    r[np.argsort(a)] = np.arange(len(a))
    return r


def cmd_analyze(args) -> None:
    out = Path(args.dir)
    plan = json.loads((out / "plan.json").read_text(encoding="utf-8"))
    info = {v["id"]: v for v in plan["versions"]}
    runs = {}
    for line in (out / "results.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        tp = out / "telemetry" / (r["tag"] + ".json")
        if r.get("val_bpb") and r.get("num_steps") and tp.exists():
            r["f"] = features(json.loads(tp.read_text(encoding="utf-8")))
            if r["f"]:
                runs[r["tag"]] = r
    rs = list(runs.values())
    boxes = sorted({r["box"] for r in rs})
    L = []
    L.append("# Training-trajectory telemetry: does it predict the result?\n")
    L.append("Source run: `%s`. %d runs with telemetry on %d boxes (%d reference runs).\n"
             % (plan["source"], len(rs), len(boxes), sum(r["id"] == "ref" for r in rs)))

    # noise: the reference version repeated on each box
    L.append("## 1. Noise: the reference version repeated\n")
    L.append("| box | runs | val_bpb | spread (sd, %) | steps |")
    L.append("| --- | --- | --- | --- | --- |")
    sds = []
    for b in boxes:
        v = [r["val_bpb"] for r in rs if r["id"] == "ref" and r["box"] == b]
        st = [r["num_steps"] for r in rs if r["id"] == "ref" and r["box"] == b]
        if v:
            sd = 100 * float(np.std(np.log(v), ddof=1)) if len(v) > 1 else float("nan")
            sds.append(sd)
            L.append("| %s | %d | %s | %.2f | %s |" % (
                b, len(v), ", ".join("%.4f" % x for x in v), sd, ", ".join(map(str, st))))
    noise = float(np.nanmean(sds)) if sds else float("nan")

    y = np.log([r["val_bpb"] for r in rs])
    ls = np.log([r["num_steps"] for r in rs])
    D = np.array([[1.0 if r["box"] == b else 0.0 for b in boxes] for r in rs])
    base_X = np.column_stack([D, ls])
    res = _resid(y, base_X)
    r2_base = 1 - float((res ** 2).sum() / ((y - y.mean()) ** 2).sum())
    L.append("\n## 2. What box and step count already explain\n")
    L.append("- Box and log steps explain %.1f%% of the variance of log val_bpb across these runs."
             % (100 * r2_base))
    L.append("- What is left has a spread of %.2f%% (sd); repeat noise is %.2f%%."
             % (100 * float(res.std(ddof=base_X.shape[1])), noise))
    L.append("- Leave-one-out R2 of box + steps alone: %.3f.\n" % _loo_r2(y, base_X))

    names = sorted({k for r in rs for k in r["f"]})
    names = [k for k in names if all(k in r["f"] for r in rs)]
    rng = np.random.default_rng(0)
    table = []
    for k in names:
        x = np.array([r["f"][k] for r in rs])
        xr = _resid(x, base_X)                       # the part box and steps do not explain
        if xr.std() < 1e-12:
            continue
        rho = float(np.corrcoef(_rank(xr), _rank(res))[0, 1])
        z = (x - x.mean()) / (x.std() or 1.0)

        def fit(v):
            return (_loo_r2(y, np.column_stack([base_X, v])),
                    _loo_r2(y, np.column_stack([base_X, v, v * v])))
        lin, quad = fit(z)
        # p: how often a shuffled statistic predicts as well (line or curve, whichever is better)
        perm = [max(fit(rng.permutation(z))) for _ in range(PERMS)]
        p = (1 + sum(q >= max(lin, quad) for q in perm)) / (PERMS + 1)
        table.append((k, rho, p, lin, quad, float(x.std())))
    r2_loo_base = _loo_r2(y, base_X)
    proc = [t for t in table if not t[0].startswith("loss_at")]
    L.append("## 3. Each trajectory statistic against what is left\n")
    L.append("Rank correlation is between the statistic and log val_bpb after box and steps are "
             "removed from both. The R2 columns are leave-one-out R2 of box + steps + the "
             "statistic (straight line, then with a squared term: a target level would show "
             "as a gain from the squared term). p is how often a shuffled statistic reaches "
             "the better of the two (%d shuffles); with %d process statistics a single one "
             "needs p < %.4f. Box + steps alone give R2 %.3f.\n"
             % (PERMS, len(proc), 0.05 / max(len(proc), 1), r2_loo_base))
    L.append("| statistic | spread (sd) | rank corr. | R2 linear | R2 quadratic | p |")
    L.append("| --- | --- | --- | --- | --- | --- |")
    for k, rho, p, lin, quad, sd in sorted(table, key=lambda t: t[2]):
        L.append("| %s | %.3f | %+.2f | %.3f | %.3f | %.4f |" % (k, sd, rho, lin, quad, p))

    # do the statistics add to what the named constants already say?
    L.append("\n## 4. Verdict\n")
    thr = 0.05 / max(len(proc), 1)
    hits = [t for t in proc if t[2] < thr and max(t[3], t[4]) > r2_loo_base + 0.01]
    if hits:
        L.append("Candidate manifold variables (pass the corrected test and improve "
                 "leave-one-out prediction): " + ", ".join("`%s`" % t[0] for t in hits) + ".")
    else:
        L.append("No process statistic passes the corrected test and improves leave-one-out "
                 "prediction over box + steps.")
    lh = [t for t in table if t[0].startswith("loss_at")]
    if lh:
        L.append("\nFor scale, the loss-curve statistics (not controllable): " + "; ".join(
            "`%s` rank corr. %+.2f, R2 %.3f" % (t[0], t[1], t[3]) for t in lh) + ".")
    kept = [(r, info[r["id"]]) for r in rs if r["id"] in info]
    if kept:
        orig = np.log([i["val"] for _, i in kept])
        now = np.log([r["val_bpb"] for r, _ in kept])
        L.append("\nRe-run against the original measurement (all boxes differ from the source "
                 "run unless reused): rank correlation %.2f over %d versions."
                 % (float(np.corrcoef(_rank(orig), _rank(now))[0, 1]), len(kept)))
    (out / "report.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan")
    p.add_argument("run")
    p.add_argument("--n", type=int, default=45)
    p.add_argument("--boxes", type=int, default=3)
    p.add_argument("--ref-repeats", type=int, default=3)
    p.add_argument("--log2-batch", type=int, default=15)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None)
    p.set_defaults(fn=cmd_plan)
    p = sub.add_parser("run")
    p.add_argument("dir")
    p.set_defaults(fn=cmd_run)
    p = sub.add_parser("analyze")
    p.add_argument("dir")
    p.set_defaults(fn=cmd_analyze)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
