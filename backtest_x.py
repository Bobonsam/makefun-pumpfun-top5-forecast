#!/usr/bin/env python3
"""Expanded backtest (helper, not the entry): pseudo-windows at every
feasible cut of the practice history (cuts after marks 1..4; mark 0
is empty and mark 5 has no future), each scored with the RULES log
score on every subsequent mark as a proxy answer read. Usage:

  python3 backtest_x.py <path-to-main.py> [label]

Prints per-cut means and the pooled mean over all reads (the number
to beat), with order/size splits.
"""
import json
import math
import os
import subprocess
import sys
import tempfile

KIT = os.path.expanduser("~/workspace/makefun/pumpfun-top5-forecast-01-kit/input")
FLOOR = 0.0001


def logcap(lamports):
    return math.log10(int(lamports)) - 9.0


def bin_of(lg, edges):
    if lg < edges[0]:
        return 0
    if lg >= edges[-1]:
        return len(edges)
    return 1 + int((lg - edges[0]) / 0.05)


def score(forecast, actual_rows, task):
    places = task["places"]
    edges = task["size"]["edges"]
    chart = sorted(actual_rows, key=lambda r: r["rank"])[:task["chart_size"]]
    chart_mints = [r["mint"] for r in chart]
    named = {}
    for r in forecast["probabilities"]:
        named.setdefault(r["place"], {})[r["mint"]] = r["p"]
    snamed = {}
    for r in forecast["sizes"]:
        snamed.setdefault(r["place"], {})[r["bin"]] = r["p"]
    order = 0.0
    size = 0.0
    for place in range(1, places + 1):
        actual = chart[place - 1]
        nd = named.get(place, {})
        s = sum(nd.values())
        not_named = sum(1 for m in chart_mints if m not in nd)
        unnamed_p = (1.0 - s) / not_named if not_named else 0.0
        p = nd.get(actual["mint"], unnamed_p)
        order += math.log(max(p, FLOOR))
        sd = snamed.get(place, {})
        ss = sum(sd.values())
        b = bin_of(logcap(actual["market_cap_lamports"]), edges)
        n_not = (len(edges) + 1) - len(sd)
        unnamed_b = (1.0 - ss) / n_not if n_not else 0.0
        pb = sd.get(b, unnamed_b)
        size += math.log(max(pb, FLOOR))
    return order, size


def run(main_path, marks, task, cut_idx):
    future = marks[cut_idx + 1:]
    trunc_dir = tempfile.mkdtemp(prefix=f"btx{cut_idx}-in-")
    out_dir = tempfile.mkdtemp(prefix=f"btx{cut_idx}-out-")
    with open(os.path.join(trunc_dir, "history.ndjson"), "w") as fh:
        for m in marks[:cut_idx + 1]:
            fh.write(json.dumps(m) + "\n")
    t2 = dict(task)
    t2["reads"] = [m["at"] for m in future]
    t2["marks"] = cut_idx + 1
    with open(os.path.join(trunc_dir, "task.json"), "w") as fh:
        json.dump(t2, fh)
    env = dict(os.environ, MAKEFUN_INPUT_DIR=trunc_dir, MAKEFUN_OUTPUT_DIR=out_dir)
    r = subprocess.run([sys.executable, main_path], env=env,
                       capture_output=True, text=True)
    if r.returncode != 0:
        return None, r.stderr.strip().splitlines()[-1] if r.stderr else "crash"
    with open(os.path.join(out_dir, "forecast.json")) as fh:
        fc = json.load(fh)
    per_read = []
    for m in future:
        o, s = score(fc, m["rows"], task)
        per_read.append((o, s))
    return per_read, None


def main():
    main_path = os.path.abspath(sys.argv[1])
    label = sys.argv[2] if len(sys.argv) > 2 else os.path.basename(main_path)
    with open(os.path.join(KIT, "task.json")) as fh:
        task = json.load(fh)
    marks = []
    with open(os.path.join(KIT, "history.ndjson")) as fh:
        for line in fh:
            if line.strip():
                m = json.loads(line)
                if m.get("ok") is True and isinstance(m.get("rows"), list):
                    marks.append(m)
    marks.sort(key=lambda m: m["at"])
    all_o, all_s = [], []
    print(f"== {label} ==")
    for cut in (1, 2, 3, 4):
        per_read, err = run(main_path, marks, task, cut)
        if per_read is None:
            print(f"cut {cut}: INFEASIBLE ({err})")
            continue
        mo = sum(o for o, _ in per_read) / len(per_read)
        ms = sum(s for _, s in per_read) / len(per_read)
        print(f"cut {cut}: reads={len(per_read)} order={mo:.3f} size={ms:.3f} total={mo+ms:.3f}")
        all_o.extend(o for o, _ in per_read)
        all_s.extend(s for _, s in per_read)
    n = len(all_o)
    print(f"POOLED over {n} reads: order={sum(all_o)/n:.3f} size={sum(all_s)/n:.3f} "
          f"TOTAL={sum(all_o)/n + sum(all_s)/n:.3f}")


if __name__ == "__main__":
    main()
