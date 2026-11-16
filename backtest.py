#!/usr/bin/env python3
"""Backtest helper (not the entry): truncate the practice history,
forecast with main.py, and score against the marks that really
followed, using the RULES scoring formula on those marks as proxy
answer reads. Compares main.py against a persistence forecast built
the way the house baseline is described (0.30 stay / 0.10 neighbours,
sizes normal sd 0.35 around the last mark's caps).
"""
import json
import math
import os
import subprocess
import sys
import tempfile

KIT = os.path.expanduser("~/workspace/makefun/pumpfun-top5-forecast-01-kit/input")
REPO = os.path.dirname(os.path.abspath(__file__))
FLOOR = 0.0001


def logcap(lamports):
    return math.log10(int(lamports)) - 9.0


def erf_cdf(x, mean, sd):
    return 0.5 * (1.0 + math.erf((x - mean) / (sd * math.sqrt(2.0))))


def bin_of(lg, edges):
    if lg < edges[0]:
        return 0
    if lg >= edges[-1]:
        return len(edges)
    return 1 + int((lg - edges[0]) / 0.05)


def persistence_forecast(last_rows, task):
    places = task["places"]
    edges = task["size"]["edges"]
    chart = sorted(last_rows, key=lambda r: r["rank"])
    probs = []
    for place in range(1, places + 1):
        probs.append({"mint": chart[place - 1]["mint"], "place": place, "p": 0.30})
        if place > 1:
            probs.append({"mint": chart[place - 2]["mint"], "place": place, "p": 0.10})
        if chart[place] if place < len(chart) else None:
            probs.append({"mint": chart[place]["mint"], "place": place, "p": 0.10})
    sizes = []
    for place in range(1, places + 1):
        mean = logcap(chart[place - 1]["market_cap_lamports"])
        for b in range(len(edges) + 1):
            lo = -math.inf if b == 0 else edges[b - 1]
            hi = math.inf if b == len(edges) else edges[b]
            u = math.floor((erf_cdf(hi, mean, 0.35) - erf_cdf(lo, mean, 0.35)) * 1_000_000)
            if u > 0:
                sizes.append({"place": place, "bin": b, "p": u / 1_000_000})
    return {"probabilities": probs,
            "top5": [r["mint"] for r in chart[:places]],
            "sizes": sizes}


def score(forecast, actual_rows, task, names):
    """RULES log score of one forecast at one actual mark."""
    places = task["places"]
    edges = task["size"]["edges"]
    chart_size = task["chart_size"]
    chart = sorted(actual_rows, key=lambda r: r["rank"])[:chart_size]
    chart_mints = [r["mint"] for r in chart]
    named = {}
    for r in forecast["probabilities"]:
        named.setdefault(r["place"], {})[r["mint"]] = r["p"]
    snamed = {}
    for r in forecast["sizes"]:
        snamed.setdefault(r["place"], {})[r["bin"]] = r["p"]
    order = 0.0
    size = 0.0
    detail = []
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
        detail.append({
            "place": place,
            "symbol": names.get(actual["mint"], "?"),
            "mint": actual["mint"],
            "named_p": nd.get(actual["mint"]),
            "eff_p": p,
            "size_bin": b,
            "size_p": pb,
        })
    return order, size, detail


def run_cut(marks, task, cut_idx, names):
    future = marks[cut_idx + 1:]
    trunc_dir = tempfile.mkdtemp(prefix=f"bt-cut{cut_idx}-in-")
    out_dir = tempfile.mkdtemp(prefix=f"bt-cut{cut_idx}-out-")
    with open(os.path.join(trunc_dir, "history.ndjson"), "w") as fh:
        for m in marks[:cut_idx + 1]:
            fh.write(json.dumps(m) + "\n")
    t2 = dict(task)
    t2["reads"] = [m["at"] for m in future]
    t2["marks"] = cut_idx + 1
    with open(os.path.join(trunc_dir, "task.json"), "w") as fh:
        json.dump(t2, fh)
    env = dict(os.environ, MAKEFUN_INPUT_DIR=trunc_dir, MAKEFUN_OUTPUT_DIR=out_dir)
    subprocess.run([sys.executable, os.path.join(REPO, "main.py")],
                   env=env, check=True, capture_output=True, text=True)
    with open(os.path.join(out_dir, "forecast.json")) as fh:
        mine = json.load(fh)
    base = persistence_forecast(marks[cut_idx]["rows"], task)

    print(f"\n===== CUT AFTER MARK {cut_idx} ({marks[cut_idx]['at']}) =====")
    print(f"my top5: {[names.get(m, '?') for m in mine['top5']]}")
    print(f"persistence top5: {[names.get(m, '?') for m in base['top5']]}")
    tot_mine = tot_base = 0.0
    for m in future:
        o1, s1, d1 = score(mine, m["rows"], task, names)
        o2, s2, d2 = score(base, m["rows"], task, names)
        tot_mine += o1 + s1
        tot_base += o2 + s2
        print(f"\n-- actual mark {m['at']} --")
        print(f"  {'pl':<3}{'actual':<12}{'mine p':>9}{'mine ln':>9}{'persist p':>11}{'persist ln':>11}")
        for a, b in zip(d1, d2):
            mp = a["named_p"] if a["named_p"] is not None else a["eff_p"]
            bp = b["named_p"] if b["named_p"] is not None else b["eff_p"]
            tag_m = "" if a["named_p"] is not None else " (unnamed)"
            tag_b = "" if b["named_p"] is not None else " (unnamed)"
            print(f"  {a['place']:<3}{a['symbol']:<12}{mp:>9.4f}{math.log(max(a['eff_p'],FLOOR)):>9.2f}"
                  f"{bp:>11.4f}{math.log(max(b['eff_p'],FLOOR)):>11.2f}  mine{tag_m} persist{tag_b}")
        print(f"  ORDER mine={o1:.3f} persist={o2:.3f} | SIZE mine={s1:.3f} persist={s2:.3f} "
              f"| TOTAL mine={o1+s1:.3f} persist={o2+s2:.3f}")
    n = len(future)
    print(f"\nCUT {cut_idx} MEAN over {n} reads: mine={tot_mine/n:.3f} persistence={tot_base/n:.3f} "
          f"diff={tot_mine/n - tot_base/n:+.3f}")
    return tot_mine / n, tot_base / n


def main():
    with open(os.path.join(KIT, "task.json")) as fh:
        task = json.load(fh)
    marks = []
    names = {}
    with open(os.path.join(KIT, "history.ndjson")) as fh:
        for line in fh:
            if not line.strip():
                continue
            m = json.loads(line)
            if m.get("ok") is True and isinstance(m.get("rows"), list):
                marks.append(m)
                for r in m["rows"]:
                    names[r["mint"]] = r.get("symbol") or "?"
    marks.sort(key=lambda m: m["at"])
    nonempty = [i for i, m in enumerate(marks) if m["rows"]]
    results = []
    for cut in (3, 4):
        if cut in nonempty or marks[cut]["rows"]:
            results.append(run_cut(marks, task, cut, names))
    print("\n===== SUMMARY =====")
    for cut, (a, b) in zip((3, 4), results):
        print(f"cut {cut}: mine={a:.3f} persistence={b:.3f} diff={a-b:+.3f}")


if __name__ == "__main__":
    main()
