#!/usr/bin/env python3
"""Analysis helper for pumpfun-top5-forecast-01 practice history (not the entry).

Reads the kit's history.ndjson + task.json and prints:
 - per-mark top 5 and churn between consecutive marks
 - births: top-5 coins created after the previous mark
 - growth trajectories (log10 cap per hour, early vs late slope)
 - venue + quote depth for the last mark's top 20
 - age at final cut + persistence (marks in top 20) for last mark's top 20
 - size distribution: top-5 caps in SOL per mark, spread per place
Stdlib only, deterministic.
"""
import json
import math
import os
import sys
from datetime import datetime, timezone

KIT = os.path.expanduser("~/workspace/makefun/pumpfun-top5-forecast-01-kit/input")
HIST = os.path.join(KIT, "history.ndjson")
TASK = os.path.join(KIT, "task.json")


def parse_ts(s):
    # ISO 8601 with Z, fractional seconds optional
    return datetime.fromisoformat(s.replace("Z", "+00:00")).astimezone(timezone.utc)


def cap_sol(row):
    return int(row["market_cap_lamports"]) / 1e9


def logcap(row):
    return math.log10(int(row["market_cap_lamports"])) - 9.0


def short(mint):
    return mint[:6] + ".." + mint[-4:]


def main():
    with open(TASK) as f:
        task = json.load(f)
    marks = []
    with open(HIST) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            m = json.loads(line)
            if m.get("ok") is True and isinstance(m.get("rows"), list):
                marks.append(m)
    marks.sort(key=lambda m: m["at"])
    print(f"task: t0={task['t0']} history_until={task['history_until']} "
          f"marks={len(marks)} places={task['places']} chart={task['chart_size']}")
    print(f"answer reads: {task['reads'][0]} .. {task['reads'][-1]} (n={len(task['reads'])})")
    for i, m in enumerate(marks):
        c = m.get("counts") or {}
        print(f"mark {i}: at={m['at']} rows={len(m['rows'])} eligible={c.get('eligible')} "
              f"graduated={c.get('graduated')} priced={c.get('priced')}")

    # ---- per-mark top5 + churn ----
    print("\n== TOP 5 PER MARK + CHURN ==")
    prev_top5 = None
    prev_at = None
    for i, m in enumerate(marks):
        rows = sorted(m["rows"], key=lambda r: r["rank"])
        top5 = rows[:5]
        print(f"\nmark {i} {m['at']}:")
        for r in top5:
            print(f"  #{r['rank']} {short(r['mint'])} {r.get('symbol','?'):<12} "
                  f"cap={cap_sol(r):10.2f} SOL log={logcap(r):.3f} venue={r['venue']} "
                  f"created={r['created_at']}")
        if prev_top5 is not None:
            cur = {r["mint"] for r in top5}
            prev = {r["mint"] for r in prev_top5}
            new = cur - prev
            dropped = prev - cur
            print(f"  churn vs mark {i-1}: new_entrants={len(new)} dropped={len(dropped)} "
                  f"kept={len(cur & prev)}")
            by_mint = {r["mint"]: r for r in top5}
            for mint in sorted(new):
                r = by_mint[mint]
                print(f"    NEW #{r['rank']} {short(mint)} {r.get('symbol','?')}")
            # births: created after previous mark's timestamp
            births = [r for r in top5 if parse_ts(r["created_at"]) > prev_at]
            print(f"  births (created after prev mark {prev_at.isoformat()}): {len(births)}/5")
            for r in births:
                age_min = (parse_ts(m["at"]) - parse_ts(r["created_at"])).total_seconds() / 60.0
                print(f"    BORN #{r['rank']} {short(r['mint'])} {r.get('symbol','?')} "
                      f"created={r['created_at']} age_at_mark={age_min:.1f} min")
        else:
            t0 = parse_ts(task["t0"])
            births = [r for r in top5 if parse_ts(r["created_at"]) >= t0]
            print(f"  first non-empty mark: {len(births)}/5 created at/after t0")
        if top5:
            prev_top5 = top5
            prev_at = parse_ts(m["at"])

    # ---- trajectories for union of all top-5 coins + last top 20 ----
    print("\n== GROWTH TRAJECTORIES ==")
    # mint -> list of (mark_idx, ts, rank, logcap, cap_sol)
    traj = {}
    info = {}
    for i, m in enumerate(marks):
        ts = parse_ts(m["at"])
        for r in m["rows"]:
            traj.setdefault(r["mint"], []).append(
                (i, ts, r["rank"], logcap(r), cap_sol(r)))
            info[r["mint"]] = r
    last = marks[-1]
    last_rows = sorted(last["rows"], key=lambda r: r["rank"])
    focus = []
    seen = set()
    for r in last_rows[:20]:
        if r["mint"] not in seen:
            focus.append(r["mint"]); seen.add(r["mint"])
    for m_ in marks:
        for r in sorted(m_["rows"], key=lambda x: x["rank"])[:5]:
            if r["mint"] not in seen:
                focus.append(r["mint"]); seen.add(r["mint"])

    def slope_per_hour(pts):
        # pts: list of (ts, logcap); least squares slope in log10 per hour
        if len(pts) < 2:
            return None
        t0 = pts[0][0]
        xs = [(t - t0).total_seconds() / 3600.0 for t, _ in pts]
        ys = [y for _, y in pts]
        n = len(xs)
        mx = sum(xs) / n; my = sum(ys) / n
        den = sum((x - mx) ** 2 for x in xs)
        if den <= 0:
            return None
        return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den

    for mint in focus:
        pts_all = sorted(traj[mint])
        pts = [(ts, lg) for _, ts, _, lg, _ in pts_all]
        total = slope_per_hour(pts)
        # early vs late: split at midpoint in time
        mid = len(pts) // 2
        early = slope_per_hour(pts[:mid + 1]) if mid >= 1 else None
        late = slope_per_hour(pts[mid:]) if len(pts) - mid >= 2 else None
        trend = "?"
        if early is not None and late is not None:
            if late > early + 0.05:
                trend = "ACCELERATING"
            elif late < early - 0.05:
                trend = "FADING"
            else:
                trend = "STEADY"
        elif total is not None:
            trend = "UP" if total > 0.05 else ("DOWN" if total < -0.05 else "FLAT")
        r = info[mint]
        ranks = [rk for _, _, rk, _, _ in pts_all]
        caps = [c for _, _, _, _, c in pts_all]
        print(f"{short(mint)} {r.get('symbol','?'):<12} marks_seen={len(pts_all)} "
              f"ranks={ranks} cap_first={caps[0]:.1f} cap_last={caps[-1]:.1f} SOL "
              f"slope={total if total is None else round(total,3)} log/h "
              f"early={early if early is None else round(early,3)} "
              f"late={late if late is None else round(late,3)} -> {trend}")

    # ---- venue + depth, age, persistence for last top 20 ----
    print("\n== LAST MARK TOP 20: VENUE / DEPTH / AGE / PERSISTENCE ==")
    cut = parse_ts(last["at"])
    # persistence: marks in top 20
    persist = {}
    for m in marks:
        top20 = {r["mint"] for r in sorted(m["rows"], key=lambda x: x["rank"])[:20]}
        for mint in top20:
            persist[mint] = persist.get(mint, 0) + 1
    venue_count = {}
    for r in last_rows[:20]:
        venue_count[r["venue"]] = venue_count.get(r["venue"], 0) + 1
    print(f"venue split in last top20: {venue_count}")
    for r in last_rows[:20]:
        age_h = (cut - parse_ts(r["created_at"])).total_seconds() / 3600.0
        depth_sol = int(r.get("quote_reserves_lamports") or 0) / 1e9
        print(f"#{r['rank']:>2} {short(r['mint'])} {r.get('symbol','?'):<12} "
              f"cap={cap_sol(r):9.2f} SOL venue={r['venue']:<9} complete={r.get('complete')} "
              f"mayhem={r.get('mayhem')} depth={depth_sol:9.2f} SOL "
              f"age_at_cut={age_h:5.2f} h persist_top20={persist.get(r['mint'],0)}/{len(marks)}")

    # ---- size distribution per place ----
    print("\n== SIZE DISTRIBUTION: TOP-5 CAPS (SOL) PER MARK ==")
    per_place = {k: [] for k in range(1, 6)}
    for i, m in enumerate(marks):
        rows = sorted(m["rows"], key=lambda r: r["rank"])[:5]
        vals = []
        for k, r in enumerate(rows, start=1):
            per_place[k].append(logcap(r))
            vals.append(f"#{k}={cap_sol(r):.1f}(log {logcap(r):.2f})")
        if vals:
            print(f"mark {i} {m['at']}: " + "  ".join(vals))
    print("\nper-place log10(cap SOL) across marks:")
    for k in range(1, 6):
        v = per_place[k]
        if v:
            print(f"  place {k}: n={len(v)} min={min(v):.3f} max={max(v):.3f} "
                  f"spread={max(v)-min(v):.3f} mean={sum(v)/len(v):.3f} "
                  f"values={[round(x,2) for x in v]}")

    # ---- extra churn stats: top20 turnover + distinct top5 holders ----
    print("\n== EXTRA: TOP-5 HOLDER SET OVER ALL MARKS ==")
    holders = {}
    for i, m in enumerate(marks):
        for r in sorted(m["rows"], key=lambda x: x["rank"])[:5]:
            holders.setdefault(r["mint"], []).append(i)
    print(f"distinct coins ever in top5: {len(holders)}")
    for mint, idxs in sorted(holders.items(), key=lambda kv: kv[1][0]):
        print(f"  {short(mint)} {info[mint].get('symbol','?'):<12} in top5 at marks {idxs}")
    # top20 turnover consecutive
    print("\ntop20 turnover between consecutive marks:")
    prev20 = None
    for i, m in enumerate(marks):
        cur20 = {r["mint"] for r in sorted(m["rows"], key=lambda x: x["rank"])[:20]}
        if prev20 is not None and cur20:
            print(f"  mark {i-1}->{i}: new_in_top20={len(cur20-prev20)} kept={len(cur20 & prev20)}")
        if cur20:
            prev20 = cur20


if __name__ == "__main__":
    main()
