#!/usr/bin/env python3
"""pumpfun-top5-forecast-01 entry (Python 3, stdlib only, deterministic).

Forecasts the top 5 pump.fun coins by market cap at the house's answer
reads. Reads history.ndjson + task.json from $MAKEFUN_INPUT_DIR (else
/input) and writes forecast.json to $MAKEFUN_OUTPUT_DIR (else /output).

Method, in one paragraph: the house baseline just persists the last
mark, but the practice history churns violently -- most early top-5
places were taken by coins born minutes earlier, and even late marks
saw a newborn take #1. So every coin is scored on more than its last
rank: a recency-weighted log-cap growth slope projected over the
forecast horizon (soft-capped so a vertical first hour cannot
extrapolate forever), venue durability (a graduated PumpSwap pool
outlasts a bonding curve), SOL depth, persistence in the top 20, a
small bonus for young coins that are already large, and a staleness
penalty for coins missing from the last mark. Predicted caps are
ranked; a Laplace kernel around each place's predicted cap spreads
probability over the top ~15 candidates, with only ~0.55 of each
place named -- the remaining ~0.45 stays unnamed for the coins born
after the cut, which the practice answer shows take top places
regularly. Sizes are a normal in log10(cap SOL) around each place's
drifted predicted cap (sd ~0.5, wider than the baseline's 0.35
because the horizon is long), blended with the empirical
distribution of top-5 caps across all marks, floored so no bin is
ever given zero.

Revision (after the first submission scored -39.10 on the practice
board, outside the paid places): two mechanisms, both validated on
truncated practice histories. (1) Horizon robustness -- the final
runs this same program on a fresh cut whose reads are ~7-13 h out,
not the practice's ~1 h, so the named budget, the growth damping
and the size sd are now explicit functions of the horizon computed
from task.json: longer horizon, larger unnamed share, stronger
mean reversion, wider sizes; at practice horizons all three sit
at (or negligibly near) the v1 values above. (2) A survivor term
-- credit for the share of the whole window's marks a coin spent
in the top 5 -- was built and tested at weights 0.16 and 0.08; it
hurt both truncation backtests at both weights, so it ships
disabled (SURVIVOR_W = 0.0). The horizon functions are the round's
shipped change: backtest-neutral at practice horizons (+0.01 /
+0.03 vs v1) and aimed squarely at the final's geometry.

Second improvement round (v3): an empirical transition component.
A diagnostic over all mark pairs of the practice history showed
the finish distribution is governed by state far more than by cap
proximity: the coin ranked #1 at any origin mark finished in the
top 5 at every later mark, coins ranked 6-20 at an origin
essentially never finished top-5, and no bonding-curve coin ever
did, while slope and depth-growth splits were weak and mutually
collinear (the model already uses slope). So the per-place
candidate weights now blend the Laplace kernel (weight 0.15) with
a transition table counted from the input history itself -- for
each mark pair, where coins in each (rank bucket x venue) state
actually finished -- shrunk toward the pooled outcome rates. On
the expanded backtest set (all four feasible cuts, 10 proxy
reads) the blend improves the pooled score from -44.362 (v2) to
-43.830, clearing the pre-set +0.15 shipping bar; weights were
grid-searched on that set only. Sizes are unchanged from v2.
"""
import json
import math
import os
from datetime import datetime, timezone

INPUT_DIR = os.environ.get("MAKEFUN_INPUT_DIR") or "/input"
OUTPUT_DIR = os.environ.get("MAKEFUN_OUTPUT_DIR") or "/output"

# Fraction of each place given to named coins at short horizons;
# the rest is left unnamed for post-cut births and unmodelled
# movers. The budget is a function of the forecast horizon (see
# main): the longer the reads are past the cut, the more of the
# chart is expected to be coins nobody has seen yet, so the named
# budget falls toward NAMED_BUDGET_FLOOR.
NAMED_BUDGET = 0.55
NAMED_BUDGET_FLOOR = 0.40
BUDGET_DECAY_PER_H = 0.0167  # per horizon hour beyond the first
# Mean reversion: growth slopes are damped by 1 / (1 + h / DAMP_H),
# so a hot first hour counts for less the further out the reads are.
DAMP_H = 6.0
# How many leading candidates share each place's named budget.
N_CANDIDATES = 15
# Growth soft-cap: projected log-cap growth over the whole horizon
# saturates at this many log10 units (tanh shape).
MAX_GROWTH_LOG = 1.05
# Survivor bonus weight: share of the window's marks a coin spent
# in the top 5 (see the scoring loop). Tested at 0.16 and 0.08 in
# the improvement round: it hurt BOTH truncation backtests at both
# weights (the practice's most persistent coin is a fader, and the
# boost mostly dilutes the leaders' kernel mass), so by the rule
# that a change hurting both truncations does not ship, the weight
# is 0.0 in the shipped version. The mechanism stays in place,
# documented, in case a longer practice history ever vindicates it.
SURVIVOR_W = 0.0
# Empirical transition component: weight of the transition table
# (rank bucket x venue at an origin mark -> place at a later mark,
# counted over all mark pairs of the input history itself) in the
# per-place candidate weights, blended against the Laplace kernel.
# Grid-searched on an expanded backtest set (every feasible cut of
# the practice history, 10 proxy reads): the pooled score rises
# monotonically from the kernel-only model to a plateau at
# 0.80-0.90 and falls at 1.00 (pure empirical loses the kernel's
# cap-proximity information); 0.85 is the plateau's centre.
EMPIRICAL_W = 0.85
# Shrinkage: pseudo-count mass pulling each state's outcome
# distribution toward the pooled outcome distribution.
TRANSITION_ALPHA = 6.0
# Rows per place below this probability are dropped (their mass
# returns to the unnamed share).
MIN_ROW_P = 0.003
# Size distribution: normal sd in log10(cap SOL) at short horizons,
# widening with the horizon (SIZE_SD_GROWTH per hour beyond the
# first, capped at SIZE_SD_MAX), and blend weight of the empirical
# top-5 cap distribution.
SIZE_SD = 0.5
SIZE_SD_GROWTH = 0.035
SIZE_SD_MAX = 0.85
SIZE_EMPIRICAL_W = 0.28
SIZE_FLOOR = 0.0006
SIZE_TOTAL = 0.99


def parse_ts(text):
    return datetime.fromisoformat(text.replace("Z", "+00:00")).astimezone(timezone.utc)


def logcap_of(lamports):
    return math.log10(int(lamports)) - 9.0


def median(values):
    vals = sorted(values)
    n = len(vals)
    if n == 0:
        return 0.0
    mid = n // 2
    if n % 2:
        return vals[mid]
    return 0.5 * (vals[mid - 1] + vals[mid])


def erf_cdf(x, mean, sd):
    return 0.5 * (1.0 + math.erf((x - mean) / (sd * math.sqrt(2.0))))


def weighted_slope(points, cut, half_life_h):
    """Recency-weighted least-squares slope of log-cap per hour.

    points: list of (timestamp, logcap). Recent marks weigh more.
    Returns None with fewer than 2 usable points.
    """
    if len(points) < 2:
        return None
    t0 = points[0][0]
    xs, ys, ws = [], [], []
    for ts, y in points:
        age_h = (cut - ts).total_seconds() / 3600.0
        w = 0.5 ** (age_h / half_life_h) if half_life_h > 0 else 1.0
        xs.append((ts - t0).total_seconds() / 3600.0)
        ys.append(y)
        ws.append(w)
    sw = sum(ws)
    mx = sum(w * x for x, w in zip(xs, ws)) / sw
    my = sum(w * y for y, w in zip(ys, ws)) / sw
    den = sum(w * (x - mx) ** 2 for x, w in zip(xs, ws))
    if den <= 1e-12:
        return None
    num = sum(w * (x - mx) * (y - my) for x, y, w in zip(xs, ys, ws))
    return num / den


def main():
    with open(os.path.join(INPUT_DIR, "task.json"), "r", encoding="utf-8") as fh:
        task = json.load(fh)
    places = int(task["places"])
    edges = [float(e) for e in task["size"]["edges"]]
    out_name = task.get("output") or "forecast.json"
    t0 = parse_ts(task["t0"])

    marks = []
    with open(os.path.join(INPUT_DIR, "history.ndjson"), "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            mark = json.loads(line)
            if mark.get("ok") is True and isinstance(mark.get("rows"), list):
                marks.append(mark)
    marks.sort(key=lambda m: m["at"])
    if not marks:
        raise ValueError("history holds no ok mark")
    cut = parse_ts(marks[-1]["at"])
    first_t = parse_ts(marks[0]["at"])
    span_h = max((cut - first_t).total_seconds() / 3600.0, 0.05)
    half_life_h = max(span_h / 2.0, 0.05)

    # Forecast horizon: mean lead of the answer reads past the cut,
    # all read from task.json (practice ~0.7 h, final ~10 h).
    leads_h = []
    for read_at in task.get("reads") or []:
        leads_h.append((parse_ts(read_at) - cut).total_seconds() / 3600.0)
    horizon_h = sum(leads_h) / len(leads_h) if leads_h else span_h
    horizon_h = max(horizon_h, 0.05)

    # Horizon-dependent calibration. All three are anchored to the
    # tuned v1 values at horizons up to ~1 h (the practice geometry)
    # and move only as the horizon grows past that (the final's
    # reads sit ~7-13 h past the cut):
    # - named budget falls toward its floor: more post-cut births;
    # - growth damping strengthens: more mean reversion;
    # - size sd widens: cap levels that far out are less certain.
    hours_beyond = max(0.0, horizon_h - 1.0)
    named_budget = max(NAMED_BUDGET_FLOOR,
                       NAMED_BUDGET - BUDGET_DECAY_PER_H * hours_beyond)
    growth_damp = 1.0 / (1.0 + horizon_h / DAMP_H)
    size_sd = min(SIZE_SD_MAX, SIZE_SD + SIZE_SD_GROWTH * hours_beyond)

    # ---- collect per-coin observations -------------------------------
    coins = {}
    top_caps_empirical = []  # log-caps of top-`places` rows, every mark
    for idx, mark in enumerate(marks):
        ts = parse_ts(mark["at"])
        rows = sorted(mark["rows"], key=lambda r: r["rank"])
        for place_idx, row in enumerate(rows[:places], start=1):
            top_caps_empirical.append((place_idx, logcap_of(row["market_cap_lamports"])))
        for row in rows:
            mint = row["mint"]
            c = coins.get(mint)
            if c is None:
                c = coins[mint] = {
                    "mint": mint,
                    "created": parse_ts(row["created_at"]),
                    "obs": [],          # (mark_idx, ts, rank, logcap)
                    "top20_marks": 0,
                    "top5_marks": 0,
                    "last": None,       # row at its most recent sighting
                    "last_idx": -1,
                }
            c["obs"].append((idx, ts, int(row["rank"]),
                             logcap_of(row["market_cap_lamports"])))
            if int(row["rank"]) <= 20:
                c["top20_marks"] += 1
            if int(row["rank"]) <= 5:
                c["top5_marks"] += 1
            if idx >= c["last_idx"]:
                c["last"] = row
                c["last_idx"] = idx
    last_idx = len(marks) - 1

    # Cohort slope: median slope of coins seen at least twice and
    # present at the last mark -- the market's background drift.
    slopes = {}
    for mint, c in coins.items():
        pts = [(ts, lg) for _, ts, _, lg in c["obs"]]
        s = weighted_slope(pts, cut, half_life_h)
        if s is not None:
            # Shrink by observation count: a 2-point slope is a rumour.
            n = len(pts)
            slopes[mint] = s * (n / (n + 1.0))
    cohort_slope = median([s for mint, s in slopes.items()
                           if coins[mint]["last_idx"] == last_idx]) if slopes else 0.0
    cohort_slope = max(min(cohort_slope, 1.5), -1.5)

    # ---- score every coin: predicted log-cap at the horizon ---------
    window_h = max((cut - t0).total_seconds() / 3600.0, span_h)
    # Marks that actually ranked coins (the opening mark may be
    # empty): the survivor share's denominator.
    marks_window = max(1, sum(1 for m in marks if m["rows"]))
    scored = []
    for mint, c in coins.items():
        last_row = c["last"]
        last_log = c["obs"][-1][3]
        last_rank = c["obs"][-1][2]
        slope = slopes.get(mint)
        if slope is None:
            slope = 0.5 * cohort_slope
        slope = max(min(slope, 2.5), -2.5)
        raw_growth = slope * growth_damp * horizon_h
        growth = MAX_GROWTH_LOG * math.tanh(raw_growth / MAX_GROWTH_LOG)
        pred = last_log + growth

        # Venue durability: graduated pools hold value; curves are
        # one sell wave from sliding back.
        venue = last_row.get("venue")
        if venue == "pool":
            pred += 0.06
        elif venue == "migrating":
            pred += 0.02
        else:  # curve
            pred -= 0.04

        # Depth: log SOL actually behind the price.
        depth_sol = int(last_row.get("quote_reserves_lamports") or 0) / 1e9
        depth_bonus = 0.02 * (math.log10(depth_sol + 1.0) - 1.5)
        pred += max(min(depth_bonus, 0.06), -0.04)

        # Persistence: share of the marks since birth spent in top 20.
        alive_marks = sum(1 for m in marks if parse_ts(m["at"]) >= c["created"])
        persist = c["top20_marks"] / alive_marks if alive_marks else 0.0
        pred += 0.08 * persist

        # Survivor term: share of the whole window's marks spent in
        # the top 5 (not just since birth -- a coin born one mark ago
        # has not survived anything yet). Deliberately separate from
        # the slope, which already rewards the young risers.
        pred += SURVIVOR_W * (c["top5_marks"] / marks_window)

        # Young and already large: the pattern of every practice
        # rocket (born minutes before topping the chart).
        age_h = max((cut - c["created"]).total_seconds() / 3600.0, 0.0)
        if last_rank <= 10 and age_h <= 0.25 * window_h:
            pred += 0.05

        # Staleness: absent from later marks means it faded or the
        # recorder stopped ranking it -- either way it is not leading.
        stale_marks = last_idx - c["last_idx"]
        if stale_marks:
            pred -= 0.25 * stale_marks

        scored.append({
            "mint": mint,
            "pred": pred,
            "last_log": last_log,
            "last_rank": last_rank,
            "venue": last_row.get("venue"),
        })
    # Rank by predicted cap; mint order breaks ties deterministically.
    scored.sort(key=lambda c: (-c["pred"], c["mint"]))
    candidates = scored[:max(N_CANDIDATES, places)]

    # ---- empirical transitions --------------------------------------
    # From the input history alone: for every ordered pair of marks
    # (origin i, target j > i), every coin ranked <= 20 at i donates
    # its place at j (1..places, or 0 = not top-5 / absent) to the
    # count of its state (rank bucket x venue group) at i. Pairs are
    # weighted by sqrt(gap / horizon) capped at 1, floored at 0.2:
    # longer gaps resemble the forecast task more. Each state's
    # outcome distribution is shrunk toward the pooled distribution
    # (TRANSITION_ALPHA pseudo-counts), so sparse states fall back
    # to the base rates instead of shouting. In the practice history
    # this table is stark -- the #1 coin at any origin finished top-5
    # at every later mark, coins ranked 6-20 essentially never did,
    # and no curve coin ever did -- which the parametric kernel,
    # built on cap proximity alone, cannot express.
    def rank_bucket(rank):
        if rank <= 1:
            return 0
        if rank <= 3:
            return 1
        if rank <= 5:
            return 2
        if rank <= 10:
            return 3
        return 4  # 11..20

    mark_maps = []
    for mark in marks:
        mmap = {}
        for row in mark["rows"]:
            mmap[row["mint"]] = row
        mark_maps.append((parse_ts(mark["at"]), mmap))
    trans_counts = {}   # (bucket, venue_group) -> [c0, c1..c_places]
    pooled_counts = [0.0] * (places + 1)
    for i in range(len(mark_maps)):
        ti, mi = mark_maps[i]
        origins = [r for r in mi.values() if int(r["rank"]) <= 20]
        if not origins:
            continue
        for j in range(i + 1, len(mark_maps)):
            tj, mj = mark_maps[j]
            gap_h = (tj - ti).total_seconds() / 3600.0
            w_pair = min(1.0, math.sqrt(gap_h / horizon_h)) if horizon_h > 0 else 1.0
            w_pair = max(0.2, w_pair)
            for row in origins:
                key = (rank_bucket(int(row["rank"])),
                       1 if row.get("venue") == "pool" else 0)
                target = mj.get(row["mint"])
                outcome = int(target["rank"]) if target is not None and int(target["rank"]) <= places else 0
                cell = trans_counts.get(key)
                if cell is None:
                    cell = trans_counts[key] = [0.0] * (places + 1)
                cell[outcome] += w_pair
                pooled_counts[outcome] += w_pair
    pooled_total = sum(pooled_counts)
    if pooled_total > 0:
        prior = [(pooled_counts[k] + 1.0) / (pooled_total + places + 1.0)
                 for k in range(places + 1)]
    else:
        prior = [1.0 / (places + 1.0)] * (places + 1)

    def empirical_place(cand, place):
        """Shrunk P(candidate finishes at `place` | its last state)."""
        if cand["last_rank"] > 20:
            return prior[place]
        key = (rank_bucket(cand["last_rank"]), 1 if cand["venue"] == "pool" else 0)
        cell = trans_counts.get(key)
        if cell is None:
            return prior[place]
        total = sum(cell)
        return (cell[place] + TRANSITION_ALPHA * prior[place]) / (total + TRANSITION_ALPHA)

    # ---- probabilities: Laplace kernel around each place's cap ------
    kernel_b = min(0.50, 0.28 + 0.02 * horizon_h)
    probabilities = []
    for place in range(1, places + 1):
        if place <= len(candidates):
            ref = candidates[place - 1]["pred"]
        else:
            ref = candidates[-1]["pred"] - 0.15 * (place - len(candidates))
        weights = []
        for cand in candidates:
            w_kernel = math.exp(-abs(cand["pred"] - ref) / kernel_b)
            w = (1.0 - EMPIRICAL_W) * w_kernel \
                + EMPIRICAL_W * empirical_place(cand, place)
            weights.append((cand["mint"], w))
        total_w = sum(w for _, w in weights)
        if total_w <= 0:
            continue
        for mint, w in weights:
            p = named_budget * w / total_w
            p = min(p, 0.50)
            p = math.floor(p * 1_000_000) / 1_000_000
            if p >= MIN_ROW_P:
                probabilities.append({"mint": mint, "place": place, "p": p})
    probabilities.sort(key=lambda r: (r["place"], -r["p"], r["mint"]))

    top5 = [c["mint"] for c in scored[:places]]

    # ---- sizes -------------------------------------------------------
    # Empirical distribution of top-5 caps across all marks (pooled,
    # lightly smoothed over neighbouring bins).
    n_bins = len(edges) + 1
    emp_counts = [0.0] * n_bins
    for _, lg in top_caps_empirical:
        if lg < edges[0]:
            b = 0
        elif lg >= edges[-1]:
            b = n_bins - 1
        else:
            b = 1 + int((lg - edges[0]) / 0.05)
            b = max(1, min(n_bins - 2, b))
        emp_counts[b] += 1.0
    smoothed = []
    for i in range(n_bins):
        v = emp_counts[i]
        if i > 0:
            v += 0.25 * emp_counts[i - 1]
        if i < n_bins - 1:
            v += 0.25 * emp_counts[i + 1]
        smoothed.append(v)
    emp_total = sum(smoothed)
    empirical = [v / emp_total for v in smoothed] if emp_total > 0 else [1.0 / n_bins] * n_bins

    # Cohort drift for place centres: last mark's place cap pushed up
    # by the (soft-capped) background slope over the horizon.
    # Backtest note (truncated practice histories, cuts after marks
    # 3 and 4): two alternative centres were tried and both scored
    # worse on sizes at both cuts -- (a) per-place empirical blends
    # with place-trajectory drift, (b) the drifted place cap alone.
    # The v1 centre below (mean of the predicted holder's cap and
    # the drifted place cap) stayed best; the remaining size gap to
    # persistence at 10-20 minute backtest horizons is the price of
    # sd 0.5, kept deliberately for the far longer real horizon.
    last_rows = sorted(marks[-1]["rows"], key=lambda r: r["rank"])
    drift = MAX_GROWTH_LOG * math.tanh(cohort_slope * growth_damp * horizon_h / MAX_GROWTH_LOG)
    sizes = []
    for place in range(1, places + 1):
        if place <= len(candidates):
            centre_pred = candidates[place - 1]["pred"]
        else:
            centre_pred = candidates[-1]["pred"] - 0.15 * (place - len(candidates))
        if place <= len(last_rows):
            centre_last = logcap_of(last_rows[place - 1]["market_cap_lamports"]) + drift
        else:
            centre_last = centre_pred
        centre = 0.5 * (centre_pred + centre_last)
        raw = []
        for b in range(n_bins):
            lo = -math.inf if b == 0 else edges[b - 1]
            hi = math.inf if b == n_bins - 1 else edges[b]
            norm_p = erf_cdf(hi, centre, size_sd) - erf_cdf(lo, centre, size_sd)
            p = (1.0 - SIZE_EMPIRICAL_W) * norm_p + SIZE_EMPIRICAL_W * empirical[b]
            raw.append(max(p, SIZE_FLOOR))
        scale = SIZE_TOTAL / sum(raw)
        for b, p in enumerate(raw):
            p = math.floor(p * scale * 1_000_000) / 1_000_000
            if p > 0:
                sizes.append({"place": place, "bin": b, "p": p})
    sizes.sort(key=lambda r: (r["place"], r["bin"]))

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    payload = {"probabilities": probabilities, "top5": top5, "sizes": sizes}
    with open(os.path.join(OUTPUT_DIR, out_name), "w", encoding="utf-8") as fh:
        fh.write(json.dumps(payload) + "\n")


if __name__ == "__main__":
    main()
