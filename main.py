"""gta6-trophy-rarity-forecast-01 entry: damped logit-trend forecast (Python 3.13, standard library only).

For each trophy, fit logit(earn rate) against time over the case's read history
with recency weights, shrink each slope a little toward the cross-trophy median
slope, then extrapolate to the target with the slope damped over the horizon
(a trend that ran for two days rarely runs unchanged for four more). The
quantile spread combines the fit's residual scale, a per-day random-walk term,
and a trend-break term proportional to the extrapolated move. The spread was
recalibrated wider after a live check against Exophase's current rates showed
launch-week dilution moves rates by several logit units in a day — far beyond
the first calibration's bands.

Deterministic: no randomness, no wall clock, files processed in name order.

Run locally:
    mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
"""

import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path

# The grader runs with -P, which keeps /bot off sys.path: put this file's directory back first.
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]

# Model constants.
SHRINK_TO_MEDIAN = 0.25   # weight of the cross-trophy median slope in each trophy's slope
DAMP_HALF_LIFE_H = 72.0   # hours; the trend's pull decays as horizon / (horizon + this)
MAX_LOGIT_MOVE = 2.5      # cap on the extrapolated logit move
# Spread constants, recalibrated 2026-11-20 against live Exophase reads: launch-week
# rates move far more than the first calibration allowed (tracked-player growth dilutes
# hard-trophy rates by several logit units in a day), and pinball loss punishes a narrow
# miss much harder than a wide hit. Medians are unchanged; only the bands widen.
WALK_SIGMA_PER_SQRT_DAY = 0.60
TREND_BREAK_FRAC = 0.45   # extra spread as a fraction of the extrapolated move
SIGMA_FLOOR = 0.20
SIGMA_CAP = 2.20
FALLBACK_RESID = 0.25     # residual scale assumed when a trophy has too few reads to fit


def clamp_pct(x: float) -> float:
    return min(99.99, max(0.01, x))


def logit(pct: float) -> float:
    p = clamp_pct(pct) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit_pct(x: float) -> float:
    return 100.0 / (1.0 + math.exp(-x))


def round4(x: float) -> float | int:
    """x rounded to 4 decimals, half up (matching the starter's convention)."""
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if value.is_integer() else value


def parse_ts(s: str) -> float:
    return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()


def fit_trend(times_h: list[float], logits: list[float]) -> tuple[float, float, float]:
    """Recency-weighted linear fit. Returns (slope per hour, fitted value at last time, residual std)."""
    n = len(times_h)
    if n < 3:
        return 0.0, logits[-1], FALLBACK_RESID
    t0, t1 = times_h[0], times_h[-1]
    span = max(t1 - t0, 1e-9)
    w = [0.5 + 0.5 * (t - t0) / span for t in times_h]
    sw = sum(w)
    mt = sum(wi * ti for wi, ti in zip(w, times_h)) / sw
    my = sum(wi * yi for wi, yi in zip(w, logits)) / sw
    sxx = sum(wi * (ti - mt) ** 2 for wi, ti in zip(w, times_h))
    sxy = sum(wi * (ti - mt) * (yi - my) for wi, ti, yi in zip(w, times_h, logits))
    slope = sxy / sxx if sxx > 1e-12 else 0.0
    fitted_last = my + slope * (t1 - mt)
    resid = math.sqrt(
        sum(wi * (yi - (my + slope * (ti - mt))) ** 2 for wi, ti, yi in zip(w, times_h, logits)) / sw
    )
    return slope, fitted_last, resid


def median(values: list[float]) -> float:
    s = sorted(values)
    m = len(s)
    if m == 0:
        return 0.0
    return s[m // 2] if m % 2 else 0.5 * (s[m // 2 - 1] + s[m // 2])


def forecast_case(case: dict, name: str) -> dict:
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    history = case["history"]
    if not history:
        raise ValueError(f"{name} has no history")
    horizon_h = case["horizon_minutes"] / 60.0
    t_first = parse_ts(history[0]["at"])
    read_times_h = [(parse_ts(r["at"]) - t_first) / 3600.0 for r in history]

    # First pass: fit every trophy's trend.
    fits = []
    for i in range(len(case["trophies"])):
        ts, ys = [], []
        for t_h, read in zip(read_times_h, history):
            rate = read["rates"][i]
            if rate is not None:
                ts.append(t_h)
                ys.append(logit(rate))
        if not ys:
            fits.append((0.0, logit(0.0), FALLBACK_RESID, 0.0))
        else:
            slope, fitted_last, resid = fit_trend(ts, ys)
            # Anchor halfway between the fitted line's end and the last observation:
            # the line smooths quantization noise, the observation keeps us honest.
            anchor = 0.5 * fitted_last + 0.5 * ys[-1]
            fits.append((slope, anchor, resid, ys[-1]))

    median_slope = median([f[0] for f in fits])
    damp = DAMP_HALF_LIFE_H / (DAMP_HALF_LIFE_H + horizon_h)
    days = horizon_h / 24.0

    rows = []
    for trophy, (slope, anchor, resid, _last) in zip(case["trophies"], fits):
        slope = (1.0 - SHRINK_TO_MEDIAN) * slope + SHRINK_TO_MEDIAN * median_slope
        move = slope * horizon_h * damp
        move = max(-MAX_LOGIT_MOVE, min(MAX_LOGIT_MOVE, move))
        center = anchor + move
        sigma = math.sqrt(resid**2 + (WALK_SIGMA_PER_SQRT_DAY**2) * days + (TREND_BREAK_FRAC * abs(move)) ** 2)
        sigma = max(SIGMA_FLOOR, min(SIGMA_CAP, sigma))
        q = [round4(inv_logit_pct(center + z * sigma)) for z in Z]
        # Quantiles must be non-decreasing; rounding can only break that at the extremes.
        for k in range(1, len(q)):
            if q[k] < q[k - 1]:
                q[k] = q[k - 1]
        rows.append({"id": trophy["id"], "q": q})
    return {"trophies": rows}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
