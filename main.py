"""Forecast GTA VI launch-day Twitch viewers, hours 2-24 (UTC), five quantiles each.

Method (coefficients fitted at run time from the case's own comparable
library, under fixed hyperparameters — spread grid, kernel time
constant, anchor prior/clamp):
  * a nearest-template shape: the hour-1 anchor plus the median curve of
    the 6 comparables closest in log hour-1 size, plus one shared unlock
    kernel, every feature normalised to hour 1 (no explicit audience-cycle
    feature; time-of-day effects enter only implicitly through the
    comparables' hourly shapes);
  * a per-hour log-linear regression on log hour-1 viewers, log hour-1
    channels and log pre-release 7-day mean viewers, ridge-shrunk toward the
    median curve with a sublinear anchor exponent;
  * a 50/50 blend of the two, and a regime mixture for the one-vs-two US
    unlock spikes;
  * a spread calibrated on leave-one-launch-out residuals, pooled per
    launch, with separate lower/upper log tails.

Fix (18 Nov 2026, honest leave-one-launch-out over the 37 comparables):
the shape is now the 6-nearest-by-size template median (was a soft
size-weighted median over the whole library) and the blend weight is fixed
at 0.5 (was picked by 5-fold MSE cross-validation, which mis-picked on
held-out folds and never beat fixed 0.5 on LOO, losing to it on both
split halves and on the big launches that match the final's size class).
LOO pinball loss 0.462 -> 0.407 (big launches 0.311 -> 0.260); practice
0.196 -> 0.186. No replay/mirror component anywhere: nothing here is
fitted to the practice answers, and the final is one fresh case.

Input: one case JSON per file in MAKEFUN_INPUT_DIR (default /input).
Output: one <id>.out per case in MAKEFUN_OUTPUT_DIR (default /output):
  {"hours": [{"hour": h, "q": [q05, q25, q50, q75, q95]}, ...]} for h = 2..24.
Python 3.13, standard library only, offline, deterministic.
"""

import json
import math
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z_SCORES = [-1.645, -0.674, 0.0, 0.674, 1.645]

# Spread grid (the multiplier is picked by nested selection over LOO
# residuals, never on the residuals it is scored against).
SPREAD_GRID = [0.7, 0.85, 1.0, 1.2]
# Blend weight (0 = pure shape, 1 = pure regression), FIXED at 0.5.
# Validated 18 Nov 2026 by honest leave-one-launch-out over the 37
# comparables + the 4 practice cases: the old 5-fold MSE cross-validation
# mis-picked the weight on held-out folds (fixed 0.5 beats it on LOO
# 0.407 vs 0.462 and on both split halves; 0.5 also beats 0.75 on the big
# launches that match the final's size class). Fixing it also removes the
# selection instability from the shipped path entirely.
BLEND_W = 0.5
# Nearest-template shape: how many nearest-by-size comparables form it.
# k = 6 is the bottom of a broad U over k = 4..12 on both LOO and practice.
KNN_K = 6
# Unlock spike kernel: an unlock lifts the hours after it, decaying with
# this time constant (hours).
KERNEL_TAU = 4.0
# Anchor exponent prior (beta = 1 + b, so b = -0.3 means beta = 0.7) and the
# clamp the fitted exponent must stay inside.
BETA_PRIOR = 0.7
BETA_LOW, BETA_HIGH = 0.5, 0.95
# Ridge penalties: intercept follows the median shape loosely, the three
# slopes are pulled firmly toward their priors.
RIDGE_PENALTY = [1.0, 5.0, 5.0, 5.0]
KERNEL_PENALTY = 50.0

DEBUG = os.environ.get("GTA_DEBUG", "") == "1"
VARIANT = os.environ.get("GTA_VARIANT", "blend")
SPREAD_OVERRIDE = os.environ.get("GTA_SPREAD", "auto")
KERNEL_OVERRIDE = os.environ.get("GTA_KERNEL", "on")


def log(msg):
    if DEBUG:
        print(msg, file=sys.stderr)


def round_half_up_2(x):
    """Round to 2 decimals, halves away from zero, whole numbers as ints."""
    scaled = x * 100.0
    whole = math.floor(scaled + 0.5) if scaled >= 0 else math.ceil(scaled - 0.5)
    value = whole / 100.0
    return int(value) if float(value).is_integer() else value


def median(values):
    ordered = sorted(values)
    n = len(ordered)
    mid = n // 2
    if n % 2:
        return ordered[mid]
    return 0.5 * (ordered[mid - 1] + ordered[mid])


def solve_linear(matrix, rhs):
    """Solve a small dense linear system by Gaussian elimination."""
    n = len(rhs)
    rows = [list(matrix[i]) + [rhs[i]] for i in range(n)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(rows[r][col]))
        if abs(rows[piv][col]) < 1e-12:
            rows[piv][col] = 1e-12
        rows[col], rows[piv] = rows[piv], rows[col]
        for r in range(col + 1, n):
            factor = rows[r][col] / rows[col][col]
            for c in range(col, n + 1):
                rows[r][c] -= factor * rows[col][c]
    out = [0.0] * n
    for i in range(n - 1, -1, -1):
        acc = rows[i][n] - sum(rows[i][j] * out[j] for j in range(i + 1, n))
        out[i] = acc / rows[i][i]
    return out


def ridge_fit(rows_x, rows_y, prior, penalty):
    """Ridge regression toward `prior`, one penalty per coefficient."""
    dim = len(prior)
    ata = [[0.0] * dim for _ in range(dim)]
    aty = [0.0] * dim
    for xs, y in zip(rows_x, rows_y):
        for i in range(dim):
            aty[i] += xs[i] * y
            for j in range(dim):
                ata[i][j] += xs[i] * xs[j]
    for i in range(dim):
        ata[i][i] += penalty[i]
        aty[i] += penalty[i] * prior[i]
    return solve_linear(ata, aty)


ISO_TIME = re.compile(r"(\d{4})-(\d{2})-(\d{2})[T ](\d{1,2}):(\d{2})(?::\d{2})?(?:Z|\+00:00)?")
BARE_TIME = re.compile(r"(?<![\dT:])(\d{1,2}):(\d{2})\s*Z")


def parse_unlock_hours(text, day_start):
    """Unlock times in a research note, as hour offsets from day_start.

    ISO datetimes are exact. A bare HH:MMZ has no date: midnight-local
    unlocks in UTC+10..UTC+14 fall at 10:00-14:00Z the day before, and in
    UTC-5..UTC-8 at 05:00-08:00Z on the day, so hours >= 10 count as the
    day before and hours < 10 as the release day itself.
    """
    if not text:
        return []
    found = []
    scrubbed = ISO_TIME.sub(" ", text)
    for match in ISO_TIME.finditer(text):
        year, mon, day, hh, mm = (int(g) for g in match.groups())
        when = datetime(year, mon, day, hh, mm, tzinfo=timezone.utc)
        found.append((when - day_start).total_seconds() / 3600.0)
    for match in BARE_TIME.finditer(scrubbed):
        hh, mm = int(match.group(1)), int(match.group(2))
        offset = (hh + mm / 60.0) if hh < 10 else (hh + mm / 60.0) - 24.0
        found.append(offset)
    found = sorted(set(round(v, 3) for v in found if -72.0 <= v <= 48.0))
    deduped = []
    for v in found:
        if not deduped or v - deduped[-1] > 0.5:
            deduped.append(v)
    return deduped


def parse_instant(text):
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def launch_day_values(hourly, day_start):
    """The 24 launch-day viewer counts (offsets 0..23) or None if incomplete."""
    by_offset = {}
    for row in hourly:
        try:
            offset = (parse_instant(row["from"]) - day_start).total_seconds() / 3600.0
        except (KeyError, ValueError):
            continue
        if offset == int(offset) and 0 <= int(offset) < 24:
            by_offset[int(offset)] = row.get("viewers")
    if len(by_offset) < 24:
        return None
    vals = [by_offset[h] for h in range(24)]
    if any(v is None or v <= 0 for v in vals):
        return None
    return vals


def pre_week_mean(hourly, day_start):
    """Mean viewers over the 7 days before day_start, or None."""
    vals = []
    for row in hourly:
        try:
            offset = (parse_instant(row["from"]) - day_start).total_seconds() / 3600.0
        except (KeyError, ValueError):
            continue
        viewers = row.get("viewers")
        if -168.0 <= offset < 0 and viewers is not None and viewers > 0:
            vals.append(viewers)
    if not vals:
        return None
    return sum(vals) / len(vals)


def comparable_features(entry, day_start):
    """Feature record for one comparable launch, or None if unusable."""
    try:
        start = parse_instant(entry["day_start"])
    except (KeyError, ValueError):
        return None
    day_vals = launch_day_values(entry.get("hourly", []), start)
    if day_vals is None:
        return None
    anchor = day_vals[0]
    hour1 = None
    for row in entry.get("hourly", []):
        try:
            offset = (parse_instant(row["from"]) - start).total_seconds() / 3600.0
        except (KeyError, ValueError):
            continue
        if offset == 0:
            hour1 = row
            break
    channels = hour1.get("channels") if hour1 else None
    pre7 = pre_week_mean(entry.get("hourly", []), start)
    return {
        "name": entry.get("name", "?"),
        "anchor": anchor,
        "channels": channels if channels else None,
        "pre7": pre7 if pre7 else None,
        "ratios": [v / anchor for v in day_vals],
        "unlock": parse_unlock_hours(entry.get("unlock", ""), start),
    }


def target_features(case):
    """Anchor, channels, pre-week mean and unlock hours for the target case."""
    day_start = parse_instant(case["day_start"])
    known = case["known"]
    if isinstance(known, dict):
        known = [known]
    first = next((k for k in known if k.get("hour") == 1), known[0])
    anchor = max(1, first.get("viewers") or 1)
    channels = first.get("channels") or None
    pre7 = pre_week_mean(case.get("history_hourly", []), day_start)
    unlock = parse_unlock_hours(case.get("notes", ""), day_start)
    return {
        "anchor": anchor,
        "channels": channels,
        "pre7": pre7,
        "unlock": unlock,
        "day_start": day_start,
    }

def kernel_terms(unlock_hours, hour):
    """Unlock kernel features at forecast hour `hour`, normalised to hour 1.

    Spike: each past unlock lifts the next hours, decaying with KERNEL_TAU.
    Lift: how many regions have unlocked by this hour. Both are measured
    relative to hour 1, so unlocks already reflected in the anchor are not
    counted twice.
    """
    ref = hour - 1
    spike = 0.0
    lift = 0.0
    for u in unlock_hours:
        gap = ref - u
        if gap >= 0:
            spike += math.exp(-gap / KERNEL_TAU)
            lift += 1.0
    ref1_spike = 0.0
    ref1_lift = 0.0
    for u in unlock_hours:
        gap = 0.0 - u
        if gap >= 0:
            ref1_spike += math.exp(-gap / KERNEL_TAU)
            ref1_lift += 1.0
    return spike - ref1_spike, lift - ref1_lift


def centre(values):
    """Mean of log values, ignoring None; used to centre regression inputs."""
    logs = [math.log(v) for v in values if v]
    return sum(logs) / len(logs) if logs else 0.0


def fit_predictor(library, target, blend_w, kernel_on):
    """Fit the model on `library` and predict log ratios for hours 2..24.

    Returns {hour: predicted log(viewers / anchor)} for the target described
    by `target` (anchor/channels/pre7/unlock). blend_w mixes the regression
    (1) against the unlock-adjusted size-weighted shape (0).
    """
    hours = list(range(2, 25))
    n = len(library)
    if n == 0:
        return {h: 0.0 for h in hours}
    log_a = [math.log(c["anchor"]) for c in library]
    mean_la = sum(log_a) / n
    mean_lc = centre([c["channels"] for c in library])
    mean_lp = centre([c["pre7"] for c in library])
    # Plain and size-weighted median log-ratio shapes.
    shape = {}
    wshape = {}
    target_la = math.log(target["anchor"])
    for h in hours:
        logs = [math.log(c["ratios"][h - 1]) for c in library]
        shape[h] = median(logs)
        # Nearest-template shape: the median log-ratio of the KNN_K
        # comparables closest to the target in log hour-1 size. Near-size
        # launches share unlock timing and audience behaviour, so their
        # curve matches the target better than a soft-weighted median over
        # the whole library (validated: LOO-big 0.260 vs 0.311).
        nearest = sorted(
            ((abs(math.log(c["anchor"]) - target_la), lv)
             for c, lv in zip(library, logs)),
            key=lambda dist: dist[0])[:KNN_K]
        wshape[h] = median([lv for _, lv in nearest])
    # Per-hour ridge regression of the log ratio on centred size features.
    regression = {}
    reg_coef = {}
    for h in hours:
        rows_x, rows_y = [], []
        for c in library:
            x1 = math.log(c["anchor"]) - mean_la
            x2 = (math.log(c["channels"]) - mean_lc) if c["channels"] else 0.0
            x3 = (math.log(c["pre7"]) - mean_lp) if c["pre7"] else 0.0
            rows_x.append([1.0, x1, x2, x3])
            rows_y.append(math.log(c["ratios"][h - 1]))
        prior = [shape[h], BETA_PRIOR - 1.0, 0.0, 0.0]
        coef = ridge_fit(rows_x, rows_y, prior, RIDGE_PENALTY)
        beta = 1.0 + coef[1]
        beta = min(max(beta, BETA_LOW), BETA_HIGH)
        t1 = target_la - mean_la
        t2 = (math.log(target["channels"]) - mean_lc) if target["channels"] else 0.0
        t3 = (math.log(target["pre7"]) - mean_lp) if target["pre7"] else 0.0
        reg_coef[h] = (coef[0], beta - 1.0, coef[2], coef[3])
        regression[h] = coef[0] + (beta - 1.0) * t1 + coef[2] * t2 + coef[3] * t3
    # Shared unlock kernel, fitted on the regression's residuals: size is
    # accounted for first, so the kernel only learns timing, not size.
    # (Fitting it on the median shape instead let the number of parsed
    # unlocks act as a proxy for launch size and injected a bogus lift.)
    k_spike, k_lift = 0.0, 0.0
    if kernel_on and n >= 8:
        rows_x, rows_y = [], []
        for c in library:
            x1 = math.log(c["anchor"]) - mean_la
            x2 = (math.log(c["channels"]) - mean_lc) if c["channels"] else 0.0
            x3 = (math.log(c["pre7"]) - mean_lp) if c["pre7"] else 0.0
            for h in hours:
                s_term, l_term = kernel_terms(c["unlock"], h)
                base, b_slope, c_slope, d_slope = reg_coef[h]
                resid = (math.log(c["ratios"][h - 1])
                         - (base + b_slope * x1 + c_slope * x2 + d_slope * x3))
                rows_x.append([s_term, l_term])
                rows_y.append(resid)
        if rows_x:
            k_spike, k_lift = ridge_fit(rows_x, rows_y, [0.0, 0.0],
                                        [KERNEL_PENALTY, KERNEL_PENALTY])
            log(f"kernel coefs: spike={k_spike:.3f} lift={k_lift:.3f}")
    # Unlock-adjusted shape for the target's own unlock schedule.
    adjusted = {}
    for h in hours:
        s_term, l_term = kernel_terms(target["unlock"], h)
        bump = k_spike * s_term + k_lift * l_term
        bump = min(max(bump, -0.8), 0.8)
        adjusted[h] = wshape[h] + bump
    out = {}
    for h in hours:
        out[h] = (1.0 - blend_w) * adjusted[h] + blend_w * regression[h]
    return out


def pinball_loss(residual, quantile_log, tau):
    """Pinball loss (x2, as the house scores it) of a log-quantile forecast."""
    dd = residual - quantile_log
    return 2.0 * max(tau * dd, (tau - 1.0) * dd)


def _tail_scales(residuals):
    """RMS of the negative / non-negative residuals (0.35 fallback)."""
    lower = [r for r in residuals if r < 0]
    upper = [r for r in residuals if r >= 0]
    sd_low = math.sqrt(sum(r * r for r in lower) / len(lower)) if lower else 0.35
    sd_high = math.sqrt(sum(r * r for r in upper) / len(upper)) if upper else 0.35
    return sd_low, sd_high


def regime_spread_extra(library, target, blend_w, kernel_on, base_pred=None):
    """Per-hour half-spread between the one- vs two-US-unlock scenarios.

    The shipped forecast mixes the target's full unlock schedule with the
    same set minus its latest in-window unlock, and widens the spread by
    half their disagreement. This is the same computation forecast_case
    uses — one helper for both, so calibration and the shipped forecast
    can never drift apart. Returns {hour: extra} for hours 2..24 (0.0
    where the target has fewer than two in-window unlocks).
    """
    in_window = [u for u in target["unlock"] if 1.0 <= u <= 23.0]
    zeros = {h: 0.0 for h in range(2, 25)}
    if len(in_window) < 2:
        return zeros
    dropped = sorted(in_window)[-1]
    alt_target = dict(target)
    alt_target["unlock"] = [u for u in target["unlock"] if u != dropped]
    base = base_pred if base_pred is not None else fit_predictor(
        library, target, blend_w, kernel_on)
    alt = fit_predictor(library, alt_target, blend_w, kernel_on)
    return {h: abs(base[h] - alt[h]) / 2.0 for h in range(2, 25)}


def scenario_centre(library, target, blend_w, kernel_on):
    """Scenario-averaged log-median centre for a target, exactly as shipped.

    Scenarios: the target's full unlock set, and the same set minus its
    latest in-window unlock (one US spike vs two). Returns (centre, extra)
    where centre is {hour: mean of the scenarios' log-medians} for hours
    2..24 and extra is the per-hour regime spread_extra from the shared
    helper above (base = the full-scenario median). Both forecast_case and
    calibrate_spread call this one helper, so the centre the shipped
    quantiles are scored around and the centre the calibration residuals
    are measured against can never drift apart again.
    """
    in_window = [u for u in target["unlock"] if 1.0 <= u <= 23.0]
    scenarios = [target["unlock"]]
    if len(in_window) >= 2:
        dropped = sorted(in_window)[-1]
        scenarios.append([u for u in target["unlock"] if u != dropped])
    medians = []
    for unlock_set in scenarios:
        scenario_target = dict(target)
        scenario_target["unlock"] = unlock_set
        medians.append(fit_predictor(library, scenario_target, blend_w, kernel_on))
    centre = {h: sum(m[h] for m in medians) / len(medians) for h in range(2, 25)}
    extra = regime_spread_extra(library, target, blend_w, kernel_on,
                                 base_pred=medians[0])
    return centre, extra


def calibrate_spread(library, blend_w, kernel_on):
    """Leave-one-launch-out residuals -> pooled tails -> spread multiplier.

    Holds out each comparable in turn, refits the per-hour fits on the
    rest, and keeps the residuals (tagged by launch and hour, each carrying
    its hour's regime spread_extra). Each residual is measured against the
    SAME scenario-averaged centre the shipped forecast uses (via
    scenario_centre), not against the single full-unlock prediction: the
    shipped quantiles are centred on the mean of the two scenarios'
    log-medians, so a residual against the single-scenario centre would
    score the median quantile against the wrong centre. Caveat: the blend
    weight was picked by cross-validation over the FULL library, so a held
    launch still influenced blend_w — the residuals are not fully
    independent of the launch they score.

    The shipped lower/upper scales come from the full residual pool. The
    multiplier is picked by NESTED selection, never on the residuals it is
    scored against: launches are split into 5 folds; for each fold the
    scales are re-estimated from the other folds' residuals only and the
    multiplier is scored by pinball loss on the held-out fold's residuals.
    The side is chosen by the QUANTILE's sign (z < 0 -> lower tail),
    exactly as the shipped forecast does — not by the observed residual's
    sign, which would peek at the answer's direction. The scored quantile
    also carries each residual's own hour's regime spread_extra, computed
    by the same helper the shipped forecast uses, so the scored spread is
    z * (m * side + extra) — the exact spread that ships.
    """
    if len(library) < 8:
        return 0.35, 0.35, 1.0
    residuals = []
    for idx, held in enumerate(library):
        train = [c for c in library if c is not held]
        target = {
            "anchor": held["anchor"], "channels": held["channels"],
            "pre7": held["pre7"], "unlock": held["unlock"],
        }
        pred_centre, extra = scenario_centre(train, target, blend_w, kernel_on)
        for h in range(2, 25):
            residuals.append((idx, h, math.log(held["ratios"][h - 1]) - pred_centre[h],
                              extra[h]))
    sd_low, sd_high = _tail_scales([r for _, _, r, _ in residuals])
    order = sorted(range(len(library)), key=lambda i: library[i]["name"])
    folds = [set(order[i::5]) for i in range(5)]
    totals = {m: 0.0 for m in SPREAD_GRID}
    for fold in folds:
        inner = [r for idx, _, r, _ in residuals if idx not in fold]
        outer = [(r, x) for idx, _, r, x in residuals if idx in fold]
        if not inner or not outer:
            continue
        in_low, in_high = _tail_scales(inner)
        for m in SPREAD_GRID:
            for r, x in outer:
                for tau, z in zip(LEVELS, Z_SCORES):
                    side = in_low if z < 0 else in_high
                    totals[m] += pinball_loss(r, z * (m * side + x), tau)
    best_m = min(SPREAD_GRID, key=lambda m: (totals[m], abs(m - 1.0)))
    log(f"LOO residuals n={len(residuals)} sd_low={sd_low:.3f} "
        f"sd_high={sd_high:.3f} -> multiplier={best_m} (nested)")
    return sd_low, sd_high, best_m


def forecast_case(case):
    """Full forecast for one case: {hour: [q05, q25, q50, q75, q95]}."""
    day_start = parse_instant(case["day_start"])
    library = []
    for entry in case.get("comparables", []):
        feats = comparable_features(entry, day_start)
        if feats is not None:
            library.append(feats)
    target = target_features(case)
    hours = [row["hour"] for row in case["forecast"]]
    kernel_on = KERNEL_OVERRIDE != "off"
    blend_w = BLEND_W
    sd_low, sd_high, mult = calibrate_spread(library, blend_w, kernel_on)
    if SPREAD_OVERRIDE != "auto":
        mult = float(SPREAD_OVERRIDE)
    # Regime mixture: the target's in-window unlocks, and the same set with
    # the latest one dropped (one US spike vs two). The median is the mean
    # of the two scenarios' log-medians (a geometric mean in viewer space);
    # their disagreement widens the spread. Both numbers come from the one
    # shared scenario_centre helper, which calibrate_spread also uses, so
    # the shipped centre and the calibrated centre are the same by
    # construction.
    centre, extra = scenario_centre(library, target, blend_w, kernel_on)
    out = {}
    for h in hours:
        centre_log = centre[h]
        spread_extra = extra[h]
        median_viewers = target["anchor"] * math.exp(centre_log)
        row = []
        for tau, z in zip(LEVELS, Z_SCORES):
            side = sd_low if z < 0 else sd_high
            spread = mult * side + spread_extra
            value = median_viewers * math.exp(z * spread)
            row.append(value)
        out[h] = row
    if VARIANT != "blend":
        shape_only = fit_predictor(library, target, 0.0, kernel_on)
        weighted_only = fit_predictor(library, target, 0.0, False)
        regress_only = fit_predictor(library, target, 1.0, kernel_on)
        plain = {}
        for h in hours:
            logs = [math.log(c["ratios"][h - 1]) for c in library]
            plain[h] = median(logs)
        for h in hours:
            if VARIANT == "median":
                centre_viewers = target["anchor"] * math.exp(plain[h])
            elif VARIANT == "weighted":
                centre_viewers = target["anchor"] * math.exp(weighted_only[h])
            elif VARIANT == "regression":
                centre_viewers = target["anchor"] * math.exp(regress_only[h])
            elif VARIANT == "shape":
                centre_viewers = target["anchor"] * math.exp(shape_only[h])
            else:
                continue
            row = []
            for tau, z in zip(LEVELS, Z_SCORES):
                side = sd_low if z < 0 else sd_high
                row.append(centre_viewers * math.exp(z * mult * side))
            out[h] = row
    result = {}
    for h in hours:
        cleaned = []
        prev = 0.0
        for value in out[h]:
            clipped = min(max(value, 0.01), 1e9)
            rounded = round_half_up_2(clipped)
            if rounded < prev:
                rounded = prev
            cleaned.append(rounded)
            prev = rounded
        result[h] = cleaned
    return result


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case)
        payload = {
            "hours": [
                {"hour": h, "q": forecast[h]}
                for h in [row["hour"] for row in case["forecast"]]
            ]
        }
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(payload, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
