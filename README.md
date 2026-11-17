[![make.fun · gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/syahruldyh/gta6-twitch-ccu-forecast-01/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/syahruldyh/gta6-twitch-ccu-forecast-01/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)

# GTA VI Twitch Launch-Day Forecaster

Entry for [gta6-twitch-ccu-forecast-01](https://make.fun/bounty/gta6-twitch-ccu-forecast-01)
on [make.fun](https://make.fun): one hour after GTA VI's release day starts
(19 Nov 2026, 00:00 UTC), forecast the live Twitch viewers in the GTA VI
category for hours 2–24 (UTC), five quantiles each (5 / 25 / 50 / 75 / 95%).
Score is mean pinball loss on the log scale — lower is better.

## How it works

`main.py` (Python 3.13, standard library only, offline, deterministic) fits
its coefficients at run time from the comparable launch library shipped
inside each case file, under fixed hyperparameters (the blend/spread grids,
ridge penalties, kernel time constant, size-weighting width, and the
anchor-exponent prior and clamp). The same program runs the practice cases
and the final case.

1. **Unlock-aligned median curve.** The hour-1 print anchors the forecast;
   one shared unlock kernel (a spike that decays over ~4 hours, plus a
   sustained lift per unlocked region) shifts the curve. There is no
   explicit UTC audience-cycle feature: time-of-day effects enter only
   implicitly, through the comparables' hourly shapes that the median
   curve and the regression both use. Unlock times are parsed out of the
   case's research notes (`notes` / each comparable's `unlock` field).
   Every feature is normalised against hour 1, so unlocks already
   reflected in the anchor are never counted twice.
2. **Per-hour log-linear regression.** `log(viewers_h)` on log hour-1
   viewers, log hour-1 channels and log pre-release 7-day mean viewers,
   ridge-shrunk toward the median curve with a sublinear anchor exponent
   (beta clamped to 0.5–0.95, prior 0.7).
3. **Blend + regime mixture.** The blend weight between the regression and
   the unlock-adjusted shape is picked by 5-fold cross-validation over the
   comparables. The US unlock is genuinely ambiguous (one spike at hour 6
   vs two spikes at hours 6 and 9), so the forecast mixes both scenarios:
   the median is the mean of the two scenarios' log-medians (a geometric
   mean in viewer space) and the spread widens where they disagree
   (from hour 9 onward, once the second spike would or would not have
   happened).
4. **Honest spread.** Leave-one-launch-out residuals give pooled lower and
   upper log tails, and the multiplier (0.7 / 0.85 / 1.0 / 1.2) is picked
   by nested selection: launches are split into 5 folds, the tails are
   re-estimated from the other folds only, and the multiplier is scored by
   pinball loss on the held-out fold — so it is never picked on the
   residuals it is scored against, and the side is chosen by the
   quantile's sign, exactly as the shipped forecast does. No spread is
   tuned on the practice cases. One caveat: the blend weight was picked by
   cross-validation over the full library, so a held-out launch still
   influenced it — the LOO residuals are not fully independent of the
   launch they score.

## Run it

```sh
mkdir -p out
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Each `<id>.in` in the input directory produces one `<id>.out`:
`{"hours": [{"hour": h, "q": [q05, q25, q50, q75, q95]}, ...]}` for
hours 2–24. Runtime is about a second per case (max 0.96s for the shipped
blend on the practice set, limit 8s); two runs are byte-identical.

## Practice validation (local, SullyGnome Average viewers)

Pinball loss on the log scale, same convention as the house scorer
(baseline 0.3871 reproduced exactly):

| model | p00 Wolverine | p01 RDR2 | p02 GoW | p03 Spider-Man 2 | mean |
|---|---:|---:|---:|---:|---:|
| median shape | 0.429 | 0.202 | 0.212 | 0.225 | 0.267 |
| size-weighted shape | 0.329 | 0.235 | 0.202 | 0.226 | 0.248 |
| regression | 0.196 | 0.192 | 0.189 | 0.233 | 0.202 |
| shape + unlock kernel | 0.188 | 0.185 | 0.202 | 0.226 | 0.200 |
| **blend, LOO spread (this entry)** | **0.188** | **0.184** | **0.192** | **0.222** | **0.196** |

Board: #1 0.122, #20 (paid line) 0.144. The practice mean (0.196) is
still above the paid line — the final is a single case, so this is a
calculated bet, not a proven paid finish. These numbers were recomputed
after fixing the spread calibration (the tail is now picked by the
quantile's sign, and the multiplier by nested 5-fold selection); the
corrected selection still lands on multiplier 0.85 on all four cases, so
the corrected table coincides with the pre-fix one — a measured
coincidence, not a carried-over figure. For reference, the same blend
with the spread tuned on the practice set (0.7x) scores 0.172, but that
number is optimistic by construction and is not what ships.

## Files

- `main.py` — the whole forecaster, one file, no dependencies.
- `README.md` — this file.
