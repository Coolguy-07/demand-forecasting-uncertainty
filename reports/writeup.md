# Write-up — Demand Forecasting with Uncertainty Quantification

*Round 2 · AIML-04 · Kotti Yaswanth (25BCE2784)*

## 1. Problem & data

A retailer needs demand forecasts whose **confidence** can be trusted, because
overstocking and stockouts cost differently. The deliverable is therefore not
a model score but a decision: how much safety stock the forecast implies, and
what that costs versus ordering on the point forecast alone.

**Data.** The assignment suggested M5 (Walmart) or Store Item Demand from
Kaggle; both require a Kaggle login, which would block reviewers from
reproducing the run. I used the **Rossmann Store Sales** competition data
(public mirror, anonymous download): 1.02 M daily sales rows × 1,115 German
pharmacy stores, 2013-01 → 2015-07, with `Promo`/`Promo2` promotion flags and
`StateHoliday`/`SchoolHoliday` calendars. This keeps the scenario intact —
multi-series retail demand with promotions and holidays to forecast around.
Splits are strictly temporal: train ≤ 2015-04-30, validation = May 2015, test
= June–July 2015 (58,611 open store-days).

## 2. Approach

**Features (leak-free).** Every target-derived feature is lagged ≥ 1 day: 13
lags (1 → 364 days, incl. 52-week seasonality), rolling mean/std over 7/14/28
days, momentum, mean of the previous four same-weekday sales, promo streaks,
schedule-derived Promo2 activity, weekday/month/week-of-year calendar terms,
store metadata and lagged cross-store sales. The raw files miss 33,121
store-days, so each store is first reindexed to a complete daily grid —
otherwise "lag 7" silently skips gaps and the weekly seasonality signal
corrupts. ~47 features in total; closed days are excluded from modelling but
still provide lag context.

**Model.** LightGBM quantile regression (pinball loss), one model per level
{0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95}, early-stopped on the validation
month (47 features × 760k training rows; all seven models fit in ~8 minutes on
CPU). The median model is the point forecast; a seasonal-naive (lag-7)
baseline anchors the accuracy numbers. Non-monotonic quantile crossings
(32.6k of 84.5k rows) are sorted into order and counted.

**Calibration.** Raw q10–q90 coverage rarely equals 80% exactly. I fit a
single **scale factor per interval level** on the validation month — widen the
interval multiplicatively around the median until validation coverage hits the
nominal level — then freeze it and apply it to test. Calibration never sees
test data; raw and calibrated results are always reported side by side.

**Decision.** Daily newsvendor per store: order `Q`, leftover wasted at
`Co = 1`, lost demand charged at `Cu = ratio·Co`. A calibrated forecast says:
order at the `ratio/(1+ratio)` quantile. I compare ordering on the median
(point) vs. the upper bound of the 80%/90% interval across cost ratios, with a
hindsight oracle as reference.

## 3. Results

**Point accuracy (test = June–July 2015, 58,611 open store-days).** The
median quantile model reaches **MAE 543.9 / RMSE 794.2**, versus 2,391.6 /
3,208.4 for a seasonal-naive (same weekday last week) baseline —
MASE 0.246.

**Interval calibration.** Raw quantile intervals are systematically too
narrow — a common and important finding:

| Interval | Nominal | Raw coverage (test) | Calibrated (test) |
|---|---:|---:|---:|
| q25–q75 | 50% | 40.8% | 54.0% |
| q10–q90 | 80% | 72.3% | **81.6%** |
| q05–q95 | 90% | 84.3% | **89.9%** |

The single scale factor per level, fitted on May and frozen, brings the 80%
and 90% intervals within ~1.5 pp of nominal on the untouched test month (the
50% interval overshoots to 54% — the test month drifted differently than the
calibration month; reported honestly). A useful diagnostic: the raw 80%
interval's failure is asymmetric — 17.5% of actuals fall **below** q10 while
only 10.2% exceed q90, i.e. the lower tail is the main miscalibration.

**Decision.** With a stockout costing 9× an overstock unit (Cu/Co = 9, the
newsvendor-optimal order quantile is the 90th percentile):

| Policy | Total cost | vs. point | Fill rate | Stockout days |
|---|---:|---:|---:|---:|
| Seasonal naive | 779.5M | −430% | 81.0% | 53.5% |
| Point forecast (median) | 147.2M | — | 96.6% | 44.2% |
| **Upper 80% PI (q90)** | **85.8M** | **−41.7%** | 99.3% | 10.2% |
| Upper 80% PI, calibrated | 90.5M | −38.5% | 99.5% | 6.9% |
| Upper 90% PI (q95) | 92.8M | −37.0% | 99.5% | 5.8% |

Ordering on the interval's upper bound cuts cost ~42% while dropping
store-days with any stockout from 44% to 10%. Note the unit-fill-rate nuance:
the median policy already serves 96.6% of *units* but still stockouts on
44% of *days* — exactly the risk a point forecast hides. The calibrated upper
bound costs ~5% more than raw q90 at this ratio (it overshoots the true 90th
percentile, which raw q90 already tracked) but buys a guaranteed coverage
level and fewer stockout days (6.9% vs 10.2%). Figures:
`fig_cost_tradeoff.png`, `fig_calibration.png`, `fig_interval_samples.png`.

## 4. Where the model is overconfident

Coverage is cut across segments (raw 80% interval, test):

| Segment | Coverage |
|---|---:|
| overall | 72.3% |
| demand spikes (actual ≥ store's train-period p95) | **67.3%** |
| normal demand | 72.6% |
| promo days | 71.6% |
| no-promo days | 72.8% |
| school holidays | 74.3% |

Findings:

1. **Demand spikes are the clearest failure** (67.3% vs 72.6%) — the band
   does not fully stretch when demand jumps to record levels, by construction
   of the quantile loss on historically rare events.
2. **Promotions hurt less than expected — and that's informative.** The model
   *does* widen on promo days (mean width ×1.40), so promo coverage (71.6%)
   is only 1.2 pp below no-promo (72.8%). High-side misses are essentially
   promo-neutral (47.0% occur on promos vs. a 46.3% promo base rate) —
   contrary to the "overconfident around promotions" hypothesis, this model's
   promotion weakness shows up in the *worst weeks* instead: the worst test
   week (2015-06-29, 83% promo share) covers only 64.7% raw / 74.0%
   calibrated.
3. **The failure is store-skewed.** Per-store raw coverage spans p10 = 63.5%
   to p90 = 81.1%; **428 of 1,115 stores sit below 70%**, worst = store 722
   (45.3%). A single global scale factor cannot repair store-specific
   miscalibration — the strongest argument for per-store or hierarchical
   calibration in production.
4. Width and difficulty don't fully co-move: worst-store MAE (e.g. store 909,
   MAE 3,778) identifies stores where both point accuracy and interval
   coverage collapse — model staleness or regime change, not just noise.

Figures: `fig_coverage_segments.png`, `fig_weekly_coverage.png`,
`fig_store_coverage_hist.png`; tables: `outputs/tables/segment_coverage.csv`,
`store_coverage.csv`, `weekly_coverage.csv`, `worst_interval_misses.csv`.

## 5. Bonus: quantile vs. residual-based intervals

Both calibrated on the same May validation month, judged on June–July:

| Method | Coverage valid → test | Width | Pinball (q10/q90) | Cost @ Cu/Co=9 |
|---|---:|---:|---:|---:|
| Quantile GBM, raw | 70.8% → 72.3% | 1,564 | **135.4** | 85.8M |
| Quantile GBM + scale calibration | 80.0% → **81.6%** | 2,009 | 137.6 | 90.5M |
| Residual – global | 80.0% → 79.2% | 1,714 | 140.8 | 88.1M |
| Residual – per store (absolute) | 74.2% → 73.7% | 1,599 | 141.0 | 85.8M |
| Residual – per store (relative) | 74.2% → 73.3% | 1,647 | 138.0 | 85.8M |

Does calibration improve things? **Yes for coverage**: the calibrated quantile
interval reaches 81.6% test coverage against 72.3% raw, and the residual
method confirms the same design point from the other direction (0.800 valid →
0.792 test). But three nuances make the comparison worth more than a
checkmark:

* **Sharpness vs. coverage trade-off is real** — pinball loss is lowest for
  the *raw* quantile models (135.4); widening to hit nominal coverage costs a
  little pinball (137.6). Interval coverage and quantile loss measure
  different things, and improving one slightly degrades the other.
* **Global residual intervals are a strong baseline** — almost no drift
  between months, zero extra training. Their weakness is structural: a
  constant-width band that cannot widen on promotion days or narrow on quiet
  days, so it over-covers easy days and under-covers hard ones without
  anyone noticing.
* **Per-store residual quantiles fail** — 26 validation points cannot
  estimate 10th/90th percentiles per store (coverage drops to ~73%), a
  textbook small-sample problem. Global or hierarchical pooling is required.

At Cu/Co = 9 the calibrated upper bound costs slightly more (90.5M vs 85.8M)
because raw q90 already sat near the true 90th percentile — the interval's
coverage failure was dominated by its *lower* bound (§3). Cost at one ratio
is therefore a poor sole judge of interval quality; coverage, width and
pinball together tell the full story.

Figure: `fig_bonus_comparison.png`;
table: `outputs/tables/bonus_comparison.csv`.


## 6. Key decisions

1. **Dataset choice** — reproducibility without credentials over dataset
   prestige; Rossmann preserves promotions/holidays/spikes.
2. **One-step-ahead framing** — matches daily ordering decisions; a single
   model with lagged features gives honest, leak-free test predictions.
3. **Validation month does three jobs** — early stopping, scale-factor
   calibration, and residual-interval fitting; test stays untouched until the
   final report.
4. **Interpretability over marginal accuracy** — no per-store models or
   hyperparameter sweeps; the scientific question is interval behaviour, not
   leaderboard rank.

## 7. Limitations & next steps

- Daily independent newsvendor ignores carry-over of leftovers; a
  rolling-horizon simulation would be more realistic.
- One global scale factor cannot fix segment-level undercoverage; per-store or
  promo/non-promo factors target exactly the failures found in §4.
- Promotion-aware quantile training (interaction features or asymmetric
  penalties on promo days) is the natural model-side fix.
- Multi-step horizons would show how width degrades with lead time.

---
*All figures: `reports/figures/` · all numbers: `outputs/metrics.json` ·
reproduce: `python run_all.py`*
