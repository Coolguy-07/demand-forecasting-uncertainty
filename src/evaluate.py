"""Stage 5: evaluation - point accuracy, interval calibration, figures.

Point accuracy : MAE / RMSE (+ seasonal-naive baseline, MASE)
Intervals      : coverage of 50/80/90% PIs on valid & test, raw and
                 scale-calibrated (factor fitted on the validation month so the
                 interval hits its nominal coverage there).
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.common import (
    COLORS,
    load_predictions,
    quantile_col,
    save_fig,
    save_predictions,
    tables_dir,
    update_metrics,
)
from src.config import load_config


# ---------------------------------------------------------------------------
# metrics helpers
# ---------------------------------------------------------------------------
def pinball(y, q, alpha):
    d = y - q
    return np.maximum(alpha * d, (alpha - 1.0) * d)


def point_metrics(y, p):
    err = y - p
    return {
        "MAE": float(np.mean(np.abs(err))),
        "RMSE": float(np.sqrt(np.mean(err**2))),
    }


def find_scale_factor(lo, hi, mid, y, nominal):
    """Multiplicative widening of [lo, hi] around the median `mid` so that
    empirical coverage on the calibration sample equals `nominal`."""

    def coverage(k):
        l = np.maximum(mid - k * (mid - lo), 0.0)
        h = mid + k * (hi - mid)
        return float(np.mean((y >= l) & (y <= h)))

    lo_k, hi_k = 0.0, 1.0
    while coverage(hi_k) < nominal and hi_k < 64:
        hi_k *= 2
    if coverage(hi_k) < nominal:
        return np.nan
    for _ in range(60):
        mid_k = (lo_k + hi_k) / 2
        if coverage(mid_k) < nominal:
            lo_k = mid_k
        else:
            hi_k = mid_k
    return (lo_k + hi_k) / 2


def interval_stats(y, lo, hi):
    inside = (y >= lo) & (y <= hi)
    width = hi - lo
    return {
        "coverage": float(np.mean(inside)),
        "mean_width": float(np.mean(width)),
        "median_width": float(np.median(width)),
        "miss_hi_rate": float(np.mean(y > hi)),
        "miss_lo_rate": float(np.mean(y < lo)),
    }


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------
def fig_calibration(cfg, cov):
    nominal = [iv["nominal"] for iv in cfg["intervals"]]
    raw = [cov["test"][iv["name"]]["raw"]["coverage"] for iv in cfg["intervals"]]
    cal = [cov["test"][iv["name"]]["calibrated"]["coverage"] for iv in cfg["intervals"]]
    val = [cov["valid"][iv["name"]]["raw"]["coverage"] for iv in cfg["intervals"]]

    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    ax.plot([0, 1], [0, 1], ls="--", c="grey", lw=1, label="perfect calibration")
    ax.plot(nominal, val, "o--", c=COLORS["naive"], alpha=0.7, label="raw (valid)")
    ax.plot(nominal, raw, "o-", c=COLORS["point"], label="raw (test)")
    ax.plot(nominal, cal, "s-", c=COLORS["cal"], label="scale-calibrated (test)")
    for x, yv, yc in zip(nominal, raw, cal):
        ax.annotate(f"{yv:.2f}", (x, yv), textcoords="offset points", xytext=(4, -12),
                    color=COLORS["point"], fontsize=9)
        ax.annotate(f"{yc:.2f}", (x, yc), textcoords="offset points", xytext=(4, 6),
                    color=COLORS["cal"], fontsize=9)
    ax.set_xlabel("Nominal coverage")
    ax.set_ylabel("Empirical coverage (test)")
    ax.set_xlim(0.4, 1.0)
    ax.set_ylim(0.4, 1.0)
    ax.set_title("Interval calibration: predicted vs. actual coverage")
    ax.legend(loc="upper left")
    save_fig(cfg, "fig_calibration.png")


def fig_interval_samples(cfg, preds):
    test = preds[preds["split"] == "test"].copy()
    means = test.groupby("Store", observed=True)["y"].mean()
    means = means.sort_values()
    picks = [int(means.index[0]), int(means.index[len(means) // 2]), int(means.index[-1])]
    labels = ["low-volume store", "median store", "high-volume store"]

    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8), sharex=True)
    for ax, store, label in zip(axes, picks, labels):
        d = test[test["Store"] == store].sort_values("Date")
        y = d["y"].to_numpy()
        lo, hi = d["cal_lo_80"].to_numpy(), d["cal_hi_80"].to_numpy()
        cov_raw = float(np.mean((y >= d["q_0.10"]) & (y <= d["q_0.90"])))
        cov_cal = float(np.mean((y >= lo) & (y <= hi)))
        for _, row in d[d["Promo"] == 1].iterrows():
            ax.axvspan(row["Date"], row["Date"] + pd.Timedelta(days=1),
                       color="#ffda7f", alpha=0.55, lw=0)
        ax.fill_between(d["Date"], lo, hi, color=COLORS["band"], alpha=0.25,
                        label="80% PI (calibrated)")
        ax.plot(d["Date"], d["q_0.50"], color=COLORS["point"], lw=1.3, label="median")
        ax.plot(d["Date"], y, color=COLORS["actual"], lw=1.1, marker=".",
                ms=2.5, label="actual")
        ax.set_title(
            f"Store {store} ({label})\ncoverage raw {cov_raw:.0%} -> cal {cov_cal:.0%}"
        )
        ax.tick_params(axis="x", rotation=30)
    axes[0].set_ylabel("daily sales")
    axes[0].legend(loc="upper left", fontsize=8, ncol=3)
    fig.suptitle("Prediction intervals over the test period (shaded = promo days)",
                 y=1.04, fontweight="bold")
    save_fig(cfg, "fig_interval_samples.png")


def fig_aggregate_band(cfg, preds):
    test = preds[preds["split"] == "test"]
    agg = test.groupby("Date").agg(
        y=("y", "sum"),
        q50=(quantile_col(0.50), "sum"),
        lo=("cal_lo_80", "sum"),
        hi=("cal_hi_80", "sum"),
        promo=("Promo", "max"),
    ).sort_index()

    fig, ax = plt.subplots(figsize=(11, 4.4))
    for _, row in agg[agg["promo"] == 1].iterrows():
        ax.axvspan(row.name, row.name + pd.Timedelta(days=1),
                   color="#ffda7f", alpha=0.55, lw=0)
    ax.fill_between(agg.index, agg["lo"], agg["hi"], color=COLORS["band"],
                    alpha=0.25, label="80% PI (sum of per-store bounds)")
    ax.plot(agg.index, agg["q50"], color=COLORS["point"], lw=1.4, label="median forecast")
    ax.plot(agg.index, agg["y"], color=COLORS["actual"], lw=1.4, label="actual total")
    ax.set_ylabel("total daily sales (all stores)")
    ax.set_title("Chain-wide demand: actual vs. forecast with 80% interval")
    ax.legend(loc="upper right", ncol=3, fontsize=8)
    ax.tick_params(axis="x", rotation=30)
    save_fig(cfg, "fig_aggregate_band.png")


# ---------------------------------------------------------------------------
def evaluate(cfg=None):
    cfg = cfg or load_config()
    preds = load_predictions(cfg)
    q50 = quantile_col(0.50)

    # ---- point accuracy ----------------------------------------------------
    feats = pd.read_parquet(cfg["paths"]["cache_dir"] / "features.parquet")
    tr = feats[feats["split"] == "train"]
    naive_denom = float(np.mean(np.abs(tr["Sales"] - tr["lag_7"])))

    point = {}
    for sp in ["valid", "test"]:
        d = preds[preds["split"] == sp]
        y = d["y"].to_numpy()
        point[sp] = {
            "q50_median": point_metrics(y, d[q50].to_numpy()),
            "seasonal_naive_lag7": point_metrics(y, d["naive"].to_numpy()),
            "n_rows": int(len(d)),
        }
    point["test"]["MASE_q50_vs_naive"] = (
        point["test"]["q50_median"]["MAE"] / naive_denom
    )

    # ---- pinball loss per quantile ----------------------------------------
    pinball_res = {}
    for sp in ["valid", "test"]:
        d = preds[preds["split"] == sp]
        y = d["y"].to_numpy()
        pinball_res[sp] = {
            f"{q:.2f}": float(np.mean(pinball(y, d[quantile_col(q)].to_numpy(), q)))
            for q in cfg["quantiles"]
        }

    # ---- interval calibration ---------------------------------------------
    scales, coverage = {}, {"valid": {}, "test": {}}
    yv = preds.loc[preds["split"] == "valid", "y"].to_numpy()

    for iv in cfg["intervals"]:
        lo_c, hi_c, mid_c = quantile_col(iv["lo"]), quantile_col(iv["hi"]), q50
        name, nominal = iv["name"], iv["nominal"]

        va = preds[preds["split"] == "valid"]
        k = find_scale_factor(
            va[lo_c].to_numpy(), va[hi_c].to_numpy(), va[mid_c].to_numpy(), yv, nominal
        )
        scales[name] = None if np.isnan(k) else float(k)

        # calibrated bounds for every row
        preds[f"cal_lo_{name}"] = np.maximum(
            preds[mid_c] - k * (preds[mid_c] - preds[lo_c]), 0.0
        )
        preds[f"cal_hi_{name}"] = preds[mid_c] + k * (preds[hi_c] - preds[mid_c])

        for sp in ["valid", "test"]:
            d = preds[preds["split"] == sp]
            y = d["y"].to_numpy()
            coverage[sp][name] = {
                "nominal": nominal,
                "scale_k": scales[name],
                "raw": interval_stats(y, d[lo_c].to_numpy(), d[hi_c].to_numpy()),
                "calibrated": interval_stats(
                    y,
                    d[f"cal_lo_{name}"].to_numpy(),
                    d[f"cal_hi_{name}"].to_numpy(),
                ),
            }
        c_raw = coverage["test"][name]["raw"]["coverage"]
        c_cal = coverage["test"][name]["calibrated"]["coverage"]
        k_txt = "n/a" if scales[name] is None else f"{scales[name]:.3f}"
        print(
            f"[evaluate] {int(nominal * 100)}% interval: test coverage "
            f"{c_raw:.3f} raw -> {c_cal:.3f} calibrated (k={k_txt})"
        )

    # ---- persist -----------------------------------------------------------
    save_predictions(cfg, preds)

    tab = tables_dir(cfg)
    pd.DataFrame(
        [
            {
                "split": sp,
                "interval": name,
                "nominal": v["nominal"],
                "scale_k": v["scale_k"],
                "coverage_raw": v["raw"]["coverage"],
                "coverage_calibrated": v["calibrated"]["coverage"],
                "mean_width_raw": v["raw"]["mean_width"],
                "mean_width_calibrated": v["calibrated"]["mean_width"],
                "miss_hi_raw": v["raw"]["miss_hi_rate"],
                "miss_lo_raw": v["raw"]["miss_lo_rate"],
            }
            for sp in coverage for name, v in coverage[sp].items()
        ]
    ).to_csv(tab / "coverage.csv", index=False)

    pd.DataFrame(
        [
            {"split": sp, "model": name, **vals}
            for sp, models in point.items()
            for name, vals in models.items()
            if isinstance(vals, dict)
        ]
    ).to_csv(tab / "point_metrics.csv", index=False)

    pd.DataFrame(
        [{"split": sp, "quantile": q, "pinball": v}
         for sp, qs in pinball_res.items() for q, v in qs.items()]
    ).to_csv(tab / "pinball.csv", index=False)

    update_metrics(
        cfg,
        "evaluation",
        {
            "point": point,
            "pinball": pinball_res,
            "coverage": coverage,
            "calibration_scale_factors": scales,
            "calibration_method": (
                "multiplicative widening around the median; factor fitted on the "
                "validation month so the interval attains nominal coverage there"
            ),
        },
    )

    # ---- figures -----------------------------------------------------------
    fig_calibration(cfg, coverage)
    fig_interval_samples(cfg, preds)
    fig_aggregate_band(cfg, preds)
    return preds


def main():
    evaluate()
