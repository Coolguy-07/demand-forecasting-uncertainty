"""Stage 7: overconfidence analysis - where are the intervals too narrow?

Cuts the test period by promo days, school holidays, demand spikes
(actual >= store's train-period p95), weekday, week and store, and reports
where empirical coverage falls below the nominal 80%.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.common import COLORS, load_predictions, save_fig, tables_dir, update_metrics
from src.config import load_config


def _row(seg, label, d):
    y = d["y"].to_numpy()
    raw = (y >= d["q_0.10"].to_numpy()) & (y <= d["q_0.90"].to_numpy())
    cal = (y >= d["cal_lo_80"].to_numpy()) & (y <= d["cal_hi_80"].to_numpy())
    return {
        "segment": seg,
        "value": label,
        "n": int(len(d)),
        "coverage_raw": float(np.mean(raw)),
        "coverage_cal": float(np.mean(cal)),
        "mean_width_raw": float(np.mean(d["q_0.90"] - d["q_0.10"])),
        "mean_width_cal": float(np.mean(d["cal_hi_80"] - d["cal_lo_80"])),
        "miss_hi_rate": float(np.mean(y > d["cal_hi_80"].to_numpy())),
        "miss_lo_rate": float(np.mean(y < d["cal_lo_80"].to_numpy())),
        "promo_share": float(d["Promo"].mean()),
    }


def segment_table(test):
    rows = []
    rows.append(_row("overall", "all", test))
    rows.append(_row("promo", "no promo", test[test["Promo"] == 0]))
    rows.append(_row("promo", "promo day", test[test["Promo"] == 1]))
    rows.append(_row("school_holiday", "no school holiday",
                     test[test["SchoolHoliday"] == 0]))
    rows.append(_row("school_holiday", "school holiday",
                     test[test["SchoolHoliday"] == 1]))
    rows.append(_row("demand_spike", "normal demand", test[test["is_spike"] == 0]))
    rows.append(_row("demand_spike", "spike (>= train p95)",
                     test[test["is_spike"] == 1]))
    for dow, grp in test.groupby("dow", observed=True):
        rows.append(_row("weekday", str(dow), grp))
    for month, grp in test.groupby(test["Date"].dt.to_period("M")):
        rows.append(_row("month", str(month), grp))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------
def fig_segments(cfg, seg):
    keep = ["overall", "promo", "school_holiday", "demand_spike"]
    d = seg[seg["segment"].isin(keep)]
    x = np.arange(len(d))
    fig, ax = plt.subplots(figsize=(10, 4.6))
    ax.bar(x - 0.2, d["coverage_raw"], 0.4, label="raw q10-q90", color=COLORS["point"])
    ax.bar(x + 0.2, d["coverage_cal"], 0.4, label="scale-calibrated",
           color=COLORS["cal"])
    ax.axhline(0.8, color="black", ls="--", lw=1.2, label="nominal 80%")
    for xi, (r, c, n) in enumerate(zip(d["coverage_raw"], d["coverage_cal"], d["n"])):
        ax.annotate(f"{r:.0%}", (xi - 0.2, r), ha="center", va="bottom",
                    fontsize=8, color=COLORS["point"])
        ax.annotate(f"{c:.0%}", (xi + 0.2, c), ha="center", va="bottom",
                    fontsize=8, color=COLORS["cal"])
        ax.annotate(f"n={n:,}", (xi, 0.02), ha="center", fontsize=8, color="white")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{a}\n{b}" for a, b in zip(d["segment"], d["value"])],
                       fontsize=8)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Empirical coverage of 80% interval (test)")
    ax.set_title("Where the interval is too narrow")
    ax.legend(loc="upper right", fontsize=8)
    save_fig(cfg, "fig_coverage_segments.png")


def fig_store_coverage(cfg, store_cov):
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    bins = np.linspace(0.4, 1.0, 25)
    ax.hist(store_cov["coverage_raw"], bins=bins, alpha=0.75,
            label="raw q10-q90", color=COLORS["point"])
    ax.hist(store_cov["coverage_cal"], bins=bins, alpha=0.6,
            label="scale-calibrated", color=COLORS["cal"])
    ax.axvline(0.8, color="black", ls="--", lw=1.3, label="nominal 80%")
    ax.set_xlabel("Per-store coverage of the 80% interval")
    ax.set_ylabel("Number of stores")
    ax.set_title("Per-store coverage: many stores are under-covered")
    ax.legend(fontsize=8)
    save_fig(cfg, "fig_store_coverage_hist.png")


def fig_weekly(cfg, weekly):
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(10, 6), sharex=True, gridspec_kw={"height_ratios": [2, 1]}
    )
    x = np.arange(len(weekly))
    ax1.plot(x, weekly["coverage_raw"], "o-", color=COLORS["point"],
             label="raw q10-q90")
    ax1.plot(x, weekly["coverage_cal"], "s-", color=COLORS["cal"],
             label="scale-calibrated")
    ax1.axhline(0.8, color="black", ls="--", lw=1.2, label="nominal 80%")
    ax1.set_ylim(0, 1.02)
    ax1.set_ylabel("Weekly coverage")
    ax1.set_title("Coverage by week: collapse during high-miss weeks")
    ax1.legend(fontsize=8, ncol=3, loc="lower left")
    ax2.bar(x, weekly["promo_share"], color="#ffda7f", edgecolor="grey", lw=0.5,
            label="promo share of week")
    ax2.set_ylabel("promo share")
    ax2.set_ylim(0, 1)
    ax2.set_xticks(x)
    ax2.set_xticklabels(weekly["week_label"], rotation=45, ha="right", fontsize=8)
    ax2.legend(fontsize=8, loc="upper left")
    save_fig(cfg, "fig_weekly_coverage.png")


# ---------------------------------------------------------------------------
def main(cfg=None):
    cfg = cfg or load_config()
    preds = load_predictions(cfg)
    feats = pd.read_parquet(cfg["paths"]["cache_dir"] / "features.parquet")

    # ---- demand-spike threshold per store (from the TRAIN period only) -----
    q = cfg["analysis"]["spike_train_quantile"]
    thr_train = (
        feats[feats["split"] == "train"]
        .groupby("Store", observed=True)["Sales"]
        .quantile(q)
    )
    thr = {int(k): float(v) for k, v in thr_train.items()}

    test = preds[preds["split"] == "test"].copy()
    test["is_spike"] = [y >= thr[int(s)]
                        for y, s in zip(test["y"], test["Store"])]

    # ---- segment coverage --------------------------------------------------
    seg = segment_table(test)
    tab = tables_dir(cfg)
    seg.to_csv(tab / "segment_coverage.csv", index=False)

    # ---- per-store coverage ------------------------------------------------
    test["_raw_in"] = (test["y"] >= test["q_0.10"]) & (test["y"] <= test["q_0.90"])
    test["_cal_in"] = (test["y"] >= test["cal_lo_80"]) & (test["y"] <= test["cal_hi_80"])
    test["_width"] = test["q_0.90"] - test["q_0.10"]
    test["_abs_err"] = (test["y"] - test["q_0.50"]).abs()
    store_cov = (
        test.groupby("Store", observed=True)
        .agg(
            n=("y", "size"),
            coverage_raw=("_raw_in", "mean"),
            coverage_cal=("_cal_in", "mean"),
            mean_width_raw=("_width", "mean"),
            promo_share=("Promo", "mean"),
            mae=("_abs_err", "mean"),
        )
        .reset_index()
    )
    store_cov.sort_values("coverage_raw").to_csv(tab / "store_coverage.csv",
                                                 index=False)
    worst_stores = store_cov.nsmallest(10, "coverage_raw")

    # ---- weekly coverage ---------------------------------------------------
    week_key = (test["Date"].dt.isocalendar().week.astype(int) * 10000
                + test["Date"].dt.year)  # unique (iso-week, year)
    tw = test.assign(_week=week_key)
    weekly = tw.groupby("_week").agg(
        week_label=("Date", lambda s: str(s.min().date())),
        n=("y", "size"),
        coverage_raw=("_raw_in", "mean"),
        coverage_cal=("_cal_in", "mean"),
        promo_share=("Promo", "mean"),
        mae=("_abs_err", "mean"),
    ).sort_index().reset_index()
    weekly.to_csv(tab / "weekly_coverage.csv", index=False)

    # ---- worst single-day misses (actual above the calibrated upper bound) --
    hi_miss = test[test["y"] > test["cal_hi_80"]].copy()
    hi_miss["excess"] = hi_miss["y"] - hi_miss["cal_hi_80"]
    worst_days = hi_miss.nlargest(10, "excess")[
        ["Store", "Date", "y", "q_0.50", "cal_hi_80", "excess", "Promo",
         "SchoolHoliday", "is_spike"]
    ]
    worst_days.to_csv(tab / "worst_interval_misses.csv", index=False)

    # ---- figures -----------------------------------------------------------
    fig_segments(cfg, seg)
    fig_store_coverage(cfg, store_cov)
    fig_weekly(cfg, weekly)

    # ---- headline numbers --------------------------------------------------
    def cov(seg_type, value, col="coverage_raw"):
        r = seg[(seg["segment"] == seg_type) & (seg["value"] == value)]
        return float(r[col].iloc[0])

    n_hi = len(hi_miss)
    hi_on_promo = float(hi_miss["Promo"].mean()) if n_hi else float("nan")
    promo_share_all = float(test["Promo"].mean())
    width_promo = float(
        seg[(seg["segment"] == "promo") & (seg["value"] == "promo day")]
        ["mean_width_raw"].iloc[0]
    )
    width_nopromo = float(
        seg[(seg["segment"] == "promo") & (seg["value"] == "no promo")]
        ["mean_width_raw"].iloc[0]
    )

    summary = {
        "spike_definition": f"actual >= store's train-period p{int(q * 100)}",
        "coverage": {
            "overall_raw": cov("overall", "all"),
            "overall_cal": cov("overall", "all", "coverage_cal"),
            "promo_raw": cov("promo", "promo day"),
            "no_promo_raw": cov("promo", "no promo"),
            "spike_raw": cov("demand_spike", "spike (>= train p95)"),
            "normal_raw": cov("demand_spike", "normal demand"),
            "school_holiday_raw": cov("school_holiday", "school holiday"),
        },
        "interval_width": {
            "promo_days": width_promo,
            "no_promo_days": width_nopromo,
            "promo_over_nopromo_ratio": width_promo / width_nopromo,
        },
        "high_misses": {
            "n_above_calibrated_hi": n_hi,
            "share_of_high_misses_on_promo": hi_on_promo,
            "overall_promo_share": promo_share_all,
        },
        "per_store": {
            "p10_coverage_raw": float(store_cov["coverage_raw"].quantile(0.10)),
            "median_coverage_raw": float(store_cov["coverage_raw"].median()),
            "p90_coverage_raw": float(store_cov["coverage_raw"].quantile(0.90)),
            "stores_below_70pct": int((store_cov["coverage_raw"] < 0.70).sum()),
            "n_stores": int(len(store_cov)),
            "worst_stores": worst_stores[
                ["Store", "coverage_raw", "coverage_cal", "promo_share", "mae"]
            ].to_dict("records"),
        },
        "worst_weeks": weekly.nsmallest(5, "coverage_raw")[
            ["week_label", "n", "coverage_raw", "coverage_cal", "promo_share", "mae"]
        ].to_dict("records"),
    }
    update_metrics(cfg, "overconfidence", summary)
    print(
        f"[analyze] coverage: promo {summary['coverage']['promo_raw']:.3f} vs "
        f"no-promo {summary['coverage']['no_promo_raw']:.3f} | spikes "
        f"{summary['coverage']['spike_raw']:.3f} | width ratio "
        f"promo/no-promo {width_promo / width_nopromo:.2f}"
    )
    return seg


if __name__ == "__main__":
    main()
