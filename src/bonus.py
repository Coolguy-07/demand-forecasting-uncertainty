"""Stage 8 (bonus): quantile regression vs. residual-based intervals.

Both interval methods are calibrated on the SAME validation month:
  * quantile GBM: the q10/q90 models themselves (plus the scale-calibrated
    version of them);
  * residual: take the point forecast (median model), look at its validation
    residuals, and add the 10th/90th percentile of those residuals.
Then both are compared on the untouched test period: coverage, width,
pinball loss and the inventory decision at Cu/Co = 9.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.common import COLORS, load_predictions, save_fig, tables_dir, update_metrics
from src.config import load_config


# ---------------------------------------------------------------------------
# residual-interval machinery
# ---------------------------------------------------------------------------
def _offsets(val, kind, relative):
    """Validation residual quantiles. Returns (lookup, fallback).

    lookup maps store -> (q10, q90) of that store's residuals;
    fallback is the global (q10, q90) used when a store has too few points.
    """
    res = (val["y"].to_numpy() - val["q_0.50"].to_numpy())
    if relative:
        q50 = val["q_0.50"].to_numpy()
        mask = q50 > 0
        res = res[mask] / q50[mask]

    fallback = tuple(np.quantile(res, [0.10, 0.90]))
    if kind == "global":
        return {"__global__": fallback}, fallback

    stores = val["Store"].astype(int).to_numpy()[mask] if relative \
        else val["Store"].astype(int).to_numpy()
    frame = pd.DataFrame({"store": stores, "r": res})
    lookup = {}
    for store, grp in frame.groupby("store"):
        if len(grp) >= 5:
            lookup[store] = tuple(np.quantile(grp["r"], [0.10, 0.90]))
    return lookup, fallback


def _apply_offsets(test, lookup, fallback, relative):
    q50 = test["q_0.50"].to_numpy()
    stores = test["Store"].astype(int).to_numpy()
    lo = np.empty(len(test))
    hi = np.empty(len(test))
    global_off = lookup.get("__global__", fallback)
    for i, s in enumerate(stores):
        ql, qh = lookup.get(s, global_off)
        if relative:
            lo[i] = max(q50[i] * (1 + ql), 0.0)
            hi[i] = q50[i] * (1 + qh)
        else:
            lo[i] = max(q50[i] + ql, 0.0)
            hi[i] = q50[i] + qh
    return lo, hi


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------
def _pinball(y, q, alpha):
    d = y - q
    return np.maximum(alpha * d, (alpha - 1.0) * d)


def _stats(y, lo, hi):
    return {
        "coverage": float(np.mean((y >= lo) & (y <= hi))),
        "mean_width": float(np.mean(hi - lo)),
        "pinball_10_90": float(
            (np.mean(_pinball(y, lo, 0.10)) + np.mean(_pinball(y, hi, 0.90))) / 2
        ),
    }


def _cost_cu9(y, Q):
    return float(np.maximum(Q - y, 0).sum() + 9.0 * np.maximum(y - Q, 0).sum())


# ---------------------------------------------------------------------------
def main(cfg=None):
    cfg = cfg or load_config()
    preds = load_predictions(cfg)
    val = preds[preds["split"] == "valid"]
    test = preds[preds["split"] == "test"]
    y_val, y = val["y"].to_numpy(), test["y"].to_numpy()

    rows = []

    def add(name, lo_va, hi_va, lo_te, hi_te):
        sva = _stats(y_val, lo_va, hi_va)
        ste = _stats(y, lo_te, hi_te)
        rows.append(
            {
                "method": name,
                "coverage_valid": sva["coverage"],
                "coverage_test": ste["coverage"],
                "mean_width_test": ste["mean_width"],
                "pinball_10_90_test": ste["pinball_10_90"],
                "cost_Cu9_test": _cost_cu9(y, hi_te),
            }
        )

    # 1) quantile GBM, raw
    add(
        "Quantile GBM (q10/q90, raw)",
        val["q_0.10"].to_numpy(), val["q_0.90"].to_numpy(),
        test["q_0.10"].to_numpy(), test["q_0.90"].to_numpy(),
    )
    # 2) quantile GBM + scale calibration (the headline method)
    add(
        "Quantile GBM + scale calibration",
        val["cal_lo_80"].to_numpy(), val["cal_hi_80"].to_numpy(),
        test["cal_lo_80"].to_numpy(), test["cal_hi_80"].to_numpy(),
    )
    # 3) residual intervals, global
    lk, fb = _offsets(val, "global", relative=False)
    lv, hv = _apply_offsets(val, lk, fb, relative=False)
    lt, ht = _apply_offsets(test, lk, fb, relative=False)
    add("Residual - global (point + residual q10/q90)", lv, hv, lt, ht)
    # 4) residual intervals, per store (absolute)
    lk, fb = _offsets(val, "store", relative=False)
    lv, hv = _apply_offsets(val, lk, fb, relative=False)
    lt, ht = _apply_offsets(test, lk, fb, relative=False)
    add("Residual - per store (absolute)", lv, hv, lt, ht)
    # 5) residual intervals, per store (relative to forecast level)
    lk, fb = _offsets(val, "store", relative=True)
    lv, hv = _apply_offsets(val, lk, fb, relative=True)
    lt, ht = _apply_offsets(test, lk, fb, relative=True)
    add("Residual - per store (relative)", lv, hv, lt, ht)

    table = pd.DataFrame(rows)
    tab = tables_dir(cfg)
    table.to_csv(tab / "bonus_comparison.csv", index=False)

    # ---- figure ------------------------------------------------------------
    labels = [m.replace(" (", "\n(") for m in table["method"]]
    x = np.arange(len(table))
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.8))

    ax = axes[0]
    ax.bar(x - 0.2, table["coverage_valid"], 0.4, label="valid month",
           color=COLORS["naive"])
    ax.bar(x + 0.2, table["coverage_test"], 0.4, label="test period",
           color=COLORS["point"])
    ax.axhline(0.8, color="black", ls="--", lw=1.2, label="nominal 80%")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=7.5)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Coverage of 80% interval")
    ax.set_title("Calibration holds, or drifts?")
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.bar(x, table["mean_width_test"], color=COLORS["ok"])
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=7.5)
    ax.set_ylabel("Mean interval width (test)")
    ax.set_title("How wide is the interval?")

    ax = axes[2]
    ax.bar(x, table["cost_Cu9_test"] / 1e6, color=COLORS["orange"])
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=7.5)
    ax.set_ylabel("Inventory cost @ Cu/Co=9 (millions)")
    ax.set_title("Downstream inventory cost (lower is better)")

    fig.suptitle("Bonus: quantile regression vs. residual-based intervals",
                 y=1.02, fontweight="bold")
    save_fig(cfg, "fig_bonus_comparison.png")

    # ---- metrics -----------------------------------------------------------
    point_cost = _cost_cu9(y, test["q_0.50"].to_numpy())
    update_metrics(
        cfg,
        "bonus",
        {
            "setup": (
                "both methods calibrated on the validation month; "
                "residual intervals = point forecast + quantiles of its residuals"
            ),
            "point_forecast_cost_Cu9_test": point_cost,
            "methods": table.to_dict("records"),
        },
    )
    print("[bonus]\n" + table.to_string(index=False))
    return table


if __name__ == "__main__":
    main()
