"""Stage 6: translate the prediction interval into an inventory decision.

Newsvendor-style daily policy per store:
    order Q units for tomorrow, leftover is wasted (overage cost Co),
    unmet demand is lost (shortage cost Cu = ratio * Co).
A forecast at the (ratio / (1 + ratio)) quantile is cost-optimal when the
forecast is perfectly calibrated - so we compare:
    * ordering on the point forecast (median),
    * ordering on the upper bound of the 80% / 90% PI (raw + calibrated),
    * a seasonal-naive order,
    * a hindsight oracle (per-store test-period quantile - unattainable reference).
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.common import (
    COLORS,
    load_predictions,
    save_fig,
    tables_dir,
    update_metrics,
)
from src.config import load_config


def policy_quantiles(preds):
    return {
        "Seasonal naive (lag 7)": preds["naive"].to_numpy(),
        "Point forecast (q50)": preds["q_0.50"].to_numpy(),
        "Upper 80% PI (q90)": preds["q_0.90"].to_numpy(),
        "Upper 80% PI calibrated": preds["cal_hi_80"].to_numpy(),
        "Upper 90% PI (q95)": preds["q_0.95"].to_numpy(),
    }


def evaluate_policy(Q, D, ratio, overage):
    shortage_cost = ratio * overage
    over = np.maximum(Q - D, 0.0)
    short = np.maximum(D - Q, 0.0)
    return {
        "total_cost": float(over.sum() * overage + short.sum() * shortage_cost),
        "overage_units": float(over.sum()),
        "shortage_units": float(short.sum()),
        "fill_rate": float(1.0 - short.sum() / max(D.sum(), 1.0)),
        "stockout_day_rate": float(np.mean(short > 0)),
        "avg_order": float(Q.mean()),
    }


def oracle_quantiles(test, ratios, store_col="Store", y_col="y"):
    """Per-store hindsight quantile of test demand for each cost ratio."""
    u_grid = np.array([r / (r + 1.0) for r in ratios])
    q_star = {}
    for store, grp in test.groupby(store_col, observed=True, sort=False):
        y = grp[y_col].to_numpy()
        q_star[store] = np.quantile(y, u_grid)
    return q_star, u_grid


def main(cfg=None):
    cfg = cfg or load_config()
    preds = load_predictions(cfg)
    test = preds[preds["split"] == "test"].sort_values(["Store", "Date"])
    D = test["y"].to_numpy()

    overage = cfg["decision"]["overage_cost"]
    ratios = cfg["decision"]["cost_ratios"]
    plot_max = cfg["decision"]["plot_ratio_max"]
    plot_ratios = np.linspace(1, plot_max, 60)

    policies = policy_quantiles(test)

    # ---- table over the configured ratios ---------------------------------
    rows = []
    for ratio in ratios:
        for name, Q in policies.items():
            rows.append({"policy": name, "cost_ratio": ratio,
                         **evaluate_policy(Q, D, ratio, overage)})
    table = pd.DataFrame(rows)
    tab = tables_dir(cfg)
    table.to_csv(tab / "decision_costs.csv", index=False)

    # ---- headline: shortage 9x as expensive as overstock -------------------
    r_head = 9
    head = {}
    sub = table[table["cost_ratio"] == r_head].set_index("policy")
    for name in policies:
        head[name] = {
            "total_cost": float(sub.loc[name, "total_cost"]),
            "fill_rate": float(sub.loc[name, "fill_rate"]),
            "stockout_day_rate": float(sub.loc[name, "stockout_day_rate"]),
            "overage_units": float(sub.loc[name, "overage_units"]),
            "shortage_units": float(sub.loc[name, "shortage_units"]),
        }
    point_cost = head["Point forecast (q50)"]["total_cost"]
    savings = {
        name: 100.0 * (point_cost - v["total_cost"]) / point_cost
        for name, v in head.items()
    }

    # ---- oracle cost curve -------------------------------------------------
    q_star, u_grid = oracle_quantiles(test, list(plot_ratios))
    oracle_costs = np.zeros(len(plot_ratios))
    store_vals = test["Store"].to_numpy()
    for i, store in enumerate(q_star):
        mask = store_vals == store
        Ds = D[mask]
        for j, u in enumerate(u_grid):
            Q = q_star[store][j]
            oracle_costs[j] += np.maximum(Q - Ds, 0).sum() * overage + np.maximum(
                Ds - Q, 0
            ).sum() * plot_ratios[j] * overage

    # ---- figure: cost + fill rate vs cost ratio ----------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.6))
    style = {
        "Seasonal naive (lag 7)": dict(c=COLORS["naive"], ls="--"),
        "Point forecast (q50)": dict(c=COLORS["point"], lw=2),
        "Upper 80% PI (q90)": dict(c=COLORS["ok"], lw=2),
        "Upper 80% PI calibrated": dict(c=COLORS["cal"], lw=2),
        "Upper 90% PI (q95)": dict(c=COLORS["purple"], lw=2),
    }
    for name, Q in policies.items():
        costs, fills = [], []
        for r in plot_ratios:
            res = evaluate_policy(Q, D, r, overage)
            costs.append(res["total_cost"])
            fills.append(res["fill_rate"])
        ax1.plot(plot_ratios, costs, label=name, **style[name])
        ax2.plot(plot_ratios, fills, label=name, **style[name])
    ax1.plot(plot_ratios, oracle_costs, c="black", ls=":", lw=1.6,
             label="Hindsight oracle (reference)")
    ax1.set_yscale("log")
    ax1.set_xlabel("Shortage cost as multiple of overstock cost (Cu/Co)")
    ax1.set_ylabel("Total test-period cost (log scale, Co = 1)")
    ax1.set_title("Inventory cost vs. cost assumptions")
    ax1.legend(fontsize=8, loc="upper left")

    ax2.set_xlabel("Shortage cost as multiple of overstock cost (Cu/Co)")
    ax2.set_ylabel("Fill rate (units served / units demanded)")
    ax2.set_title("Service level vs. cost assumptions")
    ax2.legend(fontsize=8, loc="lower right")
    save_fig(cfg, "fig_cost_tradeoff.png")

    update_metrics(
        cfg,
        "decision",
        {
            "setup": (
                "daily newsvendor per store: order Q, leftover wasted at Co=1, "
                "lost demand charged at Cu = ratio * Co"
            ),
            "cost_ratios": ratios,
            "headline_ratio": r_head,
            f"at_ratio_{r_head}": head,
            f"savings_vs_point_pct_at_ratio_{r_head}": savings,
            "fill_rate_note": "fill rate of the point (median) policy is ~50% by construction",
        },
    )
    print(
        f"[decision] Cu/Co = {r_head}: point cost {point_cost:,.0f} | "
        f"q90 cost {head['Upper 80% PI (q90)']['total_cost']:,.0f} "
        f"({savings['Upper 80% PI (q90)']:.1f}% cheaper) | "
        f"calibrated cost {head['Upper 80% PI calibrated']['total_cost']:,.0f} "
        f"({savings['Upper 80% PI calibrated']:.1f}% cheaper)"
    )
    return table


if __name__ == "__main__":
    main()
