"""Run the full pipeline (or selected stages).

    python run_all.py                       # everything, in order
    python run_all.py --stage train evaluate
    python run_all.py --stage analyze

Stage order:
    download -> prepare -> features -> train -> evaluate -> decision -> analyze -> bonus
"""

import argparse
import time

from src import bonus, decision, download, evaluate, features, overconfidence
from src import prepare as prepare_mod
from src import train as train_mod

STAGES = [
    ("download", download.main),
    ("prepare", prepare_mod.main),
    ("features", features.main),
    ("train", train_mod.main),
    ("evaluate", evaluate.main),
    ("decision", decision.main),
    ("analyze", overconfidence.main),
    ("bonus", bonus.main),
]


def main():
    parser = argparse.ArgumentParser(description="Demand forecasting pipeline")
    parser.add_argument(
        "--stage",
        nargs="+",
        choices=[name for name, _ in STAGES] + ["all"],
        default=["all"],
        help="stages to run (default: all, in pipeline order)",
    )
    args = parser.parse_args()
    selected = set(args.stage)

    for name, fn in STAGES:
        if "all" in selected or name in selected:
            t0 = time.time()
            print(f"\n=== stage: {name} " + "=" * max(1, 58 - len(name)))
            fn()
            print(f"=== {name} finished in {time.time() - t0:.1f}s ===")

    # short end-of-run summary if metrics exist
    try:
        from src.common import load_metrics
        from src.config import load_config

        m = load_metrics(load_config())
        if "evaluation" in m:
            ev = m["evaluation"]
            c80 = ev["coverage"]["test"]["80"]
            print("\n--- results ---")
            print(
                f"point MAE (test): {ev['point']['test']['q50_median']['MAE']:,.1f} | "
                f"naive: {ev['point']['test']['seasonal_naive_lag7']['MAE']:,.1f}"
            )
            print(
                f"80% interval coverage (test): raw {c80['raw']['coverage']:.3f} "
                f"-> calibrated {c80['calibrated']['coverage']:.3f} "
                f"(nominal {c80['nominal']:.2f})"
            )
        if "decision" in m:
            s = m["decision"].get("savings_vs_point_pct_at_ratio_9", {})
            if "Upper 80% PI (q90)" in s:
                print(
                    f"inventory cost saving vs point forecast @ Cu/Co=9: "
                    f"{s['Upper 80% PI (q90)']:.1f}% (q90 policy)"
                )
    except Exception as exc:  # summary is best-effort only
        print(f"(summary skipped: {exc})")


if __name__ == "__main__":
    main()
