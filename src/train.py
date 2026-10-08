"""Stage 4: train LightGBM quantile models (pinball loss) and predict.

Outputs
-------
* outputs/models/q_0.xx.txt       - one LightGBM booster per quantile level
* outputs/predictions.parquet     - quantile predictions for valid + test rows
* outputs/metrics.json            - training section
"""

import time

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.common import quantile_col, save_predictions, update_metrics
from src.config import load_config

# columns that are keys/targets rather than features
EXCLUDE = {"Date", "split", "Sales"}
PRED_KEYS = ["Store", "Date", "split", "Sales", "Promo", "SchoolHoliday",
             "dow", "lag_7"]


def load_data(cfg):
    df = pd.read_parquet(cfg["paths"]["cache_dir"] / "features.parquet")
    # integer categoricals don't survive the parquet roundtrip - restore them
    # so LightGBM treats store/day-of-week/month as categories, not ordered numbers
    for col in ["Store", "dow", "month"]:
        df[col] = df[col].astype("category")
    feats = [c for c in df.columns if c not in EXCLUDE]
    parts = {sp: df[df["split"] == sp] for sp in ["train", "valid", "test"]}
    return df, feats, parts


def train(cfg=None):
    cfg = cfg or load_config()
    df, feats, parts = load_data(cfg)

    X_tr, y_tr = parts["train"][feats], parts["train"]["Sales"].to_numpy()
    X_va, y_va = parts["valid"][feats], parts["valid"]["Sales"].to_numpy()
    X_te = parts["test"][feats]

    m = cfg["model"]
    base_params = dict(
        n_estimators=m["n_estimators"],
        learning_rate=m["learning_rate"],
        num_leaves=m["num_leaves"],
        min_child_samples=m["min_child_samples"],
        feature_fraction=m["feature_fraction"],
        bagging_fraction=m["bagging_fraction"],
        bagging_freq=m["bagging_freq"],
        lambda_l1=m["lambda_l1"],
        lambda_l2=m["lambda_l2"],
        deterministic=True,
        force_row_wise=True,
        num_threads=cfg["num_threads"],
        seed=cfg["seed"],
        verbosity=-1,
    )

    qcols = [quantile_col(q) for q in cfg["quantiles"]]
    val_preds, test_preds, model_info = {}, {}, {}

    for q in cfg["quantiles"]:
        col = quantile_col(q)
        t0 = time.time()
        model = lgb.LGBMRegressor(
            objective="quantile",
            alpha=q,
            eval_metric="quantile",
            **base_params,
        )
        model.fit(
            X_tr,
            y_tr,
            eval_set=[(X_va, y_va)],
            callbacks=[
                lgb.early_stopping(
                    m["early_stopping_rounds"], first_metric_only=True, verbose=False
                )
            ],
        )
        pv = np.maximum(model.predict(X_va), 0.0)
        pte = np.maximum(model.predict(X_te), 0.0)
        val_preds[col] = pv
        test_preds[col] = pte

        pinball_va = float(np.mean(_pinball(y_va, pv, q)))
        model_info[col] = {
            "alpha": q,
            "best_iteration": int(model.best_iteration_),
            "fit_seconds": round(time.time() - t0, 1),
            "valid_pinball": round(pinball_va, 3),
        }
        cfg["paths"]["models_dir"].mkdir(parents=True, exist_ok=True)
        model.booster_.save_model(str(cfg["paths"]["models_dir"] / f"{col}.txt"))
        print(
            f"[train] {col}: {model.best_iteration_} trees, "
            f"val pinball {pinball_va:.2f}, {model_info[col]['fit_seconds']}s"
        )

    # ---- assemble predictions + fix quantile crossing ----------------------
    frames = []
    for sp, preds in [("valid", val_preds), ("test", test_preds)]:
        part = parts[sp][PRED_KEYS].rename(columns={"Sales": "y", "lag_7": "naive"})
        for col, p in preds.items():
            part[col] = p
        frames.append(part)
    out = pd.concat(frames, ignore_index=True)

    arr = out[qcols].to_numpy()
    crossed = ~np.all(np.diff(arr, axis=1) >= -1e-9, axis=1)
    n_crossed = int(crossed.sum())
    if n_crossed:
        out[qcols] = np.sort(arr, axis=1)
        print(f"[train] quantile crossing in {n_crossed:,} rows - sorted into order")
    out["point"] = out[quantile_col(0.50)]

    save_predictions(cfg, out)
    update_metrics(
        cfg,
        "training",
        {
            "library": f"lightgbm {lgb.__version__}",
            "objective": "quantile (pinball loss)",
            "n_features": len(feats),
            "feature_names": feats,
            "rows": {sp: int(len(p)) for sp, p in parts.items()},
            "models": model_info,
            "quantile_crossed_rows": n_crossed,
            "crossing_note": "rows whose quantiles were non-monotonic and got sorted",
        },
    )
    return out


def _pinball(y, q, alpha):
    d = y - q
    return np.maximum(alpha * d, (alpha - 1.0) * d)


def main():
    train()
