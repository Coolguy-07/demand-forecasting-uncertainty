"""Shared helpers: metrics JSON IO, prediction IO, plotting style."""

import json

import matplotlib

matplotlib.use("Agg")  # headless backend - figures are written to disk
import matplotlib.pyplot as plt
import pandas as pd

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update(
    {
        "figure.dpi": 110,
        "savefig.dpi": 170,
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "legend.fontsize": 9,
        "axes.grid": True,
        "grid.alpha": 0.3,
    }
)

# palette
COLORS = {
    "actual": "#1b1b1b",
    "point": "#1f77b4",
    "band": "#1f77b4",
    "cal": "#d62728",
    "naive": "#7f7f7f",
    "ok": "#2ca02c",
    "orange": "#ff7f0e",
    "purple": "#9467bd",
    "brown": "#8c564b",
}


def quantile_col(q):
    """Column name for a quantile prediction, e.g. 0.10 -> 'q_0.10'."""
    return f"q_{q:.2f}"


def metrics_path(cfg):
    return cfg["paths"]["outputs_dir"] / "metrics.json"


def load_metrics(cfg):
    p = metrics_path(cfg)
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return {}


def save_metrics(cfg, metrics):
    p = metrics_path(cfg)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(metrics, indent=2, default=float), encoding="utf-8")


def update_metrics(cfg, section, data):
    """Merge one top-level section into outputs/metrics.json."""
    metrics = load_metrics(cfg)
    metrics[section] = data
    save_metrics(cfg, metrics)
    print(f"[metrics] wrote section '{section}' -> {metrics_path(cfg)}")


def tables_dir(cfg):
    d = cfg["paths"]["tables_dir"]
    d.mkdir(parents=True, exist_ok=True)
    return d


def predictions_path(cfg):
    return cfg["paths"]["outputs_dir"] / "predictions.parquet"


def load_predictions(cfg):
    return pd.read_parquet(predictions_path(cfg))


def save_predictions(cfg, preds):
    cfg["paths"]["outputs_dir"].mkdir(parents=True, exist_ok=True)
    preds.to_parquet(predictions_path(cfg), index=False)
    print(f"[predictions] {len(preds):,} rows -> {predictions_path(cfg)}")


def save_fig(cfg, name):
    d = cfg["paths"]["figures_dir"]
    d.mkdir(parents=True, exist_ok=True)
    out = d / name
    plt.savefig(out, bbox_inches="tight")
    plt.close("all")
    print(f"[figure] {out}")
