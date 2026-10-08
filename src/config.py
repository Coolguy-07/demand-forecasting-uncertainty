"""Central configuration: loads config.yaml and resolves relative paths."""

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_config(path=None):
    """Load config.yaml (defaults to the project root copy)."""
    path = Path(path) if path else ROOT / "config.yaml"
    with open(path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    cfg["root"] = ROOT
    for key, val in list(cfg["paths"].items()):
        p = Path(val)
        cfg["paths"][key] = p if p.is_absolute() else ROOT / p
    return cfg
