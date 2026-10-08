"""Stage 1: download the Rossmann dataset (public mirror, no credentials)."""

import urllib.request

from src.config import load_config

USER_AGENT = "Mozilla/5.0 (compatible; demand-forecasting-repro/1.0)"


def _fetch(url, dest):
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=120) as resp, open(dest, "wb") as out:
        while True:
            chunk = resp.read(1 << 20)
            if not chunk:
                break
            out.write(chunk)


def validate(cfg):
    """Verify the downloaded files match the expected dataset shape."""
    import pandas as pd

    raw = cfg["paths"]["raw_dir"]
    train = pd.read_csv(raw / "train.csv", usecols=["Store", "Date", "Sales"])
    stores = pd.read_csv(raw / "store.csv")
    assert len(train) == cfg["data"]["expected_rows"], (
        f"train.csv has {len(train):,} rows, expected {cfg['data']['expected_rows']:,}"
    )
    n_stores = train["Store"].nunique()
    assert n_stores == cfg["data"]["expected_stores"], (
        f"train.csv has {n_stores} stores, expected {cfg['data']['expected_stores']}"
    )
    assert len(stores) == cfg["data"]["expected_stores"], "store.csv row count mismatch"
    print(
        f"[download] validation OK: {len(train):,} rows, {n_stores} stores, "
        f"{train['Date'].min()} .. {train['Date'].max()}"
    )


def download(force=False):
    cfg = load_config()
    raw = cfg["paths"]["raw_dir"]
    raw.mkdir(parents=True, exist_ok=True)
    for name, url in cfg["data"]["urls"].items():
        dest = raw / f"{name}.csv"
        if dest.exists() and dest.stat().st_size > 0 and not force:
            print(f"[download] {dest.name} already present ({dest.stat().st_size:,} bytes)")
            continue
        print(f"[download] fetching {url}")
        _fetch(url, dest)
        print(f"[download] saved {dest} ({dest.stat().st_size:,} bytes)")
    validate(cfg)


def main():
    download()
