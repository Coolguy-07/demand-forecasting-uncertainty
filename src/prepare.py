"""Stage 2: load raw CSVs, clean, merge store metadata, write base cache."""

import pandas as pd

from src.config import load_config

HOLIDAY_MAP = {
    "0": "none",
    "nan": "none",
    "a": "public",
    "b": "easter",
    "c": "christmas",
}


def load_raw(cfg):
    raw = cfg["paths"]["raw_dir"]
    train = pd.read_csv(
        raw / "train.csv", dtype={"StateHoliday": str}, low_memory=False, parse_dates=["Date"]
    )
    stores = pd.read_csv(raw / "store.csv")
    return train, stores


def prepare():
    cfg = load_config()
    train, stores = load_raw(cfg)

    # --- cleaning decisions -------------------------------------------------
    # StateHoliday arrives with mixed int/str types -> normalise to labels.
    train["StateHoliday"] = train["StateHoliday"].map(HOLIDAY_MAP).fillna("none")
    # Customers is only observable after the day ends -> excluded (would leak).
    train = train.drop(columns=["Customers"])
    # DayOfWeek is recomputed from Date in the feature stage -> drop raw copy.
    train = train.drop(columns=["DayOfWeek"])

    assert (train["Sales"] >= 0).all(), "negative sales found"

    df = train.merge(stores, on="Store", how="left", validate="m:1")
    df = df.sort_values(["Store", "Date"]).reset_index(drop=True)

    cache = cfg["paths"]["cache_dir"]
    cache.mkdir(parents=True, exist_ok=True)
    out = cache / "base.parquet"
    df.to_parquet(out, index=False)
    print(
        f"[prepare] {len(df):,} rows | {df['Store'].nunique()} stores | "
        f"{df['Date'].min().date()} .. {df['Date'].max().date()} -> {out}"
    )
    return df


def main():
    prepare()
