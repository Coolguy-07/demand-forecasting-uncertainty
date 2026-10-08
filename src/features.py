"""Stage 3: time-based feature engineering.

Design rules
------------
* Every target-derived feature is lagged by >= 1 day, so a row's features only
  use information available *before* that day (leak-free one-step-ahead setup).
* The raw data has gaps (~33k missing store-days). We first reindex every store
  to a complete daily grid so that lag-7 always means "same weekday last week",
  then compute features on the full grid and only afterwards keep open days.
* Known-in-advance covariates (promo schedule, school-holiday calendar, store
  metadata, date features) may be used at time t - they are published ahead.
"""

import numpy as np
import pandas as pd

from src.config import load_config

HELPER_COLS = ["_s1", "_prev_dow"]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _complete_grid(df):
    """Reindex every store to a continuous daily date range (gaps -> NaN)."""
    dates = pd.date_range(df["Date"].min(), df["Date"].max(), freq="D")
    stores = np.sort(df["Store"].unique())
    idx = pd.MultiIndex.from_product([stores, dates], names=["Store", "Date"])
    out = df.set_index(["Store", "Date"]).reindex(idx).reset_index()
    n_added = len(out) - len(df)
    print(f"[features] complete grid: {len(out):,} rows ({n_added:,} imputed gap days)")
    return out


def _month_set(interval):
    if not isinstance(interval, str) or not interval:
        return frozenset()
    return frozenset(m.strip()[:3] for m in interval.split(",") if m.strip())


def _iso_week_start(year, week):
    try:
        return pd.Timestamp.fromisocalendar(int(year), int(week), 1)
    except (ValueError, OverflowError):
        return pd.NaT


def _promo2_features(df):
    """Promo2 is a rolling second promotion - schedule is known in advance."""
    sy, sw = df["Promo2SinceYear"], df["Promo2SinceWeek"]
    pairs = pd.DataFrame({"y": sy, "w": sw}).dropna().drop_duplicates()
    mapping = {
        (int(r.y), int(r.w)): _iso_week_start(r.y, r.w) for r in pairs.itertuples()
    }
    starts = []
    for y, w in zip(sy, sw):
        if pd.isna(y) or pd.isna(w):
            starts.append(pd.NaT)
        else:
            starts.append(mapping.get((int(y), int(w)), pd.NaT))
    start = pd.Series(pd.to_datetime(starts), index=df.index)

    months = df["PromoInterval"].map(_month_set)
    in_month = pd.Series(
        [d.month_name()[:3] in m for d, m in zip(df["Date"], months)], index=df.index
    )
    started = start.notna() & (df["Date"] - start).dt.days.ge(0)
    active = started & in_month & df["Promo2"].eq(1)

    weeks = (df["Date"] - start).dt.days.div(7).where(start.notna(), other=np.nan)
    out = pd.DataFrame(index=df.index)
    out["promo2_active"] = active.astype(np.int8)
    out["promo2_weeks_active"] = weeks.astype(np.float32)
    return out


def _competition_months_open(df):
    """Months since the nearest competitor opened (schedule partly unknown -> NaN)."""
    cy, cm = df["CompetitionOpenSinceYear"], df["CompetitionOpenSinceMonth"]
    valid = cy.between(1990, 2016) & cm.between(1, 12)
    since = cy.fillna(0) * 12 + cm.fillna(0)
    now = df["Date"].dt.year * 12 + df["Date"].dt.month
    return (now - since).where(valid, other=np.nan).astype(np.float32)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def build_features(df, cfg):
    fc = cfg["features"]
    df = _complete_grid(df)

    # ---- calendar (known in advance) --------------------------------------
    date = df["Date"]
    df["dow"] = date.dt.dayofweek.astype(np.int8)          # 0 = Monday
    df["month"] = date.dt.month.astype(np.int8)
    df["year"] = date.dt.year.astype(np.int16)
    df["weekofyear"] = date.dt.isocalendar().week.astype(np.int16)
    df["dayofyear"] = date.dt.dayofyear.astype(np.int16)
    df["quarter"] = date.dt.quarter.astype(np.int8)
    df["is_weekend"] = (df["dow"] >= 5).astype(np.int8)
    df["is_month_start"] = date.dt.is_month_start.astype(np.int8)

    store_grp = df.groupby("Store", sort=False)

    # ---- target lags -------------------------------------------------------
    for lag in fc["lags"]:
        df[f"lag_{lag}"] = store_grp["Sales"].shift(lag).astype(np.float32)

    # ---- rolling statistics over the previous day's sales ------------------
    df["_s1"] = store_grp["Sales"].shift(1)
    s1_grp = df.groupby("Store", sort=False)["_s1"]
    for w in fc["rolling_windows"]:
        df[f"roll_mean_{w}"] = s1_grp.transform(
            lambda x, w=w: x.rolling(w, min_periods=max(3, w // 2)).mean()
        ).astype(np.float32)
    for w in fc["std_windows"]:
        df[f"roll_std_{w}"] = s1_grp.transform(
            lambda x, w=w: x.rolling(w, min_periods=max(3, w // 2)).std()
        ).astype(np.float32)

    ratio = np.where(
        df["roll_mean_28"] > 0, df["roll_mean_7"] / df["roll_mean_28"], np.nan
    )
    df["mean7_over_mean28"] = pd.Series(ratio, index=df.index).astype(np.float32)

    # ---- same-weekday history (mean of previous 4 weeks' same weekday) -----
    prev_dow = df.groupby(["Store", "dow"], sort=False, observed=True)["Sales"].shift(1)
    df["_prev_dow"] = prev_dow
    df["same_dow_mean_4"] = (
        df.groupby(["Store", "dow"], sort=False, observed=True)["_prev_dow"]
        .transform(lambda x: x.rolling(4, min_periods=1).mean())
        .astype(np.float32)
    )

    # ---- closure / promotion context --------------------------------------
    df["open_lag1"] = store_grp["Open"].shift(1).astype(np.float32)

    promo = df["Promo"].fillna(0).astype(np.int8)
    streak_id = (promo == 0).cumsum()
    streak = promo.groupby(streak_id).cumcount() + 1
    df["promo_streak"] = np.where(promo == 1, streak, 0).astype(np.int16)

    df["promo_lag1"] = store_grp["Promo"].shift(1).astype(np.float32)
    df["promo_sum_7"] = (
        store_grp["Promo"]
        .shift(1)
        .groupby(df["Store"], sort=False)
        .transform(lambda x: x.rolling(7, min_periods=1).sum())
        .astype(np.float32)
    )

    df = pd.concat([df, _promo2_features(df)], axis=1)

    # ---- competition / store metadata --------------------------------------
    df["competition_months_open"] = _competition_months_open(df)
    df["comp_dist"] = df["CompetitionDistance"].astype(np.float32)
    df["comp_dist_log"] = np.log1p(df["CompetitionDistance"]).astype(np.float32)
    df["has_competition"] = df["CompetitionDistance"].notna().astype(np.int8)
    df["Promo2"] = df["Promo2"].fillna(0).astype(np.int8)
    df["SchoolHoliday"] = df["SchoolHoliday"].fillna(0).astype(np.int8)

    # ---- market-wide context (previous day, cross-store mean) --------------
    market = df.groupby("Date", sort=False)["Sales"].transform("mean")
    df["market_sales_lag1"] = (
        market.groupby(df["Store"], sort=False).shift(1).astype(np.float32)
    )

    return df


def finalize(df, cfg):
    """Keep open days, assign split labels, cast dtypes, save."""
    mask = df["Open"].eq(1) & df["Sales"].notna()
    out = df[mask].copy()

    train_end = pd.Timestamp(cfg["splits"]["train_end"])
    valid_end = pd.Timestamp(cfg["splits"]["valid_end"])
    out["split"] = np.select(
        [out["Date"] <= train_end, out["Date"] <= valid_end],
        ["train", "valid"],
        default="test",
    )

    drop_cols = [c for c in HELPER_COLS if c in out.columns] + [
        c for c in ["Open", "CompetitionOpenSinceMonth", "CompetitionOpenSinceYear",
                    "Promo2SinceWeek", "Promo2SinceYear", "PromoInterval"]
        if c in out.columns
    ]
    out = out.drop(columns=drop_cols)

    out["Sales"] = out["Sales"].astype(np.float32)
    for col in out.columns:
        if out[col].dtype == "float64":
            out[col] = out[col].astype(np.float32)

    # categorical features for LightGBM
    for col in ["Store", "StoreType", "Assortment", "dow", "month", "StateHoliday"]:
        out[col] = out[col].astype("category")

    out = out.sort_values(["Store", "Date"]).reset_index(drop=True)
    return out


def build(cfg=None):
    cfg = cfg or load_config()
    base = pd.read_parquet(cfg["paths"]["cache_dir"] / "base.parquet")
    feats = build_features(base, cfg)
    feats = finalize(feats, cfg)

    cache = cfg["paths"]["cache_dir"]
    out = cache / "features.parquet"
    feats.to_parquet(out, index=False)

    sizes = feats["split"].value_counts().to_dict()
    print(
        f"[features] {len(feats):,} modelled rows | train {sizes.get('train', 0):,} | "
        f"valid {sizes.get('valid', 0):,} | test {sizes.get('test', 0):,} -> {out}"
    )
    return feats


def main():
    build()
