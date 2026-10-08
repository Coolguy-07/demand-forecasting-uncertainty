# Data

## Source

**Rossmann Store Sales** — daily sales for 1,115 German pharmacy stores,
2013-01-01 .. 2015-07-31 (the Kaggle *Rossmann Store Sales* competition data).

The assignment suggested the M5 or Store Item Demand datasets from Kaggle.
Both sit behind Kaggle authentication, which would make the pipeline
non-reproducible for a reviewer without an account. This project therefore uses
a public mirror of the Rossmann competition files that downloads with a plain
HTTP request — no API key, no login:

- `train.csv` — 1,017,209 rows: `Store, Date, Sales, Customers, Open, Promo,
  StateHoliday, SchoolHoliday`
- `store.csv` — 1,115 rows: store metadata, assortment, competition and Promo2
  schedule

Mirror: <https://huggingface.co/datasets/AiiN-aini/rossmann-store-sales>
(original competition: <https://www.kaggle.com/competitions/rossmann-store-sales>)

The scenario fits the assignment well: a retailer forecasting **product demand
per store** with promotion flags (`Promo`, `Promo2`) and holiday calendars
(`StateHoliday`, `SchoolHoliday`) — exactly the "holidays/promotions" signal
the requirements ask for, plus real demand spikes around promotions to analyse
overconfidence against.

## Getting the files

```
python run_all.py --stage download
```

`data/raw/*.csv` is git-ignored; the download stage validates row/store counts
against `config.yaml` (`data.expected_rows`, `data.expected_stores`).

## Processing

| File | Produced by | Description |
|---|---|---|
| `raw/*.csv` | `download` | verbatim mirror files |
| `cache/base.parquet` | `prepare` | cleaned + merged with store metadata |
| `cache/features.parquet` | `features` | complete daily grid (33,121 gap days imputed as NaN), ~47 engineered features, split labels |

Only rows with `Open == 1` and a known `Sales` value are modelled
(844,392 store-days: train 759,901 / valid 25,880 / test 58,611).
