#!/usr/bin/env python
"""One-time download & cleanup of NYC TLC 2023 yellow-taxi data.

Produces:
    data/train.parquet       -- 11.5 months of 2023, ~37M trips after cleaning
    data/dev.parquet         -- last 2 weeks of 2023, ~1M trips (for local grading)
    data/sample_1M.parquet   -- 1M-row subset of train for fast iteration

The held-out Eval set (a 2024 slice) is kept by Gobblecube and never distributed.

Takes ~5 minutes on a fast connection, ~20 minutes on a slow one.
"""

from __future__ import annotations

from pathlib import Path
from urllib.request import urlretrieve

import pandas as pd

BASE_URL = "https://d37ci6vzurychx.cloudfront.net/trip-data"
MONTHS = [f"2023-{m:02d}" for m in range(1, 13)]

DATA_DIR = Path(__file__).parent
RAW_DIR = DATA_DIR / "raw"

CUTOFF = pd.Timestamp("2023-12-18")   # dev = last ~2 weeks of Dec
SAMPLE_SIZE = 1_000_000


def download_month(yyyymm: str) -> Path:
    RAW_DIR.mkdir(exist_ok=True)
    url = f"{BASE_URL}/yellow_tripdata_{yyyymm}.parquet"
    out = RAW_DIR / f"yellow_{yyyymm}.parquet"
    if out.exists():
        print(f"  cached   {out.name}")
        return out
    print(f"  fetching {url}")
    urlretrieve(url, out)
    return out


def clean(paths: list[Path]) -> pd.DataFrame:
    frames = []
    for p in paths:
        df = pd.read_parquet(
            p,
            columns=[
                "tpep_pickup_datetime",
                "tpep_dropoff_datetime",
                "PULocationID",
                "DOLocationID",
                "passenger_count",
            ],
        )
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)

    duration = (
        df["tpep_dropoff_datetime"] - df["tpep_pickup_datetime"]
    ).dt.total_seconds()

    clean_df = pd.DataFrame({
        "pickup_zone":      df["PULocationID"].astype("int32"),
        "dropoff_zone":     df["DOLocationID"].astype("int32"),
        "requested_at":     df["tpep_pickup_datetime"].dt.strftime("%Y-%m-%dT%H:%M:%S"),
        "passenger_count":  df["passenger_count"].fillna(1).astype("int8"),
        "duration_seconds": duration.astype("float64"),
        "_ts":              df["tpep_pickup_datetime"],
    })

    mask = (
        (clean_df["duration_seconds"] >= 30)
        & (clean_df["duration_seconds"] <= 3 * 3600)
        & (clean_df["pickup_zone"].between(1, 265))
        & (clean_df["dropoff_zone"].between(1, 265))
        & (clean_df["_ts"].dt.year == 2023)
    )
    return clean_df.loc[mask].reset_index(drop=True)


def split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = df[df["_ts"] < CUTOFF].drop(columns=["_ts"]).reset_index(drop=True)
    dev = df[df["_ts"] >= CUTOFF].drop(columns=["_ts"]).reset_index(drop=True)
    return train, dev


def main() -> None:
    print("Step 1: download monthly parquets")
    paths = [download_month(m) for m in MONTHS]

    print("\nStep 2: clean & combine")
    df = clean(paths)
    print(f"  cleaned: {len(df):,} trips")

    print("\nStep 3: train/dev split")
    train, dev = split(df)
    train.to_parquet(DATA_DIR / "train.parquet", index=False)
    dev.to_parquet(DATA_DIR / "dev.parquet", index=False)
    print(f"  train.parquet: {len(train):,} rows")
    print(f"  dev.parquet:   {len(dev):,} rows")

    print("\nStep 4: 1M-row training sample")
    sample = train.sample(n=min(SAMPLE_SIZE, len(train)), random_state=42)
    sample.reset_index(drop=True).to_parquet(
        DATA_DIR / "sample_1M.parquet", index=False
    )
    print(f"  sample_1M.parquet: {len(sample):,} rows")

    print("\nDone. Next: `python baseline.py`")


if __name__ == "__main__":
    main()
