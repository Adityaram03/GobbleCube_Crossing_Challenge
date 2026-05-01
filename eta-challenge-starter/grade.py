#!/usr/bin/env python
"""Scoring harness — mirrors Gobblecube's grader.

Two modes:

  Local dev grading:
      python grade.py
          Reads data/dev.parquet, prints MAE on stdout.

  Grader mode (used inside the Docker container by Gobblecube):
      python grade.py <input_parquet> <output_csv>
          Reads requests from input_parquet (duration_seconds not required),
          writes one prediction per row to output_csv.
          Gobblecube computes MAE server-side against held-out truth.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from predict import predict

DATA_DIR = Path(__file__).parent / "data"
REQUEST_FIELDS = ["pickup_zone", "dropoff_zone", "requested_at", "passenger_count"]


def run(input_path: Path, output_path: Path | None, sample_n: int | None = None) -> None:
    df = pd.read_parquet(input_path)
    if sample_n is not None and len(df) > sample_n:
        df = df.sample(n=sample_n, random_state=42).reset_index(drop=True)
    print(f"Predicting {len(df):,} rows from {input_path.name}...", file=sys.stderr)

    preds = np.empty(len(df), dtype=np.float64)
    records = df[REQUEST_FIELDS].to_dict("records")
    for i, req in enumerate(records):
        preds[i] = predict(req)

    if output_path is not None:
        # Echo the input's row_idx so the grader can verify row order on
        # eval. Local Dev parquet has no row_idx — synthesize one from the
        # current row position.
        if "row_idx" in df.columns:
            row_idx = df["row_idx"].to_numpy()
        else:
            row_idx = np.arange(len(df), dtype=np.int64)
        pd.DataFrame({"row_idx": row_idx, "prediction": preds}).to_csv(output_path, index=False)
        print(f"Wrote {len(preds):,} predictions to {output_path}", file=sys.stderr)
        return

    if "duration_seconds" not in df.columns:
        raise SystemExit(
            "Local grading needs a `duration_seconds` column in the parquet."
        )
    truth = df["duration_seconds"].to_numpy()
    mae = float(np.mean(np.abs(preds - truth)))
    if not np.isfinite(mae):
        raise SystemExit(f"Non-finite MAE ({mae}) — predictions contain NaN/Inf.")
    print(f"MAE: {mae:.1f} seconds")


def main(argv: list[str]) -> None:
    if len(argv) == 1:
        # Local: sample 50k rows of Dev for fast feedback. Deterministic seed
        # so the printed MAE is stable across runs. Matches Eval set size.
        run(DATA_DIR / "dev.parquet", None, sample_n=50_000)
    elif len(argv) == 3:
        # Grader mode: no sampling, predict every row in the input parquet.
        run(Path(argv[1]), Path(argv[2]))
    else:
        print(
            "Usage:\n"
            "  python grade.py                              # local Dev grading (50k sample)\n"
            "  python grade.py <input.parquet> <output.csv>  # grader mode (no sampling)",
            file=sys.stderr,
        )
        sys.exit(2)


if __name__ == "__main__":
    main(sys.argv)
