#!/usr/bin/env python
"""Scoring harness — mirrors Gobblecube's grader.

Two modes:

  Local dev grading:
      python grade.py
          Reads data/dev.parquet, prints the composite score.

  Grader mode (used inside the Docker container by Gobblecube):
      python grade.py <input_parquet> <output_csv>
          Reads requests from input_parquet (targets not required),
          writes one row per input: intent + 4 bboxes (flattened).
          Gobblecube computes the score server-side.

Scoring formula:
    composite = 0.5 * (BCE / BCE_FLOOR)  +  0.5 * (mean_pixel_ADE / ADE_FLOOR)

Floor constants are measured on the Eval set:
    BCE_FLOOR = H(p_eval)                -- entropy of class prior on Eval
    ADE_FLOOR = zero-velocity mean ADE   -- predict current bbox for all horizons

A zero-work submission (class prior + zero velocity) scores exactly 1.0
on Eval by construction. On Dev, the same floors apply but Dev's
positive rate is slightly different, so zero-work drifts modestly.
Lower is
better. The numbers printed here must match what the real grader prints on
the same predictions (to the last digit) — this file's scoring logic is
held in lock-step with the grader's `score.py`.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from predict import predict

DATA = Path(__file__).parent / "data"
REQUEST_FIELDS = [
    "ped_id", "frame_w", "frame_h",
    "time_of_day", "weather", "location", "ego_available",
    "bbox_history", "ego_speed_history", "ego_yaw_history",
    "requested_at_frame",
]
HORIZONS = ["bbox_500ms", "bbox_1000ms", "bbox_1500ms", "bbox_2000ms"]
# The grader verifies the row order of predictions against the input by
# comparing this column. Don't drop it — your Docker image will be rejected.
OUT_COLS = ["ped_id", "intent"] + [f"{h}_{c}" for h in HORIZONS for c in ("x1", "y1", "x2", "y2")]

# KEEP IN SYNC with the grader's score.py — both files must use identical
# floors, clamp, and safe-intent/bbox logic, or local grade and real grade
# disagree. Floors are measured on the Eval set.
BCE_FLOOR = 0.2488
ADE_FLOOR = 49.80
BBOX_CLAMP = (-2000.0, 4000.0)


def _flatten(pred: dict, ped_id: str) -> list:
    row = [ped_id, float(pred["intent"])]
    for h in HORIZONS:
        row.extend(float(v) for v in pred[h])
    return row


def _safe_intent(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float64)
    arr = np.nan_to_num(arr, nan=0.5, posinf=1.0 - 1e-6, neginf=1e-6)
    return np.clip(arr, 1e-6, 1.0 - 1e-6)


def _safe_bbox(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float64)
    arr = np.nan_to_num(arr, nan=960.0, posinf=BBOX_CLAMP[1], neginf=BBOX_CLAMP[0])
    return np.clip(arr, *BBOX_CLAMP)


def score(preds_df: pd.DataFrame, truth_df: pd.DataFrame) -> dict[str, float]:
    """Both frames must be row-aligned. Returns the composite score + terms."""
    if "ped_id" in preds_df.columns and "ped_id" in truth_df.columns:
        if not (preds_df["ped_id"].to_numpy() == truth_df["ped_id"].to_numpy()).all():
            raise SystemExit(
                "Row-order mismatch: predictions and truth disagree on ped_id order."
            )
    ip = _safe_intent(preds_df["intent"].to_numpy())
    it = truth_df["will_cross_2s"].to_numpy(dtype=np.float64)
    bce = -float(np.mean(it * np.log(ip) + (1 - it) * np.log(1 - ip)))

    ades = []
    for h in HORIZONS:
        pcols = [f"{h}_x1", f"{h}_y1", f"{h}_x2", f"{h}_y2"]
        pb = _safe_bbox(preds_df[pcols].to_numpy())
        tb = np.stack([np.asarray(x, dtype=np.float64) for x in truth_df[h].to_numpy()])
        pcx = (pb[:, 0] + pb[:, 2]) * 0.5
        pcy = (pb[:, 1] + pb[:, 3]) * 0.5
        tcx = (tb[:, 0] + tb[:, 2]) * 0.5
        tcy = (tb[:, 1] + tb[:, 3]) * 0.5
        ades.append(float(np.hypot(pcx - tcx, pcy - tcy).mean()))
    mean_ade = float(np.mean(ades))

    composite = 0.5 * (bce / BCE_FLOOR) + 0.5 * (mean_ade / ADE_FLOOR)
    return {
        "score": composite,
        "intent_term": bce / BCE_FLOOR,
        "traj_term": mean_ade / ADE_FLOOR,
        "intent_bce": bce,
        "mean_ade_px": mean_ade,
    }


def run(input_path: Path, output_path: Path | None, sample_n: int | None = None) -> None:
    df = pd.read_parquet(input_path)
    if sample_n is not None and len(df) > sample_n:
        df = df.sample(n=sample_n, random_state=42).reset_index(drop=True)
    print(f"Predicting {len(df):,} rows from {input_path.name}...", file=sys.stderr)

    records = df[REQUEST_FIELDS].to_dict("records")
    flat = [_flatten(predict(r), r["ped_id"]) for r in records]
    preds_df = pd.DataFrame(flat, columns=OUT_COLS)

    if output_path is not None:
        preds_df.to_csv(output_path, index=False)
        print(f"Wrote {len(preds_df):,} predictions to {output_path}", file=sys.stderr)
        return

    if "will_cross_2s" not in df.columns:
        raise SystemExit("Local grading needs targets in the parquet.")
    s = score(preds_df, df)
    print(
        f"Score: {s['score']:.4f}   "
        f"(intent_term {s['intent_term']:.3f}, traj_term {s['traj_term']:.3f}; "
        f"BCE {s['intent_bce']:.4f}, ADE {s['mean_ade_px']:.1f} px)"
    )


def main(argv: list[str]) -> None:
    if len(argv) == 1:
        run(DATA / "dev.parquet", None, sample_n=5000)
    elif len(argv) == 3:
        run(Path(argv[1]), Path(argv[2]))
    else:
        print(
            "Usage:\n"
            "  python grade.py                              # local Dev grading (5k sample)\n"
            "  python grade.py <input.parquet> <output.csv>  # grader mode",
            file=sys.stderr,
        )
        sys.exit(2)


if __name__ == "__main__":
    main(sys.argv)
