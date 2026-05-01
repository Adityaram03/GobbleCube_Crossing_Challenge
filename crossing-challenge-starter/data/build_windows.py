#!/usr/bin/env python
"""Slice per-frame tracklets into prediction windows + video-level split.

Output (each row = one window):
    request fields (input to predict):
        ped_id             str    hashed 12-char token, globally unique
        frame_w, frame_h   int    frame dimensions
        time_of_day, weather, location  str  (JAAD only; empty for PIE)
        ego_available      bool   True when OBD ego motion is valid
        bbox_history       list[16][4]   past bboxes at 15 Hz (oldest → current)
        ego_speed_history  list[16]      past OBD speeds m/s (0.0 if unavailable)
        ego_yaw_history    list[16]      past yaw rates (0.0 if unavailable)
        requested_at_frame int    native 30fps frame id of current observation

    targets (held out of request; only in dev/eval truth):
        will_cross_2s      bool   any crossing event in next 2s
        bbox_500ms         list[4]    +8 frames @ 15Hz
        bbox_1000ms        list[4]    +15 frames
        bbox_1500ms        list[4]    +23 frames
        bbox_2000ms        list[4]    +30 frames

Splits by VIDEO ID (no video appears in multiple splits — and since peds are
one-to-one with a video, no pedestrian appears either). Stratified by source
× "video has any positive window" to keep class ratios matched. This is
stricter than ped-level split: it also holds out scene / intersection /
lighting / ego-vehicle context.

Pedestrian IDs are hashed with `HASH_SALT` to prevent candidates from
re-identifying eval peds in the raw JAAD/PIE XMLs. The `source` field is
dropped — `ego_available` carries the same training signal without
exposing which upstream dataset produced the row.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
TRACKLETS = ROOT / "tracklets_raw.parquet"

PAST_LEN = 16          # frames at 15 Hz
FUTURE_LEN = 30        # 2 s at 15 Hz
HORIZON_IDX = [8, 15, 23, 30]   # +0.53, +1.0, +1.53, +2.0 s (at 15 Hz)
WINDOW_LEN = PAST_LEN + FUTURE_LEN
STRIDE = 5             # every 0.33 s at 15 Hz

SEED = 2024
# With only ~400 videos and a 80/10/10 split, 10 % per holdout is only ~40
# videos — too few to stay representative across sources × size × pos-rate
# variance. 70/15/15 keeps ~60 videos per holdout, which settles the
# positive-rate spread to within ~1 % between dev and eval.
SPLIT_RATIOS = {"train": 0.70, "dev": 0.15, "eval": 0.15}

# Internal salt — read from a gitignored sibling file so the public starter
# repo never contains it. A candidate with the shipped parquets but no salt
# cannot reverse the ped_id → upstream XML mapping.
_SALT_FILE = ROOT.parent / ".hash_salt"


def _load_salt() -> str:
    if not _SALT_FILE.exists():
        raise SystemExit(
            f"Missing {_SALT_FILE}. Create it with 16 bytes of hex "
            "(e.g. `python -c \"import secrets; print(secrets.token_hex(16))\" > .hash_salt`). "
            "Never commit this file."
        )
    s = _SALT_FILE.read_text().strip()
    if len(s) < 32:
        raise SystemExit(f"{_SALT_FILE} looks too short; expected ≥32 hex chars")
    return s


def _hash_id(raw: str, _cache: dict[str, str] = {}) -> str:
    salt = _cache.setdefault("salt", _load_salt())
    return hashlib.sha256((salt + raw).encode("utf-8")).hexdigest()[:12]


def downsample_to_15hz(df: pd.DataFrame) -> pd.DataFrame:
    # JAAD and PIE are both 30 fps. Keep even-numbered frames to get 15 Hz.
    df = df[df["frame"] % 2 == 0].copy()
    df.sort_values(["ped_id", "frame"], inplace=True)
    return df.reset_index(drop=True)


def contiguous_runs(frames: np.ndarray) -> list[tuple[int, int]]:
    """Given a sorted array of native frame indices (all even, 15 Hz), return
    index ranges [start, end) of contiguous runs (frame_diff == 2)."""
    if len(frames) == 0:
        return []
    diffs = np.diff(frames)
    breaks = np.where(diffs != 2)[0]
    starts = np.concatenate(([0], breaks + 1))
    ends = np.concatenate((breaks + 1, [len(frames)]))
    return list(zip(starts.tolist(), ends.tolist()))


def build_windows(df: pd.DataFrame) -> list[dict]:
    windows: list[dict] = []
    drop_full_occlusion = 0
    drop_not_predictable = 0

    for ped_id, g in df.groupby("ped_id", sort=False):
        g = g.reset_index(drop=True)
        frames = g["frame"].to_numpy()
        for run_start, run_end in contiguous_runs(frames):
            if run_end - run_start < WINDOW_LEN:
                continue
            for w_start in range(run_start, run_end - WINDOW_LEN + 1, STRIDE):
                past = g.iloc[w_start : w_start + PAST_LEN]
                future = g.iloc[w_start + PAST_LEN : w_start + WINDOW_LEN]
                current = past.iloc[-1]

                if current["occlusion"] == "full":
                    drop_full_occlusion += 1
                    continue
                if current["cross"] not in ("not-crossing", "crossing-irrelevant"):
                    drop_not_predictable += 1
                    continue

                will_cross_2s = bool((future["cross"] == "crossing").any())

                bbox_hist = past[["x1", "y1", "x2", "y2"]].to_numpy().tolist()
                ego_speed_hist = past["ego_speed_ms"].fillna(0.0).to_numpy().tolist()
                ego_yaw_hist = past["ego_yaw_rate"].fillna(0.0).to_numpy().tolist()
                # ego_available reflects actual OBD-data presence, not dataset
                # provenance. A PIE window with fully-NaN OBD reads False.
                ego_available = bool(past["ego_speed_ms"].notna().all())

                # Horizon bboxes: index in the future slice is (h-1) because
                # h=8 means "8 frames into future" which is future.iloc[7].
                horizons = {}
                for h in HORIZON_IDX:
                    row = future.iloc[h - 1]
                    horizons[f"bbox_{h}"] = [row["x1"], row["y1"], row["x2"], row["y2"]]

                windows.append({
                    # Internal-only columns used for splitting + stratification;
                    # we drop them before writing the shipped parquets.
                    "_video_id": str(current["video_id"]),
                    "_source": current["source"],
                    "_raw_ped_id": str(ped_id),

                    "ped_id": _hash_id(str(ped_id)),
                    "frame_w": int(current["frame_w"]),
                    "frame_h": int(current["frame_h"]),
                    "time_of_day": current["time_of_day"] or "",
                    "weather": current["weather"] or "",
                    "location": current["location"] or "",
                    "ego_available": ego_available,
                    "bbox_history": bbox_hist,
                    "ego_speed_history": ego_speed_hist,
                    "ego_yaw_history": ego_yaw_hist,
                    "requested_at_frame": int(current["frame"]),
                    "will_cross_2s": will_cross_2s,
                    "bbox_500ms":  horizons["bbox_8"],
                    "bbox_1000ms": horizons["bbox_15"],
                    "bbox_1500ms": horizons["bbox_23"],
                    "bbox_2000ms": horizons["bbox_30"],
                })

    print(f"  dropped full-occlusion current frames: {drop_full_occlusion:,}")
    print(f"  dropped currently-crossing frames    : {drop_not_predictable:,}")
    return windows


def split_by_video(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Split windows so no video appears in more than one split.

    Greedy assignment per source: walk videos in decreasing total-window
    order, place each into whichever split is furthest behind on its
    JOINT (total-windows, positive-windows) quota. Balancing both quotas
    together keeps dev/eval positive rates within a point or two of train,
    which matters because the published BCE_FLOOR is measured on Eval.
    """
    rng = np.random.default_rng(SEED)
    vid_df = df.groupby("_video_id").agg(
        source=("_source", "first"),
        n_windows=("will_cross_2s", "size"),
        n_positive=("will_cross_2s", "sum"),
    ).reset_index()
    vid_df["n_positive"] = vid_df["n_positive"].astype(int)

    out: dict[str, list[str]] = {"train": [], "dev": [], "eval": []}

    for src in sorted(vid_df["source"].unique()):
        vids = vid_df[vid_df["source"] == src].copy()
        total_w = int(vids["n_windows"].sum())
        total_p = int(vids["n_positive"].sum())
        # Normalize each quota separately, then sum — a split behind on
        # *either* window count or positive count rises to the top.
        tgt_w = {k: total_w * r for k, r in SPLIT_RATIOS.items()}
        tgt_p = {k: max(total_p, 1) * r for k, r in SPLIT_RATIOS.items()}
        got_w = {k: 0 for k in SPLIT_RATIOS}
        got_p = {k: 0 for k in SPLIT_RATIOS}

        order = rng.permutation(len(vids))
        vids = vids.iloc[order].sort_values("n_windows", ascending=False, kind="stable")

        for _, v in vids.iterrows():
            # Shortfall as a fraction of target, summed across the two axes.
            # Picking the split with the largest shortfall keeps all three
            # splits moving toward their joint quota.
            deficits = {
                k: (tgt_w[k] - got_w[k]) / max(tgt_w[k], 1)
                + (tgt_p[k] - got_p[k]) / max(tgt_p[k], 1)
                for k in SPLIT_RATIOS
            }
            pick = max(deficits, key=deficits.get)
            out[pick].append(v["_video_id"])
            got_w[pick] += int(v["n_windows"])
            got_p[pick] += int(v["n_positive"])

    splits = {k: df[df["_video_id"].isin(set(v))].reset_index(drop=True) for k, v in out.items()}
    return splits


def main() -> None:
    df = pd.read_parquet(TRACKLETS)
    print(f"Loaded {len(df):,} frame-rows, {df['ped_id'].nunique():,} peds")

    df15 = downsample_to_15hz(df)
    print(f"After 15 Hz downsample: {len(df15):,} rows")

    print("\nBuilding windows...")
    windows = build_windows(df15)
    wdf = pd.DataFrame(windows)
    print(f"Built {len(wdf):,} windows from {wdf['_video_id'].nunique():,} videos")
    print(f"  will_cross_2s positive rate: {wdf['will_cross_2s'].mean():.3f}")
    print(f"  by source:\n{wdf.groupby('_source')['will_cross_2s'].agg(['count','mean'])}")

    print("\nSplitting by VIDEO (stratified by source × has_any_positive)...")
    splits = split_by_video(wdf)
    for name, s in splits.items():
        print(f"  {name:5s}: {len(s):,} windows, "
              f"{s['_video_id'].nunique():,} videos, "
              f"{s['ped_id'].nunique():,} hashed peds, "
              f"positive rate {s['will_cross_2s'].mean():.3f}")

    # Sanity: no video appears in multiple splits (and therefore no ped either)
    train_vids = set(splits["train"]["_video_id"])
    dev_vids = set(splits["dev"]["_video_id"])
    eval_vids = set(splits["eval"]["_video_id"])
    assert not (train_vids & dev_vids), "Video leakage train↔dev"
    assert not (train_vids & eval_vids), "Video leakage train↔eval"
    assert not (dev_vids & eval_vids), "Video leakage dev↔eval"

    # Strip internal-only columns before shipping
    internal = ["_video_id", "_source", "_raw_ped_id"]
    for name, s in splits.items():
        out = ROOT / f"{name}.parquet"
        s.drop(columns=internal).to_parquet(out, index=False)
        print(f"  wrote {out}")


if __name__ == "__main__":
    main()
