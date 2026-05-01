# Data schema

Every row of `train.parquet` / `dev.parquet` is one **prediction window**:
16 frames of history at 15 Hz (≈1.07 s of past) plus the targets the
model should predict.

## Request fields (inputs to `predict()`)

| Column                | Dtype            | Meaning |
|-----------------------|------------------|---------|
| `ped_id`              | str              | Opaque 12-char token, stable within the dataset, disjoint across splits. Not a natural key — don't try to decode it. |
| `frame_w`, `frame_h`  | int              | Frame dimensions (px). Always 1920 × 1080 in v1. |
| `time_of_day`         | str              | Optional: `"daytime"`, `"nighttime"`, or `""`. |
| `weather`             | str              | Optional: `"clear"`, `"cloudy"`, `"rain"`, `"snow"`, or `""`. |
| `location`            | str              | Optional scene type (`"plaza"`, `"street"`…) or `""`. |
| `ego_available`       | bool             | `True` when `ego_speed_history` / `ego_yaw_history` reflect real OBD data. `False` when they're zero-filled (no vehicle telemetry available). |
| `bbox_history`        | list[16] of [4 float] | Past 16 bboxes `[x1, y1, x2, y2]` in pixels, oldest → current. 15 Hz. |
| `ego_speed_history`   | list[16] float   | Ego vehicle speed (m/s) per past frame. Zeros when `ego_available=False`. |
| `ego_yaw_history`     | list[16] float   | Ego gyro-Z (yaw rate, rad/s). Zeros when `ego_available=False`. |
| `requested_at_frame`  | int              | Native-30fps frame id of the current (most-recent) observation. |

## Target fields (what you predict; present in Dev, not in Eval input)

| Column            | Dtype      | Meaning |
|-------------------|------------|---------|
| `will_cross_2s`   | bool       | `True` iff any frame in the next 2 s has `cross == "crossing"`. |
| `bbox_500ms`      | [4 float]  | Ground-truth bbox at +0.5 s (native frame + 16). |
| `bbox_1000ms`     | [4 float]  | +1.0 s (native frame + 30). |
| `bbox_1500ms`     | [4 float]  | +1.5 s (native frame + 46). |
| `bbox_2000ms`     | [4 float]  | +2.0 s (native frame + 60). |

## Notes

- **Windows only exist when the pedestrian is currently not crossing.**
  We drop windows whose current frame has `cross == "crossing"`, so the
  task is always "predict whether / when / where this pedestrian will
  cross," not "recognize that they already are."
- **Current-frame occlusion="full" windows are dropped.** The model
  always sees a visible pedestrian at the moment of prediction.
- **Video-disjoint split.** No video appears in more than one split,
  which also means no pedestrian does. This holds out the scene, the
  intersection, the lighting, and the ego-vehicle trajectory — not just
  the pedestrian's identity. A model that learns "this specific
  intersection has a lot of jaywalkers" gets no free points on Eval.
- **Class imbalance is real.** Roughly 7 % of windows are
  `will_cross_2s=True` (a couple of points each way by split). Design
  for it — log-loss punishes over-confident minority-class predictions
  harder than you'd expect.
