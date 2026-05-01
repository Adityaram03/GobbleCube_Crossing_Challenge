"""Your submission entry point — replace the baseline model here.

Contract (do NOT change the signature):

    predict(request: dict) -> dict

Input: one request dict (keys below). Output: a dict with intent probability
and 4 future bounding boxes.

Request keys (all required):
    ped_id                str       opaque token, stable within the dataset
    frame_w, frame_h      int
    time_of_day, weather, location   str  (may be empty strings)
    ego_available         bool      True when ego speed/yaw history is valid
    bbox_history          list[16][4]  past bboxes at 15 Hz, oldest → current
                                       each bbox is [x1, y1, x2, y2] in pixels
    ego_speed_history     list[16]  past ego speeds (m/s); zeros if unavailable
    ego_yaw_history       list[16]  past yaw rates (rad/s); zeros if unavailable
    requested_at_frame    int       native-30fps frame id of current observation

Output keys (all required):
    intent                float     P(crossing within next 2s), in [0, 1]
    bbox_500ms            list[4]   predicted bbox at +0.5 s
    bbox_1000ms           list[4]   predicted bbox at +1.0 s
    bbox_1500ms           list[4]   predicted bbox at +1.5 s
    bbox_2000ms           list[4]   predicted bbox at +2.0 s

The grader calls predict() once per request, row-by-row. Prediction order
must match input order.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

MODEL_PATH = Path(__file__).parent / "model.pkl"
HORIZONS_FRAMES = [8, 15, 23, 30]   # at 15 Hz → 0.5, 1.0, 1.5, 2.0 s
HORIZON_KEYS = ["bbox_500ms", "bbox_1000ms", "bbox_1500ms", "bbox_2000ms"]

_cached_model = None


def _load_model():
    global _cached_model
    if _cached_model is None:
        with open(MODEL_PATH, "rb") as f:
            _cached_model = pickle.load(f)
    return _cached_model


def _as_2d(x) -> np.ndarray:
    """Coerce list-of-lists / object-array-of-arrays to (N, 4) float64."""
    return np.stack([np.asarray(r, dtype=np.float64) for r in x])


def _engineered_features(req: dict) -> np.ndarray:
    """Mirrors baseline.py's feature builder so the XGBoost model sees the same
    layout at inference as at training. Keep this in lock-step with baseline.py.
    """
    hist = _as_2d(req["bbox_history"])  # (16, 4)
    cx = (hist[:, 0] + hist[:, 2]) * 0.5
    cy = (hist[:, 1] + hist[:, 3]) * 0.5
    w = hist[:, 2] - hist[:, 0]
    h = hist[:, 3] - hist[:, 1]
    vx = np.diff(cx)
    vy = np.diff(cy)

    ego_s = np.asarray(req["ego_speed_history"], dtype=np.float64)
    ego_y = np.asarray(req["ego_yaw_history"], dtype=np.float64)

    fw = float(req["frame_w"])
    fh = float(req["frame_h"])
    feats = [
        cx[-1] / fw,
        cy[-1] / fh,
        w[-1] / fw,
        h[-1] / fh,
        vx[-4:].mean() / fw,
        vy[-4:].mean() / fh,
        vx.std() / fw,                            # normalized per frame width
        vy.std() / fh,                            # normalized per frame height
        (h / (w + 1e-6)).mean(),                  # aspect ratio (tallness)
        float(req["ego_available"]),
        ego_s.mean(), ego_s[-1], ego_s.max(),
        ego_y.mean(), ego_y[-1], np.abs(ego_y).max(),
        1.0 if req.get("time_of_day") == "daytime" else 0.0,
        1.0 if req.get("time_of_day") == "nighttime" else 0.0,
        1.0 if req.get("weather") == "rain" else 0.0,
        1.0 if req.get("weather") == "snow" else 0.0,
    ]
    # The richer motion-history block helps intent distinguish steady walkers,
    # slowing pedestrians, and camera-motion artifacts.
    feats.extend(_compact_trajectory_features(req).tolist())
    return np.asarray(feats, dtype=np.float32)


def _compact_trajectory_features(req: dict) -> np.ndarray:
    hist = _as_2d(req["bbox_history"])  # (16, 4)
    cx = (hist[:, 0] + hist[:, 2]) * 0.5
    cy = (hist[:, 1] + hist[:, 3]) * 0.5
    w = hist[:, 2] - hist[:, 0]
    h = hist[:, 3] - hist[:, 1]
    vx = np.diff(cx)
    vy = np.diff(cy)
    ax = np.diff(vx)
    ay = np.diff(vy)

    ego_s = np.asarray(req["ego_speed_history"], dtype=np.float64)
    ego_y = np.asarray(req["ego_yaw_history"], dtype=np.float64)
    fw = float(req["frame_w"])
    fh = float(req["frame_h"])

    feats: list[float] = []
    feats.extend((cx[-8:] / fw).tolist())
    feats.extend((cy[-8:] / fh).tolist())
    feats.extend((w[-8:] / fw).tolist())
    feats.extend((h[-8:] / fh).tolist())
    feats.extend((vx[-8:] / fw).tolist())
    feats.extend((vy[-8:] / fh).tolist())
    feats.extend((ax[-6:] / fw).tolist())
    feats.extend((ay[-6:] / fh).tolist())
    feats.extend([
        vx[-4:].mean() / fw,
        vy[-4:].mean() / fh,
        vx[-8:].mean() / fw,
        vy[-8:].mean() / fh,
        vx[-4:].std() / fw,
        vy[-4:].std() / fh,
        ax[-4:].mean() / fw,
        ay[-4:].mean() / fh,
        w[-1] / fw,
        h[-1] / fh,
        (w[-1] - w[-5]) / fw,
        (h[-1] - h[-5]) / fh,
        (h[-1] / (w[-1] + 1e-6)),
        float(req["ego_available"]),
        ego_s.mean(), ego_s[-1], ego_s.max(), ego_s[-1] - ego_s[0],
        ego_y.mean(), ego_y[-1], np.abs(ego_y).max(),
        1.0 if req.get("time_of_day") == "daytime" else 0.0,
        1.0 if req.get("time_of_day") == "nighttime" else 0.0,
        1.0 if req.get("weather") == "rain" else 0.0,
        1.0 if req.get("weather") == "snow" else 0.0,
    ])
    return np.asarray(feats, dtype=np.float32)


def _trajectory_features(req: dict) -> np.ndarray:
    hist = _as_2d(req["bbox_history"])  # (16, 4)
    cx = (hist[:, 0] + hist[:, 2]) * 0.5
    cy = (hist[:, 1] + hist[:, 3]) * 0.5
    w = hist[:, 2] - hist[:, 0]
    h = hist[:, 3] - hist[:, 1]
    vx = np.diff(cx)
    vy = np.diff(cy)
    ax = np.diff(vx)
    ay = np.diff(vy)

    ego_s = np.asarray(req["ego_speed_history"], dtype=np.float64)
    ego_y = np.asarray(req["ego_yaw_history"], dtype=np.float64)
    fw = float(req["frame_w"])
    fh = float(req["frame_h"])

    feats: list[float] = []
    feats.extend((cx / fw).tolist())
    feats.extend((cy / fh).tolist())
    feats.extend((w / fw).tolist())
    feats.extend((h / fh).tolist())
    feats.extend((vx / fw).tolist())
    feats.extend((vy / fh).tolist())
    feats.extend((ax / fw).tolist())
    feats.extend((ay / fh).tolist())
    for n in (3, 5, 8, 15):
        feats.extend([
            vx[-n:].mean() / fw,
            vy[-n:].mean() / fh,
            vx[-n:].std() / fw,
            vy[-n:].std() / fh,
        ])
    feats.extend([
        ax[-5:].mean() / fw,
        ay[-5:].mean() / fh,
        ax[-10:].mean() / fw,
        ay[-10:].mean() / fh,
        w[-1] / fw,
        h[-1] / fh,
        (w[-1] - w[0]) / fw,
        (h[-1] - h[0]) / fh,
        (h[-1] / (w[-1] + 1e-6)),
        float(req["ego_available"]),
        ego_s.mean(), ego_s[-1], ego_s.max(), ego_s[-1] - ego_s[0],
        ego_y.mean(), ego_y[-1], np.abs(ego_y).max(),
    ])
    return np.asarray(feats, dtype=np.float32)


def _robust_cv_trajectory(req: dict) -> dict[str, list[float]]:
    hist = _as_2d(req["bbox_history"])  # (16, 4)
    cx = (hist[:, 0] + hist[:, 2]) * 0.5
    cy = (hist[:, 1] + hist[:, 3]) * 0.5
    w_last = hist[-1, 2] - hist[-1, 0]
    h_last = hist[-1, 3] - hist[-1, 1]
    vx_hist = np.diff(cx[-9:])  # last 8 intervals
    vy_hist = np.diff(cy[-9:])
    # Robust velocity estimate against noisy frame-to-frame jitter.
    vx = float(np.median(vx_hist))
    vy = float(np.median(vy_hist))

    # Cap extreme per-frame speeds from detector glitches.
    v_norm = float(np.hypot(vx, vy))
    max_v = 12.0
    if v_norm > max_v:
        scale = max_v / (v_norm + 1e-6)
        vx *= scale
        vy *= scale

    # If motion is clearly slowing, suppress long-horizon drift.
    recent_vx = float(np.mean(vx_hist[-3:]))
    recent_vy = float(np.mean(vy_hist[-3:]))
    older_vx = float(np.mean(vx_hist[:3]))
    older_vy = float(np.mean(vy_hist[:3]))
    recent_speed = float(np.hypot(recent_vx, recent_vy))
    older_speed = float(np.hypot(older_vx, older_vy)) + 1e-6
    slowdown_ratio = recent_speed / older_speed
    slowdown_factor = float(np.clip(0.55 + 0.45 * slowdown_ratio, 0.55, 1.08))

    cur_cx, cur_cy = float(cx[-1]), float(cy[-1])

    out: dict[str, list[float]] = {}
    for h, key in zip(HORIZONS_FRAMES, HORIZON_KEYS):
        # Mild horizon-dependent damping: near-baseline at short horizon,
        # reduced overshoot at +1.5s/+2.0s.
        damp = (1.0 - 0.0015 * max(h - 8, 0)) * slowdown_factor
        nx = cur_cx + (vx * damp) * h
        ny = cur_cy + (vy * damp) * h
        out[key] = [nx - w_last / 2, ny - h_last / 2, nx + w_last / 2, ny + h_last / 2]
    return out


def _learned_trajectory(req: dict, model: dict) -> dict[str, list[float]] | None:
    if "trajectory" not in model:
        return None

    hist = _as_2d(req["bbox_history"])
    cur_cx = float((hist[-1, 0] + hist[-1, 2]) * 0.5)
    cur_cy = float((hist[-1, 1] + hist[-1, 3]) * 0.5)
    w_last = float(hist[-1, 2] - hist[-1, 0])
    h_last = float(hist[-1, 3] - hist[-1, 1])
    fw = float(req["frame_w"])
    fh = float(req["frame_h"])

    feats = _trajectory_features(req).reshape(1, -1)
    if not np.isfinite(feats).all():
        feats = np.nan_to_num(feats, nan=0.0, posinf=1.0, neginf=-1.0)

    regressors = model["trajectory"]
    direct_pred = np.asarray([reg.predict(feats)[0] for reg in regressors], dtype=np.float64)
    if not np.isfinite(direct_pred).all():
        return None

    residual_pred = None
    if "trajectory_residual" in model:
        residual_pred = np.asarray([reg.predict(feats)[0] for reg in model["trajectory_residual"]], dtype=np.float64)
        if not np.isfinite(residual_pred).all():
            residual_pred = None

    extra_pred = None
    if "trajectory_extra" in model:
        extra_pred = np.asarray(model["trajectory_extra"].predict(feats)[0], dtype=np.float64)
        if not np.isfinite(extra_pred).all():
            extra_pred = None

    cv = _robust_cv_trajectory(req)
    ensemble = model.get("trajectory_ensemble")
    blend = model.get("trajectory_blend", [1.0] * len(HORIZON_KEYS))
    out: dict[str, list[float]] = {}
    for i, key in enumerate(HORIZON_KEYS):
        direct_cx = cur_cx + float(direct_pred[2 * i]) * fw
        direct_cy = cur_cy + float(direct_pred[2 * i + 1]) * fh
        cv_box = cv[key]
        cv_cx = (cv_box[0] + cv_box[2]) * 0.5
        cv_cy = (cv_box[1] + cv_box[3]) * 0.5
        if ensemble is not None and residual_pred is not None:
            residual_cx = cv_cx + float(residual_pred[2 * i]) * fw
            residual_cy = cv_cy + float(residual_pred[2 * i + 1]) * fh
            weights = [float(v) for v in ensemble[i]]
            w_direct, w_residual, w_cv = weights[:3]
            nx = w_direct * direct_cx + w_residual * residual_cx + w_cv * cv_cx
            ny = w_direct * direct_cy + w_residual * residual_cy + w_cv * cv_cy
            if len(weights) == 4 and extra_pred is not None:
                extra_cx = cur_cx + float(extra_pred[2 * i]) * fw
                extra_cy = cur_cy + float(extra_pred[2 * i + 1]) * fh
                w_extra = weights[3]
                nx += w_extra * extra_cx
                ny += w_extra * extra_cy
        else:
            alpha = float(blend[i])
            nx = alpha * direct_cx + (1.0 - alpha) * cv_cx
            ny = alpha * direct_cy + (1.0 - alpha) * cv_cy
        out[key] = [nx - w_last / 2, ny - h_last / 2, nx + w_last / 2, ny + h_last / 2]
    return out


def _calibrate_intent(intent_prob: float, model: dict) -> float:
    if "intent_calibration" not in model:
        return intent_prob
    scale, bias = model["intent_calibration"]
    p = float(np.clip(intent_prob, 1e-6, 1.0 - 1e-6))
    logit = np.log(p / (1.0 - p))
    calibrated = 1.0 / (1.0 + np.exp(-(float(scale) * logit + float(bias))))
    return float(np.clip(calibrated, 0.0, 1.0))


def _predict_intent(feats: np.ndarray, model: dict) -> float:
    if "intent_models" not in model:
        return float(model["intent"].predict_proba(feats)[0, 1])

    probs = np.asarray([m.predict_proba(feats)[0, 1] for m in model["intent_models"]], dtype=np.float64)
    if not np.isfinite(probs).all():
        probs = np.nan_to_num(probs, nan=0.5, posinf=1.0, neginf=0.0)
    weights = np.asarray(model.get("intent_weights", [1.0 / len(probs)] * len(probs)), dtype=np.float64)
    weights = weights / max(weights.sum(), 1e-12)
    return float(np.clip(probs @ weights, 0.0, 1.0))


def predict(request: dict) -> dict:
    model = _load_model()
    feats = _engineered_features(request).reshape(1, -1)
    if not np.isfinite(feats).all():
        feats = np.nan_to_num(feats, nan=0.0, posinf=1.0, neginf=-1.0)
    intent_prob = _predict_intent(feats, model)
    if not np.isfinite(intent_prob):
        intent_prob = 0.5
    intent_prob = _calibrate_intent(intent_prob, model)

    out = _learned_trajectory(request, model)
    if out is None:
        out = _robust_cv_trajectory(request)
    for k in HORIZON_KEYS:
        out[k] = [float(v) if np.isfinite(v) else 0.0 for v in out[k]]
    out["intent"] = intent_prob
    return out
