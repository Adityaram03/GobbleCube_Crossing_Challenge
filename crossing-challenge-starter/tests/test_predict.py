"""Submission contract tests. These validate SHAPE, not quality.

Run:
    pytest tests/
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from predict import predict


HERE = Path(__file__).parent.parent
HORIZON_KEYS = ["bbox_500ms", "bbox_1000ms", "bbox_1500ms", "bbox_2000ms"]


def _synthetic_request(**over) -> dict:
    req = dict(
        ped_id="test000000ab",
        frame_w=1920,
        frame_h=1080,
        time_of_day="daytime",
        weather="clear",
        location="street",
        ego_available=True,
        bbox_history=[[100.0 + i * 2, 200.0, 180.0 + i * 2, 380.0] for i in range(16)],
        ego_speed_history=[5.0] * 16,
        ego_yaw_history=[0.0] * 16,
        requested_at_frame=100,
    )
    req.update(over)
    return req


def test_model_pkl_exists():
    assert (HERE / "model.pkl").exists(), "Run `python baseline.py` before testing."


def test_predict_returns_required_keys():
    out = predict(_synthetic_request())
    assert "intent" in out
    for h in HORIZON_KEYS:
        assert h in out


def test_intent_is_probability():
    out = predict(_synthetic_request())
    assert isinstance(out["intent"], float)
    assert 0.0 <= out["intent"] <= 1.0


def test_bbox_is_4_floats():
    out = predict(_synthetic_request())
    for h in HORIZON_KEYS:
        bbox = out[h]
        assert len(bbox) == 4
        for v in bbox:
            assert isinstance(v, (int, float))
            assert np.isfinite(v)


def test_missing_ego_handled():
    # JAAD-style: ego_available=False, histories are zeros
    req = _synthetic_request(
        ego_available=False,
        ego_speed_history=[0.0] * 16,
        ego_yaw_history=[0.0] * 16,
    )
    out = predict(req)
    assert 0.0 <= out["intent"] <= 1.0


def test_zero_velocity_bbox_is_finite():
    # Pedestrian standing still — past bboxes are identical, velocity is 0.
    req = _synthetic_request(bbox_history=[[100.0, 200.0, 180.0, 380.0]] * 16)
    out = predict(req)
    for h in HORIZON_KEYS:
        for v in out[h]:
            assert np.isfinite(v)


def test_nan_in_bbox_history_raises():
    bad = [[100.0, 200.0, 180.0, 380.0]] * 16
    bad[4][0] = float("nan")
    req = _synthetic_request(bbox_history=bad)
    # Either the model raises, or it must not return NaN. Silent NaN in the
    # output corrupts the grader's leaderboard. Pick one.
    try:
        out = predict(req)
    except Exception:
        return
    for v in [out["intent"]] + [c for h in HORIZON_KEYS for c in out[h]]:
        assert np.isfinite(v), "predict() must not return NaN on NaN input"


def test_row_order_preserved_on_dev():
    data = HERE / "data" / "dev.parquet"
    if not data.exists():
        pytest.skip("dev.parquet not built")
    df = pd.read_parquet(data).head(32).reset_index(drop=True)
    reqs = df[[
        "ped_id", "frame_w", "frame_h",
        "time_of_day", "weather", "location", "ego_available",
        "bbox_history", "ego_speed_history", "ego_yaw_history",
        "requested_at_frame",
    ]].to_dict("records")
    outs = [predict(r) for r in reqs]
    # Trivial check: intent vector length matches input row count and is finite
    intents = [o["intent"] for o in outs]
    assert len(intents) == len(df)
    assert all(np.isfinite(i) for i in intents)
