"""Minimal synthetic-request smoke test used at Docker build time.

If this script fails, the built image will not pass the grader contract.
Candidates: leave this alone unless your submission genuinely needs a
different synthetic shape.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make this script runnable from any CWD: add the repo root (parent of
# `tests/`) to sys.path so `from predict import predict` resolves whether
# we're invoked as `python tests/smoke.py` or as `python smoke.py` from
# inside Docker.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from predict import predict


def main() -> None:
    req = {
        "ped_id": "smoke0000test",
        "frame_w": 1920,
        "frame_h": 1080,
        "time_of_day": "",
        "weather": "",
        "location": "",
        "ego_available": True,
        "bbox_history": [[100.0 + i * 2, 200.0, 180.0 + i * 2, 380.0] for i in range(16)],
        "ego_speed_history": [5.0] * 16,
        "ego_yaw_history": [0.0] * 16,
        "requested_at_frame": 100,
    }
    out = predict(req)
    assert "intent" in out and 0.0 <= out["intent"] <= 1.0, out
    for h in ("bbox_500ms", "bbox_1000ms", "bbox_1500ms", "bbox_2000ms"):
        assert h in out and len(out[h]) == 4, out


if __name__ == "__main__":
    main()
