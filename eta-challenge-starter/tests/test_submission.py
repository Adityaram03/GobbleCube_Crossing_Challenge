"""Smoke tests for your submission.

Run: python -m pytest tests/

These do NOT verify prediction quality — they only check that your submission
interface matches what the grader expects. Run `python grade.py` for a real
MAE check on the Dev set.
"""

import pytest

from predict import predict


def _request(pz=100, dz=200, ts="2024-02-14T08:30:00", pc=1):
    return {
        "pickup_zone": pz,
        "dropoff_zone": dz,
        "requested_at": ts,
        "passenger_count": pc,
    }


def test_returns_float():
    assert isinstance(predict(_request()), float)


def test_positive_duration():
    assert predict(_request()) > 0, "duration must be positive"


@pytest.mark.parametrize("zone", [1, 132, 265])
def test_accepts_edge_zones(zone):
    assert isinstance(predict(_request(pz=zone, dz=zone)), float)


def test_varies_with_time_of_day():
    morning = predict(_request(ts="2024-02-14T08:00:00"))
    night = predict(_request(ts="2024-02-14T23:00:00"))
    assert morning != night, "predictions should vary by time of day"


@pytest.mark.parametrize("count", [1, 2, 4, 6])
def test_accepts_passenger_counts(count):
    assert isinstance(predict(_request(pc=count)), float)


def test_accepts_iso_with_seconds():
    assert isinstance(
        predict(_request(ts="2024-02-14T08:30:45")), float
    )
