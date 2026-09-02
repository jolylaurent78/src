import pytest
from PIL import Image

from src.assembleur_geo_map_view import CalibratedGeoMap


def test_legacy_normalization_uses_an_upward_world_y_axis():
    raw = {
        "affineWorldToLambertKm": [1, 0, 0, 0, 1, 0],
        "bgWorldRectAtCalibration": {"x0": 10, "y0": 20, "w": 100, "h": 200},
    }
    normalized = CalibratedGeoMap._normalize_calibration(raw, (1000, 2000))
    calibrated = CalibratedGeoMap("legacy", Image.new("RGB", (1000, 2000)), normalized)

    assert calibrated.pixel_to_lambert(0, 0) == pytest.approx((10_000, 220_000))
    assert calibrated.pixel_to_lambert(0, 2000) == pytest.approx((10_000, 20_000))


def test_modern_a_offset_calibration_is_returned_unchanged():
    raw = {"projection": "EPSG:2154", "A": [[1, 0], [0, 1]], "offset": [3, 4]}

    assert CalibratedGeoMap._normalize_calibration(raw, (100, 100)) is raw


def test_modern_a_offset_has_priority_over_legacy_fields_after_recalibration():
    raw = {
        "A": [[1, 0], [0, 1]], "offset": [3, 4],
        "affineWorldToLambertKm": [0, 0, 0, 0, 0, 0],
        "bgWorldRectAtCalibration": {"x0": 0, "y0": 0, "w": 1, "h": 1},
    }

    assert CalibratedGeoMap._normalize_calibration(raw, (100, 100)) is raw

