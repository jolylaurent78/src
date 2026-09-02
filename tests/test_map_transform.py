from dataclasses import dataclass
import math

import pytest

from src.assembleur_catalogue import WorldRect
from src.assembleur_map_transform import (
    MapTransform,
    scale_factor_for_world_rect,
    validate_scale_factor,
    validate_world_rect,
    world_rect_for_scale,
)


@dataclass
class _CalibratedMap:
    image_size: tuple[int, int] = (200, 100)

    def lambert_to_pixel(self, x_m: float, y_m: float) -> tuple[float, float]:
        return x_m / 10.0, y_m / 10.0

    def pixel_to_lambert(self, x_px: float, y_px: float) -> tuple[float, float]:
        return x_px * 10.0, y_px * 10.0


def test_pixel_corners_center_and_axis_orientation_are_explicit():
    transform = MapTransform(_CalibratedMap(), WorldRect(-200, -100, 400, 200))

    assert transform.pixel_to_world(0, 0) == (-200, 100)
    assert transform.pixel_to_world(200, 0) == (200, 100)
    assert transform.pixel_to_world(0, 100) == (-200, -100)
    assert transform.pixel_to_world(100, 50) == (0, 0)
    assert transform.pixel_to_world(10, 0)[1] > transform.pixel_to_world(10, 10)[1]
    assert transform.world_to_pixel(0, 0) == (100, 50)
    assert transform.world_to_pixel(*transform.pixel_to_world(123.4, 67.8)) == pytest.approx((123.4, 67.8))


def test_lambert_and_world_round_trips_are_stable():
    transform = MapTransform(_CalibratedMap(), WorldRect(-400, 200, 400, 200))
    lambert = (1500.25, -200.5)
    world = transform.lambert_to_world(*lambert)

    assert transform.world_to_lambert(*world) == pytest.approx(lambert, abs=1e-9)
    assert transform.lambert_to_world(*transform.world_to_lambert(123.0, 321.0)) == pytest.approx(
        (123.0, 321.0), abs=1e-9
    )


@pytest.mark.parametrize(
    "value",
    [
        WorldRect(0, 0, 0, 1),
        WorldRect(0, 0, 1, math.inf),
        WorldRect(True, 0, 1, 1),
        None,
    ],
)
def test_invalid_world_rects_are_rejected(value):
    with pytest.raises(ValueError):
        validate_world_rect(value)


@pytest.mark.parametrize("value", [0, -1, math.nan, math.inf, True, "12"])
def test_invalid_scale_factors_are_rejected(value):
    with pytest.raises(ValueError):
        validate_scale_factor(value)


def test_transform_rejects_anisotropic_world_rect():
    with pytest.raises(ValueError, match="ratio intrinsèque"):
        MapTransform(_CalibratedMap(), WorldRect(0, 0, 300, 100))


def test_scale_change_derives_an_homothetic_world_rect():
    reference = WorldRect(5, 7, 400, 200)

    resized = world_rect_for_scale(
        reference,
        12.0,
        18.0,
        image_aspect_ratio=2.0,
    )

    assert resized == WorldRect(5, 7, 600, 300)
    assert scale_factor_for_world_rect(resized, reference, 12.0) == pytest.approx(18.0)
