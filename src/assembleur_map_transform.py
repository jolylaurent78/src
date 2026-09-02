"""Transformations pures entre Lambert-93, pixels d'une carte et monde scénario."""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real
from typing import TYPE_CHECKING

from src.assembleur_catalogue import WorldRect

if TYPE_CHECKING:
    from src.assembleur_geo_map_view import CalibratedGeoMap


def validate_finite_number(value: object, *, label: str, strictly_positive: bool = False) -> float:
    """Valide un nombre géométrique sans accepter les booléens."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{label} doit être un nombre.")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} doit être fini.")
    if strictly_positive and number <= 0:
        raise ValueError(f"{label} doit être strictement positif.")
    return number


def validate_world_rect(value: object, *, label: str = "world_rect") -> WorldRect:
    """Retourne une copie normalisée d'un rectangle monde valide."""
    if not isinstance(value, WorldRect):
        raise ValueError(f"{label} doit être un WorldRect.")
    return WorldRect(
        validate_finite_number(value.x0, label=f"{label}.x0"),
        validate_finite_number(value.y0, label=f"{label}.y0"),
        validate_finite_number(value.w, label=f"{label}.w", strictly_positive=True),
        validate_finite_number(value.h, label=f"{label}.h", strictly_positive=True),
    )


def validate_scale_factor(value: object, *, label: str = "scale_factor") -> float:
    return validate_finite_number(value, label=label, strictly_positive=True)


def validate_world_rect_aspect(world_rect: WorldRect, image_aspect_ratio: object) -> None:
    """Refuse toute pose monde qui déformerait l'image de la carte."""
    aspect = validate_finite_number(
        image_aspect_ratio,
        label="image_aspect_ratio",
        strictly_positive=True,
    )
    actual = world_rect.w / world_rect.h
    if not math.isclose(actual, aspect, rel_tol=1e-9, abs_tol=1e-12):
        raise ValueError(
            "world_rect ne respecte pas le ratio intrinsèque de l'image "
            f"({actual!r} au lieu de {aspect!r})."
        )


def world_rect_for_scale(
    reference_world_rect: WorldRect,
    reference_scale_factor: object,
    requested_scale_factor: object,
    *,
    image_aspect_ratio: object,
    x0: object | None = None,
    y0: object | None = None,
) -> WorldRect:
    """Construit une pose homothétique, ancrée par défaut en haut à gauche.

    La largeur est proportionnelle au facteur demandé par rapport à la pose de
    référence. La hauteur est toujours déduite du ratio image, jamais fournie
    indépendamment.
    """
    reference = validate_world_rect(reference_world_rect, label="reference_world_rect")
    reference_scale = validate_scale_factor(reference_scale_factor, label="reference_scale_factor")
    requested_scale = validate_scale_factor(requested_scale_factor, label="requested_scale_factor")
    aspect = validate_finite_number(
        image_aspect_ratio,
        label="image_aspect_ratio",
        strictly_positive=True,
    )
    width = reference.w * requested_scale / reference_scale
    return WorldRect(
        reference.x0 if x0 is None else validate_finite_number(x0, label="world_rect.x0"),
        reference.y0 if y0 is None else validate_finite_number(y0, label="world_rect.y0"),
        width,
        width / aspect,
    )


def scale_factor_for_world_rect(
    world_rect: WorldRect,
    reference_world_rect: WorldRect,
    reference_scale_factor: object,
) -> float:
    """Dérive le facteur homogène correspondant à une pose monde validée."""
    current = validate_world_rect(world_rect)
    reference = validate_world_rect(reference_world_rect, label="reference_world_rect")
    reference_scale = validate_scale_factor(reference_scale_factor, label="reference_scale_factor")
    return reference_scale * current.w / reference.w


@dataclass(frozen=True)
class MapTransform:
    """Composition Lambert (m) ↔ pixels ↔ monde, sans rotation ni miroir.

    Convention : pixel ``(0, 0)`` correspond à ``world_rect.(x0, y0)`` ;
    ``x`` augmente vers la droite et ``y`` augmente vers le bas dans les deux
    repères pixel et monde. Lambert conserve son axe Y orienté vers le nord.
    """

    # Pixels: origine haut-gauche, X droite, Y bas. Monde Assembleur: X droite, Y haut.
    calibrated_map: "CalibratedGeoMap"
    world_rect: WorldRect

    def __post_init__(self) -> None:
        rect = validate_world_rect(self.world_rect)
        image_width, image_height = self.calibrated_map.image_size
        if (
            isinstance(image_width, bool)
            or isinstance(image_height, bool)
            or not isinstance(image_width, int)
            or not isinstance(image_height, int)
            or image_width <= 0
            or image_height <= 0
        ):
            raise ValueError("La carte calibrée doit avoir des dimensions image entières positives.")
        validate_world_rect_aspect(rect, image_width / image_height)
        object.__setattr__(self, "world_rect", rect)

    @property
    def image_aspect_ratio(self) -> float:
        image_width, image_height = self.calibrated_map.image_size
        return image_width / image_height

    def pixel_to_world(self, x_px: float, y_px: float) -> tuple[float, float]:
        pixel_x = validate_finite_number(x_px, label="pixel.x")
        pixel_y = validate_finite_number(y_px, label="pixel.y")
        image_width, image_height = self.calibrated_map.image_size
        return (
            self.world_rect.x0 + pixel_x * self.world_rect.w / image_width,
            self.world_rect.y0 + self.world_rect.h - pixel_y * self.world_rect.h / image_height,
        )

    def world_to_pixel(self, x_world: float, y_world: float) -> tuple[float, float]:
        world_x = validate_finite_number(x_world, label="world.x")
        world_y = validate_finite_number(y_world, label="world.y")
        image_width, image_height = self.calibrated_map.image_size
        return (
            (world_x - self.world_rect.x0) * image_width / self.world_rect.w,
            (self.world_rect.y0 + self.world_rect.h - world_y) * image_height / self.world_rect.h,
        )

    def lambert_to_world(self, x_m: float, y_m: float) -> tuple[float, float]:
        lambert_x = validate_finite_number(x_m, label="lambert.x")
        lambert_y = validate_finite_number(y_m, label="lambert.y")
        return self.pixel_to_world(*self.calibrated_map.lambert_to_pixel(lambert_x, lambert_y))

    def world_to_lambert(self, x_world: float, y_world: float) -> tuple[float, float]:
        return self.calibrated_map.pixel_to_lambert(*self.world_to_pixel(x_world, y_world))
