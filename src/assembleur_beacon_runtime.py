"""Résolution runtime des balises Catalogue dans le repère World."""

from __future__ import annotations

from collections.abc import Callable

from src.assembleur_catalogue import Catalogue


class BeaconWorldResolver:
    """Adaptateur Core minimal, sans dupliquer de données géographiques."""

    def __init__(
        self,
        catalogue: Catalogue,
        lambert_to_world: Callable[[float, float], tuple[float, float]],
    ) -> None:
        self._catalogue = catalogue
        self._lambert_to_world = lambert_to_world

    def set_catalogue(self, catalogue: Catalogue) -> None:
        self._catalogue = catalogue

    def contains(self, beacon_id: str) -> bool:
        return beacon_id in self._catalogue.beacons

    def get_world(self, beacon_id: str) -> tuple[float, float]:
        beacon = self._catalogue.get_beacon(beacon_id)
        lambert_x_m, lambert_y_m = self._catalogue.get_city_lambert(beacon.city_id)
        return self._lambert_to_world(lambert_x_m, lambert_y_m)

    def iter_beacon_ids(self) -> tuple[str, ...]:
        return tuple(beacon.beacon_id for beacon in self._catalogue.iter_beacons())
