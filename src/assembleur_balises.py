"""Catalogue global des repères géographiques partagés par les scénarios."""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass, replace
from typing import Iterable

from src.assembleur_io import TriangleFileService
from src.utils.logging_utils import get_app_logger


@dataclass(frozen=True)
class Beacon:
    """Repère géographique identifié indépendamment de son libellé."""

    beacon_id: str
    name: str
    latitude: float
    longitude: float
    lambert_x: float
    lambert_y: float
    world_x: float | None = None
    world_y: float | None = None


class BeaconCatalog:
    """Référentiel global immuable côté consommateurs, indexé par ``beacon_id``."""

    _EXPECTED_HEADER = ["Id", "Nom", "Latitude", "Longitude"]
    _DMS_RE = TriangleFileService.DMS_RE

    def __init__(self) -> None:
        self._by_id: dict[str, Beacon] = {}
        self._transformer = None

    def clear(self) -> None:
        self._by_id = {}

    def load_from_csv(self, csv_path: str) -> None:
        path = os.path.normpath(str(csv_path or "").strip())
        if not path:
            raise ValueError("Beacon CSV: missing path.")
        if not os.path.isfile(path):
            raise ValueError(f"Beacon CSV not found: {path}")

        loaded: dict[str, Beacon] = {}
        with open(path, "r", encoding="utf-8-sig", newline="") as source:
            reader = csv.reader(source, delimiter=",")
            try:
                header = [str(value).strip() for value in next(reader)]
            except StopIteration as exc:
                raise ValueError("Beacon CSV is empty; expected Id,Nom,Latitude,Longitude.") from exc
            if header != self._EXPECTED_HEADER:
                raise ValueError("Beacon CSV header must be Id,Nom,Latitude,Longitude.")

            for line_no, row in enumerate(reader, start=2):
                if not row or all(not str(value).strip() for value in row):
                    continue
                if len(row) != 4:
                    raise ValueError(f"Beacon CSV line {line_no}: expected 4 columns.")
                beacon_id, name, raw_latitude, raw_longitude = (str(value).strip() for value in row)
                if not beacon_id:
                    raise ValueError(f"Beacon CSV line {line_no}: missing beacon_id.")
                if beacon_id in loaded:
                    raise ValueError(f"Beacon CSV line {line_no}: duplicate beacon_id {beacon_id!r}.")
                if not name:
                    raise ValueError(f"Beacon CSV line {line_no}: missing name.")
                latitude = self._parse_coordinate(raw_latitude, ("N", "S"), "latitude")
                longitude = self._parse_coordinate(raw_longitude, ("E", "W"), "longitude")
                try:
                    x_m, y_m = self._get_transformer().transform(longitude, latitude)
                except Exception as exc:
                    raise RuntimeError(f"Beacon projection failure for {beacon_id!r}: {exc}") from exc
                loaded[beacon_id] = Beacon(
                    beacon_id=beacon_id,
                    name=name,
                    latitude=latitude,
                    longitude=longitude,
                    lambert_x=float(x_m) / 1000.0,
                    lambert_y=float(y_m) / 1000.0,
                )

        self._by_id = loaded
        get_app_logger().info("BeaconCatalog loaded: %s beacons", len(loaded))

    def get(self, beacon_id: str) -> Beacon:
        return self._by_id[str(beacon_id or "").strip()]

    def contains(self, beacon_id: str) -> bool:
        return str(beacon_id or "").strip() in self._by_id

    def iter_beacons(self) -> Iterable[Beacon]:
        return tuple(sorted(self._by_id.values(), key=lambda beacon: beacon.beacon_id))

    def find_by_name(self, name: str) -> list[Beacon]:
        target = str(name or "").strip().casefold()
        return [beacon for beacon in self.iter_beacons() if beacon.name.casefold() == target]

    def get_world(self, beacon_id: str) -> tuple[float, float]:
        beacon = self.get(beacon_id)
        if beacon.world_x is None or beacon.world_y is None:
            raise RuntimeError(f"Beacon {beacon_id!r} has no World coordinates.")
        return (float(beacon.world_x), float(beacon.world_y))

    def reproject_world(self, lambert_to_world) -> None:
        """Rebuild immutable beacons from the current Lambert→World projection."""
        projected: dict[str, Beacon] = {}
        for beacon in self.iter_beacons():
            wx, wy = lambert_to_world(beacon.lambert_x, beacon.lambert_y)
            projected[beacon.beacon_id] = replace(
                beacon, world_x=float(wx), world_y=float(wy)
            )
        self._by_id = projected

    def clear_world_coordinates(self) -> None:
        """Rebuild beacons without a currently usable World projection."""
        self._by_id = {
            beacon.beacon_id: replace(beacon, world_x=None, world_y=None)
            for beacon in self.iter_beacons()
        }

    def _parse_coordinate(self, raw: str, hemispheres: tuple[str, ...], label: str) -> float:
        value = str(raw or "").strip()
        try:
            numeric = float(value)
        except ValueError:
            match = self._DMS_RE.match(value)
            if match is None:
                raise ValueError(f"Invalid {label}: {raw!r}")
            degrees, minutes, seconds, hemisphere = match.groups()
            if hemisphere not in hemispheres or int(minutes) >= 60 or int(seconds) >= 60:
                raise ValueError(f"Invalid {label}: {raw!r}")
            numeric = int(degrees) + int(minutes) / 60.0 + int(seconds) / 3600.0
            if hemisphere in ("S", "W"):
                numeric = -numeric
        if not (-90.0 <= numeric <= 90.0 if label == "latitude" else -180.0 <= numeric <= 180.0):
            raise ValueError(f"Invalid {label}: {raw!r}")
        return float(numeric)

    def _get_transformer(self):
        if self._transformer is None:
            try:
                from pyproj import Transformer
                self._transformer = Transformer.from_crs("EPSG:4326", "EPSG:2154", always_xy=True)
            except Exception as exc:
                raise RuntimeError("Lambert93 projection unavailable.") from exc
        return self._transformer
