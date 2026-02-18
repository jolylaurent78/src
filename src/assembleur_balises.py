"""Gestion des balises GPS (CSV DMS -> Lambert93 km -> World)."""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass

from src.assembleur_io import TriangleFileService


@dataclass(frozen=True)
class Balise:
    nom: str
    latDms: str
    lonDms: str
    lambertXKm: float
    lambertYKm: float


class BalisesManager:
    _EXPECTED_HEADER = ["Nom", "Latitude", "Longitude"]
    _DMS_RE = TriangleFileService.DMS_RE

    def __init__(self) -> None:
        self._balisesByName: dict[str, Balise] = {}
        self._transformer = None

    def clear(self) -> None:
        self._balisesByName = {}

    def loadFromCsv(self, csvPath: str) -> None:
        csv_path = os.path.normpath(str(csvPath or "").strip())
        if not csv_path:
            raise ValueError("CSV balises: chemin vide.")
        if not os.path.isfile(csv_path):
            raise ValueError(f"CSV balises introuvable: {csv_path}")

        loaded: dict[str, Balise] = {}
        with open(csv_path, "r", encoding="utf-8-sig", newline="") as f_in:
            reader = csv.reader(f_in, delimiter=",")
            try:
                raw_header = next(reader)
            except StopIteration as e:
                raise ValueError(
                    "CSV balises vide. Entete attendue: Nom,Latitude,Longitude"
                ) from e

            header = [str(c).strip() for c in raw_header]
            if header != self._EXPECTED_HEADER:
                raise ValueError(
                    "CSV balises: entete invalide. Colonnes attendues: Nom,Latitude,Longitude"
                )

            for line_no, row in enumerate(reader, start=2):
                if len(row) == 0 or all(str(c).strip() == "" for c in row):
                    continue
                if len(row) != 3:
                    raise ValueError(
                        f"CSV balises: ligne {line_no} invalide (3 colonnes attendues)."
                    )

                nom = str(row[0]).strip()
                lat_dms = str(row[1]).strip()
                lon_dms = str(row[2]).strip()
                if not nom:
                    raise ValueError(f"CSV balises: Nom vide a la ligne {line_no}.")
                if nom in loaded:
                    raise ValueError(f"CSV balises: Nom duplique '{nom}'.")

                lat_deg = self._parseDms(lat_dms, expected_hemispheres=("N", "S"), field_label="latitude")
                lon_deg = self._parseDms(lon_dms, expected_hemispheres=("E", "W"), field_label="longitude")
                try:
                    x_m, y_m = self._getTransformer().transform(lon_deg, lat_deg)
                except Exception as e:
                    raise RuntimeError(f"Projection Lambert93 impossible pour '{nom}': {e}") from e

                loaded[nom] = Balise(
                    nom=nom,
                    latDms=lat_dms,
                    lonDms=lon_dms,
                    lambertXKm=float(x_m) / 1000.0,
                    lambertYKm=float(y_m) / 1000.0,
                )

        self._balisesByName = loaded

    def hasBalise(self, nom: str) -> bool:
        return self._normNom(nom) in self._balisesByName

    def getLambertKm(self, nom: str) -> tuple[float, float]:
        key = self._normNom(nom)
        balise = self._balisesByName[key]
        return (float(balise.lambertXKm), float(balise.lambertYKm))

    def getWorld(self, viewer, nom: str) -> tuple[float, float]:
        if viewer is None or not hasattr(viewer, "bgLambertKmToWorld"):
            raise RuntimeError("Viewer incompatible: bgLambertKmToWorld manquant.")
        x_km, y_km = self.getLambertKm(nom)
        wx, wy = viewer.bgLambertKmToWorld(x_km, y_km)
        return (float(wx), float(wy))

    def listNoms(self) -> list[str]:
        return sorted(self._balisesByName.keys(), key=str.casefold)

    def _normNom(self, nom: str) -> str:
        return str(nom or "").strip()

    def _parseDmsParts(self, dms: str):
        if not isinstance(dms, str):
            raise ValueError(f"DMS invalide: {dms!r}")
        m = self._DMS_RE.match(dms)
        if m is None:
            raise ValueError(f"DMS invalide: {dms!r}")

        deg = int(m.group(1))
        minute = int(m.group(2))
        second = int(m.group(3))
        hemisphere = m.group(4)
        if minute < 0 or minute >= 60:
            raise ValueError(f"DMS invalide (minutes): {dms!r}")
        if second < 0 or second >= 60:
            raise ValueError(f"DMS invalide (secondes): {dms!r}")
        return (deg, minute, second, hemisphere)

    def _parseDms(self, dms: str, expected_hemispheres: tuple[str, ...], field_label: str) -> float:
        deg, minute, second, hemisphere = self._parseDmsParts(dms)
        if hemisphere not in expected_hemispheres:
            raise ValueError(f"DMS {field_label} invalide: {dms!r}")
        decimal = float(deg) + (float(minute) / 60.0) + (float(second) / 3600.0)
        if hemisphere in ("S", "W"):
            decimal = -decimal
        return decimal

    def _getTransformer(self):
        if self._transformer is None:
            try:
                from pyproj import Transformer
            except Exception as e:
                raise RuntimeError("Projection Lambert93 indisponible (pyproj non importable).") from e
            try:
                self._transformer = Transformer.from_crs("EPSG:4326", "EPSG:2154", always_xy=True)
            except Exception as e:
                raise RuntimeError(f"Projection Lambert93 indisponible: {e}") from e
        return self._transformer
