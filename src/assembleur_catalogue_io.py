"""Persistance JSON V1 du catalogue, indépendante de toute interface Tk."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from src.assembleur_catalogue import (
    Catalogue,
    CatalogueBeacon,
    CatalogueCity,
    CatalogueTriangle,
    HypothesisTemplate,
)


_VERSION = 1


def catalogue_to_dict(catalogue: Catalogue) -> dict[str, Any]:
    """Produit la représentation JSON V1 déterministe d'un catalogue valide."""
    catalogue.validate()
    return {
        "version": _VERSION,
        "defaultTemplateId": catalogue.default_template_id,
        "cities": [
            {
                "cityId": city.city_id,
                "name": city.name,
                "latitude": city.latitude,
                "longitude": city.longitude,
                "archived": city.archived,
            }
            for city in catalogue.iter_cities()
        ],
        "beacons": [
            {
                "beaconId": beacon.beacon_id,
                "cityId": beacon.city_id,
                "archived": beacon.archived,
            }
            for beacon in catalogue.iter_beacons()
        ],
        "triangles": [
            {
                "triangleId": triangle.triangle_id,
                "note": triangle.note,
                "openingCityId": triangle.opening_city_id,
                "baseCityId": triangle.base_city_id,
                "lightCityId": triangle.light_city_id,
                "archived": triangle.archived,
            }
            for triangle in catalogue.iter_triangles()
        ],
        "templates": [
            {
                "templateId": template.template_id,
                "name": template.name,
                "description": template.description,
                "archived": template.archived,
                "triangleIdsByRank": list(template.triangle_ids_by_rank),
            }
            for template in catalogue.iter_templates()
        ],
    }


def _require_mapping(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"Catalogue invalide : {label} doit être un objet JSON.")
    return value


def _require_list(value: object, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"Catalogue invalide : {label} doit être une liste.")
    return value


def _require_str(value: object, label: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"Catalogue invalide : {label} doit être une chaîne.")
    return value


def _require_bool(value: object, label: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"Catalogue invalide : {label} doit être un booléen.")
    return value


def _require_number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Catalogue invalide : {label} doit être un nombre.")
    return float(value)


def _require_rank_ids(value: object, label: str) -> list[str | None]:
    ranks = _require_list(value, label)
    if len(ranks) != 32:
        raise ValueError("Catalogue invalide : triangleIdsByRank doit contenir exactement 32 éléments.")
    if any(item is not None and not isinstance(item, str) for item in ranks):
        raise ValueError("Catalogue invalide : triangleIdsByRank contient un identifiant invalide.")
    return list(ranks)


def catalogue_from_dict(data: object) -> Catalogue:
    """Réhydrate strictement un catalogue V1, sans régénérer les identifiants."""
    root = _require_mapping(data, "la racine")
    version = root["version"]
    if isinstance(version, bool) or not isinstance(version, int):
        raise ValueError("Catalogue invalide : version doit être un entier.")
    if version != _VERSION:
        raise ValueError(f"Version de catalogue non supportée : {version}")
    default_template_id = root["defaultTemplateId"]
    if default_template_id is not None and not isinstance(default_template_id, str):
        raise ValueError("Catalogue invalide : defaultTemplateId doit être une chaîne ou null.")

    catalogue = Catalogue()
    for index, raw_city in enumerate(_require_list(root["cities"], "cities"), start=1):
        item = _require_mapping(raw_city, f"cities[{index}]")
        city = CatalogueCity(
            _require_str(item["cityId"], "cityId"),
            _require_str(item["name"], "name"),
            _require_number(item["latitude"], "latitude"),
            _require_number(item["longitude"], "longitude"),
            _require_bool(item["archived"], "archived"),
        )
        if city.city_id in catalogue.cities:
            raise ValueError(f"Catalogue invalide : identifiant ville dupliqué : {city.city_id}.")
        catalogue.cities[city.city_id] = city

    for index, raw_beacon in enumerate(_require_list(root.get("beacons", []), "beacons"), start=1):
        item = _require_mapping(raw_beacon, f"beacons[{index}]")
        beacon = CatalogueBeacon(
            _require_str(item["beaconId"], "beaconId"),
            _require_str(item["cityId"], "cityId"),
            _require_bool(item["archived"], "archived"),
        )
        if beacon.beacon_id in catalogue.beacons:
            raise ValueError(f"Catalogue invalide : identifiant balise dupliqué : {beacon.beacon_id}.")
        catalogue.beacons[beacon.beacon_id] = beacon

    for index, raw_triangle in enumerate(_require_list(root["triangles"], "triangles"), start=1):
        item = _require_mapping(raw_triangle, f"triangles[{index}]")
        triangle = CatalogueTriangle(
            _require_str(item["triangleId"], "triangleId"),
            _require_str(item["note"], "note"),
            _require_str(item["openingCityId"], "openingCityId"),
            _require_str(item["baseCityId"], "baseCityId"),
            _require_str(item["lightCityId"], "lightCityId"),
            _require_bool(item["archived"], "archived"),
        )
        if triangle.triangle_id in catalogue.triangles:
            raise ValueError(f"Catalogue invalide : identifiant triangle dupliqué : {triangle.triangle_id}.")
        catalogue.triangles[triangle.triangle_id] = triangle

    for index, raw_template in enumerate(_require_list(root["templates"], "templates"), start=1):
        item = _require_mapping(raw_template, f"templates[{index}]")
        template = HypothesisTemplate(
            _require_str(item["templateId"], "templateId"),
            _require_str(item["name"], "name"),
            _require_str(item["description"], "description"),
            _require_bool(item["archived"], "archived"),
            _require_rank_ids(item["triangleIdsByRank"], "triangleIdsByRank"),
        )
        if template.template_id in catalogue.templates:
            raise ValueError(f"Catalogue invalide : identifiant template dupliqué : {template.template_id}.")
        catalogue.templates[template.template_id] = template

    catalogue.version = version
    catalogue.default_template_id = default_template_id
    catalogue.validate()
    return catalogue


def load_catalogue(path: str | Path) -> Catalogue:
    source = Path(path)
    try:
        with source.open("r", encoding="utf-8") as stream:
            data = json.load(stream)
    except json.JSONDecodeError as exc:
        raise ValueError(f"JSON catalogue invalide : {exc.msg}.") from exc
    return catalogue_from_dict(data)


def save_catalogue(catalogue: Catalogue, path: str | Path) -> None:
    """Écrit atomiquement le JSON V1, sans altérer un fichier valide existant."""
    data = catalogue_to_dict(catalogue)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            json.dump(data, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    except OSError:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
        raise
