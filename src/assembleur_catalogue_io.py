"""Persistance JSON V5 du catalogue, indépendante de toute interface Tk."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from src.assembleur_catalogue import (
    Catalogue,
    CatalogueBeacon,
    CatalogueCity,
    CatalogueMap,
    CatalogueTriangle,
    HypothesisTemplate,
    WorldRect,
)
from src.assembleur_catalogue_identity import CATALOGUE_ID_KIND_ORDER, CatalogueIdProvider


_VERSION = Catalogue.version


def catalogue_to_dict(catalogue: Catalogue) -> dict[str, Any]:
    """Produit la représentation JSON V5 déterministe d'un catalogue valide."""
    catalogue.validate()
    return {
        "version": _VERSION,
        "idCounters": {
            kind: catalogue.id_counters[kind]
            for kind in CATALOGUE_ID_KIND_ORDER
        },
        "defaultTemplateId": catalogue.default_template_id,
        "defaultMapId": catalogue.default_map_id,
        "catalogueReferenceMapId": catalogue.catalogue_reference_map_id,
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
        "maps": [
            {
                "mapId": catalogue_map.map_id,
                "name": catalogue_map.name,
                "imageFile": catalogue_map.image_file,
                "calibrationPointsFile": catalogue_map.calibration_points_file,
                "calibrationFile": catalogue_map.calibration_file,
                "projection": catalogue_map.projection,
                "defaultWorldRect": {
                    "x0": catalogue_map.default_world_rect.x0,
                    "y0": catalogue_map.default_world_rect.y0,
                    "w": catalogue_map.default_world_rect.w,
                    "h": catalogue_map.default_world_rect.h,
                },
                "defaultScaleFactor": catalogue_map.default_scale_factor,
                "archived": catalogue_map.archived,
                "description": catalogue_map.description,
                "calibrationCityIds": list(catalogue_map.calibration_city_ids),
            }
            for catalogue_map in catalogue.iter_maps()
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


def _require_field(mapping: dict[str, Any], key: str) -> Any:
    try:
        return mapping[key]
    except KeyError as exc:
        raise ValueError(f"Catalogue invalide : champ obligatoire absent : {key}.") from exc


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


def _require_optional_str(value: object, label: str) -> str | None:
    if value is not None and not isinstance(value, str):
        raise ValueError(f"Catalogue invalide : {label} doit être une chaîne ou null.")
    return value


def _require_exact_keys(mapping: dict[str, Any], label: str, expected_keys: set[str]) -> None:
    actual_keys = set(mapping)
    if actual_keys == expected_keys:
        return
    missing = sorted(expected_keys - actual_keys)
    unexpected = sorted(actual_keys - expected_keys)
    details = []
    if missing:
        details.append(f"clés manquantes : {', '.join(missing)}")
    if unexpected:
        details.append(f"clés inconnues : {', '.join(unexpected)}")
    raise ValueError(f"Catalogue invalide : {label} possède une structure invalide ({'; '.join(details)}).")


def _require_id_counters(value: object) -> dict[str, int]:
    counters = _require_mapping(value, "idCounters")
    expected_keys = set(CATALOGUE_ID_KIND_ORDER)
    actual_keys = set(counters)
    if actual_keys != expected_keys:
        missing = sorted(expected_keys - actual_keys)
        unexpected = sorted(actual_keys - expected_keys)
        details = []
        if missing:
            details.append(f"clés manquantes : {', '.join(missing)}")
        if unexpected:
            details.append(f"clés inconnues : {', '.join(unexpected)}")
        raise ValueError(
            "Catalogue invalide : idCounters doit contenir exactement les clés "
            f"{', '.join(CATALOGUE_ID_KIND_ORDER)} ({'; '.join(details)})."
        )
    restored: dict[str, int] = {}
    for kind in CATALOGUE_ID_KIND_ORDER:
        counter = counters[kind]
        if isinstance(counter, bool) or not isinstance(counter, int) or counter < 0:
            raise ValueError(
                f"Catalogue invalide : idCounters.{kind} doit être un entier positif ou nul."
            )
        restored[kind] = counter
    return restored


def _require_rank_ids(value: object, label: str) -> list[str | None]:
    ranks = _require_list(value, label)
    if len(ranks) != 32:
        raise ValueError("Catalogue invalide : triangleIdsByRank doit contenir exactement 32 éléments.")
    if any(item is not None and not isinstance(item, str) for item in ranks):
        raise ValueError("Catalogue invalide : triangleIdsByRank contient un identifiant invalide.")
    return list(ranks)


def _catalogue_map_from_dict(raw_map: object, index: int) -> CatalogueMap:
    label = f"maps[{index}]"
    item = _require_mapping(raw_map, label)
    _require_exact_keys(
        item,
        label,
        {
            "mapId",
            "name",
            "imageFile",
            "calibrationPointsFile",
            "calibrationFile",
            "projection",
            "defaultWorldRect",
            "defaultScaleFactor",
            "archived",
            "description",
            "calibrationCityIds",
        },
    )
    raw_rect = _require_mapping(_require_field(item, "defaultWorldRect"), f"{label}.defaultWorldRect")
    _require_exact_keys(raw_rect, f"{label}.defaultWorldRect", {"x0", "y0", "w", "h"})
    return CatalogueMap(
        _require_str(_require_field(item, "mapId"), f"{label}.mapId"),
        _require_str(_require_field(item, "name"), f"{label}.name"),
        _require_str(_require_field(item, "imageFile"), f"{label}.imageFile"),
        _require_optional_str(_require_field(item, "calibrationPointsFile"), f"{label}.calibrationPointsFile"),
        _require_optional_str(_require_field(item, "calibrationFile"), f"{label}.calibrationFile"),
        _require_optional_str(_require_field(item, "projection"), f"{label}.projection"),
        WorldRect(
            _require_number(_require_field(raw_rect, "x0"), f"{label}.defaultWorldRect.x0"),
            _require_number(_require_field(raw_rect, "y0"), f"{label}.defaultWorldRect.y0"),
            _require_number(_require_field(raw_rect, "w"), f"{label}.defaultWorldRect.w"),
            _require_number(_require_field(raw_rect, "h"), f"{label}.defaultWorldRect.h"),
        ),
        _require_number(_require_field(item, "defaultScaleFactor"), f"{label}.defaultScaleFactor"),
        _require_bool(_require_field(item, "archived"), f"{label}.archived"),
        _require_str(_require_field(item, "description"), f"{label}.description"),
        [
            _require_str(value, f"{label}.calibrationCityIds[{city_index}]")
            for city_index, value in enumerate(
                _require_list(_require_field(item, "calibrationCityIds"), f"{label}.calibrationCityIds")
            )
        ],
    )


def catalogue_from_dict(data: object, *, id_provider: CatalogueIdProvider | None = None) -> Catalogue:
    """Réhydrate strictement un catalogue V5, sans régénérer les identifiants."""
    root = _require_mapping(data, "la racine")
    version = _require_field(root, "version")
    if isinstance(version, bool) or not isinstance(version, int):
        raise ValueError("Catalogue invalide : version doit être un entier.")
    if version != _VERSION:
        raise ValueError(f"Version de catalogue non supportée : {version}")
    _require_exact_keys(
        root,
        "la racine",
        {
            "version",
            "idCounters",
            "defaultTemplateId",
            "defaultMapId",
            "catalogueReferenceMapId",
            "cities",
            "beacons",
            "triangles",
            "templates",
            "maps",
        },
    )
    id_counters = _require_id_counters(_require_field(root, "idCounters"))
    default_template_id = _require_field(root, "defaultTemplateId")
    if default_template_id is not None and not isinstance(default_template_id, str):
        raise ValueError("Catalogue invalide : defaultTemplateId doit être une chaîne ou null.")
    default_map_id = _require_field(root, "defaultMapId")
    if default_map_id is not None and not isinstance(default_map_id, str):
        raise ValueError("Catalogue invalide : defaultMapId doit être une chaîne ou null.")
    catalogue_reference_map_id = _require_field(root, "catalogueReferenceMapId")
    if catalogue_reference_map_id is not None and not isinstance(catalogue_reference_map_id, str):
        raise ValueError("Catalogue invalide : catalogueReferenceMapId doit être une chaîne ou null.")

    catalogue = Catalogue(id_provider=id_provider)
    for index, raw_city in enumerate(_require_list(_require_field(root, "cities"), "cities"), start=1):
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

    for index, raw_beacon in enumerate(_require_list(_require_field(root, "beacons"), "beacons"), start=1):
        item = _require_mapping(raw_beacon, f"beacons[{index}]")
        beacon = CatalogueBeacon(
            _require_str(item["beaconId"], "beaconId"),
            _require_str(item["cityId"], "cityId"),
            _require_bool(item["archived"], "archived"),
        )
        if beacon.beacon_id in catalogue.beacons:
            raise ValueError(f"Catalogue invalide : identifiant balise dupliqué : {beacon.beacon_id}.")
        catalogue.beacons[beacon.beacon_id] = beacon

    for index, raw_triangle in enumerate(_require_list(_require_field(root, "triangles"), "triangles"), start=1):
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

    for index, raw_template in enumerate(_require_list(_require_field(root, "templates"), "templates"), start=1):
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

    for index, raw_map in enumerate(_require_list(_require_field(root, "maps"), "maps"), start=1):
        catalogue_map = _catalogue_map_from_dict(raw_map, index)
        if catalogue_map.map_id in catalogue.maps:
            raise ValueError(f"Catalogue invalide : identifiant carte dupliqué : {catalogue_map.map_id}.")
        catalogue.maps[catalogue_map.map_id] = catalogue_map

    catalogue.version = version
    catalogue.id_counters = id_counters
    catalogue.default_template_id = default_template_id
    catalogue.default_map_id = default_map_id
    catalogue.catalogue_reference_map_id = catalogue_reference_map_id
    catalogue.validate()
    return catalogue


def load_catalogue(path: str | Path, *, id_provider: CatalogueIdProvider | None = None) -> Catalogue:
    source = Path(path)
    try:
        with source.open("r", encoding="utf-8") as stream:
            data = json.load(stream)
    except json.JSONDecodeError as exc:
        raise ValueError(f"JSON catalogue invalide : {exc.msg}.") from exc
    return catalogue_from_dict(data, id_provider=id_provider)


def save_catalogue(catalogue: Catalogue, path: str | Path) -> None:
    """Écrit atomiquement le JSON V5, sans altérer un fichier valide existant."""
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
