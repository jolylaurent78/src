"""Migration explicite, one-shot et atomique d'un Catalogue V2 vers V3."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any

if __package__ in {None, ""}:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.assembleur_catalogue import (
    Catalogue,
    CatalogueBeacon,
    CatalogueCity,
    CatalogueTriangle,
    HypothesisTemplate,
    WorldRect,
)
from src.assembleur_catalogue_identity import SystemCatalogueIdProvider
from src.assembleur_catalogue_io import catalogue_to_dict


_V2_COUNTER_KEYS = ("city", "beacon", "triangle", "template")
_V2_ROOT_KEYS = {
    "version",
    "idCounters",
    "defaultTemplateId",
    "cities",
    "beacons",
    "triangles",
    "templates",
}
_DEFAULT_IMAGE_FILE = "899 - Alsace.jpg"
_DEFAULT_CALIBRATION_POINTS_FILE = "899 - Alsace.calib_points.json"
_DEFAULT_CALIBRATION_FILE = "899 - Alsace.json"


@dataclass(frozen=True)
class InitialMapDefinition:
    name: str
    image_file: str
    calibration_points_file: str | None
    calibration_file: str | None
    projection: str | None
    default_world_rect: WorldRect
    default_scale_factor: float


@dataclass(frozen=True)
class CatalogueV2ToV3MigrationResult:
    source: Path
    destination: Path
    backup: Path | None
    map_id: str


def _require_mapping(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"Catalogue V2 invalide : {label} doit être un objet JSON.")
    return value


def _require_list(value: object, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"Catalogue V2 invalide : {label} doit être une liste.")
    return value


def _require_field(mapping: dict[str, Any], field_name: str, label: str) -> Any:
    try:
        return mapping[field_name]
    except KeyError as exc:
        raise ValueError(f"Catalogue V2 invalide : {label} : champ obligatoire absent : {field_name}.") from exc


def _require_exact_keys(mapping: dict[str, Any], label: str, expected: set[str]) -> None:
    actual = set(mapping)
    if actual == expected:
        return
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    details = []
    if missing:
        details.append(f"clés manquantes : {', '.join(missing)}")
    if unexpected:
        details.append(f"clés inconnues : {', '.join(unexpected)}")
    raise ValueError(f"Catalogue V2 invalide : {label} possède une structure invalide ({'; '.join(details)}).")


def _require_str(value: object, label: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"Catalogue V2 invalide : {label} doit être une chaîne.")
    return value


def _require_optional_str(value: object, label: str) -> str | None:
    if value is not None and not isinstance(value, str):
        raise ValueError(f"Catalogue V2 invalide : {label} doit être une chaîne ou null.")
    return value


def _require_bool(value: object, label: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"Catalogue V2 invalide : {label} doit être un booléen.")
    return value


def _require_number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Catalogue V2 invalide : {label} doit être un nombre.")
    return float(value)


def _require_v2_counters(value: object) -> dict[str, int]:
    counters = _require_mapping(value, "idCounters")
    _require_exact_keys(counters, "idCounters", set(_V2_COUNTER_KEYS))
    restored: dict[str, int] = {}
    for kind in _V2_COUNTER_KEYS:
        counter = counters[kind]
        if isinstance(counter, bool) or not isinstance(counter, int) or counter < 0:
            raise ValueError(
                f"Catalogue V2 invalide : idCounters.{kind} doit être un entier positif ou nul."
            )
        restored[kind] = counter
    return restored


def _require_rank_ids(value: object) -> list[str | None]:
    ranks = _require_list(value, "templates.triangleIdsByRank")
    if len(ranks) != 32:
        raise ValueError("Catalogue V2 invalide : triangleIdsByRank doit contenir exactement 32 éléments.")
    if any(item is not None and not isinstance(item, str) for item in ranks):
        raise ValueError("Catalogue V2 invalide : triangleIdsByRank contient un identifiant invalide.")
    return list(ranks)


def parse_catalogue_v2(data: object, *, id_provider=None) -> Catalogue:
    """Parse et valide strictement V2 sans faire appel au loader JSON V3."""
    root = _require_mapping(data, "la racine")
    version = _require_field(root, "version", "la racine")
    if isinstance(version, bool) or not isinstance(version, int) or version != 2:
        raise ValueError(f"Catalogue V2 attendu, version reçue : {version!r}.")
    _require_exact_keys(root, "la racine", _V2_ROOT_KEYS)
    counters = _require_v2_counters(_require_field(root, "idCounters", "la racine"))
    default_template_id = _require_optional_str(
        _require_field(root, "defaultTemplateId", "la racine"), "defaultTemplateId"
    )

    catalogue = Catalogue(id_provider=id_provider or SystemCatalogueIdProvider())
    for index, raw_city in enumerate(_require_list(_require_field(root, "cities", "la racine"), "cities"), start=1):
        item = _require_mapping(raw_city, f"cities[{index}]")
        _require_exact_keys(item, f"cities[{index}]", {"cityId", "name", "latitude", "longitude", "archived"})
        city = CatalogueCity(
            _require_str(item["cityId"], f"cities[{index}].cityId"),
            _require_str(item["name"], f"cities[{index}].name"),
            _require_number(item["latitude"], f"cities[{index}].latitude"),
            _require_number(item["longitude"], f"cities[{index}].longitude"),
            _require_bool(item["archived"], f"cities[{index}].archived"),
        )
        if city.city_id in catalogue.cities:
            raise ValueError(f"Catalogue V2 invalide : identifiant ville dupliqué : {city.city_id}.")
        catalogue.cities[city.city_id] = city

    for index, raw_beacon in enumerate(_require_list(_require_field(root, "beacons", "la racine"), "beacons"), start=1):
        item = _require_mapping(raw_beacon, f"beacons[{index}]")
        _require_exact_keys(item, f"beacons[{index}]", {"beaconId", "cityId", "archived"})
        beacon = CatalogueBeacon(
            _require_str(item["beaconId"], f"beacons[{index}].beaconId"),
            _require_str(item["cityId"], f"beacons[{index}].cityId"),
            _require_bool(item["archived"], f"beacons[{index}].archived"),
        )
        if beacon.beacon_id in catalogue.beacons:
            raise ValueError(f"Catalogue V2 invalide : identifiant balise dupliqué : {beacon.beacon_id}.")
        catalogue.beacons[beacon.beacon_id] = beacon

    for index, raw_triangle in enumerate(_require_list(_require_field(root, "triangles", "la racine"), "triangles"), start=1):
        item = _require_mapping(raw_triangle, f"triangles[{index}]")
        _require_exact_keys(
            item,
            f"triangles[{index}]",
            {"triangleId", "note", "openingCityId", "baseCityId", "lightCityId", "archived"},
        )
        triangle = CatalogueTriangle(
            _require_str(item["triangleId"], f"triangles[{index}].triangleId"),
            _require_str(item["note"], f"triangles[{index}].note"),
            _require_str(item["openingCityId"], f"triangles[{index}].openingCityId"),
            _require_str(item["baseCityId"], f"triangles[{index}].baseCityId"),
            _require_str(item["lightCityId"], f"triangles[{index}].lightCityId"),
            _require_bool(item["archived"], f"triangles[{index}].archived"),
        )
        if triangle.triangle_id in catalogue.triangles:
            raise ValueError(f"Catalogue V2 invalide : identifiant triangle dupliqué : {triangle.triangle_id}.")
        catalogue.triangles[triangle.triangle_id] = triangle

    for index, raw_template in enumerate(_require_list(_require_field(root, "templates", "la racine"), "templates"), start=1):
        item = _require_mapping(raw_template, f"templates[{index}]")
        _require_exact_keys(
            item,
            f"templates[{index}]",
            {"templateId", "name", "description", "archived", "triangleIdsByRank"},
        )
        template = HypothesisTemplate(
            _require_str(item["templateId"], f"templates[{index}].templateId"),
            _require_str(item["name"], f"templates[{index}].name"),
            _require_str(item["description"], f"templates[{index}].description"),
            _require_bool(item["archived"], f"templates[{index}].archived"),
            _require_rank_ids(item["triangleIdsByRank"]),
        )
        if template.template_id in catalogue.templates:
            raise ValueError(f"Catalogue V2 invalide : identifiant template dupliqué : {template.template_id}.")
        catalogue.templates[template.template_id] = template

    catalogue.id_counters = {**counters, "map": 0}
    catalogue.version = 2
    catalogue.default_template_id = default_template_id
    catalogue.default_map_id = None
    catalogue.validate()
    return catalogue


def build_delivered_default_map_definition(
    *,
    config_path: str | Path,
    resource_maps_dir: str | Path,
) -> InitialMapDefinition:
    """Construit le seed MAP-SYS-000001 depuis les assets et config livrés audités."""
    config_source = Path(config_path)
    maps_dir = Path(resource_maps_dir)
    for asset_name in (
        _DEFAULT_IMAGE_FILE,
        _DEFAULT_CALIBRATION_POINTS_FILE,
        _DEFAULT_CALIBRATION_FILE,
    ):
        if not (maps_dir / asset_name).is_file():
            raise FileNotFoundError(f"Asset de la carte SYS initiale absent : {maps_dir / asset_name}")
    try:
        config = _require_mapping(json.loads(config_source.read_text(encoding="utf-8")), "configuration de référence")
    except json.JSONDecodeError as exc:
        raise ValueError(f"Configuration de référence JSON invalide {config_source} : {exc.msg}.") from exc
    if config.get("bgMap") != _DEFAULT_IMAGE_FILE:
        raise ValueError(
            f"La configuration de référence ne désigne pas {_DEFAULT_IMAGE_FILE!r} comme carte par défaut."
        )
    rect_data = _require_mapping(config.get("bgWorldRect"), "bgWorldRect")
    rect = WorldRect(
        _require_number(rect_data.get("x0"), "bgWorldRect.x0"),
        _require_number(rect_data.get("y0"), "bgWorldRect.y0"),
        _require_number(rect_data.get("w"), "bgWorldRect.w"),
        _require_number(rect_data.get("h"), "bgWorldRect.h"),
    )
    return InitialMapDefinition(
        name="899 - Alsace",
        image_file=_DEFAULT_IMAGE_FILE,
        calibration_points_file=_DEFAULT_CALIBRATION_POINTS_FILE,
        calibration_file=_DEFAULT_CALIBRATION_FILE,
        projection="EPSG:2154",
        default_world_rect=rect,
        default_scale_factor=12.0,
    )


def migrate_catalogue_data_v2_to_v3(
    data: object,
    *,
    initial_map: InitialMapDefinition,
) -> dict[str, Any]:
    """Convertit V2 vers V3 en mémoire, sans modifier les identités existantes."""
    catalogue = parse_catalogue_v2(data, id_provider=SystemCatalogueIdProvider())
    map_id = catalogue.add_map(
        name=initial_map.name,
        image_file=initial_map.image_file,
        calibration_points_file=initial_map.calibration_points_file,
        calibration_file=initial_map.calibration_file,
        projection=initial_map.projection,
        default_world_rect=initial_map.default_world_rect,
        default_scale_factor=initial_map.default_scale_factor,
    )
    if map_id != "MAP-SYS-000001":
        raise ValueError(f"Migration V2→V3 attendait MAP-SYS-000001, identité produite : {map_id}.")
    catalogue.set_default_map(map_id)
    catalogue.validate()
    migrated = catalogue_to_dict(catalogue)
    # Le contrat de cette migration reste explicitement V2 -> V3.  Le passage
    # V3 -> V4, qui introduit les rôles de cartes, est réalisé par l'outil
    # dédié ``migrate_catalogue_v3_to_v4``.
    migrated["version"] = 3
    migrated.pop("catalogueReferenceMapId", None)
    for catalogue_map in migrated["maps"]:
        catalogue_map.pop("description", None)
        catalogue_map.pop("calibrationCityIds", None)
    return migrated


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return _require_mapping(json.loads(path.read_text(encoding="utf-8")), "la racine")
    except json.JSONDecodeError as exc:
        raise ValueError(f"JSON catalogue invalide {path} : {exc.msg}.") from exc


def _serialized_catalogue(data: dict[str, Any]) -> bytes:
    return (json.dumps(data, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def _write_atomic(destination: Path, content: bytes) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def _backup_path_for(source: Path) -> Path:
    return source.with_name(f"{source.stem}.v2.pre-catalog-map-001c1{source.suffix}")


def migrate_catalogue_file_v2_to_v3(
    input_path: str | Path,
    *,
    initial_map: InitialMapDefinition,
    output_path: str | Path | None = None,
    force: bool = False,
) -> CatalogueV2ToV3MigrationResult:
    """Migre un fichier V2 vers V3 ; un remplacement in-place crée un backup obligatoire."""
    source = Path(input_path)
    destination = source if output_path is None else Path(output_path)
    if not source.is_file():
        raise FileNotFoundError(f"Catalogue source introuvable : {source}")
    is_in_place = source.resolve() == destination.resolve()
    if not is_in_place and destination.exists() and not force:
        raise FileExistsError(f"Destination de migration déjà existante : {destination}")
    backup = _backup_path_for(source) if is_in_place else None
    if backup is not None and backup.exists():
        raise FileExistsError(f"Backup de migration déjà existant : {backup}")

    migrated = migrate_catalogue_data_v2_to_v3(_read_json(source), initial_map=initial_map)
    if backup is not None:
        shutil.copy2(source, backup)
    _write_atomic(destination, _serialized_catalogue(migrated))
    return CatalogueV2ToV3MigrationResult(source, destination, backup, "MAP-SYS-000001")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", type=Path, help="destination ; absent = remplacement in-place avec backup")
    parser.add_argument("--config", required=True, type=Path, help="configuration livrée portant bgMap/bgWorldRect")
    parser.add_argument("--maps-dir", required=True, type=Path, help="répertoire resources/maps livré")
    parser.add_argument("--force", action="store_true", help="autorise l'écrasement d'une destination distincte")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        initial_map = build_delivered_default_map_definition(
            config_path=args.config,
            resource_maps_dir=args.maps_dir,
        )
        result = migrate_catalogue_file_v2_to_v3(
            args.input,
            output_path=args.output,
            initial_map=initial_map,
            force=args.force,
        )
    except (OSError, ValueError) as exc:
        print(f"MIGRATION FAILED: {exc}")
        return 1
    print(f"MIGRATION SUCCESS: {result.destination}")
    if result.backup is not None:
        print(f"BACKUP: {result.backup}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
