"""Migration explicite et one-shot des identités Catalogue V1 vers V2 SYS."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import re
import sys
import tempfile
from typing import Any, Iterable
import xml.etree.ElementTree as ET

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.assembleur_catalogue_identity import CATALOGUE_ID_KIND_ORDER, SystemCatalogueIdProvider
from src.assembleur_catalogue_io import catalogue_from_dict, catalogue_to_dict


_LEGACY_PREFIXES = {
    "city": "CITY",
    "beacon": "BEA",
    "triangle": "TRI",
    "template": "TPL",
}
_LEGACY_ID_RE = {
    kind: re.compile(rf"^{prefix}-(\d+)$")
    for kind, prefix in _LEGACY_PREFIXES.items()
}


@dataclass(frozen=True)
class CatalogueMappings:
    city: dict[str, str]
    beacon: dict[str, str]
    triangle: dict[str, str]
    template: dict[str, str]
    counters: dict[str, int]

    def for_kind(self, kind: str) -> dict[str, str]:
        return getattr(self, kind)


@dataclass
class ScenarioMigrationReport:
    source: Path
    cities: int = 0
    triangles: int = 0
    beacons: int = 0
    templates: int = 0
    local_scities: int = 0
    local_stris: int = 0


@dataclass
class MigrationReport:
    mappings: CatalogueMappings
    scenario_reports: list[ScenarioMigrationReport] = field(default_factory=list)

    def render(self) -> str:
        lines = [
            "Catalogue",
            f"  Cities:    {len(self.mappings.city)} migrated",
            f"  Beacons:  {len(self.mappings.beacon)} migrated",
            f"  Triangles:{len(self.mappings.triangle)} migrated",
            f"  Templates:{len(self.mappings.template)} migrated",
            "Counters",
        ]
        lines.extend(f"  {kind}: {self.mappings.counters[kind]}" for kind in CATALOGUE_ID_KIND_ORDER)
        for scenario in self.scenario_reports:
            lines.extend([
                f"Scenario {scenario.source}",
                f"  CITY references migrated: {scenario.cities}",
                f"  TRI references migrated: {scenario.triangles}",
                f"  BEA anchors migrated: {scenario.beacons}",
                f"  TPL references migrated: {scenario.templates}",
                f"  local SCITY preserved: {scenario.local_scities}",
                f"  local STRI preserved: {scenario.local_stris}",
            ])
        return "\n".join(lines)


def _require_mapping(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"Catalogue V1 invalide : {label} doit être un objet JSON.")
    return value


def _require_list(value: object, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"Catalogue V1 invalide : {label} doit être une liste.")
    return value


def _require_field(mapping: dict[str, Any], field_name: str, label: str) -> Any:
    try:
        return mapping[field_name]
    except KeyError as exc:
        raise ValueError(f"{label} : champ obligatoire absent : {field_name}.") from exc


def _legacy_id_number(value: object, kind: str, context: str) -> int:
    if not isinstance(value, str):
        raise ValueError(f"{context} : identifiant {kind} legacy invalide : {value!r}.")
    match = _LEGACY_ID_RE[kind].fullmatch(value)
    if match is None:
        raise ValueError(f"{context} : identifiant {kind} legacy invalide : {value!r}.")
    number = int(match.group(1))
    if number <= 0:
        raise ValueError(f"{context} : suffixe legacy {kind} invalide : {value!r}.")
    return number


def _build_mapping(kind: str, values: Iterable[object]) -> tuple[dict[str, str], int]:
    mapping: dict[str, str] = {}
    produced: set[str] = set()
    maximum = 0
    for value in values:
        number = _legacy_id_number(value, kind, "Catalogue V1")
        if not isinstance(value, str):
            raise ValueError(f"Catalogue V1 : identifiant {kind} legacy invalide : {value!r}.")
        new_id = f"{_LEGACY_PREFIXES[kind]}-SYS-{number:06d}"
        if value in mapping:
            raise ValueError(f"Catalogue V1 : identifiant {kind} legacy dupliqué : {value}.")
        if new_id in produced:
            raise ValueError(
                f"Catalogue V1 : collision de normalisation {kind} : {value} produit {new_id}."
            )
        mapping[value] = new_id
        produced.add(new_id)
        maximum = max(maximum, number)
    return mapping, maximum


def build_catalogue_mappings(legacy_catalogue: object) -> CatalogueMappings:
    """Construit tous les mappings avant toute transformation de document."""
    root = _require_mapping(legacy_catalogue, "la racine")
    version = _require_field(root, "version", "Catalogue V1")
    if isinstance(version, bool) or version != 1:
        raise ValueError(f"Catalogue V1 attendu, version reçue : {version!r}.")
    collections = {
        "city": ("cities", "cityId"),
        "beacon": ("beacons", "beaconId"),
        "triangle": ("triangles", "triangleId"),
        "template": ("templates", "templateId"),
    }
    mappings: dict[str, dict[str, str]] = {}
    counters: dict[str, int] = {}
    for kind, (collection_name, id_field) in collections.items():
        items = _require_list(_require_field(root, collection_name, "Catalogue V1"), collection_name)
        values = [
            _require_field(_require_mapping(item, f"{collection_name}[{index}]"), id_field, f"{collection_name}[{index}]")
            for index, item in enumerate(items, start=1)
        ]
        mappings[kind], counters[kind] = _build_mapping(kind, values)
    return CatalogueMappings(
        city=mappings["city"], beacon=mappings["beacon"], triangle=mappings["triangle"],
        template=mappings["template"], counters=counters,
    )


def _mapped(value: object, mapping: dict[str, str], kind: str, context: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{context} : référence {kind} attendue, reçue {value!r}.")
    try:
        return mapping[value]
    except KeyError as exc:
        raise ValueError(f"{context} : référence {kind} legacy introuvable : {value!r}.") from exc


def migrate_catalogue_data_v1_to_v2(legacy_catalogue: object, mappings: CatalogueMappings | None = None) -> dict[str, Any]:
    """Transforme un dictionnaire V1 validé en V2 SYS, sans provider d'allocation."""
    root = _require_mapping(legacy_catalogue, "la racine")
    mappings = build_catalogue_mappings(root) if mappings is None else mappings
    migrated = json.loads(json.dumps(root))
    migrated["version"] = 2
    migrated["idCounters"] = dict(mappings.counters)
    default_template = _require_field(root, "defaultTemplateId", "Catalogue V1")
    migrated["defaultTemplateId"] = (
        None if default_template is None else _mapped(default_template, mappings.template, "template", "defaultTemplateId")
    )
    for old_item, new_item in zip(_require_list(root["cities"], "cities"), migrated["cities"], strict=True):
        new_item["cityId"] = _mapped(old_item["cityId"], mappings.city, "city", "cities.cityId")
    for old_item, new_item in zip(_require_list(root["beacons"], "beacons"), migrated["beacons"], strict=True):
        new_item["beaconId"] = _mapped(old_item["beaconId"], mappings.beacon, "beacon", "beacons.beaconId")
        new_item["cityId"] = _mapped(old_item["cityId"], mappings.city, "city", "beacons.cityId")
    for old_item, new_item in zip(_require_list(root["triangles"], "triangles"), migrated["triangles"], strict=True):
        new_item["triangleId"] = _mapped(old_item["triangleId"], mappings.triangle, "triangle", "triangles.triangleId")
        for field_name in ("openingCityId", "baseCityId", "lightCityId"):
            new_item[field_name] = _mapped(old_item[field_name], mappings.city, "city", f"triangles.{field_name}")
    for old_item, new_item in zip(_require_list(root["templates"], "templates"), migrated["templates"], strict=True):
        new_item["templateId"] = _mapped(old_item["templateId"], mappings.template, "template", "templates.templateId")
        ranks = _require_list(old_item["triangleIdsByRank"], "templates.triangleIdsByRank")
        new_item["triangleIdsByRank"] = [
            None if triangle_id is None else _mapped(triangle_id, mappings.triangle, "triangle", "templates.triangleIdsByRank")
            for triangle_id in ranks
        ]
    catalogue = catalogue_from_dict(migrated, id_provider=SystemCatalogueIdProvider())
    return catalogue_to_dict(catalogue)


def _preserve_or_map_local(value: object, local_prefix: str, mapping: dict[str, str], kind: str, context: str, report: ScenarioMigrationReport) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{context} : référence {kind} invalide : {value!r}.")
    if value.startswith(local_prefix):
        if local_prefix == "SCITY-":
            report.local_scities += 1
        else:
            report.local_stris += 1
        return value
    return _mapped(value, mapping, kind, context)


def _migrate_snapshot(snapshot: object, mappings: CatalogueMappings, report: ScenarioMigrationReport) -> None:
    payload = _require_mapping(snapshot, "topoSnapshot")
    elements = _require_list(_require_field(payload, "elements", "topoSnapshot"), "topoSnapshot.elements")
    for index, element in enumerate(elements, start=1):
        item = _require_mapping(element, f"topoSnapshot.elements[{index}]")
        source_triangle_id = item.get("source_triangle_id")
        if source_triangle_id:
            item["source_triangle_id"] = _preserve_or_map_local(
                source_triangle_id, "STRI-", mappings.triangle, "triangle", "topoSnapshot source_triangle_id", report
            )
            if not str(source_triangle_id).startswith("STRI-"):
                report.triangles += 1
        if "vertex_business_ids" in item:
            business_ids = _require_list(item["vertex_business_ids"], "topoSnapshot vertex_business_ids")
            migrated_ids: list[str | None] = []
            for value in business_ids:
                if value is None:
                    migrated_ids.append(None)
                    continue
                migrated_ids.append(_preserve_or_map_local(
                    value, "SCITY-", mappings.city, "city", "topoSnapshot vertex_business_ids", report
                ))
                if not str(value).startswith("SCITY-"):
                    report.cities += 1
            item["vertex_business_ids"] = migrated_ids
    anchors = _require_list(payload.get("group_anchors", []), "topoSnapshot.group_anchors")
    for index, anchor in enumerate(anchors, start=1):
        item = _require_mapping(anchor, f"topoSnapshot.group_anchors[{index}]")
        item["beacon_id"] = _mapped(
            _require_field(item, "beacon_id", f"topoSnapshot.group_anchors[{index}]"),
            mappings.beacon, "beacon", "topoSnapshot group anchor",
        )
        report.beacons += 1


def migrate_scenario_xml(source: Path, mappings: CatalogueMappings) -> tuple[bytes, ScenarioMigrationReport]:
    """Migre exclusivement les emplacements Catalogue explicitement audités."""
    try:
        tree = ET.parse(source)
    except ET.ParseError as exc:
        raise ValueError(f"Scénario XML invalide {source} : {exc}.") from exc
    root = tree.getroot()
    if root.tag != "scenario":
        raise ValueError(f"Scénario {source} : racine <scenario> attendue.")
    version = root.get("version")
    if version not in {"5", "6"}:
        raise ValueError(f"Scénario {source} : version 5 ou 6 attendue, reçue {version!r}.")
    report = ScenarioMigrationReport(source=source)
    hypothesis = root.find("hypothesis")
    if hypothesis is None:
        raise ValueError(f"Scénario {source} : hypothesis absente.")
    source_template = hypothesis.get("sourceTemplateId")
    if source_template:
        hypothesis.set("sourceTemplateId", _mapped(source_template, mappings.template, "template", "hypothesis.sourceTemplateId"))
        report.templates += 1
    for rank in hypothesis.findall("rank"):
        old_triangle = _require_field(rank.attrib, "triangleId", "hypothesis.rank")
        rank.set("triangleId", _preserve_or_map_local(
            old_triangle, "STRI-", mappings.triangle, "triangle", "hypothesis.rank.triangleId", report
        ))
        if not str(old_triangle).startswith("STRI-"):
            report.triangles += 1
    reference = root.find("scenarioReference")
    if version == "6":
        if reference is None:
            raise ValueError(f"Scénario {source} : scenarioReference absente pour v6.")
        for city in reference.findall("./cities/city"):
            city_ref_id = _require_field(city.attrib, "cityRefId", "scenarioReference.city")
            if not str(city_ref_id).startswith("SCITY-"):
                raise ValueError(f"scenarioReference.cityRefId local attendu : {city_ref_id!r}.")
            report.local_scities += 1
            source_city = city.get("catalogueSourceCityId")
            if source_city is not None:
                city.set("catalogueSourceCityId", _mapped(source_city, mappings.city, "city", "scenarioReference.catalogueSourceCityId"))
                report.cities += 1
        for triangle in reference.findall("./triangles/triangle"):
            triangle_ref_id = _require_field(triangle.attrib, "triangleRefId", "scenarioReference.triangle")
            if not str(triangle_ref_id).startswith("STRI-"):
                raise ValueError(f"scenarioReference.triangleRefId local attendu : {triangle_ref_id!r}.")
            report.local_stris += 1
            source_triangle = triangle.get("catalogueSourceTriangleId")
            if source_triangle is not None:
                triangle.set("catalogueSourceTriangleId", _mapped(source_triangle, mappings.triangle, "triangle", "scenarioReference.catalogueSourceTriangleId"))
                report.triangles += 1
            for field_name in ("openingCityRefId", "baseCityRefId", "lightCityRefId"):
                old_city = _require_field(triangle.attrib, field_name, "scenarioReference.triangle")
                triangle.set(field_name, _preserve_or_map_local(
                    old_city, "SCITY-", mappings.city, "city", f"scenarioReference.{field_name}", report
                ))
                if not str(old_city).startswith("SCITY-"):
                    report.cities += 1
    elif reference is not None:
        raise ValueError(f"Scénario {source} : scenarioReference interdite pour v5.")
    snapshot_element = root.find("topoSnapshot")
    if snapshot_element is None or snapshot_element.get("encoding") != "json" or not (snapshot_element.text or "").strip():
        raise ValueError(f"Scénario {source} : topoSnapshot JSON absent.")
    try:
        snapshot = json.loads(snapshot_element.text or "")
    except json.JSONDecodeError as exc:
        raise ValueError(f"Scénario {source} : topoSnapshot JSON invalide : {exc.msg}.") from exc
    _migrate_snapshot(snapshot, mappings, report)
    snapshot_element.text = json.dumps(snapshot, ensure_ascii=False, separators=(",", ":"))
    _validate_migrated_scenario(root, mappings)
    return ET.tostring(root, encoding="utf-8", xml_declaration=True), report


def _validate_migrated_scenario(root: ET.Element, mappings: CatalogueMappings) -> None:
    """Validation croisée des seuls emplacements Catalogue migrés."""
    valid = {kind: set(mappings.for_kind(kind).values()) for kind in CATALOGUE_ID_KIND_ORDER}
    version = root.get("version")
    local_city_ids = {
        city.get("cityRefId")
        for city in root.findall("./scenarioReference/cities/city")
    }
    local_triangle_ids = {
        triangle.get("triangleRefId")
        for triangle in root.findall("./scenarioReference/triangles/triangle")
    }
    hypothesis = root.find("hypothesis")
    if hypothesis is None:
        raise ValueError("Scénario migré : hypothesis absente.")
    source_template = hypothesis.get("sourceTemplateId")
    if source_template is not None and source_template not in valid["template"]:
        raise ValueError(f"Scénario migré : template Catalogue absent : {source_template!r}.")
    for rank in hypothesis.findall("rank"):
        value = rank.get("triangleId")
        is_known_local = isinstance(value, str) and value.startswith("STRI-") and value in local_triangle_ids
        if not (isinstance(value, str) and (value in valid["triangle"] or (version == "6" and is_known_local))):
            raise ValueError(f"Scénario migré : triangle d'hypothèse invalide : {value!r}.")
    for city in root.findall("./scenarioReference/cities/city"):
        value = city.get("catalogueSourceCityId")
        if value is not None and value not in valid["city"]:
            raise ValueError(f"Scénario migré : ville Catalogue absente : {value!r}.")
    for triangle in root.findall("./scenarioReference/triangles/triangle"):
        source_triangle = triangle.get("catalogueSourceTriangleId")
        if source_triangle is not None and source_triangle not in valid["triangle"]:
            raise ValueError(f"Scénario migré : triangle Catalogue absent : {source_triangle!r}.")
        for field_name in ("openingCityRefId", "baseCityRefId", "lightCityRefId"):
            value = triangle.get(field_name)
            is_known_local = isinstance(value, str) and value.startswith("SCITY-") and value in local_city_ids
            if not (isinstance(value, str) and (value in valid["city"] or is_known_local)):
                raise ValueError(f"Scénario migré : référence ville invalide : {value!r}.")
    snapshot_element = root.find("topoSnapshot")
    if snapshot_element is None:
        raise ValueError("Scénario migré : topoSnapshot absent.")
    snapshot = json.loads(snapshot_element.text or "")
    for element in _require_list(snapshot.get("elements"), "topoSnapshot.elements"):
        item = _require_mapping(element, "topoSnapshot.element")
        source_triangle = item.get("source_triangle_id")
        is_known_local_triangle = (
            isinstance(source_triangle, str)
            and source_triangle.startswith("STRI-")
            and source_triangle in local_triangle_ids
        )
        if source_triangle and source_triangle not in valid["triangle"] and not (version == "6" and is_known_local_triangle):
            raise ValueError(f"Scénario migré : source_triangle_id invalide : {source_triangle!r}.")
        for city_id in item.get("vertex_business_ids", []):
            is_known_local_city = (
                isinstance(city_id, str)
                and city_id.startswith("SCITY-")
                and city_id in local_city_ids
            )
            if city_id is not None and city_id not in valid["city"] and not is_known_local_city:
                raise ValueError(f"Scénario migré : vertex_business_ids invalide : {city_id!r}.")
    for anchor in snapshot.get("group_anchors", []):
        beacon_id = _require_mapping(anchor, "topoSnapshot.group_anchor").get("beacon_id")
        if beacon_id not in valid["beacon"]:
            raise ValueError(f"Scénario migré : beacon d'ancrage absent : {beacon_id!r}.")


def _read_legacy_catalogue(path: Path) -> dict[str, Any]:
    try:
        return _require_mapping(json.loads(path.read_text(encoding="utf-8")), "la racine")
    except json.JSONDecodeError as exc:
        raise ValueError(f"Catalogue V1 JSON invalide {path} : {exc.msg}.") from exc


def _ensure_destinations(sources: list[Path], destinations: list[Path], force: bool) -> None:
    if len(set(destinations)) != len(destinations):
        raise ValueError("Les destinations de migration doivent être distinctes.")
    source_paths = {path.resolve() for path in sources}
    for destination in destinations:
        if destination.resolve() in source_paths:
            raise ValueError(f"La destination ne doit jamais écraser une source : {destination}.")
        if destination.exists() and not force:
            raise FileExistsError(f"Destination déjà existante : {destination} (utiliser --force).")


def _write_atomic_batch(outputs: list[tuple[Path, bytes]], force: bool) -> None:
    temporary_paths: list[tuple[Path, Path]] = []
    backups: list[tuple[Path, Path]] = []
    published: list[Path] = []
    try:
        for destination, content in outputs:
            destination.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(mode="wb", dir=destination.parent, prefix=f".{destination.name}.", suffix=".tmp", delete=False) as stream:
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
                temporary_paths.append((destination, Path(stream.name)))
        for destination, temporary in temporary_paths:
            if destination.exists() and not force:
                raise FileExistsError(f"Destination déjà existante : {destination}.")
            if destination.exists():
                with tempfile.NamedTemporaryFile(
                    mode="wb", dir=destination.parent, prefix=f".{destination.name}.", suffix=".backup", delete=False
                ) as stream:
                    backup = Path(stream.name)
                os.replace(destination, backup)
                backups.append((destination, backup))
            os.replace(temporary, destination)
            published.append(destination)
    except OSError:
        for destination in reversed(published):
            destination.unlink(missing_ok=True)
        for destination, backup in reversed(backups):
            os.replace(backup, destination)
        for _, temporary in temporary_paths:
            temporary.unlink(missing_ok=True)
        raise
    for _, backup in backups:
        backup.unlink(missing_ok=True)


def migrate_paths(
    catalogue_in: str | Path,
    catalogue_out: str | Path,
    scenarios: Iterable[tuple[str | Path, str | Path]],
    *,
    force: bool = False,
    dry_run: bool = False,
) -> MigrationReport:
    source_catalogue = Path(catalogue_in)
    destination_catalogue = Path(catalogue_out)
    scenario_paths = [(Path(source), Path(destination)) for source, destination in scenarios]
    if not scenario_paths:
        raise ValueError("Au moins un scénario doit être fourni avec --scenario.")
    _ensure_destinations(
        [source_catalogue, *(source for source, _ in scenario_paths)],
        [destination_catalogue, *(destination for _, destination in scenario_paths)], force,
    )
    legacy_catalogue = _read_legacy_catalogue(source_catalogue)
    mappings = build_catalogue_mappings(legacy_catalogue)
    migrated_catalogue = migrate_catalogue_data_v1_to_v2(legacy_catalogue, mappings)
    scenario_outputs = [
        (destination, *migrate_scenario_xml(source, mappings))
        for source, destination in scenario_paths
    ]
    report = MigrationReport(mappings, [item[2] for item in scenario_outputs])
    if not dry_run:
        outputs = [
            (destination_catalogue, (json.dumps(migrated_catalogue, ensure_ascii=False, indent=2) + "\n").encode("utf-8")),
            *((destination, xml_bytes) for destination, xml_bytes, _ in scenario_outputs),
        ]
        _write_atomic_batch(outputs, force)
    return report


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalogue-in", required=True, type=Path)
    parser.add_argument("--catalogue-out", required=True, type=Path)
    parser.add_argument("--scenario", nargs=2, action="append", metavar=("SOURCE", "DESTINATION"), required=True)
    parser.add_argument("--force", action="store_true", help="autorise l'écrasement explicite des destinations")
    parser.add_argument("--dry-run", action="store_true", help="valide toute la migration sans écrire de sortie")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        report = migrate_paths(
            args.catalogue_in, args.catalogue_out, args.scenario,
            force=args.force, dry_run=args.dry_run,
        )
    except (OSError, ValueError) as exc:
        print(f"MIGRATION FAILED: {exc}")
        return 1
    print(report.render())
    print("MIGRATION SUCCESS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
