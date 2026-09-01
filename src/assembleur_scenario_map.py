"""Références de cartes portables pour les scénarios XML."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import shutil
import tempfile
import xml.etree.ElementTree as ET


_BACKUP_SUFFIX = ".pre-packaging-004-map"


@dataclass(frozen=True)
class ScenarioMapMigrationResult:
    path: Path
    changed: bool
    old_path: str | None
    resource: str | None
    backup_path: Path | None


def validate_resource_name(resource: object) -> str:
    value = str(resource or "").strip()
    if not value:
        raise ValueError("Référence de carte resource vide.")
    if "/" in value or "\\" in value or ".." in value or Path(value).name != value:
        raise ValueError(f"Référence de carte resource invalide : {value!r}")
    return value


def resolve_resource_map(resource: object, resource_maps_dir: str | Path) -> Path:
    name = validate_resource_name(resource)
    candidate = Path(resource_maps_dir) / name
    if not candidate.is_file():
        raise FileNotFoundError(f"Ressource cartographique introuvable : {candidate}")
    return candidate


def resource_name_for_path(map_path: object, resource_maps_dir: str | Path) -> str | None:
    value = str(map_path or "").strip()
    if not value:
        return None
    name = Path(value).name
    candidate = Path(resource_maps_dir) / name
    if not candidate.is_file():
        return None
    try:
        return name if Path(value).resolve() == candidate.resolve() else None
    except OSError:
        return None


def migrate_scenario_map_path(
    scenario_path: str | Path,
    *,
    resource_maps_dir: str | Path,
    force: bool = False,
) -> ScenarioMapMigrationResult:
    """Convertit un ``map@path`` livré en ``map@resource`` avec backup atomique."""
    source = Path(scenario_path)
    tree = ET.parse(source)
    root = tree.getroot()
    if root.tag != "scenario":
        raise ValueError(f"Fichier scénario invalide (racine attendue scenario) : {source}")
    map_element = root.find("map")
    if map_element is None:
        return ScenarioMapMigrationResult(source, False, None, None, None)
    old_path = str(map_element.get("path", "") or "").strip()
    if not old_path:
        return ScenarioMapMigrationResult(source, False, old_path, map_element.get("resource"), None)
    name = Path(old_path).name
    if not (Path(resource_maps_dir) / name).is_file():
        return ScenarioMapMigrationResult(source, False, old_path, None, None)

    backup = Path(str(source) + _BACKUP_SUFFIX)
    if backup.exists() and not force:
        raise FileExistsError(f"Backup de migration déjà existant : {backup}")
    if not backup.exists():
        shutil.copy2(source, backup)
    map_element.attrib.pop("path", None)
    map_element.set("resource", name)
    _atomic_write_xml(tree, source)
    return ScenarioMapMigrationResult(source, True, old_path, name, backup)


def _atomic_write_xml(tree: ET.ElementTree, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    try:
        with os.fdopen(fd, "wb") as stream:
            tree.write(stream, encoding="utf-8", xml_declaration=True)
        os.replace(temporary_name, destination)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)
