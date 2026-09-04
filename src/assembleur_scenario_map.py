"""Références de cartes portables pour les scénarios XML."""

from __future__ import annotations

from dataclasses import dataclass
import math
import os
from pathlib import Path
import shutil
import tempfile
import xml.etree.ElementTree as ET

from src.assembleur_catalogue import Catalogue
from src.assembleur_catalogue_identity import is_catalogue_map_id
from src.assembleur_map_transform import validate_finite_number, validate_scale_factor


_BACKUP_SUFFIX = ".pre-packaging-004-map"
_LEGACY_DEFAULT_MAP_RESOURCE = "899 - Alsace.jpg"
_LEGACY_DEFAULT_MAP_ID = "MAP-SYS-000001"


@dataclass(frozen=True)
class ScenarioMapPosition:
    """Position monde explicite d'une carte de scénario, sans dimension."""

    x0: float
    y0: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "x0", validate_finite_number(self.x0, label="position_override.x0"))
        object.__setattr__(self, "y0", validate_finite_number(self.y0, label="position_override.y0"))


@dataclass(frozen=True)
class ScenarioMapState:
    """Etat runtime explicite de la carte référencée par un scénario.

    Ce modèle ne sérialise pas encore le XML legacy. La position et l'échelle
    sont indépendantes ; les dimensions restent toujours dérivées de l'échelle.
    """

    map_ref_id: str | None
    position_override: ScenarioMapPosition | None = None
    scale_factor_override: float | None = None
    visible: bool = True

    def __post_init__(self) -> None:
        if self.map_ref_id is not None:
            if not isinstance(self.map_ref_id, str) or not self.map_ref_id.strip():
                raise ValueError("map_ref_id doit être une chaîne non vide ou None.")
            if not is_catalogue_map_id(self.map_ref_id):
                raise ValueError(f"map_ref_id invalide : {self.map_ref_id!r}.")
        if self.position_override is not None and not isinstance(
            self.position_override, ScenarioMapPosition
        ):
            raise ValueError("position_override doit être un ScenarioMapPosition ou None.")
        if self.scale_factor_override is not None:
            object.__setattr__(
                self,
                "scale_factor_override",
                validate_scale_factor(self.scale_factor_override, label="scale_factor_override"),
            )
        if not isinstance(self.visible, bool):
            raise ValueError("visible doit être un booléen.")


def scenario_map_state_to_xml_attributes(state: ScenarioMapState) -> dict[str, str]:
    """Sérialise uniquement le contrat XML CatalogueMap courant."""
    if not isinstance(state, ScenarioMapState):
        raise TypeError("scenario_map_state_to_xml_attributes exige un ScenarioMapState.")
    if state.map_ref_id is None:
        raise ValueError("Une carte absente ne possède pas d'élément XML <map>.")
    attributes = {
        "refId": state.map_ref_id,
        "visible": "true" if state.visible else "false",
    }
    if state.position_override is not None:
        attributes["x0"] = f"{state.position_override.x0:.12g}"
        attributes["y0"] = f"{state.position_override.y0:.12g}"
    if state.scale_factor_override is not None:
        attributes["scale"] = f"{state.scale_factor_override:.12g}"
    return attributes


def scenario_map_state_from_xml_attributes(attributes: dict[str, str]) -> ScenarioMapState:
    """Lit le format XML cible, sans interpréter aucun chemin ou basename."""
    ref_id = str(attributes.get("refId", "") or "").strip()
    if not ref_id:
        raise ValueError("Map scenario invalide : refId est obligatoire.")
    has_x0 = "x0" in attributes
    has_y0 = "y0" in attributes
    if has_x0 != has_y0:
        raise ValueError("Map scenario invalide : x0 et y0 doivent être fournis ensemble.")
    position = (
        None
        if not has_x0
        else ScenarioMapPosition(
            _parse_map_number(attributes["x0"], "x0"),
            _parse_map_number(attributes["y0"], "y0"),
        )
    )
    scale = (
        None
        if "scale" not in attributes
        else validate_scale_factor(_parse_map_number(attributes["scale"], "scale"), label="scale")
    )
    return ScenarioMapState(
        map_ref_id=ref_id,
        position_override=position,
        scale_factor_override=scale,
        visible=_parse_map_visible(attributes.get("visible", "true")),
    )


def migrate_legacy_map_attributes(
    catalogue: Catalogue,
    attributes: dict[str, str],
) -> ScenarioMapState:
    """Convertit le seul format historique livré, sans fallback runtime."""
    resource = str(attributes.get("resource", "") or "").strip()
    path = str(attributes.get("path", "") or "").strip()
    if resource and path:
        raise ValueError("Map scenario legacy invalide : path et resource sont mutuellement exclusifs.")
    if resource and (Path(resource).name != resource or "/" in resource or "\\" in resource):
        raise ValueError(f"Map scenario legacy non migrable : resource invalide {resource!r}.")
    basename = Path(resource or path).name
    if basename != _LEGACY_DEFAULT_MAP_RESOURCE:
        raise ValueError(
            "Map scenario legacy non migrable sans import explicite : "
            f"{resource or path!r}."
        )
    catalogue_map = catalogue.get_map(_LEGACY_DEFAULT_MAP_ID)
    default = catalogue_map.default_world_rect
    x0 = _parse_map_number(attributes.get("x0", str(default.x0)), "x0")
    y0 = _parse_map_number(attributes.get("y0", str(default.y0)), "y0")
    width = _parse_map_number(attributes.get("w", str(default.w)), "w")
    height = _parse_map_number(attributes.get("h", str(default.h)), "h")
    if width <= 0 or height <= 0:
        raise ValueError("Map scenario legacy invalide : w et h doivent être strictement positifs.")
    if not math.isclose(width / height, default.w / default.h, rel_tol=1e-5, abs_tol=1e-8):
        raise ValueError("Map scenario legacy invalide : ratio w/h anisotrope.")
    same_position = math.isclose(x0, default.x0, rel_tol=1e-6, abs_tol=1e-6) and math.isclose(y0, default.y0, rel_tol=1e-6, abs_tol=1e-6)
    same_size = math.isclose(width, default.w, rel_tol=1e-6, abs_tol=1e-6) and math.isclose(height, default.h, rel_tol=1e-6, abs_tol=1e-6)
    return ScenarioMapState(
        map_ref_id=_LEGACY_DEFAULT_MAP_ID,
        position_override=None if same_position else ScenarioMapPosition(x0, y0),
        scale_factor_override=None if same_size else catalogue_map.default_scale_factor * width / default.w,
        visible=_parse_legacy_map_visible(attributes.get("visible", "1")),
    )


def _parse_map_number(value: object, label: str) -> float:
    try:
        return validate_finite_number(float(str(value)), label=label)
    except ValueError as exc:
        raise ValueError(f"Map scenario invalide : {label} doit être numérique et fini.") from exc


def _parse_map_visible(value: object) -> bool:
    normalized = str(value).strip().casefold()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise ValueError("Map scenario invalide : visible doit être true ou false.")


def _parse_legacy_map_visible(value: object) -> bool:
    normalized = str(value).strip().casefold()
    if normalized in {"1", "true"}:
        return True
    if normalized in {"0", "false"}:
        return False
    raise ValueError("Map scenario legacy invalide : visible doit être 0, 1, true ou false.")


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
