"""Contrat portable et migration one-shot de la configuration utilisateur."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping


_OBSOLETE_KEYS = frozenset(
    {
        "lastTriangleExcel",
        "lastTriangleCsvIn",
        "lastVillesCsvIn",
        "lastTriangleExcelOut",
        "cheminsBaliseRefName",
    }
)
_LEGACY_DEVELOPMENT_PATH_MARKERS = (
    "\\dropbox\\la chouette\\python\\assembleurtriangles\\",
    "\\dropbox\\la chouette\\python\\algosimulator\\",
)


def load_config_file(path: str | Path) -> dict[str, Any]:
    """Charge une configuration existante et rejette explicitement tout JSON invalide."""
    source = Path(path)
    try:
        with source.open("r", encoding="utf-8") as stream:
            data = json.load(stream)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Configuration JSON invalide : {exc.msg}.") from exc
    except OSError as exc:
        raise OSError(f"Impossible de lire la configuration {source}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError("Configuration invalide : la racine JSON doit être un objet.")
    return data


def save_config_file(config: Mapping[str, Any], path: str | Path) -> None:
    """Écrit une configuration JSON avec remplacement atomique."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            json.dump(dict(config), stream, ensure_ascii=False, indent=2)
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


def _is_legacy_development_path(value: object) -> bool:
    if not isinstance(value, str):
        return False
    normalized = value.replace("/", "\\").lower()
    return any(marker in normalized for marker in _LEGACY_DEVELOPMENT_PATH_MARKERS)


def _resource_map_name(value: object, resource_maps_dir: Path) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    candidate = Path(value).name
    if candidate != value.replace("\\", "/").split("/")[-1]:
        return None
    return candidate if (resource_maps_dir / candidate).is_file() else None


def migrate_legacy_config(
    legacy_config: Mapping[str, Any], *, resource_maps_dir: str | Path
) -> dict[str, Any]:
    """Convertit la config historique sans imposer de migration de préférences."""
    migrated = dict(legacy_config)
    maps_dir = Path(resource_maps_dir)

    for key in _OBSOLETE_KEYS:
        migrated.pop(key, None)

    legacy_background = migrated.get("bgSvgPath")
    map_name = _resource_map_name(legacy_background, maps_dir)
    if map_name is not None:
        migrated["bgMap"] = map_name
        migrated.pop("bgSvgPath", None)
    elif _is_legacy_development_path(legacy_background):
        migrated.pop("bgSvgPath", None)

    current_map = migrated.get("bgMap")
    if isinstance(current_map, str):
        normalized_map_name = _resource_map_name(current_map, maps_dir)
        if normalized_map_name is not None:
            migrated["bgMap"] = normalized_map_name
        elif _is_legacy_development_path(current_map):
            migrated.pop("bgMap", None)

    return migrated


def migrate_legacy_config_file(
    source: str | Path,
    destination: str | Path,
    *,
    resource_maps_dir: str | Path,
) -> dict[str, Any]:
    """Migre une seule fois et refuse d'écraser une configuration utilisateur."""
    target = Path(destination)
    if target.exists():
        raise FileExistsError(
            "Migration de configuration refusée : destination déjà existante : "
            f"{target}"
        )
    migrated = migrate_legacy_config(
        load_config_file(source), resource_maps_dir=resource_maps_dir
    )
    save_config_file(migrated, target)
    return load_config_file(target)
