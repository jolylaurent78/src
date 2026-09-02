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


def migrate_legacy_config(legacy_config: Mapping[str, Any]) -> dict[str, Any]:
    """Supprime les préférences obsolètes sans modifier les préférences valides."""
    migrated = dict(legacy_config)
    for key in _OBSOLETE_KEYS:
        migrated.pop(key, None)
    return migrated


def migrate_legacy_config_file(
    source: str | Path,
    destination: str | Path,
) -> dict[str, Any]:
    """Migre une seule fois et refuse d'écraser une configuration utilisateur."""
    target = Path(destination)
    if target.exists():
        raise FileExistsError(
            "Migration de configuration refusée : destination déjà existante : "
            f"{target}"
        )
    migrated = migrate_legacy_config(load_config_file(source))
    save_config_file(migrated, target)
    return load_config_file(target)
