"""Migration explicite et atomique du Catalogue V6 vers V7."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any


_V6_ROOT_KEYS = {
    "version", "idCounters", "defaultTemplateId", "defaultMapId", "defaultBookId",
    "catalogueReferenceMapId", "cities", "beacons", "triangles", "templates", "maps", "books", "geometricLayers",
}


def migrate_catalogue_data_v6_to_v7(data: object) -> dict[str, Any]:
    """Ajoute les overrides d'affichage vides sans autre mutation."""
    if not isinstance(data, dict) or set(data) != _V6_ROOT_KEYS or data.get("version") != 6:
        raise ValueError("Catalogue V6 attendu avec le contrat root strict.")
    layers = data.get("geometricLayers")
    if not isinstance(layers, dict):
        raise ValueError("Catalogue V6 attendu avec geometricLayers objet.")
    migrated_layers: dict[str, dict[str, Any]] = {}
    for base_city_id, raw_layer in layers.items():
        if not isinstance(raw_layer, dict) or set(raw_layer) != {"asset"}:
            raise ValueError("Catalogue V6 attendu avec des calques ne contenant que asset.")
        migrated_layers[base_city_id] = {"asset": raw_layer["asset"], "displayOverrides": {}}
    return {**data, "version": 7, "geometricLayers": migrated_layers}


def migrate_catalogue_file_v6_to_v7(source: str | Path, *, force: bool = False) -> Path:
    path = Path(source)
    backup = Path(str(path) + ".pre-catalogue-v7")
    if backup.exists() and not force:
        raise FileExistsError(f"Backup déjà présent : {backup}")
    migrated = migrate_catalogue_data_v6_to_v7(json.loads(path.read_text(encoding="utf-8")))
    if not backup.exists():
        shutil.copy2(path, backup)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(migrated, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return backup


def main() -> int:
    parser = argparse.ArgumentParser(description="Migre un Catalogue V6 vers V7.")
    parser.add_argument("catalogue")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    migrate_catalogue_file_v6_to_v7(args.catalogue, force=args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
