"""Migration explicite et atomique du Catalogue V7 vers V8."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any


_V7_ROOT_KEYS = {
    "version", "idCounters", "defaultTemplateId", "defaultMapId", "defaultBookId",
    "catalogueReferenceMapId", "cities", "beacons", "triangles", "templates", "maps", "books", "geometricLayers",
}


def migrate_catalogue_data_v7_to_v8(data: object) -> dict[str, Any]:
    """Déplace le contrat vers V8 en abandonnant les overrides par calque."""
    if not isinstance(data, dict) or set(data) != _V7_ROOT_KEYS or data.get("version") != 7:
        raise ValueError("Catalogue V7 attendu avec le contrat root strict.")
    layers = data.get("geometricLayers")
    if not isinstance(layers, dict):
        raise ValueError("Catalogue V7 attendu avec geometricLayers objet.")
    migrated_layers: dict[str, dict[str, Any]] = {}
    for base_city_id, raw_layer in layers.items():
        if not isinstance(raw_layer, dict) or set(raw_layer) != {"asset", "displayOverrides"}:
            raise ValueError("Catalogue V7 attendu avec asset et displayOverrides par calque.")
        migrated_layers[base_city_id] = {"asset": raw_layer["asset"]}
    return {
        **data,
        "version": 8,
        "geometricLayerDisplayOverrides": {},
        "geometricLayers": migrated_layers,
    }


def migrate_catalogue_file_v7_to_v8(source: str | Path, *, force: bool = False) -> Path:
    path = Path(source)
    backup = Path(str(path) + ".pre-catalogue-v8")
    if backup.exists() and not force:
        raise FileExistsError(f"Backup déjà présent : {backup}")
    migrated = migrate_catalogue_data_v7_to_v8(json.loads(path.read_text(encoding="utf-8")))
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
    parser = argparse.ArgumentParser(description="Migre un Catalogue V7 vers V8.")
    parser.add_argument("catalogue")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    migrate_catalogue_file_v7_to_v8(args.catalogue, force=args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
