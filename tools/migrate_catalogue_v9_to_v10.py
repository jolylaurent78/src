"""Migration explicite et atomique du Catalogue V9 vers V10."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any


_V9_ROOT_KEYS = {
    "version", "idCounters", "defaultTemplateId", "defaultMapId", "catalogueReferenceMapId",
    "cities", "beacons", "triangles", "templates", "maps", "defaultBookId", "books",
    "geometricLayerDisplayOverrides", "geometricLayers",
}


def migrate_catalogue_data_v9_to_v10(data: object) -> dict[str, Any]:
    if not isinstance(data, dict) or set(data) != _V9_ROOT_KEYS or data.get("version") != 9:
        raise ValueError("Catalogue V9 attendu avec le contrat root strict.")
    return {**data, "version": 10, "beaconGroupColors": {}}


def migrate_catalogue_file_v9_to_v10(source: str | Path, *, force: bool = False) -> Path:
    path = Path(source)
    backup = Path(str(path) + ".pre-catalogue-v10")
    if backup.exists() and not force:
        raise FileExistsError(f"Backup déjà présent : {backup}")
    migrated = migrate_catalogue_data_v9_to_v10(json.loads(path.read_text(encoding="utf-8")))
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
    parser = argparse.ArgumentParser(description="Migre un Catalogue V9 vers V10.")
    parser.add_argument("catalogue")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    migrate_catalogue_file_v9_to_v10(args.catalogue, force=args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
