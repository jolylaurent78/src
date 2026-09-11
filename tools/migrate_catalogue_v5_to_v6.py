"""Migration explicite et atomique du Catalogue V5 vers V6."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any


_V5_ROOT_KEYS = {
    "version", "idCounters", "defaultTemplateId", "defaultMapId", "defaultBookId",
    "catalogueReferenceMapId", "cities", "beacons", "triangles", "templates", "maps", "books",
}
_V5_LEGACY_ROOT_KEYS = _V5_ROOT_KEYS - {"defaultBookId", "books"}


def migrate_catalogue_data_v5_to_v6(data: object) -> dict[str, Any]:
    """Ajoute la collection vide des associations de calques, sans autre mutation."""
    if not isinstance(data, dict) or set(data) not in (_V5_ROOT_KEYS, _V5_LEGACY_ROOT_KEYS) or data.get("version") != 5:
        raise ValueError("Catalogue V5 attendu avec le contrat root strict.")
    migrated = dict(data)
    if set(migrated) == _V5_LEGACY_ROOT_KEYS:
        counters = dict(migrated["idCounters"])
        counters["book"] = 1
        migrated["idCounters"] = counters
        migrated["defaultBookId"] = "BOOK-SYS-000001"
        migrated["books"] = [{
            "bookId": "BOOK-SYS-000001", "name": "Livre", "assetFile": "books/livre.txt",
            "archived": False, "description": "",
        }]
    return {**migrated, "version": 6, "geometricLayers": {}}


def migrate_catalogue_file_v5_to_v6(source: str | Path, *, force: bool = False) -> Path:
    path = Path(source)
    backup = Path(str(path) + ".pre-catalogue-v6")
    if backup.exists() and not force:
        raise FileExistsError(f"Backup déjà présent : {backup}")
    data = json.loads(path.read_text(encoding="utf-8"))
    migrated = migrate_catalogue_data_v5_to_v6(data)
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
    parser = argparse.ArgumentParser(description="Migre un Catalogue V5 vers V6.")
    parser.add_argument("catalogue")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    migrate_catalogue_file_v5_to_v6(args.catalogue, force=args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
