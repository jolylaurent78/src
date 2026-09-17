"""Migration explicite et atomique du Catalogue V8 vers V9."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any


_V8_ROOT_KEYS = {
    "version", "idCounters", "defaultTemplateId", "defaultMapId", "catalogueReferenceMapId",
    "cities", "beacons", "triangles", "templates", "maps", "defaultBookId", "books",
    "geometricLayerDisplayOverrides", "geometricLayers",
}
_V8_BEACON_KEYS = {"beaconId", "cityId", "archived"}


def migrate_catalogue_data_v8_to_v9(data: object) -> dict[str, Any]:
    """Ajoute les métadonnées de balise, sans qualifier les données historiques."""
    if not isinstance(data, dict) or set(data) != _V8_ROOT_KEYS or data.get("version") != 8:
        raise ValueError("Catalogue V8 attendu avec le contrat root strict.")
    raw_beacons = data.get("beacons")
    if not isinstance(raw_beacons, list):
        raise ValueError("Catalogue V8 attendu avec beacons liste.")
    migrated_beacons: list[dict[str, Any]] = []
    for index, beacon in enumerate(raw_beacons, start=1):
        if not isinstance(beacon, dict) or set(beacon) != _V8_BEACON_KEYS:
            raise ValueError(f"Catalogue V8 attendu avec beacons[{index}] strict.")
        migrated_beacons.append({
            "beaconId": beacon["beaconId"],
            "cityId": beacon["cityId"],
            "group": "",
            "order": None,
            "usableAsAnchor": True,
            "note": "",
            "archived": beacon["archived"],
        })
    return {**data, "version": 9, "beacons": migrated_beacons}


def migrate_catalogue_file_v8_to_v9(source: str | Path, *, force: bool = False) -> Path:
    path = Path(source)
    backup = Path(str(path) + ".pre-catalogue-v9")
    if backup.exists() and not force:
        raise FileExistsError(f"Backup déjà présent : {backup}")
    migrated = migrate_catalogue_data_v8_to_v9(json.loads(path.read_text(encoding="utf-8")))
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
    parser = argparse.ArgumentParser(description="Migre un Catalogue V8 vers V9.")
    parser.add_argument("catalogue")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    migrate_catalogue_file_v8_to_v9(args.catalogue, force=args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
