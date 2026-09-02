"""Migration explicite, one-shot et atomique d'un Catalogue V3 vers V4."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any


_V3_ROOT_KEYS = {
    "version", "idCounters", "defaultTemplateId", "defaultMapId", "cities",
    "beacons", "triangles", "templates", "maps",
}
_REFERENCE_MAP_ID = "MAP-SYS-000002"
_REFERENCE_MAP = {
    "mapId": _REFERENCE_MAP_ID,
    "name": "France Michelin",
    "imageFile": "france_michelin.jpg",
    "calibrationPointsFile": None,
    "calibrationFile": "france_michelin.json",
    "projection": "EPSG:2154",
    "defaultWorldRect": {"x0": 0.0, "y0": 0.0, "w": 9999.0, "h": 9999.0},
    "defaultScaleFactor": 1.0,
    "archived": False,
}


def migrate_catalogue_data_v3_to_v4(data: object) -> dict[str, Any]:
    """Transforme strictement le seed SYS V3 connu en contrat V4."""
    if not isinstance(data, dict) or set(data) != _V3_ROOT_KEYS or data.get("version") != 3:
        raise ValueError("Catalogue V3 attendu avec le contrat root strict.")
    maps = data.get("maps")
    counters = data.get("idCounters")
    if not isinstance(maps, list) or not isinstance(counters, dict):
        raise ValueError("Catalogue V3 invalide : maps ou idCounters.")
    map_ids = {item.get("mapId") for item in maps if isinstance(item, dict)}
    if "MAP-SYS-000001" not in map_ids:
        raise ValueError("Catalogue V3 SYS invalide : MAP-SYS-000001 absente.")
    if _REFERENCE_MAP_ID in map_ids:
        raise ValueError(f"Catalogue V3 SYS invalide : {_REFERENCE_MAP_ID} déjà présente.")
    map_counter = counters.get("map")
    if isinstance(map_counter, bool) or not isinstance(map_counter, int) or map_counter < 1:
        raise ValueError("Catalogue V3 SYS invalide : compteur map incohérent.")
    migrated = {key: value for key, value in data.items()}
    migrated["version"] = 4
    migrated["idCounters"] = {**counters, "map": max(map_counter, 2)}
    migrated["catalogueReferenceMapId"] = _REFERENCE_MAP_ID
    migrated["maps"] = [*maps, dict(_REFERENCE_MAP)]
    return migrated


def migrate_catalogue_file_v3_to_v4(source: str | Path, *, force: bool = False) -> Path:
    path = Path(source)
    backup = Path(str(path) + ".pre-catalogue-v4")
    if backup.exists() and not force:
        raise FileExistsError(f"Backup déjà présent : {backup}")
    data = json.loads(path.read_text(encoding="utf-8"))
    migrated = migrate_catalogue_data_v3_to_v4(data)
    if not backup.exists():
        shutil.copy2(path, backup)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(migrated, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)
    return backup


def main() -> int:
    parser = argparse.ArgumentParser(description="Migre un Catalogue SYS V3 vers V4.")
    parser.add_argument("catalogue")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    migrate_catalogue_file_v3_to_v4(args.catalogue, force=args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
