"""Migration explicite et atomique du seed Catalogue V4 vers V5."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any


_V4_ROOT_KEYS = {
    "version", "idCounters", "defaultTemplateId", "defaultMapId",
    "catalogueReferenceMapId", "cities", "beacons", "triangles", "templates", "maps",
}
_V4_MAP_KEYS = {
    "mapId", "name", "imageFile", "calibrationPointsFile", "calibrationFile",
    "projection", "defaultWorldRect", "defaultScaleFactor", "archived",
}
_SYS_CALIBRATION_CITY_IDS = {
    "MAP-SYS-000001": [
        "CITY-SYS-000105",  # Plombières
        "CITY-SYS-000106",  # Mont Sainte-Odile
        "CITY-SYS-000094",  # Grand Ballon
        "CITY-SYS-000004",  # Forbach
        "CITY-SYS-000037",  # Carignan
    ],
    "MAP-SYS-000002": [
        "CITY-SYS-000009", "CITY-SYS-000002", "CITY-SYS-000010",
        "CITY-SYS-000005", "CITY-SYS-000052",
    ],
}


def migrate_catalogue_data_v4_to_v5(data: object) -> dict[str, Any]:
    if not isinstance(data, dict) or set(data) != _V4_ROOT_KEYS or data.get("version") != 4:
        raise ValueError("Catalogue V4 attendu avec le contrat root strict.")
    maps = data.get("maps")
    if not isinstance(maps, list):
        raise ValueError("Catalogue V4 invalide : maps doit être une liste.")
    cities = data.get("cities")
    city_ids = {item.get("cityId") for item in cities if isinstance(item, dict)} if isinstance(cities, list) else set()
    migrated_maps = []
    for raw_map in maps:
        if not isinstance(raw_map, dict) or set(raw_map) != _V4_MAP_KEYS:
            raise ValueError("Catalogue V4 invalide : contrat carte strict attendu.")
        map_id = raw_map.get("mapId")
        calibration_city_ids = _SYS_CALIBRATION_CITY_IDS.get(map_id, [])
        if not set(calibration_city_ids).issubset(city_ids):
            calibration_city_ids = []
        migrated_maps.append({
            **raw_map,
            "description": "Carte d'assemblage Vosges" if map_id == "MAP-SYS-000001"
            else "Carte de référence Catalogue" if map_id == "MAP-SYS-000002" else "",
            "calibrationCityIds": calibration_city_ids,
        })
    return {**data, "version": 5, "maps": migrated_maps}


def migrate_catalogue_file_v4_to_v5(source: str | Path, *, force: bool = False) -> Path:
    path = Path(source)
    backup = Path(str(path) + ".pre-catalogue-v5")
    if backup.exists() and not force:
        raise FileExistsError(f"Backup déjà présent : {backup}")
    data = json.loads(path.read_text(encoding="utf-8"))
    migrated = migrate_catalogue_data_v4_to_v5(data)
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
    parser = argparse.ArgumentParser(description="Migre un Catalogue V4 vers V5.")
    parser.add_argument("catalogue")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    migrate_catalogue_file_v4_to_v5(args.catalogue, force=args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
