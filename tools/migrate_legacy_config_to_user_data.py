"""Cutover one-shot de config/assembleur_config.json vers user-data."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.assembleur_config import load_config_file, migrate_legacy_config_file
from src.assembleur_paths import ApplicationPaths


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=PROJECT_ROOT / "config" / "assembleur_config.json",
        help="configuration historique à migrer",
    )
    args = parser.parse_args()

    paths = ApplicationPaths.from_runtime()
    source = args.source.resolve()
    destination = paths.config_path
    archive = source.parent / "obsolete" / "assembleur_config.pre-packaging-004.json"
    if not source.is_file():
        raise FileNotFoundError(f"Configuration historique absente : {source}")
    if destination.exists():
        raise FileExistsError(f"Configuration utilisateur déjà existante : {destination}")
    if archive.exists():
        raise FileExistsError(f"Archive existante, migration refusée : {archive}")

    migrated = migrate_legacy_config_file(
        source,
        destination,
        resource_maps_dir=paths.resource_maps_dir,
    )
    # Relit avec le loader normal et valide la référence de ressource convertie.
    migrated = load_config_file(destination)
    map_name = migrated.get("bgMap")
    if map_name is not None and not (paths.resource_maps_dir / str(map_name)).is_file():
        raise ValueError(f"Carte livrée introuvable après migration : {map_name!r}")

    archive.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(source), str(archive))
    print(f"Configuration migrée : {source} -> {destination}")
    print(f"Archive conservée : {archive}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
