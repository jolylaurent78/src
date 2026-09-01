"""Migration one-shot des références ``map@path`` livrées vers ``map@resource``."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.assembleur_paths import ApplicationPaths
from src.assembleur_scenario_map import migrate_scenario_map_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scenarios", nargs="+", type=Path)
    parser.add_argument("--force", action="store_true", help="autorise un backup existant")
    args = parser.parse_args()
    paths = ApplicationPaths.from_runtime()
    for scenario in args.scenarios:
        result = migrate_scenario_map_path(
            scenario, resource_maps_dir=paths.resource_maps_dir, force=args.force
        )
        if result.changed:
            print(f"MIGRÉ {result.path}: {result.old_path} -> resource={result.resource}; backup={result.backup_path}")
        else:
            print(f"INCHANGÉ {result.path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
