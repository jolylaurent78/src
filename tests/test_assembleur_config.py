from __future__ import annotations

from pathlib import Path

import pytest

from src import assembleur_io
from src.assembleur_config import (
    load_config_file,
    migrate_legacy_config,
    migrate_legacy_config_file,
    save_config_file,
)
from src.assembleur_paths import ApplicationPaths


def _paths_with_seed(tmp_path) -> ApplicationPaths:
    paths = ApplicationPaths.from_runtime(
        installation_root=tmp_path / "installation",
        user_data_root=tmp_path / "user-root",
    )
    save_config_file({"uiMapOpacity": 70}, paths.default_config_path)
    return paths


def test_first_user_config_is_copied_from_default_without_machine_path(tmp_path) -> None:
    paths = _paths_with_seed(tmp_path)
    default_bytes = paths.default_config_path.read_bytes()

    active = paths.config_path_for_runtime()

    assert active == paths.config_path
    assert active.read_bytes() == default_bytes
    assert load_config_file(active) == {"uiMapOpacity": 70}
    assert "D:\\" not in active.read_text(encoding="utf-8")


def test_existing_user_config_is_never_overwritten(tmp_path) -> None:
    paths = _paths_with_seed(tmp_path)
    save_config_file({"uiMapOpacity": 35}, paths.config_path)
    user_bytes = paths.config_path.read_bytes()

    assert paths.config_path_for_runtime() == paths.config_path
    assert paths.config_path.read_bytes() == user_bytes


def test_user_config_modification_does_not_change_default(tmp_path) -> None:
    paths = _paths_with_seed(tmp_path)
    default_bytes = paths.default_config_path.read_bytes()
    active = paths.config_path_for_runtime()
    config = load_config_file(active)
    config["uiMapOpacity"] = 42
    save_config_file(config, active)

    assert load_config_file(active)["uiMapOpacity"] == 42
    assert paths.default_config_path.read_bytes() == default_bytes


def test_missing_default_config_fails_explicitly(tmp_path) -> None:
    paths = ApplicationPaths.from_runtime(
        installation_root=tmp_path / "installation",
        user_data_root=tmp_path / "user-root",
    )

    with pytest.raises(FileNotFoundError, match="Configuration par défaut absente"):
        paths.config_path_for_runtime()
    assert not paths.config_path.exists()


def test_existing_invalid_config_is_not_replaced_silently(tmp_path) -> None:
    paths = _paths_with_seed(tmp_path)
    paths.config_dir.mkdir(parents=True)
    paths.config_path.write_text("{invalid", encoding="utf-8")

    with pytest.raises(ValueError, match="Configuration JSON invalide"):
        load_config_file(paths.config_path)
    assert paths.config_path.read_text(encoding="utf-8") == "{invalid"


def test_legacy_migration_keeps_portable_state_and_removes_dead_recent_file_keys() -> None:
    legacy = {
        "uiMapOpacity": 55,
        "simAutoPlacementByMap": {"MAP-SYS-000001": {"forward": {"ox": 1}}},
        "lastTriangleCsvIn": "triangle.csv",
        "lastVillesCsvIn": "villes.csv",
        "cheminsBaliseRefName": "Grand Ballon",
    }

    migrated = migrate_legacy_config(legacy)

    assert migrated["uiMapOpacity"] == 55
    assert migrated["simAutoPlacementByMap"] == legacy["simAutoPlacementByMap"]
    assert "lastTriangleCsvIn" not in migrated
    assert "lastVillesCsvIn" not in migrated
    assert "cheminsBaliseRefName" not in migrated


def test_migration_refuses_to_overwrite_existing_user_config(tmp_path) -> None:
    paths = _paths_with_seed(tmp_path)
    source = tmp_path / "legacy.json"
    save_config_file({"uiMapOpacity": 70}, source)
    save_config_file({"uiMapOpacity": 12}, paths.config_path)

    with pytest.raises(FileExistsError, match="destination déjà existante"):
        migrate_legacy_config_file(source, paths.config_path)


def test_load_app_config_keeps_missing_optional_but_rejects_invalid(tmp_path) -> None:
    class Viewer:
        config_path = str(tmp_path / "assembleur_config.json")
        appConfig = {"old": True}

    viewer = Viewer()
    assembleur_io.loadAppConfig(viewer)
    assert viewer.appConfig == {}
    Path(viewer.config_path).write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="racine JSON"):
        assembleur_io.loadAppConfig(viewer)
