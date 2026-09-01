from __future__ import annotations

from pathlib import Path

import pytest

from src.assembleur_config import (
    load_config_file,
    migrate_legacy_config,
    migrate_legacy_config_file,
    save_config_file,
)
from src.assembleur_paths import ApplicationPaths
from src import assembleur_io
from src.assembleur_tk import TriangleViewerManual


def _paths_with_seed(tmp_path) -> ApplicationPaths:
    paths = ApplicationPaths.from_runtime(
        installation_root=tmp_path / "installation",
        user_data_root=tmp_path / "user-data",
    )
    paths.resource_maps_dir.mkdir(parents=True)
    (paths.resource_maps_dir / "899 - Alsace.jpg").write_bytes(b"map")
    save_config_file(
        {"uiMapOpacity": 70, "bgMap": "899 - Alsace.jpg"},
        paths.default_config_path,
    )
    return paths


def test_first_user_config_is_copied_from_default_without_machine_path(tmp_path) -> None:
    paths = _paths_with_seed(tmp_path)
    default_bytes = paths.default_config_path.read_bytes()

    active = paths.config_path_for_runtime()

    assert active == paths.config_path
    assert active.read_bytes() == default_bytes
    config = load_config_file(active)
    assert config["bgMap"] == "899 - Alsace.jpg"
    assert (paths.resource_maps_dir / config["bgMap"]).is_file()
    assert "D:\\" not in active.read_text(encoding="utf-8")


def test_existing_user_config_is_never_overwritten(tmp_path) -> None:
    paths = _paths_with_seed(tmp_path)
    save_config_file({"uiMapOpacity": 35, "bgMap": "899 - Alsace.jpg"}, paths.config_path)
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
        user_data_root=tmp_path / "user-data",
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


def test_legacy_migration_keeps_portable_state_and_removes_dead_recent_file_keys(tmp_path) -> None:
    paths = _paths_with_seed(tmp_path)
    legacy = {
        "uiMapOpacity": 55,
        "simAutoPlacementByMap": {"899 - Alsace.jpg": {"forward": {"ox": 1}}},
        "bgSvgPath": r"D:\Dropbox\La Chouette\Python\AssembleurTriangles\data\maps\899 - Alsace.jpg",
        "bgWorldRect": {"x0": 1, "y0": 2, "w": 3, "h": 4},
        "lastTriangleCsvIn": r"D:\Dropbox\La Chouette\Python\AssembleurTriangles\data\triangle.csv",
        "lastVillesCsvIn": r"Z:\Documents\villes.csv",
        "cheminsBaliseRefName": "Grand Ballon",
    }

    migrated = migrate_legacy_config(legacy, resource_maps_dir=paths.resource_maps_dir)

    assert migrated["uiMapOpacity"] == 55
    assert migrated["simAutoPlacementByMap"] == legacy["simAutoPlacementByMap"]
    assert migrated["bgWorldRect"] == legacy["bgWorldRect"]
    assert migrated["bgMap"] == "899 - Alsace.jpg"
    assert "bgSvgPath" not in migrated
    assert "lastTriangleCsvIn" not in migrated
    assert "lastVillesCsvIn" not in migrated
    assert "cheminsBaliseRefName" not in migrated
    assert "Dropbox" not in repr(migrated)


def test_external_background_path_is_preserved(tmp_path) -> None:
    paths = _paths_with_seed(tmp_path)
    external_map = r"Z:\Documents\ma-carte.jpg"

    migrated = migrate_legacy_config(
        {"bgSvgPath": external_map}, resource_maps_dir=paths.resource_maps_dir
    )

    assert migrated["bgSvgPath"] == external_map
    assert "bgMap" not in migrated


def test_migration_refuses_to_overwrite_existing_user_config(tmp_path) -> None:
    paths = _paths_with_seed(tmp_path)
    source = tmp_path / "legacy.json"
    save_config_file({"bgMap": "899 - Alsace.jpg"}, source)
    save_config_file({"uiMapOpacity": 12}, paths.config_path)

    with pytest.raises(FileExistsError, match="destination déjà existante"):
        migrate_legacy_config_file(
            source, paths.config_path, resource_maps_dir=paths.resource_maps_dir
        )


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


def test_background_standard_map_is_resolved_from_resources_and_persisted_portably(tmp_path) -> None:
    paths = _paths_with_seed(tmp_path)

    class Viewer:
        _bg_startup_scheduled = True
        _bg_defer_redraw = False
        appConfig = {"bgMap": "899 - Alsace.jpg", "bgWorldRect": {"x0": 1, "y0": 2, "w": 3, "h": 4}}

        def getAppConfigValue(self, key, default=None):
            return self.appConfig.get(key, default)

        def update_idletasks(self):
            pass

        def _bg_set_map(self, path, *, rect_override, persist):
            self.loaded_path = path
            self.loaded_rect = rect_override
            self.persist_flag = persist

        def saveAppConfig(self):
            self.saved = True

    viewer = Viewer()
    viewer.paths = paths
    viewer._bg = {"path": str(paths.resource_maps_dir / "899 - Alsace.jpg"), "x0": 1, "y0": 2, "w": 3, "h": 4, "aspect": 1}
    TriangleViewerManual._autoLoadBackgroundAfterLayout(viewer)
    assert Path(viewer.loaded_path) == paths.resource_maps_dir / "899 - Alsace.jpg"
    assert viewer.loaded_rect == viewer.appConfig["bgWorldRect"]
    assert viewer.persist_flag is False

    TriangleViewerManual._persistBackgroundConfig(viewer)
    assert viewer.appConfig["bgMap"] == "899 - Alsace.jpg"
    assert "bgSvgPath" not in viewer.appConfig
