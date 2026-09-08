from __future__ import annotations

from pathlib import Path

import pytest

from src.assembleur_catalogue import Catalogue
from src.assembleur_catalogue_identity import SystemCatalogueIdProvider, UserCatalogueIdProvider
from src.assembleur_catalogue_io import load_catalogue, save_catalogue
from src.assembleur_paths import ApplicationPaths


def _paths(tmp_path, *, catalogue_mode: str = "USER", create_default_scenarios: bool = True):
    paths = ApplicationPaths.from_runtime(
        installation_root=tmp_path / "installation",
        user_data_root=tmp_path / "user-space",
        catalogue_mode=catalogue_mode,
    )
    if create_default_scenarios:
        paths.default_scenarios_dir.mkdir(parents=True)
    return paths


def _write_default_catalogue(paths: ApplicationPaths) -> None:
    catalogue = Catalogue(id_provider=SystemCatalogueIdProvider())
    catalogue.add_city("Reference", 48.8566, 2.3522)
    save_catalogue(catalogue, paths.default_catalogue_path)


def test_roots_are_distinct_and_independent_from_cwd(tmp_path, monkeypatch) -> None:
    paths = _paths(tmp_path)
    _write_default_catalogue(paths)
    alternate_cwd = tmp_path / "elsewhere"
    alternate_cwd.mkdir()
    monkeypatch.chdir(alternate_cwd)

    assert paths.resource_root == tmp_path / "installation" / "resources"
    assert paths.defaults_root == tmp_path / "installation" / "defaults"
    assert paths.default_catalogue_dir == tmp_path / "installation" / "defaults" / "catalogue"
    assert paths.default_catalogue_path == tmp_path / "installation" / "defaults" / "catalogue" / "catalogue.json"
    assert paths.default_catalogue_maps_dir == tmp_path / "installation" / "defaults" / "catalogue" / "maps"
    assert paths.user_data_root == tmp_path / "user-space"
    assert len({paths.resource_root, paths.defaults_root, paths.user_data_root}) == 3
    assert paths.default_catalogue_path.is_file()
    runtime_paths = ApplicationPaths.from_runtime()
    assert runtime_paths.installation_root == Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("catalogue_mode", ("SYS", "USER"))
def test_app_icon_path_resolves_packaged_resource_independently_from_mode(
    tmp_path, catalogue_mode: str
) -> None:
    paths = _paths(tmp_path, catalogue_mode=catalogue_mode)
    expected_icon = paths.resource_root / "icons" / "AssembleurTriangles.ico"
    expected_icon.parent.mkdir(parents=True)
    expected_icon.touch()

    assert paths.app_icon_path == expected_icon


def test_app_icon_path_requires_packaged_resource(tmp_path) -> None:
    paths = _paths(tmp_path)

    with pytest.raises(FileNotFoundError, match="Icone applicative absente") as error:
        _ = paths.app_icon_path

    assert str(paths.resource_root / "icons" / "AssembleurTriangles.ico") in str(
        error.value
    )


def test_user_first_run_copies_default_without_modifying_it(tmp_path) -> None:
    paths = _paths(tmp_path)
    _write_default_catalogue(paths)
    default_content = paths.default_catalogue_path.read_bytes()

    active_path = paths.catalogue_path_for_mode("USER")

    assert active_path == paths.user_catalogue_path
    assert active_path.read_bytes() == default_content
    assert paths.default_catalogue_path.read_bytes() == default_content
    assert list(paths.user_catalogue_maps_dir.iterdir()) == []
    assert load_catalogue(active_path).get_city("CITY-SYS-000001").name == "Reference"


def test_existing_user_catalogue_is_preserved(tmp_path) -> None:
    paths = _paths(tmp_path)
    _write_default_catalogue(paths)
    user_catalogue = Catalogue(id_provider=UserCatalogueIdProvider())
    user_catalogue.add_city("Personnel", 45.7640, 4.8357)
    save_catalogue(user_catalogue, paths.user_catalogue_path)
    user_content = paths.user_catalogue_path.read_bytes()

    active_path = paths.catalogue_path_for_mode("USER")

    assert active_path == paths.user_catalogue_path
    assert active_path.read_bytes() == user_content
    assert load_catalogue(active_path).get_city(next(iter(user_catalogue.cities))).name == "Personnel"


def test_user_without_default_fails_explicitly(tmp_path) -> None:
    paths = _paths(tmp_path)

    try:
        paths.catalogue_path_for_mode("USER")
    except FileNotFoundError as exc:
        assert str(paths.default_catalogue_path) in str(exc)
    else:
        raise AssertionError("Le default absent doit provoquer une erreur explicite.")
    assert not paths.user_catalogue_path.exists()


def test_sys_uses_default_and_creates_no_user_catalogue(tmp_path) -> None:
    paths = _paths(tmp_path)
    _write_default_catalogue(paths)

    active_path = paths.catalogue_path_for_mode("SYS")

    assert active_path == paths.default_catalogue_path
    assert not paths.user_catalogue_path.exists()


def test_mutable_directories_are_created(tmp_path) -> None:
    paths = _paths(tmp_path)

    paths.ensure_user_data_directories()

    assert all(
        directory.is_dir()
        for directory in (
            paths.user_catalogue_dir,
            paths.user_catalogue_maps_dir,
            paths.user_scenarios_dir,
            paths.config_dir,
            paths.exports_dir,
            paths.logs_dir,
        )
    )
    assert not (paths.user_data_root / "user_data").exists()
    assert not (paths.user_data_root / "user-data").exists()


def test_first_user_initialization_copies_all_default_scenarios(tmp_path) -> None:
    paths = _paths(tmp_path)
    (paths.default_scenarios_dir / "reference.xml").write_text(
        "<scenario>reference</scenario>", encoding="utf-8"
    )
    nested_default = paths.default_scenarios_dir / "samples" / "example.xml"
    nested_default.parent.mkdir()
    nested_default.write_text("<scenario>example</scenario>", encoding="utf-8")

    paths.ensure_user_data_directories()

    assert (paths.user_scenarios_dir / "reference.xml").read_text(
        encoding="utf-8"
    ) == "<scenario>reference</scenario>"
    assert (paths.user_scenarios_dir / "samples" / "example.xml").read_text(
        encoding="utf-8"
    ) == "<scenario>example</scenario>"


def test_subsequent_user_initialization_does_not_recopy_scenarios(tmp_path) -> None:
    paths = _paths(tmp_path)
    default_scenario = paths.default_scenarios_dir / "reference.xml"
    default_scenario.write_text("version-1", encoding="utf-8")

    paths.ensure_user_data_directories()
    default_scenario.write_text("version-2", encoding="utf-8")
    (paths.default_scenarios_dir / "new.xml").write_text("new", encoding="utf-8")
    paths.ensure_user_data_directories()

    assert (paths.user_scenarios_dir / "reference.xml").read_text(
        encoding="utf-8"
    ) == "version-1"
    assert not (paths.user_scenarios_dir / "new.xml").exists()


def test_user_scenario_deletion_is_preserved_after_first_initialization(tmp_path) -> None:
    paths = _paths(tmp_path)
    default_scenario = paths.default_scenarios_dir / "reference.xml"
    default_scenario.write_text("reference", encoding="utf-8")

    paths.ensure_user_data_directories()
    (paths.user_scenarios_dir / "reference.xml").unlink()
    paths.ensure_user_data_directories()

    assert not (paths.user_scenarios_dir / "reference.xml").exists()


def test_first_user_initialization_requires_default_scenarios_payload(tmp_path) -> None:
    paths = _paths(tmp_path, create_default_scenarios=False)

    with pytest.raises(FileNotFoundError, match="Scenarios par defaut absents") as error:
        paths.ensure_user_data_directories()

    assert str(paths.default_scenarios_dir) in str(error.value)
    assert not paths.user_data_root.exists()


def test_sys_uses_default_scenarios_without_creating_user_scenarios(tmp_path) -> None:
    paths = _paths(tmp_path, catalogue_mode="SYS")
    (paths.default_scenarios_dir / "reference.xml").write_text(
        "reference", encoding="utf-8"
    )

    paths.ensure_user_data_directories()

    assert paths.active_scenarios_dir == paths.default_scenarios_dir
    assert not paths.user_scenarios_dir.exists()


def test_runtime_default_uses_application_root_without_user_data_level(tmp_path) -> None:
    paths = ApplicationPaths.from_runtime(
        environ={"LOCALAPPDATA": str(tmp_path / "local-app-data")},
        installation_root=tmp_path / "installation",
    )

    assert paths.user_data_root == tmp_path / "local-app-data" / "AssembleurTriangles"


@pytest.mark.parametrize("legacy_name", ("user_data", "user-data"))
def test_legacy_nested_layout_is_migrated_before_directories_are_created(tmp_path, legacy_name) -> None:
    paths = _paths(tmp_path)
    legacy_root = paths.user_data_root / legacy_name
    (legacy_root / "catalogue").mkdir(parents=True)
    (legacy_root / "catalogue" / "catalogue.json").write_text("{}", encoding="utf-8")
    (legacy_root / "scenarios").mkdir()
    (legacy_root / "scenarios" / "scenario.xml").write_text("<scenario/>", encoding="utf-8")
    (legacy_root / "config").mkdir()
    (legacy_root / "config" / "assembleur_config.json").write_text("{}", encoding="utf-8")
    (legacy_root / "exports").mkdir()

    paths.ensure_user_data_directories()

    assert (paths.user_catalogue_dir / "catalogue.json").is_file()
    assert (paths.user_scenarios_dir / "scenario.xml").is_file()
    assert paths.config_path.is_file()
    assert paths.exports_dir.is_dir()
    assert not legacy_root.exists()


def test_partially_migrated_layout_continues_without_error(tmp_path) -> None:
    paths = _paths(tmp_path)
    legacy_scenarios = paths.user_data_root / "user-data" / "scenarios"
    legacy_scenarios.mkdir(parents=True)
    (legacy_scenarios / "test.xml").write_text("<scenario/>", encoding="utf-8")
    paths.user_catalogue_dir.mkdir(parents=True)
    (paths.user_catalogue_dir / "catalogue.json").write_text("{}", encoding="utf-8")

    paths.ensure_user_data_directories()

    assert (paths.user_catalogue_dir / "catalogue.json").is_file()
    assert (paths.user_scenarios_dir / "test.xml").is_file()


def test_legacy_directory_merges_with_existing_directory(tmp_path) -> None:
    paths = _paths(tmp_path)
    (paths.user_catalogue_dir / "maps").mkdir(parents=True)
    (paths.user_catalogue_dir / "maps" / "A.png").write_bytes(b"image")
    legacy_catalogue = paths.user_data_root / "user-data" / "catalogue"
    legacy_catalogue.mkdir(parents=True)
    (legacy_catalogue / "catalogue.json").write_text("{}", encoding="utf-8")

    paths.migrate_legacy_user_data_layout()

    assert (paths.user_catalogue_dir / "catalogue.json").is_file()
    assert (paths.user_catalogue_dir / "maps" / "A.png").read_bytes() == b"image"


def test_identical_legacy_file_is_discarded_after_merge(tmp_path) -> None:
    paths = _paths(tmp_path)
    paths.user_catalogue_dir.mkdir(parents=True)
    destination = paths.user_catalogue_dir / "catalogue.json"
    destination.write_text("same", encoding="utf-8")
    source = paths.user_data_root / "user-data" / "catalogue" / "catalogue.json"
    source.parent.mkdir(parents=True)
    source.write_text("same", encoding="utf-8")

    paths.migrate_legacy_user_data_layout()

    assert destination.read_text(encoding="utf-8") == "same"
    assert not source.exists()


def test_different_legacy_file_raises_without_data_loss(tmp_path) -> None:
    paths = _paths(tmp_path)
    paths.user_catalogue_dir.mkdir(parents=True)
    destination = paths.user_catalogue_dir / "catalogue.json"
    destination.write_text("current", encoding="utf-8")
    source = paths.user_data_root / "user-data" / "catalogue" / "catalogue.json"
    source.parent.mkdir(parents=True)
    source.write_text("legacy", encoding="utf-8")

    with pytest.raises(FileExistsError, match="fichiers différents") as error:
        paths.migrate_legacy_user_data_layout()

    assert str(source) in str(error.value)
    assert str(destination) in str(error.value)
    assert source.read_text(encoding="utf-8") == "legacy"
    assert destination.read_text(encoding="utf-8") == "current"


def test_legacy_type_conflict_is_explicit(tmp_path) -> None:
    paths = _paths(tmp_path)
    (paths.user_catalogue_dir / "maps").mkdir(parents=True)
    source = paths.user_data_root / "user-data" / "catalogue" / "maps"
    source.parent.mkdir(parents=True)
    source.write_text("not a directory", encoding="utf-8")

    with pytest.raises(FileExistsError, match="types incompatibles"):
        paths.migrate_legacy_user_data_layout()


def test_two_legacy_roots_merge_non_conflicting_entries(tmp_path) -> None:
    paths = _paths(tmp_path)
    first = paths.user_data_root / "user_data" / "catalogue"
    second = paths.user_data_root / "user-data" / "catalogue"
    first.mkdir(parents=True)
    second.mkdir(parents=True)
    (first / "catalogue.json").write_text("{}", encoding="utf-8")
    (second / "extra.json").write_text("{}", encoding="utf-8")

    paths.migrate_legacy_user_data_layout()

    assert (paths.user_catalogue_dir / "catalogue.json").is_file()
    assert (paths.user_catalogue_dir / "extra.json").is_file()


def test_legacy_migration_and_ensure_directories_are_idempotent(tmp_path) -> None:
    paths = _paths(tmp_path)
    source = paths.user_data_root / "user-data" / "logs" / "app.log"
    source.parent.mkdir(parents=True)
    source.write_text("log", encoding="utf-8")

    paths.migrate_legacy_user_data_layout()
    paths.migrate_legacy_user_data_layout()
    paths.ensure_user_data_directories()
    paths.ensure_user_data_directories()

    assert (paths.logs_dir / "app.log").read_text(encoding="utf-8") == "log"


def test_runtime_log_collision_is_preserved_under_a_legacy_name(tmp_path) -> None:
    paths = _paths(tmp_path)
    paths.logs_dir.mkdir(parents=True)
    (paths.logs_dir / "mig_geo.log").write_text("current", encoding="utf-8")
    source = paths.user_data_root / "user-data" / "logs" / "mig_geo.log"
    source.parent.mkdir(parents=True)
    source.write_text("legacy", encoding="utf-8")

    paths.migrate_legacy_user_data_layout()

    assert (paths.logs_dir / "mig_geo.log").read_text(encoding="utf-8") == "current"
    assert (paths.logs_dir / "mig_geo.legacy.log").read_text(encoding="utf-8") == "legacy"
    assert not source.exists()


def test_identical_runtime_log_collision_keeps_a_single_file(tmp_path) -> None:
    paths = _paths(tmp_path)
    paths.logs_dir.mkdir(parents=True)
    (paths.logs_dir / "mig_geo.log").write_text("same", encoding="utf-8")
    source = paths.user_data_root / "user-data" / "logs" / "mig_geo.log"
    source.parent.mkdir(parents=True)
    source.write_text("same", encoding="utf-8")

    paths.migrate_legacy_user_data_layout()

    assert sorted(path.name for path in paths.logs_dir.iterdir()) == ["mig_geo.log"]


def test_runtime_log_collision_uses_incremented_legacy_name(tmp_path) -> None:
    paths = _paths(tmp_path)
    paths.logs_dir.mkdir(parents=True)
    (paths.logs_dir / "mig_geo.log").write_text("current", encoding="utf-8")
    (paths.logs_dir / "mig_geo.legacy.log").write_text("older", encoding="utf-8")
    source = paths.user_data_root / "user-data" / "logs" / "mig_geo.log"
    source.parent.mkdir(parents=True)
    source.write_text("legacy", encoding="utf-8")

    paths.migrate_legacy_user_data_layout()

    assert (paths.logs_dir / "mig_geo.legacy-2.log").read_text(encoding="utf-8") == "legacy"


def test_runtime_log_subdirectories_merge_recursively(tmp_path) -> None:
    paths = _paths(tmp_path)
    destination = paths.logs_dir / "archive"
    destination.mkdir(parents=True)
    (destination / "current.log").write_text("current", encoding="utf-8")
    source = paths.user_data_root / "user-data" / "logs" / "archive"
    source.mkdir(parents=True)
    (source / "legacy.log").write_text("legacy", encoding="utf-8")

    paths.migrate_legacy_user_data_layout()

    assert (destination / "current.log").is_file()
    assert (destination / "legacy.log").read_text(encoding="utf-8") == "legacy"


def test_runtime_type_conflict_is_explicit(tmp_path) -> None:
    paths = _paths(tmp_path)
    (paths.logs_dir / "archive").mkdir(parents=True)
    source = paths.user_data_root / "user-data" / "logs" / "archive"
    source.parent.mkdir(parents=True)
    source.write_text("not a directory", encoding="utf-8")

    with pytest.raises(FileExistsError, match="types incompatibles"):
        paths.migrate_legacy_user_data_layout()


@pytest.mark.parametrize("directory_name", ("cache", "temp"))
def test_runtime_cache_and_temp_collisions_do_not_block_migration(tmp_path, directory_name) -> None:
    paths = _paths(tmp_path)
    destination = paths.user_data_root / directory_name
    destination.mkdir(parents=True)
    (destination / "runtime").write_text("current", encoding="utf-8")
    source = paths.user_data_root / "user-data" / directory_name / "runtime"
    source.parent.mkdir(parents=True)
    source.write_text("legacy", encoding="utf-8")

    paths.migrate_legacy_user_data_layout()

    assert (destination / "runtime.legacy").read_text(encoding="utf-8") == "legacy"


def test_legacy_directory_with_unknown_files_is_not_removed(tmp_path) -> None:
    paths = _paths(tmp_path)
    legacy_root = paths.user_data_root / "user_data"
    (legacy_root / "scenarios").mkdir(parents=True)
    (legacy_root / "foo.txt").write_text("unknown", encoding="utf-8")

    paths.ensure_user_data_directories()

    assert paths.user_scenarios_dir.is_dir()
    assert (legacy_root / "foo.txt").read_text(encoding="utf-8") == "unknown"


def test_user_paths_are_direct_children_of_the_user_root(tmp_path) -> None:
    paths = _paths(tmp_path)

    assert paths.user_catalogue_path == paths.user_data_root / "catalogue" / "catalogue.json"
    assert paths.user_scenarios_dir == paths.user_data_root / "scenarios"
    assert paths.config_path == paths.user_data_root / "config" / "assembleur_config.json"
    assert paths.exports_dir == paths.user_data_root / "exports"
