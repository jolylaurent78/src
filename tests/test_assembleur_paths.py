from __future__ import annotations

from pathlib import Path

from src.assembleur_catalogue import Catalogue
from src.assembleur_catalogue_identity import SystemCatalogueIdProvider, UserCatalogueIdProvider
from src.assembleur_catalogue_io import load_catalogue, save_catalogue
from src.assembleur_paths import ApplicationPaths


def _paths(tmp_path):
    return ApplicationPaths.from_runtime(
        installation_root=tmp_path / "installation",
        user_data_root=tmp_path / "user-space" / "user-data",
    )


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
    assert paths.user_data_root == tmp_path / "user-space" / "user-data"
    assert len({paths.resource_root, paths.defaults_root, paths.user_data_root}) == 3
    assert paths.default_catalogue_path.is_file()
    runtime_paths = ApplicationPaths.from_runtime()
    assert runtime_paths.installation_root == Path(__file__).resolve().parents[1]


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
