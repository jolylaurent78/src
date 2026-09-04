from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from src.assembleur_catalogue import Catalogue, CatalogueBook, CatalogueMap, WorldRect
from src.assembleur_catalogue_book_assets import CatalogueBookAssetResolver
from src.assembleur_catalogue_identity import SystemCatalogueIdProvider, UserCatalogueIdProvider
from src.assembleur_catalogue_map_assets import CatalogueMapAssetResolver
from src.assembleur_catalogue_map_calibration import CatalogueMapCalibrationController
from src.assembleur_paths import ApplicationPaths


def _paths(tmp_path: Path, mode: str) -> ApplicationPaths:
    return ApplicationPaths.from_runtime(
        installation_root=tmp_path / "installation",
        user_data_root=tmp_path / "user",
        catalogue_mode=mode,
    )


@pytest.mark.parametrize(
    ("mode", "map_id", "expected"),
    [
        ("SYS", "MAP-SYS-000001", "default"),
        ("USER", "MAP-SYS-000001", "user"),
        ("USER", "MAP-USR-550e8400-e29b-41d4-a716-446655440000", "user"),
    ],
)
def test_map_assets_use_one_active_root_without_fallback(tmp_path, mode, map_id, expected):
    paths = _paths(tmp_path, mode)
    root = paths.default_catalogue_maps_dir if expected == "default" else paths.user_catalogue_maps_dir
    root.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (2, 2), "white").save(root / "map.jpg")

    catalogue_map = CatalogueMap(
        map_id, "Carte", "map.jpg", None, None, WorldRect(0, 0, 1, 1), 1.0
    )

    assert CatalogueMapAssetResolver(paths).resolve(catalogue_map).image_path == (root / "map.jpg").resolve()
    other_root = paths.user_catalogue_maps_dir if expected == "default" else paths.default_catalogue_maps_dir
    (root / "map.jpg").unlink()
    other_root.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (2, 2), "white").save(other_root / "map.jpg")
    with pytest.raises(FileNotFoundError):
        CatalogueMapAssetResolver(paths).resolve(catalogue_map)


@pytest.mark.parametrize(
    ("mode", "book_id", "expected"),
    [
        ("SYS", "BOOK-SYS-000001", "default"),
        ("USER", "BOOK-SYS-000001", "user"),
        ("USER", "BOOK-USR-550e8400-e29b-41d4-a716-446655440000", "user"),
    ],
)
def test_book_assets_use_one_active_root_without_fallback(tmp_path, mode, book_id, expected):
    paths = _paths(tmp_path, mode)
    root = paths.default_catalogue_books_dir if expected == "default" else paths.user_catalogue_books_dir
    asset = root / "book.txt"
    asset.parent.mkdir(parents=True, exist_ok=True)
    asset.write_text("530 mot\n", encoding="utf-8")
    book = CatalogueBook(book_id, "Livre", "books/book.txt")

    assert CatalogueBookAssetResolver(paths).resolve(book) == asset.resolve()
    asset.unlink()
    other_root = paths.user_catalogue_books_dir if expected == "default" else paths.default_catalogue_books_dir
    other_asset = other_root / "book.txt"
    other_asset.parent.mkdir(parents=True, exist_ok=True)
    other_asset.write_text("530 mot\n", encoding="utf-8")
    with pytest.raises(FileNotFoundError):
        CatalogueBookAssetResolver(paths).resolve(book)


def test_user_seed_copies_the_complete_catalogue_once(tmp_path):
    paths = _paths(tmp_path, "USER")
    paths.default_catalogue_dir.mkdir(parents=True)
    paths.default_catalogue_path.write_text("{}", encoding="utf-8")
    (paths.default_catalogue_maps_dir / "a.jpg").parent.mkdir(parents=True)
    (paths.default_catalogue_maps_dir / "a.jpg").write_bytes(b"image")
    (paths.default_catalogue_maps_dir / "a.json").write_text("{}", encoding="utf-8")
    (paths.default_catalogue_books_dir / "a.txt").parent.mkdir(parents=True)
    (paths.default_catalogue_books_dir / "a.txt").write_text("book", encoding="utf-8")

    assert paths.catalogue_path_for_mode("USER") == paths.user_catalogue_path
    assert (paths.user_catalogue_maps_dir / "a.jpg").is_file()
    assert (paths.user_catalogue_maps_dir / "a.json").is_file()
    assert (paths.user_catalogue_books_dir / "a.txt").is_file()

    paths.user_catalogue_path.write_text('{"user": true}', encoding="utf-8")
    paths.default_catalogue_path.write_text('{"sys": true}', encoding="utf-8")
    paths.catalogue_path_for_mode("USER")
    assert json.loads(paths.user_catalogue_path.read_text(encoding="utf-8")) == {"user": True}


@pytest.mark.parametrize(
    ("mode", "expected"), [("SYS", "default"), ("USER", "user")]
)
def test_new_map_is_published_in_the_active_catalogue_root(tmp_path, mode, expected):
    paths = _paths(tmp_path, mode)
    source = tmp_path / "source.png"
    Image.new("RGB", (2, 2), "white").save(source)
    catalogue = Catalogue(id_provider=SystemCatalogueIdProvider() if mode == "SYS" else UserCatalogueIdProvider())
    controller = CatalogueMapCalibrationController(
        catalogue, paths, allow_system_map_editing=mode == "SYS"
    )

    map_id = controller.stage_map(source, name="Nouvelle", description="")
    controller.commit()
    root = paths.default_catalogue_maps_dir if expected == "default" else paths.user_catalogue_maps_dir
    assert (root / catalogue.get_map(map_id).image_file).is_file()
    assert map_id.startswith("MAP-SYS-" if mode == "SYS" else "MAP-USR-")


@pytest.mark.parametrize(
    ("mode", "expected"), [("SYS", "default"), ("USER", "user")]
)
def test_scenario_directory_follows_the_active_mode(tmp_path, mode, expected):
    paths = _paths(tmp_path, mode)
    expected_path = paths.default_scenarios_dir if expected == "default" else paths.user_scenarios_dir
    assert paths.active_scenarios_dir == expected_path
