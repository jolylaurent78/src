import json
from pathlib import Path

import pytest

from src.assembleur_catalogue import Catalogue
from src.assembleur_catalogue_geometric_layer_assets import (
    CatalogueGeometricLayerAssetController,
    CatalogueGeometricLayerAssetResolver,
)
from src import assembleur_catalogue_geometric_layer_assets as layer_assets_module
from src.assembleur_catalogue_identity import SystemCatalogueIdProvider
from src.assembleur_geometric_layer_io import geometric_layer_document_from_dict, load_geometric_layer_document
from src.assembleur_paths import ApplicationPaths


def _trace(geometry: dict) -> dict:
    return {
        "geometry": geometry,
        "graphics": {
            "name": "Trace", "color_bgr": [1, 2, 3], "width": 2,
            "style": "solid", "show_name": None, "visible": True,
            "tags": {"module": "m1"}, "tooltips": ["info"], "scenario_tooltips": ["scenario"],
        },
    }


def _document() -> dict:
    return {
        "schema_version": 2, "source": "AlgoSimulator", "algorithm": "AlgorithmeTest", "segment": "18",
        "scope": {"type": "scenario", "scenario": "Référence"},
        "modules": [
            {"id": "m1", "label": "Premier", "traces": [_trace({"type": "line_image_azimuth", "x_l93": 700000.125, "y_l93": 6600000.5, "azimuth_deg": 42.5})]},
            {"id": "m2", "label": "Second", "traces": [_trace({"type": "circle", "center_x_l93": 1, "center_y_l93": 2, "radius_km": 3})]},
        ],
    }


def _catalogue() -> tuple[Catalogue, str, str]:
    catalogue = Catalogue(id_provider=SystemCatalogueIdProvider())
    base = catalogue.add_city("Base", 47, 2)
    other = catalogue.add_city("Non base", 46, 3)
    opening = catalogue.add_city("Ouverture", 45, 4)
    light = catalogue.add_city("Lumière", 44, 5)
    catalogue.add_triangle("Trace", opening.city_id, base.city_id, light.city_id)
    return catalogue, base.city_id, other.city_id


def _paths(tmp_path: Path, mode: str = "SYS") -> ApplicationPaths:
    return ApplicationPaths.from_runtime(installation_root=tmp_path / "installation", user_data_root=tmp_path / "user", catalogue_mode=mode)


def _write_document(path: Path, payload: dict | None = None) -> None:
    path.write_text(json.dumps(_document() if payload is None else payload), encoding="utf-8")


def test_parser_reads_a_representative_v2_document(tmp_path) -> None:
    source = tmp_path / "source.traces.json"
    _write_document(source)
    document = load_geometric_layer_document(source)
    assert len(document.modules) == 2
    assert document.modules[0].traces[0].geometry_type == "line_image_azimuth"
    assert document.modules[0].traces[0].geometry["azimuth_deg"] == 42.5
    assert document.modules[0].traces[0].graphics.show_name is None
    assert document.modules[1].traces[0].graphics.color_bgr == (1.0, 2.0, 3.0)


@pytest.mark.parametrize("show_name", (None, True, False))
def test_parser_accepts_nullable_show_name(show_name) -> None:
    data = _document()
    data["modules"][0]["traces"][0]["graphics"]["show_name"] = show_name

    document = geometric_layer_document_from_dict(data)

    assert document.modules[0].traces[0].graphics.show_name is show_name


@pytest.mark.parametrize("show_name", (0, 1, "true", "false", "", [], {}))
def test_parser_rejects_non_boolean_non_null_show_name(show_name) -> None:
    data = _document()
    data["modules"][0]["traces"][0]["graphics"]["show_name"] = show_name

    with pytest.raises(ValueError, match=r"graphics\.show_name doit être un booléen ou null"):
        geometric_layer_document_from_dict(data)


@pytest.mark.parametrize("visible", (True, False))
def test_parser_accepts_boolean_visible(visible) -> None:
    data = _document()
    data["modules"][0]["traces"][0]["graphics"]["visible"] = visible

    document = geometric_layer_document_from_dict(data)

    assert document.modules[0].traces[0].graphics.visible is visible


def test_parser_rejects_null_visible() -> None:
    data = _document()
    data["modules"][0]["traces"][0]["graphics"]["visible"] = None

    with pytest.raises(ValueError, match=r"graphics\.visible doit être un booléen"):
        geometric_layer_document_from_dict(data)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda data: data.__setitem__("schema_version", 1), "schema_version"),
        (lambda data: data.__setitem__("modules", {}), "modules"),
        (lambda data: data["modules"][0]["traces"].__setitem__(0, {}), r"traces\[0\]"),
        (lambda data: data["modules"][0]["traces"][0]["geometry"].__setitem__("type", "unknown"), "inconnu"),
        (lambda data: data["modules"][0]["traces"][0]["geometry"].__setitem__("azimuth_deg", float("nan")), "nombre fini"),
        (lambda data: data["modules"][0]["traces"][0]["graphics"].__setitem__("color_bgr", [1, 2]), "trois"),
        (lambda data: data["modules"][0]["traces"][0]["graphics"].__setitem__("tags", []), "tags"),
        (lambda data: data["modules"][0]["traces"][0]["graphics"].__setitem__("tooltips", [1]), "tooltips"),
    ],
)
def test_parser_rejects_invalid_v2_documents(mutate, message) -> None:
    data = _document()
    mutate(data)
    with pytest.raises(ValueError, match=message):
        geometric_layer_document_from_dict(data)


def test_import_copies_to_the_base_named_asset_and_replaces_it(tmp_path) -> None:
    catalogue, base, _other = _catalogue()
    paths = _paths(tmp_path)
    source = tmp_path / "unrelated-name.traces.json"
    _write_document(source)
    controller = CatalogueGeometricLayerAssetController(catalogue, paths)

    controller.stage_geometric_layer(base, source)
    controller.commit()
    destination = paths.active_catalogue_geometric_layers_dir / f"{base}.traces.json"
    assert destination.read_bytes() == source.read_bytes()
    assert catalogue.get_geometric_layer(base).asset_file == f"geometric-layers/{base}.traces.json"
    controller.finalize_commit()

    replacement = tmp_path / "replacement.traces.json"
    replacement_data = _document()
    replacement_data["algorithm"] = "Remplacement"
    _write_document(replacement, replacement_data)
    controller.stage_geometric_layer(base, replacement)
    controller.commit()
    assert destination.read_bytes() == replacement.read_bytes()


def test_invalid_or_invalid_base_import_preserves_the_existing_asset(tmp_path) -> None:
    catalogue, base, other = _catalogue()
    paths = _paths(tmp_path)
    source = tmp_path / "valid.traces.json"
    _write_document(source)
    controller = CatalogueGeometricLayerAssetController(catalogue, paths)
    controller.stage_geometric_layer(base, source)
    controller.commit()
    destination = paths.active_catalogue_geometric_layers_dir / f"{base}.traces.json"
    original = destination.read_bytes()

    invalid = tmp_path / "invalid.traces.json"
    invalid.write_text("{", encoding="utf-8")
    with pytest.raises(ValueError, match="JSON Traces"):
        controller.stage_geometric_layer(base, invalid)
    with pytest.raises(ValueError, match="Base d'aucun triangle"):
        controller.stage_geometric_layer(other, source)
    with pytest.raises(KeyError, match="Ville inconnue"):
        controller.stage_geometric_layer("CITY-SYS-999999", source)
    assert destination.read_bytes() == original


def test_resolver_uses_active_catalogue_root_not_the_base_id_prefix_and_deletes(tmp_path) -> None:
    catalogue, base, _other = _catalogue()
    paths = _paths(tmp_path, "USER")
    source = tmp_path / "source.traces.json"
    _write_document(source)
    controller = CatalogueGeometricLayerAssetController(catalogue, paths)
    controller.stage_geometric_layer(base, source)
    controller.commit()
    layer = catalogue.get_geometric_layer(base)
    resolved = CatalogueGeometricLayerAssetResolver(paths).resolve(layer.asset_file)
    assert resolved == paths.user_catalogue_geometric_layers_dir / f"{base}.traces.json"
    assert not (paths.default_catalogue_geometric_layers_dir / f"{base}.traces.json").exists()
    controller.delete_geometric_layer_asset(base)
    controller.commit()
    assert not resolved.exists()
    assert catalogue.get_geometric_layer(base) is None


def test_resolver_reports_a_missing_asset(tmp_path) -> None:
    catalogue, base, _other = _catalogue()
    catalogue.set_geometric_layer(base, f"geometric-layers/{base}.traces.json")
    with pytest.raises(FileNotFoundError, match="Asset du calque géométrique absent"):
        CatalogueGeometricLayerAssetResolver(_paths(tmp_path)).resolve_layer(base, catalogue)


def test_commit_failure_restores_a_first_created_asset_and_catalogue(monkeypatch, tmp_path) -> None:
    catalogue, base, _other = _catalogue()
    second_base = catalogue.add_city("Base 2", 43, 6)
    opening = catalogue.add_city("Ouverture 2", 42, 7)
    light = catalogue.add_city("Lumière 2", 41, 8)
    catalogue.add_triangle("Trace 2", opening.city_id, second_base.city_id, light.city_id)
    paths = _paths(tmp_path)
    first, second = tmp_path / "first.traces.json", tmp_path / "second.traces.json"
    _write_document(first)
    _write_document(second)
    controller = CatalogueGeometricLayerAssetController(catalogue, paths)
    controller.stage_geometric_layer(base, first)
    controller.stage_geometric_layer(second_base.city_id, second)
    original_copy = layer_assets_module.shutil.copy2
    calls = 0

    def fail_second_publish(source, destination, *args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("publication interrompue")
        return original_copy(source, destination, *args, **kwargs)

    monkeypatch.setattr(layer_assets_module.shutil, "copy2", fail_second_publish)
    with pytest.raises(OSError, match="interrompue"):
        controller.commit()
    assert not (paths.active_catalogue_geometric_layers_dir / f"{base}.traces.json").exists()
    assert catalogue.get_geometric_layer(base) is None
    assert catalogue.get_geometric_layer(second_base.city_id) is None


def test_commit_failure_restores_a_replaced_asset_and_multiple_bases(monkeypatch, tmp_path) -> None:
    catalogue, base, _other = _catalogue()
    second_base = catalogue.add_city("Base 2", 43, 6)
    opening = catalogue.add_city("Ouverture 2", 42, 7)
    light = catalogue.add_city("Lumière 2", 41, 8)
    catalogue.add_triangle("Trace 2", opening.city_id, second_base.city_id, light.city_id)
    paths = _paths(tmp_path)
    old_first, old_second = tmp_path / "old-1.traces.json", tmp_path / "old-2.traces.json"
    _write_document(old_first)
    _write_document(old_second)
    initial = CatalogueGeometricLayerAssetController(catalogue, paths)
    initial.stage_geometric_layer(base, old_first)
    initial.stage_geometric_layer(second_base.city_id, old_second)
    initial.commit()
    initial.finalize_commit()
    first_destination = paths.active_catalogue_geometric_layers_dir / f"{base}.traces.json"
    second_destination = paths.active_catalogue_geometric_layers_dir / f"{second_base.city_id}.traces.json"
    first_before, second_before = first_destination.read_bytes(), second_destination.read_bytes()

    replacement, other_replacement = tmp_path / "new-1.traces.json", tmp_path / "new-2.traces.json"
    replacement_data, other_data = _document(), _document()
    replacement_data["algorithm"], other_data["algorithm"] = "Nouvelle 1", "Nouvelle 2"
    _write_document(replacement, replacement_data)
    _write_document(other_replacement, other_data)
    controller = CatalogueGeometricLayerAssetController(catalogue, paths)
    controller.stage_geometric_layer(base, replacement)
    controller.stage_geometric_layer(second_base.city_id, other_replacement)
    original_copy = layer_assets_module.shutil.copy2
    calls = 0

    def fail_second_publish(source, destination, *args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("publication interrompue")
        return original_copy(source, destination, *args, **kwargs)

    monkeypatch.setattr(layer_assets_module.shutil, "copy2", fail_second_publish)
    with pytest.raises(OSError):
        controller.commit()
    assert first_destination.read_bytes() == first_before
    assert second_destination.read_bytes() == second_before
    assert catalogue.get_geometric_layer(base).asset_file == f"geometric-layers/{base}.traces.json"
    assert catalogue.get_geometric_layer(second_base.city_id).asset_file == f"geometric-layers/{second_base.city_id}.traces.json"


def test_delete_then_stage_replaces_the_same_base_asset(tmp_path) -> None:
    catalogue, base, _other = _catalogue()
    paths = _paths(tmp_path)
    old, new = tmp_path / "old.traces.json", tmp_path / "new.traces.json"
    _write_document(old)
    new_data = _document()
    new_data["algorithm"] = "Nouveau"
    _write_document(new, new_data)
    controller = CatalogueGeometricLayerAssetController(catalogue, paths)
    controller.stage_geometric_layer(base, old)
    controller.commit()
    controller.finalize_commit()
    controller.delete_geometric_layer_asset(base)
    controller.stage_geometric_layer(base, new)
    controller.commit()
    destination = paths.active_catalogue_geometric_layers_dir / f"{base}.traces.json"
    assert destination.read_bytes() == new.read_bytes()


def test_stage_then_delete_cancels_publication_and_deletes_the_old_asset(tmp_path) -> None:
    catalogue, base, _other = _catalogue()
    paths = _paths(tmp_path)
    old, new = tmp_path / "old.traces.json", tmp_path / "new.traces.json"
    _write_document(old)
    _write_document(new)
    initial = CatalogueGeometricLayerAssetController(catalogue, paths)
    initial.stage_geometric_layer(base, old)
    initial.commit()
    initial.finalize_commit()
    controller = CatalogueGeometricLayerAssetController(catalogue, paths)
    controller.stage_geometric_layer(base, new)
    controller.delete_geometric_layer_asset(base)
    controller.commit()
    assert not (paths.active_catalogue_geometric_layers_dir / f"{base}.traces.json").exists()
    assert catalogue.get_geometric_layer(base) is None


def test_deletion_commit_failure_restores_file_and_catalogue(monkeypatch, tmp_path) -> None:
    catalogue, base, _other = _catalogue()
    paths = _paths(tmp_path)
    source = tmp_path / "source.traces.json"
    _write_document(source)
    initial = CatalogueGeometricLayerAssetController(catalogue, paths)
    initial.stage_geometric_layer(base, source)
    initial.commit()
    initial.finalize_commit()
    destination = paths.active_catalogue_geometric_layers_dir / f"{base}.traces.json"
    before = destination.read_bytes()
    controller = CatalogueGeometricLayerAssetController(catalogue, paths)
    controller.delete_geometric_layer_asset(base)
    original_remove = catalogue.remove_geometric_layer
    calls = 0

    def fail_once(base_city_id):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise ValueError("suppression Catalogue interrompue")
        original_remove(base_city_id)

    monkeypatch.setattr(catalogue, "remove_geometric_layer", fail_once)
    with pytest.raises(ValueError, match="interrompue"):
        controller.commit()
    assert destination.read_bytes() == before
    assert catalogue.get_geometric_layer(base).asset_file == f"geometric-layers/{base}.traces.json"


def test_external_rollback_after_a_successful_deletion_restores_file_and_catalogue(tmp_path) -> None:
    catalogue, base, _other = _catalogue()
    paths = _paths(tmp_path)
    source = tmp_path / "source.traces.json"
    _write_document(source)
    initial = CatalogueGeometricLayerAssetController(catalogue, paths)
    initial.stage_geometric_layer(base, source)
    initial.commit()
    initial.finalize_commit()
    destination = paths.active_catalogue_geometric_layers_dir / f"{base}.traces.json"
    before = destination.read_bytes()
    controller = CatalogueGeometricLayerAssetController(catalogue, paths)
    controller.delete_geometric_layer_asset(base)
    created = controller.commit()
    assert not destination.exists()
    assert catalogue.get_geometric_layer(base) is None
    controller.rollback(created)
    assert destination.read_bytes() == before
    assert catalogue.get_geometric_layer(base).asset_file == f"geometric-layers/{base}.traces.json"


def test_mixed_replacement_and_deletion_failure_restores_every_base(monkeypatch, tmp_path) -> None:
    catalogue, base, _other = _catalogue()
    second_base = catalogue.add_city("Base 2", 43, 6)
    opening = catalogue.add_city("Ouverture 2", 42, 7)
    light = catalogue.add_city("Lumière 2", 41, 8)
    catalogue.add_triangle("Trace 2", opening.city_id, second_base.city_id, light.city_id)
    paths = _paths(tmp_path)
    old_first, old_second = tmp_path / "old-1.traces.json", tmp_path / "old-2.traces.json"
    _write_document(old_first)
    _write_document(old_second)
    initial = CatalogueGeometricLayerAssetController(catalogue, paths)
    initial.stage_geometric_layer(base, old_first)
    initial.stage_geometric_layer(second_base.city_id, old_second)
    initial.commit()
    initial.finalize_commit()
    first_destination = paths.active_catalogue_geometric_layers_dir / f"{base}.traces.json"
    second_destination = paths.active_catalogue_geometric_layers_dir / f"{second_base.city_id}.traces.json"
    first_before, second_before = first_destination.read_bytes(), second_destination.read_bytes()
    replacement = tmp_path / "replacement.traces.json"
    replacement_data = _document()
    replacement_data["algorithm"] = "Remplacement"
    _write_document(replacement, replacement_data)
    controller = CatalogueGeometricLayerAssetController(catalogue, paths)
    controller.stage_geometric_layer(base, replacement)
    controller.delete_geometric_layer_asset(second_base.city_id)
    original_remove = catalogue.remove_geometric_layer

    def fail_delete(base_city_id):
        if base_city_id == second_base.city_id:
            raise ValueError("suppression interrompue")
        original_remove(base_city_id)

    monkeypatch.setattr(catalogue, "remove_geometric_layer", fail_delete)
    with pytest.raises(ValueError, match="interrompue"):
        controller.commit()
    assert first_destination.read_bytes() == first_before
    assert second_destination.read_bytes() == second_before
    assert catalogue.get_geometric_layer(base).asset_file == f"geometric-layers/{base}.traces.json"
    assert catalogue.get_geometric_layer(second_base.city_id).asset_file == f"geometric-layers/{second_base.city_id}.traces.json"
