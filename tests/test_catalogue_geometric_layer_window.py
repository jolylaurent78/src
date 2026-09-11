import json
import inspect
from types import SimpleNamespace

import pytest

from src.assembleur_catalogue import Catalogue
from src.assembleur_catalogue_geometric_layer_assets import CatalogueGeometricLayerAssetController
from src.assembleur_catalogue_identity import SystemCatalogueIdProvider, UserCatalogueIdProvider
from src.assembleur_catalogue_window import (
    CatalogueWindow,
    GeometricLayerCreationDialog,
    GeometricLayerModuleDisplayDialog,
    _bgr_to_rgb_hex,
    _rgb_hex_to_bgr,
)
from src.assembleur_geometric_layer_display import GeometricLayerModuleDisplayOverride
from src.assembleur_geometric_layer_io import geometric_layer_document_from_dict, load_geometric_layer_document
from src.assembleur_paths import ApplicationPaths


class _Value:
    def __init__(self, value=""):
        self.value = value

    def get(self):
        return self.value


class _Widget:
    def __init__(self):
        self.options = {}

    def configure(self, **kwargs):
        self.options.update(kwargs)


class _ModuleVariable:
    def __init__(self, master=None, value=False):
        self.value = value

    def get(self):
        return self.value

    def set(self, value):
        self.value = value


class _ModuleFrame:
    def __init__(self):
        self.children = []

    def winfo_children(self):
        return list(self.children)


class _ModuleCheckbox:
    def __init__(self, parent, **kwargs):
        self.parent = parent
        self.kwargs = kwargs
        parent.children.append(self)

    def destroy(self):
        self.parent.children.remove(self)

    def grid(self, **kwargs):
        self.grid_options = kwargs


@pytest.fixture(autouse=True)
def _fake_module_widgets(monkeypatch):
    monkeypatch.setattr("src.assembleur_catalogue_window.tk.BooleanVar", _ModuleVariable)
    monkeypatch.setattr("src.assembleur_catalogue_window.ttk.Checkbutton", _ModuleCheckbox)
    monkeypatch.setattr("src.assembleur_catalogue_window.ttk.Button", _ModuleCheckbox)
    monkeypatch.setattr(CatalogueWindow, "_attach_tooltip", lambda _self, _widget, _text: None)


def _document_payload(algorithm="Test"):
    return {
        "schema_version": 2,
        "source": "AlgoSimulator",
        "algorithm": algorithm,
        "segment": "18",
        "scope": {"type": "scenario", "scenario": "Reference"},
        "modules": [{
            "id": "m1",
            "label": "Module",
            "traces": [{
                "geometry": {"type": "point", "x_l93": 700000, "y_l93": 6600000},
                "graphics": {
                    "name": "Point", "color_bgr": [1, 2, 3], "width": 1,
                    "style": "solid", "show_name": False, "visible": True,
                    "tags": {}, "tooltips": [], "scenario_tooltips": [],
                },
            }],
        }],
    }


def _write_document(path, algorithm="Test"):
    path.write_text(json.dumps(_document_payload(algorithm)), encoding="utf-8")


def _catalogue_with_bases():
    catalogue = Catalogue(id_provider=SystemCatalogueIdProvider())
    base = catalogue.add_city("Base", 47, 2)
    layered_base = catalogue.add_city("Base avec calque", 46, 3)
    archived_base = catalogue.add_city("Base archivee", 45, 4)
    non_base = catalogue.add_city("Non base", 44, 5)
    opening = catalogue.add_city("Ouverture", 43, 6)
    light = catalogue.add_city("Lumiere", 42, 7)
    catalogue.add_triangle("Premier", opening.city_id, base.city_id, light.city_id)
    catalogue.add_triangle("Second", non_base.city_id, layered_base.city_id, light.city_id)
    archived_triangle = catalogue.add_triangle("Archive", opening.city_id, archived_base.city_id, non_base.city_id)
    catalogue.update_triangle(archived_triangle.triangle_id, archived=True)
    catalogue.set_geometric_layer(layered_base.city_id, f"geometric-layers/{layered_base.city_id}.traces.json")
    return catalogue, base.city_id, layered_base.city_id, archived_base.city_id, non_base.city_id


def _window_for_layer_logic(catalogue):
    window = object.__new__(CatalogueWindow)
    window.catalogue = catalogue
    window._geometric_layer_search_var = _Value()
    window._selected_geometric_layer_base_city_id = None
    window._geometric_layer_preview_document = None
    window._staged_geometric_layer_base_city_ids = set()
    window._staged_geometric_layer_documents = {}
    window._pending_geometric_layer_display_overrides = {}
    window._deleted_geometric_layer_base_city_ids = set()
    window._geometric_layer_module_vars = {}
    window._geometric_layer_modules_frame = _ModuleFrame()
    window._geometric_layer_modules_label = _Widget()
    return window


def test_available_geometric_layer_bases_are_real_bases_without_an_association():
    catalogue, base, layered_base, archived_base, non_base = _catalogue_with_bases()
    window = _window_for_layer_logic(catalogue)

    offered = {city.city_id for city in CatalogueWindow._available_geometric_layer_bases(window)}

    assert base in offered
    assert archived_base in offered
    assert layered_base not in offered
    assert non_base not in offered


def test_add_geometric_layer_opens_the_creation_dialog_with_only_available_bases(monkeypatch):
    catalogue, base, _layered_base, archived_base, _non_base = _catalogue_with_bases()
    window = _window_for_layer_logic(catalogue)
    captured = {}

    class _Dialog:
        def __init__(self, parent, cities, on_stage):
            captured["parent"] = parent
            captured["cities"] = cities
            captured["on_stage"] = on_stage

        def show(self):
            captured["shown"] = True

    monkeypatch.setattr("src.assembleur_catalogue_window.GeometricLayerCreationDialog", _Dialog)

    CatalogueWindow._add_geometric_layer(window)

    assert {city.city_id for city in captured["cities"]} == {base, archived_base}
    assert captured["on_stage"] == window._stage_new_geometric_layer
    assert captured["shown"] is True
    assert window._selected_geometric_layer_base_city_id is None


def test_creation_dialog_requires_a_base_and_file_before_staging():
    city = SimpleNamespace(city_id="CITY-1", name="Base")
    dialog = object.__new__(GeometricLayerCreationDialog)
    dialog._cities_by_name = {"Base": city}
    dialog._base_var = _Value("")
    dialog._source_var = _Value("")
    dialog._confirm_button = _Widget()
    staged = []
    dialog._on_stage = lambda base_city_id, source: staged.append((base_city_id, source)) or True
    dialog.destroy = lambda: staged.append("destroyed")

    GeometricLayerCreationDialog._update_confirm_state(dialog)
    GeometricLayerCreationDialog._confirm(dialog)

    assert dialog._confirm_button.options["state"] == "disabled"
    assert staged == []

    dialog._base_var = _Value("Base")
    dialog._source_var = _Value("source.traces.json")
    GeometricLayerCreationDialog._update_confirm_state(dialog)
    GeometricLayerCreationDialog._confirm(dialog)

    assert dialog._confirm_button.options["state"] == "normal"
    assert staged == [("CITY-1", "source.traces.json"), "destroyed"]


def test_stage_updates_preview_without_publishing_and_invalid_file_keeps_previous_preview(monkeypatch, tmp_path):
    catalogue, base, _layered_base, _archived_base, _non_base = _catalogue_with_bases()
    paths = ApplicationPaths.from_runtime(installation_root=tmp_path / "installation", user_data_root=tmp_path / "user")
    valid = tmp_path / "valid.traces.json"
    invalid = tmp_path / "invalid.traces.json"
    _write_document(valid, "Nouveau")
    invalid.write_text("{", encoding="utf-8")
    window = _window_for_layer_logic(catalogue)
    window._paths = paths
    window._geometric_layer_assets = CatalogueGeometricLayerAssetController(catalogue, paths)
    window._selected_geometric_layer_base_city_id = base
    window._geometric_layer_preview_document = None
    window._refresh_geometric_layer_list = lambda: None
    window._mark_dirty = lambda: setattr(window, "dirty", True)
    errors = []
    monkeypatch.setattr("src.assembleur_catalogue_window.messagebox.showerror", lambda *args, **kwargs: errors.append(args))

    assert CatalogueWindow._stage_geometric_layer(window, base, str(valid), select=True) is True

    preview = window._geometric_layer_preview_document
    assert preview.algorithm == "Nouveau"
    assert catalogue.get_geometric_layer(base) is None
    assert not (paths.active_catalogue_geometric_layers_dir / f"{base}.traces.json").exists()
    assert window.dirty is True

    assert CatalogueWindow._stage_geometric_layer(window, base, str(invalid), select=True) is False

    assert window._geometric_layer_preview_document is preview
    assert catalogue.get_geometric_layer(base) is None
    assert errors


def test_reimport_uses_the_selected_base_and_keeps_previous_preview_on_error(monkeypatch, tmp_path):
    catalogue, base, _layered_base, _archived_base, _non_base = _catalogue_with_bases()
    paths = ApplicationPaths.from_runtime(installation_root=tmp_path / "installation", user_data_root=tmp_path / "user")
    first, replacement, invalid = (tmp_path / "first.traces.json", tmp_path / "replacement.traces.json", tmp_path / "invalid.traces.json")
    _write_document(first, "Premier")
    _write_document(replacement, "Remplacement")
    invalid.write_text("{", encoding="utf-8")
    window = _window_for_layer_logic(catalogue)
    window._paths = paths
    window._geometric_layer_assets = CatalogueGeometricLayerAssetController(catalogue, paths)
    window._selected_geometric_layer_base_city_id = base
    window._refresh_geometric_layer_list = lambda: None
    window._mark_dirty = lambda: None
    errors = []
    monkeypatch.setattr("src.assembleur_catalogue_window.messagebox.showerror", lambda *args, **kwargs: errors.append(args))

    assert CatalogueWindow._stage_geometric_layer(window, base, str(first), select=True) is True
    first_preview = window._geometric_layer_preview_document
    monkeypatch.setattr("src.assembleur_catalogue_window.filedialog.askopenfilename", lambda **_kwargs: str(replacement))
    CatalogueWindow._reimport_selected_geometric_layer(window)

    assert window._selected_geometric_layer_base_city_id == base
    assert window._geometric_layer_preview_document.algorithm == "Remplacement"
    assert CatalogueWindow._selected_geometric_layer_module_ids(window) == {"m1"}

    replacement_preview = window._geometric_layer_preview_document
    monkeypatch.setattr("src.assembleur_catalogue_window.filedialog.askopenfilename", lambda **_kwargs: str(invalid))
    CatalogueWindow._reimport_selected_geometric_layer(window)

    assert first_preview.algorithm == "Premier"
    assert window._geometric_layer_preview_document is replacement_preview
    assert CatalogueWindow._selected_geometric_layer_module_ids(window) == {"m1"}
    assert errors


def test_existing_association_uses_resolver_and_parser_for_its_preview(monkeypatch):
    catalogue, _base, layered_base, _archived_base, _non_base = _catalogue_with_bases()
    document = SimpleNamespace(name="document", modules=())
    resolved = []

    class _Resolver:
        def __init__(self, paths):
            assert paths == "paths"

        def resolve(self, asset_file):
            resolved.append(asset_file)
            return "resolved.traces.json"

    window = _window_for_layer_logic(catalogue)
    window._paths = "paths"
    window._selected_geometric_layer_base_city_id = layered_base
    window._geometric_layer_preview_document = None
    window._geometric_layer_delete_button = _Widget()
    window._geometric_layer_reimport_button = _Widget()
    window._geometric_layer_map_view = SimpleNamespace(_request_redraw=lambda: resolved.append("redraw"))
    monkeypatch.setattr("src.assembleur_catalogue_window.CatalogueGeometricLayerAssetResolver", _Resolver)
    monkeypatch.setattr("src.assembleur_catalogue_window.load_geometric_layer_document", lambda path: document if path == "resolved.traces.json" else None)

    CatalogueWindow._load_selected_geometric_layer(window)

    assert resolved == [f"geometric-layers/{layered_base}.traces.json", "redraw"]
    assert window._geometric_layer_preview_document is document


def _csv_window(catalogue, paths):
    window = _window_for_layer_logic(catalogue)
    window._paths = paths
    window._geometric_layer_assets = CatalogueGeometricLayerAssetController(catalogue, paths)
    return window


def _write_csv(path, *rows):
    path.write_text("\ufeffBase;Fichier\n" + "\n".join(";".join(row) for row in rows), encoding="utf-8")


def test_export_geometric_layers_uses_names_basenames_and_copies_assets(tmp_path):
    catalogue, _base, layered_base, _archived_base, _non_base = _catalogue_with_bases()
    paths = ApplicationPaths.from_runtime(installation_root=tmp_path / "installation", user_data_root=tmp_path / "user")
    source = tmp_path / "source.traces.json"
    _write_document(source)
    publisher = CatalogueGeometricLayerAssetController(catalogue, paths)
    publisher.stage_geometric_layer(layered_base, source)
    publisher.commit()
    publisher.finalize_commit()
    window = _csv_window(catalogue, paths)
    exported = tmp_path / "export" / "calques.csv"
    exported.parent.mkdir()
    window._choose_export_path = lambda *_args, **_kwargs: str(exported)

    CatalogueWindow._export_geometric_layers_csv(window)

    assert exported.read_text(encoding="utf-8-sig") == f"Base;Fichier\nBase avec calque;{layered_base}.traces.json\n"
    companion = exported.parent / f"{layered_base}.traces.json"
    assert companion.read_bytes() == source.read_bytes()
    assert catalogue.get_geometric_layer(layered_base).asset_file == f"geometric-layers/{layered_base}.traces.json"


def test_import_geometric_layers_stages_a_new_base_without_publishing(tmp_path):
    catalogue, base, _layered_base, _archived_base, _non_base = _catalogue_with_bases()
    paths = ApplicationPaths.from_runtime(installation_root=tmp_path / "installation", user_data_root=tmp_path / "user")
    source = tmp_path / "new.traces.json"
    _write_document(source)
    csv_path = tmp_path / "calques.csv"
    _write_csv(csv_path, ("Base", source.name))
    window = _csv_window(catalogue, paths)

    result = CatalogueWindow._read_geometric_layers_csv(window, str(csv_path))

    assert result.staged_base_city_ids == (base,)
    assert result.errors == ()
    assert catalogue.get_geometric_layer(base) is None
    assert not (paths.active_catalogue_geometric_layers_dir / f"{base}.traces.json").exists()


def test_import_geometric_layers_stages_a_replacement(tmp_path):
    catalogue, _base, layered_base, _archived_base, _non_base = _catalogue_with_bases()
    paths = ApplicationPaths.from_runtime(installation_root=tmp_path / "installation", user_data_root=tmp_path / "user")
    old, replacement = tmp_path / "old.traces.json", tmp_path / "replacement.traces.json"
    _write_document(old, "Ancien")
    _write_document(replacement, "Nouveau")
    publisher = CatalogueGeometricLayerAssetController(catalogue, paths)
    publisher.stage_geometric_layer(layered_base, old)
    publisher.commit()
    publisher.finalize_commit()
    destination = paths.active_catalogue_geometric_layers_dir / f"{layered_base}.traces.json"
    before = destination.read_bytes()
    csv_path = tmp_path / "calques.csv"
    _write_csv(csv_path, ("Base avec calque", replacement.name))
    window = _csv_window(catalogue, paths)

    result = CatalogueWindow._read_geometric_layers_csv(window, str(csv_path))

    assert result.staged_base_city_ids == (layered_base,)
    assert destination.read_bytes() == before
    assert catalogue.get_geometric_layer(layered_base).asset_file == f"geometric-layers/{layered_base}.traces.json"


def test_import_geometric_layers_accumulates_invalid_rows_and_keeps_valid_staging(tmp_path):
    catalogue, base, _layered_base, _archived_base, non_base = _catalogue_with_bases()
    paths = ApplicationPaths.from_runtime(installation_root=tmp_path / "installation", user_data_root=tmp_path / "user")
    valid = tmp_path / "valid.traces.json"
    invalid = tmp_path / "invalid.traces.json"
    _write_document(valid)
    invalid.write_text("{", encoding="utf-8")
    csv_path = tmp_path / "calques.csv"
    _write_csv(
        csv_path,
        ("Base", valid.name),
        ("Ville inconnue", valid.name),
        ("Non base", valid.name),
        ("Base avec calque", "absent.traces.json"),
        ("Base archivee", invalid.name),
        ("Base", "autre.traces.json"),
    )
    window = _csv_window(catalogue, paths)

    result = CatalogueWindow._read_geometric_layers_csv(window, str(csv_path))

    assert result.staged_base_city_ids == (base,)
    assert len(result.errors) == 5
    assert any("Base inconnue" in error for error in result.errors)
    assert any("n'est la Base" in error for error in result.errors)
    assert any("introuvable" in error for error in result.errors)
    assert any("JSON Traces invalide" in error for error in result.errors)
    assert any("Base dupliquée" in error for error in result.errors)


def test_import_geometric_layers_rejects_absolute_and_parent_paths(tmp_path):
    catalogue, _base, _layered_base, _archived_base, _non_base = _catalogue_with_bases()
    paths = ApplicationPaths.from_runtime(installation_root=tmp_path / "installation", user_data_root=tmp_path / "user")
    csv_path = tmp_path / "calques.csv"
    _write_csv(csv_path, ("Base", "C:\\outside.traces.json"), ("Base archivee", "../outside.traces.json"))
    window = _csv_window(catalogue, paths)

    result = CatalogueWindow._read_geometric_layers_csv(window, str(csv_path))

    assert result.staged_count == 0
    assert len(result.errors) == 2
    assert all("dossier du CSV" in error for error in result.errors)


def test_geometric_layers_tab_activates_csv_actions_without_tk():
    window = object.__new__(CatalogueWindow)
    window._geometric_layers_tab = object()
    window._catalogue_notebook = SimpleNamespace(select=lambda: str(window._geometric_layers_tab))
    window._import_button = _Widget()
    window._export_button = _Widget()
    window._cities_tab = object()
    window._beacons_tab = object()
    window._triangles_tab = object()
    window._templates_tab = object()
    window._books_tab = object()

    CatalogueWindow._update_context_actions(window)

    assert window._import_button.options == {"command": window._import_geometric_layers_csv, "state": "normal"}
    assert window._export_button.options == {"command": window._export_geometric_layers_csv, "state": "normal"}


def test_preview_renderer_uses_catalogue_map_transform_and_all_modules(monkeypatch, tmp_path):
    source = tmp_path / "source.traces.json"
    _write_document(source)
    document = load_geometric_layer_document(source)
    calls = []

    class _Renderer:
        def __init__(self, canvas, context):
            calls.append((canvas, context))

        def render_document(self, received, **kwargs):
            calls.append((received, kwargs))

    map_object = SimpleNamespace(
        image_size=(320, 200),
        lambert_to_pixel=lambda x, y: (x / 10, y / 10),
    )
    canvas = object()
    window = object.__new__(CatalogueWindow)
    window._geometric_layer_preview_document = document
    window._geometric_layer_module_vars = {"m1": _ModuleVariable(value=True)}
    window._selected_geometric_layer_base_city_id = None
    window._pending_geometric_layer_display_overrides = {}
    window._geometric_layer_map_view = SimpleNamespace(map=map_object, canvas=canvas, _map_to_screen=lambda x, y: (x + 1, y + 2))
    monkeypatch.setattr("src.assembleur_catalogue_window.GeometricLayerRenderer", _Renderer)

    CatalogueWindow._render_geometric_layer_preview(window)

    assert calls[0][0] is canvas
    assert calls[0][1].image_size == (320, 200)
    assert calls[0][1].lambert_to_image(100, 50) == (10, 5)
    assert calls[0][1].image_to_canvas(10, 5) == (11, 7)
    assert calls[1] == (document, {"module_ids": {"m1"}, "display_overrides": {}, "clear": True})


def _document_with_modules(*modules):
    payload = _document_payload()
    payload["modules"] = [
        {"id": module_id, "label": label, "traces": []}
        for module_id, label in modules
    ]
    return geometric_layer_document_from_dict(payload)


def test_module_filters_are_built_from_labels_and_enabled_by_default():
    window = _window_for_layer_logic(Catalogue())
    document = _document_with_modules(("light", "Lumière"), ("shadow", "Ombre"))

    CatalogueWindow._set_geometric_layer_preview_document(window, document)

    assert set(window._geometric_layer_module_vars) == {"light", "shadow"}
    assert CatalogueWindow._selected_geometric_layer_module_ids(window) == {"light", "shadow"}
    assert [checkbox.kwargs["text"] for checkbox in window._geometric_layer_modules_frame.children if "variable" in checkbox.kwargs] == ["Lumière", "Ombre"]
    assert window._geometric_layer_modules_label.options["state"] == "normal"


def test_each_module_has_a_customization_button_for_its_own_identity(monkeypatch):
    window = _window_for_layer_logic(Catalogue())
    document = _document_with_modules(("light", "Lumière"), ("shadow", "Ombre"))
    captured = []

    class _Dialog:
        def __init__(self, _parent, module, override, _on_confirm):
            captured.append((module.module_id, module.label, override))

    monkeypatch.setattr("src.assembleur_catalogue_window.GeometricLayerModuleDisplayDialog", _Dialog)
    CatalogueWindow._set_geometric_layer_preview_document(window, document)

    buttons = [child for child in window._geometric_layer_modules_frame.children if child.kwargs.get("text") == "..."]
    assert len(buttons) == 2
    buttons[1].kwargs["command"]()
    assert captured == [("shadow", "Ombre", None)]


def test_customization_dialog_receives_the_existing_module_override(monkeypatch):
    catalogue, _base, layered_base, _archived_base, _non_base = _catalogue_with_bases()
    catalogue.set_geometric_layer_display_override(layered_base, "light", color_bgr=(3, 2, 1), width=5)
    window = _window_for_layer_logic(catalogue)
    window._selected_geometric_layer_base_city_id = layered_base
    captured = []

    class _Dialog:
        def __init__(self, _parent, module, override, _on_confirm):
            captured.append((module.module_id, override))

    monkeypatch.setattr("src.assembleur_catalogue_window.GeometricLayerModuleDisplayDialog", _Dialog)
    CatalogueWindow._set_geometric_layer_preview_document(window, _document_with_modules(("light", "Lumière")))

    next(child for child in window._geometric_layer_modules_frame.children if child.kwargs.get("text") == "...").kwargs["command"]()

    assert captured == [("light", GeometricLayerModuleDisplayOverride((3, 2, 1), 5))]


def test_dialog_helpers_convert_rgb_bgr_and_keep_cancelled_color_unchanged(monkeypatch):
    assert _bgr_to_rgb_hex((3, 2, 1)) == "#010203"
    assert _rgb_hex_to_bgr("#010203") == (3, 2, 1)
    dialog = object.__new__(GeometricLayerModuleDisplayDialog)
    dialog._color_bgr = (3, 2, 1)
    dialog._update_states = lambda: pytest.fail("A cancelled chooser must not update the dialog.")
    captured = []
    monkeypatch.setattr(
        "src.assembleur_catalogue_window.colorchooser.askcolor",
        lambda **kwargs: (captured.append(kwargs), (None, None))[1],
    )

    dialog._choose_color()

    assert dialog._color_bgr == (3, 2, 1)
    assert captured[0]["color"] == "#010203"


def test_dialog_confirm_uses_no_override_when_both_choices_are_default():
    dialog = object.__new__(GeometricLayerModuleDisplayDialog)
    dialog._color_bgr = (3, 2, 1)
    dialog._color_mode = _ModuleVariable(value="origin")
    dialog._width_mode = _ModuleVariable(value="origin")
    dialog._width_var = _ModuleVariable(value="4")
    confirmed = []
    dialog._on_confirm = lambda color_bgr, width: confirmed.append((color_bgr, width))
    dialog.destroy = lambda: confirmed.append("destroyed")

    dialog._confirm()
    assert confirmed == [(None, None), "destroyed"]

    confirmed.clear()
    dialog._color_bgr = (9, 8, 7)
    dialog._color_mode.set("custom")
    dialog._width_mode.set("custom")
    dialog._confirm()
    assert confirmed == [((9, 8, 7), 4), "destroyed"]


def test_dialog_layout_uses_compact_groups_default_labels_and_palette_button():
    layout = inspect.getsource(GeometricLayerModuleDisplayDialog.__init__)

    assert 'ttk.LabelFrame(root, text="Couleur", padding=6)' in layout
    assert 'ttk.LabelFrame(root, text="Épaisseur", padding=6)' in layout
    assert layout.count('text="Par défaut"') == 2
    assert 'text="Origine"' not in layout
    assert "_original_values" not in layout
    assert 'text="Réinitialiser"' not in layout
    assert 'text="Choisir..."' not in layout
    assert "image=parent._icon_palette" in layout
    assert 'parent._attach_tooltip(self._choose_color_button, "Choisir une couleur")' in layout
    assert 'self._color_preview.grid(row=1, column=1' in layout
    assert 'self._choose_color_button.grid(row=1, column=2' in layout


def test_catalogue_window_loads_the_supplied_palette_icon_once(monkeypatch):
    loaded = []
    monkeypatch.setattr(
        "src.assembleur_catalogue_window.tk.PhotoImage",
        lambda *, file: loaded.append(file) or file,
    )
    window = object.__new__(CatalogueWindow)

    window._load_icons()

    assert window._icon_palette.name == "palette.png"
    assert [path.name for path in loaded].count("palette.png") == 1


def test_staged_layer_override_is_previewed_without_mutating_catalogue(monkeypatch):
    catalogue, base, _layered_base, _archived_base, _non_base = _catalogue_with_bases()
    window = _window_for_layer_logic(catalogue)
    window._selected_geometric_layer_base_city_id = base
    window._geometric_layer_map_view = SimpleNamespace(_request_redraw=lambda: None)
    window._mark_dirty = lambda: setattr(window, "dirty", True)
    calls = []

    class _Renderer:
        def __init__(self, _canvas, _context):
            pass

        def render_document(self, document, **kwargs):
            calls.append((document, kwargs))

    CatalogueWindow._set_geometric_layer_module_display_override(window, "m1", color_bgr=(7, 8, 9), width=3)
    assert catalogue.get_geometric_layer(base) is None
    assert window._pending_geometric_layer_display_overrides == {
        base: {"m1": GeometricLayerModuleDisplayOverride((7, 8, 9), 3)},
    }
    window._geometric_layer_preview_document = _document_with_modules(("m1", "Module"))
    window._geometric_layer_module_vars = {"m1": _ModuleVariable(value=True)}
    window._geometric_layer_map_view = SimpleNamespace(
        map=SimpleNamespace(image_size=(100, 100), lambert_to_pixel=lambda x, y: (x, y)),
        canvas=object(),
        _map_to_screen=lambda x, y: (x, y),
    )
    monkeypatch.setattr("src.assembleur_catalogue_window.GeometricLayerRenderer", _Renderer)

    CatalogueWindow._render_geometric_layer_preview(window)

    assert window.dirty is True
    assert calls[0][1]["display_overrides"] == {"m1": GeometricLayerModuleDisplayOverride((7, 8, 9), 3)}


def test_module_filter_redraws_only_checked_modules_without_mutating_assets(monkeypatch):
    document = _document_with_modules(("light", "Lumière"), ("shadow", "Ombre"))
    window = _window_for_layer_logic(Catalogue())
    CatalogueWindow._set_geometric_layer_preview_document(window, document)
    window._geometric_layer_module_vars["shadow"].set(False)
    redraws = []
    window._geometric_layer_map_view = SimpleNamespace(_request_redraw=lambda: redraws.append("redraw"))
    window._geometric_layer_assets = SimpleNamespace(_staged={})

    CatalogueWindow._on_geometric_layer_module_selection_changed(window)

    assert CatalogueWindow._selected_geometric_layer_module_ids(window) == {"light"}
    assert redraws == ["redraw"]
    assert window.catalogue.geometric_layers == {}
    assert window._geometric_layer_assets._staged == {}


def test_preview_renderer_receives_empty_set_when_all_modules_are_unchecked(monkeypatch):
    document = _document_with_modules(("light", "Lumière"), ("shadow", "Ombre"))
    calls = []

    class _Renderer:
        def __init__(self, _canvas, _context):
            pass

        def render_document(self, received, **kwargs):
            calls.append((received, kwargs))

    window = _window_for_layer_logic(Catalogue())
    CatalogueWindow._set_geometric_layer_preview_document(window, document)
    for variable in window._geometric_layer_module_vars.values():
        variable.set(False)
    window._geometric_layer_map_view = SimpleNamespace(
        map=SimpleNamespace(image_size=(100, 100), lambert_to_pixel=lambda x, y: (x, y)),
        canvas=object(),
        _map_to_screen=lambda x, y: (x, y),
    )
    monkeypatch.setattr("src.assembleur_catalogue_window.GeometricLayerRenderer", _Renderer)

    CatalogueWindow._render_geometric_layer_preview(window)

    assert calls == [(document, {"module_ids": set(), "display_overrides": {}, "clear": True})]


def test_replacing_preview_document_rebuilds_its_module_filters():
    window = _window_for_layer_logic(Catalogue())
    first = _document_with_modules(("light", "Lumière"), ("shadow", "Ombre"))
    replacement = _document_with_modules(("candidate", "Candidat"),)
    CatalogueWindow._set_geometric_layer_preview_document(window, first)
    window._geometric_layer_module_vars["light"].set(False)

    CatalogueWindow._set_geometric_layer_preview_document(window, replacement)

    assert set(window._geometric_layer_module_vars) == {"candidate"}
    assert CatalogueWindow._selected_geometric_layer_module_ids(window) == {"candidate"}
    assert [checkbox.kwargs["text"] for checkbox in window._geometric_layer_modules_frame.children if "variable" in checkbox.kwargs] == ["Candidat"]


def test_no_preview_document_disables_and_empties_module_filters():
    window = _window_for_layer_logic(Catalogue())
    CatalogueWindow._set_geometric_layer_preview_document(window, _document_with_modules(("light", "Lumière")))

    CatalogueWindow._set_geometric_layer_preview_document(window, None)

    assert window._geometric_layer_module_vars == {}
    assert window._geometric_layer_modules_frame.children == []
    assert window._geometric_layer_modules_label.options["state"] == "disabled"


def test_delete_only_plans_removal_and_removes_the_line_before_apply(tmp_path):
    catalogue, _base, layered_base, _archived_base, _non_base = _catalogue_with_bases()
    paths = ApplicationPaths.from_runtime(installation_root=tmp_path / "installation", user_data_root=tmp_path / "user")
    destination = paths.active_catalogue_geometric_layers_dir / f"{layered_base}.traces.json"
    destination.parent.mkdir(parents=True)
    _write_document(destination)
    window = _window_for_layer_logic(catalogue)
    window._paths = paths
    window._geometric_layer_assets = CatalogueGeometricLayerAssetController(catalogue, paths)
    window._selected_geometric_layer_base_city_id = layered_base
    window._geometric_layer_preview_document = object()
    window._refresh_geometric_layer_list = lambda: setattr(window, "refreshed", True)
    window._mark_dirty = lambda: setattr(window, "dirty", True)

    CatalogueWindow._delete_selected_geometric_layer(window)

    assert layered_base in window._deleted_geometric_layer_base_city_ids
    assert window._selected_geometric_layer_base_city_id is None
    assert window._geometric_layer_preview_document is None
    assert destination.exists()
    assert catalogue.get_geometric_layer(layered_base) is not None
    assert window.refreshed is True
    assert window.dirty is True


def test_cancel_discards_staged_geometric_asset_and_restores_the_working_catalogue(tmp_path):
    catalogue, base, _layered_base, _archived_base, _non_base = _catalogue_with_bases()
    paths = ApplicationPaths.from_runtime(installation_root=tmp_path / "installation", user_data_root=tmp_path / "user")
    source = tmp_path / "source.traces.json"
    _write_document(source)
    controller = CatalogueGeometricLayerAssetController(catalogue, paths)
    controller.stage_geometric_layer(base, source)
    staged = controller._staged[base]
    window = object.__new__(CatalogueWindow)
    window.catalogue = catalogue
    window._validated_catalogue = catalogue.clone()
    window._map_calibration = SimpleNamespace(discard=lambda: None, rebind_catalogue=lambda _catalogue: None)
    window._book_assets = SimpleNamespace(discard=lambda: None, rebind_catalogue=lambda _catalogue: None)
    window._geometric_layer_assets = controller
    window._selected_template_rank_slot = None
    window._selected_city_id = "x"
    window._selected_beacon_id = "x"
    window._selected_triangle_id = "x"
    window._selected_template_id = None
    window._selected_map_id = None
    window._selected_book_id = None
    window._selected_geometric_layer_base_city_id = base
    window._geometric_layer_preview_document = object()
    window._geometric_layer_module_vars = {}
    window._geometric_layer_modules_frame = _ModuleFrame()
    window._geometric_layer_modules_label = _Widget()
    window._staged_geometric_layer_base_city_ids = {base}
    window._staged_geometric_layer_documents = {}
    window._pending_geometric_layer_display_overrides = {
        base: {"m1": GeometricLayerModuleDisplayOverride((1, 2, 3), 2)},
    }
    window._deleted_geometric_layer_base_city_ids = set()
    window._selected_calibration_city_id = None
    window._selected_template_triangle_id = None
    window._refresh_city_list = lambda: None
    window._refresh_beacon_list = lambda: None
    window._refresh_triangle_tree = lambda: None
    window._refresh_maps = lambda: None
    window._refresh_books = lambda: None
    window._refresh_geometric_layer_list = lambda: None
    window._refresh_templates = lambda: None
    window._update_context_actions = lambda: None
    window._load_selected_city = lambda: None
    window._set_dirty = lambda value: setattr(window, "dirty", value)

    CatalogueWindow._cancel_changes(window)

    assert not staged.exists()
    assert controller._staged == {}
    assert window.catalogue.get_geometric_layer(base) is None
    assert window._selected_geometric_layer_base_city_id is None
    assert window._pending_geometric_layer_display_overrides == {}
    assert window.dirty is False


class _Transaction:
    def __init__(self, name, calls, created):
        self.name, self.calls, self.created = name, calls, created

    def commit(self):
        self.calls.append(f"commit:{self.name}")
        return self.created

    def rollback(self, created):
        self.calls.append((f"rollback:{self.name}", created))

    def finalize_commit(self):
        self.calls.append(f"finalize:{self.name}")


def _window_for_apply(calls):
    window = object.__new__(CatalogueWindow)
    window.catalogue = Catalogue(id_provider=UserCatalogueIdProvider())
    window._map_calibration = _Transaction("map", calls, ["map-created"])
    window._book_assets = _Transaction("book", calls, ["book-created"])
    window._geometric_layer_assets = _Transaction("layer", calls, ["layer-created"])
    window._catalogue_path = "catalogue.json"
    window._on_catalogue_applied = None
    window._validated_catalogue = None
    window._staged_geometric_layer_base_city_ids = {"CITY-USR-1"}
    window._staged_geometric_layer_documents = {}
    window._pending_geometric_layer_display_overrides = {}
    window._deleted_geometric_layer_base_city_ids = {"CITY-USR-2"}
    window._refresh_geometric_layer_list = lambda: calls.append("refresh")
    window._set_dirty = lambda value: calls.append(("dirty", value))
    return window


def test_apply_commits_layer_assets_before_save_and_finalizes_only_after_success(monkeypatch):
    calls = []
    window = _window_for_apply(calls)
    monkeypatch.setattr("src.assembleur_catalogue_window.save_catalogue", lambda catalogue, path: calls.append(("save", catalogue, path)))

    CatalogueWindow._apply_changes(window)

    assert [entry for entry in calls if isinstance(entry, str)] == [
        "commit:map", "commit:book", "commit:layer", "finalize:map", "finalize:book", "finalize:layer", "refresh",
    ]
    assert calls[3][0] == "save"
    assert window._staged_geometric_layer_base_city_ids == set()
    assert window._deleted_geometric_layer_base_city_ids == set()


def test_apply_persists_pending_staged_layer_override_before_saving(monkeypatch, tmp_path):
    catalogue, base, _layered_base, _archived_base, _non_base = _catalogue_with_bases()
    paths = ApplicationPaths.from_runtime(installation_root=tmp_path / "installation", user_data_root=tmp_path / "user")
    source = tmp_path / "source.traces.json"
    _write_document(source)
    controller = CatalogueGeometricLayerAssetController(catalogue, paths)
    controller.stage_geometric_layer(base, source)
    window = _window_for_layer_logic(catalogue)
    calls = []
    window._map_calibration = _Transaction("map", calls, [])
    window._book_assets = _Transaction("book", calls, [])
    window._geometric_layer_assets = controller
    window._catalogue_path = tmp_path / "catalogue.json"
    window._on_catalogue_applied = None
    window._validated_catalogue = catalogue.clone()
    window._staged_geometric_layer_base_city_ids = {base}
    window._pending_geometric_layer_display_overrides = {
        base: {"m1": GeometricLayerModuleDisplayOverride((9, 8, 7), 4)},
    }
    window._refresh_geometric_layer_list = lambda: None
    window._set_dirty = lambda value: setattr(window, "dirty", value)
    saved = []
    monkeypatch.setattr("src.assembleur_catalogue_window.save_catalogue", lambda received, _path: saved.append(received.clone()))

    CatalogueWindow._apply_changes(window)

    assert catalogue.get_geometric_layer_display_override(base, "m1") == GeometricLayerModuleDisplayOverride((9, 8, 7), 4)
    assert saved[0].get_geometric_layer_display_override(base, "m1") == GeometricLayerModuleDisplayOverride((9, 8, 7), 4)
    assert window._pending_geometric_layer_display_overrides == {}
    assert window.dirty is False


def test_apply_save_failure_rolls_back_geometric_layer_assets(monkeypatch):
    calls = []
    window = _window_for_apply(calls)
    monkeypatch.setattr("src.assembleur_catalogue_window.save_catalogue", lambda *_args: (_ for _ in ()).throw(OSError("save failed")))
    monkeypatch.setattr("src.assembleur_catalogue_window.messagebox.showerror", lambda *args, **kwargs: calls.append("error"))

    CatalogueWindow._apply_changes(window)

    assert ("rollback:layer", ["layer-created"]) in calls
    assert ("rollback:map", ["map-created"]) in calls
    assert ("rollback:book", ["book-created"]) in calls
    assert "finalize:layer" not in calls
