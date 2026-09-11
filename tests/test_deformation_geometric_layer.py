from pathlib import Path
from types import SimpleNamespace

import pytest

from src.assembleur_catalogue import Catalogue
from src.assembleur_deformation_window import DeformationWindow
from src.assembleur_geometry_reference import GeometryReferenceResolver, ScenarioReference
from src.assembleur_geometric_layer_display import GeometricLayerModuleDisplayOverride
from src.assembleur_geometric_layer_io import geometric_layer_document_from_dict
from src.assembleur_geometric_layer_renderer import GeometricLayerRenderer
from src.assembleur_tk import TriangleViewerManual, resolve_catalogue_base_city_id_for_deformation_triangle


class _Button:
    def __init__(self):
        self.options = {}

    def configure(self, **kwargs):
        self.options.update(kwargs)


class _Window:
    def __init__(self, *, visible=True):
        self.geometric_layer_visible = visible
        self.layers = []
        self.cleared = 0

    def winfo_exists(self):
        return True

    def set_geometric_layer(self, document, *, display_overrides):
        self.layers.append((document, display_overrides))

    def clear_geometric_layer(self):
        self.cleared += 1


def _catalogue_and_resolver():
    catalogue = Catalogue()
    opening = catalogue.add_city("Ouverture", 46.0, 2.0)
    base = catalogue.add_city("Base", 47.0, 3.0)
    light = catalogue.add_city("Lumière", 48.0, 4.0)
    triangle = catalogue.add_triangle("Trace", opening.city_id, base.city_id, light.city_id)
    return catalogue, base, triangle, ScenarioReference()


def test_resolve_catalogue_base_city_id_accepts_tri_and_stri_references():
    catalogue, base, triangle, reference = _catalogue_and_resolver()
    resolver = GeometryReferenceResolver(catalogue, reference)
    assert resolve_catalogue_base_city_id_for_deformation_triangle(resolver, triangle.triangle_id) == base.city_id

    direct_stri = reference.create_triangle("Local", triangle.opening_city_id, base.city_id, triangle.light_city_id)
    assert resolve_catalogue_base_city_id_for_deformation_triangle(
        GeometryReferenceResolver(catalogue, reference), direct_stri.triangle_ref_id
    ) == base.city_id


def test_resolve_catalogue_base_city_id_follows_scity_source_without_fallback():
    catalogue, base, triangle, reference = _catalogue_and_resolver()
    local_base = reference.create_city("Base locale", 47.1, 3.1, catalogue_source_city_id=base.city_id)
    local_triangle = reference.create_triangle("Local", triangle.opening_city_id, local_base.city_ref_id, triangle.light_city_id)
    resolver = GeometryReferenceResolver(catalogue, reference)
    assert resolve_catalogue_base_city_id_for_deformation_triangle(resolver, local_triangle.triangle_ref_id) == base.city_id

    orphan_base = reference.create_city("Base isolée", 47.2, 3.2)
    orphan_triangle = reference.create_triangle("Orphelin", triangle.opening_city_id, orphan_base.city_ref_id, triangle.light_city_id)
    assert resolve_catalogue_base_city_id_for_deformation_triangle(resolver, orphan_triangle.triangle_ref_id) is None
    with pytest.raises(KeyError):
        resolve_catalogue_base_city_id_for_deformation_triangle(resolver, "STRI-9999")


def _viewer_for_loading(catalogue, reference, source_triangle_id):
    viewer = TriangleViewerManual.__new__(TriangleViewerManual)
    viewer.catalogue = catalogue
    viewer.paths = SimpleNamespace()
    viewer._deformation_window = _Window()
    viewer._deformation_geometric_layer_source_triangle_id = None
    viewer._deformation_working_reference = lambda: reference
    viewer._deformation_state.element_id = "E1"
    viewer._deformation_state.reference_world = SimpleNamespace(
        elements={"E1": SimpleNamespace(source_triangle_id=source_triangle_id)},
    )
    return viewer


def test_owner_loads_once_and_forwards_persisted_overrides(monkeypatch):
    catalogue, base, triangle, reference = _catalogue_and_resolver()
    catalogue.set_geometric_layer(base.city_id, f"geometric-layers/{base.city_id}.traces.json")
    catalogue.set_geometric_layer_display_override(base.city_id, "lumiere", color_bgr=(1, 2, 3), width=4)
    viewer = _viewer_for_loading(catalogue, reference, triangle.triangle_id)
    resolved, parsed = [], []

    class _Resolver:
        def __init__(self, _paths):
            pass

        def resolve(self, asset_file):
            resolved.append(asset_file)
            return Path("layer.traces.json")

    document = object()
    monkeypatch.setattr("src.assembleur_tk.CatalogueGeometricLayerAssetResolver", _Resolver)
    monkeypatch.setattr("src.assembleur_tk.load_geometric_layer_document", lambda path: parsed.append(path) or document)

    TriangleViewerManual._refresh_deformation_geometric_layer(viewer)
    TriangleViewerManual._refresh_deformation_geometric_layer(viewer)

    assert resolved == [f"geometric-layers/{base.city_id}.traces.json"]
    assert parsed == [Path("layer.traces.json")]
    assert viewer._deformation_window.layers == [
        (document, {"lumiere": GeometricLayerModuleDisplayOverride((1, 2, 3), 4)}),
    ]


def test_owner_clears_previous_layer_without_parsing_when_no_layer_or_toggle_off(monkeypatch):
    catalogue, _base, triangle, reference = _catalogue_and_resolver()
    viewer = _viewer_for_loading(catalogue, reference, triangle.triangle_id)
    monkeypatch.setattr("src.assembleur_tk.load_geometric_layer_document", lambda _path: pytest.fail("No layer must not be parsed."))

    TriangleViewerManual._refresh_deformation_geometric_layer(viewer)
    assert viewer._deformation_window.cleared == 1
    TriangleViewerManual._deformation_geometric_layer_visibility_changed(viewer, False)
    assert viewer._deformation_window.cleared == 2
    assert viewer._deformation_geometric_layer_source_triangle_id is None


def test_owner_replaces_the_layer_when_current_triangle_changes_and_skips_when_toggle_is_off(monkeypatch):
    catalogue, first_base, first_triangle, reference = _catalogue_and_resolver()
    opening = catalogue.add_city("Ouverture 2", 45.0, 1.0)
    second_base = catalogue.add_city("Base 2", 46.0, 2.0)
    light = catalogue.add_city("Lumière 2", 47.0, 3.0)
    second_triangle = catalogue.add_triangle("Trace 2", opening.city_id, second_base.city_id, light.city_id)
    catalogue.set_geometric_layer(first_base.city_id, f"geometric-layers/{first_base.city_id}.traces.json")
    catalogue.set_geometric_layer(second_base.city_id, f"geometric-layers/{second_base.city_id}.traces.json")
    viewer = _viewer_for_loading(catalogue, reference, first_triangle.triangle_id)
    resolved, parsed = [], []

    class _Resolver:
        def __init__(self, _paths):
            pass

        def resolve(self, asset_file):
            resolved.append(asset_file)
            return Path(asset_file)

    monkeypatch.setattr("src.assembleur_tk.CatalogueGeometricLayerAssetResolver", _Resolver)
    monkeypatch.setattr("src.assembleur_tk.load_geometric_layer_document", lambda path: parsed.append(path) or path)

    TriangleViewerManual._refresh_deformation_geometric_layer(viewer)
    viewer._deformation_state.reference_world.elements["E1"].source_triangle_id = second_triangle.triangle_id
    TriangleViewerManual._refresh_deformation_geometric_layer(viewer)
    viewer._deformation_window.geometric_layer_visible = False
    TriangleViewerManual._refresh_deformation_geometric_layer(viewer)

    assert resolved == [
        f"geometric-layers/{first_base.city_id}.traces.json",
        f"geometric-layers/{second_base.city_id}.traces.json",
    ]
    assert parsed == [Path(asset) for asset in resolved]
    assert [document for document, _overrides in viewer._deformation_window.layers] == parsed


def test_owner_reports_a_bad_asset_once_and_keeps_overlay_empty(monkeypatch):
    catalogue, base, triangle, reference = _catalogue_and_resolver()
    catalogue.set_geometric_layer(base.city_id, f"geometric-layers/{base.city_id}.traces.json")
    viewer = _viewer_for_loading(catalogue, reference, triangle.triangle_id)
    errors = []

    class _Resolver:
        def __init__(self, _paths):
            pass

        def resolve(self, _asset_file):
            raise FileNotFoundError("asset absent")

    monkeypatch.setattr("src.assembleur_tk.CatalogueGeometricLayerAssetResolver", _Resolver)
    monkeypatch.setattr("src.assembleur_tk.messagebox.showerror", lambda *args, **kwargs: errors.append((args, kwargs)))

    TriangleViewerManual._refresh_deformation_geometric_layer(viewer)
    TriangleViewerManual._refresh_deformation_geometric_layer(viewer)

    assert viewer._deformation_window.cleared == 1
    assert len(errors) == 1


def test_window_toggle_and_renderer_keep_layer_state_outside_canvas_mode(monkeypatch):
    window = DeformationWindow.__new__(DeformationWindow)
    window._geometric_layer_visible = False
    window._geometric_layer_button = _Button()
    changed, cleared = [], []
    window._on_geometric_layer_visibility_changed = changed.append
    window.clear_geometric_layer = lambda: cleared.append(True)

    window._toggle_geometric_layer_visibility()
    window._toggle_geometric_layer_visibility()

    assert changed == [True, False]
    assert cleared == [True]
    assert window._geometric_layer_button.options["relief"] == "flat"

    calls = []

    class _Renderer:
        def __init__(self, canvas, context):
            calls.append((canvas, context))

        def render_document(self, document, **kwargs):
            calls.append((document, kwargs))

    map_calls, pixel_calls = [], []
    document = object()
    window._geometric_layer_visible = True
    window._geometric_layer_document = document
    window._geometric_layer_display_overrides = {"m": GeometricLayerModuleDisplayOverride(None, 2)}
    window.map_view = SimpleNamespace(
        map=SimpleNamespace(
            image_size=(100, 80),
            lambert_to_pixel=lambda x, y: map_calls.append((x, y)) or (x / 10, y / 10),
        ),
        pixel_to_screen=lambda x, y: pixel_calls.append((x, y)) or (x + 20, y + 20),
        canvas=object(),
    )
    monkeypatch.setattr("src.assembleur_deformation_window.GeometricLayerRenderer", _Renderer)

    window._render_geometric_layer()

    assert calls[1] == (document, {"module_ids": None, "display_overrides": {"m": GeometricLayerModuleDisplayOverride(None, 2)}, "clear": False})
    context = calls[0][1]
    assert context.lambert_to_image(100, 200) == (10, 20)
    assert context.image_to_canvas(10, 20) == (30, 40)
    assert map_calls == [(100, 200)]

    line_document = geometric_layer_document_from_dict({
        "schema_version": 2,
        "source": "AlgoSimulator",
        "algorithm": "Test",
        "segment": "1",
        "scope": {"type": "automatic_aggregation"},
        "modules": [{
            "id": "m",
            "label": "M",
            "traces": [{
                "geometry": {"type": "line_image_azimuth", "x_l93": 100, "y_l93": 200, "azimuth_deg": 90},
                "graphics": {
                    "name": "", "color_bgr": [1, 2, 3], "width": 1, "style": "plein",
                    "show_name": False, "visible": True, "tags": {}, "tooltips": [], "scenario_tooltips": [],
                },
            }],
        }],
    })

    class _Canvas:
        def __init__(self):
            self.lines = []

        def create_line(self, *coordinates, **_kwargs):
            self.lines.append(coordinates)
            return len(self.lines)

    canvas = _Canvas()
    GeometricLayerRenderer(canvas, context).render_document(line_document)
    assert canvas.lines[0] == pytest.approx((20.0, 40.0, 120.0, 40.0))
    assert map_calls == [(100, 200), (100, 200)]
    assert pixel_calls[-2] == pytest.approx((0.0, 20.0))
    assert pixel_calls[-1] == pytest.approx((100.0, 20.0))
