import pytest

from src.assembleur_geometric_layer_display import GeometricLayerModuleDisplayOverride
from src.assembleur_geometric_layer_io import geometric_layer_document_from_dict
from src.assembleur_geometric_layer_renderer import (
    GeometricLayerRenderContext,
    GeometricLayerRenderer,
    bgr_to_tk_color,
    clip_infinite_line_to_image,
)


class FakeCanvas:
    def __init__(self) -> None:
        self.calls = []
        self.deleted = []

    def _create(self, kind, args, kwargs):
        item_id = len(self.calls) + 1
        self.calls.append((kind, args, kwargs, item_id))
        return item_id

    def create_line(self, *args, **kwargs): return self._create("line", args, kwargs)
    def create_oval(self, *args, **kwargs): return self._create("oval", args, kwargs)
    def create_polygon(self, *args, **kwargs): return self._create("polygon", args, kwargs)
    def create_text(self, *args, **kwargs): return self._create("text", args, kwargs)
    def delete(self, tag): self.deleted.append(tag)


def _trace(geometry, *, visible=True, show_name=False, name="Trace", style="plein"):
    return {"geometry": geometry, "graphics": {
        "name": name, "color_bgr": [36, 28, 237], "width": 3, "style": style,
        "show_name": show_name, "visible": visible, "tags": {}, "tooltips": [], "scenario_tooltips": [],
    }}


def _document(*modules):
    return geometric_layer_document_from_dict({
        "schema_version": 2, "source": "AlgoSimulator", "algorithm": "Test", "segment": "1",
        "scope": {"type": "automatic_aggregation"}, "modules": list(modules),
    })


def _context():
    return GeometricLayerRenderContext((100, 80), lambda x, y: (x, y), lambda x, y: (x * 10, y * 10))


def test_bgr_conversion_and_line_clipping_are_exact() -> None:
    assert bgr_to_tk_color((36, 28, 237)) == "#ED1C24"
    assert clip_infinite_line_to_image((50, 40), (1, 0), (100, 80)) == ((0.0, 40.0), (100.0, 40.0))
    assert clip_infinite_line_to_image((50, 40), (0, 1), (100, 80)) == ((50.0, 0.0), (50.0, 80.0))


@pytest.mark.parametrize(
    ("azimuth", "expected"), [(0, (50.0, 80.0, 50.0, 0.0)), (90, (0.0, 40.0, 100.0, 40.0))],
)
def test_image_azimuth_keeps_its_image_orientation(azimuth, expected) -> None:
    document = _document({"id": "m", "label": "M", "traces": [_trace({"type": "line_image_azimuth", "x_l93": 50, "y_l93": 40, "azimuth_deg": azimuth})]})
    canvas = FakeCanvas()
    GeometricLayerRenderer(canvas, _context()).render_document(document)
    assert canvas.calls[0][0] == "line"
    assert canvas.calls[0][1] == pytest.approx(tuple(value * 10 for value in expected))


def test_vertical_horizontal_infinite_line_and_finite_segment_are_distinct() -> None:
    document = _document({"id": "m", "label": "M", "traces": [
        _trace({"type": "vertical_image_line", "x_l93": 20, "y_l93": 30}),
        _trace({"type": "horizontal_image_line", "x_l93": 20, "y_l93": 30}),
        _trace({"type": "segment", "x1_l93": 2, "y1_l93": 3, "x2_l93": 4, "y2_l93": 5}),
        _trace({"type": "line_between_points", "x1_l93": 20, "y1_l93": 30, "x2_l93": 30, "y2_l93": 30}),
    ]})
    canvas = FakeCanvas()
    GeometricLayerRenderer(canvas, _context()).render_document(document)
    assert canvas.calls[0][1] == (200.0, 0.0, 200.0, 800.0)
    assert canvas.calls[1][1] == (0.0, 300.0, 1000.0, 300.0)
    assert canvas.calls[2][1] == (20.0, 30.0, 40.0, 50.0)
    assert canvas.calls[3][1] == (0.0, 300.0, 1000.0, 300.0)


def test_circle_and_positive_negative_arcs_are_sampled_in_image_before_canvas_projection() -> None:
    document = _document({"id": "m", "label": "M", "traces": [
        _trace({"type": "circle", "center_x_l93": 50, "center_y_l93": 40, "radius_km": 0.01}),
        _trace({"type": "arc", "center_x_l93": 50, "center_y_l93": 40, "radius_km": 0.01, "start_azimuth_deg": 0, "rotation_deg": 90}),
        _trace({"type": "arc", "center_x_l93": 50, "center_y_l93": 40, "radius_km": 0.01, "start_azimuth_deg": 350, "rotation_deg": -180}),
    ]})
    canvas = FakeCanvas()
    GeometricLayerRenderer(canvas, _context()).render_document(document)
    circle, positive_arc, negative_arc = (call[1] for call in canvas.calls)
    assert circle[:2] == pytest.approx((500.0, 300.0))
    assert circle[-2:] == pytest.approx((500.0, 300.0))
    assert positive_arc[:2] == pytest.approx((500.0, 300.0))
    assert positive_arc[-2:] == pytest.approx((600.0, 400.0))
    assert negative_arc[:2] == pytest.approx((482.635182, 301.519225))


def test_point_symbol_visibility_module_filter_name_and_clear() -> None:
    document = _document(
        {"id": "one", "label": "One", "traces": [
            _trace({"type": "point", "x_l93": 1, "y_l93": 2}, show_name=True, name="Point"),
            _trace({"type": "symbol", "x_l93": 3, "y_l93": 4, "source": "https://example.test/symbol"}),
        ]},
        {"id": "two", "label": "Two", "traces": [_trace({"type": "point", "x_l93": 5, "y_l93": 6}, visible=False)]},
    )
    canvas = FakeCanvas()
    renderer = GeometricLayerRenderer(canvas, _context())
    result = renderer.render_document(document, module_ids={"one"})
    assert result.traces_drawn == 2
    assert [call[0] for call in canvas.calls] == ["oval", "text", "polygon"]
    assert canvas.calls[0][2]["fill"] == "#ED1C24"
    assert canvas.calls[0][2]["width"] == 3.0
    assert canvas.calls[0][2]["tags"] == ("geometric-layer", "geometric-layer:one")
    renderer.clear()
    assert canvas.deleted == ["geometric-layer"]


def test_nullable_show_name_does_not_draw_a_label() -> None:
    document = _document({"id": "one", "label": "One", "traces": [
        _trace({"type": "point", "x_l93": 1, "y_l93": 2}, show_name=None, name="Sans libellé"),
    ]})
    canvas = FakeCanvas()

    GeometricLayerRenderer(canvas, _context()).render_document(document)

    assert [call[0] for call in canvas.calls] == ["oval"]


def test_module_display_overrides_apply_to_primitives_labels_and_arrow_without_mutating_document() -> None:
    document = _document(
        {"id": "light", "label": "Light", "traces": [
            _trace({"type": "point", "x_l93": 1, "y_l93": 2}, show_name=True, name="Point"),
            _trace({"type": "arc", "center_x_l93": 50, "center_y_l93": 40, "radius_km": 0.01, "start_azimuth_deg": 0, "rotation_deg": 90}, style="Arrow"),
        ]},
        {"id": "shadow", "label": "Shadow", "traces": [_trace({"type": "point", "x_l93": 3, "y_l93": 4})]},
    )
    canvas = FakeCanvas()
    original_color = document.modules[0].traces[0].graphics.color_bgr
    original_width = document.modules[0].traces[0].graphics.width

    GeometricLayerRenderer(canvas, _context()).render_document(
        document,
        display_overrides={"light": GeometricLayerModuleDisplayOverride((1, 2, 3), 7), "absent": GeometricLayerModuleDisplayOverride((9, 9, 9), 9)},
    )

    light_calls = [call for call in canvas.calls if "geometric-layer:light" in call[2]["tags"]]
    shadow_call = next(call for call in canvas.calls if "geometric-layer:shadow" in call[2]["tags"])
    assert all(call[2].get("fill") == "#030201" for call in light_calls)
    assert all(call[2].get("width") == 7 for call in light_calls if call[0] != "text")
    assert shadow_call[2]["fill"] == "#ED1C24"
    assert shadow_call[2]["width"] == 3.0
    assert document.modules[0].traces[0].graphics.color_bgr == original_color
    assert document.modules[0].traces[0].graphics.width == original_width


def test_circle_is_constructed_in_image_space_after_a_non_uniform_calibration() -> None:
    document = _document({"id": "m", "label": "M", "traces": [
        _trace({"type": "circle", "center_x_l93": 10, "center_y_l93": 20, "radius_km": 0.001}),
    ]})
    context = GeometricLayerRenderContext((100, 80), lambda x, y: (2 * x, 3 * y), lambda x, y: (x, y))
    canvas = FakeCanvas()
    GeometricLayerRenderer(canvas, context).render_document(document)
    coordinates = canvas.calls[0][1]
    # Centre image (20, 60), rayon image dérivé de l'arête Lambert +X : 2 px.
    assert coordinates[:2] == pytest.approx((20.0, 58.0))
    assert coordinates[24 * 2:24 * 2 + 2] == pytest.approx((20.0, 62.0))


def test_arc_keeps_image_azimuth_and_arrow_follows_its_final_tangent() -> None:
    document = _document({"id": "m", "label": "M", "traces": [
        _trace({"type": "arc", "center_x_l93": 50, "center_y_l93": 40, "radius_km": 0.01, "start_azimuth_deg": 0, "rotation_deg": 90}, style="Arrow"),
        _trace({"type": "arc", "center_x_l93": 50, "center_y_l93": 40, "radius_km": 0.01, "start_azimuth_deg": 90, "rotation_deg": -90}),
    ]})
    canvas = FakeCanvas()
    GeometricLayerRenderer(canvas, _context()).render_document(document)
    positive_arc = canvas.calls[0][1]
    arrow = canvas.calls[1]
    negative_arc = canvas.calls[2][1]
    assert positive_arc[:2] == pytest.approx((500.0, 300.0))
    assert positive_arc[-2:] == pytest.approx((600.0, 400.0))
    assert arrow[0] == "line" and arrow[2]["arrow"] == "last"
    assert arrow[1][-2:] == pytest.approx((600.0, 400.0))
    assert negative_arc[:2] == pytest.approx((600.0, 400.0))
    assert negative_arc[-2:] == pytest.approx((500.0, 300.0))


def test_outside_infinite_line_does_not_create_a_label_or_rendered_trace() -> None:
    document = _document({"id": "m", "label": "M", "traces": [
        _trace({"type": "horizontal_image_line", "x_l93": 50, "y_l93": 200}, show_name=True, name="Invisible"),
    ]})
    canvas = FakeCanvas()
    result = GeometricLayerRenderer(canvas, _context()).render_document(document)
    assert result.traces_drawn == 0
    assert result.rendered_traces == ()
    assert canvas.calls == []
