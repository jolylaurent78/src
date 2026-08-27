"""Manual viewer path backed exclusively by ScenarioHypothesis and Catalogue."""

from types import SimpleNamespace

import numpy as np
import pytest

from src.assembleur_catalogue import Catalogue
from src.assembleur_core import ScenarioAssemblage, TopologyElement
from src.assembleur_geometry_reference import GeometryReferenceResolver
from src.assembleur_scenario import (
    ScenarioHypothesis,
    materialize_catalogue_triangle,
    materialize_triangle,
)
from src.assembleur_tk import TriangleViewerManual
from src.canvas_objects_collection import CanvasObjectsCollection


class _Listbox:
    def __init__(self):
        self.entries = []
        self.colours = {}
        self.selected = ()

    def delete(self, *_args):
        self.entries.clear()

    def insert(self, _index, value):
        self.entries.append(value)

    def size(self):
        return len(self.entries)

    def itemconfig(self, index, **kwargs):
        self.colours[index] = kwargs

    def curselection(self):
        return self.selected

    def selection_clear(self, *_args):
        self.selected = ()

    def selection_set(self, index):
        self.selected = tuple(sorted(set(self.selected + (index,))))

    def nearest(self, _y):
        return 0

    def yview(self):
        return (0.0, 1.0)

    def yview_moveto(self, _position):
        pass


class _Canvas:
    def delete(self, *_args):
        pass

    def focus_set(self):
        pass

    def configure(self, **_kwargs):
        pass


class _Status:
    def config(self, **_kwargs):
        pass


def _catalogue_and_hypothesis():
    catalogue = Catalogue()
    triangle_ids = []
    for pair_index in range(16):
        base = catalogue.add_city(f"Base {pair_index}", 42.0 + pair_index / 100, 2.0)
        for item_in_pair in range(2):
            rank = pair_index * 2 + item_in_pair + 1
            opening = catalogue.add_city(f"Opening {rank}", 44.0 + rank / 100, 3.0)
            light = catalogue.add_city(f"Light {rank}", 46.0 + rank / 100, 4.0)
            triangle_ids.append(
                catalogue.add_triangle(
                    f"Note {rank}", opening.city_id, base.city_id, light.city_id
                ).triangle_id
            )
    return catalogue, ScenarioHypothesis(triangle_ids, "TPL-0001")


def _viewer(catalogue, hypothesis):
    viewer = TriangleViewerManual.__new__(TriangleViewerManual)
    scenario = ScenarioAssemblage("Manual", hypothesis=hypothesis)
    viewer.catalogue = catalogue
    viewer._triangle_list_triangle_ids = []
    viewer.listbox = _Listbox()
    viewer.status = _Status()
    viewer._last_triangle_selection = None
    viewer._in_triangle_select_guard = False
    viewer._drag = None
    viewer._list_drag_pending = None
    viewer._get_active_scenario = lambda: scenario
    viewer._last_drawn = []
    viewer.canvas_objects = CanvasObjectsCollection(viewer._last_drawn)
    viewer._project_core_element_to_last_drawn = lambda world, element_id: None
    viewer._redraw_from = lambda _entries: None
    viewer._rebuild_triangle_listbox_from_core = lambda: None
    return viewer, scenario


def test_manual_listbox_order_and_labels_come_from_hypothesis_and_catalogue():
    catalogue, hypothesis = _catalogue_and_hypothesis()
    hypothesis.triangle_ids_by_rank[0], hypothesis.triangle_ids_by_rank[2] = (
        hypothesis.triangle_ids_by_rank[2],
        hypothesis.triangle_ids_by_rank[0],
    )
    viewer, scenario = _viewer(catalogue, hypothesis)

    TriangleViewerManual._rebuild_triangle_listbox_from_core(viewer)

    assert viewer._triangle_list_triangle_ids == hypothesis.triangle_ids_by_rank
    assert len(viewer.listbox.entries) == 32
    assert viewer._get_triangle_id_from_listbox_index(0) == hypothesis.triangle_ids_by_rank[0]
    assert viewer._get_triangle_id_from_listbox_index(1) == hypothesis.triangle_ids_by_rank[1]
    first = catalogue.get_triangle(hypothesis.triangle_ids_by_rank[0])
    assert viewer.listbox.entries[0] == (
        f"01. B:{catalogue.get_city(first.base_city_id).name}  "
        f"L:{catalogue.get_city(first.light_city_id).name}"
    )
    assert scenario.topoWorld.get_used_source_triangle_ids() == frozenset()


def test_manual_preview_and_placement_use_catalogue_triangle_id_only():
    catalogue, hypothesis = _catalogue_and_hypothesis()
    viewer, scenario = _viewer(catalogue, hypothesis)
    TriangleViewerManual._rebuild_triangle_listbox_from_core(viewer)
    triangle_id = hypothesis.triangle_ids_by_rank[0]

    preview = viewer._triangle_from_index(0)
    assert preview["triangle_id"] == triangle_id
    assert "model" not in preview
    viewer._drag = {
        "triangle_id": triangle_id,
        "world_pts": {
            "O": np.array([10.0, 20.0]),
            "B": np.array([13.0, 20.0]),
            "L": np.array([10.0, 24.0]),
        },
    }

    viewer._place_dragged_triangle()

    assert scenario.is_placeholder is False
    element = next(iter(scenario.topoWorld.elements.values()))
    assert element.source_triangle_id == triangle_id
    assert "triRank" not in element.meta
    assert "modelId" not in element.meta
    assert scenario.topoWorld.get_used_source_triangle_ids() == frozenset({triangle_id})
    entry = {"topoElementId": element.element_id}
    assert viewer._build_triangle_display_label(entry) == "T1"

    element.set_pose(np.eye(2), np.zeros(2), mirrored=True)
    assert viewer._build_triangle_display_label(entry) == "T1S"

    with pytest.raises(ValueError, match="already used"):
        viewer._place_dragged_triangle()


def test_list_drag_keeps_only_the_catalogue_triangle_id_and_ui_state():
    catalogue, hypothesis = _catalogue_and_hypothesis()
    viewer, _scenario = _viewer(catalogue, hypothesis)
    TriangleViewerManual._rebuild_triangle_listbox_from_core(viewer)
    viewer.canvas = _Canvas()
    viewer._drag_preview_id = None

    TriangleViewerManual._on_list_mouse_down(
        viewer, SimpleNamespace(y=0, x_root=100, y_root=200),
    )

    assert viewer._drag["triangle_id"] == hypothesis.triangle_ids_by_rank[0]
    assert viewer._drag["triangle_ids"] == (hypothesis.triangle_ids_by_rank[0],)


def test_drag_world_points_are_resolved_from_the_catalogue_triangle_id():
    catalogue, hypothesis = _catalogue_and_hypothesis()
    viewer, _scenario = _viewer(catalogue, hypothesis)
    triangle_id = hypothesis.triangle_ids_by_rank[0]

    world_pts = TriangleViewerManual._build_drag_world_points(
        viewer, triangle_id, (10.0, 20.0),
    )

    assert world_pts["O"] == pytest.approx((10.0, 20.0))
    element = materialize_catalogue_triangle(catalogue, triangle_id)
    expected_bl = np.asarray(element.vertex_local_xy[2]) - np.asarray(element.vertex_local_xy[0])
    assert world_pts["L"] - world_pts["O"] == pytest.approx(expected_bl)


def test_modern_display_label_resolves_rank_from_hypothesis_order():
    catalogue, hypothesis = _catalogue_and_hypothesis()
    viewer, scenario = _viewer(catalogue, hypothesis)
    triangle_id = hypothesis.triangle_ids_by_rank[1]
    element = materialize_catalogue_triangle(catalogue, triangle_id)
    scenario.topoWorld.add_element_as_new_group(element)

    assert viewer._build_triangle_display_label({"topoElementId": element.element_id}) == "T2"


def test_modern_display_label_does_not_infer_rank_from_element_id():
    catalogue, hypothesis = _catalogue_and_hypothesis()
    viewer, scenario = _viewer(catalogue, hypothesis)
    element = materialize_catalogue_triangle(catalogue, hypothesis.triangle_ids_by_rank[5])
    element.element_id = "T05"
    scenario.topoWorld.add_element_as_new_group(element)

    assert viewer._build_triangle_display_label({"topoElementId": "T05"}) == "T6"


def test_deformed_local_triangle_keeps_its_catalogue_rank_in_main_and_deform_labels():
    catalogue, hypothesis = _catalogue_and_hypothesis()
    viewer, scenario = _viewer(catalogue, hypothesis)
    source = catalogue.get_triangle(hypothesis.triangle_ids_by_rank[24])
    local_light = scenario.reference.create_city("Lumiere deformee", 49.0, 3.0)
    local_triangle = scenario.reference.create_triangle(
        "Do deforme", source.opening_city_id, source.base_city_id, local_light.city_ref_id,
        catalogue_source_triangle_id=source.triangle_id,
    )
    hypothesis.triangle_ids_by_rank[24] = local_triangle.triangle_ref_id
    element = materialize_triangle(
        GeometryReferenceResolver(catalogue, scenario.reference),
        local_triangle.triangle_ref_id,
    )
    element.element_id = "T25"
    scenario.topoWorld.add_element_as_new_group(element)
    viewer._deformation_state.enter()
    viewer._deformation_state.select(element.element_id, scenario.topoWorld)

    assert viewer._build_triangle_display_label({"topoElementId": element.element_id}) == "T25"
    assert viewer._deformation_occurrence_label(element.element_id, "L") == "T25:L - Lumiere deformee"


def test_deform_occurrences_sort_local_triangles_by_their_catalogue_rank():
    catalogue, hypothesis = _catalogue_and_hypothesis()
    viewer, scenario = _viewer(catalogue, hypothesis)
    local_light = scenario.reference.create_city("Lumiere partagee", 49.0, 3.0)
    resolver = GeometryReferenceResolver(catalogue, scenario.reference)
    for rank, element_id in ((24, "T25"), (1, "T02")):
        source = catalogue.get_triangle(hypothesis.triangle_ids_by_rank[rank])
        local_triangle = scenario.reference.create_triangle(
            f"Do local {rank}", source.opening_city_id, source.base_city_id,
            local_light.city_ref_id, catalogue_source_triangle_id=source.triangle_id,
        )
        hypothesis.triangle_ids_by_rank[rank] = local_triangle.triangle_ref_id
        element = materialize_triangle(resolver, local_triangle.triangle_ref_id)
        element.element_id = element_id
        scenario.topoWorld.add_element_as_new_group(element)
    viewer._deformation_state.enter()
    viewer._deformation_state.select("T25", scenario.topoWorld)

    assert viewer._deformation_occurrences_for_city(local_light.city_ref_id) == (
        ("T02", "L"), ("T25", "L"),
    )


def test_effective_local_reference_rebuilds_the_list_and_can_be_placed_again():
    catalogue, hypothesis = _catalogue_and_hypothesis()
    viewer, scenario = _viewer(catalogue, hypothesis)
    source = catalogue.get_triangle(hypothesis.triangle_ids_by_rank[24])
    local_light = scenario.reference.create_city("Lumiere validee", 49.0, 3.0)
    local_triangle = scenario.reference.create_triangle(
        "Do local", source.opening_city_id, source.base_city_id, local_light.city_ref_id,
        catalogue_source_triangle_id=source.triangle_id,
    )
    hypothesis.triangle_ids_by_rank[24] = local_triangle.triangle_ref_id

    TriangleViewerManual._rebuild_triangle_listbox_from_core(viewer)
    assert viewer.listbox.entries[24] == "25. B:Base 12  L:Lumiere validee"
    assert viewer.listbox.colours[24]["fg"] == "black"

    used = materialize_triangle(
        GeometryReferenceResolver(catalogue, scenario.reference), local_triangle.triangle_ref_id,
    )
    used.element_id = "T25"
    scenario.topoWorld.add_element_as_new_group(used)
    TriangleViewerManual._update_triangle_listbox_colors(viewer)
    assert viewer.listbox.colours[24]["fg"] == "gray50"

    scenario.topoWorld.removeElementsAndRebuild([used.element_id])
    TriangleViewerManual._rebuild_triangle_listbox_from_core(viewer)
    assert viewer.listbox.colours[24]["fg"] == "black"
    assert TriangleViewerManual._validate_triangle_list_selection(viewer, (24,)) == (True, "")

    viewer._drag = {
        "triangle_id": local_triangle.triangle_ref_id,
        "world_pts": {
            "O": np.array([10.0, 20.0]),
            "B": np.array([13.0, 20.0]),
            "L": np.array([10.0, 24.0]),
        },
    }
    viewer._place_dragged_triangle()

    placed = next(iter(scenario.topoWorld.elements.values()))
    assert placed.source_triangle_id == local_triangle.triangle_ref_id
    assert placed.vertex_business_ids[2] == local_light.city_ref_id


def test_modern_display_label_rejects_missing_or_unknown_source_triangle_id():
    catalogue, hypothesis = _catalogue_and_hypothesis()
    viewer, scenario = _viewer(catalogue, hypothesis)
    missing = TopologyElement(
        name="Missing source", vertex_labels=["O", "B", "L"],
        vertex_types=["O", "B", "L"], edge_lengths_km=[3.0, 5.0, 4.0],
    )
    scenario.topoWorld.add_element_as_new_group(missing)
    with pytest.raises(ValueError, match="source_triangle_id absent"):
        viewer._build_triangle_display_label({"topoElementId": missing.element_id})

    unknown = TopologyElement(
        name="Unknown source", vertex_labels=["O", "B", "L"],
        vertex_types=["O", "B", "L"], edge_lengths_km=[3.0, 5.0, 4.0],
        source_triangle_id="TRI-9999",
    )
    scenario.topoWorld.add_element_as_new_group(unknown)
    with pytest.raises(ValueError, match="absent de l'hypoth"):
        viewer._build_triangle_display_label({"topoElementId": unknown.element_id})


def test_display_label_rejects_a_scenario_without_hypothesis():
    catalogue, _hypothesis = _catalogue_and_hypothesis()
    viewer, scenario = _viewer(catalogue, None)
    element = TopologyElement(
        name="Legacy", vertex_labels=["O", "B", "L"],
        vertex_types=["O", "B", "L"], edge_lengths_km=[3.0, 5.0, 4.0],
    )
    scenario.topoWorld.add_element_as_new_group(element)
    entry = {"topoElementId": element.element_id}
    with pytest.raises(ValueError, match="ScenarioHypothesis absente"):
        viewer._build_triangle_display_label(entry)


def test_used_catalogue_triangle_is_grey_and_becomes_available_after_removal():
    catalogue, hypothesis = _catalogue_and_hypothesis()
    viewer, scenario = _viewer(catalogue, hypothesis)
    TriangleViewerManual._rebuild_triangle_listbox_from_core(viewer)
    triangle_id = hypothesis.triangle_ids_by_rank[0]
    preview = viewer._triangle_from_index(0)
    viewer._drag = {
        "triangle_id": triangle_id,
        "world_pts": {"O": (0.0, 0.0), "B": (3.0, 0.0), "L": (0.0, 4.0)},
    }
    viewer._place_dragged_triangle()
    element_id = next(iter(scenario.topoWorld.elements))

    TriangleViewerManual._rebuild_triangle_listbox_from_core(viewer)
    assert viewer.listbox.colours[0]["fg"] == "gray50"
    assert hypothesis.triangle_ids_by_rank[0] == triangle_id

    scenario.topoWorld.removeElementsAndRebuild([element_id])
    TriangleViewerManual._rebuild_triangle_listbox_from_core(viewer)
    assert viewer.listbox.colours[0]["fg"] == "black"
    assert hypothesis.triangle_ids_by_rank[0] == triangle_id


def test_manual_scenarios_keep_distinct_hypothesis_order_and_usage_state():
    catalogue, first_hypothesis = _catalogue_and_hypothesis()
    second_hypothesis = first_hypothesis.clone()
    second_hypothesis.triangle_ids_by_rank[0], second_hypothesis.triangle_ids_by_rank[1] = (
        second_hypothesis.triangle_ids_by_rank[1],
        second_hypothesis.triangle_ids_by_rank[0],
    )
    viewer, first = _viewer(catalogue, first_hypothesis)
    second = ScenarioAssemblage("Second", hypothesis=second_hypothesis)

    first.topoWorld.add_element_as_new_group(
        materialize_catalogue_triangle(catalogue, first_hypothesis.triangle_ids_by_rank[0])
    )
    viewer._get_active_scenario = lambda: first
    TriangleViewerManual._rebuild_triangle_listbox_from_core(viewer)
    assert viewer._triangle_list_triangle_ids[0] == first_hypothesis.triangle_ids_by_rank[0]
    assert viewer.listbox.colours[0]["fg"] == "gray50"

    viewer._get_active_scenario = lambda: second
    TriangleViewerManual._rebuild_triangle_listbox_from_core(viewer)
    assert viewer._triangle_list_triangle_ids[0] == second_hypothesis.triangle_ids_by_rank[0]
    assert viewer.listbox.colours[0]["fg"] == "black"
