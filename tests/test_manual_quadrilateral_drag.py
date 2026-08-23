"""Core-first contract for the manual two-triangle drag."""

import numpy as np
import pytest
from types import SimpleNamespace

from src.assembleur_catalogue import Catalogue
from src.assembleur_core import ScenarioAssemblage, TopologyEdgeEdgeAttachment
from src.assembleur_projection import getCoreTriangleWorldPoints
from src.assembleur_scenario import ScenarioHypothesis
from src.assembleur_tk import TriangleViewerManual


class _Status:
    def config(self, **_kwargs):
        pass


class _Listbox:
    def __init__(self):
        self.selected = ()
        self.count = 2

    def curselection(self):
        return self.selected

    def selection_clear(self, *_args):
        self.selected = ()

    def selection_set(self, index):
        self.selected = tuple(sorted(set(self.selected + (int(index),))))

    def size(self):
        return self.count

    def nearest(self, y):
        return int(y)


class _Canvas:
    def __init__(self):
        self.created = []

    def delete(self, *_args):
        pass

    def focus_set(self):
        pass

    def configure(self, **_kwargs):
        pass

    def winfo_rootx(self):
        return 0

    def winfo_rooty(self):
        return 0

    def create_polygon(self, *coords, **_kwargs):
        self.created.append(coords)
        return len(self.created)

    def coords(self, _item_id, *_coords):
        pass


def _selection_viewer():
    catalogue = Catalogue()
    opening = catalogue.add_city("Opening", 1.0, 1.0)
    base = catalogue.add_city("Base", 2.0, 2.0)
    first_id = catalogue.add_triangle("A", opening.city_id, base.city_id, catalogue.add_city("L1", 3.0, 3.0).city_id).triangle_id
    second_id = catalogue.add_triangle("B", opening.city_id, base.city_id, catalogue.add_city("L2", 4.0, 4.0).city_id).triangle_id
    viewer = TriangleViewerManual.__new__(TriangleViewerManual)
    viewer.catalogue = catalogue
    viewer._get_active_scenario = lambda: ScenarioAssemblage("Manual", hypothesis=ScenarioHypothesis([first_id, second_id], "TPL"))
    viewer._triangle_list_triangle_ids = [first_id, second_id]
    viewer._last_triangle_selection = ()
    viewer._in_triangle_select_guard = False
    viewer._deformation_state = SimpleNamespace(active=False)
    viewer.listbox = _Listbox()
    viewer.canvas = _Canvas()
    viewer.status = _Status()
    viewer._drag_preview_id = None
    viewer._drag_preview_ids = []
    viewer._drag = None
    viewer._list_drag_pending = None
    viewer.offset = np.array([0.0, 0.0])
    viewer.zoom = 1.0
    viewer._world_to_screen = lambda point: (float(point[0]), float(-point[1]))
    return viewer


@pytest.mark.parametrize("first,second", [(0, 1), (1, 0)])
def test_mouse_selection_controller_is_deterministic_in_both_ctrl_orders(first, second):
    viewer = _selection_viewer()
    viewer._on_list_mouse_down(SimpleNamespace(y=first, x_root=0, y_root=0, state=0))
    assert viewer._last_triangle_selection == (first,)

    assert viewer._on_list_mouse_down(SimpleNamespace(y=second, x_root=0, y_root=0, state=0x0004)) == "break"
    assert viewer._last_triangle_selection == (0, 1)
    assert viewer.listbox.curselection() == (0, 1)

    assert viewer._drag["kind"] == "quadrilateral"
    assert len(viewer._drag["triangle_ids"]) == 2


def test_dragging_the_second_member_of_a_pair_is_quadrilateral():
    viewer = _selection_viewer()
    viewer._set_triangle_list_selection((0, 1))
    viewer._on_list_mouse_down(SimpleNamespace(y=1, x_root=0, y_root=0, state=0))
    assert viewer._drag["kind"] == "quadrilateral"


def test_simple_click_prepares_placement_without_button_motion():
    viewer = _selection_viewer()
    viewer._on_list_mouse_down(SimpleNamespace(y=0, x_root=0, y_root=0, state=0))
    assert viewer._last_triangle_selection == (0,)
    assert viewer._drag["kind"] == "triangle"

    viewer._on_list_mouse_down(SimpleNamespace(y=1, x_root=0, y_root=0, state=0x0004))
    assert viewer._last_triangle_selection == (0, 1)
    assert viewer._drag["kind"] == "quadrilateral"


def test_quad_preview_is_created_by_plain_canvas_motion_without_button():
    viewer = _selection_viewer()
    viewer._on_list_mouse_down(SimpleNamespace(y=0, x_root=0, y_root=0, state=0))
    viewer._on_list_mouse_down(SimpleNamespace(y=1, x_root=0, y_root=0, state=0x0004))
    viewer._update_list_drag_preview_at_canvas_xy(10, 10)
    assert viewer._drag["kind"] == "quadrilateral"
    assert len(viewer._drag_preview_ids) == 2
    reference_id = viewer._drag["reference_triangle_id"]
    assert viewer._drag["world_pts_by_triangle"][reference_id]["O"] == pytest.approx((10.0, -10.0))

    # Déplacements horizontal, vertical et diagonal : O suit toujours le curseur.
    for canvas_x, canvas_y in ((20, 10), (20, 30), (40, 50)):
        viewer._update_list_drag_preview_at_canvas_xy(canvas_x, canvas_y)
        assert viewer._drag["world_pts_by_triangle"][reference_id]["O"] == pytest.approx(
            (float(canvas_x), -float(canvas_y))
        )


def test_selection_cycles_and_invalid_base_preserve_controller_selection():
    viewer = _selection_viewer()
    # Construire / détruire / reconstruire la paire dans l'ordre inverse.
    viewer._on_list_mouse_down(SimpleNamespace(y=0, x_root=0, y_root=0, state=0))
    viewer._drag = None
    viewer._on_list_mouse_down(SimpleNamespace(y=1, x_root=0, y_root=0, state=0x0004))
    viewer._on_list_mouse_down(SimpleNamespace(y=1, x_root=0, y_root=0, state=0x0004))
    assert viewer._last_triangle_selection == (0,)
    viewer._on_list_mouse_down(SimpleNamespace(y=1, x_root=0, y_root=0, state=0x0004))
    assert viewer._last_triangle_selection == (0, 1)

    other_base = viewer.catalogue.add_city("Other base", 5.0, 5.0)
    other_light = viewer.catalogue.add_city("Other light", 6.0, 6.0)
    opening_id = viewer.catalogue.get_triangle(viewer._triangle_list_triangle_ids[0]).opening_city_id
    invalid_id = viewer.catalogue.add_triangle("Invalid", opening_id, other_base.city_id, other_light.city_id).triangle_id
    viewer._triangle_list_triangle_ids.append(invalid_id)
    viewer.listbox.count = 3
    viewer._on_list_mouse_down(SimpleNamespace(y=1, x_root=0, y_root=0, state=0x0004))
    assert viewer._last_triangle_selection == (0,)
    viewer._on_list_mouse_down(SimpleNamespace(y=2, x_root=0, y_root=0, state=0x0004))
    assert viewer._last_triangle_selection == (0,)
    assert viewer.listbox.curselection() == (0,)


def test_manual_quadrilateral_commit_matches_core_preview_and_groups_both_triangles():
    catalogue = Catalogue()
    opening = catalogue.add_city("Opening", 1.0, 1.0)
    base = catalogue.add_city("Base", 2.0, 2.0)
    light_one = catalogue.add_city("Light one", 3.0, 3.0)
    light_two = catalogue.add_city("Light two", 4.0, 4.0)
    first_id = catalogue.add_triangle("A", opening.city_id, base.city_id, light_one.city_id).triangle_id
    second_id = catalogue.add_triangle("B", opening.city_id, base.city_id, light_two.city_id).triangle_id

    viewer = TriangleViewerManual.__new__(TriangleViewerManual)
    scenario = ScenarioAssemblage("Manual", hypothesis=ScenarioHypothesis([first_id, second_id], "TPL"))
    viewer.catalogue = catalogue
    viewer._get_active_scenario = lambda: scenario
    viewer._last_drawn = []
    viewer._rebuild_active_projection_from_core = lambda: None
    viewer._redraw_from = lambda _entries: None
    viewer._rebuild_triangle_listbox_from_core = lambda: None
    viewer.status = _Status()

    geometry = viewer._build_quadrilateral_drag_geometry((first_id, second_id))
    offset = np.array([12.0, -7.0])
    preview = {
        triangle_id: {key: point + offset for key, point in points.items()}
        for triangle_id, points in geometry["relative_world_pts"].items()
    }
    viewer._drag = {
        "kind": "quadrilateral",
        "triangle_ids": (first_id, second_id),
        "world_pts_by_triangle": preview,
    }

    viewer._place_dragged_quadrilateral()

    assert scenario.is_placeholder is False
    world = scenario.topoWorld
    assert len(world.elements) == 2
    assert world.get_used_source_triangle_ids() == frozenset({first_id, second_id})
    attachments = list(world.attachments.values())
    assert len(attachments) == 1
    assert isinstance(attachments[0], TopologyEdgeEdgeAttachment)
    assert attachments[0].mob_edge == attachments[0].dest_edge == "OB"
    element_by_source = {element.source_triangle_id: element.element_id for element in world.elements.values()}
    assert world.get_group_of_element(element_by_source[first_id]) == world.get_group_of_element(element_by_source[second_id])
    for triangle_id, element_id in element_by_source.items():
        for key, point in preview[triangle_id].items():
            assert getCoreTriangleWorldPoints(world, element_id)[key] == pytest.approx(point)
