"""REF-001B: commits DEFORM copy-on-write sans mutation Catalogue."""

from types import SimpleNamespace

import numpy as np

import pytest

from src.assembleur_catalogue import Catalogue
from src.assembleur_core import ScenarioAssemblage, TopologyWorld
from src.assembleur_deformation import (
    commit_deformation_copy_on_write,
    simulate_occurrence_deformation,
)
from src.assembleur_geometry_reference import GeometryReferenceResolver
from src.assembleur_deformation_ui import WorkingPoint
from src.assembleur_scenario import ScenarioHypothesis, materialize_triangle
from src.assembleur_tk import TriangleViewerManual


def _scenario_with_catalogue_triangle():
    catalogue = Catalogue()
    triangle_ids = []
    for pair_index in range(16):
        base = catalogue.add_city(f"Base {pair_index}", 45.0 + pair_index / 10, 2.0)
        for member_index in range(2):
            rank = pair_index * 2 + member_index + 1
            opening = catalogue.add_city(f"Ouverture {rank}", 44.0 + rank / 100, 1.5)
            light = catalogue.add_city(f"Lumiere {rank}", 46.0 + rank / 100, 2.5)
            triangle_ids.append(
                catalogue.add_triangle("Do", opening.city_id, base.city_id, light.city_id).triangle_id
            )
    triangle = catalogue.get_triangle(triangle_ids[24])
    scenario = ScenarioAssemblage("DEFORM", hypothesis=ScenarioHypothesis(triangle_ids))
    resolver = GeometryReferenceResolver(catalogue, scenario.reference)
    world = TopologyWorld()
    element = materialize_triangle(resolver, triangle.triangle_id)
    element.element_id = "T25"
    world.add_element_as_new_group(element)
    scenario.topoWorld = world
    return catalogue, scenario, triangle, element.element_id


def _preview(catalogue, scenario, element_id, overrides):
    return simulate_occurrence_deformation(
        resolver=GeometryReferenceResolver(catalogue, scenario.reference),
        initial_world=scenario.topoWorld,
        occurrence_lambert_overrides=overrides,
    ).world


def _working_points(*items):
    """(point_id, coordonnee, occurrences) -> points DEFORM explicites."""
    return {
        point_id: WorkingPoint(point_id, coordinate, set(occurrences))
        for point_id, coordinate, occurrences in items
    }


def _viewer_with_dirty_cow_preview():
    catalogue, scenario, _triangle, element_id = _scenario_with_catalogue_triangle()
    viewer = TriangleViewerManual.__new__(TriangleViewerManual)
    viewer.catalogue = catalogue
    viewer._get_active_scenario = lambda: scenario
    viewer.status = type("Status", (), {"config": lambda self, **_kwargs: None})()
    viewer._show_deformation_preview = lambda _world: None
    viewer._refresh_deformation_window = lambda **_kwargs: None
    viewer._rebuild_triangle_listbox_from_core = lambda: None
    viewer._screen_to_world = lambda x, y: (float(x), float(y))
    state = viewer._deformation_state
    state.enter()
    state.working_reference = scenario.reference.clone()
    state.working_hypothesis = scenario.hypothesis.clone()
    state.select(element_id, scenario.topoWorld)
    state.begin_drag("L")
    original = GeometryReferenceResolver(
        catalogue, scenario.reference
    ).get_city_lambert(scenario.topoWorld.elements[element_id].vertex_business_ids[2])
    state.ensure_working_point((element_id, "L"), original)
    preview = viewer._apply_deformation_occurrence_overrides(
        state.candidate_occurrence_overrides((700000.0, 6600000.0))
    )
    state.accept_occurrence_candidate((700000.0, 6600000.0), preview)
    return viewer, scenario, element_id


def _published_snapshot(scenario):
    return {
        "reference": scenario.reference.clone(),
        "hypothesis": list(scenario.hypothesis.triangle_ids_by_rank),
        "poses": {
            element_id: scenario.topoWorld.getElementPose(element_id)
            for element_id in scenario.topoWorld.elements
        },
        "sources": {
            element_id: element.source_triangle_id
            for element_id, element in scenario.topoWorld.elements.items()
        },
    }


def _assert_published_snapshot(scenario, snapshot):
    assert scenario.reference.cities == snapshot["reference"].cities
    assert scenario.reference.triangles == snapshot["reference"].triangles
    assert scenario.hypothesis.triangle_ids_by_rank == snapshot["hypothesis"]
    assert {
        element_id: element.source_triangle_id
        for element_id, element in scenario.topoWorld.elements.items()
    } == snapshot["sources"]
    for element_id, expected in snapshot["poses"].items():
        actual = scenario.topoWorld.getElementPose(element_id)
        np.testing.assert_allclose(actual[0], expected[0])
        np.testing.assert_allclose(actual[1], expected[1])
        assert actual[2] is expected[2]


def _prepare_deformation_release(viewer):
    viewer._clock_dragging = False
    viewer._bg_resizing = None
    viewer._bg_moving = None
    viewer._pan_anchor = None
    viewer._drag = None
    viewer._reset_assist = lambda: None


def test_dirty_deformation_main_rotation_uses_stri_preview_without_publishing():
    viewer, scenario, element_id = _viewer_with_dirty_cow_preview()
    state = viewer._deformation_state
    published_before = _published_snapshot(scenario)
    preview_before = state.last_accepted_world.clonePhysicalState()
    group_id = preview_before.get_group_of_element(element_id)
    viewer._sel = {
        "mode": "rotate_group_anchor_drag",
        "core_group_id": group_id,
        "pivot_world": np.array((0.0, 0.0)),
        "mouse_angle_start": 0.0,
        "deformation_base_world": preview_before,
    }

    _prepare_deformation_release(viewer)
    viewer._on_canvas_left_move(SimpleNamespace(x=0.0, y=1.0))
    first_motion_pose = state.last_accepted_world.getElementPose(element_id)
    viewer._on_canvas_left_move(SimpleNamespace(x=0.0, y=1.0))
    second_motion_pose = state.last_accepted_world.getElementPose(element_id)
    np.testing.assert_allclose(second_motion_pose[0], first_motion_pose[0])
    np.testing.assert_allclose(second_motion_pose[1], first_motion_pose[1])
    viewer._on_canvas_left_up(SimpleNamespace(x=0.0, y=1.0))

    assert state.last_accepted_world.elements[element_id].source_triangle_id.startswith("STRI-")
    assert state.working_hypothesis.get_rank_for_triangle_ref(
        state.last_accepted_world.elements[element_id].source_triangle_id
    ) == 25
    assert state.dirty is True
    assert viewer._sel is None
    _assert_published_snapshot(scenario, published_before)

    viewer._validate_deformation_session()

    assert scenario.topoWorld.elements[element_id].source_triangle_id.startswith("STRI-")
    assert scenario.hypothesis.get_rank_for_triangle_ref(
        scenario.topoWorld.elements[element_id].source_triangle_id
    ) == 25
    assert state.dirty is False


def test_dirty_deformation_main_move_is_candidate_only_and_close_discards_it():
    viewer, scenario, element_id = _viewer_with_dirty_cow_preview()
    state = viewer._deformation_state
    published_before = _published_snapshot(scenario)
    preview_before = state.last_accepted_world.clonePhysicalState()
    group_id = preview_before.get_group_of_element(element_id)
    viewer._sel = {
        "mode": "move_group",
        "core_group_id": group_id,
        "mouse_world_start": np.array((0.0, 0.0)),
        "deformation_base_world": preview_before,
    }

    _prepare_deformation_release(viewer)
    viewer._on_canvas_left_move(SimpleNamespace(x=10.0, y=-5.0))
    viewer._on_canvas_left_up(SimpleNamespace(x=20.0, y=-10.0))

    before_pose = preview_before.getElementPose(element_id)
    after_pose = state.last_accepted_world.getElementPose(element_id)
    np.testing.assert_allclose(after_pose[1], before_pose[1] + np.array((20.0, -10.0)))
    assert state.last_accepted_world.elements[element_id].source_triangle_id.startswith("STRI-")
    assert state.dirty is True
    assert viewer._sel is None
    _assert_published_snapshot(scenario, published_before)

    state.exit()
    _assert_published_snapshot(scenario, published_before)


def test_canvas_display_context_uses_the_preview_world_and_hypothesis():
    catalogue, scenario, triangle, element_id = _scenario_with_catalogue_triangle()
    viewer = TriangleViewerManual.__new__(TriangleViewerManual)
    viewer.catalogue = catalogue
    viewer._get_active_scenario = lambda: scenario
    preview = _preview(
        catalogue, scenario, element_id, {(element_id, "L"): (700000.0, 6600000.0)},
    )
    committed = commit_deformation_copy_on_write(
        catalogue=catalogue,
        scenario=scenario,
        preview_world=preview,
        working_points=_working_points(
            ("TMP-1", (700000.0, 6600000.0), ((element_id, "L"),)),
        ),
    )
    entry = {"topoElementId": element_id}

    assert viewer._get_canvas_display_world() is scenario.topoWorld
    viewer._deformation_state.enter()
    viewer._deformation_state.reference_world = scenario.topoWorld
    viewer._deformation_state.last_accepted_world = None
    assert viewer._get_canvas_display_world() is scenario.topoWorld

    viewer._deformation_state.last_accepted_world = committed.world
    viewer._deformation_state.working_reference = committed.reference
    viewer._deformation_state.working_hypothesis = committed.hypothesis
    assert viewer._get_canvas_display_world() is committed.world
    assert viewer._get_canvas_display_hypothesis() is committed.hypothesis
    assert viewer._get_core_element_from_last_drawn_entry(entry) is committed.world.elements[element_id]
    assert viewer._get_core_vertex_labels(entry)[2] == "Temp Lumiere 25"
    assert viewer._build_triangle_display_label(entry) == "T25"

    class _Tooltip:
        def winfo_width(self):
            return 100

        def winfo_height(self):
            return 40

        def wm_geometry(self, _value):
            pass

        def deiconify(self):
            pass

    class _Canvas:
        def winfo_rootx(self):
            return 0

        def winfo_rooty(self):
            return 0

    tooltip_texts = []
    viewer._clock_trace_active = False
    viewer._clock_measure_active = False
    viewer._clock_arc_active = False
    viewer._clock_setref_active = False
    viewer._ensure_pick_cache = lambda: None
    viewer._drag = None
    viewer._sel = None
    viewer._ctrl_down = False
    viewer._last_drawn = [{"topoElementId": element_id, "pts": {"L": (0.0, 0.0)}}]
    viewer._hit_test = lambda *_args: ("vertex", 0, "L")
    viewer._world_to_screen = lambda point: point
    viewer._ensure_canvas_tooltip = lambda text: tooltip_texts.append(text)
    viewer._tooltip = _Tooltip()
    viewer._computeNodeTooltipCanvasPosition = lambda *_args: (0, 0)
    viewer._hide_tooltip = lambda: None
    viewer._pick_cache_valid = True
    viewer.canvas = _Canvas()

    viewer._on_canvas_motion_update_drag(type("Event", (), {"x": 0, "y": 0})())

    assert "Temp Lumiere 25" in tooltip_texts[0]

    viewer._deformation_state.exit()
    assert viewer._get_canvas_display_world() is scenario.topoWorld
    assert viewer._get_core_vertex_labels(entry)[2] == catalogue.cities[
        triangle.light_city_id
    ].name


def test_first_deformation_creates_one_local_triangle_and_city_without_mutating_catalogue():
    catalogue, scenario, triangle, element_id = _scenario_with_catalogue_triangle()
    catalogue_before = catalogue.clone()
    preview = _preview(catalogue, scenario, element_id, {(element_id, "L"): (700000.0, 6600000.0)})

    committed = commit_deformation_copy_on_write(
        catalogue=catalogue,
        scenario=scenario,
        preview_world=preview,
        working_points=_working_points(
            ("TMP-1", (700000.0, 6600000.0), ((element_id, "L"),)),
        ),
    )

    assert scenario.reference.cities == {}
    assert scenario.topoWorld.elements[element_id].source_triangle_id == triangle.triangle_id
    assert len(committed.reference.cities) == 1
    assert len(committed.reference.triangles) == 1
    local = next(iter(committed.reference.triangles.values()))
    assert local.opening_city_ref_id == triangle.opening_city_id
    assert local.base_city_ref_id == triangle.base_city_id
    assert local.light_city_ref_id.startswith("SCITY-")
    assert committed.reference.cities[local.light_city_ref_id].name == "Temp Lumiere 25"
    assert committed.world.elements[element_id].source_triangle_id == local.triangle_ref_id
    assert committed.hypothesis.triangle_ids_by_rank[24] == local.triangle_ref_id
    assert committed.hypothesis.get_rank_for_triangle_ref(local.triangle_ref_id) == 25
    with pytest.raises(ValueError, match="Référence effective absente"):
        committed.hypothesis.get_rank_for_triangle_ref(triangle.triangle_id)
    assert committed.world.elements[element_id].vertex_business_ids == [
        triangle.opening_city_id, triangle.base_city_id, local.light_city_ref_id,
    ]
    assert catalogue.cities == catalogue_before.cities
    assert catalogue.triangles == catalogue_before.triangles


def test_preview_uses_the_temporary_cow_reference_before_validation():
    catalogue, scenario, _triangle, element_id = _scenario_with_catalogue_triangle()
    viewer = TriangleViewerManual.__new__(TriangleViewerManual)
    viewer.catalogue = catalogue
    viewer._get_active_scenario = lambda: scenario
    viewer.status = type("Status", (), {"config": lambda self, **_kwargs: None})()
    state = viewer._deformation_state
    state.enter()
    state.working_reference = scenario.reference.clone()
    state.working_hypothesis = scenario.hypothesis.clone()
    state.select(element_id, scenario.topoWorld)
    state.begin_drag("L")
    original_point = GeometryReferenceResolver(
        catalogue, scenario.reference
    ).get_city_lambert(_triangle.light_city_id)
    state.ensure_working_point((element_id, "L"), original_point)
    candidate_point = (700000.0, 6600000.0)

    preview = viewer._apply_deformation_occurrence_overrides(
        state.candidate_occurrence_overrides(candidate_point)
    )
    assert preview is not None
    state.accept_occurrence_candidate(candidate_point, preview)

    local_triangle = next(iter(state.working_reference.triangles.values()))
    local_city_id = local_triangle.light_city_ref_id
    assert state.working_reference.cities[local_city_id].name == "Temp Lumiere 25"
    assert scenario.reference.cities == {}
    assert preview.elements[element_id].vertex_labels[2] == "Temp Lumiere 25"
    assert viewer._deformation_vertices()["L"].name == "Temp Lumiere 25"

    state.restore_working_point((element_id, "L"))
    restored_preview = viewer._apply_deformation_occurrence_overrides({})
    assert restored_preview is scenario.topoWorld
    assert state.working_reference.cities == {}
    assert scenario.reference.cities == {}

    state.begin_drag("L")
    state.ensure_working_point((element_id, "L"), original_point)
    preview = viewer._apply_deformation_occurrence_overrides(
        state.candidate_occurrence_overrides(candidate_point)
    )
    assert preview is not None
    state.accept_occurrence_candidate(candidate_point, preview)
    viewer._rebuild_triangle_listbox_from_core = lambda: None
    viewer._show_deformation_preview = lambda _world: None
    viewer._refresh_deformation_window = lambda **_kwargs: None

    viewer._validate_deformation_session()

    assert len(scenario.reference.cities) == 1
    assert next(iter(scenario.reference.cities.values())).name == "Temp Lumiere 25"
    assert scenario.topoWorld.elements[element_id].vertex_labels[2] == "Temp Lumiere 25"


def test_temporary_city_rename_stays_in_working_reference_until_validation(monkeypatch):
    catalogue, scenario, triangle, element_id = _scenario_with_catalogue_triangle()
    viewer = TriangleViewerManual.__new__(TriangleViewerManual)
    viewer.catalogue = catalogue
    viewer._get_active_scenario = lambda: scenario
    viewer.status = type("Status", (), {"config": lambda self, **_kwargs: None})()
    state = viewer._deformation_state
    state.enter()
    state.working_reference = scenario.reference.clone()
    state.working_hypothesis = scenario.hypothesis.clone()
    state.select(element_id, scenario.topoWorld)
    state.begin_drag("L")
    original_point = GeometryReferenceResolver(
        catalogue, scenario.reference
    ).get_city_lambert(triangle.light_city_id)
    state.ensure_working_point((element_id, "L"), original_point)
    candidate_point = (700000.0, 6600000.0)
    preview = viewer._apply_deformation_occurrence_overrides(
        state.candidate_occurrence_overrides(candidate_point)
    )
    assert preview is not None
    state.accept_occurrence_candidate(candidate_point, preview)
    state.select_occurrence(element_id, "L")
    city_ref_id = viewer._deformation_city_id_for_occurrence(element_id, "L")
    viewer._show_deformation_preview = lambda _world: None
    viewer._refresh_deformation_window = lambda **_kwargs: None
    monkeypatch.setattr(
        "src.assembleur_tk.simpledialog.askstring", lambda *_args, **_kwargs: "Point 560"
    )

    assert viewer._deformation_selected_occurrence_is_local_city() is True
    viewer._deformation_rename_selected()

    assert state.dirty is True
    assert state.working_reference.cities[city_ref_id].name == "Point 560"
    assert scenario.reference.cities == {}
    assert state.last_accepted_world.elements[element_id].vertex_labels[2] == "Point 560"

    state.exit()
    assert scenario.reference.cities == {}


def test_temporary_city_rename_is_published_by_deformation_validation():
    catalogue, scenario, triangle, element_id = _scenario_with_catalogue_triangle()
    viewer = TriangleViewerManual.__new__(TriangleViewerManual)
    viewer.catalogue = catalogue
    viewer._get_active_scenario = lambda: scenario
    viewer.status = type("Status", (), {"config": lambda self, **_kwargs: None})()
    state = viewer._deformation_state
    state.enter()
    state.working_reference = scenario.reference.clone()
    state.working_hypothesis = scenario.hypothesis.clone()
    state.select(element_id, scenario.topoWorld)
    state.begin_drag("L")
    original_point = GeometryReferenceResolver(
        catalogue, scenario.reference
    ).get_city_lambert(triangle.light_city_id)
    state.ensure_working_point((element_id, "L"), original_point)
    candidate_point = (700000.0, 6600000.0)
    preview = viewer._apply_deformation_occurrence_overrides(
        state.candidate_occurrence_overrides(candidate_point)
    )
    assert preview is not None
    state.accept_occurrence_candidate(candidate_point, preview)
    city_ref_id = viewer._deformation_city_id_for_occurrence(element_id, "L")
    viewer._rename_working_deformation_city(city_ref_id, "Point 560")
    local_triangle_id = next(iter(state.working_reference.triangles))
    state.begin_drag("L")
    second_point = (710000.0, 6610000.0)
    second_preview = viewer._apply_deformation_occurrence_overrides(
        state.candidate_occurrence_overrides(second_point)
    )
    assert second_preview is not None
    state.accept_occurrence_candidate(second_point, second_preview)

    assert set(state.working_reference.cities) == {city_ref_id}
    assert set(state.working_reference.triangles) == {local_triangle_id}
    assert state.working_reference.cities[city_ref_id].name == "Point 560"
    assert state.last_accepted_world.elements[element_id].vertex_labels[2] == "Point 560"

    viewer._rebuild_triangle_listbox_from_core = lambda: None
    viewer._show_deformation_preview = lambda _world: None
    viewer._refresh_deformation_window = lambda **_kwargs: None

    viewer._validate_deformation_session()

    assert scenario.reference.cities[city_ref_id].name == "Point 560"
    assert scenario.topoWorld.elements[element_id].vertex_labels[2] == "Point 560"


def test_temporary_shared_city_rename_rematerializes_every_preview_occurrence():
    catalogue, scenario, _triangle, first_id = _scenario_with_catalogue_triangle()
    second_triangle_id = scenario.hypothesis.triangle_ids_by_rank[23]
    second = materialize_triangle(
        GeometryReferenceResolver(catalogue, scenario.reference), second_triangle_id,
    )
    scenario.topoWorld.add_element_as_new_group(second)
    second_id = second.element_id
    viewer = TriangleViewerManual.__new__(TriangleViewerManual)
    viewer.catalogue = catalogue
    viewer._get_active_scenario = lambda: scenario
    viewer.status = type("Status", (), {"config": lambda self, **_kwargs: None})()
    state = viewer._deformation_state
    state.enter()
    state.working_reference = scenario.reference.clone()
    state.working_hypothesis = scenario.hypothesis.clone()
    state.select(first_id, scenario.topoWorld)
    state.begin_drag("L")
    original_point = GeometryReferenceResolver(
        catalogue, scenario.reference
    ).get_city_lambert(_triangle.light_city_id)
    state.working_points["TMP-shared"] = WorkingPoint(
        "TMP-shared", original_point, {(first_id, "L"), (second_id, "O")},
    )
    candidate_point = (700000.0, 6600000.0)
    preview = viewer._apply_deformation_occurrence_overrides(
        state.candidate_occurrence_overrides(candidate_point)
    )
    assert preview is not None
    state.accept_occurrence_candidate(candidate_point, preview)
    city_ref_id = viewer._deformation_city_id_for_occurrence(first_id, "L")

    viewer._rename_working_deformation_city(city_ref_id, "Point 560")
    state.begin_drag("L")
    second_preview = viewer._apply_deformation_occurrence_overrides(
        state.candidate_occurrence_overrides((710000.0, 6610000.0))
    )
    assert second_preview is not None
    state.accept_occurrence_candidate((710000.0, 6610000.0), second_preview)

    assert len(state.working_reference.cities) == 1
    assert state.working_reference.cities[city_ref_id].name == "Point 560"
    assert state.last_accepted_world.elements[first_id].vertex_labels[2] == "Point 560"
    assert state.last_accepted_world.elements[second_id].vertex_labels[0] == "Point 560"


def test_successive_deformations_reuse_the_same_local_triangle_and_city():
    catalogue, scenario, _triangle, element_id = _scenario_with_catalogue_triangle()
    first_preview = _preview(catalogue, scenario, element_id, {(element_id, "L"): (700000.0, 6600000.0)})
    first = commit_deformation_copy_on_write(
        catalogue=catalogue, scenario=scenario, preview_world=first_preview,
        working_points=_working_points(
            ("TMP-1", (700000.0, 6600000.0), ((element_id, "L"),)),
        ),
    )
    scenario.reference, scenario.hypothesis, scenario.topoWorld = (
        first.reference, first.hypothesis, first.world,
    )
    local_triangle = next(iter(scenario.reference.triangles.values()))
    local_city_id = local_triangle.light_city_ref_id
    viewer = TriangleViewerManual.__new__(TriangleViewerManual)
    viewer.catalogue = catalogue
    viewer._get_active_scenario = lambda: scenario
    viewer._commit_deformation_city_rename(local_city_id, "Point 560")

    second_preview = _preview(catalogue, scenario, element_id, {(element_id, "L"): (710000.0, 6610000.0)})
    second = commit_deformation_copy_on_write(
        catalogue=catalogue, scenario=scenario, preview_world=second_preview,
        working_points=_working_points(
            ("TMP-2", (710000.0, 6610000.0), ((element_id, "L"),)),
        ),
    )

    assert len(second.reference.triangles) == 1
    assert len(second.reference.cities) == 1
    assert second.reference.triangles[local_triangle.triangle_ref_id].light_city_ref_id == local_city_id
    assert second.reference.cities[local_city_id].name == "Point 560"
    assert second.world.elements[element_id].source_triangle_id == local_triangle.triangle_ref_id
    assert second.hypothesis.triangle_ids_by_rank[24] == local_triangle.triangle_ref_id
    assert len(second.reference.triangles) == 1


def test_explicitly_shared_temporary_point_creates_one_city_for_two_occurrences():
    catalogue, scenario, _triangle, first_id = _scenario_with_catalogue_triangle()
    second_triangle_id = scenario.hypothesis.triangle_ids_by_rank[23]
    second = materialize_triangle(
        GeometryReferenceResolver(catalogue, scenario.reference), second_triangle_id,
    )
    scenario.topoWorld.add_element_as_new_group(second)
    second_id = second.element_id
    # The Core permits two instances here; COW must still only share when asked.
    preview = _preview(
        catalogue, scenario, first_id,
        {(first_id, "L"): (700000.0, 6600000.0), (second_id, "O"): (700000.0, 6600000.0)},
    )
    committed = commit_deformation_copy_on_write(
        catalogue=catalogue, scenario=scenario, preview_world=preview,
        working_points=_working_points(
            ("TMP-shared", (700000.0, 6600000.0), ((first_id, "L"), (second_id, "O"))),
        ),
    )

    assert len(committed.reference.cities) == 1
    triangles = list(committed.reference.triangles.values())
    shared_city_id = next(iter(committed.reference.cities))
    assert sum(
        shared_city_id in {
            triangle.opening_city_ref_id,
            triangle.base_city_ref_id,
            triangle.light_city_ref_id,
        }
        for triangle in triangles
    ) == 2
    assert committed.hypothesis.triangle_ids_by_rank[24].startswith("STRI-")
    assert committed.hypothesis.triangle_ids_by_rank[23].startswith("STRI-")


def test_renaming_a_shared_scenario_city_updates_all_materialized_elements():
    catalogue, scenario, _triangle, first_id = _scenario_with_catalogue_triangle()
    second_triangle_id = scenario.hypothesis.triangle_ids_by_rank[23]
    second = materialize_triangle(
        GeometryReferenceResolver(catalogue, scenario.reference), second_triangle_id,
    )
    scenario.topoWorld.add_element_as_new_group(second)
    second_id = second.element_id
    preview = _preview(
        catalogue, scenario, first_id,
        {(first_id, "L"): (700000.0, 6600000.0), (second_id, "O"): (700000.0, 6600000.0)},
    )
    committed = commit_deformation_copy_on_write(
        catalogue=catalogue, scenario=scenario, preview_world=preview,
        working_points=_working_points(
            ("TMP-shared", (700000.0, 6600000.0), ((first_id, "L"), (second_id, "O"))),
        ),
    )
    scenario.reference, scenario.hypothesis, scenario.topoWorld = (
        committed.reference, committed.hypothesis, committed.world,
    )
    shared_city_id = next(iter(scenario.reference.cities))
    viewer = TriangleViewerManual.__new__(TriangleViewerManual)
    viewer.catalogue = catalogue
    viewer._get_active_scenario = lambda: scenario

    viewer._commit_deformation_city_rename(shared_city_id, "Point 560")

    assert tuple(scenario.reference.cities) == (shared_city_id,)
    assert scenario.reference.cities[shared_city_id].name == "Point 560"
    for element in scenario.topoWorld.elements.values():
        shared_index = element.vertex_business_ids.index(shared_city_id)
        assert element.vertex_labels[shared_index] == "Point 560"


def test_invalid_scenario_city_rename_is_atomic():
    catalogue, scenario, _triangle, element_id = _scenario_with_catalogue_triangle()
    preview = _preview(catalogue, scenario, element_id, {(element_id, "L"): (700000.0, 6600000.0)})
    committed = commit_deformation_copy_on_write(
        catalogue=catalogue, scenario=scenario, preview_world=preview,
        working_points=_working_points(
            ("TMP-1", (700000.0, 6600000.0), ((element_id, "L"),)),
        ),
    )
    scenario.reference, scenario.hypothesis, scenario.topoWorld = (
        committed.reference, committed.hypothesis, committed.world,
    )
    city_ref_id = next(iter(scenario.reference.cities))
    before_reference = scenario.reference.clone()
    before_world = scenario.topoWorld._exportPhysicalSnapshot()
    viewer = TriangleViewerManual.__new__(TriangleViewerManual)
    viewer.catalogue = catalogue
    viewer._get_active_scenario = lambda: scenario

    with pytest.raises(ValueError, match="ne peut pas être vide"):
        viewer._commit_deformation_city_rename(city_ref_id, "   ")

    assert scenario.reference.cities == before_reference.cities
    assert scenario.topoWorld._exportPhysicalSnapshot() == before_world


def test_rename_callback_ignores_catalogue_cities_and_cancelled_dialogs(monkeypatch):
    catalogue, scenario, triangle, element_id = _scenario_with_catalogue_triangle()
    viewer = TriangleViewerManual.__new__(TriangleViewerManual)
    viewer.catalogue = catalogue
    viewer._get_active_scenario = lambda: scenario
    viewer.status = type("Status", (), {"config": lambda self, **_kwargs: None})()
    state = viewer._deformation_state
    state.enter()
    state.select(element_id, scenario.topoWorld)
    state.select_occurrence(element_id, "L")

    assert viewer._deformation_selected_occurrence_is_local_city() is False
    before_catalogue = catalogue.clone()
    viewer._deformation_rename_selected()
    assert catalogue.cities == before_catalogue.cities

    preview = _preview(catalogue, scenario, element_id, {(element_id, "L"): (700000.0, 6600000.0)})
    committed = commit_deformation_copy_on_write(
        catalogue=catalogue, scenario=scenario, preview_world=preview,
        working_points=_working_points(
            ("TMP-1", (700000.0, 6600000.0), ((element_id, "L"),)),
        ),
    )
    scenario.reference, scenario.hypothesis, scenario.topoWorld = (
        committed.reference, committed.hypothesis, committed.world,
    )
    state.rebase_after_commit(scenario.topoWorld, scenario.reference)
    state.select(element_id, scenario.topoWorld)
    state.select_occurrence(element_id, "L")
    before_reference = scenario.reference.clone()
    before_world = scenario.topoWorld._exportPhysicalSnapshot()
    monkeypatch.setattr(
        "src.assembleur_tk.simpledialog.askstring", lambda *_args, **_kwargs: None
    )

    assert viewer._deformation_selected_occurrence_is_local_city() is True
    viewer._deformation_rename_selected()

    assert scenario.reference.cities == before_reference.cities
    assert scenario.topoWorld._exportPhysicalSnapshot() == before_world


def test_rename_callback_commits_immediately_without_dirty_state(monkeypatch):
    catalogue, scenario, _triangle, element_id = _scenario_with_catalogue_triangle()
    viewer = TriangleViewerManual.__new__(TriangleViewerManual)
    viewer.catalogue = catalogue
    viewer._get_active_scenario = lambda: scenario
    viewer.status = type("Status", (), {"config": lambda self, **_kwargs: None})()
    state = viewer._deformation_state
    state.enter()
    state.select(element_id, scenario.topoWorld)
    state.select_occurrence(element_id, "L")
    preview = _preview(
        catalogue, scenario, element_id, {(element_id, "L"): (700000.0, 6600000.0)},
    )
    committed = commit_deformation_copy_on_write(
        catalogue=catalogue,
        scenario=scenario,
        preview_world=preview,
        working_points=_working_points(
            ("TMP-1", (700000.0, 6600000.0), ((element_id, "L"),)),
        ),
    )
    scenario.reference, scenario.hypothesis, scenario.topoWorld = (
        committed.reference, committed.hypothesis, committed.world,
    )
    state.rebase_after_commit(scenario.topoWorld, scenario.reference)
    state.select(element_id, scenario.topoWorld)
    state.select_occurrence(element_id, "L")
    refreshes = []
    viewer._restore_deformation_real_projection = lambda: refreshes.append("projection")
    viewer._refresh_deformation_window = lambda: refreshes.append("window")
    monkeypatch.setattr(
        "src.assembleur_tk.simpledialog.askstring", lambda *_args, **_kwargs: "Point 560"
    )

    viewer._deformation_rename_selected()

    city_ref_id = viewer._deformation_city_id_for_occurrence(element_id, "L")
    assert scenario.reference.cities[city_ref_id].name == "Point 560"
    assert state.dirty is False
    assert refreshes == ["projection", "window"]


def test_identical_temporary_coordinates_do_not_merge_without_shared_identity():
    catalogue, scenario, triangle, first_id = _scenario_with_catalogue_triangle()
    second_triangle_id = scenario.hypothesis.triangle_ids_by_rank[23]
    second = materialize_triangle(
        GeometryReferenceResolver(catalogue, scenario.reference), second_triangle_id,
    )
    scenario.topoWorld.add_element_as_new_group(second)
    second_id = second.element_id
    point = (700000.0, 6600000.0)
    preview = _preview(
        catalogue, scenario, first_id,
        {(first_id, "L"): point, (second_id, "O"): point},
    )

    committed = commit_deformation_copy_on_write(
        catalogue=catalogue,
        scenario=scenario,
        preview_world=preview,
        working_points=_working_points(
            ("TMP-1", point, ((first_id, "L"),)),
            ("TMP-2", point, ((second_id, "O"),)),
        ),
    )

    assert len(committed.reference.cities) == 2
    triangles = list(committed.reference.triangles.values())
    local_city_ids = set(committed.reference.cities)
    assert sum(
        city_id in local_city_ids
        for triangle in triangles
        for city_id in (
            triangle.opening_city_ref_id,
            triangle.base_city_ref_id,
            triangle.light_city_ref_id,
        )
    ) == 2


def test_rejected_preview_leaves_reference_and_world_untouched():
    catalogue, scenario, _triangle, element_id = _scenario_with_catalogue_triangle()
    before_reference = scenario.reference.clone()
    before_hypothesis = scenario.hypothesis.clone()
    before_snapshot = scenario.topoWorld._exportPhysicalSnapshot()

    with pytest.raises(ValueError, match="Point temporaire DEFORM invalide"):
        commit_deformation_copy_on_write(
            catalogue=catalogue, scenario=scenario, preview_world=scenario.topoWorld,
            working_points=_working_points(
                ("TMP-1", (float("nan"), 0.0), ((element_id, "L"),)),
            ),
        )

    assert scenario.reference.cities == before_reference.cities
    assert scenario.reference.triangles == before_reference.triangles
    assert scenario.hypothesis == before_hypothesis
    assert scenario.topoWorld._exportPhysicalSnapshot() == before_snapshot
