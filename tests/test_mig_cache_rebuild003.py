from types import SimpleNamespace

import numpy as np
import pytest

from src.assembleur_core import (
    ScenarioAssemblage,
    TopologyElement,
    TopologyNodeType,
    TopologyWorld,
)
from src.assembleur_projection import (
    buildLastDrawnFromTopology,
    getManualProjectionElementIds,
)
from src.assembleur_tk import TriangleViewerManual


def _element(element_id: str, rank: int) -> TopologyElement:
    return TopologyElement(
        element_id=element_id,
        name=f"Triangle {rank}",
        vertex_labels=["O", "B", "L"],
        vertex_types=[
            TopologyNodeType.OUVERTURE,
            TopologyNodeType.BASE,
            TopologyNodeType.LUMIERE,
        ],
        edge_lengths_km=[3.0, 5.0, 4.0],
        meta={"triRank": rank},
    )


def _world_with_three_elements():
    world = TopologyWorld()
    group_a = world.add_element_as_new_group(_element("T01", 1))
    group_b = world.add_element_as_new_group(_element("T02", 2))
    group_c = world.add_element_as_new_group(_element("T03", 3))
    world.union_groups(group_a, group_b)
    for index, element_id in enumerate(("T01", "T02", "T03")):
        world.setElementPose(
            element_id,
            np.eye(2),
            np.array((10.0 * index, -2.0 * index)),
            mirrored=False,
        )
    return world, group_c


def _assert_projection_matches_world(world, projection):
    assert {entry["topoElementId"] for entry in projection} == set(world.elements)
    for entry in projection:
        element = world.elements[entry["topoElementId"]]
        key_by_type = {
            TopologyNodeType.OUVERTURE: "O",
            TopologyNodeType.BASE: "B",
            TopologyNodeType.LUMIERE: "L",
        }
        for index, vertex_type in enumerate(element.vertex_types):
            np.testing.assert_allclose(
                entry["pts"][key_by_type[vertex_type]],
                element.localToWorld(element.vertex_local_xy[index]),
            )


def test_core_only_builder_ignores_any_former_projection_and_preserves_requested_order():
    world, _group_c = _world_with_three_elements()
    former_projection = [
        {"topoElementId": "T01", "pts": {"O": (999.0, 999.0)}},
    ]

    projection = buildLastDrawnFromTopology(
        topologyWorld=world,
        elementIds=["T03", "T01", "T02"],
    )

    assert [entry["topoElementId"] for entry in projection] == ["T03", "T01", "T02"]
    assert projection is not former_projection
    _assert_projection_matches_world(world, projection)
    assert not any(np.allclose(entry["pts"]["O"], (999.0, 999.0)) for entry in projection)


def test_manual_projection_ids_are_complete_and_contiguous_by_live_group():
    world, group_c = _world_with_three_elements()

    element_ids = getManualProjectionElementIds(world)

    assert set(element_ids) == {"T01", "T02", "T03"}
    assert element_ids[-1:] == world.getGroupElementIds(group_c)
    projection = buildLastDrawnFromTopology(topologyWorld=world, elementIds=element_ids)
    _assert_projection_matches_world(world, projection)


def test_core_transformations_and_successive_rebuilds_are_independent():
    world, _group_c = _world_with_three_elements()
    element_ids = ["T01", "T02", "T03"]
    first = buildLastDrawnFromTopology(topologyWorld=world, elementIds=element_ids)
    first[0]["pts"]["O"][:] = (1234.0, 5678.0)

    group_id = world.get_group_of_element("T01")
    world.move_group(group_id, 3.0, -4.0)
    world.rotate_group(group_id, np.array((0.0, 0.0)), np.pi / 2.0)
    world.flip_group(group_id, np.array((0.0, 0.0)), np.array((1.0, 0.0)))
    rebuilt = buildLastDrawnFromTopology(topologyWorld=world, elementIds=element_ids)

    _assert_projection_matches_world(world, rebuilt)
    assert not np.allclose(rebuilt[0]["pts"]["O"], (1234.0, 5678.0))


@pytest.mark.parametrize("element_ids, error", [
    (["T01", "", "T02"], "vide"),
    (["T01", "T01", "T02"], "dupliqué"),
    (["T01", "T02", "T99"], "absent"),
])
def test_core_only_builder_rejects_invalid_element_id_contract(element_ids, error):
    world, _group_c = _world_with_three_elements()
    with pytest.raises((KeyError, ValueError), match=error):
        buildLastDrawnFromTopology(topologyWorld=world, elementIds=element_ids)


def test_active_rebuild_discards_a_false_cache_and_replaces_its_objects():
    world, _group_c = _world_with_three_elements()
    scenario = ScenarioAssemblage("manuel", source_type="manual")
    scenario.topoWorld = world
    scenario.last_drawn = [
        {"topoElementId": "T404", "pts": {"O": (9, 9), "B": (9, 9), "L": (9, 9)}},
    ]
    old_projection = scenario.last_drawn
    viewer = TriangleViewerManual.__new__(TriangleViewerManual)
    viewer.scenarios = [scenario]
    viewer.active_scenario_index = 0

    viewer._rebuild_active_projection_from_core()

    assert scenario.last_drawn is viewer._last_drawn
    assert scenario.last_drawn is not old_projection
    _assert_projection_matches_world(world, scenario.last_drawn)
    assert viewer.canvas_objects.get_by_topology_id("T404") is None


def test_auto_scenario_rebuild_uses_ordered_element_ids_not_its_cache_order():
    world, _group_c = _world_with_three_elements()
    scenario = SimpleNamespace(
        name="auto",
        source_type="auto",
        topoWorld=world,
        orderedElementIds=["T02", "T03", "T01"],
        last_drawn=[{"topoElementId": "T01", "pts": {}}],
    )
    viewer = TriangleViewerManual.__new__(TriangleViewerManual)
    viewer.scenarios = [scenario]
    viewer.active_scenario_index = 0

    viewer._rebuild_active_projection_from_core()

    assert [entry["topoElementId"] for entry in viewer._last_drawn] == ["T02", "T03", "T01"]
    _assert_projection_matches_world(world, viewer._last_drawn)


def test_scenario_activation_rebuilds_the_new_active_projection_from_core():
    first_world, _first_group = _world_with_three_elements()
    second_world, _second_group = _world_with_three_elements()
    first = ScenarioAssemblage("premier", source_type="manual")
    second = ScenarioAssemblage("second", source_type="manual")
    first.topoWorld = first_world
    second.topoWorld = second_world
    first.last_drawn = []
    second.last_drawn = [{"topoElementId": "T404", "pts": {}}]
    stale_second_cache = second.last_drawn

    viewer = TriangleViewerManual.__new__(TriangleViewerManual)
    viewer.scenarios = [first, second]
    viewer.active_scenario_index = 0
    viewer._discard_manual_move_preview = lambda: False
    viewer._discard_manual_rotate_preview = lambda: False
    viewer._discard_auto_transform_preview = lambda: False
    viewer._attach_catalog_to_world = lambda _world: None
    viewer._capture_view_state = lambda: {}
    viewer._capture_map_state = lambda: {}
    viewer._apply_map_state = lambda *_args, **_kwargs: None
    viewer._apply_view_state = lambda *_args, **_kwargs: None
    viewer._rebuild_triangle_listbox_from_core = lambda: None
    viewer._invalidate_pick_cache = lambda: None
    viewer._redraw_from = lambda _entries: None
    viewer._redraw_overlay_only = lambda: None
    viewer.refreshCheminTreeView = lambda: None
    viewer._update_compass_ctx_menu_and_dico_state = lambda: None
    viewer.auto_fit_scenario_select = SimpleNamespace(get=lambda: False)
    viewer.status = SimpleNamespace(config=lambda **_kwargs: None)
    viewer.scenario_tree = None

    viewer._set_active_scenario(1)

    assert viewer.active_scenario_index == 1
    assert second.last_drawn is viewer._last_drawn
    assert second.last_drawn is not stale_second_cache
    _assert_projection_matches_world(second_world, second.last_drawn)
