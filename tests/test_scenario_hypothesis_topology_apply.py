import numpy as np
import pytest

from src.assembleur_catalogue import Catalogue
from src.assembleur_core import (
    ResolvedEdgeEdgeAttachment,
    ScenarioAssemblage,
    TopologyEdgeEdgeAttachment,
    TopologyGroupAnchor,
    TopologyVertexEdgeAttachment,
)
from src.assembleur_scenario import (
    ScenarioHypothesis,
    apply_hypothesis_change_to_manual_scenario,
    materialize_catalogue_triangle,
)


def _catalogue_with_hypothesis():
    catalogue = Catalogue()
    ranks = []
    for index in range(16):
        base = catalogue.add_city(f"Base {index}", 40.0 + index, 2.0)
        opening = catalogue.add_city(f"O {index}", 40.0 + index, 3.0)
        for parity in range(2):
            light = catalogue.add_city(f"L {index}-{parity}", 40.0 + index, 4.0 + parity)
            ranks.append(
                catalogue.add_triangle(
                    f"N {index}-{parity}", opening.city_id, base.city_id, light.city_id
                ).triangle_id
            )

    old = catalogue.get_triangle(ranks[0])
    replay_light = catalogue.add_city("Replay light", 55.0, 6.0)
    replay = catalogue.add_triangle(
        "Replay", old.opening_city_id, old.base_city_id, replay_light.city_id
    )
    return catalogue, ScenarioHypothesis(ranks, "TPL-A"), replay.triangle_id


def _replay_triangle_for_rank(catalogue, hypothesis, rank, name):
    old = catalogue.get_triangle(hypothesis.triangle_ids_by_rank[rank - 1])
    light = catalogue.add_city(f"{name} light", 60.0 + rank, 7.0)
    return catalogue.add_triangle(name, old.opening_city_id, old.base_city_id, light.city_id)


class _BeaconResolver:
    def __init__(self, beacon_id, world_xy):
        self._beacon_id = beacon_id
        self._world_xy = world_xy

    def contains(self, beacon_id):
        return beacon_id == self._beacon_id

    def get_world(self, beacon_id):
        if not self.contains(beacon_id):
            raise KeyError(beacon_id)
        return self._world_xy


def _attach_ob_edge_edge(world, first_element_id, second_element_id):
    world.apply_attachment(
        TopologyEdgeEdgeAttachment(
            attachment_id=world.new_attachment_id(),
            mob_element_id=first_element_id,
            mob_edge="OB",
            dest_element_id=second_element_id,
            dest_edge="OB",
        )
    )


def _attach_lo_vertex_edge(world, first_element_id, second_element_id):
    world.apply_attachment(
        TopologyVertexEdgeAttachment(
            attachment_id=world.new_attachment_id(),
            mob_element_id=first_element_id,
            mob_vertex="L",
            creation_mob_edge="LO",
            dest_element_id=second_element_id,
            dest_vertex="O",
            creation_dest_edge="LO",
            mob_orientation="CCW",
            dest_orientation="CW",
        )
    )


def _assert_resolved_edge_edge_is_coincident(world, attachment):
    resolved = world.getResolvedAttachment(attachment.attachment_id)
    assert isinstance(resolved, ResolvedEdgeEdgeAttachment)
    for mob_vertex, dest_vertex in (
        (resolved.mob_vertex_1, resolved.dest_vertex_1),
        (resolved.mob_vertex_2, resolved.dest_vertex_2),
    ):
        mob_local = world.elements[resolved.mob_element_id].vertex_local_xy[
            world.elements[resolved.mob_element_id].vertex_types.index(mob_vertex)
        ]
        dest_local = world.elements[resolved.dest_element_id].vertex_local_xy[
            world.elements[resolved.dest_element_id].vertex_types.index(dest_vertex)
        ]
        assert world.elementLocalToWorld(
            resolved.mob_element_id, mob_local
        ) == pytest.approx(
            world.elementLocalToWorld(resolved.dest_element_id, dest_local)
        )


def test_apply_replaces_materialized_triangle_on_a_clone_and_replays_ob_connection():
    catalogue, old_hypothesis, replay_id = _catalogue_with_hypothesis()
    scenario = ScenarioAssemblage("Manuel", hypothesis=old_hypothesis)
    first = materialize_catalogue_triangle(catalogue, old_hypothesis.triangle_ids_by_rank[0])
    second = materialize_catalogue_triangle(catalogue, old_hypothesis.triangle_ids_by_rank[1])
    scenario.topoWorld.add_element_as_new_group(first)
    scenario.topoWorld.add_element_as_new_group(second)
    scenario.topoWorld.setElementPose(first.element_id, np.eye(2), np.array([10.0, 20.0]))
    scenario.topoWorld.setElementPose(second.element_id, np.eye(2), np.array([20.0, 30.0]))
    _attach_ob_edge_edge(scenario.topoWorld, second.element_id, first.element_id)
    source_snapshot = scenario.topoWorld._exportPhysicalSnapshot()
    source_world = scenario.topoWorld
    source_hypothesis = scenario.hypothesis

    draft = old_hypothesis.clone()
    draft.triangle_ids_by_rank[0] = replay_id
    result = apply_hypothesis_change_to_manual_scenario(catalogue, scenario, draft)

    assert scenario.hypothesis.triangle_ids_by_rank == draft.triangle_ids_by_rank
    assert scenario.topoWorld._exportPhysicalSnapshot() != source_snapshot
    assert source_world._exportPhysicalSnapshot() == source_snapshot
    assert source_hypothesis.triangle_ids_by_rank[0] != replay_id
    assert result.replayed_attachment_count == 1
    assert len(scenario.topoWorld.attachments) == 1
    attachment = next(iter(scenario.topoWorld.attachments.values()))
    assert isinstance(attachment, TopologyEdgeEdgeAttachment)
    assert attachment.mob_edge == attachment.dest_edge == "OB"
    new_element = scenario.topoWorld.elements[attachment.mob_element_id]
    assert new_element.source_triangle_id == replay_id
    _assert_resolved_edge_edge_is_coincident(scenario.topoWorld, attachment)


def test_apply_replay_without_ob_does_not_reconnect_a_vertex_edge_link():
    catalogue, old_hypothesis, replay_id = _catalogue_with_hypothesis()
    scenario = ScenarioAssemblage("Manuel", hypothesis=old_hypothesis)
    first = materialize_catalogue_triangle(catalogue, old_hypothesis.triangle_ids_by_rank[0])
    second = materialize_catalogue_triangle(catalogue, old_hypothesis.triangle_ids_by_rank[1])
    scenario.topoWorld.add_element_as_new_group(first)
    scenario.topoWorld.add_element_as_new_group(second)
    _attach_lo_vertex_edge(scenario.topoWorld, first.element_id, second.element_id)
    draft = old_hypothesis.clone()
    draft.triangle_ids_by_rank[0] = replay_id

    result = apply_hypothesis_change_to_manual_scenario(catalogue, scenario, draft)

    assert result.plan.global_impact.name == "REPLAY"
    assert result.replayed_attachment_count == 0
    assert not scenario.topoWorld.attachments


def test_apply_detach_does_not_recreate_an_existing_ob_link():
    catalogue, old_hypothesis, _replay_id = _catalogue_with_hypothesis()
    scenario = ScenarioAssemblage("Manuel", hypothesis=old_hypothesis)
    first = materialize_catalogue_triangle(catalogue, old_hypothesis.triangle_ids_by_rank[0])
    second = materialize_catalogue_triangle(catalogue, old_hypothesis.triangle_ids_by_rank[1])
    scenario.topoWorld.add_element_as_new_group(first)
    scenario.topoWorld.add_element_as_new_group(second)
    _attach_ob_edge_edge(scenario.topoWorld, first.element_id, second.element_id)
    old = catalogue.get_triangle(old_hypothesis.triangle_ids_by_rank[0])
    new_opening = catalogue.add_city("Opening detach", 58.0, 1.0)
    detached = catalogue.add_triangle(
        "Detach", new_opening.city_id, old.base_city_id, old.light_city_id
    )
    draft = old_hypothesis.clone()
    draft.triangle_ids_by_rank[0] = detached.triangle_id

    result = apply_hypothesis_change_to_manual_scenario(catalogue, scenario, draft)

    assert result.plan.global_impact.name == "DETACH"
    assert result.replayed_attachment_count == 0
    assert not scenario.topoWorld.attachments


def test_apply_replays_a_changed_pair_only_once():
    catalogue, old_hypothesis, replay_first = _catalogue_with_hypothesis()
    old_second = catalogue.get_triangle(old_hypothesis.triangle_ids_by_rank[1])
    second_light = catalogue.add_city("Replay second light", 56.0, 6.0)
    replay_second = catalogue.add_triangle(
        "Replay second", old_second.opening_city_id, old_second.base_city_id, second_light.city_id
    )
    scenario = ScenarioAssemblage("Manuel", hypothesis=old_hypothesis)
    first = materialize_catalogue_triangle(catalogue, old_hypothesis.triangle_ids_by_rank[0])
    second = materialize_catalogue_triangle(catalogue, old_hypothesis.triangle_ids_by_rank[1])
    scenario.topoWorld.add_element_as_new_group(first)
    scenario.topoWorld.add_element_as_new_group(second)
    _attach_ob_edge_edge(scenario.topoWorld, first.element_id, second.element_id)
    draft = old_hypothesis.clone()
    draft.triangle_ids_by_rank[0] = replay_first
    draft.triangle_ids_by_rank[1] = replay_second.triangle_id

    result = apply_hypothesis_change_to_manual_scenario(catalogue, scenario, draft)

    assert result.replayed_attachment_count == 1
    assert len(scenario.topoWorld.attachments) == 1
    assert {element.source_triangle_id for element in scenario.topoWorld.elements.values()} == {
        replay_first,
        replay_second.triangle_id,
    }


def test_apply_replays_independent_neighbour_connections_and_preserves_anchor():
    catalogue, old_hypothesis, _unused_replay = _catalogue_with_hypothesis()
    replay_second = _replay_triangle_for_rank(catalogue, old_hypothesis, 2, "Replay rank 2")
    replay_third = _replay_triangle_for_rank(catalogue, old_hypothesis, 3, "Replay rank 3")
    scenario = ScenarioAssemblage("Manuel", hypothesis=old_hypothesis)
    beacon_id = catalogue.add_beacon(catalogue.get_triangle(old_hypothesis.triangle_ids_by_rank[0]).opening_city_id).beacon_id
    scenario.topoWorld.attachBeaconResolver(_BeaconResolver(beacon_id, (500.0, -200.0)))
    elements = []
    for rank in range(1, 5):
        element = materialize_catalogue_triangle(catalogue, old_hypothesis.triangle_ids_by_rank[rank - 1])
        scenario.topoWorld.add_element_as_new_group(element)
        elements.append(element)
    _attach_ob_edge_edge(scenario.topoWorld, elements[0].element_id, elements[1].element_id)
    _attach_ob_edge_edge(scenario.topoWorld, elements[2].element_id, elements[3].element_id)
    anchor = scenario.topoWorld.createGroupAnchor(
        scenario.topoWorld.get_group_of_element(elements[0].element_id),
        beacon_id,
        scenario.topoWorld.get_element_vertex_node_id_by_type(elements[0].element_id, "L"),
    )

    draft = old_hypothesis.clone()
    draft.triangle_ids_by_rank[1] = replay_second.triangle_id
    draft.triangle_ids_by_rank[2] = replay_third.triangle_id
    result = apply_hypothesis_change_to_manual_scenario(catalogue, scenario, draft)

    assert result.replayed_attachment_count == 2
    assert len(scenario.topoWorld.attachments) == 2
    restored_anchor = scenario.topoWorld.getGroupAnchor(anchor.anchor_id)
    assert restored_anchor.node_id == anchor.node_id
    assert restored_anchor.beacon_id == beacon_id
    assert restored_anchor.group_id == scenario.topoWorld.getGroupIdFromConceptNode(anchor.node_id)
    assert scenario.topoWorld.getAnchorForGroup(restored_anchor.group_id) is restored_anchor
    assert scenario.topoWorld.getConceptNodeWorldXY(
        restored_anchor.node_id, restored_anchor.group_id
    ) == pytest.approx((500.0, -200.0))

    for attachment in scenario.topoWorld.attachments.values():
        assert isinstance(attachment, TopologyEdgeEdgeAttachment)
        _assert_resolved_edge_edge_is_coincident(scenario.topoWorld, attachment)


def test_apply_rejects_a_direct_anchor_without_mutating_the_scenario():
    catalogue, old_hypothesis, replay_id = _catalogue_with_hypothesis()
    scenario = ScenarioAssemblage("Manuel", hypothesis=old_hypothesis)
    element = materialize_catalogue_triangle(catalogue, old_hypothesis.triangle_ids_by_rank[0])
    group_id = scenario.topoWorld.add_element_as_new_group(element)
    scenario.topoWorld.groupAnchors["AN001"] = TopologyGroupAnchor(
        anchor_id="AN001", group_id=group_id, beacon_id="B001", node_id=f"{element.element_id}:N0"
    )
    source_world = scenario.topoWorld
    source_hypothesis = scenario.hypothesis
    draft = old_hypothesis.clone()
    draft.triangle_ids_by_rank[0] = replay_id

    with pytest.raises(ValueError, match="porte l'ancre"):
        apply_hypothesis_change_to_manual_scenario(catalogue, scenario, draft)

    assert scenario.topoWorld is source_world
    assert scenario.hypothesis is source_hypothesis
