import math
from types import SimpleNamespace

import numpy as np
import pytest

from src.assembleur_catalogue import Catalogue
from src.assembleur_core import (
    TopologyAttachment,
    TopologyFeatureRef,
    TopologyFeatureType,
    TopologyWorld,
)
from src.assembleur_deformation import simulate_triangle_deformation
from src.assembleur_edgechoice import materialize_vertex_edge_attachments
from src.assembleur_scenario import ScenarioHypothesis, materialize_catalogue_triangle
from src.assembleur_sim import AlgoQuadrisParPaires, MoteurSimulationAssemblage


class _BeaconResolver:
    def __init__(self, world_by_id):
        self._world_by_id = dict(world_by_id)

    def contains(self, beacon_id):
        return beacon_id in self._world_by_id

    def get_world(self, beacon_id):
        return self._world_by_id[beacon_id]


def _catalogue(second_light_latitude: float = 44.0) -> tuple[Catalogue, str, str]:
    catalogue = Catalogue()
    opening = catalogue.add_city("Ouverture", 45.0, 2.0)
    base = catalogue.add_city("Base", 45.0, 3.0)
    first_light = catalogue.add_city("Lumière nord", 46.0, 2.5)
    second_light = catalogue.add_city("Lumière sud", second_light_latitude, 2.5)
    first = catalogue.add_triangle(
        "Do", opening.city_id, base.city_id, first_light.city_id
    )
    second = catalogue.add_triangle(
        "Si", opening.city_id, base.city_id, second_light.city_id
    )
    return catalogue, first.triangle_id, second.triangle_id


def _attachment_signature(world):
    return [
        (
            attachment.attachment_id,
            attachment.kind,
            attachment.feature_a.feature_type,
            attachment.feature_a.element_id,
            attachment.feature_a.index,
            attachment.feature_b.feature_type,
            attachment.feature_b.element_id,
            attachment.feature_b.index,
            dict(attachment.params),
            attachment.source,
        )
        for attachment in sorted(world.attachments.values(), key=lambda item: item.attachment_id)
    ]


def _anchored_two_triangle_world():
    catalogue, first_triangle_id, second_triangle_id = _catalogue()
    world = TopologyWorld()
    first = materialize_catalogue_triangle(catalogue, first_triangle_id)
    second = materialize_catalogue_triangle(catalogue, second_triangle_id)
    world.add_element_as_new_group(first)
    world.add_element_as_new_group(second)
    world.apply_attachment(TopologyAttachment(
        attachment_id="A001",
        kind="edge-edge",
        feature_a=TopologyFeatureRef(TopologyFeatureType.EDGE, first.element_id, 0),
        feature_b=TopologyFeatureRef(TopologyFeatureType.EDGE, second.element_id, 0),
        params={"mapping": "direct"},
    ))
    anchor_node_id = world.get_element_vertex_node_id_by_type(first.element_id, "O")
    resolver = _BeaconResolver({"BEA-0001": (500.0, -200.0)})
    world.attachBeaconResolver(resolver)
    anchor = world.createGroupAnchor(
        world.get_group_of_element(first.element_id), "BEA-0001", anchor_node_id
    )
    world.applyGroupAnchor(anchor.anchor_id)
    return catalogue, world, first.element_id, second.element_id, anchor


def _anchored_atomic_vertex_edge_world(second_light_latitude: float):
    catalogue, first_triangle_id, second_triangle_id = _catalogue(
        second_light_latitude
    )
    world = TopologyWorld()
    first = materialize_catalogue_triangle(catalogue, first_triangle_id)
    second = materialize_catalogue_triangle(catalogue, second_triangle_id)
    world.add_element_as_new_group(first)
    world.add_element_as_new_group(second)
    world.apply_attachments(materialize_vertex_edge_attachments(
        world=world,
        element_id_src=first.element_id,
        src_edge="LO",
        src_anchor_vkey="O",
        element_id_dst=second.element_id,
        dst_edge="LO",
        dst_anchor_vkey="O",
    ))
    world.attachBeaconResolver(_BeaconResolver({"BEA-0001": (0.0, 0.0)}))
    anchor = world.createGroupAnchor(
        world.get_group_of_element(first.element_id),
        "BEA-0001",
        world.get_element_vertex_node_id_by_type(first.element_id, "O"),
    )
    world.applyGroupAnchor(anchor.anchor_id)
    return catalogue, world, first.element_id, second.element_id, second_triangle_id


def _candidate_lambert(catalogue, triangle_id, dx=0.0, dy=0.0):
    city_id = catalogue.get_triangle(triangle_id).light_city_id
    x, y = catalogue.get_city_lambert(city_id)
    return (x + dx, y + dy)


def _candidate_with_same_ol_length(catalogue, triangle_id, angle_rad):
    triangle = catalogue.get_triangle(triangle_id)
    opening = np.asarray(catalogue.get_city_lambert(triangle.opening_city_id))
    light = np.asarray(catalogue.get_city_lambert(triangle.light_city_id))
    cosine, sine = math.cos(angle_rad), math.sin(angle_rad)
    rotation = np.array([[cosine, -sine], [sine, cosine]])
    return tuple(opening + (rotation @ (light - opening)))


def _candidate_with_same_distance_from_opening(catalogue, triangle_id, role, angle_rad):
    triangle = catalogue.get_triangle(triangle_id)
    city_id_by_role = {
        "O": triangle.opening_city_id,
        "B": triangle.base_city_id,
        "L": triangle.light_city_id,
    }
    opening = np.asarray(catalogue.get_city_lambert(triangle.opening_city_id))
    point = np.asarray(catalogue.get_city_lambert(city_id_by_role[role]))
    cosine, sine = math.cos(angle_rad), math.sin(angle_rad)
    rotation = np.array([[cosine, -sine], [sine, cosine]])
    return tuple(opening + (rotation @ (point - opening)))


def _anchored_single_triangle_world():
    catalogue, triangle_id, _other_triangle_id = _catalogue()
    world = TopologyWorld()
    element = materialize_catalogue_triangle(catalogue, triangle_id)
    world.add_element_as_new_group(element)
    world.attachBeaconResolver(_BeaconResolver({"BEA-0001": (20.0, -30.0)}))
    anchor = world.createGroupAnchor(
        world.get_group_of_element(element.element_id),
        "BEA-0001",
        world.get_element_vertex_node_id_by_type(element.element_id, "O"),
    )
    world.applyGroupAnchor(anchor.anchor_id)
    return catalogue, world, element.element_id, triangle_id, anchor


def _attachment_distance(world, attachment):
    if attachment.kind == "vertex-vertex":
        first = world.elements[attachment.feature_a.element_id].vertex_local_xy[
            attachment.feature_a.index
        ]
        second = world.elements[attachment.feature_b.element_id].vertex_local_xy[
            attachment.feature_b.index
        ]
        return float(np.linalg.norm(
            world.elementLocalToWorld(attachment.feature_a.element_id, first)
            - world.elementLocalToWorld(attachment.feature_b.element_id, second)
        ))

    vertex = attachment.feature_a
    edge_ref = attachment.feature_b
    edge = world.get_edge(edge_ref.element_id, edge_ref.index)
    edge_t = float(attachment.params["t"])
    if attachment.params["edgeFrom"] == edge.v_end.node_id:
        edge_t = 1.0 - edge_t
    edge_point = world._localPointOnEdge(
        world.elements[edge_ref.element_id], edge, edge_t
    )
    vertex_point = world.elements[vertex.element_id].vertex_local_xy[vertex.index]
    return float(np.linalg.norm(
        world.elementLocalToWorld(vertex.element_id, vertex_point)
        - world.elementLocalToWorld(edge_ref.element_id, edge_point)
    ))


def test_deformation_is_pure_deterministic_and_preserves_attachments_and_anchor():
    catalogue, initial_world, _first_id, second_id, anchor = _anchored_two_triangle_world()
    initial_snapshot = initial_world._exportPhysicalSnapshot()
    catalogue_before = catalogue.clone()
    triangle_id = initial_world.elements[second_id].source_triangle_id
    candidate_one = _candidate_lambert(catalogue, triangle_id, dx=7_000.0)
    candidate_two = _candidate_lambert(catalogue, triangle_id, dy=-11_000.0)

    first = simulate_triangle_deformation(
        catalogue=catalogue,
        initial_world=initial_world,
        element_id=second_id,
        vertex_lambert_overrides={"L": candidate_one},
    )
    _second = simulate_triangle_deformation(
        catalogue=catalogue,
        initial_world=initial_world,
        element_id=second_id,
        vertex_lambert_overrides={"L": candidate_two},
    )
    repeated = simulate_triangle_deformation(
        catalogue=catalogue,
        initial_world=initial_world,
        element_id=second_id,
        vertex_lambert_overrides={"L": candidate_one},
    )

    assert first.accepted
    assert repeated.accepted
    assert initial_world._exportPhysicalSnapshot() == initial_snapshot
    assert catalogue.clone().__dict__ == catalogue_before.__dict__
    assert first.world._exportPhysicalSnapshot() == repeated.world._exportPhysicalSnapshot()
    assert _attachment_signature(first.world) == _attachment_signature(initial_world)
    restored_anchor = first.world.getGroupAnchor(anchor.anchor_id)
    assert restored_anchor.node_id == anchor.node_id
    assert restored_anchor.beacon_id == anchor.beacon_id
    assert restored_anchor.group_id == first.world.getGroupIdFromConceptNode(anchor.node_id)
    assert first.world.getConceptNodeWorldXY(
        anchor.node_id, restored_anchor.group_id
    ) == pytest.approx((500.0, -200.0))


def test_deformation_rejects_a_degenerate_candidate_without_mutating_source():
    catalogue, initial_world, _first_id, second_id, _anchor = _anchored_two_triangle_world()
    before = initial_world._exportPhysicalSnapshot()
    triangle = catalogue.get_triangle(initial_world.elements[second_id].source_triangle_id)
    candidate = catalogue.get_city_lambert(triangle.opening_city_id)

    result = simulate_triangle_deformation(
        catalogue=catalogue,
        initial_world=initial_world,
        element_id=second_id,
        vertex_lambert_overrides={"L": candidate},
    )

    assert not result.accepted
    assert result.world is None
    assert initial_world._exportPhysicalSnapshot() == before


@pytest.mark.parametrize(
    "overrides",
    [
        {},
        {"L": (650_000.0, 6_600_000.0)},
        {"B": (655_000.0, 6_600_000.0)},
        {"O": (645_000.0, 6_595_000.0)},
        {"B": (655_000.0, 6_600_000.0), "L": (650_000.0, 6_600_000.0)},
        {
            "O": (645_000.0, 6_595_000.0),
            "B": (655_000.0, 6_600_000.0),
            "L": (650_000.0, 6_610_000.0),
        },
    ],
)
def test_deformation_materializes_each_supported_override_combination(overrides):
    catalogue, initial_world, element_id, triangle_id, anchor = _anchored_single_triangle_world()
    initial_snapshot = initial_world._exportPhysicalSnapshot()
    catalogue_before = catalogue.clone()
    expected = materialize_catalogue_triangle(
        catalogue,
        triangle_id,
        vertex_lambert_overrides=overrides,
    )

    result = simulate_triangle_deformation(
        catalogue=catalogue,
        initial_world=initial_world,
        element_id=element_id,
        vertex_lambert_overrides=overrides,
    )

    assert result.accepted
    assert dict(result.vertex_lambert_overrides) == overrides
    temporary = result.world.elements[element_id]
    assert temporary.source_triangle_id == triangle_id
    assert temporary.vertex_labels == expected.vertex_labels
    for vertex_index, point in expected.vertex_local_xy.items():
        assert temporary.vertex_local_xy[vertex_index] == pytest.approx(point)
    restored_anchor = result.world.getGroupAnchor(anchor.anchor_id)
    assert result.world.getConceptNodeWorldXY(
        restored_anchor.node_id, restored_anchor.group_id
    ) == pytest.approx((20.0, -30.0))
    assert initial_world._exportPhysicalSnapshot() == initial_snapshot
    assert catalogue.clone().__dict__ == catalogue_before.__dict__


def test_deformation_multi_overrides_are_independent_of_drag_order():
    catalogue, initial_world, element_id, triangle_id, _anchor = _anchored_single_triangle_world()
    final_overrides = {
        "B": _candidate_with_same_distance_from_opening(catalogue, triangle_id, "B", 0.02),
        "L": _candidate_with_same_ol_length(catalogue, triangle_id, -0.01),
    }

    simulate_triangle_deformation(
        catalogue=catalogue,
        initial_world=initial_world,
        element_id=element_id,
        vertex_lambert_overrides={"L": final_overrides["L"]},
    )
    from_light_then_base = simulate_triangle_deformation(
        catalogue=catalogue,
        initial_world=initial_world,
        element_id=element_id,
        vertex_lambert_overrides=final_overrides,
    )
    simulate_triangle_deformation(
        catalogue=catalogue,
        initial_world=initial_world,
        element_id=element_id,
        vertex_lambert_overrides={"B": final_overrides["B"]},
    )
    from_base_then_light = simulate_triangle_deformation(
        catalogue=catalogue,
        initial_world=initial_world,
        element_id=element_id,
        vertex_lambert_overrides={"L": final_overrides["L"], "B": final_overrides["B"]},
    )

    assert from_light_then_base.accepted
    assert from_base_then_light.accepted
    assert (
        from_light_then_base.world._exportPhysicalSnapshot()
        == from_base_then_light.world._exportPhysicalSnapshot()
    )


def test_deformation_rejects_invalid_override_point_and_unknown_role():
    catalogue, world, element_id, _triangle_id, _anchor = _anchored_single_triangle_world()

    with pytest.raises(ValueError, match="Rôle d'override"):
        materialize_catalogue_triangle(
            catalogue,
            world.elements[element_id].source_triangle_id,
            vertex_lambert_overrides={"X": (1.0, 2.0)},
        )
    with pytest.raises(ValueError, match="Rôle d'override"):
        simulate_triangle_deformation(
            catalogue=catalogue,
            initial_world=world,
            element_id=element_id,
            vertex_lambert_overrides={"X": (1.0, 2.0)},
        )
    result = simulate_triangle_deformation(
        catalogue=catalogue,
        initial_world=world,
        element_id=element_id,
        vertex_lambert_overrides={"O": (float("nan"), 2.0)},
    )
    assert not result.accepted
    assert result.rejection_reason == "Override Lambert candidat invalide"


def test_deformation_keeps_attachments_frozen_with_base_and_light_overrides():
    catalogue, initial_world, _first_id, second_id, anchor = _anchored_two_triangle_world()
    triangle_id = initial_world.elements[second_id].source_triangle_id
    overrides = {
        "B": _candidate_with_same_distance_from_opening(catalogue, triangle_id, "B", 0.02),
        "L": _candidate_lambert(catalogue, triangle_id, dx=1_000.0),
    }
    topology_before = _attachment_signature(initial_world)

    result = simulate_triangle_deformation(
        catalogue=catalogue,
        initial_world=initial_world,
        element_id=second_id,
        vertex_lambert_overrides=overrides,
    )

    assert result.accepted
    assert _attachment_signature(result.world) == topology_before
    restored_anchor = result.world.getGroupAnchor(anchor.anchor_id)
    assert result.world.getConceptNodeWorldXY(
        restored_anchor.node_id, restored_anchor.group_id
    ) == pytest.approx((500.0, -200.0))


@pytest.mark.parametrize(
    ("second_light_latitude", "light_scale", "vertex_element"),
    [
        (44.0, 1.01, "source"),
        (45.5, 1.01, "destination"),
        (44.0, 0.99, "destination"),
    ],
)
def test_deformation_rematerializes_atomic_vertex_edge_with_current_edge_ratio(
    second_light_latitude,
    light_scale,
    vertex_element,
):
    catalogue, world, source_element_id, destination_element_id, triangle_id = (
        _anchored_atomic_vertex_edge_world(second_light_latitude)
    )
    old_vertex_edge = next(
        attachment for attachment in world.attachments.values()
        if attachment.kind == "vertex-edge"
    )
    opening_city_id = catalogue.get_triangle(triangle_id).opening_city_id
    light_city_id = catalogue.get_triangle(triangle_id).light_city_id
    opening = np.asarray(catalogue.get_city_lambert(opening_city_id))
    light = np.asarray(catalogue.get_city_lambert(light_city_id))

    result = simulate_triangle_deformation(
        catalogue=catalogue,
        initial_world=world,
        element_id=destination_element_id,
        vertex_lambert_overrides={"L": tuple(opening + light_scale * (light - opening))},
    )

    assert result.accepted
    rematerialized_vertex_edge = next(
        attachment for attachment in result.world.attachments.values()
        if attachment.kind == "vertex-edge"
    )
    expected_vertex_element_id = (
        source_element_id if vertex_element == "source" else destination_element_id
    )
    assert rematerialized_vertex_edge.feature_a.element_id == expected_vertex_element_id
    assert rematerialized_vertex_edge.params["t"] != pytest.approx(
        old_vertex_edge.params["t"]
    )
    assert rematerialized_vertex_edge.params["incident_edge_by_element"] == {
        source_element_id: "LO",
        destination_element_id: "LO",
    }
    for attachment in result.world.attachments.values():
        if attachment.kind in ("vertex-vertex", "vertex-edge"):
            assert _attachment_distance(result.world, attachment) <= 1e-8


def test_deformation_ignores_an_overlapping_independent_group():
    catalogue, initial_world, first_id, second_id, _anchor = _anchored_two_triangle_world()
    external = materialize_catalogue_triangle(
        catalogue, initial_world.elements[first_id].source_triangle_id
    )
    initial_world.add_element_as_new_group(external)
    first_rotation, first_translation, first_mirrored = initial_world.getElementPose(first_id)
    initial_world.setElementPose(
        external.element_id,
        first_rotation,
        first_translation,
        mirrored=first_mirrored,
    )
    external_before = initial_world.getElementPose(external.element_id)

    result = simulate_triangle_deformation(
        catalogue=catalogue,
        initial_world=initial_world,
        element_id=second_id,
        vertex_lambert_overrides={"L": _candidate_lambert(
            catalogue, initial_world.elements[second_id].source_triangle_id, dx=1_000.0
        )},
    )

    assert result.accepted
    external_after = result.world.getElementPose(external.element_id)
    assert external_after[0] == pytest.approx(external_before[0])
    assert external_after[1] == pytest.approx(external_before[1])
    assert external_after[2] is external_before[2]


def test_deformation_requires_one_resolvable_group_anchor():
    catalogue, world, _first_id, second_id, anchor = _anchored_two_triangle_world()
    world.removeGroupAnchor(anchor.anchor_id)

    with pytest.raises(ValueError, match="exactement une ancre"):
        simulate_triangle_deformation(
            catalogue=catalogue,
            initial_world=world,
            element_id=second_id,
            vertex_lambert_overrides={"L": _candidate_lambert(
                catalogue, world.elements[second_id].source_triangle_id
            )},
        )


@pytest.mark.parametrize(
    ("kind", "params"),
    [
        ("vertex-vertex", {}),
        ("vertex-edge", {"t": 0.5}),
    ],
)
def test_deformation_replays_frozen_point_attachments(kind, params):
    catalogue, first_triangle_id, second_triangle_id = _catalogue()
    world = TopologyWorld()
    first = materialize_catalogue_triangle(catalogue, first_triangle_id)
    second = materialize_catalogue_triangle(catalogue, second_triangle_id)
    world.add_element_as_new_group(first)
    world.add_element_as_new_group(second)
    if kind == "vertex-vertex":
        attachment = TopologyAttachment(
            "A001",
            kind,
            TopologyFeatureRef(TopologyFeatureType.VERTEX, second.element_id, 0),
            TopologyFeatureRef(TopologyFeatureType.VERTEX, first.element_id, 2),
            params,
        )
    else:
        attachment = TopologyAttachment(
            "A001",
            kind,
            TopologyFeatureRef(TopologyFeatureType.VERTEX, second.element_id, 0),
            TopologyFeatureRef(TopologyFeatureType.EDGE, first.element_id, 2),
            {**params, "edgeFrom": first.vertexes[2].node_id},
        )
    world.apply_attachment(attachment)
    world.attachBeaconResolver(_BeaconResolver({"BEA-0001": (0.0, 0.0)}))
    anchor = world.createGroupAnchor(
        world.get_group_of_element(first.element_id),
        "BEA-0001",
        world.get_element_vertex_node_id_by_type(first.element_id, "O"),
    )
    world.applyGroupAnchor(anchor.anchor_id)

    if kind == "vertex-edge":
        with pytest.raises(ValueError, match="incident_edge_by_element absent"):
            simulate_triangle_deformation(
                catalogue=catalogue,
                initial_world=world,
                element_id=second.element_id,
                vertex_lambert_overrides={"L": _candidate_lambert(
                    catalogue, second_triangle_id, dx=1_000.0
                )},
            )
        return

    result = simulate_triangle_deformation(
        catalogue=catalogue,
        initial_world=world,
        element_id=second.element_id,
        vertex_lambert_overrides={"L": _candidate_lambert(catalogue, second_triangle_id, dx=1_000.0)},
    )

    assert result.accepted
    replayed = next(iter(result.world.attachments.values()))
    assert replayed.kind == kind
    assert replayed.params == attachment.params


def test_deformation_reanchors_a_group_when_anchor_is_remote_from_pilot():
    catalogue, first_triangle_id, second_triangle_id = _catalogue()
    world = TopologyWorld()
    first = materialize_catalogue_triangle(catalogue, first_triangle_id)
    second = materialize_catalogue_triangle(catalogue, second_triangle_id)
    world.add_element_as_new_group(first)
    world.add_element_as_new_group(second)
    world.apply_attachment(TopologyAttachment(
        "A001",
        "vertex-vertex",
        TopologyFeatureRef(TopologyFeatureType.VERTEX, first.element_id, 2),
        TopologyFeatureRef(TopologyFeatureType.VERTEX, second.element_id, 0),
        {"incident_edge_by_element": {
            first.element_id: "LO",
            second.element_id: "LO",
        }},
    ))
    world.attachBeaconResolver(_BeaconResolver({"BEA-0001": (100.0, 200.0)}))
    anchor = world.createGroupAnchor(
        world.get_group_of_element(first.element_id),
        "BEA-0001",
        world.get_element_vertex_node_id_by_type(second.element_id, "O"),
    )
    world.applyGroupAnchor(anchor.anchor_id)
    positions_before = {
        element_id: world.getElementPose(element_id)[1]
        for element_id in world.elements
    }

    result = simulate_triangle_deformation(
        catalogue=catalogue,
        initial_world=world,
        element_id=first.element_id,
        vertex_lambert_overrides={"L": _candidate_lambert(catalogue, first_triangle_id, dx=1_000.0)},
    )

    assert result.accepted
    restored_anchor = result.world.getGroupAnchor(anchor.anchor_id)
    assert result.world.getConceptNodeWorldXY(
        restored_anchor.node_id,
        restored_anchor.group_id,
    ) == pytest.approx((100.0, 200.0))
    assert not np.allclose(
        result.world.getElementPose(first.element_id)[1], positions_before[first.element_id]
    )
    assert not np.allclose(
        result.world.getElementPose(second.element_id)[1], positions_before[second.element_id]
    )


def test_deformation_replays_vertex_vertex_and_vertex_edge_as_one_rigid_link():
    catalogue, first_triangle_id, second_triangle_id = _catalogue(43.0)
    world = TopologyWorld()
    first = materialize_catalogue_triangle(catalogue, first_triangle_id)
    second = materialize_catalogue_triangle(catalogue, second_triangle_id)
    world.add_element_as_new_group(first)
    world.add_element_as_new_group(second)
    opening = np.asarray(catalogue.get_city_lambert(
        catalogue.get_triangle(first_triangle_id).opening_city_id
    ))
    first_light = np.asarray(catalogue.get_city_lambert(
        catalogue.get_triangle(first_triangle_id).light_city_id
    ))
    second_light = np.asarray(catalogue.get_city_lambert(
        catalogue.get_triangle(second_triangle_id).light_city_id
    ))
    attachment_t = float(
        np.linalg.norm(first_light - opening) / np.linalg.norm(second_light - opening)
    )
    world.apply_attachment(TopologyAttachment(
        "A001",
        "vertex-vertex",
        TopologyFeatureRef(TopologyFeatureType.VERTEX, first.element_id, 2),
        TopologyFeatureRef(TopologyFeatureType.VERTEX, second.element_id, 0),
        {"incident_edge_by_element": {
            first.element_id: "LO",
            second.element_id: "LO",
        }},
    ))
    world.apply_attachment(TopologyAttachment(
        "A002",
        "vertex-edge",
        TopologyFeatureRef(TopologyFeatureType.VERTEX, first.element_id, 0),
        TopologyFeatureRef(TopologyFeatureType.EDGE, second.element_id, 2),
        {
            "t": attachment_t,
            "edgeFrom": second.vertexes[0].node_id,
            "incident_edge_by_element": {
                first.element_id: "LO",
                second.element_id: "LO",
            },
        },
    ))
    world.attachBeaconResolver(_BeaconResolver({"BEA-0001": (0.0, 0.0)}))
    anchor = world.createGroupAnchor(
        world.get_group_of_element(first.element_id),
        "BEA-0001",
        world.get_element_vertex_node_id_by_type(first.element_id, "O"),
    )
    world.applyGroupAnchor(anchor.anchor_id)
    rotation_before = world.getElementPose(second.element_id)[0]
    attachment_ids_before = sorted(world.attachments)

    result = simulate_triangle_deformation(
        catalogue=catalogue,
        initial_world=world,
        element_id=first.element_id,
        vertex_lambert_overrides={"L": _candidate_with_same_ol_length(
            catalogue, first_triangle_id, 0.03
        )},
    )

    assert result.accepted
    assert not np.allclose(result.world.getElementPose(second.element_id)[0], rotation_before)
    assert sorted(result.world.attachments) == attachment_ids_before
    replayed = sorted(result.world.attachments.values(), key=lambda item: item.attachment_id)
    assert replayed[1].params["t"] == pytest.approx(attachment_t)
    assert _attachment_distance(result.world, replayed[0]) <= 1e-8
    assert _attachment_distance(result.world, replayed[1]) <= 1e-8


def test_deformation_propagates_across_pairs_linked_by_an_atomic_point_link():
    catalogue = Catalogue()
    opening = catalogue.add_city("Ouverture chaîne", 45.0, 2.0)
    base = catalogue.add_city("Base chaîne", 45.0, 3.0)
    triangle_ids = []
    for name, latitude in (("Nord A", 46.0), ("Sud A", 44.5), ("Nord B", 46.0), ("Sud B", 44.5)):
        light = catalogue.add_city(name, latitude, 2.5)
        triangle_ids.append(catalogue.add_triangle(
            name, opening.city_id, base.city_id, light.city_id
        ).triangle_id)
    world = TopologyWorld()
    elements = []
    for triangle_id in triangle_ids:
        element = materialize_catalogue_triangle(catalogue, triangle_id)
        world.add_element_as_new_group(element)
        elements.append(element)
    for attachment_id, first_index, second_index in (("A001", 0, 1), ("A002", 2, 3)):
        world.apply_attachment(TopologyAttachment(
            attachment_id,
            "edge-edge",
            TopologyFeatureRef(TopologyFeatureType.EDGE, elements[first_index].element_id, 0),
            TopologyFeatureRef(TopologyFeatureType.EDGE, elements[second_index].element_id, 0),
            {"mapping": "direct"},
        ))
    opening_xy = np.asarray(catalogue.get_city_lambert(opening.city_id))
    light_a_xy = np.asarray(catalogue.get_city_lambert(
        catalogue.get_triangle(triangle_ids[1]).light_city_id
    ))
    light_b_xy = np.asarray(catalogue.get_city_lambert(
        catalogue.get_triangle(triangle_ids[2]).light_city_id
    ))
    attachment_t = float(
        np.linalg.norm(light_a_xy - opening_xy) / np.linalg.norm(light_b_xy - opening_xy)
    )
    world.apply_attachment(TopologyAttachment(
        "A003",
        "vertex-vertex",
        TopologyFeatureRef(TopologyFeatureType.VERTEX, elements[1].element_id, 2),
        TopologyFeatureRef(TopologyFeatureType.VERTEX, elements[2].element_id, 0),
        {"incident_edge_by_element": {
            elements[1].element_id: "LO",
            elements[2].element_id: "LO",
        }},
    ))
    world.apply_attachment(TopologyAttachment(
        "A004",
        "vertex-edge",
        TopologyFeatureRef(TopologyFeatureType.VERTEX, elements[1].element_id, 0),
        TopologyFeatureRef(TopologyFeatureType.EDGE, elements[2].element_id, 2),
        {
            "t": attachment_t,
            "edgeFrom": elements[2].vertexes[0].node_id,
            "incident_edge_by_element": {
                elements[1].element_id: "LO",
                elements[2].element_id: "LO",
            },
        },
    ))
    world.attachBeaconResolver(_BeaconResolver({"BEA-0001": (0.0, 0.0)}))
    anchor = world.createGroupAnchor(
        world.get_group_of_element(elements[0].element_id),
        "BEA-0001",
        world.get_element_vertex_node_id_by_type(elements[0].element_id, "O"),
    )
    world.applyGroupAnchor(anchor.anchor_id)
    attachment_ids_before = sorted(world.attachments)
    rotation_before = world.getElementPose(elements[1].element_id)[0]

    result = simulate_triangle_deformation(
        catalogue=catalogue,
        initial_world=world,
        element_id=elements[2].element_id,
        vertex_lambert_overrides={"L": _candidate_with_same_ol_length(
            catalogue, triangle_ids[2], 0.02
        )},
    )

    assert result.accepted
    assert sorted(result.world.attachments) == attachment_ids_before
    assert not np.allclose(
        result.world.getElementPose(elements[1].element_id)[0], rotation_before
    )
    restored_anchor = result.world.getGroupAnchor(anchor.anchor_id)
    assert result.world.getConceptNodeWorldXY(
        restored_anchor.node_id, restored_anchor.group_id
    ) == pytest.approx((0.0, 0.0))
    for attachment in result.world.attachments.values():
        if attachment.kind in ("vertex-vertex", "vertex-edge"):
            assert _attachment_distance(result.world, attachment) <= 1e-8


def test_deformation_keeps_reverse_edge_mapping_frozen():
    catalogue, _first_triangle_id, second_triangle_id = _catalogue(46.0)
    world = TopologyWorld()
    first = materialize_catalogue_triangle(catalogue, "TRI-0001")
    second = materialize_catalogue_triangle(catalogue, second_triangle_id)
    world.add_element_as_new_group(first)
    world.add_element_as_new_group(second)
    world.apply_attachment(TopologyAttachment(
        attachment_id="A001",
        kind="edge-edge",
        feature_a=TopologyFeatureRef(TopologyFeatureType.EDGE, first.element_id, 0),
        feature_b=TopologyFeatureRef(TopologyFeatureType.EDGE, second.element_id, 0),
        params={"mapping": "reverse"},
    ))
    world.attachBeaconResolver(_BeaconResolver({"BEA-0001": (0.0, 0.0)}))
    anchor = world.createGroupAnchor(
        world.get_group_of_element(first.element_id),
        "BEA-0001",
        world.get_element_vertex_node_id_by_type(first.element_id, "O"),
    )
    world.applyGroupAnchor(anchor.anchor_id)

    result = simulate_triangle_deformation(
        catalogue=catalogue,
        initial_world=world,
        element_id=second.element_id,
        vertex_lambert_overrides={"L": _candidate_lambert(catalogue, second_triangle_id, dx=3_000.0)},
    )

    assert result.accepted
    assert next(iter(result.world.attachments.values())).params["mapping"] == "reverse"


def test_deformation_accepts_a_world_produced_by_the_auto_simulator():
    catalogue, first_triangle_id, second_triangle_id = _catalogue()
    hypothesis = ScenarioHypothesis(
        [first_triangle_id, second_triangle_id] + [None] * 30,
        "TPL-0001",
    )
    engine = MoteurSimulationAssemblage(
        SimpleNamespace(catalogue=catalogue),
        source_hypothesis=hypothesis,
    )
    scenario = AlgoQuadrisParPaires(engine).run(
        [first_triangle_id, second_triangle_id]
    )[0]
    assert scenario.source_type == "auto"
    world = scenario.topoWorld
    element_ids = sorted(world.elements)
    world.attachBeaconResolver(_BeaconResolver({"BEA-0001": (0.0, 0.0)}))
    anchor = world.createGroupAnchor(
        world.get_group_of_element(element_ids[0]),
        "BEA-0001",
        world.get_element_vertex_node_id_by_type(element_ids[0], "O"),
    )
    world.applyGroupAnchor(anchor.anchor_id)

    result = simulate_triangle_deformation(
        catalogue=catalogue,
        initial_world=world,
        element_id=element_ids[1],
        vertex_lambert_overrides={"B": _candidate_with_same_distance_from_opening(
            catalogue, world.elements[element_ids[1]].source_triangle_id, "B", 0.01
        )},
    )

    assert result.accepted
