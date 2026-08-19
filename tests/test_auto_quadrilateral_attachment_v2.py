from types import SimpleNamespace

import numpy as np
import pytest

from src.assembleur_catalogue import Catalogue
from src.assembleur_core import (
    ResolvedEdgeEdgeAttachment,
    TopologyAttachmentResolutionError,
    TopologyEdgeEdgeAttachment,
    TopologyElement,
    TopologyWorld,
    TopologyVertexEdgeAttachment,
)
from src.assembleur_projection import buildLastDrawnFromTopology, getCoreTriangleWorldPoints
from src.assembleur_scenario import ScenarioHypothesis
from src.assembleur_sim import AlgoQuadrisParPaires, MoteurSimulationAssemblage, createTopoQuadrilateral


def _points(*, opening, base, light):
    return {
        "O": np.asarray(opening, dtype=float),
        "B": np.asarray(base, dtype=float),
        "L": np.asarray(light, dtype=float),
    }


def _create_quadrilateral(*, second_local, second_world):
    first = _points(opening=(0.0, 0.0), base=(3.0, 0.0), light=(0.0, 4.0))
    world = TopologyWorld()

    def factory(triangle_id):
        points = first if triangle_id == "TRI-01" else second_local
        labels = (
            ["Opening", "Base", "Odd light"]
            if triangle_id == "TRI-01"
            else ["Opening", "Base", "Even light"]
        )
        return TopologyElement(
            name=triangle_id,
            vertex_labels=labels,
            vertex_types=["O", "B", "L"],
            edge_lengths_km=[3.0, 5.0, 4.0],
            vertex_local_xy={index: tuple(points[key]) for index, key in enumerate(("O", "B", "L"))},
            source_triangle_id=triangle_id,
        )

    result = createTopoQuadrilateral(
        world=world,
        triangleMobFromId="TRI-01",
        triangleMobToId="TRI-02",
        triangleMobFrom={},
        triangleMobTo={},
        triangleMobFrom_PtsLocal=first,
        triangleMobTo_PtsLocal=second_local,
        triangleMobFromPts=first,
        triangleMobToPts=second_world,
        element_factory=factory,
    )
    return world, first, result


def _catalogue_and_auto_hypothesis():
    catalogue = Catalogue()
    lambert = {}
    triangle_ids = []
    for pair_index in range(16):
        base = catalogue.add_city(f"Base {pair_index}", 45.0, 2.0)
        opening = catalogue.add_city(f"Opening {pair_index}", 45.0, 2.0)
        lambert[base.city_id] = (pair_index * 100.0 + 3.0, 0.0)
        lambert[opening.city_id] = (pair_index * 100.0, 0.0)
        for parity in range(2):
            rank = pair_index * 2 + parity + 1
            light = catalogue.add_city(f"Light {rank}", 45.0, 2.0)
            lambert[light.city_id] = (
                pair_index * 100.0 + (0.0 if parity == 0 else 3.0),
                4.0 if parity == 0 else -4.0,
            )
            triangle_ids.append(
                catalogue.add_triangle(
                    f"Note {rank}", opening.city_id, base.city_id, light.city_id
                ).triangle_id
            )
    catalogue.get_city_lambert = lambda city_id: lambert[city_id]
    return catalogue, ScenarioHypothesis(triangle_ids, "TPL-0001")


def _chain_pairs_by_hypothesis_rank(scenario, triangle_ids):
    rank_by_triangle_id = {
        triangle_id: rank
        for rank, triangle_id in enumerate(triangle_ids, start=1)
    }
    return {
        (
            rank_by_triangle_id[scenario.topoWorld.elements[attachment.dest_element_id].source_triangle_id],
            rank_by_triangle_id[scenario.topoWorld.elements[attachment.mob_element_id].source_triangle_id],
        )
        for attachment in scenario.topoWorld.attachments.values()
        if isinstance(attachment, TopologyVertexEdgeAttachment)
    }


def _internal_pair_edges_by_hypothesis_rank(scenario, triangle_ids):
    rank_by_triangle_id = {
        triangle_id: rank
        for rank, triangle_id in enumerate(triangle_ids, start=1)
    }
    return {
        (
            rank_by_triangle_id[scenario.topoWorld.elements[attachment.mob_element_id].source_triangle_id],
            rank_by_triangle_id[scenario.topoWorld.elements[attachment.dest_element_id].source_triangle_id],
        )
        for attachment in scenario.topoWorld.attachments.values()
        if isinstance(attachment, TopologyEdgeEdgeAttachment)
    }


def _ordered_source_triangle_ids(scenario):
    return [
        scenario.topoWorld.elements[element_id].source_triangle_id
        for element_id in scenario.orderedElementIds
    ]


def test_create_topo_quadrilateral_records_one_edge_edge_attachment_v2():
    second = _points(opening=(0.0, 0.0), base=(3.0, 0.0), light=(3.0, -4.0))
    world, first, (group_id, odd_id, even_id, src_edge, dst_edge) = _create_quadrilateral(
        second_local=second,
        second_world=second,
    )

    assert (odd_id, even_id, src_edge, dst_edge) == ("T01", "T02", "OB", "OB")
    assert set(world.elements) == {odd_id, even_id}
    assert len(world.attachments) == 1
    attachment = next(iter(world.attachments.values()))
    assert isinstance(attachment, TopologyEdgeEdgeAttachment)
    assert not hasattr(attachment, "params")
    assert world.get_group_of_element(odd_id) == group_id
    assert world.get_group_of_element(even_id) == group_id
    assert isinstance(world.getResolvedAttachment(attachment.attachment_id), ResolvedEdgeEdgeAttachment)
    for key, point in first.items():
        assert np.allclose(getCoreTriangleWorldPoints(world, odd_id)[key], point)


def test_quadrilateral_uses_resolver_reverse_mapping_without_sim_endpoint_mapping():
    first = _points(opening=(0.0, 0.0), base=(3.0, 0.0), light=(0.0, 4.0))
    second_local = _points(opening=(3.0, 0.0), base=(0.0, 0.0), light=(3.0, -4.0))
    world, _first, (_group_id, odd_id, even_id, _src_edge, _dst_edge) = _create_quadrilateral(
        second_local=second_local,
        second_world=second_local,
    )

    attachment = next(iter(world.attachments.values()))
    resolved = world.getResolvedAttachment(attachment.attachment_id)
    assert (resolved.mob_element_id, resolved.dest_element_id) == (odd_id, even_id)
    assert (resolved.mob_vertex_1, resolved.mob_vertex_2) == ("O", "B")
    assert (resolved.dest_vertex_1, resolved.dest_vertex_2) == ("B", "O")
    assert np.allclose(getCoreTriangleWorldPoints(world, odd_id)["O"], getCoreTriangleWorldPoints(world, even_id)["B"])
    assert np.allclose(getCoreTriangleWorldPoints(world, odd_id)["B"], getCoreTriangleWorldPoints(world, even_id)["O"])


def test_quadrilateral_projection_is_rebuilt_from_the_v2_world():
    second = _points(opening=(0.0, 0.0), base=(3.0, 0.0), light=(3.0, -4.0))
    world, _first, (_group_id, odd_id, even_id, _src_edge, _dst_edge) = _create_quadrilateral(
        second_local=second,
        second_world=second,
    )

    projection = buildLastDrawnFromTopology(topologyWorld=world, elementIds=[odd_id, even_id])

    assert [entry["topoElementId"] for entry in projection] == [odd_id, even_id]
    assert all(set(entry["pts"]) == {"O", "B", "L"} for entry in projection)


def test_quadrilateral_rejects_equal_lengths_without_a_common_business_edge():
    first = _points(opening=(0.0, 0.0), base=(3.0, 0.0), light=(0.0, 4.0))
    world = TopologyWorld()

    def factory(triangle_id):
        labels = ["A", "B", "C"] if triangle_id == "TRI-01" else ["D", "E", "F"]
        return TopologyElement(
            name=triangle_id,
            vertex_labels=labels,
            vertex_types=["O", "B", "L"],
            edge_lengths_km=[3.0, 5.0, 4.0],
            vertex_local_xy={index: tuple(first[key]) for index, key in enumerate(("O", "B", "L"))},
        )

    with pytest.raises(ValueError, match="aucune arête métier commune"):
        createTopoQuadrilateral(
            world=world,
            triangleMobFromId="TRI-01",
            triangleMobToId="TRI-02",
            triangleMobFrom={},
            triangleMobTo={},
            triangleMobFrom_PtsLocal=first,
            triangleMobTo_PtsLocal=first,
            triangleMobFromPts=first,
            triangleMobToPts=first,
            element_factory=factory,
        )


def test_quadrilateral_rejects_ambiguous_business_edges():
    first = _points(opening=(0.0, 0.0), base=(3.0, 0.0), light=(0.0, 4.0))
    world = TopologyWorld()

    def factory(triangle_id):
        return TopologyElement(
            name=triangle_id,
            vertex_labels=["A", "B", "C"],
            vertex_types=["O", "B", "L"],
            edge_lengths_km=[3.0, 5.0, 4.0],
            vertex_local_xy={index: tuple(first[key]) for index, key in enumerate(("O", "B", "L"))},
        )

    with pytest.raises(ValueError, match="ambiguës"):
        createTopoQuadrilateral(
            world=world,
            triangleMobFromId="TRI-01",
            triangleMobToId="TRI-02",
            triangleMobFrom={},
            triangleMobTo={},
            triangleMobFrom_PtsLocal=first,
            triangleMobTo_PtsLocal=first,
            triangleMobFromPts=first,
            triangleMobToPts=first,
            element_factory=factory,
        )


def test_auto_two_triangle_scenario_uses_a_single_v2_edge_edge_attachment():
    catalogue, hypothesis = _catalogue_and_auto_hypothesis()

    scenarios = AlgoQuadrisParPaires(
        MoteurSimulationAssemblage(SimpleNamespace(catalogue=catalogue), hypothesis)
    ).run(hypothesis.triangle_ids_by_rank[:2])

    assert len(scenarios) == 1
    scenario = scenarios[0]
    assert scenario.status == "complete"
    assert len(scenario.topoWorld.elements) == 2
    assert len(scenario.topoWorld.getLiveGroupIds()) == 1
    assert len(scenario.topoWorld.attachments) == 1
    assert isinstance(next(iter(scenario.topoWorld.attachments.values())), TopologyEdgeEdgeAttachment)
    assert len(scenario.last_drawn) == 2


def test_auto_four_triangles_uses_only_v2_attachments_and_explores_all_edges(monkeypatch):
    catalogue, hypothesis = _catalogue_and_auto_hypothesis()
    attempts = []
    original_simulate = TopologyWorld.simulate_topological_overlap

    def record_simulation(world, group_dest_id, group_mob_id, attachment):
        if isinstance(attachment, TopologyVertexEdgeAttachment):
            attempts.append((attachment.creation_mob_edge, attachment.creation_dest_edge))
        return original_simulate(
            world,
            group_dest_id,
            group_mob_id,
            attachment,
        )

    monkeypatch.setattr(TopologyWorld, "simulate_topological_overlap", record_simulation)
    scenarios = AlgoQuadrisParPaires(
        MoteurSimulationAssemblage(SimpleNamespace(catalogue=catalogue), hypothesis)
    ).run(hypothesis.triangle_ids_by_rank[:4])

    assert {("LO", "LO"), ("LO", "BL"), ("BL", "LO"), ("BL", "BL")} <= set(attempts)
    assert scenarios
    for scenario in scenarios:
        attachments = list(scenario.topoWorld.attachments.values())
        assert len(scenario.topoWorld.elements) == 4
        assert len(scenario.topoWorld.getLiveGroupIds()) == 1
        assert len(attachments) == 3
        assert sum(isinstance(attachment, TopologyEdgeEdgeAttachment) for attachment in attachments) == 2
        assert sum(isinstance(attachment, TopologyVertexEdgeAttachment) for attachment in attachments) == 1
        projection = buildLastDrawnFromTopology(
            topologyWorld=scenario.topoWorld,
            elementIds=scenario.orderedElementIds,
        )
        assert [entry["topoElementId"] for entry in scenario.last_drawn] == [
            entry["topoElementId"] for entry in projection
        ]
        for drawn, projected in zip(scenario.last_drawn, projection):
            for vertex in ("O", "B", "L"):
                assert np.allclose(drawn["pts"][vertex], projected["pts"][vertex])


def test_auto_six_triangles_keeps_the_attachment_count_invariant():
    catalogue, hypothesis = _catalogue_and_auto_hypothesis()

    scenarios = AlgoQuadrisParPaires(
        MoteurSimulationAssemblage(SimpleNamespace(catalogue=catalogue), hypothesis)
    ).run(hypothesis.triangle_ids_by_rank[:6])

    assert scenarios
    for scenario in scenarios:
        assert len(scenario.topoWorld.elements) == 6
        assert len(scenario.topoWorld.attachments) == 5
        assert all(
            isinstance(attachment, (TopologyEdgeEdgeAttachment, TopologyVertexEdgeAttachment))
            for attachment in scenario.topoWorld.attachments.values()
        )


def test_auto_four_triangles_chains_only_rank_two_to_rank_three():
    catalogue, hypothesis = _catalogue_and_auto_hypothesis()
    triangle_ids = hypothesis.triangle_ids_by_rank[:4]

    scenarios = AlgoQuadrisParPaires(
        MoteurSimulationAssemblage(SimpleNamespace(catalogue=catalogue), hypothesis)
    ).run(triangle_ids)

    assert scenarios
    for scenario in scenarios:
        assert _internal_pair_edges_by_hypothesis_rank(scenario, triangle_ids) == {
            (1, 2), (3, 4),
        }
        assert _chain_pairs_by_hypothesis_rank(scenario, triangle_ids) == {(2, 3)}
        assert _ordered_source_triangle_ids(scenario) == triangle_ids


def test_auto_six_triangles_preserves_chain_order_when_element_ids_are_permuted():
    catalogue, hypothesis = _catalogue_and_auto_hypothesis()
    triangle_ids = hypothesis.triangle_ids_by_rank[:6]
    engine = MoteurSimulationAssemblage(SimpleNamespace(catalogue=catalogue), hypothesis)
    original_materialize = engine.materialize_triangle
    element_id_by_rank = {
        1: "T06", 2: "T04", 3: "T03", 4: "T02", 5: "T01", 6: "T05",
    }

    def materialize_with_permuted_element_id(triangle_id):
        element = original_materialize(triangle_id)
        rank = triangle_ids.index(triangle_id) + 1
        element.element_id = element_id_by_rank[rank]
        return element

    engine.materialize_triangle = materialize_with_permuted_element_id
    scenarios = AlgoQuadrisParPaires(engine).run(triangle_ids)

    assert scenarios
    for scenario in scenarios:
        assert scenario.topoWorld.elements["T05"].source_triangle_id == triangle_ids[5]
        assert _chain_pairs_by_hypothesis_rank(scenario, triangle_ids) == {
            (2, 3), (4, 5),
        }
        assert _ordered_source_triangle_ids(scenario) == triangle_ids


def test_auto_eight_triangles_chains_successive_quadrilaterals_by_rank():
    catalogue, hypothesis = _catalogue_and_auto_hypothesis()
    triangle_ids = hypothesis.triangle_ids_by_rank[:8]

    scenarios = AlgoQuadrisParPaires(
        MoteurSimulationAssemblage(SimpleNamespace(catalogue=catalogue), hypothesis)
    ).run(triangle_ids)

    assert scenarios
    for scenario in scenarios:
        assert _chain_pairs_by_hypothesis_rank(scenario, triangle_ids) == {
            (2, 3), (4, 5), (6, 7),
        }
        assert _ordered_source_triangle_ids(scenario) == triangle_ids


def test_auto_rejects_a_branch_when_only_the_forbidden_even_mobile_could_continue(monkeypatch):
    catalogue, hypothesis = _catalogue_and_auto_hypothesis()
    triangle_ids = hypothesis.triangle_ids_by_rank[:4]
    attempted_mobile_sources = []
    simulated_mobile_sources = []
    original_apply_attachment = TopologyWorld.apply_attachment
    original_simulate = TopologyWorld.simulate_topological_overlap

    def reject_odd_chain_attachment(world, attachment):
        if isinstance(attachment, TopologyVertexEdgeAttachment):
            source_triangle_id = world.elements[attachment.mob_element_id].source_triangle_id
            attempted_mobile_sources.append(source_triangle_id)
            if source_triangle_id == triangle_ids[2]:
                raise TopologyAttachmentResolutionError("raccord impair forcé en échec")
        return original_apply_attachment(world, attachment)

    monkeypatch.setattr(TopologyWorld, "apply_attachment", reject_odd_chain_attachment)

    def record_simulation(world, group_dest_id, group_mob_id, attachment):
        if isinstance(attachment, TopologyVertexEdgeAttachment):
            simulated_mobile_sources.append(
                world.elements[attachment.mob_element_id].source_triangle_id
            )
        return original_simulate(
            world,
            group_dest_id,
            group_mob_id,
            attachment,
        )

    monkeypatch.setattr(TopologyWorld, "simulate_topological_overlap", record_simulation)
    engine = MoteurSimulationAssemblage(SimpleNamespace(catalogue=catalogue), hypothesis)
    scenarios = AlgoQuadrisParPaires(engine).run(triangle_ids)

    assert scenarios == []
    assert simulated_mobile_sources == [triangle_ids[2]] * 4
    assert attempted_mobile_sources
    assert set(attempted_mobile_sources) == {triangle_ids[2]}
    assert engine.debug_last["step"] == "chain_connect"
