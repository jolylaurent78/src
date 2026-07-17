from types import SimpleNamespace

import src.assembleur_tk as tk_module

from src.assembleur_core import (
    TopologyAttachment,
    TopologyElement,
    TopologyFeatureRef,
    TopologyFeatureType,
    TopologyNodeType,
    TopologyWorld,
)
from src.assembleur_edgechoice import buildEdgeChoiceEptsFromBest
from src.assembleur_tk import TriangleViewerManual


def _element(element_id: str) -> TopologyElement:
    return TopologyElement(
        element_id=element_id,
        name=element_id,
        vertex_labels=["O", "B", "L"],
        vertex_types=[
            TopologyNodeType.OUVERTURE,
            TopologyNodeType.BASE,
            TopologyNodeType.LUMIERE,
        ],
        edge_lengths_km=[3.0, 5.0, 4.0],
    )


def _vertex_edge_world_and_projection():
    world = TopologyWorld()
    world.add_element_as_new_group(_element("T02"))
    world.add_element_as_new_group(_element("T01"))
    world.apply_attachment(TopologyAttachment(
        "A-VE",
        "vertex-edge",
        TopologyFeatureRef(TopologyFeatureType.VERTEX, "T02", 1),
        TopologyFeatureRef(TopologyFeatureType.EDGE, "T01", 0),
        params={"t": 0.5, "edgeFrom": "T01:N0"},
    ))
    last_drawn = [
        {
            "topoElementId": "T02",
            "labels": ("O", "B", "L"),
            "pts": {"O": (0.0, 0.0), "B": (3.0, 0.0), "L": (0.0, 4.0)},
        },
        {
            "topoElementId": "T01",
            "labels": ("O", "B", "L"),
            "pts": {"O": (0.0, 0.0), "B": (3.0, 0.0), "L": (0.0, 4.0)},
        },
    ]
    return world, last_drawn


def _build(world, last_drawn, best, target_anchor):
    return buildEdgeChoiceEptsFromBest(
        best,
        world=world,
        mob_idx=0,
        tgt_idx=1,
        mob_tids=[0],
        tgt_tids=[1],
        last_drawn=last_drawn,
        eps_world=1e-6,
        mATmpId="T02:N0",
        tATmpId=target_anchor,
    )


def test_boundary_owner_without_physical_anchor_is_rejected_without_exception():
    world, last_drawn = _vertex_edge_world_and_projection()

    result = _build(
        world,
        last_drawn,
        (0.0, ((0.0, 0.0), (3.0, 0.0)), ((1.5, 0.0), (3.0, 0.0))),
        "T02:N1",
    )

    assert result is None


def test_representable_boundary_owner_still_builds_edge_choice():
    world, last_drawn = _vertex_edge_world_and_projection()

    result = _build(
        world,
        last_drawn,
        (0.0, ((0.0, 0.0), (3.0, 0.0)), ((0.0, 0.0), (3.0, 0.0))),
        "T01:N0",
    )

    assert result is not None
    epts, _meta = result
    assert epts.elementIdSrc == "T02"
    assert epts.elementIdDst == "T01"


def test_snap_loop_continues_after_unrepresentable_boundary_candidate(monkeypatch):
    class _World:
        def format_node_id(self, element_id, vertex_index):
            return f"{element_id}:N{vertex_index}"

        def getBoundarySegments(self, group_id):
            return [f"outline:{group_id}"]

        def getIncidentBoundarySegments(self, group_id, _node_id):
            return [f"incident:{group_id}"]

        def simulateOverlapTopologique(self, *_args, **_kwargs):
            return False

    class _Epts:
        def createTopologyAttachments(self, **_kwargs):
            return [object()]

    world = _World()
    viewer = TriangleViewerManual.__new__(TriangleViewerManual)
    viewer._last_drawn = [
        {
            "group_id": 1,
            "topoElementId": "T02",
            "pts": {"O": (0.0, 0.0), "B": (3.0, 0.0), "L": (0.0, 4.0)},
        },
        {
            "group_id": 2,
            "topoElementId": "T01",
            "pts": {"O": (10.0, 0.0), "B": (13.0, 0.0), "L": (10.0, 4.0)},
        },
    ]
    viewer.scenarios = [SimpleNamespace(topoWorld=world)]
    viewer.active_scenario_index = 0
    viewer._edge_choice = None
    viewer._edge_highlights = None
    viewer._clear_edge_highlights = lambda: None
    viewer._redraw_edge_highlights = lambda: None
    viewer._get_core_group_id_for_triangle_index = lambda index: "GM" if index == 0 else "GT"
    viewer._get_projected_elements_for_core_group = lambda group_id: (
        (viewer._last_drawn[0],) if group_id == "GM" else (viewer._last_drawn[1],)
    )
    viewer._get_projected_element_tids = lambda entries: [
        0 if entry is viewer._last_drawn[0] else 1 for entry in entries
    ]
    viewer._project_boundary_segments = lambda _group_id, segments: (
        [((0.0, 0.0), (3.0, 0.0)), ((0.0, 0.0), (0.0, 4.0))]
        if segments == ["incident:GM"]
        else [((10.0, 0.0), (13.0, 0.0))]
    )

    calls = []

    def _build_or_reject(*_args, **_kwargs):
        calls.append(True)
        return None if len(calls) == 1 else (_Epts(), {})

    monkeypatch.setattr(tk_module, "buildEdgeChoiceEptsFromBest", _build_or_reject)

    viewer._update_edge_highlights(0, "O", 1, "O")

    assert len(calls) >= 3  # rejeté, candidat suivant évalué, puis construction finale
    assert viewer._edge_choice is not None
