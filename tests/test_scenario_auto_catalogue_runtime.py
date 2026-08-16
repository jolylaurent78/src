from types import SimpleNamespace

import numpy as np
import pytest

from src.assembleur_catalogue import Catalogue
from src.assembleur_core import ScenarioAssemblage
from src.assembleur_scenario import ScenarioHypothesis, materialize_catalogue_triangle
from src.assembleur_sim import AlgoQuadrisParPaires, MoteurSimulationAssemblage
from src.assembleur_tk import TriangleViewerManual


def _catalogue_and_hypothesis():
    catalogue = Catalogue()
    triangle_ids = []
    lambert = {}
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


def _engine(catalogue, hypothesis):
    return MoteurSimulationAssemblage(
        SimpleNamespace(catalogue=catalogue), source_hypothesis=hypothesis
    )


def test_auto_build_local_triangle_uses_catalogue_factory_without_excel_dataframe():
    catalogue, hypothesis = _catalogue_and_hypothesis()
    engine = _engine(catalogue, hypothesis)
    triangle_id = hypothesis.triangle_ids_by_rank[0]

    local = engine.build_local_triangle(triangle_id)
    element = materialize_catalogue_triangle(catalogue, triangle_id)

    assert local["triangle_id"] == triangle_id
    assert local["labels"] == tuple(element.vertex_labels)
    assert np.allclose(local["pts"]["O"], element.vertex_local_xy[0])
    assert np.allclose(local["pts"]["B"], element.vertex_local_xy[1])
    assert np.allclose(local["pts"]["L"], element.vertex_local_xy[2])


def test_auto_simulation_uses_catalogue_ids_and_clones_the_source_hypothesis():
    catalogue, hypothesis = _catalogue_and_hypothesis()
    first = AlgoQuadrisParPaires(_engine(catalogue, hypothesis)).run(
        hypothesis.triangle_ids_by_rank[:2]
    )[0]
    second = AlgoQuadrisParPaires(_engine(catalogue, hypothesis)).run(
        hypothesis.triangle_ids_by_rank[:2]
    )[0]

    assert first.hypothesis is not hypothesis
    assert second.hypothesis is not hypothesis
    assert first.hypothesis is not second.hypothesis
    first.hypothesis.triangle_ids_by_rank[0] = hypothesis.triangle_ids_by_rank[1]
    assert hypothesis.triangle_ids_by_rank[0] != first.hypothesis.triangle_ids_by_rank[0]
    assert second.hypothesis.triangle_ids_by_rank[0] == hypothesis.triangle_ids_by_rank[0]
    for scenario in (first, second):
        assert len(scenario.topoWorld.elements) == 2
        assert {
            element.source_triangle_id for element in scenario.topoWorld.elements.values()
        } == set(hypothesis.triangle_ids_by_rank[:2])
        assert all("triRank" not in element.meta for element in scenario.topoWorld.elements.values())


def test_simulation_order_is_derived_from_the_hypothesis_not_the_listbox():
    _catalogue, source_hypothesis = _catalogue_and_hypothesis()
    triangle_ids = list(source_hypothesis.triangle_ids_by_rank)
    triangle_ids[:4] = [triangle_ids[9], triangle_ids[3], triangle_ids[21], triangle_ids[0]]
    hypothesis = ScenarioHypothesis(triangle_ids, source_hypothesis.source_template_id)
    scenario = ScenarioAssemblage("Manual", hypothesis=hypothesis)
    viewer = SimpleNamespace(_get_active_scenario=lambda: scenario)
    viewer._simulation_get_triangle_ids_first_n = lambda n: (
        TriangleViewerManual._simulation_get_triangle_ids_first_n(viewer, n)
    )

    assert TriangleViewerManual._simulation_get_triangle_ids_by_order(viewer, 4) == (
        hypothesis.triangle_ids_by_rank[:4]
    )
    assert TriangleViewerManual._simulation_get_triangle_ids_by_order(viewer, 2, "reverse") == [
        hypothesis.triangle_ids_by_rank[31],
        hypothesis.triangle_ids_by_rank[30],
    ]


def test_modern_simulation_rejects_a_scenario_without_hypothesis():
    scenario = ScenarioAssemblage("Invalid modern scenario")
    viewer = SimpleNamespace(_get_active_scenario=lambda: scenario)

    with pytest.raises(ValueError, match="ScenarioHypothesis"):
        TriangleViewerManual._simulation_get_triangle_ids_first_n(viewer, 2)


def test_branch_label_rank_comes_from_the_hypothesis_order():
    catalogue, hypothesis = _catalogue_and_hypothesis()
    triangle_id = hypothesis.triangle_ids_by_rank[16]

    assert _engine(catalogue, hypothesis).get_hypothesis_rank(triangle_id) == 17


def test_auto_snapshot_manual_clones_its_hypothesis_independently():
    catalogue, hypothesis = _catalogue_and_hypothesis()

    class Listbox:
        def __init__(self):
            self.items = []

        def delete(self, _start, _end):
            self.items.clear()

        def insert(self, _position, value):
            self.items.append(value)

    auto = ScenarioAssemblage(
        "Auto", source_type="auto", hypothesis=hypothesis
    )
    viewer = SimpleNamespace(
        scenarios=[auto],
        active_scenario_index=0,
        _get_active_scenario=lambda: auto,
        _capture_view_state=lambda: {"zoom": 1.0},
        _capture_map_state=lambda: {"path": "map"},
        catalogue=catalogue,
        listbox=Listbox(),
        _update_triangle_listbox_colors=lambda: None,
    )

    manual_index = TriangleViewerManual._convertActiveAutoToManualSnapshot(viewer)
    manual = viewer.scenarios[manual_index]

    assert manual.source_type == "manual"
    assert manual.hypothesis is not auto.hypothesis
    assert manual.hypothesis.triangle_ids_by_rank == auto.hypothesis.triangle_ids_by_rank
    viewer.active_scenario_index = manual_index
    viewer._get_active_scenario = lambda: manual
    TriangleViewerManual._rebuild_triangle_listbox_from_core(viewer)
    assert len(viewer.listbox.items) == 32
    manual.hypothesis.triangle_ids_by_rank[0] = auto.hypothesis.triangle_ids_by_rank[1]
    assert manual.hypothesis.triangle_ids_by_rank[0] != auto.hypothesis.triangle_ids_by_rank[0]


def test_auto_snapshot_rejects_missing_hypothesis():
    auto = ScenarioAssemblage("Invalid auto", source_type="auto")
    viewer = SimpleNamespace(
        scenarios=[auto],
        active_scenario_index=0,
        _get_active_scenario=lambda: auto,
    )

    with pytest.raises(ValueError, match="AUTO sans ScenarioHypothesis"):
        TriangleViewerManual._convertActiveAutoToManualSnapshot(viewer)
