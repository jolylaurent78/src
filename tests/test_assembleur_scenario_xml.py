"""Persistance XML v5 de l'hypothÃ¨se propriÃ©taire d'un scÃ©nario."""

import xml.etree.ElementTree as ET

import numpy as np
import pytest

from src.assembleur_catalogue import Catalogue
from src.assembleur_core import ScenarioAssemblage, TopologyWorld
from src.assembleur_io import _load_scenario_hypothesis, loadScenarioXml, saveScenarioXml
from src.assembleur_scenario import ScenarioHypothesis
from src.assembleur_tk import TriangleViewerManual
from src.canvas_objects_collection import CanvasObjectsCollection


class _Listbox:
    def size(self):
        return 0

    def get(self, _index):
        raise IndexError

    def delete(self, *_args):
        pass

    def insert(self, *_args):
        pass


class _Canvas:
    def delete(self, *_args):
        pass

    def focus_set(self):
        pass


class _Viewer:
    _bind_canvas_objects = TriangleViewerManual._bind_canvas_objects
    _build_scenario_projection_from_core = TriangleViewerManual._build_scenario_projection_from_core
    _rebuild_active_projection_from_core = TriangleViewerManual._rebuild_active_projection_from_core
    _strip_core_duplicates_from_last_drawn_entry = staticmethod(
        TriangleViewerManual._strip_core_duplicates_from_last_drawn_entry
    )

    def __init__(self, catalogue, hypothesis=None):
        scenario = ScenarioAssemblage("XML hypothesis", hypothesis=hypothesis)
        scenario.topoWorld = TopologyWorld()
        scenario.topoScenarioId = "SCENARIO"
        self.scenarios = [scenario]
        self.active_scenario_index = 0
        self.catalogue = catalogue
        self.canvas_objects = CanvasObjectsCollection()
        self._last_drawn = self.canvas_objects.entries
        scenario.last_drawn = self._last_drawn
        self._bg = None
        self._clock_cx = self._clock_cy = 0.0
        self._clock_state = {"hour": 0, "minute": 0, "label": ""}
        self.listbox = _Listbox()
        self.canvas = _Canvas()
        self.zoom = 1.0
        self.offset = np.zeros(2)
        self._clock_ref_azimuth_deg = 0.0

    def _get_active_scenario(self):
        return self.scenarios[self.active_scenario_index]

    def _bg_clear(self, persist=False):
        pass

    def _clear_nearest_line(self):
        pass

    def _clear_edge_highlights(self):
        pass

    def _hide_tooltip(self):
        pass

    def _update_triangle_listbox_colors(self):
        pass

    def _bind_canvas_handlers(self):
        pass

    def _redraw_from(self, _entries):
        pass

    def _redraw_overlay_only(self):
        pass

    def _rebuild_pick_cache(self):
        pass


def _catalogue_with_valid_hypothesis():
    catalogue = Catalogue()
    triangle_ids = []
    for pair_index in range(16):
        base = catalogue.add_city(f"Base {pair_index}", 42.0 + pair_index / 100, 2.0)
        for item_in_pair in range(2):
            rank = pair_index * 2 + item_in_pair + 1
            opening = catalogue.add_city(f"Ouverture {rank}", 44.0 + rank / 100, 3.0)
            light = catalogue.add_city(f"LumiÃ¨re {rank}", 46.0 + rank / 100, 4.0)
            triangle = catalogue.add_triangle(
                f"Note {rank}", opening.city_id, base.city_id, light.city_id
            )
            triangle_ids.append(triangle.triangle_id)
    return catalogue, ScenarioHypothesis(triangle_ids, "TPL-0001")


def _hypothesis_xml(hypothesis, *, source_template_id="TPL-0001"):
    root = ET.Element("scenario")
    hypothesis_el = ET.SubElement(root, "hypothesis")
    if source_template_id is not None:
        hypothesis_el.set("sourceTemplateId", source_template_id)
    for rank, triangle_id in enumerate(hypothesis.triangle_ids_by_rank, start=1):
        ET.SubElement(hypothesis_el, "rank", number=str(rank), triangleId=triangle_id)
    return root


def test_round_trip_persists_an_independent_hypothesis_from_xml(tmp_path):
    catalogue, original = _catalogue_with_valid_hypothesis()
    template = catalogue.add_template("Template source")
    catalogue.set_template_ranks(template.template_id, original.triangle_ids_by_rank)
    path = tmp_path / "scenario.xml"
    saveScenarioXml(_Viewer(catalogue, original), str(path))

    root = ET.parse(path).getroot()
    hypothesis_el = root.find("hypothesis")
    assert hypothesis_el is not None
    assert hypothesis_el.get("sourceTemplateId") == "TPL-0001"
    assert [(rank.get("number"), rank.get("triangleId")) for rank in hypothesis_el.findall("rank")] == [
        (str(index), triangle_id)
        for index, triangle_id in enumerate(original.triangle_ids_by_rank, start=1)
    ]

    changed_template_ranks = list(original.triangle_ids_by_rank)
    changed_template_ranks[0], changed_template_ranks[1] = (
        changed_template_ranks[1],
        changed_template_ranks[0],
    )
    catalogue.set_template_ranks(template.template_id, changed_template_ranks)

    loaded_viewer = _Viewer(catalogue)
    loadScenarioXml(loaded_viewer, str(path))
    loaded = loaded_viewer._get_active_scenario().hypothesis
    assert loaded is not None
    assert loaded.source_template_id == original.source_template_id
    assert loaded.triangle_ids_by_rank == original.triangle_ids_by_rank
    assert loaded.triangle_ids_by_rank != template.triangle_ids_by_rank
    assert loaded is not original
    assert loaded.triangle_ids_by_rank is not original.triangle_ids_by_rank


def test_auto_scenario_persists_its_hypothesis(tmp_path):
    catalogue, hypothesis = _catalogue_with_valid_hypothesis()
    viewer = _Viewer(catalogue, hypothesis)
    viewer._get_active_scenario().source_type = "auto"
    path = tmp_path / "auto-scenario.xml"

    saveScenarioXml(viewer, str(path))

    persisted = _load_scenario_hypothesis(ET.parse(path).getroot(), catalogue)
    assert persisted is not None
    assert persisted.triangle_ids_by_rank == hypothesis.triangle_ids_by_rank
    assert persisted is not hypothesis


def test_save_rejects_a_scenario_without_hypothesis(tmp_path):
    catalogue, _original = _catalogue_with_valid_hypothesis()

    with pytest.raises(ValueError, match="ScenarioHypothesis absente"):
        saveScenarioXml(_Viewer(catalogue), str(tmp_path / "invalid.xml"))


def test_hypothesis_without_source_template_id_omits_the_attribute(tmp_path):
    catalogue, hypothesis = _catalogue_with_valid_hypothesis()
    hypothesis = ScenarioHypothesis(list(hypothesis.triangle_ids_by_rank), None)
    path = tmp_path / "without-source-template.xml"
    saveScenarioXml(_Viewer(catalogue, hypothesis), str(path))

    hypothesis_el = ET.parse(path).getroot().find("hypothesis")
    assert hypothesis_el is not None
    assert "sourceTemplateId" not in hypothesis_el.attrib


def test_obsolete_source_template_id_does_not_block_loading():
    catalogue, hypothesis = _catalogue_with_valid_hypothesis()
    loaded = _load_scenario_hypothesis(
        _hypothesis_xml(hypothesis, source_template_id="TPL-9999"), catalogue
    )
    assert loaded is not None
    assert loaded.source_template_id == "TPL-9999"
    assert loaded.triangle_ids_by_rank == hypothesis.triangle_ids_by_rank


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda ranks: ranks.pop(), "exactement 32"),
        (lambda ranks: ranks.append(ET.Element("rank", number="33", triangleId="TRI-9999")), "exactement 32"),
        (lambda ranks: ranks.__setitem__(1, ET.Element("rank", number="1", triangleId="TRI-0001")), "dupliqu"),
        (lambda ranks: ranks.__setitem__(0, ET.Element("rank", number="0", triangleId="TRI-0001")), "hors plage"),
        (lambda ranks: ranks.__setitem__(0, ET.Element("rank", number="33", triangleId="TRI-0001")), "hors plage"),
        (lambda ranks: ranks.__setitem__(0, ET.Element("rank", number="x", triangleId="TRI-0001")), "invalide"),
        (lambda ranks: ranks[0].attrib.pop("number"), "number"),
        (lambda ranks: ranks[0].attrib.pop("triangleId"), "triangleId"),
        (lambda ranks: ranks.__setitem__(0, ET.Element("rank", number="1", triangleId="   ")), "vide"),
    ],
)
def test_hypothesis_xml_rank_structure_is_strict(mutate, message):
    catalogue, hypothesis = _catalogue_with_valid_hypothesis()
    root = _hypothesis_xml(hypothesis)
    ranks = root.find("hypothesis").findall("rank")
    mutate(ranks)
    hypothesis_el = root.find("hypothesis")
    hypothesis_el[:] = ranks

    with pytest.raises(ValueError, match=message):
        _load_scenario_hypothesis(root, catalogue)


def test_hypothesis_validation_rejects_unknown_duplicate_and_mismatched_base():
    catalogue, hypothesis = _catalogue_with_valid_hypothesis()

    unknown = _hypothesis_xml(hypothesis)
    unknown.find("hypothesis").find("rank").set("triangleId", "TRI-9999")
    with pytest.raises(ValueError, match="triangle Catalogue absent"):
        _load_scenario_hypothesis(unknown, catalogue)

    duplicate = _hypothesis_xml(hypothesis)
    duplicate.find("hypothesis").findall("rank")[1].set(
        "triangleId", hypothesis.triangle_ids_by_rank[0]
    )
    with pytest.raises(ValueError, match="plusieurs rangs"):
        _load_scenario_hypothesis(duplicate, catalogue)

    different_base = _hypothesis_xml(hypothesis)
    mismatched_ranks = different_base.find("hypothesis").findall("rank")
    mismatched_ranks[1].set("triangleId", hypothesis.triangle_ids_by_rank[2])
    mismatched_ranks[2].set("triangleId", hypothesis.triangle_ids_by_rank[1])
    with pytest.raises(ValueError, match="base"):
        _load_scenario_hypothesis(different_base, catalogue)


def test_invalid_hypothesis_does_not_mutate_active_scenario_or_write_xml(tmp_path):
    catalogue, valid = _catalogue_with_valid_hypothesis()
    invalid = ScenarioHypothesis(valid.triangle_ids_by_rank[:-1], valid.source_template_id)
    target = _Viewer(catalogue, valid)
    original = target._get_active_scenario().hypothesis

    root = _hypothesis_xml(invalid)
    path = tmp_path / "invalid.xml"
    path.write_text(
        '<scenario version="5" topo_tx_orientation="cw"><topoSnapshot encoding="json">'
        "{}</topoSnapshot></scenario>",
        encoding="utf-8",
    )
    # Replace the minimal document's root with the malformed hypothesis while retaining v5 snapshot data.
    document = ET.parse(path)
    document.getroot().append(root.find("hypothesis"))
    document.write(path, encoding="utf-8", xml_declaration=True)

    with pytest.raises(ValueError, match="exactement 32"):
        loadScenarioXml(target, str(path))
    assert target._get_active_scenario().hypothesis is original

    output = tmp_path / "invalid-save.xml"
    target._get_active_scenario().hypothesis = invalid
    with pytest.raises(ValueError, match="exactement 32"):
        saveScenarioXml(target, str(output))
    assert not output.exists()
