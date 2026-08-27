"""Persistance XML v5 de l'hypothÃ¨se propriÃ©taire d'un scÃ©nario."""

import json
import xml.etree.ElementTree as ET

import numpy as np
import pytest

import src.assembleur_io as assembleur_io_module
from src.assembleur_catalogue import Catalogue
from src.assembleur_core import ScenarioAssemblage, TopologyEdgeEdgeAttachment, TopologyWorld
from src.assembleur_geometry_reference import (
    GeometryReferenceResolver,
    ScenarioCity,
    ScenarioTriangle,
)
from src.assembleur_io import _load_scenario_hypothesis, loadScenarioXml, saveScenarioXml
from src.assembleur_scenario import ScenarioHypothesis, materialize_triangle
from src.assembleur_tk import TriangleViewerManual
from src.canvas_objects_collection import CanvasObjectsCollection


class _Listbox:
    def __init__(self):
        self.entries = []
        self.colours = {}

    def size(self):
        return len(self.entries)

    def get(self, _index):
        return self.entries[_index]

    def delete(self, *_args):
        self.entries.clear()

    def insert(self, _index, value):
        self.entries.append(value)

    def itemconfig(self, index, **kwargs):
        self.colours[index] = kwargs

    def yview(self):
        return (0.0, 1.0)

    def yview_moveto(self, _position):
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
    _reapply_scenario_group_anchors = TriangleViewerManual._reapply_scenario_group_anchors
    _rebuild_triangle_listbox_from_core = TriangleViewerManual._rebuild_triangle_listbox_from_core
    _update_triangle_listbox_colors = TriangleViewerManual._update_triangle_listbox_colors
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

    def _attach_beacon_resolver_to_world(self, world):
        world.attachBeaconResolver(None)

    def _bg_clear(self, persist=False):
        pass

    def _clear_nearest_line(self):
        pass

    def _clear_edge_highlights(self):
        pass

    def _hide_tooltip(self):
        pass

    def _update_triangle_listbox_colors(self):
        TriangleViewerManual._update_triangle_listbox_colors(self)

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


def _add_local_triangle(viewer, rank, *, city_ref_id, triangle_ref_id, city_name):
    scenario = viewer._get_active_scenario()
    source = viewer.catalogue.get_triangle(scenario.hypothesis.triangle_ids_by_rank[rank - 1])
    if city_ref_id not in scenario.reference.cities:
        scenario.reference.add_city(ScenarioCity(
            city_ref_id, city_name, 47.123456789, 2.987654321, source.light_city_id,
        ))
    scenario.reference.add_triangle(ScenarioTriangle(
        triangle_ref_id,
        source.note,
        source.opening_city_id,
        source.base_city_id,
        city_ref_id,
        source.triangle_id,
    ))
    scenario.hypothesis.triangle_ids_by_rank[rank - 1] = triangle_ref_id
    resolver = GeometryReferenceResolver(viewer.catalogue, scenario.reference)
    scenario.hypothesis.validate(resolver)
    element = materialize_triangle(resolver, triangle_ref_id)
    scenario.topoWorld.add_element_as_new_group(element)
    return element


def _add_catalogue_triangle(viewer, rank):
    scenario = viewer._get_active_scenario()
    triangle_id = scenario.hypothesis.triangle_ids_by_rank[rank - 1]
    element = materialize_triangle(
        GeometryReferenceResolver(viewer.catalogue, scenario.reference), triangle_id
    )
    scenario.topoWorld.add_element_as_new_group(element)
    return element


def _as_legacy_v5(path, *, clear_business_ids=False):
    tree = ET.parse(path)
    root = tree.getroot()
    root.set("version", "5")
    root.remove(root.find("scenarioReference"))
    if clear_business_ids:
        snapshot = root.find("topoSnapshot")
        payload = json.loads(snapshot.text)
        for element in payload["elements"]:
            element["vertex_business_ids"] = [None] * len(element["vertex_labels"])
        snapshot.text = json.dumps(payload)
    tree.write(path, encoding="utf-8", xml_declaration=True)


def _without_business_ids(snapshot):
    structural_snapshot = json.loads(json.dumps(snapshot))
    for element in structural_snapshot["elements"]:
        element.pop("vertex_business_ids", None)
    return structural_snapshot


def test_v6_tri_only_round_trip_has_an_explicit_empty_reference(tmp_path):
    catalogue, hypothesis = _catalogue_with_valid_hypothesis()
    path = tmp_path / "tri-only.xml"
    saveScenarioXml(_Viewer(catalogue, hypothesis), str(path))

    root = ET.parse(path).getroot()
    assert root.get("version") == "6"
    assert root.find("scenarioReference/cities") is not None
    assert root.find("scenarioReference/triangles") is not None
    assert root.findall("scenarioReference/cities/city") == []
    assert root.findall("scenarioReference/triangles/triangle") == []

    loaded_viewer = _Viewer(catalogue)
    loadScenarioXml(loaded_viewer, str(path))
    loaded = loaded_viewer._get_active_scenario()
    assert loaded.reference.cities == {}
    assert loaded.reference.triangles == {}
    assert loaded.hypothesis.triangle_ids_by_rank == hypothesis.triangle_ids_by_rank


def test_v6_round_trip_preserves_stri_scity_rename_and_business_ids(tmp_path):
    catalogue, hypothesis = _catalogue_with_valid_hypothesis()
    source_viewer = _Viewer(catalogue, hypothesis)
    element = _add_local_triangle(
        source_viewer, 25, city_ref_id="SCITY-0001", triangle_ref_id="STRI-0001", city_name="Tmp",
    )
    path = tmp_path / "stri.xml"
    saveScenarioXml(source_viewer, str(path))

    loaded_viewer = _Viewer(catalogue)
    loadScenarioXml(loaded_viewer, str(path))
    loaded = loaded_viewer._get_active_scenario()
    city = loaded.reference.cities["SCITY-0001"]
    triangle = loaded.reference.triangles["STRI-0001"]
    restored = loaded.topoWorld.elements[element.element_id]

    assert city.name == "Tmp"
    assert (city.latitude, city.longitude) == pytest.approx((47.123456789, 2.987654321))
    assert city.catalogue_source_city_id is not None
    assert triangle.catalogue_source_triangle_id is not None
    assert loaded.hypothesis.triangle_ids_by_rank[24] == "STRI-0001"
    assert restored.source_triangle_id == "STRI-0001"
    assert restored.vertex_business_ids[2] == "SCITY-0001"
    assert restored.vertex_labels[2] == "Tmp"


def test_v6_load_rebuilds_listbox_from_effective_stri_and_marks_it_used(tmp_path):
    catalogue, hypothesis = _catalogue_with_valid_hypothesis()
    source_viewer = _Viewer(catalogue, hypothesis)
    _add_local_triangle(
        source_viewer, 25, city_ref_id="SCITY-0001", triangle_ref_id="STRI-0001",
        city_name="Tmp Lumière",
    )
    path = tmp_path / "effective-list.xml"
    saveScenarioXml(source_viewer, str(path))

    loaded_viewer = _Viewer(catalogue, hypothesis)
    loadScenarioXml(loaded_viewer, str(path))

    assert loaded_viewer._triangle_list_triangle_ids[24] == "STRI-0001"
    assert loaded_viewer.listbox.entries[24] == "25. B:Base 12  L:Tmp Lumière"
    assert loaded_viewer.listbox.colours[24]["fg"] == "gray50"

    element_id = next(iter(loaded_viewer._get_active_scenario().topoWorld.elements))
    loaded_viewer._get_active_scenario().topoWorld.removeElementsAndRebuild([element_id])
    loaded_viewer._rebuild_triangle_listbox_from_core()
    assert loaded_viewer._triangle_list_triangle_ids[24] == "STRI-0001"
    assert loaded_viewer.listbox.colours[24]["fg"] == "black"


def test_v6_round_trip_preserves_shared_and_orphan_reference_ids(tmp_path):
    catalogue, hypothesis = _catalogue_with_valid_hypothesis()
    source_viewer = _Viewer(catalogue, hypothesis)
    _add_local_triangle(
        source_viewer, 25, city_ref_id="SCITY-0001", triangle_ref_id="STRI-0002", city_name="Shared",
    )
    _add_local_triangle(
        source_viewer, 27, city_ref_id="SCITY-0001", triangle_ref_id="STRI-0009", city_name="Shared",
    )
    scenario = source_viewer._get_active_scenario()
    scenario.reference.add_city(ScenarioCity("SCITY-0007", "Orpheline", 45.0, 2.0, None))
    source = catalogue.get_triangle(hypothesis.triangle_ids_by_rank[0])
    scenario.reference.add_triangle(ScenarioTriangle(
        "STRI-0008", source.note, source.opening_city_id, source.base_city_id,
        "SCITY-0007", source.triangle_id,
    ))
    path = tmp_path / "shared.xml"
    saveScenarioXml(source_viewer, str(path))

    loaded_viewer = _Viewer(catalogue)
    loadScenarioXml(loaded_viewer, str(path))
    reference = loaded_viewer._get_active_scenario().reference

    assert set(reference.cities) == {"SCITY-0001", "SCITY-0007"}
    assert reference.triangles["STRI-0002"].light_city_ref_id == "SCITY-0001"
    assert reference.triangles["STRI-0009"].light_city_ref_id == "SCITY-0001"
    assert "STRI-0008" in reference.triangles
    assert reference.next_city_ref_id() == "SCITY-0008"
    assert reference.next_triangle_ref_id() == "STRI-0010"


def test_v6_round_trip_preserves_an_attachment_on_a_stri(tmp_path):
    catalogue, hypothesis = _catalogue_with_valid_hypothesis()
    source_viewer = _Viewer(catalogue, hypothesis)
    first = _add_local_triangle(
        source_viewer, 25, city_ref_id="SCITY-0001", triangle_ref_id="STRI-0001", city_name="Tmp 25",
    )
    second = _add_local_triangle(
        source_viewer, 26, city_ref_id="SCITY-0002", triangle_ref_id="STRI-0002", city_name="Tmp 26",
    )
    world = source_viewer._get_active_scenario().topoWorld
    world.apply_attachment(TopologyEdgeEdgeAttachment("A001", first.element_id, "OB", second.element_id, "OB"))
    path = tmp_path / "attachment.xml"
    saveScenarioXml(source_viewer, str(path))

    loaded_viewer = _Viewer(catalogue)
    loadScenarioXml(loaded_viewer, str(path))
    restored = loaded_viewer._get_active_scenario().topoWorld
    attachment = restored.attachments["A001"]

    assert attachment.mob_element_id == first.element_id
    assert attachment.dest_element_id == second.element_id
    assert restored.elements[first.element_id].source_triangle_id == "STRI-0001"
    assert restored.validate_world() == []


def test_v5_tri_only_loads_with_an_empty_reference_and_rejects_stri(tmp_path):
    catalogue, hypothesis = _catalogue_with_valid_hypothesis()
    v6_path = tmp_path / "source.xml"
    saveScenarioXml(_Viewer(catalogue, hypothesis), str(v6_path))
    tree = ET.parse(v6_path)
    root = tree.getroot()
    root.set("version", "5")
    root.remove(root.find("scenarioReference"))
    v5_path = tmp_path / "legacy-v5.xml"
    tree.write(v5_path, encoding="utf-8", xml_declaration=True)

    loaded_viewer = _Viewer(catalogue)
    loadScenarioXml(loaded_viewer, str(v5_path))
    assert loaded_viewer._get_active_scenario().reference.cities == {}
    assert loaded_viewer._triangle_list_triangle_ids == hypothesis.triangle_ids_by_rank
    assert loaded_viewer.listbox.entries[0].startswith("01. B:Base 0")

    invalid_tree = ET.parse(v5_path)
    invalid_tree.getroot().find("hypothesis/rank").set("triangleId", "STRI-0001")
    invalid_path = tmp_path / "legacy-stri.xml"
    invalid_tree.write(invalid_path, encoding="utf-8", xml_declaration=True)
    original = loaded_viewer._get_active_scenario().topoWorld
    with pytest.raises(ValueError, match="référence effective|STRI"):
        loadScenarioXml(loaded_viewer, str(invalid_path))
    assert loaded_viewer._get_active_scenario().topoWorld is original


def test_v6_cross_reference_errors_do_not_mutate_the_active_scenario(tmp_path):
    catalogue, hypothesis = _catalogue_with_valid_hypothesis()
    source_viewer = _Viewer(catalogue, hypothesis)
    _add_local_triangle(
        source_viewer, 25, city_ref_id="SCITY-0001", triangle_ref_id="STRI-0001", city_name="Tmp",
    )
    valid_path = tmp_path / "valid.xml"
    saveScenarioXml(source_viewer, str(valid_path))
    tree = ET.parse(valid_path)
    snapshot = tree.getroot().find("topoSnapshot")
    payload = json.loads(snapshot.text)
    payload["elements"][0]["vertex_business_ids"][2] = "SCITY-9999"
    snapshot.text = json.dumps(payload)
    invalid_path = tmp_path / "invalid.xml"
    tree.write(invalid_path, encoding="utf-8", xml_declaration=True)

    target = _Viewer(catalogue, hypothesis)
    before_world = target._get_active_scenario().topoWorld
    before_reference = target._get_active_scenario().reference
    before_clock = dict(target._clock_state)
    with pytest.raises(ValueError, match="vertex_business_ids|SCITY"):
        loadScenarioXml(target, str(invalid_path))
    assert target._get_active_scenario().topoWorld is before_world
    assert target._get_active_scenario().reference is before_reference
    assert target._clock_state == before_clock


def test_v5_legacy_business_ids_migrate_to_runtime_and_save_as_strict_v6(tmp_path):
    catalogue, hypothesis = _catalogue_with_valid_hypothesis()
    source_viewer = _Viewer(catalogue, hypothesis)
    first = _add_catalogue_triangle(source_viewer, 1)
    second = _add_catalogue_triangle(source_viewer, 2)
    source_world = source_viewer._get_active_scenario().topoWorld
    source_world.apply_attachment(
        TopologyEdgeEdgeAttachment("A001", first.element_id, "OB", second.element_id, "OB")
    )
    source_structure = _without_business_ids(source_world._exportPhysicalSnapshot())
    source_path = tmp_path / "source.xml"
    saveScenarioXml(source_viewer, str(source_path))
    _as_legacy_v5(source_path, clear_business_ids=True)

    loaded_viewer = _Viewer(catalogue)
    loadScenarioXml(loaded_viewer, str(source_path))
    loaded_world = loaded_viewer._get_active_scenario().topoWorld
    first_loaded = loaded_world.elements[first.element_id]
    second_loaded = loaded_world.elements[second.element_id]
    first_triangle = catalogue.get_triangle(first.source_triangle_id)

    assert first_loaded.source_triangle_id == first.source_triangle_id
    assert first_loaded.vertex_business_ids == [
        first_triangle.opening_city_id, first_triangle.base_city_id, first_triangle.light_city_id,
    ]
    assert [vertex.business_id for vertex in first_loaded.vertexes] == first_loaded.vertex_business_ids
    assert first_loaded.vertex_business_ids[1] == second_loaded.vertex_business_ids[1]
    assert loaded_world.attachments == source_world.attachments
    assert loaded_world.validate_world() == []
    assert _without_business_ids(loaded_world._exportPhysicalSnapshot()) == source_structure

    migrated_path = tmp_path / "migrated-v6.xml"
    saveScenarioXml(loaded_viewer, str(migrated_path))
    assert ET.parse(migrated_path).getroot().get("version") == "6"
    reloaded_viewer = _Viewer(catalogue)
    loadScenarioXml(reloaded_viewer, str(migrated_path))
    assert reloaded_viewer._get_active_scenario().topoWorld.validate_world() == []


def test_v5_business_id_migration_is_idempotent_when_ids_are_already_correct(tmp_path):
    catalogue, hypothesis = _catalogue_with_valid_hypothesis()
    source_viewer = _Viewer(catalogue, hypothesis)
    _add_catalogue_triangle(source_viewer, 1)
    source_world = source_viewer._get_active_scenario().topoWorld
    expected_snapshot = source_world._exportPhysicalSnapshot()
    path = tmp_path / "complete-v5.xml"
    saveScenarioXml(source_viewer, str(path))
    _as_legacy_v5(path)

    loaded_viewer = _Viewer(catalogue)
    loadScenarioXml(loaded_viewer, str(path))
    assert loaded_viewer._get_active_scenario().topoWorld._exportPhysicalSnapshot() == expected_snapshot


def test_v5_business_id_migration_rejects_explicitly_contradictory_ids_atomically(tmp_path):
    catalogue, hypothesis = _catalogue_with_valid_hypothesis()
    source_viewer = _Viewer(catalogue, hypothesis)
    _add_catalogue_triangle(source_viewer, 1)
    path = tmp_path / "contradictory-v5.xml"
    saveScenarioXml(source_viewer, str(path))
    _as_legacy_v5(path)
    tree = ET.parse(path)
    snapshot = tree.getroot().find("topoSnapshot")
    payload = json.loads(snapshot.text)
    payload["elements"][0]["vertex_business_ids"][0] = "CITY-9999"
    snapshot.text = json.dumps(payload)
    tree.write(path, encoding="utf-8", xml_declaration=True)

    target = _Viewer(catalogue, hypothesis)
    before = target._get_active_scenario().topoWorld
    with pytest.raises(ValueError, match="contradictoire"):
        loadScenarioXml(target, str(path))
    assert target._get_active_scenario().topoWorld is before


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_stri", "missing_scity", "unknown_world_source", "wrong_business_id",
        "missing_business_id", "wrong_scity_label",
    ],
)
def test_v6_invalid_reference_contracts_are_rejected_atomically(tmp_path, mutation):
    catalogue, hypothesis = _catalogue_with_valid_hypothesis()
    source_viewer = _Viewer(catalogue, hypothesis)
    _add_local_triangle(
        source_viewer, 25, city_ref_id="SCITY-0001", triangle_ref_id="STRI-0001", city_name="Tmp",
    )
    source = tmp_path / "source.xml"
    saveScenarioXml(source_viewer, str(source))
    root = ET.parse(source).getroot()

    if mutation == "missing_stri":
        root.find("scenarioReference/triangles").remove(
            root.find("scenarioReference/triangles/triangle")
        )
    elif mutation == "missing_scity":
        root.find("scenarioReference/triangles/triangle").set("lightCityRefId", "SCITY-9999")
    else:
        snapshot = root.find("topoSnapshot")
        payload = json.loads(snapshot.text)
        element = payload["elements"][0]
        if mutation == "unknown_world_source":
            element["source_triangle_id"] = "STRI-9999"
        elif mutation == "wrong_business_id":
            element["vertex_business_ids"][2] = "SCITY-9999"
        elif mutation == "missing_business_id":
            element["vertex_business_ids"][2] = None
        else:
            element["vertex_labels"][2] = "Nom incohérent"
        snapshot.text = json.dumps(payload)
    invalid = tmp_path / f"{mutation}.xml"
    ET.ElementTree(root).write(invalid, encoding="utf-8", xml_declaration=True)

    target = _Viewer(catalogue, hypothesis)
    before_world = target._get_active_scenario().topoWorld
    before_reference = target._get_active_scenario().reference
    with pytest.raises(ValueError):
        loadScenarioXml(target, str(invalid))
    assert target._get_active_scenario().topoWorld is before_world
    assert target._get_active_scenario().reference is before_reference


def test_v6_save_keeps_the_existing_file_when_replace_fails(monkeypatch, tmp_path):
    catalogue, hypothesis = _catalogue_with_valid_hypothesis()
    target = tmp_path / "atomic.xml"
    target.write_text("ancien contenu", encoding="utf-8")

    def fail_replace(*_args):
        raise OSError("replace impossible")

    monkeypatch.setattr(assembleur_io_module.os, "replace", fail_replace)

    with pytest.raises(OSError, match="replace impossible"):
        saveScenarioXml(_Viewer(catalogue, hypothesis), str(target))
    assert target.read_text(encoding="utf-8") == "ancien contenu"
    assert not list(tmp_path.glob(".atomic.xml.*.tmp"))
