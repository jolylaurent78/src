from types import SimpleNamespace

import pytest

from src.assembleur_core import ScenarioAssemblage, TopologyElement, TopologyWorld
from src.assembleur_tk import TriangleViewerManual


class _Status:
    def __init__(self):
        self.messages = []

    def config(self, **kwargs):
        self.messages.append(kwargs)


def _save_viewer(scenario):
    viewer = TriangleViewerManual.__new__(TriangleViewerManual)
    viewer.scenarios = [scenario]
    viewer.active_scenario_index = 0
    viewer.status = _Status()
    viewer._get_active_scenario = lambda: viewer.scenarios[viewer.active_scenario_index]
    viewer._is_active_auto_scenario = lambda: viewer._get_active_scenario().source_type == "auto"
    viewer._refresh_scenario_listbox = lambda: None
    viewer._set_active_scenario = lambda index: setattr(viewer, "active_scenario_index", index)
    return viewer


def test_scenario_runtime_file_identity_and_placeholder_are_explicit():
    scenario = ScenarioAssemblage("Initial", is_placeholder=True)

    assert scenario.file_path is None
    assert scenario.is_placeholder is True


def test_save_reuses_existing_path_without_opening_a_dialog(tmp_path):
    target = tmp_path / "manuel.xml"
    scenario = ScenarioAssemblage("Brouillon", file_path=str(target), is_placeholder=True)
    viewer = _save_viewer(scenario)
    saved_paths = []
    viewer.save_scenario_xml = saved_paths.append
    viewer._scenario_save_as_dialog = lambda: (_ for _ in ()).throw(AssertionError("dialogue inattendu"))

    TriangleViewerManual._scenario_save(viewer)

    assert saved_paths == [str(target)]
    assert scenario.file_path == str(target)
    assert scenario.name == "manuel"
    assert scenario.is_placeholder is False


def test_save_as_associates_file_name_and_converts_placeholder(monkeypatch, tmp_path):
    target = tmp_path / "nouveau.xml"
    scenario = ScenarioAssemblage("Brouillon", is_placeholder=True)
    viewer = _save_viewer(scenario)
    viewer.scenario_dir = str(tmp_path)
    saved_paths = []
    viewer.save_scenario_xml = saved_paths.append
    monkeypatch.setattr(
        "src.assembleur_tk.filedialog.asksaveasfilename", lambda **_kwargs: str(target)
    )

    TriangleViewerManual._scenario_save_as_dialog(viewer)

    assert saved_paths == [str(target)]
    assert scenario.file_path == str(target)
    assert scenario.name == "nouveau"
    assert scenario.is_placeholder is False


def test_save_of_auto_scenario_remembers_the_file(tmp_path):
    target = tmp_path / "scenario-auto.xml"
    auto = ScenarioAssemblage("ScenarioAuto", source_type="auto")
    viewer = _save_viewer(auto)
    saved_paths = []
    viewer.save_scenario_xml = saved_paths.append

    TriangleViewerManual._save_active_scenario_to_path(viewer, str(target))

    saved = viewer._get_active_scenario()
    assert saved_paths == [str(target)]
    assert saved is auto
    assert saved.source_type == "auto"
    assert saved.file_path == str(target)
    assert saved.name == "scenario-auto"
    assert saved.is_placeholder is False


def _load_viewer(active_scenario):
    viewer = TriangleViewerManual.__new__(TriangleViewerManual)
    viewer.scenarios = [active_scenario]
    viewer.active_scenario_index = 0
    viewer.status = _Status()
    viewer.canvas = SimpleNamespace(winfo_width=lambda: 800, winfo_height=lambda: 600)
    viewer._get_active_scenario = lambda: viewer.scenarios[viewer.active_scenario_index]
    viewer._exit_deformation_mode = lambda: None
    viewer._create_manual_scenario_hypothesis = lambda **_kwargs: object()
    viewer._capture_view_state = lambda: {}
    viewer._capture_map_state = lambda: {}
    viewer._set_active_scenario = lambda index: setattr(viewer, "active_scenario_index", index)
    viewer.load_scenario_xml = lambda _path: None
    viewer._fit_to_view = lambda _entries: None
    viewer._screen_to_world = lambda _x, _y: (0.0, 0.0)
    viewer._clock_clear_anchor_binding = lambda: None
    viewer._draw_clock_overlay = lambda: None
    viewer._redraw_overlay_only = lambda: None
    viewer._refresh_scenario_listbox = lambda: None
    viewer._last_drawn = []
    return viewer


def test_loading_replaces_only_the_active_placeholder(tmp_path):
    placeholder = ScenarioAssemblage("Initial", is_placeholder=True)
    viewer = _load_viewer(placeholder)
    source = tmp_path / "reference.xml"

    TriangleViewerManual._load_scenario_into_new_scenario(viewer, str(source))

    assert len(viewer.scenarios) == 1
    loaded = viewer._get_active_scenario()
    assert loaded.name == "reference"
    assert loaded.file_path == str(source)
    assert loaded.is_placeholder is False


def test_loading_keeps_a_real_active_scenario(tmp_path):
    manual = ScenarioAssemblage("Manuel", is_placeholder=False)
    viewer = _load_viewer(manual)
    source = tmp_path / "reference.xml"

    TriangleViewerManual._load_scenario_into_new_scenario(viewer, str(source))

    assert viewer.scenarios[0] is manual
    assert len(viewer.scenarios) == 2
    assert viewer._get_active_scenario().name == "reference"


def test_new_empty_scenario_keeps_current_map_and_view_context(tmp_path):
    map_path = tmp_path / "carte.svg"
    map_path.write_text("<svg/>", encoding="utf-8")
    previous = ScenarioAssemblage("Manuel")
    viewer = TriangleViewerManual.__new__(TriangleViewerManual)
    viewer.scenarios = [previous]
    viewer.active_scenario_index = 0
    viewer.status = _Status()
    viewer._bg = {"path": str(map_path), "x0": 1.0, "y0": 2.0, "w": 30.0, "h": 40.0}
    viewer._bg_scale_factor_override = None
    current_view = {"zoom": 2.0, "offset_x": 12.0, "offset_y": 34.0}
    current_map = {
        "path": str(map_path), "x0": 1.0, "y0": 2.0, "w": 30.0, "h": 40.0,
        "visible": True, "opacity": 100, "scale": None,
    }
    viewer._capture_view_state = lambda: dict(current_view)
    viewer._capture_map_state = lambda: dict(current_map)
    viewer.show_map_layer = SimpleNamespace(set=lambda _value: None)
    viewer.map_opacity = SimpleNamespace(
        set=lambda _value: None,
        get=lambda: 100,
    )
    viewer._create_manual_scenario_hypothesis = lambda **_kwargs: object()
    viewer._exit_deformation_mode = lambda: None
    viewer._attach_beacon_resolver_to_world = lambda _world: None
    viewer._refresh_scenario_listbox = lambda: None

    def activate(index):
        viewer.active_scenario_index = index
        TriangleViewerManual._apply_map_state(
            viewer, viewer.scenarios[index].map_state, persist=False, redraw=False
        )

    viewer._set_active_scenario = activate

    TriangleViewerManual._new_empty_scenario(viewer)

    created = viewer.scenarios[1]
    assert viewer.scenarios[0] is previous
    assert viewer.active_scenario_index == 1
    assert created.last_drawn == []
    assert created.view_state == current_view
    assert created.map_state == current_map
    assert viewer._bg is not None
    assert viewer._bg["path"] == str(map_path)


def test_duplicate_keeps_current_map_context_and_reapplies_group_anchor(tmp_path):
    map_path = tmp_path / "carte.svg"
    map_path.write_text("<svg/>", encoding="utf-8")
    viewer = TriangleViewerManual.__new__(TriangleViewerManual)

    class _Hypothesis:
        def clone(self):
            return _Hypothesis()

    class _MapDependentBeaconResolver:
        def __init__(self):
            self.anchor_reapplied = False

        def contains(self, beacon_id):
            return beacon_id == "BEA-0001"

        def get_world(self, beacon_id):
            assert beacon_id == "BEA-0001"
            assert viewer._bg is not None
            self.anchor_reapplied = True
            return (10.0, 20.0)

    resolver = _MapDependentBeaconResolver()
    source = ScenarioAssemblage("Source", hypothesis=_Hypothesis())
    source.topoWorld = TopologyWorld(beacon_resolver=resolver)
    group_id = source.topoWorld.add_element_as_new_group(TopologyElement(
        element_id="T01",
        name="Triangle",
        vertex_labels=["O", "B", "L"],
        vertex_types=["O", "B", "L"],
        edge_lengths_km=[3.0, 4.0, 5.0],
    ))
    node_id = source.topoWorld.get_element_vertex_node_id_by_type("T01", "O")
    source.topoWorld.createGroupAnchor(group_id, "BEA-0001", node_id)

    viewer.scenarios = [source]
    viewer.active_scenario_index = 0
    viewer.status = _Status()
    viewer._beacon_world_resolver = resolver
    viewer._bg = {"path": str(map_path), "x0": 1.0, "y0": 2.0, "w": 30.0, "h": 40.0}
    viewer._bg_scale_factor_override = None
    current_view = {"zoom": 2.0, "offset_x": 12.0, "offset_y": 34.0}
    current_map = {
        "path": str(map_path), "x0": 1.0, "y0": 2.0, "w": 30.0, "h": 40.0,
        "visible": True, "opacity": 100, "scale": None,
    }
    viewer._capture_view_state = lambda: dict(current_view)
    viewer._capture_map_state = lambda: dict(current_map)
    viewer.show_map_layer = SimpleNamespace(set=lambda _value: None)
    viewer.map_opacity = SimpleNamespace(set=lambda _value: None, get=lambda: 100)
    viewer._exit_deformation_mode = lambda: None
    viewer._refresh_scenario_listbox = lambda: None

    def activate(index):
        duplicate = viewer.scenarios[index]
        TriangleViewerManual._apply_map_state(
            viewer, duplicate.map_state, persist=False, redraw=False
        )
        TriangleViewerManual._reapply_scenario_group_anchors(viewer, duplicate)
        viewer.active_scenario_index = index

    viewer._set_active_scenario = activate

    TriangleViewerManual._scenario_duplicate(viewer)

    duplicate = viewer.scenarios[1]
    assert viewer.scenarios[0] is source
    assert viewer.active_scenario_index == 1
    assert duplicate.view_state == current_view
    assert duplicate.map_state == current_map
    assert viewer._bg is not None
    assert resolver.anchor_reapplied is True
    assert duplicate.topoWorld is not source.topoWorld
    duplicate.topoWorld.removeGroupAnchor("AN001")
    assert source.topoWorld.getGroupAnchor("AN001") is not None


def _delete_viewer(scenarios, active_index):
    viewer = TriangleViewerManual.__new__(TriangleViewerManual)
    viewer.scenarios = scenarios
    viewer.active_scenario_index = active_index
    viewer.ref_scenario_token = None
    viewer.status = _Status()
    viewer._exit_deformation_mode = lambda: None
    viewer._refresh_scenario_listbox = lambda: None
    return viewer


def test_delete_active_last_scenario_activates_replacement_before_removal(monkeypatch):
    first = ScenarioAssemblage("A")
    removed = ScenarioAssemblage("B")
    viewer = _delete_viewer([first, removed], active_index=1)
    activations = []
    observed_during_pop = []

    class _ObservedScenarios(list):
        def pop(self, index=-1):
            result = super().pop(index)
            observed_during_pop.append(TriangleViewerManual._get_active_scenario(viewer))
            return result

    viewer.scenarios = _ObservedScenarios(viewer.scenarios)

    def activate(index):
        # Le scénario supprimé est encore présent pendant l'activation du
        # remplaçant ; tout callback peut donc lire un actif valide.
        assert viewer.scenarios == [first, removed]
        assert TriangleViewerManual._get_active_scenario(viewer) is removed
        viewer.active_scenario_index = index
        assert TriangleViewerManual._get_active_scenario(viewer) is first
        activations.append(index)

    viewer._set_active_scenario = activate
    monkeypatch.setattr("src.assembleur_tk.messagebox.askyesno", lambda *_args, **_kwargs: True)

    TriangleViewerManual._scenario_delete(viewer)

    assert activations == [0]
    assert viewer.scenarios == [first]
    assert viewer.active_scenario_index == 0
    assert TriangleViewerManual._get_active_scenario(viewer) is first
    assert observed_during_pop == [first]


def test_delete_first_of_two_manual_scenarios_is_allowed(monkeypatch):
    removed = ScenarioAssemblage("A", source_type="manual")
    successor = ScenarioAssemblage("B", source_type="manual")
    viewer = _delete_viewer([removed, successor], active_index=0)
    activations = []

    def activate(index):
        assert viewer.scenarios == [removed, successor]
        viewer.active_scenario_index = index
        activations.append(index)

    viewer._set_active_scenario = activate
    monkeypatch.setattr("src.assembleur_tk.messagebox.askyesno", lambda *_args, **_kwargs: True)

    TriangleViewerManual._scenario_delete(viewer)

    assert activations == [1]
    assert viewer.scenarios == [successor]
    assert viewer.active_scenario_index == 0
    assert TriangleViewerManual._get_active_scenario(viewer) is successor


def test_delete_active_middle_scenario_remaps_the_already_activated_successor(monkeypatch):
    first = ScenarioAssemblage("A")
    removed = ScenarioAssemblage("B")
    successor = ScenarioAssemblage("C")
    viewer = _delete_viewer([first, removed, successor], active_index=1)
    activations = []
    observed_during_pop = []

    class _ObservedScenarios(list):
        def pop(self, index=-1):
            result = super().pop(index)
            observed_during_pop.append(TriangleViewerManual._get_active_scenario(viewer))
            return result

    viewer.scenarios = _ObservedScenarios(viewer.scenarios)

    def activate(index):
        assert viewer.scenarios == [first, removed, successor]
        viewer.active_scenario_index = index
        assert TriangleViewerManual._get_active_scenario(viewer) is successor
        activations.append(index)

    viewer._set_active_scenario = activate
    monkeypatch.setattr("src.assembleur_tk.messagebox.askyesno", lambda *_args, **_kwargs: True)

    TriangleViewerManual._scenario_delete(viewer)

    assert activations == [2]
    assert viewer.scenarios == [first, successor]
    assert viewer.active_scenario_index == 1
    assert TriangleViewerManual._get_active_scenario(viewer) is successor
    assert observed_during_pop == [successor]


def test_delete_refuses_the_last_manual_even_with_automatic_scenarios(monkeypatch):
    manual = ScenarioAssemblage("Manuel", source_type="manual")
    auto_a = ScenarioAssemblage("Auto A", source_type="auto")
    auto_b = ScenarioAssemblage("Auto B", source_type="auto")
    viewer = _delete_viewer([manual, auto_a, auto_b], active_index=0)
    viewer._set_active_scenario = lambda _index: pytest.fail("activation inattendue")
    monkeypatch.setattr(
        "src.assembleur_tk.messagebox.askyesno",
        lambda *_args, **_kwargs: pytest.fail("confirmation inattendue"),
    )
    messages = []
    monkeypatch.setattr(
        "src.assembleur_tk.messagebox.showinfo",
        lambda *_args, **_kwargs: messages.append(_args),
    )

    TriangleViewerManual._scenario_delete(viewer)

    assert viewer.scenarios == [manual, auto_a, auto_b]
    assert viewer.active_scenario_index == 0
    assert TriangleViewerManual._get_active_scenario(viewer) is manual
    assert messages == [("Supprimer le scénario", "Impossible de supprimer le dernier scénario manuel.")]


def test_delete_refuses_the_only_manual_scenario(monkeypatch):
    manual = ScenarioAssemblage("Manuel", source_type="manual")
    viewer = _delete_viewer([manual], active_index=0)
    viewer._set_active_scenario = lambda _index: pytest.fail("activation inattendue")
    monkeypatch.setattr(
        "src.assembleur_tk.messagebox.askyesno",
        lambda *_args, **_kwargs: pytest.fail("confirmation inattendue"),
    )
    monkeypatch.setattr("src.assembleur_tk.messagebox.showinfo", lambda *_args, **_kwargs: None)

    TriangleViewerManual._scenario_delete(viewer)

    assert viewer.scenarios == [manual]
    assert TriangleViewerManual._get_active_scenario(viewer) is manual


def test_delete_manual_is_allowed_when_another_manual_exists_among_automatics(monkeypatch):
    removed = ScenarioAssemblage("A", source_type="manual")
    auto_a = ScenarioAssemblage("Auto A", source_type="auto")
    retained_manual = ScenarioAssemblage("B", source_type="manual")
    auto_b = ScenarioAssemblage("Auto B", source_type="auto")
    viewer = _delete_viewer([removed, auto_a, retained_manual, auto_b], active_index=0)

    viewer._set_active_scenario = lambda index: setattr(viewer, "active_scenario_index", index)
    monkeypatch.setattr("src.assembleur_tk.messagebox.askyesno", lambda *_args, **_kwargs: True)

    TriangleViewerManual._scenario_delete(viewer)

    assert viewer.scenarios == [auto_a, retained_manual, auto_b]
    assert retained_manual in viewer.scenarios
    assert viewer.active_scenario_index == 0
    assert TriangleViewerManual._get_active_scenario(viewer) is auto_a
