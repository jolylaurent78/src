from types import SimpleNamespace

import pytest

from src.assembleur_geometry_reference import ScenarioCity, ScenarioReference
from src.assembleur_scenario_cities_view import ScenarioCitiesView


class _Listbox:
    def __init__(self):
        self.entries = []
        self.selected = ()
        self.seen_index = None

    def delete(self, *_args):
        self.entries.clear()

    def insert(self, _index, value):
        self.entries.append(value)

    def curselection(self):
        return self.selected

    def selection_clear(self, *_args):
        self.selected = ()

    def selection_set(self, index):
        self.selected = (int(index),)

    def activate(self, _index):
        pass

    def see(self, index):
        self.seen_index = int(index)


class _MapView:
    def __init__(self):
        self.markers = []
        self.selected = None
        self.recenter = False
        self.fit_calls = []

    def set_markers(self, markers):
        self.markers = list(markers)

    def set_selected_marker(self, marker_id, *, recenter=False):
        self.selected = marker_id
        self.recenter = recenter

    def fit_to_bounds(self, coordinates):
        self.fit_calls.append(list(coordinates))


class _Var:
    def __init__(self, value=""):
        self.value = value

    def get(self):
        return self.value

    def set(self, value):
        self.value = value


class _Entry:
    def configure(self, **_kwargs):
        pass


class _Button:
    def __init__(self):
        self.state = None

    def configure(self, **kwargs):
        self.state = kwargs.get("state", self.state)


def _reference(*cities):
    reference = ScenarioReference()
    for city in cities:
        reference.add_city(city)
    return reference


def _view(reference):
    view = object.__new__(ScenarioCitiesView)
    view._reference = reference
    view._scenario_city_ids = []
    view._selected_city_ref_id = None
    view.listbox = _Listbox()
    view.map_view = _MapView()
    view._rename_button = _Button()
    view._delete_button = _Button()
    view._active_triangle_ref_ids = lambda: ()
    view._on_reference_changed = None
    return view


def test_scenario_cities_view_lists_orphans_in_stable_order_and_selects_both_ways():
    reference = _reference(
        ScenarioCity("SCITY-0002", "Zed", 48.7, 2.1),
        ScenarioCity("SCITY-0001", "Alpha", 48.8, 2.2),
    )
    view = _view(reference)

    ScenarioCitiesView.refresh(view)

    assert view._scenario_city_ids == ["SCITY-0001", "SCITY-0002"]
    assert view.listbox.entries == ["Alpha", "Zed"]
    assert [marker.marker_id for marker in view.map_view.markers] == ["SCITY-0001", "SCITY-0002"]
    assert [
        (marker.latitude, marker.longitude, marker.label)
        for marker in view.map_view.markers
    ] == [(48.8, 2.2, "Alpha"), (48.7, 2.1, "Zed")]
    assert view.map_view.fit_calls == []

    view.listbox.selected = (1,)
    ScenarioCitiesView._on_list_selected(view)
    assert view.map_view.selected == "SCITY-0002"
    assert view.map_view.recenter is True

    ScenarioCitiesView._on_map_marker_selected(view, "SCITY-0001")
    assert view.listbox.selected == (0,)
    assert view.listbox.seen_index == 0


def test_scenario_cities_view_loads_only_the_catalogue_reference_map(monkeypatch):
    loaded_maps = []
    view = object.__new__(ScenarioCitiesView)
    view.map_view = SimpleNamespace(set_map=lambda calibrated: loaded_maps.append(calibrated))
    catalogue = SimpleNamespace(
        default_map_id="MAP-SCENARIO",
        catalogue_reference_map_id="MAP-FRANCE",
        get_map=lambda map_id: f"definition:{map_id}",
    )
    monkeypatch.setattr(
        "src.assembleur_scenario_cities_view.ApplicationPaths.from_runtime",
        lambda: "paths",
    )
    monkeypatch.setattr(
        "src.assembleur_scenario_cities_view.CatalogueMapAssetResolver",
        lambda paths: f"resolver:{paths}",
    )
    monkeypatch.setattr(
        "src.assembleur_scenario_cities_view.load_calibrated_catalogue_map",
        lambda map_definition, resolver: f"calibrated:{map_definition}:{resolver}",
    )

    ScenarioCitiesView._load_catalogue_reference_map(view, catalogue)

    assert loaded_maps == ["calibrated:definition:MAP-FRANCE:resolver:paths"]


def test_scenario_cities_view_does_not_fallback_when_reference_map_is_absent():
    loaded_maps = []
    view = object.__new__(ScenarioCitiesView)
    view.map_view = SimpleNamespace(set_map=lambda calibrated: loaded_maps.append(calibrated))
    catalogue = SimpleNamespace(
        default_map_id="MAP-SCENARIO",
        catalogue_reference_map_id=None,
        get_map=lambda _map_id: pytest.fail("No fallback map must be loaded"),
    )

    ScenarioCitiesView._load_catalogue_reference_map(view, catalogue)

    assert loaded_maps == []


def test_scenario_cities_view_uses_only_its_explicit_reference_and_refreshes_rename():
    first = _reference(ScenarioCity("SCITY-0001", "Temp", 48.8, 2.2))
    second = _reference()
    view = _view(first)
    ScenarioCitiesView.refresh(view)

    ScenarioCitiesView.refresh(view, second)
    assert view.listbox.entries == []
    assert view.map_view.markers == []

    first.cities["SCITY-0001"].name = "Renommée"
    ScenarioCitiesView.refresh(view, first)
    assert view.listbox.entries == ["Renommée"]
    assert view.map_view.markers[0].label == "Renommée"


def test_scenario_cities_view_renames_only_its_draft_and_preserves_selection(monkeypatch):
    original = _reference(ScenarioCity("SCITY-0001", "Temp", 48.8, 2.2))
    draft = original.clone()
    view = _view(draft)
    view._on_reference_changed = lambda: setattr(view, "changed", True)
    ScenarioCitiesView.refresh(view)
    view._selected_city_ref_id = "SCITY-0001"
    monkeypatch.setattr(
        "src.assembleur_scenario_cities_view.simpledialog.askstring",
        lambda *_args, **_kwargs: "Local",
    )
    view.winfo_toplevel = lambda: None

    ScenarioCitiesView._rename_selected_city(view)

    assert original.cities["SCITY-0001"].name == "Temp"
    assert draft.cities["SCITY-0001"].name == "Local"
    assert view.listbox.entries == ["Local"]
    assert view.map_view.markers[0].label == "Local"
    assert view._selected_city_ref_id == "SCITY-0001"
    assert view.changed is True


def test_scenario_cities_view_delete_state_and_orphan_cascade_are_draft_only():
    original = _reference(
        ScenarioCity("SCITY-0001", "Temp", 48.8, 2.2),
        ScenarioCity("SCITY-0002", "Base", 48.7, 2.1),
        ScenarioCity("SCITY-0003", "Light", 48.6, 2.0),
    )
    triangle = original.create_triangle(
        "Local", "SCITY-0001", "SCITY-0002", "SCITY-0003"
    )
    second_triangle = original.create_triangle(
        "Local deux", "SCITY-0002", "SCITY-0001", "SCITY-0003"
    )
    draft = original.clone()
    view = _view(draft)
    active_ids = []
    view._active_triangle_ref_ids = lambda: active_ids
    view._on_reference_changed = lambda: setattr(view, "changed", True)
    ScenarioCitiesView.refresh(view)
    ScenarioCitiesView._select_city(view, "SCITY-0001")

    assert view._rename_button.state == "normal"
    assert view._delete_button.state == "normal"

    active_ids.append(second_triangle.triangle_ref_id)
    ScenarioCitiesView.refresh_usage_state(view)
    assert view._delete_button.state == "disabled"

    active_ids.clear()
    ScenarioCitiesView.refresh_usage_state(view)
    ScenarioCitiesView._delete_selected_city(view)

    assert "SCITY-0001" in original.cities
    assert triangle.triangle_ref_id in original.triangles
    assert "SCITY-0001" not in draft.cities
    assert triangle.triangle_ref_id not in draft.triangles
    assert second_triangle.triangle_ref_id not in draft.triangles
    assert view._selected_city_ref_id is None
    assert view._rename_button.state == "disabled"
    assert view._delete_button.state == "disabled"
    assert view.changed is True
