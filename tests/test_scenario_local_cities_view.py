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
    view._fit_applied = False
    view.listbox = _Listbox()
    view.map_view = _MapView()
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
    assert len(view.map_view.fit_calls) == 1

    view.listbox.selected = (1,)
    ScenarioCitiesView._on_list_selected(view)
    assert view.map_view.selected == "SCITY-0002"
    assert view.map_view.recenter is True

    ScenarioCitiesView._on_map_marker_selected(view, "SCITY-0001")
    assert view.listbox.selected == (0,)
    assert view.listbox.seen_index == 0


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


def test_scenario_cities_view_renames_only_its_draft_and_preserves_selection():
    original = _reference(ScenarioCity("SCITY-0001", "Temp", 48.8, 2.2))
    draft = original.clone()
    view = _view(draft)
    view._name_var = _Var()
    view._name_entry = _Entry()
    view._on_reference_changed = lambda: setattr(view, "changed", True)
    ScenarioCitiesView.refresh(view)
    view._selected_city_ref_id = "SCITY-0001"
    view._name_var.set("Local")

    ScenarioCitiesView._on_name_committed(view)

    assert original.cities["SCITY-0001"].name == "Temp"
    assert draft.cities["SCITY-0001"].name == "Local"
    assert view.listbox.entries == ["Local"]
    assert view.map_view.markers[0].label == "Local"
    assert view._selected_city_ref_id == "SCITY-0001"
    assert view.changed is True
