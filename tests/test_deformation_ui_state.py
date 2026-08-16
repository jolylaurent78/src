from types import SimpleNamespace

from src.assembleur_deformation_ui import DeformationUiState
from src.assembleur_tk import TriangleViewerManual


class _WindowStub:
    def __init__(self):
        self.exists = True
        self.calls = []

    def winfo_exists(self):
        return self.exists

    def deiconify(self):
        self.calls.append("deiconify")

    def lift(self):
        self.calls.append("lift")

    def focus_force(self):
        self.calls.append("focus_force")

    def destroy(self):
        self.calls.append("destroy")
        self.exists = False


def test_deformation_state_accumulates_two_drags_without_mutating_reference():
    state = DeformationUiState()
    reference = object()
    candidate_l = object()
    candidate_b = object()

    state.enter()
    state.select("T08", reference)
    state.begin_drag("L", "CITY-L")
    assert state.candidate_city_overrides((10.0, 20.0)) == {"CITY-L": (10.0, 20.0)}
    state.accept_candidate((10.0, 20.0), candidate_l)
    assert state.preview_city_overrides() == {"CITY-L": (10.0, 20.0)}
    assert state.modified_occurrences == []
    assert state.end_drag((("T08", "L"),))

    state.begin_drag("B", "CITY-B")
    assert state.candidate_city_overrides((30.0, 40.0)) == {
        "CITY-L": (10.0, 20.0),
        "CITY-B": (30.0, 40.0),
    }
    state.accept_candidate((30.0, 40.0), candidate_b)
    assert state.end_drag((("T08", "B"),))

    assert state.reference_world is reference
    assert state.last_accepted_world is candidate_b
    assert state.city_lambert_overrides == {
        "CITY-L": (10.0, 20.0),
        "CITY-B": (30.0, 40.0),
    }


def test_deformation_state_redrag_replaces_only_the_dragged_role():
    state = DeformationUiState(active=True)
    state.select("T08", object())
    state.city_lambert_overrides.update({"CITY-L": (1.0, 2.0), "CITY-B": (3.0, 4.0)})

    state.begin_drag("L", "CITY-L")
    assert state.candidate_city_overrides((5.0, 6.0)) == {
        "CITY-L": (5.0, 6.0),
        "CITY-B": (3.0, 4.0),
    }
    state.accept_candidate((5.0, 6.0), object())
    assert state.end_drag((("T08", "L"),))

    assert state.city_lambert_overrides == {"CITY-L": (5.0, 6.0), "CITY-B": (3.0, 4.0)}


def test_deformation_state_rejected_drag_keeps_last_accepted_overrides():
    state = DeformationUiState(active=True)
    state.select("T08", object())
    state.city_lambert_overrides["CITY-L"] = (1.0, 2.0)

    state.begin_drag("B", "CITY-B")
    assert state.candidate_city_overrides((3.0, 4.0)) == {
        "CITY-L": (1.0, 2.0),
        "CITY-B": (3.0, 4.0),
    }
    assert not state.end_drag((("T08", "B"),))

    assert state.city_lambert_overrides == {"CITY-L": (1.0, 2.0)}


def test_deformation_state_exit_discards_all_temporary_data():
    state = DeformationUiState(active=True)
    state.select("T08", object())
    state.city_lambert_overrides["CITY-O"] = (1.0, 2.0)
    state.begin_drag("L", "CITY-L")

    state.exit()

    assert not state.active
    assert state.element_id is None
    assert state.reference_world is None
    assert state.city_lambert_overrides == {}
    assert state.dragging_role is None
    assert state.last_accepted_world is None
    assert state.modified_occurrences == []
    assert state.selected_occurrence is None


def test_deformation_state_records_each_modified_occurrence_once_in_order():
    state = DeformationUiState()
    state.enter()
    state.select("T08", object())

    state.begin_drag("L", "CITY-SHARED")
    state.accept_candidate((1.0, 2.0), object())
    assert state.end_drag((("T08", "L"), ("T12", "B")))

    state.begin_drag("B", "CITY-B")
    state.accept_candidate((3.0, 4.0), object())
    assert state.end_drag((("T08", "B"),))

    state.begin_drag("B", "CITY-SHARED")
    state.accept_candidate((5.0, 6.0), object())
    assert state.end_drag((("T08", "L"), ("T12", "B")))

    assert state.modified_occurrences == [("T08", "L"), ("T12", "B"), ("T08", "B")]
    assert state.modified_roles_for_element("T08") == {"L", "B"}
    assert state.city_lambert_overrides == {
        "CITY-SHARED": (5.0, 6.0),
        "CITY-B": (3.0, 4.0),
    }


def test_shared_city_redrag_from_second_occurrence_keeps_one_list_entry_per_occurrence():
    state = DeformationUiState()
    shared_occurrences = (("T05", "B"), ("T06", "B"))
    state.enter()
    state.select("T05", object())
    state.begin_drag("B", "CITY-BASE")
    state.accept_candidate((10.0, 20.0), object())
    assert state.end_drag(shared_occurrences)
    assert state.selected_occurrence == ("T05", "B")

    state.select("T06", object())
    state.begin_drag("B", "CITY-BASE")
    state.accept_candidate((30.0, 40.0), object())
    assert state.end_drag(shared_occurrences)

    assert state.modified_occurrences == list(shared_occurrences)
    assert state.selected_occurrence == ("T06", "B")
    assert state.city_lambert_overrides == {"CITY-BASE": (30.0, 40.0)}


def test_deformation_state_keeps_occurrences_while_navigating_triangles():
    state = DeformationUiState()
    first_world = object()
    second_world = object()
    state.enter()
    state.select("T08", first_world)
    state.begin_drag("O", "CITY-O")
    state.accept_candidate((1.0, 2.0), first_world)
    assert state.end_drag((("T08", "O"),))

    state.select("T12", second_world)

    assert state.modified_occurrences == [("T08", "O")]
    assert state.last_accepted_world is first_world
    assert state.city_lambert_overrides == {"CITY-O": (1.0, 2.0)}


def test_deformation_state_selects_a_non_modified_occurrence_without_adding_it():
    state = DeformationUiState()
    state.enter()
    state.select("T08", object())

    state.select_occurrence("T08", "B")

    assert state.selected_occurrence == ("T08", "B")
    assert state.modified_occurrences == []


def test_drag_preview_does_not_rebuild_the_occurrence_list():
    class _PreviewWindow:
        def __init__(self):
            self.occurrence_calls = 0
            self.triangle_calls = 0

        def set_occurrences(self, *_args):
            self.occurrence_calls += 1

        def set_triangle(self, **_kwargs):
            self.triangle_calls += 1

    viewer = TriangleViewerManual.__new__(TriangleViewerManual)
    viewer._deformation_state.enter()
    viewer._deformation_state.select("T08", object())
    window = _PreviewWindow()
    viewer._ensure_deformation_window = lambda: window
    viewer._deformation_vertices = lambda: {}
    viewer._deformation_assembly_rotation_deg = lambda: 0.0

    viewer._refresh_deformation_window(refresh_occurrences=False)

    assert window.occurrence_calls == 0
    assert window.triangle_calls == 1


def test_deformation_triangle_navigation_does_not_simulate(monkeypatch):
    world = SimpleNamespace(
        elements={
            "T2": SimpleNamespace(source_triangle_id="TRI-0002"),
            "T5": SimpleNamespace(source_triangle_id="TRI-0005"),
        }
    )
    world.clonePhysicalState = lambda: world
    viewer = TriangleViewerManual.__new__(TriangleViewerManual)
    viewer._deformation_state.enter()
    viewer._get_active_scenario = lambda: SimpleNamespace(topoWorld=world)
    viewer._show_deformation_preview = lambda preview: setattr(
        viewer, "_shown_world", preview
    )
    viewer._refresh_deformation_window = lambda: None
    viewer.status = SimpleNamespace(config=lambda **_kwargs: None)

    def fail_if_simulated(**_kwargs):
        raise AssertionError("La navigation ne doit pas simuler une deformation")

    monkeypatch.setattr(
        "src.assembleur_tk.simulate_city_deformation",
        fail_if_simulated,
    )

    assert viewer._select_deformation_element("T2")
    viewer._deformation_state.modified_occurrences.append(("T2", "L"))

    assert viewer._select_deformation_element("T5")
    assert viewer._deformation_state.element_id == "T5"
    assert viewer._deformation_state.last_accepted_world is world
    assert viewer._shown_world is world

    assert viewer._select_deformation_element("T2")
    assert viewer._deformation_state.element_id == "T2"
    assert viewer._deformation_state.modified_occurrences == [("T2", "L")]


def test_canvas_navigation_clears_an_old_occurrence_selection():
    viewer = TriangleViewerManual.__new__(TriangleViewerManual)
    viewer._deformation_state.enter()
    viewer._deformation_state.select("T04", object())
    viewer._deformation_state.modified_occurrences.append(("T04", "L"))
    viewer._deformation_state.select_occurrence("T04", "L")
    viewer._ensure_pick_cache = lambda: None
    viewer._hit_test = lambda _x, _y: ("edge", 1, None)
    viewer._last_drawn = [{}, {"topoElementId": "T10"}]
    requested = []
    viewer._select_deformation_element = requested.append

    assert viewer._handle_deformation_left_down(SimpleNamespace(x=10, y=20)) == "break"

    assert requested == ["T10"]
    assert viewer._deformation_state.selected_occurrence is None
    assert viewer._deformation_state.modified_occurrences == [("T04", "L")]


def test_occurrence_navigation_keeps_the_requested_occurrence_selected():
    viewer = TriangleViewerManual.__new__(TriangleViewerManual)
    viewer._deformation_state.enter()
    viewer._deformation_state.select("T10", object())
    viewer._deformation_state.modified_occurrences.append(("T04", "L"))
    viewer._select_deformation_element = lambda element_id: setattr(
        viewer._deformation_state, "element_id", element_id
    ) or True
    viewer._refresh_deformation_window = lambda: None

    viewer._deformation_window_occurrence_selected("T04", "L")

    assert viewer._deformation_state.element_id == "T04"
    assert viewer._deformation_state.selected_occurrence == ("T04", "L")


def test_delete_last_city_override_restores_the_reference_world():
    viewer = TriangleViewerManual.__new__(TriangleViewerManual)
    reference_world = object()
    viewer._deformation_state.enter()
    viewer._deformation_state.select("T04", reference_world)
    viewer._deformation_state.city_lambert_overrides = {"CITY-L": (1.0, 2.0)}
    viewer._deformation_state.modified_occurrences = [("T04", "L")]
    viewer._deformation_state.select_occurrence("T04", "L")
    viewer._deformation_city_id_for_occurrence = lambda *_args: "CITY-L"
    viewer._deformation_occurrences_for_city = lambda *_args: (("T04", "L"),)
    viewer._show_deformation_preview = lambda world: setattr(viewer, "_shown_world", world)
    viewer._refresh_deformation_window = lambda: None

    viewer._deformation_delete_selected()

    assert viewer._deformation_state.city_lambert_overrides == {}
    assert viewer._deformation_state.modified_occurrences == []
    assert viewer._deformation_state.selected_occurrence is None
    assert viewer._deformation_state.last_accepted_world is reference_world
    assert viewer._shown_world is reference_world


def test_map_pin_commits_source_city_override_without_changing_its_occurrence(monkeypatch):
    class _Dialog:
        def __init__(self, *_args):
            pass

        def show(self):
            return "CITY-TARGET"

    viewer = TriangleViewerManual.__new__(TriangleViewerManual)
    viewer._deformation_state.enter()
    viewer._deformation_state.select("T04", object())
    viewer._deformation_state.select_occurrence("T04", "L")
    viewer.catalogue = SimpleNamespace(
        cities={"CITY-TARGET": SimpleNamespace(archived=False)},
        get_city_lambert=lambda city_id: (30.0, 40.0),
    )
    viewer._deformation_city_id_for_occurrence = lambda *_args: "CITY-SOURCE"
    viewer._deformation_occurrences_for_city = lambda *_args: (("T04", "L"),)
    candidate_world = object()
    viewer._apply_deformation_city_overrides = lambda overrides: candidate_world
    viewer._show_deformation_preview = lambda _world: None
    viewer._refresh_deformation_window = lambda: None
    monkeypatch.setattr("src.assembleur_tk.CitySelectionDialog", _Dialog)

    viewer._deformation_map_pin_selected()

    assert viewer._deformation_state.city_lambert_overrides == {
        "CITY-SOURCE": (30.0, 40.0),
    }
    assert viewer._deformation_state.modified_occurrences == [("T04", "L")]
    assert viewer._deformation_state.selected_occurrence == ("T04", "L")


def test_deformation_drag_coalesces_to_the_latest_pending_point():
    viewer = TriangleViewerManual.__new__(TriangleViewerManual)
    viewer._deformation_state.enter()
    viewer._deformation_state.select("T04", object())
    viewer._deformation_state.begin_drag("L", "CITY-L")
    scheduled = []
    processed = []
    viewer.after = lambda _delay, callback: scheduled.append(callback) or "after-1"
    viewer.after_cancel = lambda _after_id: None
    viewer._apply_deformation_city_overrides = lambda overrides: processed.append(overrides) or object()
    viewer._show_deformation_preview = lambda _world: None
    viewer._refresh_deformation_window = lambda **_kwargs: None

    for point in ((1.0, 1.0), (2.0, 2.0), (3.0, 3.0)):
        viewer._deformation_window_dragged("L", point)

    assert len(scheduled) == 1
    scheduled.pop()()

    assert processed == [{"CITY-L": (3.0, 3.0)}]


def test_main_canvas_vertex_click_does_not_start_a_deformation_drag():
    viewer = TriangleViewerManual.__new__(TriangleViewerManual)
    viewer._deformation_state.enter()
    viewer._deformation_state.select("T08", object())
    viewer._ensure_pick_cache = lambda: None
    viewer._hit_test = lambda _x, _y: ("vertex", 0, "L")
    viewer._last_drawn = [{"topoElementId": "T08"}]

    assert viewer._handle_deformation_left_down(SimpleNamespace(x=0, y=0)) == "break"
    assert viewer._deformation_state.dragging_role is None


def test_open_deformation_window_reuses_the_existing_session_window():
    viewer = TriangleViewerManual.__new__(TriangleViewerManual)
    viewer._deformation_state.enter()
    window = _WindowStub()
    viewer._deformation_window = window

    viewer._open_deformation_window()

    assert window.calls == ["deiconify", "lift", "focus_force"]
    assert viewer._deformation_state.active is True


def test_escape_keeps_an_open_deformation_session():
    viewer = TriangleViewerManual.__new__(TriangleViewerManual)
    viewer._deformation_state.enter()
    window = _WindowStub()
    viewer._deformation_window = window
    viewer._clock_trace_active = False
    viewer._clock_arc_active = False
    viewer._clock_measure_active = False
    viewer._clock_setref_active = False
    viewer._bg_calib_active = False
    viewer._drag = None
    viewer._sel = None

    viewer._on_escape_key(SimpleNamespace())

    assert window.winfo_exists() is True
    assert viewer._deformation_state.active is True


def test_close_deformation_session_clears_runtime_state_once():
    viewer = TriangleViewerManual.__new__(TriangleViewerManual)
    viewer._deformation_state.enter()
    window = _WindowStub()
    viewer._deformation_window = window
    viewer._sel = None
    viewer._reset_assist = lambda: None
    viewer._restore_deformation_real_projection = lambda: None
    viewer.canvas = SimpleNamespace(configure=lambda **_kwargs: None)
    viewer.status = SimpleNamespace(config=lambda **_kwargs: None)

    viewer._exit_deformation_mode()

    assert window.calls == ["destroy"]
    assert viewer._deformation_window is None
    assert viewer._deformation_state.active is False
