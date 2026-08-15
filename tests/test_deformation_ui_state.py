from types import SimpleNamespace

from src.assembleur_deformation_ui import DeformationUiState
from src.assembleur_tk import TriangleViewerManual


def test_deformation_state_accumulates_two_drags_without_mutating_reference():
    state = DeformationUiState()
    reference = object()
    candidate_l = object()
    candidate_b = object()

    state.enter()
    state.select("T08", reference)
    state.begin_drag("L")
    assert state.candidate_overrides((10.0, 20.0)) == {"L": (10.0, 20.0)}
    state.accept_candidate((10.0, 20.0), candidate_l)
    assert state.preview_overrides() == {"L": (10.0, 20.0)}
    assert state.end_drag()

    state.begin_drag("B")
    assert state.candidate_overrides((30.0, 40.0)) == {
        "L": (10.0, 20.0),
        "B": (30.0, 40.0),
    }
    state.accept_candidate((30.0, 40.0), candidate_b)
    assert state.end_drag()

    assert state.reference_world is reference
    assert state.last_accepted_world is candidate_b
    assert state.vertex_lambert_overrides == {
        "L": (10.0, 20.0),
        "B": (30.0, 40.0),
    }


def test_deformation_state_redrag_replaces_only_the_dragged_role():
    state = DeformationUiState(active=True)
    state.select("T08", object())
    state.vertex_lambert_overrides.update({"L": (1.0, 2.0), "B": (3.0, 4.0)})

    state.begin_drag("L")
    assert state.candidate_overrides((5.0, 6.0)) == {
        "L": (5.0, 6.0),
        "B": (3.0, 4.0),
    }
    state.accept_candidate((5.0, 6.0), object())
    assert state.end_drag()

    assert state.vertex_lambert_overrides == {"L": (5.0, 6.0), "B": (3.0, 4.0)}


def test_deformation_state_rejected_drag_keeps_last_accepted_overrides():
    state = DeformationUiState(active=True)
    state.select("T08", object())
    state.vertex_lambert_overrides["L"] = (1.0, 2.0)

    state.begin_drag("B")
    assert state.candidate_overrides((3.0, 4.0)) == {
        "L": (1.0, 2.0),
        "B": (3.0, 4.0),
    }
    assert not state.end_drag()

    assert state.vertex_lambert_overrides == {"L": (1.0, 2.0)}


def test_deformation_state_exit_discards_all_temporary_data():
    state = DeformationUiState(active=True)
    state.select("T08", object())
    state.vertex_lambert_overrides["O"] = (1.0, 2.0)
    state.begin_drag("L")

    state.exit()

    assert not state.active
    assert state.element_id is None
    assert state.reference_world is None
    assert state.vertex_lambert_overrides == {}
    assert state.dragging_role is None
    assert state.last_accepted_world is None


def test_main_canvas_vertex_click_does_not_start_a_deformation_drag():
    viewer = TriangleViewerManual.__new__(TriangleViewerManual)
    viewer._deformation_state.enter()
    viewer._deformation_state.select("T08", object())
    viewer._ensure_pick_cache = lambda: None
    viewer._hit_test = lambda _x, _y: ("vertex", 0, "L")
    viewer._last_drawn = [{"topoElementId": "T08"}]

    assert viewer._handle_deformation_left_down(SimpleNamespace(x=0, y=0)) == "break"
    assert viewer._deformation_state.dragging_role is None
