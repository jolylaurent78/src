from src.assembleur_deformation_points import WorkingPoint
from src.assembleur_deformation_ui import DeformationUiState
from src.assembleur_geometry_reference import ScenarioReference


def test_state_is_directly_occurrence_based_without_a_cow_feature_flag():
    state = DeformationUiState()
    state.enter()
    state.select("T08", object())
    state.begin_drag("L")
    point = state.ensure_working_point(("T08", "L"), (1.0, 2.0))
    assert point.occurrences == {("T08", "L")}
    assert not hasattr(state, "copy_on_write_enabled")
    assert not hasattr(state, "city_lambert_overrides")


def test_working_point_drag_accept_release_and_restore():
    state = DeformationUiState()
    reference = object()
    candidate = object()
    state.enter()
    state.select("T08", reference)
    state.begin_drag("L")
    state.ensure_working_point(("T08", "L"), (1.0, 2.0))
    state.accept_occurrence_candidate((10.0, 20.0), candidate)
    assert state.end_occurrence_drag()
    assert state.working_point_for_occurrence(("T08", "L")).lambert_xy == (10.0, 20.0)
    assert state.dirty
    assert state.restore_working_point(("T08", "L")) == {("T08", "L")}
    assert state.last_accepted_world is candidate
    assert not state.dirty


def test_explicit_shared_working_point_never_merges_another_identity():
    state = DeformationUiState()
    state.enter()
    state.select("T12", object())
    shared = state.ensure_working_point(
        ("T12", "B"), (1.0, 2.0), (("T13", "B"),)
    )
    state.set_shared_working_point((("T14", "B"),), (1.0, 2.0))
    assert state.working_point_for_occurrence(("T13", "B")) is shared
    assert state.working_point_for_occurrence(("T14", "B")) is not shared


def test_rebase_discards_only_unvalidated_working_state():
    state = DeformationUiState()
    reference = ScenarioReference()
    state.enter()
    state.select("T08", object())
    state.working_points["TMP-1"] = WorkingPoint("TMP-1", (1.0, 2.0), {("T08", "L")})
    state.toggle_pivoted_attachment("A001")
    state.rebase_after_commit(object(), reference)
    assert state.working_points == {}
    assert state.pivoted_attachment_ids == set()
    assert state.working_reference is not reference
    assert not state.dirty


def test_exit_discards_temporary_working_points():
    state = DeformationUiState()
    state.enter()
    state.select("T08", object())
    state.working_points["TMP-1"] = WorkingPoint("TMP-1", (1.0, 2.0), {("T08", "L")})
    state.exit()
    assert not state.active
    assert state.working_points == {}
    assert state.reference_world is None
