from src.assembleur_chemin_edit_dialog import (
    CheminEditResult,
    selection_mask_from_view,
    snapshot_index_from_view,
)


def test_snapshot_index_from_view_keeps_snapshot_order_when_not_inverted():
    assert [snapshot_index_from_view(index, 5, False) for index in range(5)] == [0, 1, 2, 3, 4]


def test_snapshot_index_from_view_reverses_after_the_anchor_node():
    assert [snapshot_index_from_view(index, 5, True) for index in range(5)] == [0, 4, 3, 2, 1]


def test_selection_mask_from_view_returns_snapshot_order_for_reverse_orientation():
    assert selection_mask_from_view((True, False, True, False, True), "ccw", "cw") == (
        True,
        True,
        False,
        True,
        False,
    )


def test_chemin_edit_result_is_an_explicit_immutable_commit_payload():
    result = CheminEditResult("cw", (True, True, True), ("angle", "azimut"))

    assert result.orientation_user == "cw"
    assert result.selection_mask == (True, True, True)
    assert result.selected_measures == ("angle", "azimut")
