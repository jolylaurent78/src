from src.assembleur_compass_controller import CompassController
from src.assembleur_compass_state import CompassState


class _DictionaryPanel:
    def __init__(self, filter_active: bool) -> None:
        self.filter_active = filter_active
        self.clear_calls = 0

    def clear_angle_filter(self) -> bool:
        self.clear_calls += 1
        self.filter_active = False
        return True


class _Viewer:
    def __init__(self, filter_active: bool) -> None:
        self.dictionary_panel = _DictionaryPanel(filter_active)
        self.compass_state = CompassState()
        self.compass_state.arc.last = {"angle": 42}
        self.compass_state.arc.last_angle_deg = 42.0
        self.cancel_calls = 0
        self.menu_updates = 0
        self.compass_controller = CompassController(
            self.compass_state, lambda p: p, lambda x, y: (x, y), lambda: None,
            lambda: [], lambda: None, lambda: True, lambda: self.dictionary_panel.filter_active,
            lambda: False, lambda text: None, lambda element_id: None,
            self._simulation_cancel_dictionary_filter, lambda: None, lambda: "#000000",
            lambda: None, self._update_compass_ctx_menu_and_dico_state, lambda value: None,
            lambda: False, lambda: None, lambda beacon_id: (0, 0), lambda: [], lambda: None,
        )

    def _simulation_cancel_dictionary_filter(self) -> None:
        self.cancel_calls += 1
        self.dictionary_panel.clear_angle_filter()

    def _update_compass_ctx_menu_and_dico_state(self) -> None:
        self.menu_updates += 1


def test_clear_last_arc_without_dictionary_filter() -> None:
    viewer = _Viewer(filter_active=False)

    viewer.compass_controller.clear_arc_last()

    assert viewer.cancel_calls == 0
    assert viewer.compass_state.arc.last is None
    assert viewer.compass_state.arc.last_angle_deg is None
    assert viewer.menu_updates == 1


def test_clear_last_arc_cancels_active_filter_via_standard_action() -> None:
    viewer = _Viewer(filter_active=True)

    viewer.compass_controller.clear_arc_last()

    assert viewer.cancel_calls == 1
    assert viewer.dictionary_panel.clear_calls == 1
    assert viewer.compass_state.arc.last is None
    assert viewer.compass_state.arc.last_angle_deg is None
    assert viewer.menu_updates == 1
