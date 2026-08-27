from src.assembleur_deformation_window import DeformationWindow


class _Tree:
    def __init__(self):
        self.items = {}
        self.selected_iid = None

    def get_children(self):
        return tuple(self.items)

    def delete(self, *iids):
        for iid in iids:
            self.items.pop(iid, None)
        self.selected_iid = None

    def insert(self, _parent, _index, *, iid, text, values=()):
        self.items[iid] = text

    def selection_set(self, iid):
        self.selected_iid = iid

    def focus(self, _iid):
        pass

    def see(self, _iid):
        pass

    def selection(self):
        return () if self.selected_iid is None else (self.selected_iid,)


class _Button:
    def __init__(self):
        self.options = {}

    def configure(self, **kwargs):
        self.options.update(kwargs)


def _window_stub(callback):
    window = DeformationWindow.__new__(DeformationWindow)
    window._occurrence_tree = _Tree()
    window._occurrence_by_iid = {}
    window._updating_occurrences = False
    window._occurrences_update_generation = 0
    window._occurrences_guard_after_id = None
    window._map_pin_button = _Button()
    window._delete_button = _Button()
    window._restore_button = _Button()
    window._on_occurrence_selected = callback
    window.idle_callbacks = []
    window.cancelled = []
    window.after_idle = lambda callback: window.idle_callbacks.append(callback) or f"idle-{len(window.idle_callbacks)}"
    window.after_cancel = lambda after_id: window.cancelled.append(after_id)
    return window


def test_delayed_programmatic_treeview_selection_cannot_restore_old_deform_occurrence():
    selected_element = {"id": "T03"}
    window = _window_stub(lambda element_id, _role: selected_element.update(id=element_id))

    window.set_occurrences((("T02", "L", "T02:L", True, False),), ("T02", "L"))
    assert window._updating_occurrences is True

    # T03 vient d'être sélectionné au canvas ; Tk livre seulement maintenant
    # le <<TreeviewSelect>> produit par selection_set(T02:L).
    window._occurrence_tree_selected()
    assert selected_element["id"] == "T03"

    window.idle_callbacks.pop()()
    assert window._updating_occurrences is False


def test_treeview_user_selection_is_forwarded_after_programmatic_update_stabilizes():
    selected = []
    window = _window_stub(lambda element_id, role: selected.append((element_id, role)))

    window.set_occurrences((("T02", "L", "T02:L", True, False),), ("T02", "L"))
    window.idle_callbacks.pop()()
    window._occurrence_tree_selected()

    assert selected == [("T02", "L")]


def test_new_occurrence_update_replaces_the_previous_idle_guard_release():
    window = _window_stub(lambda *_args: None)

    window.set_occurrences((("T02", "L", "T02:L", True, False),), ("T02", "L"))
    first_release = window.idle_callbacks[-1]
    window.set_occurrences((("T03", "B", "T03:B", True, False),), ("T03", "B"))
    second_release = window.idle_callbacks[-1]

    first_release()
    assert window._updating_occurrences is True
    second_release()
    assert window._updating_occurrences is False
    assert window.cancelled == ["idle-1"]


def test_restore_button_tracks_the_selected_deformation_occurrence():
    window = _window_stub(lambda *_args: None)

    window.set_occurrences((('T02', 'L', 'T02:L', True, False),), None)
    assert window._restore_button.options['state'] == 'disabled'

    window.set_occurrences((('T02', 'L', 'T02:L', True, False),), ('T02', 'L'))
    assert window._restore_button.options['state'] == 'normal'


def test_validate_button_tracks_the_session_dirty_state():
    window = DeformationWindow.__new__(DeformationWindow)
    window._validate_button = _Button()

    window.set_validate_enabled(False)
    assert window._validate_button.options["state"] == "disabled"

    window.set_validate_enabled(True)
    assert window._validate_button.options["state"] == "normal"


def test_rename_button_tracks_the_local_scenario_city_state():
    window = DeformationWindow.__new__(DeformationWindow)
    window._rename_button = _Button()

    window.set_rename_enabled(False)
    assert window._rename_button.options["state"] == "disabled"

    window.set_rename_enabled(True)
    assert window._rename_button.options["state"] == "normal"
