from types import SimpleNamespace

import src.assembleur_tk as assembleur_tk
import src.assembleur_tooltip as tooltip_module
from src.assembleur_deformation_window import DeformationWindow
from src.assembleur_tk import TriangleViewerManual
from src.assembleur_tooltip import attach_tooltip


class _Widget:
    def __init__(self):
        self.bindings = {}
        self.after_calls = []
        self.cancelled = []

    def bind(self, sequence, callback, add=None):
        self.bindings[sequence] = callback

    def after(self, delay, callback):
        self.after_calls.append((delay, callback))
        return "after-1"

    def after_cancel(self, after_id):
        self.cancelled.append(after_id)

    def winfo_exists(self):
        return True

    def winfo_rootx(self):
        return 95

    def winfo_rooty(self):
        return 96

    def winfo_height(self):
        return 10

    def winfo_screenwidth(self):
        return 100

    def winfo_screenheight(self):
        return 100


class _TooltipWindow:
    def __init__(self):
        self.geometry_value = None
        self.destroyed = False

    def wm_overrideredirect(self, _value):
        pass

    def attributes(self, *_args):
        pass

    def update_idletasks(self):
        pass

    def winfo_width(self):
        return 60

    def winfo_height(self):
        return 30

    def geometry(self, value):
        self.geometry_value = value

    def destroy(self):
        self.destroyed = True


class _TooltipLabel:
    def __init__(self, *_args, **_kwargs):
        pass

    def pack(self):
        pass


def test_widget_tooltip_delays_hides_cleans_up_and_clamps(monkeypatch):
    widget = _Widget()
    window = _TooltipWindow()
    monkeypatch.setattr(tooltip_module.tk, "Toplevel", lambda _parent: window)
    monkeypatch.setattr(tooltip_module.tk, "Label", _TooltipLabel)

    tooltip = attach_tooltip(widget, "Aide", delay_ms=321)

    assert set(widget.bindings) == {"<Enter>", "<Leave>", "<ButtonPress>", "<Destroy>"}
    widget.bindings["<Enter>"]()
    assert widget.after_calls == [(321, tooltip._show)]
    widget.bindings["<Leave>"]()
    assert widget.cancelled == ["after-1"]

    widget.bindings["<Enter>"]()
    widget.after_calls[-1][1]()
    assert window.geometry_value == "+40+70"

    widget.bindings["<ButtonPress>"]()
    assert window.destroyed is True
    widget.bindings["<Destroy>"]()


def test_ui_tooltip_wrapper_delegates_and_legacy_members_are_absent(monkeypatch):
    widget = object()
    delegated = []
    expected = SimpleNamespace()
    monkeypatch.setattr(
        assembleur_tk, "attach_tooltip",
        lambda received_widget, text: delegated.append((received_widget, text)) or expected,
    )
    viewer = TriangleViewerManual.__new__(TriangleViewerManual)

    assert viewer._ui_attach_tooltip(widget, "Aide") is expected
    assert delegated == [(widget, "Aide")]
    assert viewer._ui_attach_tooltip(None, "Aide") is None
    assert not hasattr(TriangleViewerManual, "_ui_show_tooltip")
    assert not hasattr(TriangleViewerManual, "_ui_hide_tooltip")
    assert "_ui_tooltip" not in viewer.__dict__
    assert "_ui_tooltip_label" not in viewer.__dict__


def test_deformation_window_keeps_using_the_shared_widget_tooltip_helper():
    assert "attach_tooltip" in DeformationWindow.__init__.__code__.co_names
