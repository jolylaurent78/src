"""Tooltip Tk réutilisable, sans dépendance externe."""

from __future__ import annotations

import tkinter as tk


class Tooltip:
    """Attache une aide flottante différée à un widget Tk."""

    def __init__(self, widget, text: str, *, delay_ms: int = 450):
        self.widget = widget
        self.text = str(text or "").strip()
        self.delay_ms = delay_ms
        self._after_id = None
        self._window = None
        widget.bind("<Enter>", self._schedule, add="+")
        widget.bind("<Leave>", self._hide, add="+")
        widget.bind("<ButtonPress>", self._hide, add="+")
        widget.bind("<Destroy>", self._destroy, add="+")

    def set_text(self, text: str) -> None:
        self.text = str(text or "").strip()

    def _schedule(self, _event=None):
        self._hide()
        try:
            self._after_id = self.widget.after(self.delay_ms, self._show)
        except tk.TclError:
            self._after_id = None

    def _show(self):
        self._after_id = None
        try:
            if not self.widget.winfo_exists():
                return
            window = tk.Toplevel(self.widget)
            window.wm_overrideredirect(True)
            window.attributes("-topmost", True)
            tk.Label(
                window, text=self.text, bg="#ffffe0", relief=tk.SOLID,
                borderwidth=1, padx=5, pady=2, justify=tk.LEFT,
            ).pack()
            window.update_idletasks()
            x = self.widget.winfo_rootx() + 10
            y = self.widget.winfo_rooty() + self.widget.winfo_height() + 8
            tw = max(1, int(window.winfo_width()))
            th = max(1, int(window.winfo_height()))
            sw = int(self.widget.winfo_screenwidth())
            sh = int(self.widget.winfo_screenheight())
            x = max(0, min(x, sw - tw))
            y = max(0, min(y, sh - th))
            window.geometry(f"+{x}+{y}")
            self._window = window
        except tk.TclError:
            self._window = None

    def _hide(self, _event=None):
        if self._after_id is not None:
            try:
                self.widget.after_cancel(self._after_id)
            except tk.TclError:
                pass
            self._after_id = None
        if self._window is not None:
            try:
                self._window.destroy()
            except tk.TclError:
                pass
            self._window = None

    def _destroy(self, _event=None):
        self._hide()


def attach_tooltip(widget, text: str, *, delay_ms: int = 450) -> Tooltip:
    """Crée et conserve le tooltip sur le widget pour la durée de sa vie."""
    tooltip = Tooltip(widget, text, delay_ms=delay_ms)
    widget._tooltip = tooltip
    return tooltip
