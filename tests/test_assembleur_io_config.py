from __future__ import annotations

import pytest

from src import assembleur_io
from src.assembleur_tk import TriangleViewerManual


class _Viewer:
    def __init__(self, app_config_marker=...):
        if app_config_marker is not ...:
            self.appConfig = app_config_marker
        self.save_calls = 0

    def saveAppConfig(self):
        self.save_calls += 1


def test_set_app_config_value_creates_missing_mapping_and_saves() -> None:
    viewer = _Viewer()

    assembleur_io.setAppConfigValue(viewer, "key", "value")

    assert viewer.appConfig == {"key": "value"}
    assert viewer.save_calls == 1


def test_set_app_config_value_replaces_none_mapping_and_saves() -> None:
    viewer = _Viewer(None)

    assembleur_io.setAppConfigValue(viewer, "key", "value")

    assert viewer.appConfig == {"key": "value"}
    assert viewer.save_calls == 1


def test_set_app_config_value_updates_existing_mapping_and_saves() -> None:
    viewer = _Viewer({"key": "old", "other": 1})

    assembleur_io.setAppConfigValue(viewer, "key", "new")

    assert viewer.appConfig == {"key": "new", "other": 1}
    assert viewer.save_calls == 1


def test_set_app_config_value_propagates_save_failure() -> None:
    class FailingViewer(_Viewer):
        def saveAppConfig(self):
            self.save_calls += 1
            raise OSError("disk full")

    viewer = FailingViewer({})

    with pytest.raises(OSError, match="disk full"):
        assembleur_io.setAppConfigValue(viewer, "key", "value")
    assert viewer.appConfig == {"key": "value"}
    assert viewer.save_calls == 1


def test_map_opacity_slider_writes_only_the_global_config_and_redraws() -> None:
    class _Variable:
        def __init__(self, value):
            self.value = value

        def get(self):
            return self.value

        def set(self, value):
            self.value = value

    class _OpacityViewer:
        _on_map_opacity_change = TriangleViewerManual._on_map_opacity_change

        def __init__(self):
            self.map_opacity = _Variable(35)
            self._map_opacity_redraw_job = None
            self._last_drawn = [object()]
            self.config = {}
            self.redraws = 0

        def setAppConfigValue(self, key, value):
            self.config[key] = value

        def after(self, _delay, callback):
            callback()
            return "redraw-job"

        def after_cancel(self, _job):
            raise AssertionError("Aucun redraw en attente n'est attendu.")

        def _redraw_from(self, entries):
            assert entries is self._last_drawn
            self.redraws += 1

    viewer = _OpacityViewer()

    viewer._on_map_opacity_change()

    assert viewer.map_opacity.get() == 35
    assert viewer.config == {"uiMapOpacity": 35}
    assert viewer.redraws == 1
