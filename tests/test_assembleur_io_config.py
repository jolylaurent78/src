from __future__ import annotations

import pytest

from src import assembleur_io


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
