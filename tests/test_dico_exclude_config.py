import pytest

import src.assembleur_tk_mixin_dictionary as dico_mixin


class _FakeBooleanVar:
    def __init__(self, value=False):
        self._value = bool(value)

    def get(self):
        return bool(self._value)

    def set(self, value):
        self._value = bool(value)


class _ViewerStub(dico_mixin.TriangleViewerDictionaryMixin):
    def __init__(self, config_value):
        self._config_value = config_value

    def getAppConfigValue(self, key, default=None):
        return self._config_value


@pytest.mark.parametrize(
    "value,expected_tag",
    [
        (False, None),
        (True, dico_mixin.DICO_TAG_EXCLURE),
    ],
)
def test_dico_exclude_bool_to_tag_mapping(monkeypatch, value, expected_tag):
    monkeypatch.setattr(dico_mixin.tk, "BooleanVar", _FakeBooleanVar)
    viewer = _ViewerStub(value)
    viewer._initDicoExcludeMotsCodesFromConfig()
    assert viewer._getDicoTagExclure() == expected_tag


def test_dico_exclude_config_type_is_strict_bool(monkeypatch):
    monkeypatch.setattr(dico_mixin.tk, "BooleanVar", _FakeBooleanVar)
    viewer = _ViewerStub("true")
    with pytest.raises(ValueError):
        viewer._initDicoExcludeMotsCodesFromConfig()
