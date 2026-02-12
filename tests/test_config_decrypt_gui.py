import src.assembleur_io as assembleur_io


class ViewerStub:
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.appConfig = {}
        self.debug_io = False

    def saveAppConfig(self):
        return assembleur_io.saveAppConfig(self)

    def loadAppConfig(self):
        return assembleur_io.loadAppConfig(self)


def test_decrypt_patterns_roundtrip(tmp_path):
    cfg_path = tmp_path / "assembleur_config.json"
    viewer = ViewerStub(str(cfg_path))

    patterns = [
        {"text": "alpha [*]", "active": True},
        {"text": "[localise] beta", "active": False},
    ]

    assembleur_io.setAppConfigValue(viewer, "decryptPatterns", patterns)

    # Reload from disk into a fresh viewer
    viewer2 = ViewerStub(str(cfg_path))
    assembleur_io.loadAppConfig(viewer2)

    assert assembleur_io.getAppConfigValue(viewer2, "decryptPatterns", None) == patterns
