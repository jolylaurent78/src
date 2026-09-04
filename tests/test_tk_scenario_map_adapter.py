import json

from PIL import Image

from src.assembleur_catalogue import Catalogue, WorldRect
from src.assembleur_catalogue_identity import SystemCatalogueIdProvider
from src.assembleur_core import ScenarioAssemblage
from src.assembleur_paths import ApplicationPaths
from src.assembleur_scenario_map import ScenarioMapPosition, ScenarioMapState
from src.assembleur_tk_scenario_map import TriangleViewerScenarioMapMixin


class _Variable:
    def __init__(self, value):
        self.value = value

    def get(self):
        return self.value

    def set(self, value):
        self.value = value


class _Viewer(TriangleViewerScenarioMapMixin):
    def __init__(self, catalogue, paths, scenario):
        self.catalogue = catalogue
        self.paths = paths
        self.scenarios = [scenario]
        self.active_scenario_index = 0
        self.show_map_layer = _Variable(True)
        self.map_opacity = _Variable(100)
        self._last_drawn = []
        self._bg = None
        self._bg_base_pil = None
        self._bg_photo = None
        self._bg_resizing = None

    def _redraw_from(self, _entries):
        pass


def test_adapter_projects_resolved_map_and_uses_its_transform(tmp_path) -> None:
    paths = ApplicationPaths.from_runtime(
        installation_root=tmp_path / "installation", user_data_root=tmp_path / "user-root", catalogue_mode="SYS"
    )
    catalogue = Catalogue(id_provider=SystemCatalogueIdProvider())
    map_id = catalogue.add_map(
        name="Carte", image_file="map.jpg", calibration_file="map.json", projection="EPSG:2154",
        default_world_rect=WorldRect(10, 20, 400, 200), default_scale_factor=12,
    )
    paths.default_catalogue_maps_dir.mkdir(parents=True)
    Image.new("RGB", (200, 100), "white").save(paths.default_catalogue_maps_dir / "map.jpg")
    (paths.default_catalogue_maps_dir / "map.json").write_text(
        json.dumps({"projection": "EPSG:2154", "A": [[0.01, 0], [0, 0.01]], "offset": [0, 0]}),
        encoding="utf-8",
    )
    scenario = ScenarioAssemblage("Carte")
    state = ScenarioMapState(map_id, ScenarioMapPosition(30, 40), 18, False)
    scenario.map_state = state
    viewer = _Viewer(catalogue, paths, scenario)
    viewer.map_opacity.set(20)

    viewer._apply_map_state(state, redraw=False)

    assert viewer._bg["x0"] == 30
    assert viewer._bg["w"] == 600
    assert viewer._bg_base_pil.mode == "RGBA"
    assert viewer.show_map_layer.get() is False
    # L'opacité est une préférence UI globale : appliquer l'état de cette
    # carte ne peut pas la restaurer depuis le scénario.
    assert viewer.map_opacity.get() == 20
    assert viewer._catalogue_lambert_to_world(0, 0) == (30, 340)
