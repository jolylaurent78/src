from types import SimpleNamespace

from PIL import Image

from src.assembleur_catalogue import Catalogue, WorldRect
from src.assembleur_catalogue_identity import SystemCatalogueIdProvider, UserCatalogueIdProvider
from src.assembleur_catalogue_map_calibration import CatalogueMapCalibrationController
from src.assembleur_catalogue_book_asset_controller import CatalogueBookAssetController
from src.assembleur_catalogue_window import CatalogueWindow, _CALIBRATION_MAP_MAXIMUM_ZOOM
from src.assembleur_paths import ApplicationPaths


class _Value:
    def __init__(self, value):
        self.value = value

    def get(self):
        return self.value


def test_calibration_view_allows_zoom_to_eight_native_pixels():
    assert _CALIBRATION_MAP_MAXIMUM_ZOOM == 8.0


class _MapView:
    def __init__(self):
        self.handler = "unset"

    def set_map_click_handler(self, handler):
        self.handler = handler


class _Button:
    def __init__(self):
        self.options = {}

    def configure(self, **kwargs):
        self.options.update(kwargs)


def test_hand_and_focus_modes_install_the_expected_click_handler():
    window = object.__new__(CatalogueWindow)
    window._map_interaction_mode = "hand"
    window._selected_calibration_city_id = "CITY-USR-550e8400-e29b-41d4-a716-446655440000"
    window._get_selected_map = lambda: SimpleNamespace(map_id="MAP-USR-550e8400-e29b-41d4-a716-446655440000")
    window._map_calibration = SimpleNamespace(is_readonly=lambda _map: False)
    window._calibration_map_view = _MapView()
    window._map_hand_button = _Button()
    window._map_focus_button = _Button()

    CatalogueWindow._set_map_interaction_mode(window, "focus")
    assert window._map_interaction_mode == "focus"
    assert window._calibration_map_view.handler == window._on_map_focus_click

    CatalogueWindow._set_map_interaction_mode(window, "hand")
    assert window._map_interaction_mode == "hand"
    assert window._calibration_map_view.handler is None


def test_focus_click_records_the_selected_city_pixel():
    calls = []
    window = object.__new__(CatalogueWindow)
    window._selected_calibration_city_id = "CITY-USR-550e8400-e29b-41d4-a716-446655440000"
    window._get_selected_map = lambda: SimpleNamespace(map_id="MAP-USR-550e8400-e29b-41d4-a716-446655440000")
    window._map_calibration = SimpleNamespace(set_pixel=lambda *args: calls.append(args))
    window._refresh_maps = lambda **kwargs: calls.append(("refresh", kwargs))
    window._mark_dirty = lambda: calls.append("dirty")

    CatalogueWindow._on_map_focus_click(window, (12.5, 24.0))

    assert calls == [
        ("MAP-USR-550e8400-e29b-41d4-a716-446655440000", "CITY-USR-550e8400-e29b-41d4-a716-446655440000", 12.5, 24.0),
        ("refresh", {"fit": False}),
        "dirty",
    ]


def test_confirmed_close_discards_staged_user_map(monkeypatch, tmp_path):
    paths = ApplicationPaths.from_runtime(installation_root=tmp_path, user_data_root=tmp_path / "user")
    source = tmp_path / "source.png"
    Image.new("RGB", (20, 10), "white").save(source)
    window = object.__new__(CatalogueWindow)
    window.catalogue = Catalogue(id_provider=UserCatalogueIdProvider())
    window._paths = paths
    window._map_calibration = CatalogueMapCalibrationController(window.catalogue, paths)
    window._book_assets = CatalogueBookAssetController(window.catalogue, paths)
    window._map_calibration.stage_user_map(source, name="Brouillon", description="")
    staged = next((paths.user_catalogue_maps_dir / ".staging").iterdir())
    window._is_dirty = True
    window.destroy = lambda: setattr(window, "destroyed", True)
    monkeypatch.setattr("src.assembleur_catalogue_window.messagebox.askyesno", lambda *args, **kwargs: True)

    assert CatalogueWindow.request_close(window) is True
    assert not staged.exists()
    assert window.destroyed is True


def test_replacing_working_catalogue_rebinds_the_calibration_controller(tmp_path):
    paths = ApplicationPaths.from_runtime(installation_root=tmp_path, user_data_root=tmp_path / "user")
    first = Catalogue(id_provider=UserCatalogueIdProvider())
    second = Catalogue(id_provider=UserCatalogueIdProvider())
    window = object.__new__(CatalogueWindow)
    window.catalogue = first
    window._paths = paths
    window._map_calibration = CatalogueMapCalibrationController(first, paths)
    window._book_assets = CatalogueBookAssetController(first, paths)
    controller = window._map_calibration

    CatalogueWindow._replace_working_catalogue(window, second)

    assert window.catalogue is second
    assert window._map_calibration is controller
    assert window._map_calibration.catalogue is second


def test_rebound_controller_uses_cities_from_the_replaced_catalogue(tmp_path):
    paths = ApplicationPaths.from_runtime(installation_root=tmp_path, user_data_root=tmp_path / "user")
    first = Catalogue(id_provider=UserCatalogueIdProvider())
    second = Catalogue(id_provider=UserCatalogueIdProvider())
    cities = [second.add_city(name, latitude, longitude) for name, latitude, longitude in (
        ("A", 47.0, 2.0), ("B", 48.0, 3.0), ("C", 46.5, 4.0),
    )]
    map_id = second.add_map(
        name="Carte", image_file="map.png", calibration_file="map.json", projection=None,
        default_world_rect=WorldRect(0, 0, 100, 100), default_scale_factor=1,
        calibration_city_ids=[city.city_id for city in cities],
    )
    paths.user_catalogue_maps_dir.mkdir(parents=True)
    Image.new("RGB", (100, 100), "white").save(paths.user_catalogue_maps_dir / "map.png")
    (paths.user_catalogue_maps_dir / "map.json").write_text('{"points": []}', encoding="utf-8")
    window = object.__new__(CatalogueWindow)
    window.catalogue = first
    window._paths = paths
    window._map_calibration = CatalogueMapCalibrationController(first, paths)
    window._book_assets = CatalogueBookAssetController(first, paths)

    CatalogueWindow._replace_working_catalogue(window, second)
    for city, pixel in zip(cities, ((10, 10), (60, 20), (30, 70))):
        window._map_calibration.set_pixel(map_id, city.city_id, *pixel)

    assert set(window._map_calibration.points_for(second.get_map(map_id))) == {city.city_id for city in cities}


def test_rebind_preserves_staged_user_map_and_its_preview_source(tmp_path):
    paths = ApplicationPaths.from_runtime(installation_root=tmp_path, user_data_root=tmp_path / "user")
    source = tmp_path / "source.png"
    Image.new("RGB", (20, 10), "white").save(source)
    catalogue = Catalogue(id_provider=UserCatalogueIdProvider())
    controller = CatalogueMapCalibrationController(catalogue, paths)
    map_id = controller.stage_user_map(source, name="Brouillon", description="")
    staged = controller._staged_images[map_id]

    replacement = catalogue.clone()
    controller.rebind_catalogue(replacement)

    assert controller.catalogue is replacement
    assert staged.exists()
    assert controller.preview_map(replacement.get_map(map_id)).image_size == (20, 10)


def test_rebind_preserves_pending_calibration_points(tmp_path):
    paths = ApplicationPaths.from_runtime(installation_root=tmp_path, user_data_root=tmp_path / "user")
    catalogue = Catalogue(id_provider=UserCatalogueIdProvider())
    city = catalogue.add_city("Ville", 47.0, 2.0)
    map_id = catalogue.add_map(
        name="Carte", image_file="map.png", calibration_file="map.json", projection=None,
        default_world_rect=WorldRect(0, 0, 100, 100), default_scale_factor=1,
        calibration_city_ids=[city.city_id],
    )
    paths.user_catalogue_maps_dir.mkdir(parents=True)
    Image.new("RGB", (100, 100), "white").save(paths.user_catalogue_maps_dir / "map.png")
    (paths.user_catalogue_maps_dir / "map.json").write_text('{"points": []}', encoding="utf-8")
    controller = CatalogueMapCalibrationController(catalogue, paths)
    controller.set_pixel(map_id, city.city_id, 10, 20)

    replacement = catalogue.clone()
    controller.rebind_catalogue(replacement)

    assert controller.points_for(replacement.get_map(map_id))[city.city_id].pixel_x == 10


def test_finalize_commit_removes_staging_without_removing_published_asset(tmp_path):
    paths = ApplicationPaths.from_runtime(installation_root=tmp_path, user_data_root=tmp_path / "user")
    source = tmp_path / "source.png"
    Image.new("RGB", (20, 10), "white").save(source)
    catalogue = Catalogue(id_provider=UserCatalogueIdProvider())
    controller = CatalogueMapCalibrationController(catalogue, paths)
    map_id = controller.stage_user_map(source, name="Publiée", description="")
    catalogue_map = catalogue.get_map(map_id)
    staged = controller._staged_images[map_id]

    controller.commit()
    published = paths.user_catalogue_maps_dir / catalogue_map.image_file
    controller.finalize_commit()

    assert not staged.exists()
    assert published.exists()


def test_city_import_rebind_keeps_a_staged_user_map(monkeypatch, tmp_path):
    paths = ApplicationPaths.from_runtime(installation_root=tmp_path, user_data_root=tmp_path / "user")
    source = tmp_path / "source.png"
    Image.new("RGB", (20, 10), "white").save(source)
    window = object.__new__(CatalogueWindow)
    window.catalogue = Catalogue(id_provider=UserCatalogueIdProvider())
    window._paths = paths
    window._map_calibration = CatalogueMapCalibrationController(window.catalogue, paths)
    window._book_assets = CatalogueBookAssetController(window.catalogue, paths)
    map_id = window._map_calibration.stage_user_map(source, name="Brouillon", description="")
    staged = window._map_calibration._staged_images[map_id]
    window._read_cities_csv = lambda _path: ([('Ville importée', 47.0, 2.0)], [])
    window._refresh_city_list = lambda: None
    window._refresh_triangle_tree = lambda: None
    window._refresh_beacon_list = lambda: None
    window._mark_dirty = lambda: None
    monkeypatch.setattr("src.assembleur_catalogue_window.filedialog.askopenfilename", lambda **_kwargs: "cities.csv")
    monkeypatch.setattr("src.assembleur_catalogue_window.messagebox.showinfo", lambda *args, **kwargs: None)
    monkeypatch.setattr("src.assembleur_catalogue_window.messagebox.showwarning", lambda *args, **kwargs: None)

    CatalogueWindow._import_csv(window)

    assert map_id in window.catalogue.maps
    assert staged.exists()
    assert window._map_calibration.preview_map(window.catalogue.get_map(map_id)).image_size == (20, 10)


class _PreviewMapView:
    def __init__(self):
        self.map = None
        self.markers = None
        self.fit_calls = []
        self.selected_calls = []
        self.pixel_markers = None
        self.pixel_polylines = None

    def set_map(self, catalogue_map, *, preserve_view=False):
        self.map = catalogue_map
        self.preserve_view = preserve_view

    def set_markers(self, markers):
        self.markers = list(markers)

    def set_pixel_markers(self, markers):
        self.pixel_markers = list(markers)

    def set_pixel_polylines(self, polylines):
        self.pixel_polylines = list(polylines)

    def fit_to_bounds(self, coordinates, *, margin):
        self.fit_calls.append((coordinates, margin))

    def fit_to_pixel_bounds(self, points, *, margin):
        self.fit_calls.append((list(points), margin))

    def set_selected_marker(self, marker_id, *, recenter):
        self.selected_calls.append((marker_id, recenter))


class _Label:
    def configure(self, **_kwargs):
        pass


class _TextValue:
    def __init__(self, value=""):
        self.value = value

    def get(self):
        return self.value

    def set(self, value):
        self.value = value


class _ScaleWidget:
    def __init__(self):
        self.value = None
        self.state = None

    def set(self, value):
        self.value = value

    def configure(self, **kwargs):
        self.state = kwargs.get("state", self.state)


def _window_for_map_add_button(provider):
    window = object.__new__(CatalogueWindow)
    window.catalogue = Catalogue(id_provider=provider)
    window._selected_map_id = None
    window._selected_calibration_city_id = None
    window._map_interaction_mode = "hand"
    window._map_calibration = SimpleNamespace(is_readonly=lambda _catalogue_map: False)
    window._map_description_entry = _ScaleWidget()
    window._map_scale_entry = _ScaleWidget()
    window._map_scale_slider = _ScaleWidget()
    window._map_default_check = _ScaleWidget()
    window._map_archive_button = _ScaleWidget()
    window._map_delete_button = _ScaleWidget()
    window._calibration_city_add_button = _ScaleWidget()
    window._calibration_city_remove_button = _ScaleWidget()
    window._map_add_button = _ScaleWidget()
    window._map_role_label = _Label()
    window._icon_archive = object()
    window._icon_archive_off = object()
    window._set_map_interaction_mode = lambda _mode: None
    return window


def test_map_add_button_is_enabled_for_both_catalogue_modes():
    for provider in (SystemCatalogueIdProvider(), UserCatalogueIdProvider()):
        window = _window_for_map_add_button(provider)

        CatalogueWindow._update_map_action_buttons(window)

        assert window._map_add_button.state == "normal"


def _window_with_pointed_map():
    catalogue = Catalogue(id_provider=UserCatalogueIdProvider())
    city = catalogue.add_city("Ville", 47.0, 2.0)
    map_id = catalogue.add_map(
        name="Carte", image_file="map.png", calibration_file="map.json", projection="EPSG:2154",
        default_world_rect=WorldRect(0, 0, 100, 100), default_scale_factor=1,
        calibration_city_ids=[city.city_id],
    )
    window = object.__new__(CatalogueWindow)
    window.catalogue = catalogue
    window._selected_map_id = map_id
    window._calibration_map_view = _PreviewMapView()
    window._map_status_label = _Label()
    window._map_calibration = SimpleNamespace(
        preview_map=lambda catalogue_map: SimpleNamespace(
            map_id=catalogue_map.map_id,
            geographic_to_pixel=lambda latitude, longitude: (longitude * 100, latitude * 100),
        ),
        points_for=lambda _catalogue_map: {city.city_id: SimpleNamespace(pixel_x=123.0, pixel_y=456.0)},
        leave_one_out_residuals=lambda _catalogue_map: {
            city.city_id: SimpleNamespace(
                predicted_x=200.0, predicted_y=4700.0, error_px=4244.7, dx=77.0, dy=4244.0,
            )
        },
        status_for=lambda _catalogue_map: "Valide",
    )
    return window, city


def _window_with_placement():
    window, _city = _window_with_pointed_map()
    catalogue_map = window.catalogue.get_map(window._selected_map_id)
    window.catalogue.update_map(
        catalogue_map.map_id,
        default_world_rect=WorldRect(-2963.38, 1642.99, 100, 100),
        default_scale_factor=12.0,
    )
    window._map_scale_var = _TextValue()
    window._map_scale_slider = _ScaleWidget()
    window._updating_map_placement = False
    window._map_calibration.is_readonly = lambda _catalogue_map: False
    return window


def test_initial_calibration_map_load_fits_the_pointed_cities():
    window, city = _window_with_pointed_map()

    CatalogueWindow._load_selected_catalogue_map(window, fit=True)

    assert window._calibration_map_view.fit_calls == [([(123.0, 456.0)], 0.20)]
    assert [marker.marker_id for marker in window._calibration_map_view.pixel_markers] == [
        ("projected", city.city_id), city.city_id
    ]


def test_live_calibration_load_preserves_the_current_view_and_selected_city():
    window, city = _window_with_pointed_map()
    CatalogueWindow._load_selected_catalogue_map(window, fit=True)
    window._selected_calibration_city_id = city.city_id

    CatalogueWindow._load_selected_catalogue_map(window, fit=False)

    assert window._calibration_map_view.preserve_view is True
    assert window._calibration_map_view.fit_calls == [([(123.0, 456.0)], 0.20)]
    assert window._calibration_map_view.selected_calls == [(city.city_id, False)]


def test_calibration_preview_keeps_observed_pixel_separate_from_projection():
    window, city = _window_with_pointed_map()

    CatalogueWindow._load_selected_catalogue_map(window, fit=True)

    observed = next(
        marker for marker in window._calibration_map_view.pixel_markers if marker.marker_id == city.city_id
    )
    projected = next(
        marker
        for marker in window._calibration_map_view.pixel_markers
        if marker.marker_id == ("projected", city.city_id)
    )
    assert (observed.pixel_x, observed.pixel_y) == (123.0, 456.0)
    assert (projected.pixel_x, projected.pixel_y) == (200.0, 4700.0)
    assert window._calibration_map_view.pixel_polylines[0].points[0] == (123.0, 456.0)


def test_calibration_preview_uses_leave_one_out_residual_for_diagnostic_marker():
    window, city = _window_with_pointed_map()

    CatalogueWindow._load_selected_catalogue_map(window, fit=True)

    projected = next(
        marker
        for marker in window._calibration_map_view.pixel_markers
        if marker.marker_id == ("projected", city.city_id)
    )
    assert (projected.pixel_x, projected.pixel_y) == (200.0, 4700.0)


def test_map_scale_detail_loads_the_catalogue_scale_only():
    window = _window_with_placement()

    CatalogueWindow._refresh_map_placement_detail(window)

    assert window._map_scale_var.get() == "12.00"
    assert window._map_scale_slider.value == 12.0


def test_map_scale_slider_updates_only_the_working_catalogue():
    window = _window_with_placement()
    calls = []
    window._mark_dirty = lambda: calls.append("dirty")

    CatalogueWindow._on_map_scale_slider_changed(window, "8.5")

    assert window.catalogue.get_map(window._selected_map_id).default_scale_factor == 8.5
    assert window._map_scale_var.get() == "8.50"
    assert calls == ["dirty"]


def test_map_scale_entry_updates_the_slider_with_its_full_precision():
    window = _window_with_placement()
    window._map_scale_var.set("15.25")
    window._mark_dirty = lambda: None

    CatalogueWindow._commit_map_scale_entry(window)

    assert window.catalogue.get_map(window._selected_map_id).default_scale_factor == 15.25
    assert window._map_scale_slider.value == 15.25


def test_map_scale_readonly_does_not_mutate_the_catalogue():
    window = _window_with_placement()
    window._map_calibration.is_readonly = lambda _catalogue_map: True
    window._mark_dirty = lambda: (_ for _ in ()).throw(AssertionError("dirty inattendu"))

    CatalogueWindow._on_map_scale_slider_changed(window, "8.5")

    assert window.catalogue.get_map(window._selected_map_id).default_scale_factor == 12.0


def test_map_scale_entry_rejects_invalid_value_and_restores_previous(monkeypatch):
    window = _window_with_placement()
    window._map_scale_var.set("20.1")
    errors = []
    monkeypatch.setattr("src.assembleur_catalogue_window.messagebox.showerror", lambda *_args, **kwargs: errors.append(kwargs))

    CatalogueWindow._commit_map_scale_entry(window)

    assert window.catalogue.get_map(window._selected_map_id).default_scale_factor == 12.0
    assert window._map_scale_var.get() == "12.00"
    assert errors


def test_list_selection_recenters_marker_without_refitting():
    window, city = _window_with_pointed_map()
    window._calibration_city_tree = SimpleNamespace(selection=lambda: (city.city_id,))
    window._update_map_action_buttons = lambda: None

    CatalogueWindow._on_calibration_city_selected(window)

    assert window._calibration_map_view.selected_calls == [(city.city_id, True)]
    assert window._calibration_map_view.fit_calls == []


def test_marker_selection_selects_the_matching_calibration_city_in_the_tree():
    window, city = _window_with_pointed_map()
    calls = []
    window._calibration_city_tree = SimpleNamespace(
        selection_set=lambda value: calls.append(("selection", value)),
        focus=lambda value: calls.append(("focus", value)),
        see=lambda value: calls.append(("see", value)),
    )
    window._update_map_action_buttons = lambda: calls.append(("buttons", None))

    CatalogueWindow._on_calibration_marker_selected(window, city.city_id)

    assert calls == [("selection", city.city_id), ("focus", city.city_id), ("see", city.city_id), ("buttons", None)]
