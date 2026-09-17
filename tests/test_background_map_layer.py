from PIL import Image

from src.assembleur_background_map_layer import (
    BackgroundMapLayer,
    BackgroundMapWorldRect,
    format_scale,
)


def _layer(on_geometry_changed=lambda: None):
    return BackgroundMapLayer(
        lambda point: point,
        lambda x, y: (x, y),
        on_geometry_changed,
    )


def test_set_map_clear_and_interaction_state_are_headless():
    layer = _layer()
    layer.set_map(Image.new("RGBA", (20, 10)), BackgroundMapWorldRect(1, 2, 20, 10), "map.png")

    assert layer.has_map is True
    assert layer.world_rect == BackgroundMapWorldRect(1, 2, 20, 10)
    layer.start_move(1, 2)
    assert layer.is_moving is True
    layer.cancel_interaction()
    assert layer.is_moving is False
    layer.clear()
    assert layer.has_map is False
    assert layer.world_rect is None


def test_move_updates_world_rect_and_notifies_callback():
    notifications = []
    layer = _layer(lambda: notifications.append("changed"))
    layer.set_map(Image.new("RGBA", (20, 10)), BackgroundMapWorldRect(1, 2, 20, 10))
    layer.start_move(10, 10)

    assert layer.update_move(15, 3) is True
    assert layer.world_rect == BackgroundMapWorldRect(6, -5, 20, 10)
    assert notifications == ["changed"]


def test_resize_preserves_image_aspect_ratio_and_notifies_callback():
    notifications = []
    layer = _layer(lambda: notifications.append("changed"))
    layer.set_map(Image.new("RGBA", (40, 20)), BackgroundMapWorldRect(0, 0, 40, 20))
    layer.start_resize("tr", 40, 20)

    assert layer.update_resize(80, 20) is True
    assert layer.world_rect == BackgroundMapWorldRect(0, 0, 80, 40)
    assert notifications == ["changed"]


def test_handle_hit_test_uses_screen_coordinates_without_canvas():
    layer = _layer()
    layer.set_map(Image.new("RGBA", (40, 20)), BackgroundMapWorldRect(0, 0, 40, 20))

    assert layer.hit_test_handle(40, 20) == "tr"
    assert layer.hit_test_handle(100, 100) is None


def test_format_scale_preserves_historical_display():
    assert format_scale(None) == "x?"
    assert format_scale(1.0) == "x1"
    assert format_scale(2.5) == "x2.50"
    assert format_scale(0.5) == "x1/2.00"
