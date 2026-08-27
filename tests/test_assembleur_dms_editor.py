import pytest

from src.assembleur_dms_editor import DmsCoordinateEditor


def test_parse_coordinate_pair_accepts_french_wikipedia_dms():
    latitude, longitude = DmsCoordinateEditor.parse_coordinate_pair(
        "49° 11′ 24″ nord, 2° 06′ 36″ ouest"
    )

    assert latitude == pytest.approx(49 + 11 / 60 + 24 / 3600)
    assert longitude == pytest.approx(-(2 + 6 / 60 + 36 / 3600))


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("49° 11′ 24″ NORD, 2° 06′ 36″ OUEST", (49 + 11 / 60 + 24 / 3600, -(2 + 6 / 60 + 36 / 3600))),
        ("49° 11′ 24″ sud, 2° 06′ 36″ est", (-(49 + 11 / 60 + 24 / 3600), 2 + 6 / 60 + 36 / 3600)),
        ("49° 11′ 24″ N, 2° 06′ 36″ W", (49 + 11 / 60 + 24 / 3600, -(2 + 6 / 60 + 36 / 3600))),
        ("47° 19′ 59″ N, 3° 11′ 13″ O", (47 + 19 / 60 + 59 / 3600, -(3 + 11 / 60 + 13 / 3600))),
        ("49° 11' 24\" N, 2° 06' 36\" W", (49 + 11 / 60 + 24 / 3600, -(2 + 6 / 60 + 36 / 3600))),
        ("49 11 24 N, 2 06 36 W", (49 + 11 / 60 + 24 / 3600, -(2 + 6 / 60 + 36 / 3600))),
        ("49.19, -2.11", (49.19, -2.11)),
    ],
)
def test_parse_coordinate_pair_preserves_supported_formats(text, expected):
    assert DmsCoordinateEditor.parse_coordinate_pair(text) == pytest.approx(expected)


@pytest.mark.parametrize(
    "text",
    [
        "49° 11′ 24″ nord, 2° 06′ 36″ sud",
        "49° 11′ 24″ est, 2° 06′ 36″ ouest",
        "91° 00′ 00″ nord, 2° 06′ 36″ ouest",
        "49° 11′ 24″ nord, 181° 00′ 00″ est",
        "49° 60′ 00″ nord, 2° 06′ 36″ ouest",
        "49° 11′ 60″ nord, 2° 06′ 36″ ouest",
    ],
)
def test_parse_coordinate_pair_rejects_invalid_dms(text):
    with pytest.raises(ValueError):
        DmsCoordinateEditor.parse_coordinate_pair(text)


def test_parse_coordinate_accepts_french_hemisphere_name():
    assert DmsCoordinateEditor.parse_coordinate("49° 11′ 24″ nord", "latitude") == pytest.approx(
        49 + 11 / 60 + 24 / 3600
    )
    assert DmsCoordinateEditor.parse_coordinate("2° 06′ 36″ O", "longitude") == pytest.approx(
        -(2 + 6 / 60 + 36 / 3600)
    )
