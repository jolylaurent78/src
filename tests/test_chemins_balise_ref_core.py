import math

import pytest

from src.assembleur_core import TopologyCheminTriplet


class _WorldStub:
    def __init__(self) -> None:
        self._nodes = {
            "A": (1.0, 0.0),
            "O": (0.0, 0.0),
            "B": (0.0, 1.0),
        }
        self._balises = {
            "RefEast": (1.0, 0.0),
            "RefNorth": (0.0, 1.0),
        }

    def getConceptNodeWorldXY(self, node_id: str, _group_id: str) -> tuple[float, float]:
        return self._nodes[str(node_id)]

    def hasBalise(self, name: str) -> bool:
        return str(name) in self._balises

    def getBaliseWorldXY(self, name: str) -> tuple[float, float]:
        return self._balises[str(name)]

    def azimutDegFromDxDy(self, dx: float, dy: float) -> float:
        # 0 deg = Nord, sens horaire (convention Core)
        return float((math.degrees(math.atan2(dx, dy)) + 360.0) % 360.0)


def test_triplet_calculer_geometrie_balise_ref_name_required() -> None:
    t = TopologyCheminTriplet("A", "O", "B")
    world = _WorldStub()

    with pytest.raises(ValueError, match="baliseRefName invalide"):
        t.calculerGeometrie(world, "G1", "cw", "")


def test_triplet_calculer_geometrie_balise_ref_must_exist() -> None:
    t = TopologyCheminTriplet("A", "O", "B")
    world = _WorldStub()

    with pytest.raises(ValueError, match="balise de reference introuvable"):
        t.calculerGeometrie(world, "G1", "cw", "MissingBalise")


def test_triplet_calculer_geometrie_uses_selected_balise_reference() -> None:
    world = _WorldStub()
    t_east = TopologyCheminTriplet("A", "O", "B")
    t_north = TopologyCheminTriplet("A", "O", "B")

    t_east.calculerGeometrie(world, "G1", "cw", "RefEast")
    t_north.calculerGeometrie(world, "G1", "cw", "RefNorth")

    assert t_east.isGeometrieValide
    assert t_north.isGeometrieValide
    assert t_east.azOA == pytest.approx(0.0)
    assert t_east.azOB == pytest.approx(90.0)
    assert t_east.angleDeg == pytest.approx(270.0)
    assert t_north.azOA == pytest.approx(90.0)
    assert t_north.azOB == pytest.approx(0.0)
    assert t_north.angleDeg == pytest.approx(270.0)
