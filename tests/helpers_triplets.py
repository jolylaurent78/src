import pytest

from src.assembleur_core import TopologyCheminTriplet


def makeTriplet(
    nodeA: str,
    nodeO: str,
    nodeB: str,
    azOA: float,
    azOB: float,
    distOA_km: float,
    distOB_km: float,
    angleDeg: float,
) -> TopologyCheminTriplet:
    t = TopologyCheminTriplet(nodeA=nodeA, nodeO=nodeO, nodeB=nodeB)
    t.azOA = float(azOA)
    t.azOB = float(azOB)
    t.distOA_km = float(distOA_km)
    t.distOB_km = float(distOB_km)
    t.angleDeg = float(angleDeg)
    t.isGeometrieValide = True
    return t


@pytest.fixture
def triplets_smoke():
    return [
        makeTriplet("T01:N2", "T02:N1", "T03:N2", 248.08, 57.12, 159.360, 262.820, 169.04),
        makeTriplet("T03:N2", "T04:N1", "T05:N1", 317.78, 39.30, 474.190, 198.370, 81.53),
        makeTriplet("T05:N1", "T07:N1", "T09:N2", 173.60, 107.41, 14.960, 171.740, 293.81),
        makeTriplet("T09:N2", "T10:N1", "T11:N2", 269.76, 148.64, 351.450, 485.690, 238.89),
    ]


class CheminsStub:
    def __init__(self, triplets):
        self.triplets = list(triplets)
