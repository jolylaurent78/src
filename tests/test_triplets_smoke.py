import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

from helpers_triplets import triplets_smoke


def test_triplets_smoke_are_valid(triplets_smoke):
    assert len(triplets_smoke) == 4
    for t in triplets_smoke:
        assert t.isGeometrieValide is True
        assert isinstance(t.azOA, float)
        assert isinstance(t.azOB, float)
        assert isinstance(t.angleDeg, float)
