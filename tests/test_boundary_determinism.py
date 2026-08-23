import json
import math
import os
from pathlib import Path
import subprocess
import sys
import textwrap

import numpy as np

from src.assembleur_core import (
    TopologyElement,
    TopologyVertexEdgeAttachment,
    TopologyWorld,
)


def _triangle(element_id: str, light_xy: tuple[float, float]) -> TopologyElement:
    opening = (0.0, 0.0)
    base = (10.0, 0.0)
    return TopologyElement(
        element_id=element_id,
        name=element_id,
        vertex_labels=["O", "B", "L"],
        vertex_types=["O", "B", "L"],
        edge_lengths_km=[
            10.0,
            math.dist(base, light_xy),
            math.dist(light_xy, opening),
        ],
        vertex_local_xy={0: opening, 1: base, 2: light_xy},
    )


def _ambiguous_radial_boundary_world() -> tuple[TopologyWorld, str]:
    """Construit deux demi-arêtes distinctes sur le même rayon de T02:N1."""
    world = TopologyWorld()
    world.add_element_as_new_group(_triangle("T01", (3.0, 4.0)))
    world.add_element_as_new_group(_triangle("T02", (6.0, 8.0)))
    world.setElementPose("T01", np.eye(2), np.zeros(2), mirrored=True)
    world.apply_attachment(
        TopologyVertexEdgeAttachment(
            "A001", "T01", "O", "OB", "T02", "B", "OB", "CCW", "CW"
        )
    )
    group_id = world.get_group_of_element("T02")
    world.replay_group_attachment_poses(group_id, "T02")
    world.setElementPose("T02", np.eye(2), np.zeros(2), mirrored=True)
    preview_world = world.clonePhysicalState()
    return preview_world, preview_world.get_group_of_element("T02")


def _ambiguous_boundary_payload() -> dict[str, object]:
    world, group_id = _ambiguous_radial_boundary_world()
    segments = world.getBoundarySegments(group_id)
    return {
        "cycle": world._concept_cache(group_id).boundaryCycle,
        "segments": [(segment.conceptA, segment.conceptB) for segment in segments],
    }


def test_ambiguous_radial_neighbors_make_boundary_unusable():
    payload = _ambiguous_boundary_payload()

    assert payload == {"cycle": [], "segments": []}


def test_boundary_is_identical_for_multiple_python_hash_seeds():
    script = textwrap.dedent(
        """
        import json
        import runpy

        helpers = runpy.run_path("tests/test_boundary_determinism.py")
        print(json.dumps(helpers["_ambiguous_boundary_payload"](), sort_keys=True))
        """
    )
    root = Path(__file__).resolve().parents[1]
    payloads = []
    for seed in ("1", "2", "3", "42", "123"):
        environment = os.environ.copy()
        environment["PYTHONHASHSEED"] = seed
        completed = subprocess.run(
            [sys.executable, "-c", script],
            cwd=root,
            env=environment,
            capture_output=True,
            check=True,
            text=True,
        )
        payloads.append(json.loads(completed.stdout))

    assert payloads == [{"cycle": [], "segments": []}] * 5


def test_non_degenerate_boundary_is_preserved_after_recomputation():
    world = TopologyWorld()
    group_id = world.add_element_as_new_group(_triangle("T01", (3.0, 4.0)))

    first_cycle = list(world.getBoundaryCycle(group_id, "T01:N0"))
    first_segments = world.getBoundarySegments(group_id)
    world.recomputeConceptAndBoundary(group_id)

    assert world.getBoundaryCycle(group_id, "T01:N0") == first_cycle
    assert world.getBoundarySegments(group_id) == first_segments
    assert len(first_segments) == 3
