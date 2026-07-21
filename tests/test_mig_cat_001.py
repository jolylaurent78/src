import pandas as pd
import pytest

from src.assembleur_core import (
    ScenarioTriangleSet,
    TopologyWorld,
    TriangleCatalog,
)


def _catalog_dataframe(count=32):
    return pd.DataFrame([
        {
            "id": rank,
            "B": f"Base {rank}",
            "L": f"Lumiere {rank}",
            "len_OB": 3.0,
            "len_OL": 4.0,
            "len_BL": 5.0,
            "orient": "CW" if rank % 2 else "CCW",
        }
        for rank in range(1, count + 1)
    ])


def test_triangle_catalog_v1_requires_exactly_32_distinct_ranks():
    catalog = TriangleCatalog.from_dataframe(_catalog_dataframe())

    assert catalog.ranks() == tuple(range(1, 33))
    assert catalog.get_by_rank(1).model_id.startswith("model-")
    assert catalog.get_by_rank(32).vertex_labels == ("Bourges", "Base 32", "Lumiere 32")

    with pytest.raises(ValueError, match="32 modeles attendus"):
        TriangleCatalog.from_dataframe(_catalog_dataframe(31))

    invalid = _catalog_dataframe()
    invalid.loc[31, "id"] = 31
    with pytest.raises(ValueError, match="rank duplique"):
        TriangleCatalog.from_dataframe(invalid)


def test_scenario_triangle_set_and_world_availability_are_core_driven():
    catalog = TriangleCatalog.from_dataframe(_catalog_dataframe())
    triangle_set = ScenarioTriangleSet.from_catalog(catalog)
    world = TopologyWorld()
    world.set_scenario_triangle_set(triangle_set)

    assert not hasattr(triangle_set, "_catalog")
    assert triangle_set.ranks() == tuple(range(1, 33))
    assert world.get_available_triangle_ranks() == tuple(range(1, 33))

    element = catalog.get_by_model_id(
        triangle_set.get_model_id_for_rank(7)
    ).build_topology_element()
    world.add_element_as_new_group(element)

    assert element.triangle_rank == 7
    assert world.is_triangle_rank_used(7)
    assert 7 not in world.get_available_triangle_ranks()
    assert 8 in world.get_available_triangle_ranks()


def test_catalog_rejects_invalid_orientation():
    invalid = _catalog_dataframe()
    invalid.loc[0, "orient"] = "NORTH"

    with pytest.raises(ValueError, match="ligne invalide"):
        TriangleCatalog.from_dataframe(invalid)


def test_topodump_exposes_catalog_selection_and_element_model_identity(tmp_path):
    catalog = TriangleCatalog.from_dataframe(_catalog_dataframe())
    triangle_set = ScenarioTriangleSet.from_catalog(catalog)
    world = TopologyWorld()
    world.set_scenario_triangle_set(triangle_set)
    element = catalog.get_by_model_id(
        triangle_set.get_model_id_for_rank(3)
    ).build_topology_element()
    world.add_element_as_new_group(element)

    dump_path = world.export_topo_dump_xml(str(tmp_path / "TopoDump.xml"), catalog=catalog)
    dump = dump_path.read_text(encoding="utf-8") if hasattr(dump_path, "read_text") else open(dump_path, encoding="utf-8").read()

    assert "<Catalog available=\"1\"" in dump
    assert "<ScenarioTriangleSet>" in dump
    assert f'modelId="{element.model_id}"' in dump
    assert 'triangleRank="3"' in dump
