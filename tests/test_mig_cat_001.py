"""Structural assertions for the Catalogue/Core-only triangle runtime."""

from pathlib import Path

from src.assembleur_core import TopologyWorld, build_topology_element_from_catalogue_triangle


def test_legacy_triangle_excel_classes_are_absent_from_the_core_runtime():
    import src.assembleur_core as core

    assert not hasattr(core, "TriangleModel")
    assert not hasattr(core, "TriangleCatalog")
    assert not hasattr(core, "ScenarioTriangleSet")


def test_viewer_and_io_expose_no_legacy_triangle_excel_pipeline():
    root = Path(__file__).resolve().parents[1]
    runtime_sources = (
        (root / "src" / "assembleur_tk.py").read_text(encoding="utf-8"),
        (root / "src" / "assembleur_io.py").read_text(encoding="utf-8"),
    )
    removed_symbols = (
        "triangle_catalog",
        "TriangleFileService",
        "autoLoadTrianglesFileAtStartup",
        "DialogCreateTriangleExcel",
        "open_excel",
    )

    for symbol in removed_symbols:
        assert all(symbol not in source for source in runtime_sources)


def test_topodump_contains_only_core_triangle_identity(tmp_path):
    world = TopologyWorld()
    world.add_element_as_new_group(
        build_topology_element_from_catalogue_triangle(
            triangle_id="TRI-0042",
            opening_name="O",
            base_name="B",
            light_name="L",
            opening_lambert_xy=(0.0, 0.0),
            base_lambert_xy=(3000.0, 0.0),
            light_lambert_xy=(0.0, 4000.0),
        )
    )

    dump_path = world.export_topo_dump_xml(str(tmp_path / "TopoDump.xml"))
    dump = dump_path.read_text(encoding="utf-8") if hasattr(dump_path, "read_text") else open(dump_path, encoding="utf-8").read()

    assert 'sourceTriangleId="TRI-0042"' in dump
    assert "<Catalog" not in dump
    assert "ScenarioTriangleSet" not in dump
    assert "modelId" not in dump
    assert "triRank" not in dump
