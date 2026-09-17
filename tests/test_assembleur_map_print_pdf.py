from __future__ import annotations

from PIL import Image
import fitz

from src.assembleur_map_print import AssembleurPrintMap, AssembleurPrintSettings, AssembleurPrintSnapshot, AssembleurPrintViewport
from src.assembleur_map_print_pdf import export_print_pdf


def test_export_a4_portrait_and_landscape(tmp_path):
    snapshot = AssembleurPrintSnapshot("Assemblage", AssembleurPrintMap(Image.new("RGB", (10, 10), "blue"), 0, 0, 10, 10), (), ())
    viewport = AssembleurPrintViewport(0, 0, 10, 10)
    portrait = export_print_pdf(tmp_path / "portrait.pdf", snapshot, AssembleurPrintSettings(title="Titre court"), viewport)
    landscape = export_print_pdf(tmp_path / "landscape.pdf", snapshot, AssembleurPrintSettings(orientation="landscape", title="Un titre volontairement très long pour vérifier la réduction automatique"), viewport)
    assert portrait.stat().st_size > 0
    assert landscape.stat().st_size > 0
    portrait_page = fitz.open(portrait)[0].rect
    landscape_page = fitz.open(landscape)[0].rect
    assert portrait_page.height > portrait_page.width
    assert landscape_page.width > landscape_page.height
