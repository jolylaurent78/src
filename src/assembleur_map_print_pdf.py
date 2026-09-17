"""PDF backend for the independent Assembleur print workflow."""

from __future__ import annotations

from pathlib import Path

from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas

from src.assembleur_map_print import (
    AssembleurPrintSettings,
    AssembleurPrintSnapshot,
    AssembleurPrintViewport,
    render_print_page_raster,
)


def export_print_pdf(path: str | Path, snapshot: AssembleurPrintSnapshot, settings: AssembleurPrintSettings, viewport: AssembleurPrintViewport, *, dpi: int = 300) -> Path:
    """Write one A4 page using the same raster page renderer as the preview."""
    destination = Path(path)
    page = render_print_page_raster(snapshot, settings, viewport, dpi)
    width_mm, height_mm = settings.page_size_mm
    points_per_mm = 72.0 / 25.4
    pdf = canvas.Canvas(str(destination), pagesize=(width_mm * points_per_mm, height_mm * points_per_mm))
    pdf.drawImage(ImageReader(page), 0, 0, width=width_mm * points_per_mm, height=height_mm * points_per_mm)
    pdf.showPage()
    pdf.save()
    if not destination.is_file() or destination.stat().st_size == 0:
        raise OSError(f"Le PDF n'a pas été créé : {destination}")
    return destination
