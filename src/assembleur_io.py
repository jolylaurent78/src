"""assembleur_io.py
Persistance (config JSON + scénario XML) isolée du GUI.
Les fonctions prennent 'viewer' en paramètre (duck-typing) pour éviter les imports circulaires.
"""

import os
import json
import datetime as _dt
import xml.etree.ElementTree as ET
import re
import unicodedata
import traceback
import numpy as np
import pandas as pd
from math import hypot, acos, degrees
from src.utils.logging_utils import get_app_logger

CFG_KEY_CHEMINS_BALISE_REF = "cheminsBaliseRefName"
APP_LOGGER = get_app_logger()


class TriangleFileService:
    CFG_KEYS = (
        "lastTriangleCsvIn",
        "lastVillesCsvIn",
        "lastTriangleExcelOut",
        "lastTriangleExcel",
    )
    DMS_RE = re.compile(r"^\s*(\d+)\s+(\d+)\s+(\d+)\s+([NSEW])\s*$")

    def __init__(self, viewer):
        self.viewer = viewer
        self._transformer = None

    def createExcelFromCsv(self, triCsvPath: str, villesCsvPath: str, excelOutPath: str) -> str:
        tri_df = pd.read_csv(triCsvPath, sep=";")
        self._validatedColumns(
            tri_df,
            expected=["Ouverture", "Base", "Lumiere"],
            fileLabel="CSV triangles",
        )
        if tri_df.empty:
            raise ValueError("CSV triangles vide.")

        villes_df = pd.read_csv(villesCsvPath, sep=",")
        self._validatedColumns(
            villes_df,
            expected=["Nom", "Latitude", "Longitude"],
            fileLabel="CSV villes",
            allowExtra=True,
        )
        villes_df = villes_df[["Nom", "Latitude", "Longitude"]]
        if villes_df.empty:
            raise ValueError("CSV villes vide.")

        villesByName = {}
        for i, row in villes_df.iterrows():
            if pd.isna(row["Nom"]) or pd.isna(row["Latitude"]) or pd.isna(row["Longitude"]):
                raise ValueError(f"CSV villes: ligne incomplète à la ligne {i + 2}.")
            name = str(row["Nom"]).strip()
            lat_dms = str(row["Latitude"]).strip()
            lon_dms = str(row["Longitude"]).strip()
            if not name:
                raise ValueError(f"CSV villes: Nom vide à la ligne {i + 2}.")
            if name in villesByName:
                raise ValueError(f"CSV villes: Nom dupliqué '{name}'.")

            (_, _, _, h_lat) = self._parseDmsParts(lat_dms)
            (_, _, _, h_lon) = self._parseDmsParts(lon_dms)
            if h_lat not in ("N", "S"):
                raise ValueError(f"CSV villes: latitude invalide pour '{name}': {lat_dms!r}")
            if h_lon not in ("E", "W"):
                raise ValueError(f"CSV villes: longitude invalide pour '{name}': {lon_dms!r}")
            villesByName[name] = (lat_dms, lon_dms)

        transformer = self._getTransformer()

        projectedByName = {}
        for name, (lat_dms, lon_dms) in villesByName.items():
            lat = self._parseDms(lat_dms)
            lon = self._parseDms(lon_dms)
            try:
                x_m, y_m = transformer.transform(lon, lat)
            except Exception as e:
                raise RuntimeError(f"Projection Lambert93 impossible pour '{name}': {e}") from e
            projectedByName[name] = (float(x_m), float(y_m))

        outRows = []
        for i, row in tri_df.iterrows():
            if pd.isna(row["Ouverture"]) or pd.isna(row["Base"]) or pd.isna(row["Lumiere"]):
                raise ValueError(f"CSV triangles: ligne incomplète à la ligne {i + 2}.")
            o = str(row["Ouverture"]).strip()
            b = str(row["Base"]).strip()
            l = str(row["Lumiere"]).strip()
            if not o or not b or not l:
                raise ValueError(f"CSV triangles: ligne incomplète à la ligne {i + 2}.")
            for city in (o, b, l):
                if city not in projectedByName:
                    raise ValueError(f"Ville introuvable dans CSV villes: '{city}'")

            (xo, yo) = projectedByName[o]
            (xb, yb) = projectedByName[b]
            (xl, yl) = projectedByName[l]

            d_ob = round(hypot(xb - xo, yb - yo) / 1000.0, 2)
            d_ol = round(hypot(xl - xo, yl - yo) / 1000.0, 2)
            d_bl = round(hypot(xl - xb, yl - yb) / 1000.0, 2)

            a_o = round(self._angleDeg(xo, yo, xb, yb, xl, yl), 2)
            a_b = round(self._angleDeg(xb, yb, xo, yo, xl, yl), 2)
            a_l = round(self._angleDeg(xl, yl, xo, yo, xb, yb), 2)
            if abs((a_o + a_b + a_l) - 180.0) > 0.05:
                raise ValueError("Triangle dégénéré / données incohérentes.")

            orientation = self._orientationAtL(xo, yo, xb, yb, xl, yl)

            outRows.append([
                int(i + 1),      # Rang
                o, b, l,         # Ouverture/Base/Lumiere (noms)
                float(a_o),      # Ouverture (angle)
                float(a_b),      # Base (angle)
                float(a_l),      # Lumiere (angle)
                float(d_ob),     # Ouverture-Base (km)
                float(d_ol),     # Ouverture-Lumiere (km)
                float(d_bl),     # Lumiere-Base (km)
                orientation,     # Orientation
            ])

        outColumns = [
            "Rang",
            "Ouverture",
            "Base",
            "Lumiere",
            "Ouverture",
            "Base",
            "Lumiere",
            "Ouverture-Base",
            "Ouverture-Lumiere",
            "Lumiere-Base",
            "Orientation",
        ]
        out_df = pd.DataFrame(outRows, columns=outColumns)

        out_dir = os.path.dirname(excelOutPath)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        tmp_path = excelOutPath + ".tmp.xlsx"
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        try:
            with pd.ExcelWriter(tmp_path, engine="openpyxl") as writer:
                out_df.to_excel(writer, sheet_name="Triangles", index=False)
            os.replace(tmp_path, excelOutPath)
        except Exception:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            raise

        return excelOutPath

    def loadExcel(self, path: str) -> tuple[pd.DataFrame, str]:
        df0 = pd.read_excel(path, header=None)
        header_row = self._findHeaderRow(df0)
        df = pd.read_excel(path, header=header_row)
        dfCanon = self._buildCanonDf(df)
        path_norm = os.path.normpath(os.path.abspath(path))
        return (dfCanon, path_norm)

    def listTriangleExcelFiles(self) -> list[str]:
        files = [
            f for f in os.listdir(self.viewer.data_dir)
            if f.lower().endswith(".xlsx") and not f.lstrip().startswith("~$")
        ]
        files.sort(key=str.lower)
        return files

    def getLastPaths(self) -> dict:
        return {
            k: str(self.viewer.getAppConfigValue(k, "") or "")
            for k in self.CFG_KEYS
        }

    def setLastPaths(
        self,
        *,
        lastTriangleCsvIn: str | None = None,
        lastVillesCsvIn: str | None = None,
        lastTriangleExcelOut: str | None = None,
        lastTriangleExcel: str | None = None,
    ) -> None:
        updates = {
            "lastTriangleCsvIn": lastTriangleCsvIn,
            "lastVillesCsvIn": lastVillesCsvIn,
            "lastTriangleExcelOut": lastTriangleExcelOut,
            "lastTriangleExcel": lastTriangleExcel,
        }
        for key, value in updates.items():
            if value is not None:
                self.viewer.setAppConfigValue(key, value)
        self.viewer.saveAppConfig()

    def _parseDmsParts(self, dms: str):
        if not isinstance(dms, str):
            raise ValueError(f"DMS invalide: {dms!r}")
        m = self.DMS_RE.match(dms)
        if m is None:
            raise ValueError(f"DMS invalide: {dms!r}")
        deg = int(m.group(1))
        minute = int(m.group(2))
        second = int(m.group(3))
        hemisphere = m.group(4)
        if minute < 0 or minute >= 60:
            raise ValueError(f"DMS invalide (minutes): {dms!r}")
        if second < 0 or second >= 60:
            raise ValueError(f"DMS invalide (secondes): {dms!r}")
        return (deg, minute, second, hemisphere)

    def _parseDms(self, dms: str) -> float:
        deg, minute, second, hemisphere = self._parseDmsParts(dms)
        decimal = float(deg) + (float(minute) / 60.0) + (float(second) / 3600.0)
        if hemisphere in ("S", "W"):
            decimal = -decimal
        return decimal

    def _clamp(self, x: float, lo: float, hi: float) -> float:
        return min(hi, max(lo, x))

    def _angleDeg(self, px: float, py: float, ax: float, ay: float, cx: float, cy: float) -> float:
        v1x = ax - px
        v1y = ay - py
        v2x = cx - px
        v2y = cy - py
        n1 = hypot(v1x, v1y)
        n2 = hypot(v2x, v2y)
        if n1 <= 0.0 or n2 <= 0.0:
            raise ValueError("Triangle dégénéré / données incohérentes.")
        c = self._clamp(((v1x * v2x) + (v1y * v2y)) / (n1 * n2), -1.0, 1.0)
        return degrees(acos(c))

    def _orientationAtL(self, xo: float, yo: float, xb: float, yb: float, xl: float, yl: float) -> str:
        vlo_x = xo - xl
        vlo_y = yo - yl
        vlb_x = xb - xl
        vlb_y = yb - yl
        cross = (vlo_x * vlb_y) - (vlo_y * vlb_x)
        if abs(cross) <= 1e-12:
            raise ValueError("Points alignés: orientation indéfinie.")
        if cross > 0.0:
            return "CCW"
        return "CW"

    def _validatedColumns(self, df: pd.DataFrame, expected: list[str], fileLabel: str, allowExtra: bool = False):
        cols = [str(c).replace("\ufeff", "").strip() for c in df.columns.tolist()]
        if allowExtra:
            if cols[:len(expected)] != expected:
                raise ValueError(
                    f"{fileLabel}: entête invalide. Colonnes attendues (préfixe): {', '.join(expected)}"
                )
            return
        if cols != expected:
            raise ValueError(
                f"{fileLabel}: entête invalide. Colonnes attendues: {', '.join(expected)}"
            )

    def _triNorm(self, s: str) -> str:
        raw = str(s)
        norm = "".join(c for c in unicodedata.normalize("NFKD", raw) if not unicodedata.combining(c)).lower()
        return re.sub(r"[^a-z0-9]+", "", norm)

    def _findHeaderRow(self, df0: pd.DataFrame) -> int:
        for i in range(min(12, len(df0))):
            row_norm = [self._triNorm(x) for x in df0.iloc[i].tolist()]
            if any("ouverture" in x for x in row_norm) and any("base" in x for x in row_norm) and any("lumiere" in x for x in row_norm):
                return i
        raise KeyError("Impossible de détecter l'entête ('Ouverture', 'Base', 'Lumière').")

    def _buildCanonDf(self, df: pd.DataFrame) -> pd.DataFrame:
        cmap = {self._triNorm(c): c for c in df.columns}
        col_id = cmap.get("rang") or cmap.get("id")
        col_B = cmap.get("base")
        col_L = cmap.get("lumiere")
        col_OB = cmap.get("ouverturebase")
        col_OL = cmap.get("ouverturelumiere")
        col_BL = cmap.get("lumierebase")
        col_OR = cmap.get("orientation")
        missing = [n for n, c in {
            "Base": col_B,
            "Lumière": col_L,
            "Ouverture-Base": col_OB,
            "Ouverture-Lumière": col_OL,
            "Lumière-Base": col_BL,
        }.items() if c is None]
        if missing:
            raise KeyError("Colonnes manquantes: " + ", ".join(missing))

        out = pd.DataFrame({
            "id": df[col_id] if col_id else range(1, len(df) + 1),
            "B": df[col_B],
            "L": df[col_L],
            "len_OB": pd.to_numeric(df[col_OB], errors="coerce"),
            "len_OL": pd.to_numeric(df[col_OL], errors="coerce"),
            "len_BL": pd.to_numeric(df[col_BL], errors="coerce"),
            "orient": (df[col_OR].astype(str).str.upper().str.strip() if col_OR else pd.Series(["CCW"] * len(df))),
        }).dropna(subset=["len_OB", "len_OL", "len_BL"]).sort_values("id")
        return out.reset_index(drop=True)

    def _getTransformer(self):
        if self._transformer is None:
            try:
                from pyproj import Transformer
            except Exception as e:
                raise RuntimeError("Projection Lambert93 indisponible (pyproj non importable).") from e
            try:
                self._transformer = Transformer.from_crs("EPSG:4326", "EPSG:2154", always_xy=True)
            except Exception as e:
                raise RuntimeError(f"Projection Lambert93 indisponible: {e}") from e
        return self._transformer


def _ioWarn(viewer, where: str, exc: Exception):
    """
    Best-effort logging (console) sans casser l'IHM.
    Active si:
      - viewer.debug_io == True
      - ou variable d'env ASSEMBLEUR_DEBUG_IO=1
    """
    try:
        if getattr(viewer, "debug_io", False) or os.environ.get("ASSEMBLEUR_DEBUG_IO", "") in ("1", "true", "True"):
            msg = f"[IO][WARN] {where}: {type(exc).__name__}: {exc}"
            print(msg)
            # trace utile en dev
            print(traceback.format_exc())
    except Exception:
        # dernier filet: on ne casse jamais sur le logger
        return


def loadAppConfig(viewer):
    """Charge la config JSON (best-effort)."""
    viewer.appConfig = {}
    try:
        path = getattr(viewer, "config_path", "")
        if not path:
            return
        if not os.path.isfile(path):
            return
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            viewer.appConfig = data
    except (OSError, json.JSONDecodeError) as e:
        # Jamais bloquant : si la config est corrompue on repart de zéro.
        _ioWarn(viewer, f"loadAppConfig(path={getattr(viewer, 'config_path', '')})", e)
        viewer.appConfig = {}


def saveAppConfig(viewer):
    """Sauvegarde la config JSON (best-effort)."""
    try:
        path = getattr(viewer, "config_path", "")
        if not path:
            return
        cfg_dir = os.path.dirname(path)
        if cfg_dir:
            os.makedirs(cfg_dir, exist_ok=True)
        # écriture atomique (évite un fichier vide si un souci survient)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(getattr(viewer, "appConfig", {}) or {}, f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)
    except OSError as e:
        _ioWarn(viewer, f"saveAppConfig(path={getattr(viewer, 'config_path', '')})", e)


def getAppConfigValue(viewer, key: str, default=None):
    return (getattr(viewer, "appConfig", {}) or {}).get(key, default)


def setAppConfigValue(viewer, key: str, value):
    try:
        if not hasattr(viewer, "appConfig") or viewer.appConfig is None:
            viewer.appConfig = {}
        viewer.appConfig[key] = value
        viewer.saveAppConfig()
    except Exception as e:
        # Ici on loggue: si ça casse, tu veux le savoir en dev.
        _ioWarn(viewer, f"setAppConfigValue(key={key!r})", e)


def saveScenarioXml(viewer, path: str):
    """
    Sauvegarde XML Core-only v5 :
      - source excel, view (zoom/offset), clock (pos + hm),
      - ids restants (listbox),
      - snapshot physique TopologyWorld et états UI persistants.
    """
    scen = viewer._get_active_scenario()
    world = scen.topoWorld

    topo_tx_orientation = world._topoTxOrientation

    snapshot = world._exportPhysicalSnapshot()
    if not isinstance(snapshot, dict):
        raise ValueError(
            f"saveScenarioXml: snapshot topo invalide ({type(snapshot).__name__}), dict attendu."
        )

    root = ET.Element("scenario", {
        "version": "5",
        "saved_at": _dt.datetime.now().isoformat(timespec="seconds"),
        "topo_tx_orientation": topo_tx_orientation,
    })
    topo_snapshot_el = ET.SubElement(root, "topoSnapshot", {"encoding": "json"})
    topo_snapshot_el.text = json.dumps(snapshot, ensure_ascii=False, separators=(",", ":"))
    world.topologyChemins._saveToXml(root)
    # source
    ET.SubElement(root, "source", {
        "excel": os.path.abspath(viewer.excel_path) if getattr(viewer, "excel_path", None) else ""
    })
    ET.SubElement(root, "view", {
        "zoom": f"{float(getattr(viewer, 'zoom', 1.0)):.6g}",
        "offset_x": f"{float(getattr(viewer, 'offset', (0.0, 0.0))[0]):.6g}",
        "offset_y": f"{float(getattr(viewer, 'offset', (0.0, 0.0))[1]):.6g}",
    })

    # map (fond) : fichier + worldRect + opacité/visibilité + scale
    bg = getattr(viewer, "_bg", None)
    map_path = str(bg.get("path")) if isinstance(bg, dict) and bg.get("path") else ""
    # scale affiché (best-effort)
    map_scale = viewer._bg_compute_scale_factor() if hasattr(viewer, "_bg_compute_scale_factor") else None
    if map_scale is None:
        map_scale = getattr(viewer, "_bg_scale_factor_override", None)

    ET.SubElement(root, "map", {
        "path": os.path.abspath(map_path) if map_path else "",
        "x0": f"{float(bg.get('x0')) if isinstance(bg, dict) and bg.get('x0') is not None else 0.0:.6g}",
        "y0": f"{float(bg.get('y0')) if isinstance(bg, dict) and bg.get('y0') is not None else 0.0:.6g}",
        "w":  f"{float(bg.get('w')) if isinstance(bg, dict) and bg.get('w') is not None else 0.0:.6g}",
        "h":  f"{float(bg.get('h')) if isinstance(bg, dict) and bg.get('h') is not None else 0.0:.6g}",
        "visible": "1" if (getattr(viewer, "show_map_layer", None) and viewer.show_map_layer.get()) else "0",
        "opacity": f"{int(viewer.map_opacity.get()) if hasattr(viewer, 'map_opacity') else 100}",
        "scale": f"{float(map_scale):.6g}" if map_scale is not None else "",
    })

    # clock
    ET.SubElement(root, "clock", {
        "x": f"{float(viewer._clock_cx) if viewer._clock_cx is not None else 0.0:.6g}",
        "y": f"{float(viewer._clock_cy) if viewer._clock_cy is not None else 0.0:.6g}",
        "hour": f"{int(viewer._clock_state.get('hour', 0))}",
        "minute": f"{int(viewer._clock_state.get('minute', 0))}",
        "label": str(viewer._clock_state.get("label", "")),
    })

    # clockRef (persist only when complete)
    if (
        scen.clockRefTopoGroupId is not None
        and scen.clockRefNodeId is not None
        and scen.clockRefEdgeId is not None
    ):
        ET.SubElement(root, "clockRef", {
            "topoGroupId": str(scen.clockRefTopoGroupId),
            "nodeId": str(scen.clockRefNodeId),
            "edgeId": str(scen.clockRefEdgeId),
        })

    # guides (persist per scenario)
    traits = scen.clockAzimuthTraits
    if len(traits) > 0:
        guides_xml = ET.SubElement(root, "guides")
        for g in traits:
            color_hex = str(g["colorHex"]) if ("colorHex" in g) else "#0b3d91"
            guide_attrs = {
                "topoGroupId": str(g["topoGroupId"]),
                "nodeId": str(g["nodeId"]),
                "deltaAzDeg": f"{float(g['deltaAzDeg']):.6g}",
                "colorHex": color_hex,
            }
            edge_ref_id = str(g.get("edgeRefId", "") or "").strip()
            if edge_ref_id:
                guide_attrs["edgeRefId"] = edge_ref_id
            ET.SubElement(guides_xml, "guide", guide_attrs)

    # listbox: ids restants
    lb = ET.SubElement(root, "listbox")
    try:
        for i in range(viewer.listbox.size()):
            txt = viewer.listbox.get(i)
            m = re.match(r"\s*(\d+)\.", str(txt))
            if m:
                ET.SubElement(lb, "tri", {"id": m.group(1)})
    except Exception as e:
        _ioWarn(viewer, "saveScenarioXml(listbox)", e)
    # écrire
    tree = ET.ElementTree(root)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tree.write(path, encoding="utf-8", xml_declaration=True)


def loadScenarioXml(viewer, path: str):
    """
    Recharge un scénario Core-only v5 :
      - tente de recharger le fichier Excel source (mode degrade si absent),
      - restaure la topologie Core depuis <topoSnapshot encoding="json">,
      - restaure vue, horloge et listbox,
      - reconstruit la projection runtime depuis le Core,
      - redessine.
    """
    tree = ET.parse(path)
    root = tree.getroot()
    if root.tag != "scenario":
        raise ValueError("Fichier scenario invalide (balise racine).")

    ver = str(root.get("version", "") or "").strip()
    if ver != "5":
        raise ValueError(f"Unsupported scenario version: expected 5, got {ver}.")
    if root.find("triangles") is not None:
        raise ValueError("Invalid scenario v5: legacy <triangles> section is forbidden.")

    topo_tx_orientation = str(root.get("topo_tx_orientation", "") or "").strip().lower()
    if topo_tx_orientation not in {"cw", "ccw"}:
        raise ValueError("Missing or invalid topo_tx_orientation (expected cw|ccw).")

    topo_snapshot_el = root.find("topoSnapshot")
    topo_snapshot_txt = ""
    if topo_snapshot_el is not None:
        topo_snapshot_txt = str(topo_snapshot_el.text or "").strip()
    if (
        topo_snapshot_el is None
        or str(topo_snapshot_el.get("encoding", "") or "").strip().lower() != "json"
        or not topo_snapshot_txt
    ):
        raise ValueError("Missing topoSnapshot (encoding=json) in scenario v5.")

    snapshot = json.loads(topo_snapshot_txt)
    if not isinstance(snapshot, dict):
        raise ValueError("Missing topoSnapshot (encoding=json) in scenario v5.")

    scen = viewer._get_active_scenario()
    if scen is None:
        raise ValueError("loadScenarioXml: active scenario is required for v5 load.")

    # clockRef optional: clear the runtime reference when it is incomplete.
    clock_ref_el = root.find("clockRef")
    if clock_ref_el is None:
        scen.clockRefTopoGroupId = None
        scen.clockRefNodeId = None
        scen.clockRefEdgeId = None
    else:
        topo_group_id = str(clock_ref_el.get("topoGroupId", "") or "").strip()
        node_id = str(clock_ref_el.get("nodeId", "") or "").strip()
        edge_id = str(clock_ref_el.get("edgeId", "") or "").strip()
        if (not topo_group_id) or (not node_id) or (not edge_id):
            scen.clockRefTopoGroupId = None
            scen.clockRefNodeId = None
            scen.clockRefEdgeId = None
        else:
            scen.clockRefTopoGroupId = topo_group_id
            scen.clockRefNodeId = node_id
            scen.clockRefEdgeId = edge_id

    # guides (v4 compatibility: conversion vers le modèle runtime actuel après restauration du world)
    raw_guides_specs = []
    guides_el = root.find("guides")
    if guides_el is not None:
        for guide_el in guides_el.findall("guide"):
            topo_group_id = str(guide_el.get("topoGroupId", "") or "").strip()
            node_id = str(guide_el.get("nodeId", "") or "").strip()
            edge_ref_id = str(guide_el.get("edgeRefId", "") or "").strip()
            delta_az_raw = str(guide_el.get("deltaAzDeg", "") or "").strip()
            color_hex = str(guide_el.get("colorHex", "") or "").strip()

            if (not topo_group_id) or (not node_id):
                continue
            if not re.fullmatch(r"#[0-9A-Fa-f]{6}", color_hex):
                continue

            try:
                delta_az = float(delta_az_raw)
            except ValueError:
                continue

            raw_guides_specs.append({
                "topoGroupId": topo_group_id,
                "nodeId": node_id,
                "edgeRefId": edge_ref_id,
                "deltaAzDeg": float(delta_az) % 360.0,
                "colorHex": color_hex,
            })

    # 1) Excel source (degraded mode allowed)
    src = root.find("source")
    excel = str(src.get("excel", "") if src is not None else "").strip()
    excel_loaded = False
    if excel:
        if os.path.isfile(excel):
            try:
                viewer.load_excel(excel)
                excel_loaded = True
            except Exception as e:
                _ioWarn(viewer, "loadScenarioXml(excel)", e)
                if hasattr(viewer, "_status_warn"):
                    viewer._status_warn(
                        "Excel source not found; scenario loaded in degraded mode (no full triangle catalog)."
                    )
        else:
            if hasattr(viewer, "_status_warn"):
                viewer._status_warn(
                    "Excel source not found; scenario loaded in degraded mode (no full triangle catalog)."
                )

    # 2) vue
    view = root.find("view")
    if view is not None:
        viewer.zoom = float(view.get("zoom", viewer.zoom))
        ox = float(view.get("offset_x", viewer.offset[0] if hasattr(viewer, "offset") else 0.0))
        oy = float(view.get("offset_y", viewer.offset[1] if hasattr(viewer, "offset") else 0.0))
        viewer.offset = np.array([ox, oy], dtype=float)

    # 2bis) map (fond)
    map_el = root.find("map")
    if map_el is not None:
        map_path = str(map_el.get("path", "") or "").strip()
        if map_path and os.path.isfile(map_path):
            rect = {
                "x0": float(map_el.get("x0", "0") or 0.0),
                "y0": float(map_el.get("y0", "0") or 0.0),
                "w": float(map_el.get("w", "0") or 0.0),
                "h": float(map_el.get("h", "0") or 0.0),
            }
            try:
                viewer._bg_set_map(map_path, rect_override=rect, persist=False)
            except Exception as e:
                _ioWarn(viewer, "loadScenarioXml(map)", e)
                viewer._bg_clear(persist=False)
        else:
            if map_path and hasattr(viewer, "_status_warn"):
                viewer._status_warn(f"Carte introuvable: {map_path}")
            viewer._bg_clear(persist=False)

        if hasattr(viewer, "show_map_layer"):
            viewer.show_map_layer.set(str(map_el.get("visible", "1")) not in ("0", "false", "False"))
        if hasattr(viewer, "map_opacity"):
            viewer.map_opacity.set(int(map_el.get("opacity", "100") or 100))
        sc = str(map_el.get("scale", "") or "").strip()
        if sc:
            viewer._bg_scale_factor_override = float(sc)

    # Etat d'interaction propre
    viewer._sel = {"mode": None}
    viewer._clear_nearest_line()
    viewer._clear_edge_highlights()
    viewer._hide_tooltip()
    viewer._ctx_target_element_id = None
    viewer._edge_choice = None
    viewer._drag_preview_id = None
    viewer.canvas.delete("preview")

    # 3) horloge
    clock = root.find("clock")
    if clock is not None:
        viewer._clock_cx = float(clock.get("x", "0"))
        viewer._clock_cy = float(clock.get("y", "0"))
        h = int(clock.get("hour", "0"))
        m = int(clock.get("minute", "0"))
        lbl = clock.get("label", "")
        viewer._clock_state.update({"hour": h, "minute": m, "label": lbl})

    # 4) listbox (ids restants)
    lb = root.find("listbox")
    if lb is not None:
        remain_ids = [int(e.get("id")) for e in lb.findall("tri") if e.get("id")]
        viewer.listbox.delete(0, "end")

        can_use_df = (
            getattr(viewer, "df", None) is not None
            and not viewer.df.empty
            and (excel_loaded or not excel)
        )
        if can_use_df:
            if remain_ids:
                wanted = set(remain_ids)
                for _, r in viewer.df.iterrows():
                    tid = int(r["id"])
                    if tid in wanted:
                        viewer.listbox.insert("end", f"{tid:02d}. B:{r['B']}  L:{r['L']}")
            else:
                for _, r in viewer.df.iterrows():
                    tid = int(r["id"])
                    viewer.listbox.insert("end", f"{tid:02d}. B:{r['B']}  L:{r['L']}")
        else:
            for tid in remain_ids:
                viewer.listbox.insert("end", f"{int(tid):02d}.")

    # 5) Core snapshot first; the Canvas projection is rebuilt afterwards.
    from src.assembleur_core import TopologyWorld

    sid = str(getattr(scen, "topoScenarioId", "") or "").strip() or "SCENARIO"
    # Core restored in isolation; the active scenario is replaced only once
    # the physical snapshot has been imported successfully.
    world = TopologyWorld()
    # MIG-CAT-001 : rattache la selection globale deja chargee ; le format
    # XML reste strictement inchange.
    if hasattr(viewer, "_attach_catalog_to_world"):
        viewer._attach_catalog_to_world(world)
    world._topoTxOrientation = topo_tx_orientation
    world._importPhysicalSnapshot(snapshot)

    def _canonical_group_from_snapshot(persisted_group_id, node_id, label):
        """Resolve an obsolete XML group id from a physical node in the Core."""
        group_id = str(persisted_group_id or "").strip()
        if world.hasLiveGroup(group_id):
            return group_id
        match = re.fullmatch(r"(T\d+):N\d+", str(node_id or "").strip())
        if match is None or match.group(1) not in world.elements:
            raise ValueError(
                f"{label} references missing Core group {group_id!r} "
                f"and cannot be resolved from node {node_id!r}."
            )
        core_group_id = world.get_group_of_element(match.group(1))
        if not core_group_id:
            raise ValueError(
                f"{label} cannot resolve a Core group from node {node_id!r}."
            )
        return str(core_group_id)

    # MIG-XML-001 -- anciens snapshots peuvent reconstruire un identifiant
    # canonique different. Les references persistantes sont re-resolues par
    # leur noeud physique, jamais par un groupe UI legacy.
    chemins_el = root.find("chemins")
    if chemins_el is not None and str(chemins_el.get("isDefined", "") or "") == "1":
        chemins_el.set(
            "groupId",
            _canonical_group_from_snapshot(
                chemins_el.get("groupId"),
                chemins_el.get("startNodeId"),
                "Chemins",
            ),
        )
    world.topologyChemins._loadFromXml(chemins_el)

    if scen.clockRefTopoGroupId and scen.clockRefNodeId:
        scen.clockRefTopoGroupId = _canonical_group_from_snapshot(
            scen.clockRefTopoGroupId,
            scen.clockRefNodeId,
            "clockRef",
        )

    # The XML contains no projection. A v5 load is always editable manually.
    scen.topoScenarioId = sid
    scen.topoWorld = world
    scen.source_type = "manual"
    scen.algo_id = None
    scen.tri_ids = []
    scen.orderedElementIds = []
    scen.first_triangle_id = None
    scen.traversal_direction = None
    scen.status = None
    viewer.canvas_objects.clear()
    scen.last_drawn = []
    viewer._rebuild_active_projection_from_core()
    viewer.canvas_objects.dump(APP_LOGGER, "chargement XML")
    viewer._update_triangle_listbox_colors()

    scen.clockAzimuthTraits = []
    ref_az = float(getattr(viewer, "_clock_ref_azimuth_deg", 0.0)) % 360.0
    for guide_spec in raw_guides_specs:
        topo_group_id = _canonical_group_from_snapshot(
            guide_spec["topoGroupId"],
            guide_spec["nodeId"],
            "Guide",
        )
        node_id = str(guide_spec["nodeId"])
        delta_az = float(guide_spec["deltaAzDeg"]) % 360.0
        color_hex = str(guide_spec["colorHex"])
        edge_ref_id = str(guide_spec.get("edgeRefId", "") or "").strip()

        if edge_ref_id:
            try:
                node_world = np.array(world.getConceptNodeWorldXY(node_id, topo_group_id), dtype=float)
                other_node_id = world.getEdgeOtherNodeId(topo_group_id, edge_ref_id, node_id)
                other_world = np.array(world.getConceptNodeWorldXY(other_node_id, topo_group_id), dtype=float)
                az_edge_abs = float(viewer._azimuth_world_deg(node_world, other_world))
                az_trait_abs = (az_edge_abs + delta_az) % 360.0
                delta_az = (az_trait_abs - ref_az + 360.0) % 360.0
            except Exception:
                continue

        scen.clockAzimuthTraits.append({
            "topoGroupId": topo_group_id,
            "nodeId": node_id,
            "deltaAzDeg": float(delta_az) % 360.0,
            "colorHex": color_hex,
        })

    # 8) selection et aides reset
    viewer._sel = {"mode": None}
    viewer._clear_nearest_line()
    viewer._clear_edge_highlights()

    # 9) re-appliquer les bindings
    viewer._bind_canvas_handlers()

    # 10) redraw complet
    viewer._redraw_from(viewer._last_drawn)
    viewer._redraw_overlay_only()
    viewer._rebuild_pick_cache()
    viewer._pick_cache_valid = True
    if hasattr(viewer, "refreshCheminTreeView"):
        viewer.refreshCheminTreeView()

    viewer.canvas.focus_set()
    viewer._bind_canvas_handlers()

# ---------- Horloge : test de hit ----------
