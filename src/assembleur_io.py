"""Services d'entrée/sortie de l'assembleur.

Le module gère la configuration JSON et les scénarios XML v5. Le TopologyWorld est persisté comme source de vérité ; la
projection Canvas et ``ScenarioAssemblage.last_drawn`` sont des caches runtime
reconstruits depuis le Core après chargement.
Persistance (config JSON + scénario XML) isolée du GUI.
Les fonctions prennent 'viewer' en paramètre (duck-typing) pour éviter les imports circulaires.
"""

import os
import json
import datetime as _dt
import xml.etree.ElementTree as ET
import re
import traceback
import numpy as np

from src.assembleur_catalogue import Catalogue
from src.assembleur_scenario import ScenarioHypothesis

CFG_KEY_CHEMINS_BEACON_REF = "cheminsBeaconRefId"


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


def _save_scenario_hypothesis(
    root: ET.Element,
    hypothesis: ScenarioHypothesis,
    catalogue: Catalogue,
) -> None:
    """Sérialise une hypothèse de scénario déjà  validée."""
    hypothesis.validate(catalogue)

    hypothesis_el = ET.SubElement(root, "hypothesis")
    if hypothesis.source_template_id is not None:
        hypothesis_el.set("sourceTemplateId", hypothesis.source_template_id)

    for rank, triangle_id in enumerate(hypothesis.triangle_ids_by_rank, start=1):
        ET.SubElement(
            hypothesis_el,
            "rank",
            {"number": str(rank), "triangleId": triangle_id},
        )


def _load_scenario_hypothesis(
    root: ET.Element,
    catalogue: Catalogue,
) -> ScenarioHypothesis | None:
    """Construit et valide l'hypothèse exclusivement depuis le XML v5."""
    hypothesis_el = root.find("hypothesis")
    if hypothesis_el is None:
        return None

    ranks_by_number: list[str | None] = [None] * 32
    rank_elements = hypothesis_el.findall("rank")
    if len(rank_elements) != 32:
        raise ValueError(
            "L'hypothèse du scénario doit contenir exactement 32 rangs XML."
        )

    for rank_el in rank_elements:
        number_raw = rank_el.get("number")
        triangle_id_raw = rank_el.get("triangleId")
        if number_raw is None:
            raise ValueError("Un rang de l'hypothèse du scénario n'a pas d'attribut number.")
        if triangle_id_raw is None:
            raise ValueError("Un rang de l'hypothèse du scénario n'a pas d'attribut triangleId.")
        try:
            number = int(number_raw)
        except ValueError as exc:
            raise ValueError(
                f"Numéro de rang d'hypothèse invalide : {number_raw!r}."
            ) from exc
        if not 1 <= number <= 32:
            raise ValueError(f"Numéro de rang d'hypothèse hors plage : {number}.")
        if ranks_by_number[number - 1] is not None:
            raise ValueError(f"Rang d'hypothèse dupliqué : {number}.")

        triangle_id = triangle_id_raw.strip()
        if not triangle_id:
            raise ValueError(f"Le rang d'hypothèse {number} a un triangleId vide.")
        ranks_by_number[number - 1] = triangle_id

    if any(triangle_id is None for triangle_id in ranks_by_number):
        raise ValueError("L'hypothèse du scénario contient un rang absent.")

    source_template_id = str(hypothesis_el.get("sourceTemplateId", "") or "").strip() or None
    hypothesis = ScenarioHypothesis(
        triangle_ids_by_rank=[triangle_id for triangle_id in ranks_by_number if triangle_id is not None],
        source_template_id=source_template_id,
    )
    hypothesis.validate(catalogue)
    return hypothesis


def saveScenarioXml(viewer, path: str):
    """
    Sauvegarde XML Core-only v5 :
      - ScenarioHypothesis, vue (zoom/offset), horloge (position + heure),
      - snapshot physique TopologyWorld et états UI persistants.
    """
    scen = viewer._get_active_scenario()
    hypothesis = scen.hypothesis
    if hypothesis is None:
        raise ValueError("saveScenarioXml: ScenarioHypothesis absente du scénario actif.")
    catalogue = viewer.catalogue
    hypothesis.validate(catalogue)
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
    _save_scenario_hypothesis(root, hypothesis, catalogue)
    world.topologyChemins._saveToXml(root)
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

    # écrire
    tree = ET.ElementTree(root)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tree.write(path, encoding="utf-8", xml_declaration=True)


def loadScenarioXml(viewer, path: str):
    """
    Recharge un scénario Core-only v5 :
      - restaure la topologie Core depuis <topoSnapshot encoding="json">,
      - restaure vue et horloge,
      - reconstruit la projection runtime depuis le Core,
      - remplace le scénario actif par un scénario manuel ;
      - vide le cache puis le reconstruit depuis le Core ;
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

    loaded_hypothesis = _load_scenario_hypothesis(root, viewer.catalogue)
    if loaded_hypothesis is None:
        raise ValueError("loadScenarioXml: ScenarioHypothesis absente du scénario XML.")

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

    # 1) vue
    view = root.find("view")
    if view is not None:
        viewer.zoom = float(view.get("zoom", viewer.zoom))
        ox = float(view.get("offset_x", viewer.offset[0] if hasattr(viewer, "offset") else 0.0))
        oy = float(view.get("offset_y", viewer.offset[1] if hasattr(viewer, "offset") else 0.0))
        viewer.offset = np.array([ox, oy], dtype=float)

    # 1bis) map (fond)
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

    # 2) horloge
    clock = root.find("clock")
    if clock is not None:
        viewer._clock_cx = float(clock.get("x", "0"))
        viewer._clock_cy = float(clock.get("y", "0"))
        h = int(clock.get("hour", "0"))
        m = int(clock.get("minute", "0"))
        lbl = clock.get("label", "")
        viewer._clock_state.update({"hour": h, "minute": m, "label": lbl})

    # 3) Core snapshot first; the Canvas projection is rebuilt afterwards.
    from src.assembleur_core import TopologyWorld

    # topoScenarioId non défini si on charge au debut
    sid = getattr(scen, "topoScenarioId", "SCENARIO")
    # Core restored in isolation; the active scenario is replaced only once
    # the physical snapshot has been imported successfully.
    world = TopologyWorld()
    # Le catalogue déjà chargé par le viewer est partagé avec le monde restauré.
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

    # Les références persistées sont recanonisées à partir du nœud physique,
    # car l'identifiant de groupe peut changer pendant l'import du snapshot.
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
    scen.hypothesis = loaded_hypothesis
    scen.source_type = "manual"
    scen.algo_id = None
    scen.orderedElementIds = []
    scen.traversal_direction = None
    scen.status = None
    viewer.canvas_objects.clear()
    scen.last_drawn = []
    viewer._rebuild_active_projection_from_core()
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
