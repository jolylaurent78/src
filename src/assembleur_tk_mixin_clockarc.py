"""
TriangleViewerClockArcMixin

Ce module est généré pour découper assembleur_tk.py.
"""

from __future__ import annotations
import numpy as np

EPS_WORLD = 1e-6

class TriangleViewerClockArcMixin:
    """Mixin: méthodes extraites de assembleur_tk.py."""
    pass

    def _clock_update_snap_target(self, sx: float, sy: float):
        """Calcule le sommet de triangle le plus proche du curseur (en écran/canvas) et l'affiche en rouge."""
        prev = getattr(self, "_clock_snap_target", None)
        world = self._get_active_scenario().topoWorld
        w = self._screen_to_world(sx, sy)
        v_world = np.array([float(w[0]), float(w[1])], dtype=float)

        # On trouve le Node DSU       
        hit = world.findNearestBoundaryNode(None, v_world)
        if hit is None:
            self._clock_clear_snap_target()
            if prev is not None and self._clock_arc_is_available():
                self._clock_arc_clear_last()
            return
        nodeDsu = str(hit["nodeId"])
        topoGroupId = str(hit["groupId"])

        # On trouve son Vertex
        ref = world.getElementVertexFromAnyNodeId(hit["nodeId"], groupId=hit["groupId"])
        if ref is None:
            raise RuntimeError(f"[ClockSnap] Topo hit but cannot resolve nodeId to element/vertex: {hit}")
        vkey = str(ref["vkey"])
        wbest = tuple(ref["wbest"])

        # On récupère son Tid
        idx = self.getTidForTopoElementId(ref["elementId"])
        if idx is None:
            raise RuntimeError(f"[ClockSnap] Topo elementId '{ref['elementId']}' not found in GUI last_drawn")

        # Si on change de noeud (idx/vkey), l'arc mesuré n'est plus valide => on reset arc + dico
        if prev is not None:
            prev_key = (prev.get("topoGroupId"), prev.get("nodeDsu"))
            new_key = (topoGroupId, nodeDsu)
            if new_key != prev_key and self._clock_arc_is_available():
                self._clock_arc_clear_last()

        self._clock_snap_target = {
            "idx": int(idx), 
            "vkey": str(vkey), 
            "world": np.array(wbest, dtype=float),
            "nodeDsu": nodeDsu,
            "topoGroupId": topoGroupId,
            }

        # Auto-mesure : si on vient de s'accrocher à un nouveau noeud,
        # on calcule automatiquement l'angle entre les 2 segments EXTÉRIEURS
        # incidents à ce noeud (sur le contour du groupe).
        prev_key = (prev.get("topoGroupId"), prev.get("nodeDsu")) if isinstance(prev, dict) else None
        new_key  = (topoGroupId, nodeDsu)

        if new_key is not None and new_key != prev_key:
            self._clock_arc_auto_from_snap_target(self._clock_snap_target , drag = True)

        # Marqueur visuel : un anneau rouge autour du sommet
        self.canvas.delete("clock_snap_target")
        px, py = self._world_to_screen(wbest)
        r = 10  # rayon px fixe
        self.canvas.create_oval(px - r, py - r, px + r, py + r,
                                outline="#FF0000", width=3,
                                fill="", tags="clock_snap_target")
        self.canvas.tag_raise("clock_snap_target")



    def _clock_arc_auto_from_snap_target(self, snap_tgt: dict, drag: bool, 
        prevNodeDsu: str | None = None, nextNodeDsu: str | None = None
    ) -> bool:
        """Auto-mesure d'arc (EXT) UNIQUEMENT via topologie, azimuts en MONDE.

        - drag=True  : pendant le drag -> update overlay (sans persister l'arc).
        - drag=False : au relâché / compas fixé -> persiste la mesure (_clock_arc_last...).

        snap_tgt doit contenir:
        - topoGroupId
        - nodeDsu
        """
        if not self.show_clock_overlay.get():
            return False
        if not isinstance(snap_tgt, dict):
            return False
        if not snap_tgt.get("topoGroupId") or not snap_tgt.get("nodeDsu"):
            return False

        scen = self._get_active_scenario()
        world = scen.topoWorld

        gid = str(snap_tgt["topoGroupId"])
        node = str(snap_tgt["nodeDsu"])

        if (prevNodeDsu is None) ^ (nextNodeDsu is None):
            raise RuntimeError("[ClockArc] prevNodeDsu and nextNodeDsu must be provided together")

        if prevNodeDsu is None:
            # Mode auto (boundary)
            prevNode, nextNode = world.getBoundaryNeighbors(gid, node)
            if prevNode is None or nextNode is None:
                raise RuntimeError(f"[ClockArc] Boundary neighbors not found for node '{node}' in group '{gid}'")
        else:
            # Mode imposé (TreeView triplet)
            prevNode = str(prevNodeDsu)
            nextNode = str(nextNodeDsu)

        # coords monde (concept)
        v_world = np.array(world.getConceptNodeWorldXY(node, gid), dtype=float)
        w1 = np.array(world.getConceptNodeWorldXY(prevNode, gid), dtype=float)
        w2 = np.array(world.getConceptNodeWorldXY(nextNode, gid), dtype=float)

        # centre compas (affichage seulement)
        cx, cy = self._world_to_screen(v_world)
        self._clock_cx = float(cx)
        self._clock_cy = float(cy)

        # azimuts EN MONDE (0°=Nord, horaire)
        az1 = float(self._azimuth_world_deg(v_world, w1))
        az2 = float(self._azimuth_world_deg(v_world, w2))
        angle_deg = float(self._clock_arc_compute_angle_deg(az1, az2))

        # persister (même format que la mesure manuelle)
        self._clock_arc_last = {"az1": az1, "az2": az2, "angle": angle_deg}
        self._clock_arc_last_angle_deg = float(angle_deg)

        self._draw_clock_overlay()
        self._redraw_overlay_only()

        if drag:
            # preview uniquement (ne pas persister)
            return True

        # Filtrer automatiquement le dictionnaire par l'arc courant
        self._ctx_filter_dictionary_by_clock_arc()

        return True


    # ==========================
    # Calibration fond (3 points)
    # ==========================


    def _clock_arc_is_available(self) -> bool:
        """True si une mesure d'arc persistee est disponible (et affichee tant que le compas reste sur le meme noeud)."""
        return (getattr(self, '_clock_arc_last_angle_deg', None) is not None) and isinstance(getattr(self, '_clock_arc_last', None), dict)


    def _clock_arc_handle_click(self, sx: int, sy: int):
        """Gestion des clics gauche dans le mode 'arc d'angle'."""
        if not getattr(self, "_clock_arc_active", False):
            return

        # Snap sur noeud (CTRL = pas de snap)
        sx2, sy2 = self._clock_apply_optional_snap(int(sx), int(sy), enable_snap=True)

        az_abs = float(self._clock_compute_azimuth_deg(sx2, sy2))

        if int(getattr(self, "_clock_arc_step", 0)) == 0:
            self._clock_arc_p1 = (int(sx2), int(sy2), float(az_abs))
            self._clock_arc_step = 1
            self.status.config(text="Mesurer un arc d'angle : sélectionne le point P2 (clic gauche), ESC pour annuler. (Snap noeuds, CTRL = désactiver snap)")
            # Premier rendu immédiat (au même point)
            self._clock_arc_update_preview(int(sx2), int(sy2))
            return

        # step==1 : valider P2 et conclure
        self._clock_arc_p2 = (int(sx2), int(sy2), float(az_abs))
        az1 = float(self._clock_arc_p1[2])
        az2 = float(self._clock_arc_p2[2])
        angle_deg = float(self._clock_arc_compute_angle_deg(az1, az2))

        # Persister la mesure : on l'affichera lors des redraw de l'overlay.
        self._clock_arc_last = {"az1": az1, "az2": az2, "angle": angle_deg}
        self._clock_arc_last_angle_deg = float(angle_deg)
        self._update_compass_ctx_menu_and_dico_state()

        # Sortir du mode interactif en nettoyant uniquement le preview
        self._clock_arc_cancel(silent=True)
        self._redraw_overlay_only()
        self.status.config(text=f"Arc mesuré : {angle_deg:0.0f}°")


    def _clock_arc_update_preview(self, sx: int, sy: int):
        """Met à jour le preview (2 rayons + arc + label) pendant la sélection de P2."""
        if not getattr(self, "_clock_arc_active", False):
            return
        if int(getattr(self, "_clock_arc_step", 0)) != 1:
            return
        if self._clock_arc_p1 is None:
            return
        if not hasattr(self, "canvas") or self.canvas is None:
            return

        cx = float(getattr(self, "_clock_cx", 0.0))
        cy = float(getattr(self, "_clock_cy", 0.0))
        R  = float(getattr(self, "_clock_R", getattr(self, "_clock_radius", 69)))

        # Snap P2 si activé
        sx2, sy2 = self._clock_apply_optional_snap(int(sx), int(sy), enable_snap=True)
        az2 = float(self._clock_compute_azimuth_deg(sx2, sy2))
        az1 = float(self._clock_arc_p1[2])

        # 2 rayons (centre->P1 et centre->P2)
        x1, y1 = int(self._clock_arc_p1[0]), int(self._clock_arc_p1[1])
        if self._clock_arc_line1_id is None:
            self._clock_arc_line1_id = self.canvas.create_line(
                cx, cy, x1, y1, width=2, dash=(4, 3),
                fill="#202020", tags=("clock_overlay", "clock_arc_preview")
            )
        else:
            self.canvas.coords(self._clock_arc_line1_id, cx, cy, x1, y1)

        if self._clock_arc_line2_id is None:
            self._clock_arc_line2_id = self.canvas.create_line(
                cx, cy, sx2, sy2, width=2, dash=(4, 3),
                fill="#202020", tags=("clock_overlay", "clock_arc_preview")
            )
        else:
            self.canvas.coords(self._clock_arc_line2_id, cx, cy, sx2, sy2)

        # Arc (plus petit arc entre az1 et az2)
        start_deg_tk, extent_deg_tk, angle_deg, mid_az = self._clock_arc_compute_tk_arc(az1, az2)

        bbox = (cx - R, cy - R, cx + R, cy + R)
        if self._clock_arc_arc_id is None:
            self._clock_arc_arc_id = self.canvas.create_arc(
                *bbox,
                start=float(start_deg_tk),
                extent=float(extent_deg_tk),
                style="arc",
                width=2,
                outline="#202020",
                tags=("clock_overlay", "clock_arc_preview")
            )
        else:
            self.canvas.coords(self._clock_arc_arc_id, *bbox)
            self.canvas.itemconfig(self._clock_arc_arc_id, start=float(start_deg_tk), extent=float(extent_deg_tk))

        # Texte angle, placé près du milieu de l'arc (légèrement à l'extérieur)
        tx, ty = self._clock_point_on_circle(mid_az, R * 1.08)
        label = f"{angle_deg:0.0f}°"
        if self._clock_arc_text_id is None:
            self._clock_arc_text_id = self.canvas.create_text(
                tx, ty, text=label, anchor="center",
                fill="#202020", font=("Arial", 12, "bold"),
                tags=("clock_overlay", "clock_arc_preview")
            )
        else:
            self.canvas.itemconfig(self._clock_arc_text_id, text=label)
            self.canvas.coords(self._clock_arc_text_id, tx, ty)


    def _clock_arc_cancel(self, silent: bool=False):
        if not getattr(self, "_clock_arc_active", False):
            return
        self._clock_arc_active = False
        self._clock_arc_step = 0
        self._clock_arc_p1 = None
        self._clock_arc_p2 = None

        # Nettoyage items (pas de try/except silencieux : on veut voir les erreurs)
        for attr in ("_clock_arc_line1_id", "_clock_arc_line2_id", "_clock_arc_arc_id", "_clock_arc_text_id"):
            item_id = getattr(self, attr, None)
            if item_id is not None and getattr(self, "canvas", None):
                self.canvas.delete(item_id)
            setattr(self, attr, None)

        self._clock_clear_snap_target()
        if not silent:
            self.status.config(text="Mesure d'arc annulée.")


    def _clock_arc_clear_last(self):
        """Efface la dernière mesure persistée (appelé notamment quand le compas bouge)."""
        # Si le dico était filtré sur la base de cet arc, on doit annuler le filtrage
        # dès que l'arc n'est plus disponible (changement de noeud compas).
        if bool(getattr(self, "_dico_filter_active", False)) or (getattr(self, "_dico_filter_ref_angle_deg", None) is not None):
            # Réutilise l'action standard de l'app (menu simulation "Annuler filtrage")
            if hasattr(self, "_simulation_cancel_dictionary_filter"):
                self._simulation_cancel_dictionary_filter()
            else:
                # fallback minimal (au cas où)
                self._dico_filter_active = False
                self._dico_filter_ref_angle_deg = None
                self._dico_clear_filter_styles()

        self._clock_arc_last = None
        self._clock_arc_last_angle_deg = None
        self._update_compass_ctx_menu_and_dico_state()





    def _clock_arc_draw_last(self, cx: float, cy: float, R: float):
        """Dessine la dernière mesure persistée (si présente) dans l'overlay."""
        last = getattr(self, "_clock_arc_last", None)
        if not isinstance(last, dict):
            return
        if not getattr(self, "canvas", None):
            return

        az1 = float(last.get("az1"))
        az2 = float(last.get("az2"))
        start_deg_tk, extent_deg_tk, angle_deg, mid_az = self._clock_arc_compute_tk_arc(az1, az2)

        # 2 rayons (centre->az1 et centre->az2)
        x1, y1 = self._clock_point_on_circle(az1, R)
        x2, y2 = self._clock_point_on_circle(az2, R)
        self.canvas.create_line(
            cx, cy, x1, y1, width=2, dash=(4, 3),
            fill="#202020", tags=("clock_overlay", "clock_arc_persist")
        )
        self.canvas.create_line(
            cx, cy, x2, y2, width=2, dash=(4, 3),
            fill="#202020", tags=("clock_overlay", "clock_arc_persist")
        )

        # Arc
        bbox = (cx - R, cy - R, cx + R, cy + R)
        self.canvas.create_arc(
            *bbox,
            start=float(start_deg_tk),
            extent=float(extent_deg_tk),
            style="arc",
            width=2,
            outline="#202020",
            tags=("clock_overlay", "clock_arc_persist"),
        )

        # Texte angle
        tx, ty = self._clock_point_on_circle(mid_az, R * 1.08)
        self.canvas.create_text(
            tx, ty,
            text=f"{float(angle_deg):0.0f}°",
            anchor="center",
            fill="#202020",
            font=("Arial", 12, "bold"),
            tags=("clock_overlay", "clock_arc_persist"),
        )


    def _clock_arc_compute_angle_deg(self, az1: float, az2: float) -> float:
        """Retourne le plus petit angle (0..180) entre deux azimuts."""
        d = (float(az2) - float(az1)) % 360.0
        if d > 180.0:
            d = 360.0 - d
        return float(d)


    def _clock_arc_compute_tk_arc(self, az1: float, az2: float):
        """Prépare les paramètres Tk (start/extent) pour dessiner le plus petit arc entre az1 et az2.

        Retourne (start_deg_tk, extent_deg_tk, angle_deg, mid_az).
        - start/extent sont dans le repère Tk (0° à 3h, CCW+)
        - mid_az est l'azimut (0°=Nord, horaire) au milieu de l'arc choisi, utile pour placer le label.
        """
        a1 = float(az1) % 360.0
        a2 = float(az2) % 360.0

        # delta horaire (cw) de a1 vers a2
        d_cw = (a2 - a1) % 360.0
        d_ccw = (a1 - a2) % 360.0  # delta si on va anti-horaire

        # Choisir le plus petit arc
        if d_cw <= d_ccw:
            # arc horaire de taille d_cw : Tk extent doit être négatif
            angle = d_cw
            start_az = a1
            mid_az = (a1 + angle * 0.5) % 360.0
            start_tk = (90.0 - start_az) % 360.0
            extent_tk = -angle
        else:
            # arc anti-horaire de taille d_ccw : Tk extent positif
            angle = d_ccw
            start_az = a1
            mid_az = (a1 - angle * 0.5) % 360.0
            start_tk = (90.0 - start_az) % 360.0
            extent_tk = angle

        return (float(start_tk), float(extent_tk), float(angle), float(mid_az))


