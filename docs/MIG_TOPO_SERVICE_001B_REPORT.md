# MIG-TOPO-SERVICE-001B — Utilisation explicite des services Boundary

Date : 2026-07-17  
Statut : clarification de frontière terminée ; aucun changement de règle de snap.

## Modification réalisée

Les wrappers UI `_group_outline_segments_topo()` et `_incident_boundary_segments_topo()` ont été supprimés. `_update_edge_highlights()` consulte désormais le Core dans son propre corps, puis délègue uniquement la conversion visuelle à `_project_boundary_segments()`.

Extrait final :

```python
mob_boundary_segments = world.getBoundarySegments(core_gid_m)
tgt_boundary_segments = world.getBoundarySegments(core_gid_t)
mob_incident_boundary_segments = world.getIncidentBoundarySegments(core_gid_m, mAId)
tgt_incident_boundary_segments = world.getIncidentBoundarySegments(core_gid_t, tAId)

mob_outline = self._project_boundary_segments(core_gid_m, mob_boundary_segments)
tgt_outline = self._project_boundary_segments(core_gid_t, tgt_boundary_segments)
m_inc_raw = self._project_boundary_segments(core_gid_m, mob_incident_boundary_segments)
t_inc_raw = self._project_boundary_segments(core_gid_t, tgt_incident_boundary_segments)
```

Les `mAId`/`tAId` restent atomiques dans l'UI. Ils sont canonisés à l'intérieur de `getIncidentBoundarySegments()` par le Core ; aucun `find_node()` ou `find_group()` n'a été ajouté dans l'UI.

## Fichiers modifiés

| Fichier | Rôle |
|---|---|
| `src/assembleur_tk.py` | appels Boundary explicites dans le snap et dans le rendu des contours ; suppression des wrappers. |
| `tests/test_mig_geo001_group_linking.py` | test de rendu adapté au getter Core direct. |
| `docs/MIG_TOPO_SERVICE_001B_REPORT.md` | présent rapport. |

Le Core et l'algorithme de sélection ne sont pas modifiés par cette passe.

## Rôle conservé de `_project_boundary_segments()`

Ce helper UI ne reçoit que des `BoundarySegment` déjà filtrés par le Core. Il résout `elementId` vers l'entrée `_last_drawn`, convertit les nodes physiques `O/B/L`, interpole `t0/t1` et retourne des couples de points monde pour l'affichage et les calculs UI. Il ne calcule pas de Boundary, ne filtre aucune arête interne, ne parcourt ni attachments ni cache Boundary, et n'appelle ni `computeBoundary()` ni `ensureBoundary()`.

## Invariants de snap préservés

Les lignes d'orientation et de classement restent inchangées : `m_oriented`, `t_oriented`, les deux boucles de candidates, le score angulaire, l'anti-chevauchement, le meilleur candidat, le preview, le release et les attachments sont conservés. Seule la provenance de `m_inc_raw` et `t_inc_raw` est explicitée : services Core puis projection UI.

## Recherches statiques

Résultats exécutés après migration :

| Recherche | Résultat exact |
|---|---|
| `_group_outline_segments_topo` dans `src/` et `tests/` | 0 occurrence |
| `_incident_boundary_segments_topo` dans `src/` et `tests/` | 0 occurrence |
| `getBoundarySegments(` dans `src/assembleur_tk.py` | 3 occurrences : rendu contours, `_update_edge_highlights`, contexte chemin |
| `getIncidentBoundarySegments(` dans `src/assembleur_tk.py` | 2 occurrences, toutes deux dans `_update_edge_highlights` |
| `ensureBoundary(` dans `src/assembleur_tk.py` | 0 occurrence |
| `computeBoundary(` dans `src/assembleur_tk.py` | 0 occurrence |

## Tests exécutés

```text
python -m py_compile src/assembleur_core.py src/assembleur_tk.py
python -m pytest tests/test_topology_boundary_services.py -q
python -m pytest tests/test_mig_geo001_group_linking.py -q
python -m pytest tests/test_topology_comparison.py tests/test_logging_utils.py -q
```

Résultats : **5 passed**, **18 passed**, puis **12 passed**. `git diff --check` est propre.
