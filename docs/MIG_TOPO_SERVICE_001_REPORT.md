# MIG-TOPO-SERVICE-001 — Services Boundary Core

Date : 2026-07-17  
Statut : migration de la vérité topologique du contour vers le Core terminée.

## Fichiers modifiés

| Fichier | Rôle |
|---|---|
| `src/assembleur_core.py` | Façade Boundary publique et autonome. |
| `src/assembleur_tk.py` | Projection UI des segments Core et migration du snap. |
| `src/assembleur_tk_mixin_frontier.py` | Suppression des reconstructions locales de graphe/contour ; mixin de compatibilité vide. |
| `tests/test_topology_boundary_services.py` | Tests Core Boundary ajoutés. |
| `tests/test_mig_geo001_group_linking.py` | Tests UI de compatibilité mis à jour vers le contrat getter autonome. |

## APIs Core ajoutées et modifiées

### `ensureBoundary(core_group_id)`

Le contrat prend un `core_group_id` déjà canonique. Il vérifie son existence dans le registre Core, calcule le Boundary si le cache conceptuel/géométrique ou les données Boundary sont absents, et ne recalcule pas si le cache est valide. Il ne retourne pas `TopologyBoundaries` ni aucune structure de cache mutable.

Un alias DSU n'est pas accepté par ce contrat. La canonisation reste une responsabilité interne du Core lors de la création ou de la résolution d'un groupe, jamais de l'UI appelante.

### Getters Boundary

`getBoundaryCycle`, `getBoundaryOrientation`, `getBoundaryNeighbors`, `isBoundaryEdge` et `getBoundarySegments` appellent désormais `ensureBoundary()` eux-mêmes. L'appelant n'a plus à faire `computeBoundary()` avant un getter.

### `getIncidentBoundarySegments(core_group_id, concept_node_id)`

Le service garantit d'abord le Boundary, canonise le node dans le Core puis retourne, dans l'ordre déterministe de `getBoundarySegments`, les `BoundarySegment` dont `conceptA` ou `conceptB` est ce node. Les segments sont reconstruits comme valeurs ; le cache interne n'est pas exposé. Une arête interne ne peut pas apparaître puisque la source est le cycle extérieur Core.

## Logique retirée de l'UI

`_update_edge_highlights()` ne calcule plus le contour par union Shapely, ne construit plus un graphe de frontière local et ne recherche plus les demi-arêtes incidentes par adjacence géométrique. Les anciens helpers de `assembleur_tk_mixin_frontier.py` ont été retirés : `_build_boundary_graph`, `_incident_half_edges_at_vertex`, `_incident_half_edges_at_point` et `_normalize_to_outline_granularity`.

L'ancienne union Shapely de contour (`_group_outline_segments` / `_outline_for_item`) a aussi été retirée. L'union Shapely restante dans `_update_edge_highlights()` sert uniquement au calcul de preview anti-chevauchement existant ; elle ne reconstruit plus la vérité topologique du contour.

## Logique conservée dans l'UI

L'UI conserve volontairement :

- identification des sommets mobile et cible ;
- projection des `BoundarySegment` vers les coordonnées `_last_drawn` ;
- angle, score, génération et classement des couples ;
- preview Canvas et instrumentation MIG-DRAG-001A ;
- filtre de chevauchement existant ;
- release et décision de commit.

Elle appelle `getBoundarySegments()` et `getIncidentBoundarySegments()` avec des `core_group_id` déjà canoniques, sans `find_group`, `computeBoundary` ni accès au cache Boundary.

## Tests exécutés

```text
python -m py_compile src/assembleur_core.py src/assembleur_tk.py src/assembleur_tk_mixin_frontier.py tests/test_mig_geo001_group_linking.py tests/test_topology_boundary_services.py
python -m pytest tests/test_topology_boundary_services.py tests/test_mig_geo001_group_linking.py tests/test_topology_comparison.py tests/test_logging_utils.py -q
```

Résultat : **36 passed**.

Les nouveaux tests couvrent : premier calcul lazy, absence de recalcul inutile, recalcul après invalidation, rejet d'un groupe inconnu, triangle isolé (deux segments incidents par sommet), deux triangles attachés (arête commune absente du contour et des incidences), ordre et métadonnées physiques stables.

## Recherche statique finale

Hors `assembleur_core.py` :

| Recherche | Occurrences | Analyse |
|---|---:|---|
| `computeBoundary(` | 1 | `src/assembleur_debug.py:131`, outil de diagnostic explicite, hors UI de snap. |
| `TopologyBoundaries` | 0 | aucune exposition hors Core. |
| `_boundaries` / `.boundaries` | 0 | aucun accès direct à l'objet `TopologyBoundaries` hors Core. |
| `_concept_cache` / `boundaryCycle` | 3 | `src/assembleur_debug.py`, outil de diagnostic explicite ; aucune occurrence UI. |
| `[Boundary] not computed` | 0 | aucun appelant externe dépendant de cet ancien contrat. |
| anciens helpers de graphe UI | 0 | retirés du projet. |

`git diff --check` est propre.

## TopoDump.xml

Le dépôt ne contient pas de `TopoDump.xml` de référence exploitable pour ce scénario et le drag Tk ne peut pas être rejoué automatiquement dans cette validation statique. La comparaison demandée reste donc à effectuer lors du rejeu manuel MIG-DRAG-001A : exporter le dump avant/après et comparer groupes canoniques, membres, nodes, attachments, aliases et connectivité. Aucun format TopoDump n'a été modifié.

## Risques et suite justifiée

Le changement de source du contour est volontaire : le snap s'appuie désormais sur le boundary topologique plutôt que sur une union géométrique tolérante aux micro-écarts. Les tests garantissent le contrat Core, mais le scénario manuel G002/G004 doit confirmer que candidats, transform final et attachments restent identiques. Le seul `computeBoundary` externe restant est un outil de diagnostic ; sa migration éventuelle est indépendante du flux UI et n'est pas nécessaire à cette étape.
