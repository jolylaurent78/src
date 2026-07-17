# MIG-DRAG-001A — Rapport d'instrumentation

Date : 2026-07-17  
Statut : instrumentation de diagnostic ; aucun changement fonctionnel intentionnel.

## Fichiers modifiés

| Fichier | Objet |
|---|---|
| `src/assembleur_tk.py` | Logs, déduplication UI de debug et pause visuelle opt-in. |
| `docs/MIG_DRAG_001A_INSTRUMENTATION_REPORT.md` | Présent rapport. |

Ni le Core, ni `EdgeChoiceEpts`, ni les tests ne sont modifiés.

## Activation et emplacement des logs

Le mécanisme existant `MIG_GEO_LOGGER = get_mig_geo_logger()` est réutilisé. Les entrées vont dans `logs/mig_geo.log` et tous les messages ajoutés commencent par `[MIG-DRAG-001]`.

L'activation reprend le flag debug existant `_debug_snap_assist`, basculé par **F10** via `_toggle_debug_snap_assist()`. `_mig_drag_001_enabled()` ne fait que lire ce flag. Hors F10, aucun log MIG-DRAG-001A et aucune pause ne sont produits. Aucun second framework ni raccourci n'a été ajouté.

## Pause de diagnostic

`MIG_DRAG_001_PAUSE_SECONDS = 1.0` est défini dans `assembleur_tk.py`. La pause est exécutée seulement avec F10 actif, après une nouvelle proposition, un changement de signature et le redessin des highlights. `update_idletasks()` force d'abord le rendu Canvas ; `update()` n'est pas utilisé. La pause ne s'exécute jamais pour chaque motion, candidate intermédiaire ou proposition inchangée.

## Logs ajoutés

Lors d'un changement de proposition :

- `PROPOSAL_BEGIN` : indices UI, éléments, sommets et groupes Core mobile/cible.
- `INCIDENT_EDGES` : nodes, extrémités arrondies à 6 décimales et azimut.
- `CANDIDATE` : index, arêtes, score, `overlap_rejected` explicite et décision.
- `SELECTED` : candidate, éléments/nodes/arêtes, `kind` et `tRaw` de `epts`.
- `PROPOSAL_CLEARED` : une seule fois lors de la disparition d'un choix existant.

Les arêtes incidentes sont des segments de preview ; leur propriétaire Core n'est pas disponible avant `buildEdgeChoiceEptsFromBest()`. Le log `SELECTED` complète donc les propriétaires d'élément et d'arête. Avec `debug_skip_overlap_highlight`, la valeur est `overlap_rejected='skipped_debug'` : la trace ne prétend pas avoir exécuté la simulation.

Au release, les logs sont `RELEASE_BEGIN`, `RELEASE_TRANSFORM`, `ATTACHMENTS_PREPARED`, `TRANSACTION_BEGIN`, `TRANSACTION_COMMIT` et `RELEASE_RESULT` (groupe Core final, membres, poses projetées compactes). Le bloc existant n'a pas de chemin explicite de rollback ; cette intervention n'ajoute donc ni `try/except`, ni rollback, ni ligne `TRANSACTION_ROLLBACK` artificielle.

## Déduplication

`self._mig_drag_001_last_signature` est un état UI de debug sans effet sur la sélection. Il contient les deux groupes Core, nodes, arêtes sélectionnées et résultat overlap. Une trace complète n'est écrite qu'au premier choix, lors du changement de meilleure paire/cible/résultat overlap, ou à la disparition de la proposition. Les répétitions de mouvements identiques ne saturent pas le log.

## Procédure de rejeu G002 → G004

1. Charger le scénario : G002 = triangles 1-2 ; G004 = triangles 3-4.
2. Presser **F10** et vérifier `Snap assist: ON` dans la barre d'état.
3. Ouvrir `logs/mig_geo.log` dans l'IDE.
4. Saisir le sommet **L** du triangle 3 et déplacer G004 vers le sommet cible de G002 jusqu'aux deux arêtes rouges.
5. Observer la pause d'environ une seconde pour chaque nouvelle proposition, puis relâcher la souris.
6. Extraire le bloc `[MIG-DRAG-001]` de `PROPOSAL_BEGIN` à `TRANSACTION_COMMIT` / `RELEASE_RESULT`.
7. Presser F10 pour désactiver les traces.

Cette étape ne produit pas de golden file : la trace sera fournie par le rejeu manuel.

## Validation

Exécuté :

```text
python -m py_compile src/assembleur_tk.py src/assembleur_edgechoice.py
python -m pytest tests/test_mig_geo001_group_linking.py tests/test_topology_comparison.py tests/test_logging_utils.py -q
```

Résultat : **31 passed**.

La recherche statique confirme que les nouveaux appels de logger et la pause sont conditionnés par `_mig_drag_001_enabled()` ; l'unique pause est postérieure à `update_idletasks()`.

## Garantie de périmètre

L'instrumentation n'altère ni les candidates, ni leur score, ni le résultat de `simulateOverlapTopologique()`, ni `_edge_choice`, ni `EdgeChoiceEpts`, ni les attachments, ni les transactions, ni les données Core/XML. Elle observe les valeurs produites par le flux existant.
