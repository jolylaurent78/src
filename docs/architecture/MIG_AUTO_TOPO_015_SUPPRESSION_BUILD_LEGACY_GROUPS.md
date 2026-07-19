# MIG-AUTO-TOPO-015 — Suppression des builders de groupes legacy

## Décision

Les fonctions `buildLegacyGroupIdMappingFromTopology()` et
`buildLegacyGroupsFromTopology()` ont été supprimées de `assembleur_sim.py`.
Elles reconstruisaient une structure historique : une numérotation locale de
groupes, des listes `nodes` indexées par projection et des champs
`edge_in` / `edge_out`.

Cette projection n'était plus produite par le simulateur depuis
MIG-AUTO-TOPO-013. Depuis MIG-AUTO-TOPO-014, `last_drawn` ne contient plus de
`group_id`. Il ne restait donc aucun consommateur de production : les seuls
appelants étaient six tests vérifiant le builder lui-même.

## Audit et suppressions

| Élément | Appelants avant | Action |
| --- | --- | --- |
| `buildLegacyGroupIdMappingFromTopology()` | Uniquement `buildLegacyGroupsFromTopology()` | Supprimé. |
| `buildLegacyGroupsFromTopology()` | Six appels directs dans les tests de `test_topology_element_ids.py` | Supprimé avec les tests dédiés. |
| Reconstruction `nodes`, `edge_in`, `edge_out` | Corps du builder supprimé | Supprimée. |
| Numérotation `Core group -> legacy group_id` | Corps du mapping supprimé | Supprimée. |
| `validateLegacyGroupsEquivalent()` | Tests du builder uniquement | Supprimé comme helper orphelin. |

Les occurrences restantes de `Scenario.groups`, `group_id`, `topoGroupId` et
`groups[...]` appartiennent au Core, aux flux UI/XML manuels ou à leur
compatibilité historique. Elles ne sont ni appelantes ni dépendantes des deux
builders supprimés et restent hors périmètre.

## Architecture résultante

```text
AlgoQuadrisParPaires
    -> TopologyWorld
    -> orderedElementIds
    -> buildLegacyLastDrawnFromTopology()
    -> Scenario.last_drawn

Groupes métier : TopologyWorld.get_group_of_element()
                  TopologyWorld.getLiveGroupIds()
                  TopologyWorld.attachments
```

Le simulateur ne reconstruit plus de groupes UI. Les opérations déjà migrées
(MOVE, ROTATE, FLIP, sélection et suppression de groupe) interrogent le Core
à partir de `topoElementId` et ne nécessitent donc aucune adaptation.

## Tests adaptés

- suppression des tests qui validaient exclusivement la projection legacy
  supprimée ;
- ajout d'un test d'API qui vérifie l'absence des deux helpers dans le module ;
- conservation des tests de scénarios automatiques, de projection
  `last_drawn`, de groupes Core, et des interactions déjà Core-first.

## Validation

Les résultats de validation sont consignés avec l'intervention : compilation,
tests ciblés, suite complète et `git diff --check`. Les éventuels échecs
historiques de la suite complète ne concernent pas cette suppression.

## Hors périmètre

Cette migration ne modifie pas `TopologyWorld`, les attachments,
`orderedElementIds`, la géométrie, les opérations d'interaction ou les formats
XML historiques. Elle retire seulement une reconstruction de groupes legacy
qui n'avait plus de chemin de production.
