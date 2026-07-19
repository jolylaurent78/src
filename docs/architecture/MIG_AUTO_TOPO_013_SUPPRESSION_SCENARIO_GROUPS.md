# MIG-AUTO-TOPO-013 — Suppression de `Scenario.groups` dans le simulateur

## Objectif et décision

Le simulateur automatique ne produit plus `Scenario.groups`. Cette structure
était une projection legacy construite par `buildLegacyGroupsFromTopology()` à
la finalisation des scénarios deux-triangles et des scénarios chaînés.

Pendant la simulation, la représentation de groupe autoritaire est désormais
exclusivement :

```text
TopologyWorld + orderedElementIds
```

`last_drawn` reste une projection graphique/compatibilité. Il contient encore
`group_id` et `topoGroupId`, tous deux calculés depuis le Core par
`buildLegacyLastDrawnFromTopology()`.

## Audit des consommateurs

### `buildLegacyGroupsFromTopology()`

Avant ce round, les seuls appels de production étaient les deux finalisations
de `AlgoQuadrisParPaires.run()` dans `src/assembleur_sim.py`. Ils ont été
supprimés.

Après ce round, la fonction restait définie et utilisée uniquement par ses
tests unitaires (`tests/test_topology_element_ids.py`) : reconstruction
edge-edge, vertex-edge, vertex-vertex, clonage et conflits. Elle a été
supprimée ultérieurement par MIG-AUTO-TOPO-015.

### `Scenario.groups`

- `ScenarioAssemblage` conserve l'attribut initialisé à `{}` pour la
  compatibilité des autres scénarios ;
- le mode manuel, l'UI et le XML possèdent encore leurs propres chemins legacy
  de groupes ; ils sont hors périmètre ;
- `_set_active_scenario()` rattache la projection `last_drawn` et ne lit pas
  `scen.groups` ; aucun adaptateur UI n'a donc été requis.

Les tests de comparaison topologique déjà existants couvrent explicitement des
scénarios avec `groups = None`, ce qui confirme que les parcours Core concernés
ne dépendent plus de cette projection.

## Modifications réalisées

- retrait de `scen.groups = buildLegacyGroupsFromTopology(...)` dans la
  finalisation des scénarios automatiques à deux triangles ;
- retrait de la même construction dans les finalisations de branches chaînées ;
- adaptation des tests de simulation : ils vérifient désormais `groups == {}`
  tout en conservant la vérification de `last_drawn`, de l'ordre de construction
  et des identifiants Core.

## Architecture obtenue

```text
AlgoQuadrisParPaires
  -> TopologyWorld
  -> orderedElementIds
  -> buildLegacyLastDrawnFromTopology
  -> Scenario.last_drawn

Scenario.groups = {}  # non produit par le simulateur automatique
```

Les connexions et l'appartenance de groupe ne sont plus reconstruits dans une
liste de noeuds legacy par le simulateur. Toute décision métier future doit
interroger `TopologyWorld`.

## Validation

- tests ciblés simulation/topologie/groupes : **53 passed**, un avertissement
  préexistant d'échappement invalide dans `src/assembleur_debug.py` ;
- suite complète : **180 passed, 9 failed, 1 warning**. Les neuf échecs sont
  hors périmètre : un test de cadence runtime, cinq tests de décryptage/angle,
  et trois tests MIG-GEO historiques incompatibles avec l'état legacy actuel ;
- `git diff --check` : aucune anomalie introduite par MIG-013. Il signale un
  espace final préexistant dans `src/assembleur_tk.py:7535`.

## Impact et suites

Cette migration ne supprime ni `Scenario.groups` de la classe, ni les groupes
du mode manuel, ni les groupes XML/UI. Elle supprime seulement leur **production
automatique**. Le builder de groupes qui restait temporairement présent a été
retiré ultérieurement par MIG-AUTO-TOPO-015 avec ses consommateurs unitaires.
