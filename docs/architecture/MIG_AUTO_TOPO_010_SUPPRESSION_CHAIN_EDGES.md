# MIG-AUTO-TOPO-010 — Suppression des chain edges

## Objet

MIG-AUTO-TOPO-009 avait remplacé la seconde source de vérité de raccordement
par les `TopologyAttachment`. Ce round retire définitivement les reliquats de
validation qui restaient dans les projections de simulation.

Le flux unique est désormais :

```text
buildEdgeChoiceEptsForAutoChain
  -> EdgeChoiceEpts.createTopologyAttachments
  -> TopologyWorld.apply_attachments
  -> buildLegacyGroupsFromTopology + orderedElementIds
  -> Scenario.groups[*].nodes[*].edge_in / edge_out
```

## Éléments supprimés

- `PlacedTriangle.chainEdgeIn` ;
- `PlacedTriangle.chainEdgeOut` ;
- import des clés `_chain_edge_in` / `_chain_edge_out` dans
  `PlacedTriangle.fromLegacyDict()` ;
- export de ces deux clés par `PlacedTriangle.toLegacyDict()` ;
- affectations de ces champs dans la génération des candidats automatiques ;
- `DEBUG_VALIDATE_CORE_CONNECTIONS` ;
- `validateCoreConnectionsAgainstLegacyChain()` et ses deux branchements de
  finalisation de scénario.

## Vérification préalable des usages

La recherche dans `src/` et `tests/` a identifié, avant suppression, seulement
les éléments listés ci-dessus et les tests consacrés à l'oracle temporaire.
Elle n'a révélé aucun lecteur fonctionnel de production hors de cet oracle.

Après suppression, aucune occurrence ne subsiste dans `src/`. Les occurrences
dans les tests se limitent aux assertions d'absence dans les scénarios générés
et aux deux clés littérales d'un test de compatibilité d'import : un
dictionnaire historique les contenant est accepté et les ignore. Elles ne sont
ni stockées ni réémises.

## Tests adaptés et conservés

- création de `PlacedTriangle` sans les champs supprimés ;
- import d'un dictionnaire contenant les deux clés privées, puis vérification
  qu'elles ne sont pas exportées ;
- les quatre raccords vertex-edge LO/BL ;
- vertex-vertex avec ordre des features inversé ;
- clonage des métadonnées `incident_edge_by_element` ;
- détection des attachments contradictoires ;
- scénarios automatiques : absence des deux clés privées dans `last_drawn`.

Les tests fonctionnels de `buildLegacyGroupsFromTopology()` sont conservés :
ils vérifient `edge_in` et `edge_out` à partir des attachments Core.

## Résultats de validation

- ciblé : `python -m pytest -q tests/test_topology_element_ids.py` —
  **29 passed**, un avertissement préexistant d'échappement invalide dans
  `src/assembleur_debug.py` ;
- suite complète : **177 passed, 9 failed, 1 warning**. Les neuf échecs sont
  hors périmètre : un test de cadence runtime, cinq tests de décryptage/angle,
  et trois tests MIG-GEO historiques qui attendent des helpers/une maintenance
  legacy déjà absents de l'arbre de travail ;
- `git diff --check` : aucune anomalie introduite par ce round. Il signale un
  espace final préexistant dans `src/assembleur_tk.py:7535`.

## Anomalies hors périmètre

Aucune anomalie fonctionnelle liée aux chain edges n'a été rencontrée. Les
échecs historiques éventuels de la suite globale sont documentés séparément du
résultat ciblé et ne sont pas corrigés par ce round.

## Confirmation

Le moteur ne contient plus de donnée de chaîne parallèle dans
`PlacedTriangle`. Les connexions legacy `edge_in` et `edge_out` sont
reconstruites exclusivement depuis `TopologyWorld.attachments` et la
chronologie `orderedElementIds`.
