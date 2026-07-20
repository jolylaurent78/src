# CORE-CLEAN-GROUPS-001 — Audit final des reliquats de groupes legacy

## Périmètre et méthode

Cet audit couvre le simulateur, `TopologyWorld`, l'UI, les helpers
d'interaction, les validateurs/debug et les tests. La persistance XML
(`assembleur_io.py`), le chargement d'anciens fichiers et les formats
historiques sont explicitement exclus. Ils sont mentionnés seulement pour ne
pas confondre leurs clés de compatibilité avec une source de vérité métier.

Les recherches ciblées (`Scenario.groups`, `self.groups`, `group_id`,
`edge_in`, `edge_out`, `_chain_edge_*`, `buildLegacy*`, `legacy_group`,
`topoGroupId` et les validateurs supprimés) ont examiné **80 lignes uniques**
hors `assembleur_io.py`. Ce total contient les homonymes Core et les tests ; il
ne représente pas 80 dépendances legacy.

## Chemin de production automatique

Le chemin constaté est :

```text
AlgoQuadrisParPaires
    -> TopologyWorld
    -> orderedElementIds
    -> buildLegacyLastDrawnFromTopology()
    -> Scenario.last_drawn
```

Il ne reconstruit ni `Scenario.groups`, ni `nodes`, ni `edge_in`/`edge_out`, ni
`last_drawn["group_id"]`. `topoGroupId` reste une projection temporaire obtenue
directement depuis `TopologyWorld.get_group_of_element(topoElementId)`.

## Inventaire synthétique

| Fichier / famille | Rôle actuel | Classification | Action |
| --- | --- | --- | --- |
| `assembleur_core.py` — `TopologyWorld.groups`, DSU, `TopologyGroup` | Stockage et canonicalisation des groupes métier ; éléments, attachments, frontières et dégroupage | Core moderne et légitime | Conservé. |
| `assembleur_sim.py` — `buildLegacyLastDrawnFromTopology()` | Projection graphique de `PlacedTriangle` ; valide chaque `topoElementId` puis projette le groupe Core en `topoGroupId` | UI/simulateur moderne reposant sur le Core | Conservé. |
| `assembleur_tk.py` — MOVE, ROTATE, FLIP, suppression, contours, hit-tests | Résolution par `topoElementId`, `get_group_of_element()`, `getGroupElementIds()` et Boundary Services | UI moderne reposant sur le Core | Conservé. |
| `assembleur_debug.py` | Affiche `getLiveGroupIds()` et `getGroupElementIds()` | Core moderne et légitime | Conservé. |
| `ScenarioAssemblage.groups`, `_group_nodes()`, copies manuelles | Miroir historique utilisé par la persistance XML ; `_group_nodes()` n'a qu'un appel dans `assembleur_io.py` | Compatibilité XML hors périmètre | Conservé. |
| `topoGroupId` dans projections UI | Identifiant Core projeté pour l'UI et la compatibilité ; non utilisé comme groupe UI numéroté | UI moderne reposant sur le Core | Conservé. |
| Clés historiques dans les tests d'import | Vérifient que des dictionnaires anciens sont tolérés puis non réémis | Compatibilité de projection testée | Conservé. |
| `_chain_edge_in`, `_chain_edge_out` dans `test_topology_element_ids.py` | Vérifie l'import tolérant de dictionnaires historiques et l'absence de réémission | Compatibilité de projection testée | Conservé. |

## Reliquats supprimés

Sept tests MIG-GEO devenus obsolètes ont été retirés de
`tests/test_mig_geo001_group_linking.py` :

1. test d'un `_ensure_group_fields()` déjà supprimé ;
2. test de `_merge_legacy_groups()` déjà supprimé ;
3. test qui exigeait encore le remappage de `self.groups` et de
   `last_drawn["group_id"]` après une suppression de groupe Core-first.
4. quatre tests comparatifs qui opposaient explicitement les membres Core à
   `group_id`, `nodes`, `edge_in`/`edge_out` ou à une projection UI historique.

Ils ne couvraient plus une API de production. Les tests Core-first existants
restent en place pour MOVE, ROTATE, FLIP, contours, snapping, dégroupage et
suppression de groupe.

## Vérification des opérations UI

| Opération | Résolution constatée | Dépendance legacy comme source de vérité |
| --- | --- | --- |
| Sélection / MOVE | `topoElementId -> get_group_of_element() -> getGroupElementIds()` | Aucune. |
| ROTATE / FLIP | `_prepare_core_group_operation_members()` et projection indexée par élément | Aucune. |
| Suppression de groupe | Élément sélectionné -> groupe Core -> `removeElementsAndRebuild()` | Aucune. |
| Contours / hit-tests / snapping | `getLiveGroupIds()`, Boundary Services et groupe Core de l'entrée | Aucune. |
| Dégroupage | Résultat de `TopologyWorld.degrouperAtNode()` et IDs Core | Aucune. |

## Occurrences XML exclues

`assembleur_io.py` continue volontairement à lire/écrire des groupes UI,
`group_id`, `nodes`, `edge_in` et `edge_out` afin de prendre en charge les
documents historiques. Ces occurrences ne sont ni modifiées ni comptées comme
consommateurs fonctionnels du simulateur ou des interactions Core-first.

## Conclusion

**Hors persistance XML et compatibilité historique, aucune structure legacy de
groupes n'est encore utilisée comme source de vérité.** Les opérations
fonctionnelles résolvent les groupes exclusivement depuis `TopologyWorld`.

Les seules occurrences restantes sont :

- les structures modernes du Core (`TopologyWorld.groups`, DSU et
  `TopologyGroup`) ;
- la projection `topoGroupId` calculée depuis le Core ;
- les miroirs nécessaires aux formats XML historiques ;
- des assertions de compatibilité d'import qui garantissent la non-réémission
  des clés historiques.
