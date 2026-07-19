# MIG-AUTO-TOPO-014 — Suppression de `group_id` dans `last_drawn`

## Décision

Le simulateur automatique ne produit plus `group_id` dans les dictionnaires
`Scenario.last_drawn`. L'appartenance à un groupe est désormais projetée
uniquement sous la forme temporaire `topoGroupId`, calculée pour chaque entrée
depuis :

```text
entry["topoElementId"]
    -> TopologyWorld.get_group_of_element(...)
    -> entry["topoGroupId"]
```

Il n'existe donc plus de numérotation locale des groupes dans les projections
automatiques. `TopologyWorld` reste la source de vérité.

## Audit préalable

La recherche ciblée a distingué les occurrences suivantes :

| Emplacement | Nature | Décision |
| --- | --- | --- |
| `buildLegacyLastDrawnFromTopology()` | Dernier producteur automatique de `last_drawn["group_id"]` | Migré dans ce ticket. |
| `buildLegacyGroupIdMappingFromTopology()` / `buildLegacyGroupsFromTopology()` | Projection legacy de groupes/nœuds, alors encore conservée pour ses tests | Supprimée ultérieurement par MIG-AUTO-TOPO-015. |
| `assembleur_tk.py` | MOVE, ROTATE, FLIP, sélection/suppression de groupe | Déjà Core-first : `topoElementId` puis `TopologyWorld`. Aucun lecteur de `last_drawn["group_id"]` n'a été migré. |
| `assembleur_io.py` | Chargement/sauvegarde du format XML manuel historique et miroir `self.groups` | Hors périmètre. Ces écritures concernent les scénarios manuels/anciens documents, pas la projection automatique produite ici. |
| tests MIG-GEO | Fixtures et assertions du miroir UI legacy | Conservés : ils ne testent pas la production automatique de `Scenario.last_drawn`. |
| `assembleur_core.py` | Identifiants internes de `TopologyGroup` et DSU | Moderne, hors champ. |

Le changement ne touche ni les structures Core, ni le XML, ni les groupes UI
manuels. En particulier, une lecture résiduelle de `group_id` dans ces chemins
ne justifie pas de réintroduire la clé dans les scénarios automatiques.

## Modifications

### Builder de projection

`buildLegacyLastDrawnFromTopology()` ne construit plus le mapping
`Core group -> entier legacy`. Il valide toujours, pour chaque triangle
projeté :

1. la présence de `topoElementId` ;
2. l'existence de l'élément dans `TopologyWorld` ;
3. l'existence de son groupe Core.

Il produit ensuite `id`, `pts`, `labels`, `mirrored`, `topoElementId` et
`topoGroupId`. La clé `group_id` n'est plus écrite.

`orderedElementIds` est maintenu dans la signature de compatibilité du builder,
mais ne participe plus à une numérotation artificielle.

### Lecteurs métier

Aucun lecteur métier additionnel n'était concerné : les opérations interactives
déjà migrées déterminent leurs membres via les helpers Core, notamment
`TopologyWorld.get_group_of_element()` et `getGroupElementIds()`.

## Compatibilité

`PlacedTriangle.fromLegacyDict()` accepte toujours un dictionnaire historique
qui contient `group_id` ; la valeur est ignorée. La projection suivante depuis
un `TopologyWorld` ne réémet pas cette clé.

Les chemins XML/UI manuels qui maintiennent encore `self.groups` et leur propre
`group_id` sont expressément conservés : leur retrait exige une migration de
format et de projection distincte.

## Tests ajoutés ou adaptés

- scénarios automatiques à deux triangles et chaînés : aucune entrée ne porte
  `group_id`, avec conservation de l'ordre, des identifiants topologiques et du
  groupe Core projeté ;
- monde à groupes Core indépendants : les groupes sont distingués uniquement
  par `topoGroupId` ;
- import d'une projection historique avec `group_id` : toléré, puis non réémis
  par le builder ;
- projection de `topoGroupId` : toujours calculée depuis le Core, sans
  dépendance à `group_id`.

## Résultat architectural

```text
PlacedTriangles + TopologyWorld
    -> buildLegacyLastDrawnFromTopology()
    -> Scenario.last_drawn (sans group_id)
```

La suppression de la clé ne modifie ni la géométrie, ni les attachments, ni la
chronologie `orderedElementIds`.
