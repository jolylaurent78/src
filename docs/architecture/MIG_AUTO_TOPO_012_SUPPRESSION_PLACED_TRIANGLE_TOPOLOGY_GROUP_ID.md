# MIG-AUTO-TOPO-012 — Suppression de `PlacedTriangle.topologyGroupId`

## Situation initiale

`PlacedTriangle.topologyGroupId` était une copie du groupe Core portée par
chaque triangle projeté. Elle était écrite lors de la création du premier
quadrilatère puis réécrite à la finalisation de la branche. Aucun calcul du
simulateur ne la relisait : elle servait seulement à l'export `last_drawn`.

La source autoritaire est et reste :

```text
TopologyWorld.get_group_of_element(topologyElementId)
```

## Audit des homonymes

L'audit a distingué les quatre catégories suivantes :

- `PlacedTriangle.topologyGroupId` : champ supprimé par ce round ;
- `last_drawn[*]["topoGroupId"]` : projection de compatibilité conservée ;
- `Scenario.groups[*]["topoGroupId"]` : métadonnée du format legacy de
  groupes, conservée ;
- `topoGroupId` des guides, horloges, bindings UI et XML : autres structures
  Core/UI, hors périmètre.

Les lecteurs de projection restants se trouvent notamment dans
`assembleur_io.py` (chargement, contrôle de cohérence et XML) et
`assembleur_tk.py` (ponts de l'UI avec le Core). Aucun de ces consommateurs ne
lit un attribut de `PlacedTriangle`.

## Modifications réalisées

- suppression de `topologyGroupId` du dataclass `PlacedTriangle` ;
- `fromLegacyDict()` accepte encore la clé `topoGroupId`, mais l'ignore ;
- `toLegacyDict()` ne l'exporte plus ;
- suppression des affectations dans `createTopoQuadrilateral()` ;
- suppression de la boucle de finalisation qui recopiait le groupe Core dans
  tous les `PlacedTriangle` ;
- extension de `buildLegacyLastDrawnFromTopology()` : il écrit
  `entry["topoGroupId"]` directement depuis
  `TopologyWorld.get_group_of_element(element_id)`.

La clé est donc régénérée lors de la projection finale, en même temps que
`group_id`, sans stockage intermédiaire dans le triangle.

## Compatibilité

Les dictionnaires et scénarios historiques qui comportent `topoGroupId` restent
acceptés par `PlacedTriangle.fromLegacyDict()`. La valeur importée n'est pas
conservée : dès qu'une projection finale est construite, elle provient du
`TopologyWorld` courant.

Le format UI/XML conserve `last_drawn[*]["topoGroupId"]`; ce round ne modifie
ni le chargement XML, ni `Scenario.groups`.

## Tests

Les tests adaptés vérifient :

- absence de l'attribut sur les objets `PlacedTriangle` ;
- import tolérant d'un ancien `topoGroupId` et export brut sans cette clé ;
- projection de `topoGroupId` depuis le Core ;
- remplacement artificiel de `get_group_of_element()` : la projection finale
  reflète immédiatement la valeur retournée ;
- cohérence conservée des groupes et projections des scénarios automatiques.

Résultats de validation :

- tests ciblés simulation/topologie/groupes : **53 passed**, un avertissement
  préexistant d'échappement invalide dans `src/assembleur_debug.py` ;
- suite complète : **180 passed, 9 failed, 1 warning**. Les neuf échecs sont
  hors périmètre : un test de cadence runtime, cinq tests de décryptage/angle,
  et trois tests MIG-GEO historiques incompatibles avec l'état legacy actuel ;
- `git diff --check` : aucune anomalie introduite par MIG-012. Il signale un
  espace final préexistant dans `src/assembleur_tk.py:7535`.

## Architecture finale

```text
PlacedTriangle.topologyElementId
        |
TopologyWorld.get_group_of_element(...)
        |
buildLegacyLastDrawnFromTopology(...)
        |
last_drawn[*]["topoGroupId"]
```

Le simulateur ne conserve plus aucune information de groupe dans
`PlacedTriangle`; les deux clés legacy finales sont construites à partir du
Core.
