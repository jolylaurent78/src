# MIG-AUTO-TOPO-011 — Suppression de `PlacedTriangle.groupId`

## 1. Situation initiale

`PlacedTriangle.groupId` était un reliquat du modèle UI historique. Le
simulateur l'écrivait systématiquement avec la constante `1`, puis
`PlacedTriangle.toLegacyDict()` le transportait vers
`last_drawn[*]["group_id"]`. Ce champ ne représentait ni un groupe Core ni une
information utilisée par les calculs de branche.

Le groupe autoritaire est désormais obtenu par :

```text
TopologyWorld.get_group_of_element(PlacedTriangle.topologyElementId)
```

## 2. Audit des consommateurs

### Lecteurs directs de `PlacedTriangle.groupId`

Aucun lecteur direct de production n'a été trouvé dans `src/` avant la
migration. L'attribut était seulement :

- importé par `PlacedTriangle.fromLegacyDict()` ;
- écrit dans les constructeurs automatiques sous la forme `groupId=1` ;
- réémis par `PlacedTriangle.toLegacyDict()`.

Il n'existait donc aucun comportement du simulateur dépendant de ce champ.

### Lecteurs de `last_drawn[*]["group_id"]`

Ils restent hors du champ de suppression de ce round :

- `src/assembleur_io.py` : sérialisation XML (`group`) et reconstruction de la
  compatibilité de groupes au chargement ;
- `src/assembleur_tk.py` : portions de projection UI et de maintenance legacy
  de `self.groups` ;
- `src/assembleur_debug.py` et tests : diagnostics et compatibilité de
  migration.

Ces consommateurs lisent une **clé de projection dictionnaire**. Ils ne sont
pas des lecteurs de `PlacedTriangle.groupId` et continuent à recevoir la clé
via la nouvelle projection explicite.

### `Scenario.groups`

`Scenario.groups` conserve son format legacy (`id`, `topoGroupId`, `nodes`) et
est toujours construit par `buildLegacyGroupsFromTopology()`. Il n'est ni
supprimé ni modifié fonctionnellement par MIG-011.

### Homonymes hors périmètre

Les identifiants `group_id` / `groupId` de `TopologyGroup`, du DSU,
`TopologyChemins`, des widgets UI et des tests Core ne sont pas des attributs
de `PlacedTriangle`; ils ne sont pas modifiés.

## 3. Modifications réalisées

- suppression du champ `groupId` du dataclass `PlacedTriangle` ;
- suppression des `groupId=1` des constructeurs du simulateur ;
- `fromLegacyDict()` accepte encore un dictionnaire comportant `group_id`, mais
  ignore sa valeur ;
- `toLegacyDict()` ne produit plus directement `group_id` ;
- ajout de `buildLegacyGroupIdMappingFromTopology()` : mapping déterministe
  `core_group_id -> legacy_group_id`, selon la première apparition dans
  `orderedElementIds` ;
- ajout de `buildLegacyLastDrawnFromTopology()` : projection individuelle via
  `toLegacyDict()`, puis injection de `group_id` depuis ce mapping Core ;
- `buildLegacyGroupsFromTopology()` réutilise exactement le même mapping ;
- les deux finalisations de scénarios automatiques (deux triangles et chaînes)
  utilisent `buildLegacyLastDrawnFromTopology()`.

Ainsi, `last_drawn` et `Scenario.groups` partagent la même numérotation sans
dépendre de l'ordre de `TopologyWorld.groups`, des attachments, ni d'une valeur
stockée dans un triangle projeté.

## 4. Compatibilité

Un dictionnaire historique avec `group_id` est accepté par
`PlacedTriangle.fromLegacyDict()` sans erreur. Cette valeur n'est pas mémorisée.
Lorsqu'une projection finale est construite avec un `TopologyWorld` complet,
la valeur est reconstruite depuis le groupe Core canonique et
`orderedElementIds`.

Le XML reste compatible à l'écriture car la projection finale conserve
`last_drawn[*]["group_id"]`. MIG-011 ne modifie pas les chemins de chargement
XML ni `Scenario.groups`.

## 5. Tests

Les tests adaptés couvrent :

- absence de l'attribut `groupId` sur `PlacedTriangle` ;
- import tolérant d'une valeur legacy `group_id` et export brut sans cette clé ;
- deux éléments d'un même groupe Core projetés avec `group_id == 1` ;
- deux groupes Core numérotés selon `orderedElementIds`, même si l'ordre de
  création interne du monde est inverse ;
- cohérence `last_drawn[tid]["group_id"]` ↔ `Scenario.groups[*].nodes[*].tid`
  pour les scénarios automatiques chaînés ;
- conservation des tests de connexions Core et des scénarios à deux triangles.

Résultats de validation :

- tests ciblés simulation/topologie/groupes : **51 passed**, un avertissement
  préexistant d'échappement invalide dans `src/assembleur_debug.py` ;
- suite complète : **178 passed, 9 failed, 1 warning**. Les neuf échecs sont
  hors périmètre : un test de cadence runtime, cinq tests de décryptage/angle,
  et trois tests MIG-GEO historiques incompatibles avec l'état legacy actuel ;
- `git diff --check` : aucune anomalie introduite par MIG-011. Il signale un
  espace final préexistant dans `src/assembleur_tk.py:7535`.

## 6. Architecture finale

```text
TopologyWorld + orderedElementIds + PlacedTriangles
       |
       +-- buildLegacyGroupIdMappingFromTopology
       |       `-- core_group_id -> legacy_group_id déterministe
       |
       +-- buildLegacyLastDrawnFromTopology
       |       `-- last_drawn[*]["group_id"]
       |
       `-- buildLegacyGroupsFromTopology
               `-- Scenario.groups avec la même numérotation
```

`PlacedTriangle` ne stocke plus l'appartenance de groupe. La clé legacy reste
une projection de compatibilité UI/XML construite depuis la topologie Core.
