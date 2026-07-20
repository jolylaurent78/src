# MIG-AUTO-TOPO-009.5 — Audit des projections `PlacedTriangle`

Date de l'audit : 2026-07-20.  
Périmètre : lecture seule du dépôt, avec recherche des accès typés à
`PlacedTriangle` et des clés émises dans `ScenarioAssemblage.last_drawn`.
Ce document ne modifie ni l'algorithme de simulation, ni les projections, ni
la persistance.

## 1. Résumé exécutif

`PlacedTriangle` est maintenant le conteneur de projection d'une branche du
simulateur. Il ne porte pas la vérité topologique : celle-ci est dans
`TopologyWorld` et ses `TopologyAttachment`. Il reste cependant le point de
passage qui transforme l'état d'une branche en dictionnaires `last_drawn`.

Constats principaux :

- `triangleId`, `points` et `topologyElementId` ont encore des lecteurs directs
  dans le moteur ou `assembleur_edgechoice`; ils sont indispensables au contrat
  actuel de projection et à la construction des candidats.
- `labels` et `mirrored` ne sont plus relus directement comme attributs de
  `PlacedTriangle` après leur construction. Ils sont toutefois exportés vers
  `last_drawn`, puis consommés par le rendu, l'interaction et le XML : ils ne
  sont donc pas supprimables sans migration de cette projection.
- `groupId` n'a aucun lecteur direct sur `PlacedTriangle` dans le code de
  production. Il est écrit avec la constante `1`, puis transporté vers
  `last_drawn["group_id"]`. Cette clé est encore lue par la persistance XML,
  des chemins legacy de `self.groups`, du debug et des tests. Le champ typé est
  donc un producteur de compatibilité, pas une donnée métier du simulateur.
- `topologyGroupId` n'a aucun lecteur direct sur `PlacedTriangle`. Sa valeur
  est une projection redondante du Core, obtenue par
  `TopologyWorld.get_group_of_element(elementId)`. Elle reste émise dans
  `last_drawn["topoGroupId"]`, où elle est encore consommée par le chargement,
  les contrôles de cohérence XML et certains états UI.
- `chainEdgeIn` et `chainEdgeOut` ne participent plus à la reconstruction des
  groupes : depuis MIG-AUTO-TOPO-009,
  `buildLegacyGroupsFromTopology()` lit uniquement les attachments. Leurs
  seules lectures de production sont l'oracle debug
  `validateCoreConnectionsAgainstLegacyChain()`. Ce sont les meilleurs
  candidats au nettoyage suivant, après une décision explicite sur le maintien
  du validateur temporaire.

Le risque principal est de confondre « aucun lecteur direct de l'attribut
Python » avec « aucun consommateur du champ projeté ». `toLegacyDict()` rend
les deux niveaux différents : certains champs sont devenus passifs dans le
simulateur, mais actifs dans l'UI et le XML une fois projetés.

## 2. Méthode et frontières de l'audit

Les recherches ont été faites sur `src/` et `tests/` :

1. accès typés `PlacedTriangle` (`.triangleId`, `.points`, `.labels`,
   `.mirrored`, `.topologyElementId`, `.groupId`, `.topologyGroupId`,
   `.chainEdgeIn`, `.chainEdgeOut`), ainsi que tous les constructeurs ;
2. sérialisation/désérialisation dans `PlacedTriangle.fromLegacyDict()` et
   `PlacedTriangle.toLegacyDict()` (`src/assembleur_sim.py:52-84`) ;
3. consommateurs des clés projetées : `id`, `pts`, `labels`, `mirrored`,
   `topoElementId`, `group_id`, `topoGroupId`, `_chain_edge_in` et
   `_chain_edge_out` ;
4. exports, UI, debug et tests identifiés par fichier et par chaîne d'appel.

Les occurrences de `TopologyChemins.groupId`, `TopologyGroup.group_id` et
autres identifiants Core homonymes ne sont pas attribuées à
`PlacedTriangle.groupId` : elles sont hors de cette structure.

## 3. Tableau de synthèse

| Champ | Écrit par | Lu par | Source métier actuelle | Projection | Peut être supprimé ? | Commentaire |
|---|---|---|---|---|---|---|
| `triangleId` | Constructeurs simulation ; `fromLegacyDict()` | recherche `findByTriangleId()` ; EdgeChoice ; tests | identifiant catalogue/UI | `id` | Non | Sert aussi de propriétaire de trace dans EdgeChoice et de clé UI/XML. |
| `points` | Constructeurs de branches | EdgeChoice ; clone/test ; export `pts` | pose/projection géométrique de branche | `pts` | Non | L'UI travaille encore sur ces coordonnées monde. |
| `labels` | Constructeurs depuis le catalogue ; `fromLegacyDict()` | pas de lecteur typé après construction ; rendu après projection | catalogue / `TopologyElement.vertex_labels` | `labels` | Pas maintenant | Redondant avec le Core à terme, mais encore requis par le rendu legacy. |
| `mirrored` | Constructeurs ; `fromLegacyDict()` | pas de lecteur typé après construction ; UI et XML après projection | pose Core (`TopologyElement.pose.mirrored`) | `mirrored` | Pas maintenant | Transport de projection ; l'UI lit et modifie encore ce flag. |
| `topologyElementId` | `createTopoQuadrilateral()` ; helper d'affectation | builder, recherches `PlacedTriangles`, EdgeChoice, overlap, finalisation, validateur | `TopologyElement.element_id` | `topoElementId` | Non | Identité de liaison indispensable Core ↔ projection. |
| `groupId` | Tous les constructeurs auto (`1`) ; `fromLegacyDict()` | aucune lecture typée ; `toLegacyDict()` | aucune dans le simulateur ; legacy UI | `group_id` | Pas avant XML/`Scenario.groups` | Le champ typé est passif, mais sa projection est encore consommée. |
| `topologyGroupId` | bootstrap, création de quadrilatère, finalisation | aucune lecture typée ; `toLegacyDict()` | `TopologyWorld.get_group_of_element()` | `topoGroupId` | Pas avant migration IO/UI | Projection redondante, notamment réécrite depuis le Core à la finalisation. |
| `chainEdgeIn` | génération de candidat chaîné | validateur debug ; export `_chain_edge_in`; tests | aucune après MIG-009 : Core attachments | `_chain_edge_in` | Oui, après retrait de l'oracle | Ne doit plus décider du résultat. |
| `chainEdgeOut` | génération de candidat chaîné | validateur debug ; export `_chain_edge_out`; tests | aucune après MIG-009 : Core attachments | `_chain_edge_out` | Oui, après retrait de l'oracle | Même statut que `chainEdgeIn`. |

## 4. Analyse détaillée

### 4.1 `triangleId`

**Description.** Identifiant historique du triangle/cataloque, distinct de
`topologyElementId`. Il devient la clé `id` de la projection.

**Écritures.**

- `PlacedTriangle.fromLegacyDict()` lit `entry["id"]`
  (`src/assembleur_sim.py:55-64`).
- Les constructeurs de la simulation initiale, des quadrilatères mobiles et du
  cas à deux triangles renseignent `triangleId` :
  `src/assembleur_sim.py:944-957`, `1005-1019`, `1158-1179`, `1263-1287`.

**Lectures directes.**

- `PlacedTriangles.last()` et `findByTriangleId()` exposent/recherchent cette
  identité (`103-124`) ;
- `buildEdgeChoiceEptsForAutoChain()` contrôle l'identité puis l'inscrit dans
  les métadonnées `src_owner_tid` / `dst_owner_tid`
  (`src/assembleur_edgechoice.py:553-576`) ;
- le validateur de chaîne l'utilise pour ses diagnostics
  (`src/assembleur_sim.py:313`) ;
- les tests de migration couvrent la recherche et les métadonnées EdgeChoice
  (`tests/test_topology_element_ids.py:439-526`).

**Après projection.** `toLegacyDict()` produit `id`
(`src/assembleur_sim.py:71`). Cette clé est utilisée par le rendu, les listes,
la suppression/réinsertion et le XML (`src/assembleur_tk.py`,
`src/assembleur_io.py:517-521, 738-750`) ; elle est également indexée par
`CanvasObjectsCollection`.

**Classification.** Comportement métier de projection, compatibilité UI/XML,
debug et tests. **Conclusion : conserver.** Le Core fournit l'identité
d'instance topologique, mais pas l'identifiant catalogue/UI attendu par les
consommateurs actuels.

### 4.2 `points`

**Description.** Dictionnaire des points monde `O/B/L` d'un triangle placé.

**Écritures.** Toutes les constructions `PlacedTriangle` citées pour
`triangleId` lui donnent soit la projection initiale, soit les points transformés
du candidat. `fromLegacyDict()` lit `entry["pts"]`
(`src/assembleur_sim.py:57`).

**Lectures directes.**

- `toLegacyDict()` émet `pts` (`70`) ;
- `buildEdgeChoiceEptsForAutoChain()` exige un dictionnaire de points et en
  tire les points mobile/destination (`src/assembleur_edgechoice.py:553-579`) ;
- `PlacedTriangles.clone()` est testé pour l'indépendance profonde des points
  (`tests/test_topology_element_ids.py:439-488`).

**Après projection.** `last_drawn[*]["pts"]` est un consommateur UI majeur :
rendu, hit-tests, drag, rotations/flips, snap, contours, synchronisation de
pose vers le Core et XML. Exemples directs :
`_sync_group_elements_pose_to_core()` (`src/assembleur_tk.py:743-823`),
dessin (`5828-5831`, `6018-6101`) et persistance
(`src/assembleur_io.py:517-521, 738-750`).

**Classification.** Comportement de simulation/EdgeChoice et projection UI.
**Conclusion : indispensable aujourd'hui.** Les points sont dérivables du
Core (forme locale + pose), mais cette dérivation n'a pas encore remplacé les
nombreux lecteurs UI ; ce serait une migration fonctionnelle distincte.

### 4.3 `labels`

**Description.** Labels O/B/L provenant de l'entrée catalogue.

**Écritures.** `fromLegacyDict()` (`58`), les constructeurs de simulation
(`946`, `953`, `1008`, `1015`, `1161`, `1173`, `1266`, `1281`) les alimentent
depuis les dictionnaires catalogue/source.

**Lectures directes.** Aucune lecture de production de `PlacedTriangle.labels`
après construction n'a été trouvée ; il est seulement émis par `toLegacyDict()`
(`69`). Les tests vérifient le format de cette projection.

**Après projection.** `last_drawn[*]["labels"]` alimente le dessin et la
présentation des triangles (`src/assembleur_tk.py:5828-5831`,
`10037-10040`) ; le XML et les outils de dictionnaire conservent encore des
flux voisins de labels.

**Classification.** Projection UI et tests. **Conclusion : pas une vérité du
simulateur.** Les labels sont disponibles dans `TopologyElement.vertex_labels`
ou dans le catalogue, mais le rendu legacy les lit encore dans la projection.

### 4.4 `mirrored`

**Description.** Flag de miroir de la projection, exporté vers la clé
`mirrored`.

**Écritures.** `fromLegacyDict()` (`59`) et les constructeurs de simulation
(`948`, `955`, `1010`, `1017`, `1162`, `1174`, `1267`, `1282`) copient le flag
de l'entrée source.

**Lectures directes.** Aucune lecture de production de l'attribut
`PlacedTriangle.mirrored` après construction ; `toLegacyDict()` le transporte
(`72`).

**Après projection.** L'UI l'affiche, le bascule lors du flip et l'utilise pour
synchroniser la pose Core : `_sync_group_elements_pose_to_core()`
(`src/assembleur_tk.py:800-823`), rendu (`5828-5831`, `6098-6101`) et flip
(`8861-8911`). XML le sérialise/désérialise
(`src/assembleur_io.py:517-521, 738-750`).

**Classification.** Projection UI, compatibilité XML et tests. **Conclusion :
conserver la projection.** Le Core possède aussi `TopologyElement.pose.mirrored`;
le champ est donc remplaçable comme stockage de simulation, mais seulement une
fois les lecteurs de `last_drawn["mirrored"]` migrés.

### 4.5 `topologyElementId`

**Description.** Identité stable de l'élément TopologyWorld correspondant à la
projection.

**Écritures.**

- `fromLegacyDict()` (`60`) ;
- `createTopoQuadrilateral()` écrit les deux nouveaux éléments
  (`587-591`) ;
- `_assignTopologyElementIdToPlacedTriangle()` écrit les éléments ajoutés à
  une branche (`801-810`, appels `1165-1179`, `1270-1287`).

**Lectures directes.**

- recherche `PlacedTriangles.findByTopologyElementId()` (`115-119`) ;
- validation de projection dans `buildLegacyGroupsFromTopology()` (`166`) ;
- groupe Core pour overlap (`1228`) ;
- métadonnées de candidat (`1319`) ;
- finalisation de scénario (`1498-1508`) ;
- EdgeChoice (`src/assembleur_edgechoice.py:555, 573-574`) ;
- validateur debug (`src/assembleur_sim.py:313`).

**Après projection.** La clé `topoElementId` est le pont utilisé par les
helpers UI de résolution Core, `CanvasObjectsCollection`, suppression,
contours, snapping, synchronisation de pose, debug et XML. Les helpers centraux
sont `src/assembleur_tk.py:679-733`.

**Classification.** Comportement métier de liaison Core ↔ projection, UI,
validation et tests. **Conclusion : indispensable.** Il n'est pas redondant
dans une projection : il est précisément la référence qui permet d'interroger
le Core.

### 4.6 `groupId`

**Description.** Ancien identifiant de groupe UI, distinct de tout identifiant
Core.

**Écritures.** `fromLegacyDict()` (`61`) et tous les constructeurs automatiques
le fixent à `1` (`949`, `956`, `1011`, `1018`, `1163`, `1175`, `1268`, `1283`).

**Lectures directes.** Aucune lecture de production de
`PlacedTriangle.groupId` n'a été trouvée. La seule sortie est
`toLegacyDict()` (`74-75`). Aucun algorithme de la simulation ne branche sur
cette valeur.

**Après projection.** `group_id` est encore consommé par les groupes UI
historiques, le XML (écriture `src/assembleur_io.py:517-521`, reconstitution
`738-845`), debug et plusieurs tests. Les opérations récentes MOVE/ROTATE/FLIP
résolvent leurs membres via le Core, mais ce champ reste requis par des flux de
compatibilité et par `Scenario.groups`.

**Classification.** Transport/compatibilité legacy, XML, debug et tests ; pas
comportement métier du simulateur. **Conclusion : suspecté mort dans
`PlacedTriangle` mais non supprimable globalement.** La preuve d'absence de
lecteur direct ne prouve pas l'absence de lecteur de `last_drawn["group_id"]`.

### 4.7 `topologyGroupId`

**Description.** Copie projetée d'un groupe Core canonique.

**Écritures.**

- `fromLegacyDict()` (`62`) ;
- `createTopoQuadrilateral()` écrit la valeur issue de
  `world.get_group_of_element()` (`645-650`) ;
- la finalisation repasse sur tous les triangles et réécrit la valeur depuis
  `topoWorld_leaf.get_group_of_element(element_id)` (`1496-1508`).

**Lectures directes.** Aucune lecture de production de
`PlacedTriangle.topologyGroupId` n'a été trouvée. Il est uniquement émis par
`toLegacyDict()` (`78-79`).

**Après projection.** `last_drawn[*]["topoGroupId"]` est relu/validé par le
chargement XML (`src/assembleur_io.py:754-939`). Certaines structures UI
secondaires (références d'horloge et guides) portent elles aussi une clé
homonyme, mais elles ne sont pas des `PlacedTriangle` et ne doivent pas être
confondues avec ce champ.

**Classification.** Projection Core → legacy, compatibilité XML et tests.
**Conclusion : valeur entièrement remplaçable par
`TopologyWorld.get_group_of_element(topologyElementId)`.** La finalisation le
démontre déjà ; la suppression exige seulement de migrer les lecteurs de la
projection/XML.

### 4.8 `chainEdgeIn`

**Description.** Arête de chaîne historique côté entrant.

**Écritures.** L'unique écriture de production est le candidat automatique
mobile, après calcul `LO/BL` (`src/assembleur_sim.py:1259-1277`).
`fromLegacyDict()` peut aussi importer `_chain_edge_in` (`63`).

**Lectures directes.** Uniquement
`validateCoreConnectionsAgainstLegacyChain()` (`295-314`), conditionnellement
au flag `DEBUG_VALIDATE_CORE_CONNECTIONS`; les tests de migration construisent
des cas oracle (`tests/test_topology_element_ids.py:303-363, 439-488`).

**Après projection.** Émis seulement si non nul (`80-81`). La recherche
exhaustive de `_chain_edge_in` ne trouve aucun consommateur de production hors
`src/assembleur_sim.py`; le second fichier est le test de migration.

**Classification.** Validation temporaire et tests. **Conclusion : candidat
de suppression direct après désactivation/retrait du validateur temporaire.**
MIG-009 a supprimé toute lecture par le builder fonctionnel.

### 4.9 `chainEdgeOut`

**Description.** Arête de chaîne historique côté sortant.

**Écritures.** L'unique écriture de production concerne le triangle destination
déjà placé (`src/assembleur_sim.py:1259-1261`). `fromLegacyDict()` peut
importer `_chain_edge_out` (`64`).

**Lectures directes.** Uniquement le validateur temporaire (`307-308`) et les
tests de migration (`303-363`, `439-488`).

**Après projection.** Émis seulement si non nul (`82-83`). La recherche de
`_chain_edge_out` ne révèle aucun lecteur fonctionnel hors
`src/assembleur_sim.py` et les tests.

**Classification.** Validation temporaire et tests. **Conclusion : même
statut que `chainEdgeIn`; candidat prioritaire pour MIG-010.**

## 5. Champs devenus purement legacy ou de projection

### Purement temporaires

- `chainEdgeIn` ;
- `chainEdgeOut`.

Ils ne servent plus au comportement de reconstruction depuis MIG-009. Le
contrat fonctionnel vient désormais de :

```text
EdgeChoiceEpts -> TopologyAttachment.incident_edge_by_element
               -> TopologyWorld.attachments
               -> buildLegacyGroupsFromTopology
               -> Scenario.groups[*].nodes[*].edge_in / edge_out
```

### Passifs dans le simulateur, mais actifs après projection

- `groupId` : pas de lecteur typé ; `group_id` reste une compatibilité UI/XML ;
- `topologyGroupId` : pas de lecteur typé ; `topoGroupId` reste une projection
  XML/UI ;
- `labels` : seulement transporté à la sortie ;
- `mirrored` : seulement transporté à la sortie dans le simulateur.

Ces champs ne doivent pas être qualifiés de « code mort » tant que leurs clés
`last_drawn` restent consommées.

## 6. Champs encore indispensables

- **`topologyElementId`** : lien non ambigu entre projection, index de canvas
  et élément Core. Sa suppression rendrait impossible la résolution Core sans
  inférence fragile.
- **`triangleId`** : identifiant catalogue/UI et métadonnée propriétaire des
  raccords EdgeChoice ; il est encore requis par le XML et l'affichage.
- **`points`** : géométrie opérationnelle de l'UI et entrée du calcul
  EdgeChoice. Une suppression demanderait la migration de l'ensemble de ces
  lecteurs vers une projection recalculée depuis les poses Core.

## 7. Recommandations et ordre des migrations

### MIG-AUTO-TOPO-010 — Retrait de l'oracle de chaîne

Supprimer `chainEdgeIn`, `chainEdgeOut`, `_chain_edge_in`, `_chain_edge_out`
et `validateCoreConnectionsAgainstLegacyChain()` **uniquement après une période
de validation suffisamment longue** des scénarios automatiques. Complexité
faible, risque faible : aucun consommateur fonctionnel de production n'a été
trouvé.

**Statut : réalisé par MIG-AUTO-TOPO-010.** Les champs, leurs clés privées et
l'oracle temporaire ont été supprimés après confirmation de l'absence de
consommateur fonctionnel. Les anciens dictionnaires qui portent ces clés sont
acceptés et les ignorent à l'import.

### MIG-AUTO-TOPO-011 — Ne plus produire `PlacedTriangle.groupId`

Supprimer les écritures constantes `groupId=1` et l'export correspondant
seulement après migration des consommateurs de `last_drawn["group_id"]`, en
particulier XML, `Scenario.groups` et les outils/debug legacy. Complexité
moyenne à forte : le champ typé est passif, mais la projection ne l'est pas.

### MIG-AUTO-TOPO-012 — Projection dynamique du groupe Core

**Statut MIG-AUTO-TOPO-011 : réalisé.** `PlacedTriangle.groupId` a été
supprimé. La clé `last_drawn["group_id"]` reste provisoirement produite par une
projection explicite dérivée du Core pour assurer la compatibilité UI/XML.

Supprimer le stockage `PlacedTriangle.topologyGroupId` au profit d'une
projection finale calculée à la demande par
`TopologyWorld.get_group_of_element(topologyElementId)`. Commencer par les
lecteurs XML/validation de `last_drawn["topoGroupId"]`. Complexité moyenne ;
risque de compatibilité de chargement XML.

### MIG-AUTO-TOPO-013 — Réduire la projection géométrique legacy

**Statut MIG-AUTO-TOPO-012 : réalisé.** `PlacedTriangle.topologyGroupId` a été
supprimé. La clé `last_drawn["topoGroupId"]` reste produite directement depuis
`TopologyWorld.get_group_of_element()`.

Étudier séparément le remplacement de `labels`, `mirrored` et finalement
`points` par une projection UI dérivée du Core. Complexité forte et risque
fonctionnel élevé : ces clés sont encore au coeur du rendu et des interactions.

### Migration ultérieure — Retrait de `Scenario.groups`

**Statut MIG-AUTO-TOPO-013 : réalisé pour le simulateur automatique.** Il ne
produit plus `Scenario.groups`. `TopologyWorld` et `orderedElementIds` sont la
seule représentation métier des groupes pendant la simulation.

**Statut MIG-AUTO-TOPO-014 : réalisé.** La clé
`last_drawn["group_id"]` a été supprimée de la projection automatique. Les
consommateurs résolvent désormais le groupe depuis `topoElementId` via
`TopologyWorld.get_group_of_element()` ; `last_drawn["topoGroupId"]` reste une
projection temporaire calculée depuis le Core.

**Statut MIG-AUTO-TOPO-015 : réalisé.**
`buildLegacyGroupIdMappingFromTopology()` et
`buildLegacyGroupsFromTopology()` ont été supprimés définitivement. La
projection historique des groupes UI a été retirée du simulateur ;
`TopologyWorld` est l'unique représentation métier des groupes.

**Statut CORE-CLEAN-GROUPS-001 : audit positif.** Hors persistance XML et
compatibilité historique, aucune structure legacy de groupes ne sert encore de
source de vérité. Les opérations fonctionnelles résolvent les groupes depuis
`TopologyWorld`.

Cette suppression doit rester distincte. `Scenario.groups` et le XML historique
portent encore `nodes` et `edge_in/out`; ils ne sont pas des champs de
`PlacedTriangle`, mais conditionnent la suppression de `groupId` projeté.

## 8. Conclusion

La réponse à la question initiale est nuancée : **seuls `chainEdgeIn` et
`chainEdgeOut` sont aujourd'hui des reliquats directement supprimables à court
terme, sous réserve du retrait assumé de leur validateur temporaire.**

`groupId` et `topologyGroupId` ne participent plus au calcul de branche sous
forme d'attributs `PlacedTriangle`, mais leurs projections dictionnaire ont des
consommateurs réels. Les supprimer maintenant modifierait la compatibilité
XML/UI, ce qui est explicitement hors périmètre de cet audit. `triangleId`,
`points` et `topologyElementId` restent nécessaires au comportement actuel ;
`labels` et `mirrored` sont des projections encore actives côté UI.
