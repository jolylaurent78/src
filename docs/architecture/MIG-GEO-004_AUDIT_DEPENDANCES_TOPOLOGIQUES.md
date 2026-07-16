# MIG-GEO-004 — Audit des dépendances topologiques restantes

## Réponse synthétique

La vérité métier restante la plus importante dans le legacy n'est pas
`group_id` isolément. C'est la **chaîne ordonnée de projection UI** :
`self.groups[ui_gid]["nodes"]`, son ordre, et les liens `edge_in` / `edge_out`
entre nœuds voisins. MOVE, ROTATE et FLIP n'utilisent plus cette chaîne pour
déterminer leurs membres ; collage, déconnexion, scénarios automatiques et XML
historique l'utilisent encore.

Le Core contient déjà les faits topologiques fondamentaux : composante d'un
élément, liste des éléments, arêtes, sommets, attachments typés, graphe
conceptuel, voisins et frontière. Il ne fournit pas encore un contrat de
**parcours ordonné** équivalent à la liste `nodes`. La prochaine migration doit
donc traiter la projection/reconstruction de chaîne, non supprimer des champs
legacy un par un.

## Méthode et périmètre

Audit en lecture seule des références à `self.groups`, `group_id`, `group_pos`,
`edge_in`, `edge_out`, `vkey_in` et `vkey_out` dans :

- `src/assembleur_tk.py` ;
- `src/assembleur_io.py` ;
- `src/assembleur_sim.py` ;
- capacités correspondantes de `src/assembleur_core.py`.

Convention : **A** désigne une vérité métier fondamentale ; **B** un cache,
raccourci, représentation ou optimisation de projection.

```text
TopologyWorld.groups[coreGroupId]
  = composante canonique : element_ids + attachment_ids

TriangleViewerManual.groups[uiGroupId]
  = projection UI : nodes[{tid, edge_in, edge_out}], bbox, topoGroupId
```

## Cartographie consolidée

| Structure Legacy | Usage métier | Fichier(s) | Criticité | Peut être fourni par Core ? |
|---|---|---|---|---|
| `group_id` | adressage de composante UI, sélection, exclusion au snap, compatibilité XML | `assembleur_tk.py`, `assembleur_io.py`, `assembleur_sim.py` | Forte | Partiel : composante/membres oui ; `ui_gid` non |
| `self.groups` / `nodes` | ordre de chaîne, voisinage local, bbox, collage, split, projection de dégroupement | mêmes fichiers | Forte | Partiel : connectivité oui ; ordre de chaîne non explicite |
| `group_pos` | copie de l'index dans `nodes` | Tk, IO, simulation | Faible | Oui si une projection ordonnée le requiert ; aucun besoin Core actuel |
| `edge_in` | arête reliée au prédécesseur de chaîne | Tk, IO, simulation | Forte | Oui pour l'attachment ; partiel pour « prédécesseur » |
| `edge_out` | arête reliée au successeur de chaîne | Tk, IO, simulation | Forte | Oui pour l'attachment ; partiel pour « successeur » |
| `vkey_in` | sommet opposé à l'arête vers le prédécesseur | Tk, simulation | Faible/Moyenne | Oui, dérivable de l'arête locale |
| `vkey_out` | sommet opposé à l'arête vers le successeur | Tk, simulation | Faible/Moyenne | Oui, dérivable de l'arête locale |

## 1. `group_id` et `self.groups`

### Inventaire par catégorie

| Catégorie | Zones observées | Raison de la consultation | A ou B ? |
|---|---|---|---|
| `[MEMBERS]` | `_group_nodes`, `_sync_group_elements_pose_to_core`, fallback MIG-GEO, dégroupement UI | énumérer des `tid` | B pour MOVE/ROTATE/FLIP ; encore nécessaire aux opérations non migrées |
| `[BOUNDARY]` | `_draw_group_outlines`, contexte de création de chemin | dessiner le contour et relier le clic UI au Core | B pour le rendu ; la frontière métier est déjà Core |
| `[NEIGHBOR]` | `_find_group_link_for_vertex`, exclusion `exclude_gid` dans le snap | reconnaître le lien avec le voisin de chaîne et éviter le snap sur soi | composante = A Core ; précédent/suivant = B de projection |
| `[ASSEMBLY]` | signature de scénarios, normalisation, collage/fusion, split CTRL, simulation | représenter une chaîne et ses liaisons adjacentes | attachment = A ; ordre linéaire = B/contrat manquant |
| `[SELECTION]` | `_get_group_of_triangle`, clic centre/sommet, contexte clic droit | retrouver rapidement le groupe UI depuis un `tid` | B ; Core répond déjà `elementId -> coreGroupId` |
| `[DISPLAY]` | réconciliation auto, contours, statuts | maintenir une projection cohérente | B |
| `[BBOX]` | `_recompute_group_bbox`, centroid, déplacement UI, dégroupement | cache géométrique d'affichage/hit-test | B |
| `[AUTRE]` | XML v4, `_new_group_id`, F8 | compatibilité, identité locale, diagnostic | B, sauf comparaison transitoire Core/UI |

### Ce que les migrations ont déjà retiré

MOVE, ROTATE et FLIP résolvent désormais `topoElementId ->
TopologyWorld.get_group_of_element() -> element_ids -> _last_drawn`. Les
groupes UI sont encore lus pour comparer Legacy/Core, pour bbox et pour la
synchronisation de pose, mais ne choisissent plus les triangles transformés.

### Dépendances restantes à forte criticité

**Snap et sélection.** `_get_group_of_triangle()` lit encore le champ porté
par le triangle. `_find_nearest_vertex` élimine les candidats du même `gid`.
Ce sont des raccourcis UI ; l'équivalent métier est la composante Core. Leur
migration touche cependant directement le collage et dépend du futur contrat
de voisinage.

**Collage/fusion.** Au relâchement d'un déplacement assisté, le code compare
le groupe mobile et cible, ordonne/réordonne `nodes`, met à jour `group_id` et
`group_pos`, puis écrit le `topoGroupId` retourné par
`TopologyWorld.apply_attachments`. Le Core est déjà autoritaire pour la
fusion topologique ; le legacy reste autoritaire pour l'ordre de projection
immédiat.

**Déconnexion/dégroupement.** CTRL découpe une liste `nodes` à une position.
Le dégroupement métier est en revanche déjà réalisé par
`TopologyWorld.degrouperAtNode`; `_applyDegrouperResultToTk` reconstruit la
projection UI, en préservant si possible l'ordre précédent. Cela confirme que
la partition est Core, mais que l'ordre affiché est legacy.

**XML et simulation.** Le XML v4 persiste `groups/nodes` avec `tid`,
`edge_in/out`, et la simulation génère également `group_id`, `group_pos` et
ces nœuds. Ce n'est donc plus un simple cache en mémoire : c'est encore une
projection persistée de compatibilité.

### Verdict

La composante connexe est **A**, déjà détenue par `TopologyWorld` via
`get_group_of_element`, `find_group` et `TopologyGroup.element_ids`.
`group_id` est majoritairement **B**, mais reste couplé à la chaîne ordonnée
que l'UI utilise pour l'assemblage et la déconnexion.

## 2. `group_pos`

### Résultat de la recherche

Toutes les occurrences applicatives sont des écritures ou initialisations :
création de singleton, chargement XML, réconciliation de scénario auto, split,
dégroupement, fusion et génération de simulation. Aucun lecteur métier de
`group_pos` n'a été trouvé : pas de test, tri, calcul, rendu ou décision qui
lise sa valeur. L'ordre réellement consommé est l'énumération de `nodes`.

### Verdict

`group_pos` est une dénormalisation de l'index dans `nodes`, donc un cache
**B** de criticité faible. Il ne doit pas être copié dans le Core comme vérité
métier. Son devenir dépend de la future projection ordonnée.

## 3. `edge_in` / `edge_out`

### Sémantique legacy

Pour un `node` UI :

- `edge_in` est l'arête locale (`OB`, `BL`, `LO`) partagée avec le
  prédécesseur ;
- `edge_out` est l'arête locale partagée avec le successeur.

Le contrat de chaîne est :

```text
nodes[i].edge_out  <->  nodes[i + 1].edge_in
```

### Consommateurs

| Zone | Ce que le code recherche | Catégorie | Couverture Core |
|---|---|---|---|
| `_scenario_connections_signature` | signature bidirectionnelle triangle/arête/voisin | `[ASSEMBLY]` | attachments oui ; ordre de chaîne non |
| `_normalize_group_nodes` | validité du contrat de chaîne | `[ASSEMBLY]` | connectivité oui ; séquence UI non |
| `_find_group_link_for_vertex` | lien du sommet au voisin précédent/suivant | `[NEIGHBOR]` | attachment oui ; direction de parcours non |
| split CTRL | suppression des liens aux nouvelles extrémités | `[ASSEMBLY]` | détachement Core oui ; projection à reconstruire |
| collage/fusion | arêtes de l'insertion et rotation de sous-chaîne | `[ASSEMBLY]` | création d'attachment oui ; insertion ordonnée non |
| simulation | arêtes réellement partagées de la chaîne générée | `[ASSEMBLY]` | résultat topologique oui si attachment autoritaire |
| XML v4 | persistance et validation des noms d'arête | `[AUTRE]` | possible seulement après définition XML cible |
| filtrage de scénarios | comparaison de séquences de connexions | `[ASSEMBLY]` | nécessite une projection de parcours stable |

### Ce que possède déjà `TopologyWorld`

Le Core possède les arêtes de `TopologyElement`, les références typées de
`TopologyAttachment`, les APIs `get_edge`, `apply_attachment`,
`getConceptEdgesForNode`, `getConceptNeighborNodes`, `computeBoundary` et
`getBoundarySegments`. Il connaît donc le fait : « l'arête A de T17 est
attachée à l'arête B de T18 ».

Il ne possède pas, sous le contrat consommé par l'UI, le fait : « T18 est le
successeur de T17 dans cette chaîne ». Une frontière Core ne suffit pas à
reconstruire nécessairement cet ordre interne.

### Verdict

L'attachement d'arêtes est **A** Core. `edge_in/out` mélange cet attachement
avec un choix de parcours UI **B**. Les déplacer tels quels créerait une
seconde autorité ; il faut d'abord définir une projection de parcours.

## 4. `vkey_in` / `vkey_out`

Ces champs désignent `O`, `B` ou `L`, soit le sommet opposé à l'arête de
liaison avec le voisin précédent/suivant.

L'état observé est transitoire : `_normalize_group_nodes` les retire et les
déclare legacy ; la détection de lien dérive déjà le sommet opposé depuis
`edge_in/out`. Mais la simulation les produit encore, et certaines branches de
collage/fusion les recopient ou les réécrivent.

Le Core connaît l'index de chaque vertex et arête. Pour un triangle, le sommet
opposé à `OB`, `BL` ou `LO` est dérivable. Les helpers Core exposent aussi
`elementId`, `vertexIndex` et `vkey` depuis un nœud topologique.

**Verdict :** `vkey_in/out` est un dérivé **B**, pas une vérité métier. Il ne
doit être traité qu'après la clarification de `edge_in/out`; le retirer avant
ne réduirait pas le couplage d'assemblage.

## Vérités et caches : décision d'architecture

| Information | Autorité métier A | Projection/cache B | Décision |
|---|---|---|---|
| appartenance d'un élément | DSU Core et `element_ids` | `group_id` UI | Core autoritaire |
| attachement entre features | `TopologyAttachment` | `edge_in/out` dans une chaîne | Core autoritaire ; projection à concevoir |
| frontière | graphe conceptuel et `BoundarySegment` | contour Canvas/bbox | Core calcule, UI dessine |
| ordre de chaîne | aucun contrat Core explicite observé | `nodes`, `group_pos` | question fonctionnelle à résoudre |
| sommet opposé à une arête | déduction topologique locale | `vkey_in/out` | ne pas persister comme autorité |
| bbox, centroid, `tid` | aucune topologie | adressage/rendu UI | cache de projection |

## Feuille de route recommandée

### MIG-GEO-005 — Caractérisation du contrat de chaîne legacy

Documenter et tester les invariants de `nodes`, l'ordre, les paires
`edge_in/out`, les signatures de scénarios, split et fusion. Comparer ces
invariants aux attachments Core sur collage, déconnexion, dégroupement,
chargement XML et scénarios auto.

**Pourquoi d'abord :** l'égalité d'ensembles ayant validé MOVE/ROTATE/FLIP ne
détecte ni inversion de chaîne ni mauvaise arête. C'est le risque métier le
plus élevé et le test le plus utile avant toute écriture.

### MIG-GEO-006 — Projection Core vers chaîne UI, en observation

Concevoir un moteur de projection qui dérive, pour un CoreGroup, les entrées
UI, adjacences et éventuellement un ou plusieurs parcours déterministes. Ne
supprimer aucun champ legacy ; comparer passivement la projection à `nodes`.

**Pourquoi ensuite :** Core fournit la connectivité mais pas la notion
prédécesseur/successeur attendue par l'UI. Cette étape résout le manque réel,
notamment pour des composantes non linéaires ou cycliques.

### MIG-GEO-007 — Migration contrôlée COLLAGE et DÉCONNEXION

Basculer lecture puis écriture, opération par opération, de la chaîne legacy
vers la projection validée. Garder temporairement `groups/nodes` comme cache
et comparateur. Le dégroupement Core, déjà existant, est un bon premier cas de
reconstruction contrôlée.

**Pourquoi en troisième :** collage et déconnexion écrivent simultanément la
partition, l'ordre, les arêtes, les groupes UI et les attachments. Leur impact
fonctionnel est maximal.

### Après MIG-GEO-007 — `group_pos` puis `vkey_in/out`

Ces champs sont à faible valeur métier. Leur retrait doit découler d'une
projection stabilisée, non constituer une migration prioritaire.

## Questions ouvertes

- Une composante Core non linéaire ou cyclique doit-elle avoir une unique
  chaîne UI, plusieurs parcours, ou une représentation de graphe ?
- Quel ancrage rend le parcours déterministe après collage, fusion et XML ?
- L'ordre de scénario automatique est-il un fait métier distinct de la
  topologie, ou une stratégie de rendu ?
- Quel XML cible porte les attachments et, si nécessaire, les choix de
  parcours sans réintroduire `tid` comme autorité ?

## Réponse finale

Le legacy conserve surtout la vérité de **parcours ordonné de projection**,
pas celle de connectivité. La prochaine étape doit formaliser et valider cette
projection de chaîne depuis les attachments Core. Toute suppression mécanique
de `group_id`, `group_pos` ou `vkey_*` avant ce travail réduirait des champs,
mais pas le couplage métier restant.
