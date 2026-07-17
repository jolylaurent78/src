# AUDIT-DRAG-SNAP-001 — Drag, snap, prévisualisation et frontière Core/UI

Date : 2026-07-17  
Statut : audit statique — aucune modification de code demandée ni effectuée.

## 1. Résumé exécutif

Le flux de drag/snap est fonctionnellement organisé autour de deux états très différents :

1. le **Core** (`TopologyWorld`) porte les éléments, leurs poses persistées, les
   attachments et la frontière topologique ;
2. l'interface fait vivre pendant le drag une géométrie transitoire dans
   `_last_drawn`, puis produit une proposition de collage dans `_edge_choice`.

`_update_edge_highlights()` est donc principalement une fonction d'interaction :
elle choisit et dessine une paire d'arêtes candidate à partir de la géométrie
visuelle courante. Elle ne crée pas directement d'attachment et ne modifie pas
les poses Core. En revanche, elle appelle `simulateOverlapTopologique()`, qui
reconstruit sa géométrie depuis les poses persistées du Core. Le rejet d'une
candidate peut donc être évalué dans un référentiel différent de celui du
curseur et de l'aperçu.

Conclusion principale : le modèle recommandé est **C — modèle partagé avec
frontière explicite**.

- UI : geste souris, géométrie temporaire, hit-testing, classement des
  candidates et rendu ;
- Core : identité topologique, groupes canoniques, boundary, validation de la
  proposition et transaction définitive ;
- contrat explicite : une proposition de snap doit contenir des références
  topologiques stables et une géométrie temporaire explicite si elle est
  validée avant le relâchement.

Il n'est pas prudent de qualifier l'appel actuel de
`simulateOverlapTopologique()` de « purement sans effet ». Il ne modifie ni
attachment, ni pose, ni transaction métier, mais `find_group()` peut compresser
le DSU et `_build_ring_from_boundary_cycle()` appelle `ensureConceptGeom()`,
susceptible d'alimenter/rafraîchir un cache conceptuel. Les effets sont des
caches internes, pas des mutations topologiques durables.

Le risque architectural dominant est **élevé** : la proposition visualisée et
le choix appliqué au release dépendent de coordonnées `_last_drawn` et d'indices
UI ; la simulation de chevauchement, elle, part des poses Core. L'intégration
manuelle doit donc vérifier précisément les scénarios de collage avant toute
migration.

## 2. Flux fonctionnel réel : drag → aperçu → choix → release → transaction

### 2.1 Sélection et début du drag

Dans `_on_canvas_left_down()` (`src/assembleur_tk.py`, vers 9922), un clic de
centre ou de sommet initialise `self._sel` en mode `move_group`.

- Le groupe à déplacer est déjà déterminé depuis le Core :
  `topoElementId -> TopologyWorld.get_group_of_element(...) -> core_group_id`.
- `move_member_entries` contient les entrées `_last_drawn` projetées des
  membres Core.
- Le clic sommet garde aussi une ancre visuelle `{type: "vertex", tid, vkey}`.
- Le `ui_group_id` peut rester présent pour la compatibilité de l'ancien flux
  de fusion au release ; il n'est pas la source utilisée pour choisir les
  membres du mouvement.

Un clic sommet sans CTRL cherche une cible voisine avec `_find_nearest_vertex`,
puis appelle `_update_nearest_line()` et `_update_edge_highlights()`.

### 2.2 Mouvement et géométrie transitoire

Dans `_on_canvas_left_move()` (vers 10123), `_move_group_world()` translate les
points des entrées concernées dans `_last_drawn`. Cette opération réalise le
drag visuel. La pose Core n'est pas mise à jour à chaque événement souris.

Après redessin, l'assistance cherche à nouveau un sommet cible et recalcule les
highlights. La géométrie qui nourrit le choix est donc celle de la projection
temporaire au moment du dernier mouvement souris.

### 2.3 Aperçu et sélection de la candidate

`_update_edge_highlights(mob_idx, vkey_m, tgt_idx, tgt_vkey)` :

1. lit les deux triangles visuels dans `_last_drawn` ;
2. reconstitue leurs identifiants de nœuds Core depuis `topoElementId` et la
   clé de sommet `O/B/L` ;
3. extrait les contours visuels de leurs groupes ;
4. calcule les demi-arêtes de contour incidentes aux deux sommets ;
5. parcourt les paires possibles et conserve celle dont l'écart angulaire est
   minimal, après le filtre anti-chevauchement ;
6. dessine le résultat et transforme la meilleure paire en `EdgeChoiceEpts` ;
7. stocke l'ensemble dans `_edge_choice`.

Le dessin est seulement un aperçu. La meilleure paire devient cependant la
source immédiate de la décision au release : le commentaire du code l'indique
explicitement.

### 2.4 Release et transaction effective

Dans `_on_canvas_left_up()` (vers 10239), si une ancre et un `_edge_choice`
valide sont présents :

1. `EdgeChoiceEpts.computeRigidTransform()` calcule la rotation/translation de
   snap ;
2. ce transformé est appliqué aux points `_last_drawn` des `nodes` du groupe UI
   mobile ;
3. `_sync_group_elements_pose_to_core()` reconstruit et persiste les poses Core
   à partir de ces points de rendu ;
4. `epts.createTopologyAttachments()` construit les attachments ;
5. `beginTopoTransaction() -> apply_attachments() -> commitTopoTransaction()`
   modifie la topologie ;
6. le code legacy fusionne ensuite les groupes UI/`nodes` et met à jour
   `group_id`/`topoGroupId`.

La transaction couvre l'application topologique, pas le transformé préalable
de `_last_drawn` ni la synchronisation préalable des poses. C'est une frontière
à documenter pour la validation : un échec tardif peut laisser une géométrie
visuelle ou une pose Core déjà modifiée avant que l'attachment ne soit commité.
Cet audit constate le comportement ; il ne propose pas de correction.

## 3. Cartographie des méthodes

| Méthode / module | Rôle dans le flux | Nature dominante |
|---|---|---|
| `_on_canvas_left_down` | sélectionne groupe/ancre et déclenche l'assistance | UI interaction + résolution Core |
| `_on_canvas_left_move` | translate la projection puis rafraîchit l'aperçu | UI drag |
| `_move_group_world` | déplace les entrées projetées sélectionnées | UI géométrie temporaire |
| `_find_nearest_vertex` | trouve le sommet cible et exclut le groupe source | UI hit-test, avec identité Core |
| `_outline_for_item` | index UI -> élément topo -> groupe Core -> contour projeté | pont mixte |
| `_group_outline_segments` | union Shapely de triangles `_last_drawn` | UI géométrie / aperçu |
| `_build_boundary_graph` | graphe de demi-arêtes quantifié sur contour visuel | UI logique pure locale |
| `_incident_half_edges_at_vertex` | candidates de contour au sommet | UI logique pure locale |
| `_update_edge_highlights` | sélectionne, filtre et dessine la proposition | orchestration mixte |
| `buildEdgeChoiceEptsFromBest` | convertit candidate géométrique en proposition topologique | adaptateur mixte |
| `EdgeChoiceEpts.createTopologyAttachments` | instancie les attachments sans les appliquer | Core data preparation |
| `TopologyWorld.simulateOverlapTopologique` | valide la cohérence du contour simulé | simulation Core avec caches |
| `_on_canvas_left_up` | pose finale, sync poses, transaction et fusion UI legacy | transaction mixte |

## 4. Audit bloc par bloc de `_update_edge_highlights()`

### 4.1 Lecture de `tri_m`, `tri_t`, `pts`, `group_id` — [UI DRAG STATE]

Les indices `mob_idx`/`tgt_idx` désignent des positions dans `_last_drawn`.
Les points `Pm`/`Pt` sont la géométrie après drag visuel. `group_id` est encore
lu pour obtenir les éléments projetés via
`_get_projected_elements_for_ui_group()`. Ce n'est plus une lecture de membres
legacy : le helper résout `ui_gid -> topoGroupId -> Core`, puis utilise les
`element_ids` Core. Il reste néanmoins un adaptateur UI dont la présence dans
le calcul de snap est une dépendance de compatibilité.

### 4.2 Construction de `mAId` / `tAId` — [CORE DATA]

`topoElementId` et `world.format_node_id(element, vertex_index)` fournissent
des références stables vers les nœuds topologiques. Cette partie est saine :
elle ne déduit pas une identité métier depuis une coordonnée ou un ordre UI.

### 4.3 `_outline_for_item()` — [MIXED / PROBLEMATIC]

La fonction fait correctement `item UI -> topoElementId -> Core group` mais le
contour retourné est une union Shapely calculée sur les coordonnées courantes
de `_last_drawn`. Cela est nécessaire à l'aperçu de drag ; ce n'est pas le
boundary persistant du Core. Le fallback aux trois arêtes du triangle est une
dégradation UI légitime quand le contour de groupe ne peut être construit.

### 4.4 Graphe de frontière et incidence — [UI PREVIEW GEOMETRY]

`_build_boundary_graph`, `_incident_half_edges_at_vertex` et
`_normalize_to_outline_granularity` travaillent sur des segments temporaires
quantifiés. Ils déterminent les candidates visibles sous le curseur. Ils ne
doivent pas modifier le monde et n'ont pas à devenir une reconstruction du DSU.
La minimisation d'écart d'azimut est une logique mathématique pure qui pourrait
être extraite dans un service sans état, mais reste actuellement attachée aux
données UI.

### 4.5 Résolution des éléments projetés — [MIXED / PROBLEMATIC]

`_get_projected_elements_for_ui_group(gid)` :

```text
ui_group_id -> topoGroupId (UI) -> hasLiveGroup(Core)
            -> getGroupElementIds(Core) -> index _last_drawn
```

La membership est Core, non `groups[*]["nodes"]`. Toutefois le point de départ
reste l'identifiant de groupe UI de l'entrée pointée. Une API appelant avec le
`core_group_id` déjà dérivé de `topoElementId` supprimerait cet aller-retour.

### 4.6 `S_m_base` et `_place_union_on_pair()` — [UI PREVIEW GEOMETRY]

Le code fabrique une union Shapely mobile, puis une copie posée pour chaque
candidate. Or `S_m_pose` n'est jamais intersectée, ni fournie au simulateur.
Sa seule utilisation est un test `is not None`, toujours vrai pour une union
géométrique valide. Dans l'état observé, c'est donc un calcul transitoire sans
effet sur l'acceptation ; il duplique un raisonnement de pose sans participer
au résultat. C'est une ambiguïté technique à traiter dans un chantier séparé,
pas à retirer dans cet audit.

### 4.7 `_overlap_topo_for_pair()` — [CORE SIMULATION]

La closure fabrique `EdgeChoiceEpts`, convertit ces données en
`TopologyAttachment`, puis appelle la simulation Core. Les attachments sont
créés en mémoire ; aucun `apply_attachments()` n'est exécuté ici.

La logique de rejet est métier et relève bien du Core. En revanche, les
attachments prennent leurs paramètres géométriques (`tRaw`, propriétaires
d'arêtes, coordonnées) dans `_last_drawn`, alors que le simulateur reconstruit
les rings depuis les poses des éléments Core. Cet écart de source géométrique
est le problème majeur de la fonction.

### 4.8 Boucle de classement — [UI INTERACTION]

La boucle oriente les demi-arêtes autour du sommet, mesure l'écart angulaire,
rejette une candidate signalée en chevauchement et prend le minimum. C'est la
règle d'assistance à l'utilisateur, non une mutation métier. Le drapeau
`debug_skip_overlap_highlight` change toutefois le choix retenu, pas seulement
l'affichage : il doit rester un outil de diagnostic clairement identifié.

### 4.9 `_edge_highlights` et `_redraw_edge_highlights()` — [UI RENDERING]

Le dictionnaire conserve contours, incidences et meilleure paire. Le rendu
dessine les contours et candidates en bleu, puis la paire retenue en rouge.
Ces lignes Canvas sont des objets temporaires, supprimés par
`_clear_edge_highlights()`. Elles ne doivent pas devenir des données Core.

### 4.10 Construction de `_edge_choice` — [MIXED / PROBLEMATIC]

La valeur réelle est :

```python
(mob_idx, vkey_m, tgt_idx, tgt_vkey, epts)
```

`epts` est séquentiel `(mA, mBEdgeVertex, tA, tBEdgeVertex)` mais enrichi de
références d'éléments, d'arêtes, de labels, de `kind` et de `tRaw`.
Cette valeur est simultanément un cache d'aperçu et l'entrée de la transaction
au release. Elle mélange des indices instables UI et une proposition
topologique stable. Le commentaire « source de vérité pour le release » est
juste fonctionnellement, mais cette responsabilité est trop large pour un
cache de hover.

Les deux appels finaux à `_outline_for_item()` ne réutilisent pas leurs valeurs
locales avant le rendu ; ils paraissent redondants dans le chemin étudié.

## 5. Sources de géométrie et statut d'autorité

| Donnée | Producteur / lecteur | Rôle | Autorité pendant le drag |
|---|---|---|---|
| `TopologyElement.pose` + coordonnées locales | Core, `_build_ring_from_boundary_cycle()` | pose persistée / géométrie métier enregistrée | autorité persistée, mais pas mise à jour à chaque motion |
| `TopologyWorld` attachments / boundary cycle | Core | connectivité et frontière conceptuelle | autorité topologique |
| `_last_drawn[*]["pts"]` | UI drag, outline, `EdgeChoiceEpts`, release | projection, hit-test et géométrie provisoire | autorité temporaire de l'aperçu |
| contour Shapely de `_group_outline_segments()` | UI | contour visuel de la géométrie provisoire | rendu / candidates seulement |
| ring de `simulateOverlapTopologique()` | Core | contour de simulation reconstruit | dépend de la pose Core persistée |
| `_edge_highlights` | UI | lignes d'assistance | cache de rendu |
| `_edge_choice` | UI, consommé au release | proposition instantanée de snap | cache décisionnel à durée de vie courte |

La règle cible reste : `TopologyWorld` est la source de vérité topologique,
`_last_drawn` est une projection. Dans le flux actuel, `_last_drawn` devient
toutefois un **buffer de commande temporaire** : le release le transforme en
poses Core. Ce rôle est acceptable seulement s'il est explicitement borné entre
le début et la validation de l'opération.

## 6. Audit de `simulateOverlapTopologique()`

### 6.1 Ce que fait la simulation

`TopologyWorld.simulateOverlapTopologique()` (`assembleur_core.py`, vers 2573)
normalise les groupes, reconstruit les deux rings via leurs boundary cycles,
résout les deux jonctions apportées par les attachments, calcule une pose rigide
mobile, insère localement les pseudo-splits éventuels et valide le polygone de
sortie. Elle retourne `True` si le placement doit être rejeté.

### 6.2 Ce qu'elle ne fait pas

Dans le chemin observé, elle n'appelle pas :

- `apply_attachments()` ;
- `setElementPose...()` ;
- une transaction topologique ;
- une écriture dans les attachments ou les éléments du groupe.

Les `TopologyAttachment` créés pour la simulation sont des objets temporaires.
Le Core n'est donc pas modifié en tant que vérité métier par l'appel courant.

### 6.3 Effets indirects réels

La garantie « Core jamais changé pendant `_update_edge_highlights()` » ne peut
pas être donnée au sens strict :

- `find_group()` peut faire de la compression de chemin DSU ;
- `_build_ring_from_boundary_cycle()` appelle `ensureConceptGeom()`, qui peut
  produire/rafraîchir le cache géométrique conceptuel du groupe.

Ces effets n'altèrent ni la topologie, ni les poses, ni les attachments ; ce
sont des états internes dérivés. La formulation exacte est donc : **la
prévisualisation ne produit aucune mutation métier durable, mais elle n'est pas
un calcul pur sans effet de cache.**

### 6.4 Problème de référentiel

`_build_ring_from_boundary_cycle()` convertit les coordonnées locales par
`TopologyElement.localToWorld()`. Il lit donc les poses Core persistées. Pendant
un drag, les points `_last_drawn` ont déjà été déplacés ; les poses Core ne sont
synchronisées qu'au release. `simulateOverlapTopologique()` peut ainsi valider
un assemblage depuis une position antérieure alors que l'assistance calcule la
candidate depuis la position actuellement affichée.

Le calcul Shapely `S_m_pose` ressemble à une tentative de simuler le référentiel
UI, mais son résultat n'est pas consommé. Cette observation explique pourquoi
il ne fournit pas aujourd'hui une garantie de correspondance preview/final.

## 7. Contrat réel de `_edge_choice`

### Structure

`_edge_choice = (mob_idx, vkey_m, tgt_idx, tgt_vkey, epts)`.

- `mob_idx` et `tgt_idx` : indices dans `_last_drawn`, donc identités UI
  instables si la liste est reconstituée ou réordonnée ;
- `vkey_m` et `tgt_vkey` : sommet visuel `O/B/L` ;
- `epts` : points d'alignement et métadonnées topologiques nécessaires à la
  création des attachments.

### Cycle de vie

La valeur est remise à `None` lors de plusieurs nettoyages ou avant un nouveau
calcul. `_clear_edge_highlights()` ne l'efface volontairement pas. Au release,
elle détermine le transformé final, les attachments et une partie de la fusion
legacy UI. Le CTRL peut également consommer un choix existant pour une
opération géométrique.

### Risques

1. **Staleness élevé** : le release ne recalcule pas systématiquement la
   proposition. Une valeur calculée à la dernière motion peut ne plus être
   cohérente si l'état change entre-temps.
2. **Indices UI élevés** : un `tid` de projection n'est pas un contrat métier
   durable ; `topoElementId` est le bon identifiant stable.
3. **Mélange géométrie/topologie élevé** : le même objet porte des points
   temporaires et la future intention topologique.
4. **Compatibilité legacy moyenne** : le tuple est déballé à plusieurs endroits
   et doit rester séquentiel dans l'état actuel.

Un futur contrat devrait conserver les éléments/nœuds/arêtes en IDs Core, les
paramètres sémantiques (`kind`, `tRaw`, orientation) et, séparément, un
instantané géométrique/numéro de version de la projection qui l'a produit.

## 8. Violations et ambiguïtés de frontière

| Référence | Constat | Gravité |
|---|---|---|
| `assembleur_tk.py:7405+` | choix basé sur `_last_drawn`, simulation basée sur poses Core | Forte |
| `assembleur_tk.py:7470+` | `_get_projected_elements_for_ui_group(gid)` repart d'un UI gid pour une donnée déjà résoluble depuis `topoElementId` | Moyenne |
| `assembleur_tk.py:7490+` | transformé Shapely calculé mais non utilisé pour la validation | Moyenne |
| `assembleur_tk.py:7605` | `_edge_choice` mêle indices UI et proposition Core et est consommé au release | Forte |
| `assembleur_tk.py:103xx` | la transformation de `_last_drawn` et la sync de pose précèdent la transaction Core | Forte |
| `assembleur_tk.py:103xx` | collage final encore dépendant de `groups[*]["nodes"]`, `group_id` et métadonnées de chaîne | Forte, mais hors périmètre du seul aperçu |
| `assembleur_edgechoice.py:356+` | le builder consulte `_last_drawn` et emploie des détails de nœud Core ; c'est un adaptateur à frontière diffuse | Moyenne |
| `assembleur_edgechoice.py` | accès à un helper Core privé de parsing de node id | Moyenne |
| `assembleur_core.py:2573+` | simulation avec effet de cache possible et sans snapshot de pose temporaire | Moyenne |

## 9. Réponses explicites aux questions d'architecture

### 9.1 Ce qui doit impérativement rester UI

- événements souris, curseur, sélection et CTRL ;
- coordonnées écran/monde et tolérances de hit-test ;
- déplacement provisoire des entrées `_last_drawn` ;
- rendu Canvas, couleurs et lignes d'aide ;
- choix ergonomique parmi plusieurs candidates équivalentes.

### 9.2 Ce qui reconstitue actuellement une information topologique persistante

`_outline_for_item()` et `_group_outline_segments()` réinterprètent, à partir de
la projection, une frontière visuelle. Ce n'est pas la frontière Core : c'est
une reconstruction géométrique utilisée pour l'aperçu. La reconstruction de
propriétaire d'arête dans `buildEdgeChoiceEptsFromBest()` convertit aussi une
paire visuelle en références topologiques. Cette conversion est nécessaire,
mais doit rester un contrat explicite plutôt qu'une inférence dispersée.

### 9.3 Logique pure pouvant être dans le Core

- validation d'une proposition de `TopologyAttachment` ;
- simulation de contour à partir d'un snapshot explicite de poses/points ;
- normalisation des références de nœuds, éléments et arêtes ;
- lecture de boundary topologique et de ses incidences.

Le hit-test, l'azimut dépendant du curseur et le rendu ne doivent pas migrer au
Core.

### 9.4 Modèle recommandé

**C — partagé avec frontière explicite.** Le Core ne doit pas recevoir le flux
Tk ni muter pour prévisualiser. L'UI ne doit pas reconstruire les attachments
finaux avec des détails Core privés. Une proposition stable doit traverser la
frontière accompagnée, lorsque nécessaire, d'une géométrie temporaire de
validation en lecture seule.

### 9.5 Peut-on garantir que le Core ne change jamais pendant update ?

Pas strictement aujourd'hui, à cause des caches et de la compression DSU.
Oui au sens métier : aucun attachment, groupe, pose ou transaction n'est
modifié par `_update_edge_highlights()` dans le chemin observé.

### 9.6 Preview égal résultat final ?

Non garanti aujourd'hui. Le choix visible est calculé depuis `_last_drawn`, la
simulation depuis les poses Core, et le release applique un choix mémorisé sans
revalidation générale. Il peut être identique dans les cas usuels, mais ce
n'est pas un invariant formel.

### 9.7 `_get_projected_elements_for_ui_group()`

Ce helper ne reconstitue plus la membership via `groups[*]["nodes"]`. Il reste
néanmoins un pont de compatibilité UI : il commence par un `ui_gid`. Dans le
chemin de snap, il est préférable à terme de partir du `core_group_id` obtenu
depuis les triangles, puis d'appeler le helper équivalent Core.

### 9.8 Indices UI vs identifiants stables

`mob_idx`/`tgt_idx` sont valables pour le frame de projection courant mais pas
pour un contrat de transaction. `topoElementId`, node id et edge id sont les
références durables. Une migration doit garder les indices seulement comme
poignées de rendu et les vérifier contre les IDs stables à la validation.

### 9.9 Risques par catégorie

| Catégorie | Niveau | Motif |
|---|---|---|
| divergence preview/simulation/final | Élevé | deux référentiels de pose et choix mémorisé |
| intégrité transactionnelle | Élevé | poses synchronisées avant transaction attachment |
| dépendance legacy de fusion | Élevé | release utilise encore `nodes`, `group_id`, chaîne UI |
| effets Core pendant hover | Moyen | caches/DSU, pas de mutation métier |
| calcul de candidate | Moyen | epsilon, fallback triangle, propriétaire d'arête inféré |
| rendu Canvas | Faible | état entièrement temporaire |

## 10. APIs Core : nécessaires, optionnelles, à éviter

### Indispensables avant une validation strictement cohérente

1. **Simulation de proposition avec géométrie explicite en lecture seule** :
   accepter les groupes canoniques, des `TopologyAttachment` candidats et un
   snapshot temporaire des poses/points des groupes impliqués. Elle doit
   retourner le diagnostic sans modifier poses, attachments, DSU ni cache
   observables.
2. **Contrat public de proposition d'attachement** : construire/valider les
   références `element_id`, `node_id`, `edge_id`, `kind`, `tRaw` sans dépendre
   d'un parser privé du Core depuis l'UI.

### Utiles mais non bloquantes

1. incidence de frontière Core autour d'un node/edge, avec références
   topologiques stables ;
2. projection de boundary topologique par une fonction qui reçoit les points
   de projection au lieu de reconstruire une union Shapely ;
3. objet de diagnostic de simulation, distinct du booléen, pour expliquer une
   candidate rejetée.

`getBoundarySegments()` est déjà un fondement utile ; il faudra déterminer si
son contrat actuel (boundary pré-calculé et cache conceptuel) suffit à ces
usages avant d'ajouter une API.

### À décourager

- faire muter `TopologyWorld` à chaque mouvement souris ;
- exposer le DSU, les alias ou `world.groups` à l'UI ;
- déplacer les coordonnées Canvas et le rendu dans le Core ;
- créer une seconde structure métier de groupe pour contourner
  `TopologyGroup`.

## 11. Couverture de tests observée et lacunes

La recherche dans `tests/` trouve des tests de liaison MIG-GEO et de
comparaison topologique, notamment un déplacement de groupe via
`_move_group_world`. Aucun test direct n'exerce `_update_edge_highlights`,
`_edge_choice`, `simulateOverlapTopologique` depuis le drag UI ou le release
de snap.

Les lacunes à couvrir avant migration sont :

- drag d'un triangle isolé et d'un groupe multi-triangles ;
- candidates edge-edge, vertex-edge et cas sans candidate ;
- candidate rejetée par chevauchement ;
- cohérence géométrie preview / simulation / résultat après release ;
- mise à jour entre la dernière motion et le release ;
- CTRL (absence de snap et consommation éventuelle d'un choix) ;
- échec de création/application d'attachment et absence de divergence
  `_last_drawn` / pose Core ;
- groupe avec contour indisponible et fallback triangle ;
- attachement de deux groupes suivie de la compatibilité de fusion UI.

## 12. Feuille de route incrémentale proposée

### MIG-DRAG-001 — Tests de caractérisation du flux actuel

Objectif : capturer la sélection de candidate, le transformé final et les
mutations Core/UI sur les scénarios listés ci-dessus.  
Complexité : moyenne. Risque : faible. Bénéfice : prérequis de sécurité.

### MIG-DRAG-002 — Contrat de `SnapProposal` à la frontière

Objectif : représenter séparément IDs Core stables, paramètres sémantiques,
points d'aperçu et version de projection ; conserver un adaptateur tuple si la
compatibilité immédiate l'exige.  
Complexité : moyenne. Risque : moyen. Bénéfice : élimine l'ambiguïté de
`_edge_choice` sans changer l'algorithme.

### MIG-DRAG-003 — Simulation Core sur snapshot explicite

Objectif : aligner le référentiel du simulateur sur la géométrie visualisée,
en lecture seule et avec diagnostic.  
Complexité : forte. Risque : fort. Bénéfice : rend vérifiable l'invariant
« candidate vue = candidate validée ».

### MIG-DRAG-004 — Revalidation atomique au release

Objectif : valider la proposition fraîchement résolue avant de synchroniser la
pose, puis regrouper les effets finaux dans une frontière transactionnelle
cohérente.  
Complexité : forte. Risque : fort. Bénéfice : réduit les états partiels.

### MIG-DRAG-005 — Retrait progressif des adaptateurs UI de collage

Objectif : déplacer le chemin final de membres Core/projection vers les IDs
canoniques, sans toucher prématurément aux besoins de chaîne UI/XML.  
Complexité : forte. Risque : fort. Bénéfice : prépare l'élimination de
`groups[*]["nodes"]` dans le release.

## 13. Recommandation finale

Ne pas migrer `_update_edge_highlights()` comme un simple déplacement de code
vers le Core. Sa majorité fonctionnelle — curseur, hit-test, classement et
rendu — appartient à l'UI. Le prochain travail utile est de caractériser le
flux puis de formaliser la proposition de snap et la simulation en lecture
seule sur une géométrie temporaire explicite.

La règle à viser est la suivante :

```text
UI : drag et projection temporaire
    -> SnapProposal (IDs Core + paramètres + snapshot/version de projection)
    -> Core : validation pure de la proposition
    -> UI : aperçu
    -> Core : transaction finale
    -> UI : projection rafraîchie
```

Ainsi, `TopologyWorld` demeure l'autorité des faits persistés, `_last_drawn`
reste une projection contrôlée, et l'aperçu ne dépend plus implicitement d'une
géométrie différente de celle qu'il prétend valider.
