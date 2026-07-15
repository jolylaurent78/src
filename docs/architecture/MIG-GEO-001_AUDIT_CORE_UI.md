# MIG-GEO-001 Audit Core ↔ UI

**Statut :** audit statique, lecture seule du code existant.  
**Date :** 15 juillet 2026.  
**Périmètre :** `TopologyWorld`, `TopologyElement`, `ScenarioAssemblage`, `TriangleViewerManual`, persistance XML v4, simulation automatique et structures UI immédiatement liées.  
**Non inclus :** changement de code, migration, refactoring, modification XML, exécution d'instrumentation ou correction de données.

## 1. Executive Summary

L'application possède déjà les fondations de la frontière cible : le Core porte une géométrie locale par `TopologyElement`, une pose monde par élément (`R`, `T`, `mirrored`), des attachements et des groupes topologiques. La formule `local → monde` est exécutable dans `TopologyElementPose2D.local_to_world()` (`src/assembleur_core.py:354-387`).

Mais `_last_drawn` est aujourd'hui beaucoup plus qu'une projection graphique. Ses points monde O/B/L sont créés et modifiés directement par les interactions ; ils alimentent le rendu, le hit-test, les collisions, les choix d'arêtes, les contours de groupes, les groupes UI et le XML. Les poses Core sont ensuite souvent recalculées depuis ces points par `_sync_group_elements_pose_to_core()` (`src/assembleur_tk.py:686-765`). La frontière actuelle est donc **hybride et UI-first pour la géométrie manuelle**.

La cible demandée reste pertinente :

```text
TopologyWorld / état de scénario métier
          ↓ projection déterministe
_last_drawn / modèle graphique projeté
          ↓ conversion monde-écran
Canvas et interactions
```

Elle ne signifie pas la suppression de `_last_drawn`. Elle exige en revanche de distinguer ce qui doit être reconstruit du Core de ce qui n'est qu'un cache, une clé de liaison ou un état d'interaction.

Les risques majeurs sont les suivants :

- **double représentation persistée :** le XML v4 sauvegarde à la fois le snapshot Core et O/B/L UI, sans les comparer au chargement ;
- **synchronisation inverse non contrôlée :** l'ajustement Kabsch de l'UI vers le Core ne calcule pas de résidu ;
- **deux groupes :** groupe topologique dans le Core, groupe ordonné dans l'UI ;
- **miroir :** `orient` intrinsèque, `mirrored` UI et `mirrored` Core coexistent ; une docstring Core annonce une matrice différente de celle réellement exécutée ;
- **scénarios auto :** `last_drawn_local` et `auto_geom_state` portent une transformation de placement hors de `TopologyWorld`.

La recommandation documentaire pour MIG-GEO-002 est de traiter `_last_drawn` comme un **DTO de projection par scénario**, sans l'éliminer, et de formaliser d'abord : (1) projection des O/B/L depuis une pose Core, (2) liaison stable UI ↔ élément, (3) séparation entre groupe topologique et groupe/ordre d'interaction UI, (4) convention unique de miroir. Aucun de ces points ne constitue une demande d'implémentation dans ce rapport.

## 2. Architecture actuelle

### 2.1 Core

`TopologyWorld` contient les `elements`, les groupes DSU, les nœuds/arêtes, les attachements et les caches conceptuels. `TopologyElement` contient :

- la forme intrinsèque (`edge_lengths_km`, `intrinsic_sides_km`) ;
- le repère local (`local_frame`, `vertex_local_xy`) ;
- les étiquettes et types de sommets ;
- `meta["orient"]` ;
- la pose monde `R`, `T`, `mirrored`.

Le Core recalcule également la géométrie conceptuelle à partir des coordonnées locales et des poses (`TopologyWorld._refreshConceptGeom()`, `src/assembleur_core.py:1901-1918`). Les attachements, les composantes et le contour conceptuel sont donc bien Core.

### 2.2 UI et `ScenarioAssemblage`

`ScenarioAssemblage` stocke `last_drawn`, `groups` et `topoWorld`. Pour le scénario manuel, `last_drawn` et `groups` sont des références directes aux structures du viewer (`src/assembleur_core.py:56-79`, `src/assembleur_tk.py:580-591`). Le changement de scénario réaffecte `self._last_drawn = scen.last_drawn` et `self.groups = scen.groups` (`assembleur_tk.py:4502-4535`).

Le viewer est donc à la fois le contrôleur d'interaction, le détenteur du cache UI courant et, pour le scénario manuel, le propriétaire effectif de plusieurs mutations de données.

### 2.3 Canvas

Le canvas ne porte pas le modèle monde : `_draw_triangle_screen()` convertit O/B/L de monde vers écran avec `_world_to_screen()` puis crée les primitives Tk (`assembleur_tk.py:6234-6305`). Le hit-test principal relit toutefois les points monde de `_last_drawn` et reconvertit vers écran (`:7636-7670`). Les identifiants de primitives canvas ne sont pas utilisés comme identités métier.

### 2.4 Flux réel, distinct de la cible

```text
Chemin manuel actuel
Canvas / geste → mutation de _last_drawn["pts"] → inférence R,T,mirrored → TopologyWorld
                                                   ↘ groupes UI / XML / rendu

Capacité Core existante
TopologyElement.local + pose → point monde / géométrie conceptuelle

Cible de contrat
TopologyWorld → projection de scénario → _last_drawn → Canvas
Canvas → intention/UI state → commande ou synchronisation explicitement validée vers Core
```

Il existe aussi un flux Core → UI partiel : dégroupement renvoie des éléments/groupes Core, chargement XML remappe les identités, et le compas peut lire directement les coordonnées conceptuelles Core. Il n'existe pas de projection générale manuelle qui reconstruise les entrées O/B/L de `_last_drawn` depuis tous les éléments Core.

## 3. Inventaire de `_last_drawn`

### 3.1 Forme de base observée

Une entrée représente un triangle affichable. Les producteurs ne garantissent pas tous les mêmes champs : placement manuel, XML, simulation automatique et cache écran sont des variantes. La forme minimale effectivement attendue par le rendu est :

```python
{
  "labels": (str, str, str),
  "pts": {"O": ndarray|tuple, "B": ndarray|tuple, "L": ndarray|tuple},
  "id": int|None,
  "mirrored": bool,
}
```

Les champs réellement rencontrés sont détaillés dans l'appendice. Ils se répartissent ainsi :

| Champ | Type observé | Rôle apparent | Variantes / remarque |
|---|---|---|---|
| `labels` | tuple/list de 3 chaînes | libellés O/B/L pour dessin et liste | XML ne le sérialise pas directement ; reconstruit depuis Excel ou vide |
| `pts` | dict O/B/L → vecteur 2D | géométrie monde affichée | lu très largement ; doit être complet pour rendu/hit-test |
| `id` | entier, rarement `None` | identifiant du triangle de catalogue | sert aussi au remappage vers `Txx` au chargement XML |
| `mirrored` | booléen | flag de flip UI/pose | sémantique distincte de `orient` intrinsèque |
| `orient` | `"CW"`/`"CCW"` ou absent | orientation issue du catalogue | créé au placement manuel, consommé par simulation ; Core l'a dans `meta` |
| `topoElementId` | chaîne | lien vers `TopologyElement` | clé UI → Core ; rétabli par inférence au chargement |
| `topoGroupId` | chaîne | lien vers groupe DSU canonique | redondant/dérivable depuis `topoElementId` dans le Core |
| `group_id` | entier ou `None` | groupe UI contenant l'entrée | n'est pas l'identité du groupe Core |
| `group_pos` | entier ou `None` | position dans l'ordre UI de groupe | réindexé lors de fusion/scission/suppression |
| `_pick_pts` | dict O/B/L → coordonnées écran | cache de hit-test | régénéré après zoom/pan/redraw, jamais XML |
| `_pick_poly` | liste de trois coordonnées écran ou `None` | cache polygonal de hit-test | régénéré, non persisté |
| `_chain_edge_in` / `_chain_edge_out` | `"OB"`/`"BL"`/`"LO"` | métadonnées temporaires du chaînage automatique | présentes dans certaines branches auto, transférées vers les nodes UI |

`world_pts` n'est pas un champ persistant d'entrée : il est utilisé par le drag-preview et accepté comme secours par le synchroniseur. Il doit être considéré comme une variante de transport de `pts`, non comme un second état de scénario.

### 3.2 Structures graphiques équivalentes ou associées

| Structure | Porteur | Rôle | Rapport à `_last_drawn` |
|---|---|---|---|
| `viewer.groups[gid]` | UI/scénario | `{id, nodes, bbox, topoGroupId}` | groupe de manipulation et d'ordre ; indexe les `tid` |
| `groups[*].nodes[*]` | UI/scénario | `tid`, `edge_in/out`, parfois `vkey_in/out` | métadonnées de chaîne et de menu, séparées du triangle |
| `scen.last_drawn_local` | auto seulement | copie de points relatifs à l'ancre auto | état intermédiaire de reconstruction auto, hors Core |
| `auto_geom_state` | viewer, partagé auto | `ox`, `oy`, `thetaDeg` | transforme les scénarios auto ; non porté par `TopologyWorld` |
| `_pick_cache_valid` | viewer | drapeau d'invalidation | cache UI global, non entrée de triangle |
| `_sel`, `_drag`, `_edge_choice` | viewer | sélection, preview et assistance de collage | état transitoire d'interaction, non persistant |
| Canvas Tk | UI | primitives graphiques | sortie du rendu, aucun contrat métier |
| `_tri_words` | viewer | association triangle → mot/position dictionnaire | donnée de fonctionnalité UI distincte, persistée dans XML `words` |

## 4. Classification des données

La catégorie décrit la cible de propriété, pas seulement l'emplacement actuel. Une même donnée peut avoir une copie de projection ; la colonne « propriétaire cible » indique l'autorité à viser.

| Donnée actuelle | Catégorie | Propriétaire cible | Justification |
|---|---|---|---|
| forme locale, longueurs, types/labels de sommets | **A** | `TopologyElement` | nécessaire à toute reconstruction et déjà dans le Core |
| pose `R`, `T`, `mirrored` | **A** | `TopologyElement.pose` | position/orientation métier reconstruisible ; déjà dans le Core |
| `orient` intrinsèque | **A** | `TopologyElement.meta` ou modèle intrinsèque typé | définit la construction locale ; sa copie UI est redondante |
| éléments, attachements, DSU, groupe canonique | **A** | `TopologyWorld` | topologie et rattachements métier |
| `id` catalogue / `topoElementId` | **B** | identifiant Core exposé à la projection | `id` doit se relier de façon stable à l'élément ; UI ne doit pas être la source de l'identité |
| `topoGroupId` | **B** | dérivé de `TopologyWorld` | utile à l'UI mais ne doit pas concurrencer le DSU Core |
| O/B/L monde (`pts`) | **C** | projection depuis local + pose | aujourd'hui actif métier de fait ; cible = DTO projeté |
| `labels` dans l'entrée UI | **C** | projection des labels Core/catalogue | nécessaires au dessin, non à la topologie si Core est complet |
| `mirrored` dans l'entrée UI | **C** | projection de `pose.mirrored` | doit refléter le Core ; exception à analyser pendant prévisualisation |
| `group_id`, `group_pos` | **E**, avec partie à qualifier | état d'interaction / ordre de présentation | pas le groupe DSU ; peut être dérivé ou maintenir une séquence UI choisie |
| `groups.nodes.edge_in/out/vkey_in/out` | **A candidate** + E | décision de contrat à prendre | la topologie Core porte les attachements, pas une séquence UI explicite ; utilisé pour des liens/dégroupements/chaînage |
| `bbox`, `_pick_pts`, `_pick_poly` | **D** | cache UI | dérivables de `pts`, zoom et offset |
| `_chain_edge_in/out` | **E** ou **A candidate** | algorithme/scénario, à trancher | état de construction auto ; devient significatif seulement si l'ordre d'assemblage est une donnée métier |
| `_sel`, `_drag`, hover, surbrillances | **E** | UI | transitoire, non sauvegardé comme état de scénario |

### Point de vigilance sur les groupes

Il serait incorrect de classer indistinctement « les groupes » en A ou en E. La **composante de connexion** est A et appartient déjà à `TopologyWorld`. L'**ordre linéaire de présentation/manipulation**, les arêtes `edge_in/out` et les positions `group_pos` sont aujourd'hui UI. Ils deviennent une donnée A seulement si les futures règles métier exigent de reconstruire exactement une chaîne d'assemblage, un parcours ou une sémantique de frontière à partir de cet ordre. Cette décision reste ouverte.

## 5. Flux Core → UI

### 5.1 Flux actuellement disponibles

| Source Core | Lecteur UI | Donnée transmise | Observation |
|---|---|---|---|
| `TopologyElement.localToWorld()` | Core conceptuel | point monde issu de local+pose | utilisé dans `_refreshConceptGeom`, pas pour remplir le cache UI général |
| `getBoundarySegments()` | `_group_outline_segments_topo()` | segments topologiques de contour | l'UI relit ensuite les points de `_last_drawn` pour interpoler les extrémités (`tk:6816-6869`) |
| `degrouperAtNode()` | `_applyDegrouperResultToTk()` | IDs de groupes et d'éléments | UI reconstruit ses groupes et effectue aussi un déplacement visuel |
| `getConceptNodeWorldXY()` | compas/ancres | coordonnées monde de nœud conceptuel | flux Core direct, avec fallback UI possible |
| snapshot XML Core | `loadScenarioXml()` | éléments, poses, attachements | réimporté après les triangles UI ; aucun recalcul O/B/L |
| `get_group_of_element()` | placement, collage, load | groupe canonique | copié dans `topoGroupId` UI |

### 5.2 Flux cible proposé

```text
TopologyWorld.elements
  [vertex_local_xy, pose, labels, element_id]
          ↓ ProjectScenarioGeometry
liste de TriangleProjection
  [topoElementId, id, labels, pts, mirrored]
          ↓
_last_drawn (cache/remplacement atomique par scénario)
          ↓
pick cache, bbox, Canvas
```

Le nom `ProjectScenarioGeometry` décrit un contrat, pas une classe ou une fonction à créer ici. Sa sortie doit être complète, déterministe, ordonnée par une clé explicitement définie et sans objets Tk.

## 6. Flux UI → Core

### 6.1 Flux réel actuel

| Départ UI | Passage | Arrivée Core | Preuve |
|---|---|---|---|
| drag depuis catalogue | `_place_dragged_triangle` ajoute `pts`, crée élément puis sync pose | `add_element_as_new_group`, `setElementPose` | `assembleur_tk.py:7247-7349` |
| move groupe | translation de O/B/L | `setElementPose` par élément | `:9531-9556`, sync `:686-765` |
| rotation | rotation directe O/B/L | `setElementPose` par élément | `:6652-6694` |
| miroir | réflexion O/B/L + toggle flag | `setElementPose(... mirrored=...)` | `:9471-9528` |
| collage | transformation O/B/L, sync, création d'attachements | `apply_attachments` puis groupes Core | `:10049-10231` |
| suppression | lecture de `topoElementId` | `removeElementsAndRebuild` | `:9123-9208` |
| dégroupement | cible UI sélectionnée | `degrouperAtNode` | `:7934-7947` |
| XML | `id`, groupes et IDs UI | import de snapshot + relink | `assembleur_io.py:732-965` |

L'interface ne devrait pas transmettre un nouveau monde au Core comme vérité implicite. Dans le contrat cible, elle transmet une intention d'interaction et la liaison stable d'élément/groupe ; le Core accepte ou refuse la mutation. Pendant une prévisualisation, des points UI temporaires peuvent exister, mais ils doivent être explicitement identifiés comme tels.

### 6.2 Synchronisation inverse actuelle

`_sync_group_elements_pose_to_core()` applique Kabsch/SVD aux trois points locaux et trois points monde. Il force une rotation de déterminant positif et reporte la réflexion dans `mirrored`. Il n'évalue cependant ni erreur par sommet ni rigidité résiduelle. Une entrée UI non isométrique peut donc mettre à jour le Core avec sa meilleure approximation.

Ce mécanisme est une **frontière de compatibilité** à caractériser, non un contrat cible d'autorité. Il confirme que la pose Core est calculable depuis l'UI mais pas que l'UI est une projection fidèle du Core.

## 7. Analyse Mirror / Orientation

### 7.1 Données et producteurs

| Notion | Emplacement | Producteur | Consommateurs |
|---|---|---|---|
| `orient` | Excel, `_last_drawn` manuel, `TopologyElement.meta` | `_triangle_from_index`, placement, simulation | construction locale Core, simulation |
| orientation locale | `vertex_local_xy` | `_try_build_default_local_coords` inverse Y de L pour CW | projection Core, inférence de pose |
| `mirrored` UI | entrée `_last_drawn` | placement, flip, XML, simulation | dessin, sync UI → Core |
| `mirrored` Core | `TopologyElement.pose` | `setElementPose`, `flipGroup`, import snapshot | `local_to_world`, géométrie conceptuelle, snapshot |
| réflexion monde | interaction ou `flipGroup` | `_ctx_flip_selected` ou `TopologyWorld.flipGroup` | points UI ou poses Core |

### 7.2 Redondances et écarts

`orient` et `mirrored` n'ont pas la même intention : `orient` est incorporé à la forme locale de catalogue ; `mirrored` est une réflexion de la pose ultérieure. L'UI utilise néanmoins un booléen `mirrored` avant que la pose soit inférée. `_triangle_from_index()` construit un triangle CW et renvoie initialement `mirrored=True` (`assembleur_tk.py:5692-5705`), alors que `_place_dragged_triangle()` réinitialise explicitement le flag UI à `False` et stocke `orient` (`:7293-7299`). Cette normalisation implicite doit être considérée comme une convention de transition, non comme une définition simple du champ.

Le Core a aussi deux formulations divergentes : la docstring de `TopologyElementPose2D` écrit `M = diag(-1,+1)`, tandis que `mirror_matrix()` et les synchroniseurs utilisent `diag(+1,-1)`. Le comportement exécutable est donc la réflexion suivant l'axe X local ; le texte ne l'est pas.

### 7.3 Frontière cible

- **A :** `orient` intrinsèque et `pose.mirrored` sont autoritaires dans le Core.
- **C :** `last_drawn.mirrored` est la copie projetée de la pose retenue pour le dessin.
- **E :** un miroir de preview peut exister dans l'état de drag/interaction, mais ne doit pas modifier silencieusement la projection persistante avant validation.
- Toute API future doit expliciter si elle traite l'orientation de forme ou le flip de pose ; le nom générique « mirror » ne suffit pas.

## 8. Données métier absentes du Core

### 8.1 Candidats réels

| Donnée hors Core | Pourquoi elle peut être métier | Impact potentiel | Décision de propriété à prendre |
|---|---|---|---|
| ordre des `groups.nodes`, `group_pos`, `edge_in/out`, `vkey_in/out` | encode un parcours de chaîne, des liens de voisinage et des choix d'arêtes ; utilisé après collage/dégroupement et en auto | impossible de reconstruire exactement la présentation/chaîne avec le seul DSU | intégrer à un modèle de scénario/topologie seulement si cet ordre est une contrainte métier ; sinon le déclarer projection UI |
| `auto_geom_state` (`ox`, `oy`, `thetaDeg`) | détermine la position/rotation de scénarios auto et reconstruit `last_drawn` | état géométrique autoritaire hors Core avant resync des poses | propriété de scénario ou de projection à formaliser ; pas de propriété Canvas |
| `last_drawn_local` auto | référence relative servant à reconstruire le monde | dépend de l'ancre et des transformations auto | cache de projection si reconstruisible du Core ; sinon révèle un manque dans le modèle auto |
| points UI O/B/L | aujourd'hui source de transformations manuelles | toute divergence peut devenir persistance/XML | ne sont pas une donnée Core cible ; doivent être projection validée |

### 8.2 Données qui ne sont pas absentes du Core

- la forme, les longueurs, les labels de sommets et l'orientation intrinsèque sont déjà portés par `TopologyElement` ;
- l'identité élément et le groupe topologique sont déjà portés par `TopologyWorld` ;
- le miroir de pose est déjà porté par `TopologyElement.pose` ;
- les attachements et les relations topologiques sont déjà Core.

Ainsi, le risque principal n'est pas d'abord une absence de structure Core pour les triangles simples. C'est le fait que l'UI conserve des copies actives, et que les séquences de groupes/auto ne disposent pas encore d'un contrat de scénario clairement séparé du canvas.

## 9. Contrat Core ↔ UI proposé

### 9.1 Entrées autoritaires du Core

Pour chaque élément projetable :

```text
element_id stable
identité catalogue si applicable
vertex_labels/types
vertex_local_xy et convention orient
pose R, T, mirrored
attachements et groupe topologique canonique
```

Pour le scénario : les éléments présents, leur ordre de projection explicite si l'UI en requiert un, et les données de groupe qui sont réellement métier après décision dédiée.

### 9.2 Sortie de projection attendue

Chaque entrée `_last_drawn` projetée devrait contenir au minimum :

```text
topoElementId        clé B obligatoire
id                   identifiant catalogue affichable, si présent
labels               projection des labels Core/catalogue
pts[O,B,L]           projection locale + pose
mirrored             copie de pose.mirrored, si nécessaire au dessin
```

`topoGroupId`, `group_id`, `group_pos`, bbox et pick-cache n'ont pas la même nature : ils sont soit dérivés, soit spécifiques à l'interaction. Ils ne doivent pas réintroduire une autorité concurrente sur la topologie ou sur la pose.

### 9.3 Commandes et interactions

```text
UI : hit-test/preview à partir de la projection
UI → Core : intention + clés stables + paramètres de geste
Core : mutation validée des poses/attachements/topologie
Core → UI : nouvelle projection de cache
```

Le contrat laisse explicitement une place à un cache `_last_drawn`, au hit-test et à une prévisualisation fluide. Il interdit seulement qu'un point de cache devienne implicitement la source métier durable après une action sans synchronisation et validation explicites.

### 9.4 Persistance

Le snapshot Core doit être l'état métier. Une projection UI peut être sauvegardée pour performance, diagnostic ou compatibilité, mais son statut doit être explicite : elle est reconstituée ou comparée au chargement, jamais une seconde autorité silencieuse. Cette phrase décrit la cible contractuelle ; l'actuel XML v4 conserve encore les deux représentations sans comparaison.

## 10. Recommandations MIG-GEO-002

Les recommandations suivantes sont des conclusions d'architecture pour une mission ultérieure. Elles n'autorisent ni n'incluent une modification dans MIG-GEO-001.

1. **Figer le contrat de projection élémentaire.** La projection O/B/L depuis `vertex_local_xy` + pose existe ; elle doit devenir la référence mesurable avant toute migration d'interaction.
2. **Conserver `_last_drawn` comme DTO/couche UI.** Préserver ses usages de rendu, hit-test, optimisation et preview, mais séparer ses champs projetés des caches et de l'interaction.
3. **Rendre la liaison stable obligatoire.** `topoElementId` doit être la clé de la projection ; les `tid` et positions de liste restent locaux à l'UI et remappables.
4. **Isoler le contrat de groupe.** Décider si ordre, `edge_in/out` et `vkey_in/out` sont un modèle métier de chaîne ou une structure UI. Ne pas les assimiler automatiquement au DSU Core.
5. **Unifier miroir/orientation.** Définir une matrice exécutée, la différence entre orientation intrinsèque et flip, et la source de `mirrored` projeté.
6. **Traiter les scénarios auto comme un cas de contrat.** `auto_geom_state` et `last_drawn_local` doivent être classés comme état de scénario/projection, jamais comme détail de canvas.
7. **Préparer la persistance à une seule autorité.** Avant toute évolution XML, identifier explicitement snapshot Core, projection de compatibilité et contrôle de cohérence attendu.

## Conclusion

La réponse à la question centrale est nette : `TopologyWorld` doit détenir la forme, la pose, les attachements, les groupes topologiques, l'orientation intrinsèque et les identités stables. `_last_drawn` doit demeurer une projection UI riche — points monde, labels de dessin, lien d'élément, caches écran et support d'interaction — mais ne pas décider seul d'un état métier durable.

Le cas qui nécessite une décision explicite n'est pas le canvas : c'est la structure de groupe ordonnée et les transformations globales auto, aujourd'hui placées autour de `_last_drawn`. Cette frontière doit être qualifiée avant de généraliser la projection Core → UI.

Les tables détaillées de lectures, écritures et persistance sont dans [l'appendice](MIG-GEO-001_AUDIT_CORE_UI_APPENDIX.md).
