# MIG-CACHE-001 — Audit de reconstruction de last_drawn depuis TopologyWorld

## 1. Objet, périmètre et conclusion

Cet audit prépare la SPEC de migration qui fera de last_drawn un cache graphique reconstruisible. Il est documentaire : aucun comportement n'a été modifié.

La cible est :

TopologyWorld
→ projection graphique
→ CanvasObjectsCollection.entries / last_drawn
→ Canvas

Le constat principal est favorable : pour un TopologyElement existant, le Core porte déjà sa géométrie intrinsèque, sa pose monde et son état de miroir. Il est donc possible de reconstruire ses points affichés O/B/L. En revanche, le Core ne porte pas encore de contrat complet pour l'identité catalogue, les libellés historiques exacts, l'ordre de projection et les données d'interaction écran. Une migration sûre doit séparer ces responsabilités au lieu de tenter de tout déduire implicitement.

## 2. Cartographie globale de last_drawn

### 2.1 Propriétaire actuel

TriangleViewerManual crée CanvasObjectsCollection au démarrage puis garde last_drawn comme alias de collection.entries. La liste active est reliée à ScenarioAssemblage.last_drawn par _bind_canvas_objects() et lors de _set_active_scenario().

La recherche de texte recense actuellement 149 occurrences dans src :

| Fichier | Occurrences | Rôle dominant |
|---|---:|---|
| src/assembleur_tk.py | 122 | interactions, transformations, rendu, scénarios |
| src/assembleur_tk_mixin_bg.py | 11 | cadrage et redessins de fond |
| src/assembleur_tk_mixin_dictionary.py | 8 | association catalogue/mots et redessin |
| src/assembleur_io.py | 5 | persistance de projection XML |
| src/canvas_objects_collection.py | 2 | alias et contrat de collection |
| src/assembleur_core.py | 1 | champ ScenarioAssemblage.last_drawn |

Ce nombre est une cartographie des lecteurs et des mutations, pas le nombre de vérités métier. La majorité des lectures servent au Canvas, au hit-test, aux outils visuels et aux exports.

### 2.2 Création et remplacement structurel

| Producteur | Fichier | Rôle | Nature |
|---|---|---|---|
| constructeur TriangleViewerManual | assembleur_tk.py | crée une CanvasObjectsCollection vide et son alias | initialisation UI |
| _bind_canvas_objects(entries) | assembleur_tk.py | rend la liste d'un scénario active | activation/scénario |
| _place_dragged_triangle() | assembleur_tk.py | ajoute un triangle manuel à la projection active | création UI puis Core |
| loadScenarioXml() | assembleur_io.py | importe points XML puis replace_all() | chargement de projection |
| buildLegacyLastDrawnFromTopology() | assembleur_sim.py | produit last_drawn des scénarios automatiques à partir de PlacedTriangles + Core | projection de simulation |
| _autoRebuildWorldGeometryScenario() | assembleur_tk.py | réécrit les points monde de scénarios automatiques depuis leur géométrie locale et auto_geom_state | reconstruction UI auto |
| CanvasObjectsCollection remove/remove_many/replace_all/clear | canvas_objects_collection.py | mutations structurelles et index | infrastructure collection |

Aucun de ces producteurs n'est aujourd'hui un rebuild générique depuis TopologyWorld seul.

### 2.3 Consommateurs

Les familles fonctionnelles sont :

| Famille | Principaux consommateurs | Dépendance réelle |
|---|---|---|
| Rendu Canvas | _redraw_from(), _draw_triangle_screen(), contours, surbrillances | points, labels, id, mirrored |
| Interaction | hit-test, sélection, drag, rotation, flip, snap | points, topoElementId, cache pick |
| Navigation Core → UI | get_last_drawn_entries_for_core_group(), CanvasObjectsCollection index | topoElementId |
| UI auxiliaire | fond/cadrage, mots de dictionnaire, PDF, différences de scénario | points, id, labels |
| Persistance XML | saveScenarioXml(), loadScenarioXml() | id, points, mirrored, topoElementId |
| Auto | activation et transform global auto | points locaux/projection de simulation |
| Tests et diagnostics | tests de migration, dump collection | structure et identité |

## 3. Champs observés dans les entrées

Les dictionnaires ne sont pas déclarés par un schéma unique. Les champs stables et les champs temporaires identifiés sont les suivants.

| Champ | Origine actuelle | Reconstructible depuis Core ? | Classe | Commentaire |
|---|---|---|---|---|
| id | catalogue Excel, PlacedTriangle.triangleId ou XML | **Partiellement** | donnée de projection/catalogue | TopologyWorld n'impose pas une correspondance élément → id catalogue. Le manuel met triRank dans meta ; ce contrat n'est pas universel. |
| labels | catalogue, PlacedTriangle ou XML enrichi par loader | **Partiellement / souvent oui** | projection UI | TopologyElement possède vertex_labels et vertex_types, mais la convention de tuple UI et les fallbacks sont externes. |
| pts | placement manuel, XML, simulation, transformations Tk | **Oui** | cache graphique aujourd'hui ; donnée Core cible | Utiliser vertex_local_xy et localToWorld avec la pose de l'élément. |
| world_pts | état de drag ponctuel | Non comme champ persistant | état UI temporaire | Variante transitoire de pts pendant un geste. |
| mirrored | placement/flip UI | **Oui** | donnée de pose Core / cache UI | TopologyElement.get_pose() retourne mirrored. La duplication doit devenir dérivée. |
| topoElementId | création Core, simulation, XML | **Oui** | lien d'identité | C'est le lien canonique projection → Core et doit rester obligatoire. |
| topoGroupId | projection automatique ou loader | **Oui** | cache de compatibilité runtime | Derivé de world.get_group_of_element(topoElementId). Les scénarios manuels le lisent encore. |
| orient | placement manuel depuis dataframe | **Oui si meta complet** | métadonnée de projection | Le manuel écrit orient ; l'élément Core reçoit meta orient. Contrat à caractériser. |
| _pick_poly | _rebuild_pick_cache() | Oui depuis pts + zoom/offset | cache UI temporaire | Polygone écran, jamais persistant. |
| _pick_pts | _rebuild_pick_cache() | Oui depuis pts + zoom/offset | cache UI temporaire | Points écran, jamais persistant. |

La cible ne doit pas mettre les caches écran dans le Core. Ils doivent être invalidés/recalculés après chaque rebuild de projection.

## 4. Flux Core → UI déjà disponibles

### 4.1 APIs Core directement réutilisables

| Fichier | API | Rôle | Réutilisation |
|---|---|---|---|
| assembleur_core.py | TopologyWorld.elements | inventaire des TopologyElement | élevée, mais l'ordre doit être fourni |
| assembleur_core.py | getElementPose(elementId) | retourne R, T, mirrored | élevée |
| assembleur_core.py | TopologyElement.vertex_local_xy | points locaux canoniques | élevée |
| assembleur_core.py | TopologyElement.localToWorld() | projection d'un point local par la pose actuelle | élevée |
| assembleur_core.py | get_group_of_element() | dérive topoGroupId | élevée pour compatibilité runtime |
| assembleur_core.py | getGroupElementIds(), getLiveGroupIds() | navigation par groupe Core | élevée pour fonctions de groupe, pas pour l'ordre Canvas |
| assembleur_core.py | ensureBoundary(), getBoundarySegments(), getIncidentBoundarySegments() | contours et bords Core | élevée pour contours/snap/highlights |

Ces APIs suffisent à reconstruire une géométrie monde par élément. Elles ne suffisent pas à déterminer un ordre de dessin, ni à retrouver à coup sûr un id catalogue externe.

### 4.2 Projections existantes mais incomplètes

| Fichier | Fonction | Rôle actuel | Réutilisation possible |
|---|---|---|---|
| assembleur_sim.py | buildLegacyLastDrawnFromTopology() | convertit PlacedTriangles en dictionnaires et calcule topoGroupId depuis le Core | moyenne : prouve le contrat de sortie, mais réutilise les points déjà portés par PlacedTriangle |
| assembleur_sim.py | PlacedTriangle.toLegacyDict() | adapte une projection de simulation au dictionnaire UI | faible à moyenne : pas de calcul Core → points |
| assembleur_tk.py | _autoEnsureLocalGeometry() | construit last_drawn_local depuis la projection auto existante | faible : source encore UI |
| assembleur_tk.py | _autoRebuildWorldGeometryScenario() | applique auto_geom_state à last_drawn_local puis réécrit pts | moyenne : mécanisme de rebuild UI, mais source locale hors Core |
| assembleur_tk.py | _redraw_from() | dessine une projection fournie | élevée : consommateur naturel d'un nouveau rebuild |
| assembleur_tk.py | _get_projected_elements_for_core_group() | résout Core group → entrées Canvas par topoElementId | élevée comme façade de lecture après rebuild |
| assembleur_io.py | loadScenarioXml() | remplit la projection depuis XML et rattache les IDs au Core | faible : conserve des points XML comme autorité de fait |

Conclusion : il existe des briques de conversion et de rendu, mais pas encore une fonction qui part exclusivement de TopologyWorld pour créer chaque entrée Canvas.

## 5. Cartographie des écritures directes de last_drawn

### 5.1 Géométrie et pose

| Fichier | Fonction / flux | Mutation | Sens actuel |
|---|---|---|---|
| assembleur_tk.py | _place_dragged_triangle() | ajoute labels, pts, id, mirrored, orient, topoElementId, topoGroupId | UI → Core : la pose est ensuite synchronisée |
| assembleur_tk.py | _move_group_world() | translate les trois points de chaque membre | UI → Core via _sync_group_elements_pose_to_core() |
| assembleur_tk.py | drag manuel continu et commit de move | déplace pts pendant le geste et au dépôt | UI → Core |
| assembleur_tk.py | rotation de groupe | remplace les trois points par rotation | UI → Core |
| assembleur_tk.py | _ctx_flip_selected() | reflète pts et bascule mirrored | UI → Core |
| assembleur_tk.py | collage/commit dans _on_canvas_left_up() | applique la transformation rigide finale aux entrées mobiles | UI → Core |
| assembleur_tk.py | dégroupage / application du résultat Core | remplace ou répartit des projections et synchronise les poses | mixte, encore couplé |
| assembleur_tk.py | _autoRebuildWorldGeometryScenario() | recalcule pts auto à partir de last_drawn_local et auto_geom_state | UI auto → Core via _autoSyncAllTopoPoses() |
| assembleur_io.py | loadScenarioXml() | construit pts depuis XML | XML projection → Core déjà importé |
| assembleur_sim.py | création de branches / PlacedTriangles | conserve points de simulation | simulation → projection ; poses Core sont aussi posées |
| canvas_objects_collection.py | add/remove/replace_all/remove_many | structure de collection, pas points | infrastructure |

### 5.2 Champs non géométriques

- labels sont complétés à la lecture XML depuis la dataframe ou par fallback.
- id est ajouté lors du placement et provient de la simulation ou de XML.
- topoElementId est écrit après création d'un TopologyElement, à la lecture XML et à la projection automatique.
- topoGroupId est dérivé à la lecture XML et dans la projection automatique ; il n'est pas autoritaire.
- _pick_poly et _pick_pts sont écrits par le cache de picking et dépendent de la vue écran.
- les annotations textuelles historiques ne sont pas un champ de last_drawn ; elles ont été supprimées du runtime.

Le mouvement, la rotation et le flip sont les principaux obstacles : ils modifient directement le cache, puis demandent au Core de copier la pose calculée.

## 6. Audit de _sync_group_elements_pose_to_core()

### 6.1 Contrat effectif

Signature : _sync_group_elements_pose_to_core(core_group_id, scen=None).

1. récupère le scénario et son TopologyWorld ;
2. obtient les elementIds par world.getGroupElementIds(core_group_id) ;
3. trouve les entrées Canvas correspondantes via CanvasObjectsCollection, ou crée une collection éphémère pour un scénario inactif ;
4. lit O, B, L depuis pts ou world_pts ;
5. lit les coordonnées locales de l'élément Core ;
6. applique une réflexion locale si mirrored est vrai ;
7. calcule R et T par fit orthonormal de Kabsch ;
8. appelle world.setElementPose(elementId, R, T, mirrored).

La fonction transfère donc précisément une pose par élément, et non une pose de groupe. La résolution des membres est déjà Core-first ; la direction du flux reste inversée.

### 6.2 Appelants identifiés

| Flux appelant | Raison |
|---|---|
| _place_dragged_triangle() | poser dans le Core le triangle créé graphiquement |
| _move_group_world() | persister une translation UI |
| rotation manuelle | persister une rotation UI |
| _ctx_flip_selected() | persister un flip UI |
| commit de drag/collage | persister la pose finale avant/après attachment |
| dégroupage | synchroniser les groupes reconstruits |
| _autoSyncAllTopoPoses() | faire suivre le Core après transformation auto de la projection |

### 6.3 Hypothèses et dépendances

- chaque élément du groupe possède une entrée projetée indexable par topoElementId ;
- chaque entrée a les points O/B/L, sans contrôle complet avant indexation ;
- l'ordre des indices locaux 0, 1, 2 correspond à O, B, L ;
- mirrored dans la projection est la seule source de la réflexion au moment du fit ;
- la pose UI est rigide et non dégénérée ;
- les scénarios inactifs conservent une liste mutable historique ;
- CanvasObjectsCollection est disponible pour le scénario actif.

### 6.4 Ce qui peut être conservé et ce qui doit disparaître

À conserver :

- le contrat de pose par élément du Core ;
- world.setElementPose() et son invalidation de géométrie conceptuelle ;
- les validations de rigidité et la convention mirrored.

À faire disparaître de la trajectoire normale :

- l'utilisation de pts Canvas comme entrée de vérité ;
- le fit Kabsch déclenché après chaque transformation UI ;
- l'écart actif/inactif fondé sur des collections différentes ;
- la nécessité de synchroniser explicitement après move/rotate/flip.

La fonction reste utile temporairement comme adaptateur de compatibilité, notamment pour chargement de vieux XML et pour tout flux UI non encore inversé. Elle deviendra un outil de migration ou devra être retirée une fois les gestes Core-first.

## 7. Faisabilité de rebuild_last_drawn_from_core()

### 7.1 Données déjà suffisantes

Pour chaque elementId sélectionné :

1. récupérer l'élément dans world.elements ;
2. récupérer ses points locaux O/B/L dans vertex_local_xy ;
3. projeter chaque point par element.localToWorld() ;
4. récupérer mirrored avec getElementPose() ;
5. dériver topoGroupId avec get_group_of_element();
6. construire une entrée avec topoElementId.

Les poses Core sont persistées dans topoSnapshot ; le rebuild sera donc stable après chargement XML dès lors que le lien projection → element est conservé.

### 7.2 Données manquantes ou à contractualiser

| Besoin de projection | Couvert aujourd'hui ? | Risque |
|---|---|---|
| ordre de rendu | Non garanti par world.elements | moyen : ordre visuel et tests peuvent changer |
| id catalogue | Non universellement garanti dans le Core | fort : mots, listbox, XML et dictionnaire l'utilisent |
| labels UI exacts | partiellement dans vertex_labels | moyen : conventions O/B/L et fallbacks |
| ordre des sommets O/B/L | dépend de l'index 0/1/2 actuel | fort : projection fausse si convention non universelle |
| annotations textuelles historiques | supprimées du runtime | sans objet |
| état de sélection/drag | hors Core | faible : doit être invalidé après rebuild |
| caches écran | hors Core | faible : recalculer |
| géométrie locale auto | aujourd'hui last_drawn_local | fort : auto_geom_state doit être absorbé par des poses Core ou rester explicitement un transform de vue |
| anciennes projections XML | points XML encore lus | moyen : caractériser la réconciliation Core/points |

### 7.3 Risques majeurs

1. Une reconstruction sur world.elements sans ordre explicite rend le dessin, les hit-tests indexés et les associations id potentiellement instables.
2. Le manuel et l'auto n'ont pas encore un contrat unique pour id catalogue → TopologyElement.
3. Les opérations auto ont un transform global partagé auto_geom_state et une géométrie locale stockée hors Core ; les inverser sans migration dédiée modifierait leur sémantique.
4. Certains consommateurs utilisent directement les indices de last_drawn pour sélection, surbrillance et dictionnaire. Une replace_all impose de remapper ou d'invalider ces indices.
5. Les points XML sont encore chargés en projection ; le choix de prévalence Core/points doit être explicite afin de détecter les écarts de pose historiques.

## 8. Architecture proposée pour la SPEC_MIG_CACHE_001

### 8.1 Responsabilités

TopologyWorld :
- unique source des éléments, géométrie locale, poses, miroir, attachments et groupes ;
- ne connaît ni Canvas, ni labels de style, ni sélection.

Service de projection Core → UI :
- reçoit world et un contrat de projection de scénario ;
- choisit un ordre explicite d'elementIds ;
- calcule pts depuis vertex_local_xy et localToWorld ;
- dérive mirrored et topoGroupId ;
- ajoute id et labels à partir d'un registre de métadonnées de scénario/catéglogue ;
- retourne une liste neuve d'entrées.

CanvasObjectsCollection :
- reçoit la liste via replace_all ;
- maintient les index topoElementId / triangle id ;
- ne contient pas de logique métier ou de calcul de pose.

TriangleViewerManual :
- déclenche le rebuild après une mutation Core ;
- invalide pick-cache, sélection transitoire et aides visuelles ;
- appelle _redraw_from().

### 8.2 Nouvelle API à spécifier, sans l'implémenter

Une future fonction rebuild_last_drawn_from_core() devrait recevoir au minimum :

- TopologyWorld ;
- une séquence ordonnée d'elementIds ;
- un fournisseur de métadonnées de projection par elementId : id catalogue, labels et éventuellement ordre ;
- une politique explicite pour topoGroupId de compatibilité.

Elle ne doit pas recevoir une ancienne liste de points comme source de vérité.

Sortie : une nouvelle liste de dictionnaires de projection compatibles avec CanvasObjectsCollection. Les caches _pick_poly/_pick_pts ne font pas partie de cette sortie.

### 8.3 Séquence cible par transformation

move / rotate / flip / collage :
1. résoudre les éléments concernés depuis le Core ;
2. muter les poses Core via une API Core de transformation ;
3. reconstruire la projection concernée ou le scénario entier ;
4. replace_all dans CanvasObjectsCollection ;
5. invalider les caches UI et redessiner.

La SPEC devra décider si le rebuild est complet ou partiel. Un rebuild complet est plus simple et plus sûr au début ; un rebuild partiel peut venir seulement après caractérisation de l'ordre, des sélections et des index.

## 9. Opportunités de simplification

- getGroupElementIds() et CanvasObjectsCollection.get_many_by_topology_ids() constituent déjà une frontière saine Core → projection.
- topoGroupId est entièrement dérivable ; le conserver est une compatibilité, pas une contrainte de modèle.
- _sync_group_elements_pose_to_core() concentre le flux inverse dans une seule fonction, ce qui réduit le nombre de points à remplacer.
- _redraw_from() accepte déjà une projection complète : il n'a pas besoin d'être réécrit pour la première étape.
- les frontières/contours viennent désormais du Core et ne dépendent plus de nodes legacy ; ils bénéficieront directement d'un rebuild.

## 10. Recommandations pour la future SPEC

1. Définir avant tout un ProjectionMetadataStore stable, au minimum elementId → id catalogue et labels.
2. Définir un ordre de projection explicite : ScenarioAssemblage.orderedElementIds pour l'auto, un équivalent persistant pour le manuel.
3. Écrire des tests de caractérisation Core pose → O/B/L, mirrored et labels, avant de migrer les gestes.
4. Introduire d'abord un rebuild de diagnostic comparant projection existante et projection Core sans modifier le flux utilisateur.
5. Migrer le placement manuel, puis move, rotate, flip, collage et dégroupage un par un ; maintenir _sync_group_elements_pose_to_core() comme pont provisoire.
6. Traiter auto_geom_state dans un chantier séparé : il est le principal état géométrique encore hors Core pour les scénarios automatiques.
7. N'autoriser les optimisations de rebuild partiel qu'après suppression des consommateurs qui dépendent implicitement des indices.

## 11. Conclusion

La migration est techniquement faisable, car TopologyWorld dispose de la géométrie locale et des poses nécessaires à la construction des points monde. Le problème n'est pas la projection géométrique elle-même ; il est le contrat de métadonnées et d'ordre qui accompagne le Canvas.

La SPEC_MIG_CACHE_001 doit donc viser une inversion progressive du flux, en conservant provisoirement le contrat de dictionnaire actuel :

Core pose
→ projection O/B/L, mirrored, topoElementId, topoGroupId
→ CanvasObjectsCollection
→ Canvas

et non :

Canvas pts
→ fit de pose
→ Core.
