# AUDIT_CODE_MIG_GEO_001 — audit contradictoire du contrat géométrique

**Statut :** audit documentaire statique du code existant.  
**Date :** 15 juillet 2026.  
**Périmètre :** `src/assembleur_core.py`, `src/assembleur_tk.py`, `src/assembleur_io.py`, les modules directement appelés pour les choix d'arêtes et les scénarios automatiques, ainsi que `tests/`.  
**Hors périmètre :** toute correction, refonte, migration, instrumentation exécutée, modification XML ou implémentation d'un validateur.

## 1. Résumé exécutif

La conclusion principale est que le diagnostic de MIG-GEO-001 est **confirmé** : l'application a déjà un modèle Core capable de représenter la géométrie locale d'un triangle et sa pose monde, mais l'exécution manuelle courante demeure majoritairement **UI-first**. Les points monde O/B/L de `_last_drawn` sont créés, transformés, consultés pour l'interaction et sauvegardés avant ou indépendamment de la pose Core. La pose Core est ensuite inférée par ajustement rigide.

Le Core dispose bien d'une projection explicite : `TopologyElementPose2D.local_to_world()` et `TopologyElement.localToWorld()` calculent un point monde à partir des coordonnées locales, de `R`, de `T` et de `mirrored`. La pose contient donc, en théorie, les paramètres nécessaires à une transformation rigide ou réfléchie. En revanche, il n'existe pas de constructeur général de `_last_drawn` depuis le Core, ni de comparaison systématique entre les trois points UI et les trois points reconstruits. La reconstructibilité est donc **mathématiquement disponible élément par élément**, mais **non démontrée comme propriété fiable de l'état applicatif actuel**.

`_last_drawn` ne peut pas être qualifié aujourd'hui de simple cache. Il est un état géométrique opérationnel : rendu, hit-test, détection de collisions, choix d'arêtes, transformation de groupes, groupes UI, chargement/sauvegarde XML et maintien de scénarios y accèdent directement. Le document cible conserve donc sa valeur comme architecture cible, non comme description de l'existant.

Le premier comparateur utile est **GEO-INV-003** : pour chaque élément relié à une entrée UI, reprojeter les sommets locaux par la pose Core, comparer aux O/B/L UI avec une tolérance explicite et rapporter l'écart sans corriger. Cette priorité doit toutefois signaler séparément la convention de miroir : la docstring de `TopologyElementPose2D` annonce `diag(-1,+1)` alors que `mirror_matrix()` retourne `diag(+1,-1)` ; la synchronisation Tk utilise également `diag(+1,-1)`. Le comportement exécutable est cohérent sur ce dernier axe, mais le contrat textuel est contradictoire.

Le meilleur premier candidat de migration contrôlée n'est pas une interaction d'assemblage : c'est la **reconstruction d'affichage d'un triangle isolé déjà posé depuis sa géométrie locale et sa pose Core**, sous observation de GEO-INV-003. Le collage ne doit pas être le premier candidat : il combine transformation UI, synchronisation de pose, création d'attachements, transaction Core et fusion manuelle des groupes UI.

## 2. Périmètre inspecté et méthode

### 2.1 Références lues

- `docs/MIG-GEO-001_Analyse_architecturale.docx` : hypothèses, invariants et décisions à instruire.
- `docs/ARCHITECTURE_CIBLE_TOPOLOGIQUE.md` : cible où la topologie est la source de vérité, la géométrie monde une projection, `_last_drawn` un cache et le canvas une vue.
- `docs/STABILISATION_UI_CORE_PHASE1.md` : stabilisation de l'architecture hybride actuelle et nécessité d'un validateur transversal.

### 2.2 Limite de l'audit

Cet audit est statique : il suit les appels, les structures et les séquences de mutations visibles dans le code. Il ne prouve pas les valeurs produites sur tous les scénarios réels, les erreurs d'arrondi, les chemins déclenchés exclusivement par l'interface, ni les données XML historiques. Toute conclusion qui dépend de ces exécutions est marquée **INDÉTERMINÉE STATIQUEMENT**.

### 2.3 Convention de statut

| Statut | Sens dans ce rapport |
|---|---|
| **CONFIRMÉE PAR LE CODE** | Le flux et les données visibles établissent l'affirmation. |
| **PARTIELLEMENT CONFIRMÉE** | Le comportement existe, mais avec un chemin divergent, une absence de garde ou une portée limitée. |
| **INFIRMÉE** | Le code met en œuvre une autre règle. |
| **INDÉTERMINÉE STATIQUEMENT** | Il faut exercer le code ou observer des données pour conclure. |

## 3. Modèle géométrique réel observé

### 3.1 Les cinq couches sont réelles

| Couche | Structure réelle | Rôle observé | Autorité effective actuelle |
|---|---|---|---|
| Catalogue / intrinsèque | ligne `df`, longueurs `len_OB`, `len_OL`, `len_BL`, `orient` | fabrique l'élément au placement | entrée de forme, pas position monde |
| Locale Core | `TopologyElement.vertex_local_xy`, `local_frame`, `meta["orient"]` | repère O/B/L stable, en km | autorité de forme dans le Core |
| Pose Core | `TopologyElementPose2D.R`, `.T`, `.mirrored` | transformation locale → monde | autorité déclarée par le Core, mais souvent écrite depuis l'UI |
| Monde UI | `TriangleViewerManual._last_drawn[*]["pts"]` et `ScenarioAssemblage.last_drawn` | O/B/L, interaction et rendu | autorité opérationnelle de nombreux chemins manuels |
| Écran | `zoom`, `offset`, `_world_to_screen`, `_screen_to_world` | conversion canvas seulement | vue, non métier |

**Conclusion : CONFIRMÉE PAR LE CODE.** Les couches ne sont pas une simple décomposition documentaire : elles coexistent dans les structures et dans la persistance.

### 3.2 Géométrie locale Core

`_build_local_triangle()` (`assembleur_core.py:42-53`) construit O=(0,0), B=(OB,0) et L par la loi des cosinus. `TopologyElement._try_build_default_local_coords()` (`:474-532`) remappe ces points vers les indices effectifs O/B/L et, si `meta["orient"] == "CW"`, inverse la coordonnée Y locale de L. Les coordonnées résultantes sont placées dans `vertex_local_xy`.

`TopologyElement` documente que cette géométrie locale reste figée (`:446-459`), et les transformations de groupe Core modifient des poses plutôt que ces coordonnées. Cette immutabilité est un contrat voulu et largement respecté par les chemins inspectés ; aucune écriture applicative de `vertex_local_xy` après construction/import n'a été trouvée.

**Conclusion : CONFIRMÉE PAR LE CODE**, sous réserve que l'import de snapshot puisse introduire toute valeur numérique fournie par le XML JSON (`_importPhysicalSnapshot`, `:3953-3967`).

### 3.3 Pose et projection locale → monde

La formule exécutable est :

```text
p_effectif = M × p_local              si mirrored
p_monde    = R × p_effectif + T
```

avec `R` et `T` stockés dans `TopologyElementPose2D` (`:370-387`). `TopologyElement.localToWorld()` (`:471-472`) et `TopologyWorld.elementLocalToWorld()` (`:3133-3137`) exposent cette projection.

**Conclusion : CONFIRMÉE PAR LE CODE.** Il n'y a ni facteur d'échelle ni pivot stocké séparément : une pose représente une isométrie affine plane (rotation, translation et une réflexion locale éventuelle). Une mise à l'échelle ou une déformation UI ne peut donc pas être reproduite exactement par ce modèle.

### 3.4 Anomalie de convention de miroir

La docstring de `TopologyElementPose2D` affirme `M = diag(-1,+1)` (`assembleur_core.py:361-367`), mais `mirror_matrix()` retourne `[[1,0],[0,-1]]` (`:379-381`). La synchronisation Tk emploie ce même second choix (`assembleur_tk.py:711`). Ainsi, le code exécutable et le synchroniseur sont alignés sur une réflexion par rapport à l'axe X local ; le commentaire de contrat ne l'est pas.

**Conclusion : CONFIRMÉE PAR LE CODE** pour l'incohérence documentaire interne. Elle ne prouve pas une divergence de géométrie à l'exécution, mais elle interdit de considérer GEO-INV-005 comme formalisé sans clarification.

## 4. Cartographie des structures et identités

| Structure | Porte | Identité principale | Géométrie portée | Risque |
|---|---|---|---|---|
| `TopologyElement` | Core | `element_id`, généralement `Txx` | local, pose, sommets et arêtes | l'UI dépend d'un pont `topoElementId` |
| `TopologyWorld` | Core | groupes DSU, éléments, attachements | topologie et géométrie conceptuelle dérivée | ne connaît pas les index `tid` UI |
| `ScenarioAssemblage` | Core/UI hybride | instance scénario | `last_drawn`, `groups`, `topoWorld` | scénario manuel par références directes |
| `_last_drawn[t]` | Viewer/UI | index `tid` instable après suppression ; `id` catalogue | `pts` O/B/L, `mirrored`, groupes et liens Core | sert à la fois de modèle UI et de rendu |
| `viewer.groups` | UI | entier `gid` | ordre de nœuds, bbox, lien `topoGroupId` | duplication de la composante Core |
| XML v4 | persistance | `id`, `tid`, groupes, `Txx` par inférence | points UI *et* snapshot Core | double représentation persistée |

`ScenarioAssemblage` déclare explicitement que, pour le scénario manuel, `last_drawn` et `groups` sont des références directes aux structures runtime du viewer (`assembleur_core.py:56-79`). `TriangleViewerManual` les relie effectivement à l'initialisation et lors du changement de scénario (`assembleur_tk.py:580-591`, `:4533-4535`).

**Conclusion : CONFIRMÉE PAR LE CODE.** La notion de « cache par scénario » existe, mais elle est un cache mutable partagé par alias dans le scénario manuel, non une projection reconstruite.

## 5. Cartographie des écritures et lectures de `_last_drawn`

### 5.1 Écritures structurantes

| Flux | Producteur | Mutation | Effet Core |
|---|---|---|---|
| Placement | `_place_dragged_triangle()` (`tk:7247-7349`) | `append` des `pts` UI avant l'élément Core | pose calculée ensuite par sync |
| Déplacement groupe | `_move_group_world()` (`tk:9531-9556`) | addition sur O/B/L de chaque triangle | `_sync_group_elements_pose_to_core()` |
| Rotation groupe | `_on_canvas_motion_update_drag()` (`tk:6652-6694`) | rotation des points à partir d'un snapshot | sync à chaque mouvement |
| Miroir groupe | `_ctx_flip_selected()` (`tk:9471-9528`) | réflexion O/B/L + bascule `mirrored` | sync après transformation |
| Collage | `_on_canvas_left_up()` (`tk:10049-10231`) | applique une matrice rigide aux points UI | pose sync puis attachements Core |
| Dégroupement | `_applyDegrouperResultToTk()` (`tk:7997-8152`) | recompose groupes et translate visuellement des points | le Core a déjà éclaté les attachements |
| Suppression | `_ctx_delete_group()` (`tk:9123-9208`) | reconstruit la liste et remappe les `tid` | suppression/rebuild Core ensuite |
| XML | `loadScenarioXml()` (`io:732-965`) | recrée `pts` et groupes depuis XML | importe ensuite snapshot Core |
| Auto | `_autoRebuildWorldGeometryScenario()` (`tk:4376-4416`) | reconstruit `scen.last_drawn` depuis une géométrie locale spécifique à l'auto | resynchronisation des poses ailleurs |

### 5.2 Lectures métier directes

`_last_drawn` est lu par le rendu (`_redraw_from`, `tk:6003`), le pick cache et le hit-test (`:5035-5054` et appels), les choix d'arêtes (`assembleur_edgechoice.py`), les collisions de groupes via `_group_shape_from_nodes()` (`assembleur_core.py:28-39`), l'assistance de collage, le calcul des bboxes, le miroir, les outils dictionnaire, la persistance XML et la sélection de scénarios. Les modules de simulation créent eux aussi des `last_drawn` indépendants (`assembleur_sim.py`).

**Conclusion : CONFIRMÉE PAR LE CODE.** `_last_drawn` est lu hors du seul rendu ; il est donc une dépendance métier opérationnelle actuelle.

### 5.3 Flux Core → `_last_drawn`

Le flux général Core → UI n'existe pas pour les scénarios manuels. Les exceptions sont :

- le résultat topologique de `degrouperAtNode()` retourne des identifiants d'éléments, que `_applyDegrouperResultToTk()` traduit en `tid` et groupes UI ;
- le chargement XML importe un snapshot Core et relie l'UI au Core, sans recalculer les points UI ;
- les scénarios automatiques ont une reconstruction de `last_drawn` depuis une géométrie locale propre au sous-système auto, qui n'est pas la projection générale de `TopologyWorld`.

**Conclusion : PARTIELLEMENT CONFIRMÉE.** Il existe des retours Core → UI d'identité ou de structure, non une projection monde uniforme de tous les éléments Core vers les points O/B/L UI.

## 6. Analyse des poses et de l'inverse monde → pose

### 6.1 Inférence actuelle

`_sync_group_elements_pose_to_core()` (`assembleur_tk.py:686-765`) lit O/B/L dans `_last_drawn`, lit les trois points locaux de l'élément, applique préalablement le miroir si le flag UI le demande, puis calcule `R` et `T` par Kabsch/SVD. Il force `det(R) > 0` et porte la réflexion dans le booléen `mirrored` avant d'appeler `TopologyWorld.setElementPose()`.

**Conclusion : CONFIRMÉE PAR LE CODE.** L'inverse utilisé n'est ni une pose par sommet, ni un angle issu d'un seul segment : il ajuste les trois sommets.

### 6.2 Limites de l'inférence

Le synchroniseur ne calcule ni résidu maximal, ni RMS, ni test d'orthogonalité de la pose obtenue, ni erreur par sommet après reprojection. Kabsch donnera la meilleure rotation rigide même si les trois points UI ne sont pas une isométrie de la géométrie locale. La pose Core peut donc devenir une approximation silencieuse d'un état UI déformé ou incohérent.

**Conclusion : CONFIRMÉE PAR LE CODE.** C'est le principal motif technique pour lequel la réponse à « peut-on reconstruire fiablement `_last_drawn` maintenant ? » est non au niveau global.

### 6.3 Contenu de la pose et miroir

Une pose contient `R` 2×2, `T` 2D et `mirrored`. Si `vertex_local_xy` est complet et si O/B/L UI résulte réellement d'une isométrie suivant la convention exécutable de miroir, ces données sont suffisantes pour récupérer chaque sommet monde. Le sens intrinsèque `orient` est déjà incorporé à la géométrie locale lors de la construction (`core:521-525`) ; `mirrored` exprime le flip ultérieur de pose.

**Conclusion : PARTIELLEMENT CONFIRMÉE.** La pose porte l'information nécessaire pour le modèle théorique courant, y compris le miroir de pose, mais elle ne porte aucune échelle, aucun historique de pivot et ne garantit pas que les points UI persistés correspondent à ce modèle.

## 7. Projection locale → monde et monde → pose

| Sens | Fonction réelle | État de fiabilité observé |
|---|---|---|
| Local → monde, un point | `TopologyElementPose2D.local_to_world()` ; `TopologyElement.localToWorld()` | déterministe pour les données présentes |
| Local → monde, triangle | aucune fonction générique qui énumère O/B/L et alimente l'UI | formule disponible mais adaptation absente |
| Monde UI → pose | `_sync_group_elements_pose_to_core()` | Kabsch sur trois points, sans validation de résidu |
| Pose → cache UI | aucune fonction manuelle générale | absent |
| Écran ↔ monde | `_world_to_screen()` / `_screen_to_world()` (`tk:5115`, `:5970`) | inverse affine de vue ; hors domaine métier |

La projection écran applique bien `x_screen = offset_x + x_world × zoom` et `y_screen = offset_y - y_world × zoom`. Elle ne doit pas être confondue avec la transformation locale → monde ; la négation de Y est un choix de canvas.

**Conclusion : CONFIRMÉE PAR LE CODE.** L'option « `TopologyWorld → _last_drawn` existe déjà » est **INFIRMÉE** dans le sens d'une fonction de reconstruction applicative complète ; seule la brique de calcul par point existe.

## 8. Matrice des opérations sensibles

| Opération | État actuel et dépendances | Autorité opérationnelle | Risque de divergence | Comportement cible attendu |
|---|---|---|---|---|
| Placement initial | UI crée `Pw`, puis Core crée l'élément puis infère pose | UI → Core | échec Core après append UI ; absence de comparaison | commande Core puis projection cache |
| Déplacement triangle | en pratique déplacement du groupe singleton | UI → Core | points modifiés avant pose, sans résidu | transformation de pose puis cache |
| Déplacement groupe | translation O/B/L, sync pose par élément | UI → Core | désalignement possible si sync incomplète | transformation Core de toutes poses puis cache |
| Rotation triangle | opération de groupe, singleton possible | UI → Core | pivot UI non inscrit dans le Core ; approximation silencieuse | transformation de pose puis cache |
| Rotation groupe | rotation depuis snapshot O/B/L, sync à chaque mouvement | UI → Core | cache mutable pendant interaction | aperçu UI puis commit Core explicite |
| Miroir triangle/groupe | réflexion UI et toggle du flag, puis Kabsch | UI → Core | convention M textuellement contradictoire ; double sémantique `orient`/`mirrored` | opération Core définie par convention unique puis cache |
| Collage | transform UI, sync pose, attachements Core, fusion groupes UI | hybride UI/Core | manque d'atomicité transverse ; plusieurs identités | commande topologique atomique et projection |
| Dégroupement | `world.degrouperAtNode`, puis groupes UI et décalage visuel | Core → UI → Core indirect | déplacement visuel post-split sans sync immédiate visible dans le même helper | split topologique puis projection déterministe |
| Suppression | liste UI/remap puis `removeElementsAndRebuild` | UI → Core | exception Core après mutation UI ; `tid` renumérotés | suppression Core transactionnelle puis cache régénéré |
| Chargement XML | lit UI puis importe Core snapshot et relie | double source | aucune égalité géométrique UI/Core | une représentation autoritaire, autres dérivées/validées |
| Sauvegarde XML | écrit snapshot Core et O/B/L UI | double source | états divergents sauvegardés ensemble | autorité métier + cache éventuellement marqué dérivé |
| Duplication scénario | deepcopy UI/groupes + clone Core | double clonage | cohérence inter-copies non vérifiée | clone d'un état autoritaire puis cache régénéré |
| Changement scénario | réaffecte les alias `last_drawn`/`groups`, redraw | cache scénario comme état | ordre d'activation et auto partagés | reconstruction indépendante depuis scénario métier |

Les futurs comportements de la dernière colonne décrivent la cible conceptuelle des documents de référence. Ils ne constituent pas des corrections proposées dans cet audit.

## 9. Groupes : UI, Core et conséquences géométriques

Le Core modélise le groupe comme une composante topologique : une pose reste portée par élément et non par groupe (`_sync_group_elements_pose_to_core`, `tk:686-693`). Le viewer maintient cependant un groupe UI séparé avec ordre de nœuds, bbox, `group_id`, `group_pos` et un lien vers `topoGroupId`.

Lors du dégroupement, le Core supprime les attachements concernés, reconstruit les composantes et renvoie `mainGroupId`, `newGroupIds` et les éléments déplacés (`core:3734-3871`). L'UI recompose ensuite sa propre liste de groupes. Elle applique enfin un décalage de 30 pixels aux nouveaux groupes (`tk:8124-8144`) en modifiant leurs points monde. Ce décalage est une décision de présentation qui affecte pourtant la géométrie monde UI ; le helper de translation lui-même ne synchronise pas la pose Core.

**Conclusion : PARTIELLEMENT CONFIRMÉE** pour l'hypothèse « dégroupement Core-first puis reconstruction UI ». L'ordre topologique est bien Core-first, mais le résultat contient aussi une mutation géométrique UI de séparation visuelle, ce qui empêche de le décrire comme une pure projection Core → UI.

## 10. Miroir : état, preuve et risque spécifique

Trois notions distinctes coexistent :

1. `orient` de catalogue, incorporé dans le Y local de L lors de la construction du Core ;
2. `mirrored` dans la pose Core, appliqué avant la rotation ;
3. `mirrored` dans l'entrée UI, basculé au flip et utilisé par la synchronisation.

Le Core possède une opération `TopologyWorld.flipGroup()` (`core:3087-3131`) qui reflète le monde puis refactorise chaque transformation en rotation pure + flag `mirrored`. Le chemin d'interface `_ctx_flip_selected()` ne l'appelle pas : il reflète les points UI et invoque l'inférence de pose. Les deux voies ne sont donc pas le même mécanisme, même si elles visent le même modèle de résultat.

**Conclusion : CONFIRMÉE PAR LE CODE.** L'hypothèse MIG selon laquelle le miroir est le risque principal est fondée. Il faut distinguer une contradiction documentaire de matrice, l'orientation intrinsèque et le flip utilisateur avant de prétendre valider GEO-INV-005.

## 11. XML : double persistance et règle de restauration

`saveScenarioXml()` (`assembleur_io.py:389-537`) écrit :

- un `topoSnapshot` JSON issu de `TopologyWorld._exportPhysicalSnapshot()`, contenant local, pose R/T/mirrored, éléments et attachements (`core:3877-3935`) ;
- les triangles UI, avec O/B/L et `mirrored` ;
- les groupes UI et leurs `tid`.

`loadScenarioXml()` charge d'abord les points et groupes UI (`io:732-844`), recrée ensuite un `TopologyWorld`, importe le snapshot (`:852-861`) et remappe strictement les identités UI/Core (`:890-948`). Ce contrôle vérifie qu'un élément existe et qu'un groupe UI correspond à une composante DSU, mais ne reprojette pas les poses et ne compare pas leurs O/B/L aux points XML.

**Conclusion : CONFIRMÉE PAR LE CODE.** Le chargement restaure deux vérités géométriques potentielles. Il contrôle fortement les liens d'identité et de groupe, mais pas la cohérence géométrique entre les deux.

L'affirmation selon laquelle le XML « tranche » une autorité en cas de divergence est **INFIRMÉE** : il ne choisit ni le snapshot Core ni les points UI pour recalculer l'autre ; il les accepte ensemble si les liens structurels passent.

## 12. Duplication et isolation des scénarios

La duplication manuelle utilise `copy.deepcopy` pour `last_drawn` et `groups`, et `clonePhysicalState()` pour le Core (`assembleur_tk.py:4695-4735`). `clonePhysicalState()` repose sur l'export/import physique et inclut local, pose et attachements (`assembleur_core.py:3874-4019`). Cela constitue une bonne séparation des objets principaux.

La garantie n'est cependant pas complète : la duplication ne copie pas explicitement `view_state` ni `map_state`, et l'architecture auto comporte des états globaux de géométrie et de vue partagés (`auto_geom_state`, `auto_view_state`, `auto_map_state`). Surtout, aucun test trouvé ne compare après mutation les deux scénarios dupliqués.

**Conclusion : PARTIELLEMENT CONFIRMÉE** pour GEO-INV-009. L'isolation des listes, groupes et Core du chemin manuel est codée ; l'isolation fonctionnelle complète des scénarios, en particulier auto et vue/carte, doit être observée dynamiquement.

## 13. Évaluation des invariants GEO-INV-001 à GEO-INV-010

| Invariant | Statut | Éléments de preuve et exception |
|---|---|---|
| GEO-INV-001 — validité numérique | **PARTIELLEMENT CONFIRMÉE** | conversions `float` nombreuses et matrices 2D ; aucun balayage global `isfinite`, ni validation de R/T/XML avant tous usages. |
| GEO-INV-002 — complétude O/B/L | **PARTIELLEMENT CONFIRMÉE** | placement et sync accèdent explicitement aux trois sommets ; le chargeur XML accepte l'absence d'un sommet et stocke alors un dict incomplet (`io:739-758`). |
| GEO-INV-003 — équivalence local+pose/monde | **PARTIELLEMENT CONFIRMÉE** | formule Core et inférence Kabsch existent ; aucun comparateur, tolérance ou résidu n'est appliqué. |
| GEO-INV-004 — rigidité | **PARTIELLEMENT CONFIRMÉE** | move/rotate/flip UI sont rigides et pose est isométrique ; Kabsch accepte néanmoins une entrée UI non rigide en l'approchant. |
| GEO-INV-005 — cohérence miroir | **PARTIELLEMENT CONFIRMÉE** | `orient` local et `mirrored` pose sont séparés ; deux chemins de flip existent ; la docstring et la matrice ne concordent pas. |
| GEO-INV-006 — cohérence des attachements | **PARTIELLEMENT CONFIRMÉE** | validation Core des références/coverages (`validate_world`) ; pas de comparaison explicite de coïncidence projetée UI/Core après collage. |
| GEO-INV-007 — transformation de groupe | **PARTIELLEMENT CONFIRMÉE** | transformations UI appliquées à tous les sommets du groupe ; groupes UI et Core sont distincts et les chemins de collage/dégroupement les modifient séparément. |
| GEO-INV-008 — post-commit UI/Core | **INFIRMÉE** comme propriété globale | aucun validateur transversal ; plusieurs commits suivent une mutation UI et le XML ne compare pas. |
| GEO-INV-009 — isolation scénarios | **PARTIELLEMENT CONFIRMÉE** | deepcopy/clone pour duplication manuelle ; état auto partagé et absence de tests d'isolation. |
| GEO-INV-010 — fraîcheur cache dérivé | **INFIRMÉE** comme propriété globale | `_last_drawn` n'a ni version, ni invalideur central, ni reconstruction systématique ; il est muté directement. |

## 14. Validation contradictoire des hypothèses de `MIG-GEO-001`

| Hypothèse du document préalable | Statut | Preuve contradictoire ou confirmante |
|---|---|---|
| Architecture hybride : local, pose, UI monde, groupes et XML coexistent | **CONFIRMÉE PAR LE CODE** | sections 3, 4 et 11 ; structures distinctes dans Core, viewer et XML. |
| Le Core porte une géométrie locale et une pose utilisable pour le monde | **CONFIRMÉE PAR LE CODE** | `TopologyElementPose2D.local_to_world`, `TopologyElement.localToWorld`. |
| Déplacement manuel UI-first puis pose Core | **CONFIRMÉE PAR LE CODE** | `_move_group_world` et rotation modifient O/B/L puis sync. |
| Rotation manuelle UI-first puis pose Core | **CONFIRMÉE PAR LE CODE** | `_on_canvas_motion_update_drag:6676-6693`. |
| Miroir UI ou convention mixte | **CONFIRMÉE PAR LE CODE** | flip UI direct, `orient` local, `mirrored` pose, `flipGroup` Core non appelé. |
| Dégroupement Core-first avant UI | **PARTIELLEMENT CONFIRMÉE** | `degrouperAtNode` précède l'adaptation UI, mais l'UI translate ensuite le monde visuel. |
| Chargement XML double source | **CONFIRMÉE PAR LE CODE** | O/B/L UI et snapshot Core sont lus sans comparaison géométrique. |
| Rendu directement depuis `_last_drawn` | **CONFIRMÉE PAR LE CODE** | `_redraw_from(self._last_drawn)` et consommateurs associés. |
| La projection Core doit être vérifiée avant inversion d'autorité | **CONFIRMÉE PAR LE CODE** comme nécessité d'architecture | projection élémentaire existe mais aucun constructeur/contrôle global ; le besoin de mesure est directement constaté. |
| Le collage ne doit pas être la première migration | **CONFIRMÉE PAR LE CODE** comme recommandation de risque | séquence multi-frontières `tk:10049-10231`, non atomique UI/Core. |
| `_last_drawn` peut déjà être supprimé | **INFIRMÉE** | nombreuses lectures métier et persistance directe. |
| La pose suffit dans tous les cas à reconstruire l'état UI | **INDÉTERMINÉE STATIQUEMENT** | elle suffit au modèle rigide ; l'absence de comparateur et les fichiers existants empêchent d'affirmer l'équivalence réelle. |

## 15. Risques, divergences et indéterminées

### 15.1 Risques confirmés

- **Double autorité persistée :** le XML conserve Core et O/B/L UI sans validateur géométrique.
- **Écriture UI avant Core :** placement, translation, rotation, miroir et collage modifient le cache avant synchronisation.
- **Synchronisation approximative silencieuse :** Kabsch ne rejette pas un triangle UI non rigide.
- **Identité UI indexée :** la suppression renumérote les `tid`, ce qui rend les groupes UI sensibles aux erreurs de remapping.
- **Miroir non contractuellement stabilisé :** écart entre docstring et matrice réellement utilisée.
- **Transaction transverse absente :** `apply_attachments()` ne couvre que la topologie ; les groupes et points UI sont déjà modifiés par ailleurs.

### 15.2 Questions indéterminées statiquement

- L'écart numérique réel local+pose/UI pour des scénarios normaux et historiques.
- Le comportement après exception entre mutation UI et commit Core de collage/suppression.
- La complétude O/B/L des XML v4 réellement stockés.
- La cohérence des miroirs après séquences combinant orientation CW, flip, chargement et duplication.
- L'isolation effective des scénarios automatiques lorsque la géométrie globale auto change.
- La conservation des attachements géométriquement coïncidents après toutes les séquences de collage/dégroupement.

## 16. Instrumentation minimale éventuellement nécessaire

La présente section ne demande aucune implémentation. Si une décision future exige de lever les indéterminées, l'instrumentation minimale et réversible est :

1. après une synchronisation UI → Core, reprojeter O/B/L de chaque élément et journaliser l'écart maximal et le résidu par sommet ;
2. identifier l'opération, le scénario, `tid`, `topoElementId`, `R`, `T`, `mirrored`, `orient` et le sens de miroir effectif ;
3. ne jamais corriger, arrondir ou réécrire le cache à cette étape ;
4. au chargement XML, comparer les deux représentations avant le premier redraw et signaler seulement ;
5. autour d'une duplication, prendre une empreinte des poses, `last_drawn` et groupes de la source puis vérifier qu'une mutation de la copie ne les altère pas.

Le premier seuil doit rester un paramètre de diagnostic explicite, distinct des tolérances de hit-test ou de canvas.

## 17. Recommandation architecturale

### 17.1 Premier validateur recommandé

**GEO-INV-003, comparateur unitaire par élément :**

```text
pour chaque entrée UI reliée à un TopologyElement :
    lire ses trois vertex_local_xy dans l'ordre O/B/L
    les projeter par pose local_to_world
    comparer chaque point à _last_drawn[tid]["pts"]
    rapporter max, RMS, détail par sommet et contexte miroir
```

Il doit être observateur : pas de réparation, pas de reconstruction d'un scénario topologique, pas de nouveau XML, pas de rendu de remplacement. Il mesure précisément le contrat qui conditionne toute migration Core → UI.

### 17.2 Ordre de lecture des résultats

- Si GEO-INV-003 échoue hors miroir, la priorité n'est pas de remplacer `_last_drawn`, mais de caractériser les chemins UI → pose.
- S'il échoue uniquement selon `mirrored`/`orient`, GEO-INV-005 devient le diagnostic suivant.
- S'il réussit sur des états manuels et XML représentatifs, une reconstruction de cache d'un triangle isolé devient le plus petit essai contrôlé.
- Les échecs de liaison XML ou de groupe doivent être distingués des écarts géométriques : le chargeur vérifie déjà surtout les premiers.

## 18. Périmètre révisé de MIG-GEO-001

La définition recommandée est à conserver, avec une précision de résultat attendue :

> **MIG-GEO-001 — Formalisation et stabilisation du contrat « géométrie locale / pose / coordonnées monde ».**

Elle ne doit pas être redéfinie comme « migration immédiate de `TopologyWorld` vers `_last_drawn` ». Le code confirme qu'il s'agit d'abord d'un chantier de connaissance, de comparabilité et de contrat. Le périmètre révisé est :

- inventaire exhaustif des producteurs/consommateurs de points UI et poses Core ;
- convention exécutable unique de miroir, distincte de l'orientation intrinsèque ;
- comparaison local+pose contre O/B/L sur les opérations et XML représentatifs ;
- caractérisation de l'isolation de scénarios ;
- décision ultérieure et fondée sur mesure du premier flux Core → cache.

Restent explicitement hors périmètre : suppression de `_last_drawn`, refonte de `TopologyWorld`, nouveau XML, migration de tous les gestes UI, refonte du collage/groupes, reconstruction topologique complète, slots, variantes, catalogue et correction silencieuse des incohérences.

## 19. Réponses directes aux décisions demandées

1. **Peut-on reconstruire fiablement `_last_drawn` maintenant depuis le local Core et les poses ?**  
   **Non, pas comme garantie applicative globale.** La formule nécessaire existe et est déterministe par élément, mais aucun flux général ni contrôle d'égalité ne confirme que chaque O/B/L UI persistant est son image rigide exacte. Une reconstruction peut être calculée ; sa fidélité aux états existants doit être mesurée.

2. **La pose contient-elle toutes les informations, notamment le miroir ?**  
   **Oui pour le modèle rigide visé, sous condition.** `R`, `T` et `mirrored`, avec la géométrie locale qui incorpore `orient`, suffisent pour positionner un triangle sans échelle. Elles ne mémorisent ni déformation, ni pivot historique, et la convention textuelle de miroir est actuellement contradictoire.

3. **`last_drawn` est-il un cache ?**  
   **C'est un cache selon la cible, pas selon le comportement actuel.** Aujourd'hui il est une représentation opérationnelle mutable et persistée, consultée par des règles d'interaction.

4. **Existe-t-il plusieurs autorités géométriques ?**  
   **Oui.** Le Core est autoritaire localement pour la forme, la pose et la topologie ; l'UI est autoritaire de fait pour de nombreuses positions monde ; le XML persiste les deux sans arbitrage géométrique.

5. **Quel premier invariant automatiser ?**  
   **GEO-INV-003.** C'est le plus petit comparateur qui décide objectivement si la projection Core peut devenir la source du cache, sans décider encore des groupes, XML cible ou reconstruction topologique.

6. **Quel premier candidat de migration contrôlée ?**  
   **La reconstruction d'affichage d'un triangle isolé déjà posé à partir de sa pose Core**, sous observation, sans interaction de collage ni mutation de topologie. C'est un flux Core → cache minimal et réversible au niveau de l'analyse.

7. **Que ne faut-il surtout pas migrer en premier ?**  
   **Le collage/attachement.** Il traverse les points UI, l'assistance géométrique, l'inférence de pose, les transactions d'attachements et la fusion de groupes UI.

8. **Faut-il garder la définition exacte de MIG-GEO-001 ?**  
   **Oui, en la précisant mais sans la transformer.** « Formalisation et stabilisation du contrat géométrie locale / pose / coordonnées monde » est juste. Elle doit explicitement inclure le cache UI et la mesure de l'équivalence, et exclure toute inversion immédiate d'autorité.

## 20. Conclusion de l'audit

MIG-GEO-001 peut maintenant s'appuyer sur un constat contradictoire clair : le Core possède les primitives d'une projection fiable, mais l'application ne les utilise pas encore comme chemin de reconstruction de l'UI. Le passage à la cible topologique ne doit donc pas commencer par supprimer le cache ou par migrer le collage. Il doit commencer par rendre observable l'écart entre les deux représentations, en commençant par GEO-INV-003 et en isolant le miroir comme convention critique.

Cette conclusion reste une analyse du code existant. Elle ne prescrit aucune correction ni aucune implémentation.
