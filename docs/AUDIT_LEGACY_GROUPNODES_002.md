# AUDIT-LEGACY-GROUPNODES-002 — état post MIG-GEO-010

Date : 16 juillet 2026  
Nature : audit statique ciblé, sans modification applicative.

## Résumé exécutif

L'audit couvre **26 usages fonctionnels groupés** du modèle UI historique dans
le code de production et les tests. Il distingue les occurrences syntaxiques
des flux réellement différents : une boucle de chargement et sa validation,
par exemple, constituent un même contrat de persistance, tandis que le collage
manuel est isolé car il porte sa propre sémantique.

MIG-GEO-010 a bien migré les lecteurs passifs annoncés : bbox monde et écran,
centroïde, contours, forme de collision et exclusion lors du snap. Aucun autre
lecteur passif de `groups[*]["nodes"]` n'a été trouvé dans ces domaines.

Le reliquat n'est toutefois pas limité au collage et au XML. Cinq usages
emploient encore la chaîne ou le `group_id` UI comme solution de repli ou comme
entrée d'un traitement actif : synchronisation de poses vers le Core, rotation
CTRL d'un groupe, fallback MOVE/ROTATE/FLIP, recherche du groupe UI depuis un
triangle et translation de la projection après dégrouper. Ils ne justifient pas
un MIG-GEO-011 consacré aux lecteurs passifs, mais ils justifient un chantier
réduit et explicitement fonctionnel.

**Verdict : Verdict 2 — MIG-GEO-011 réduit.** Il ne reste pas de migration
passive large ; un micro-chantier doit traiter les lecteurs de membres encore
utilisés comme autorité dans les flux de synchronisation et d'interaction, sans
toucher au collage ni au XML.

## Méthode et périmètre

L'audit a examiné `src/`, `tests/` et la documentation d'architecture utile.
Les recherches sur `self.groups`, `groups.get`, `groups[*]["nodes"]`,
`group.get("nodes")`, `group["nodes"]` et `group_id` ont été suivies jusqu'aux
appelants immédiats. Chaque résultat est classé une seule fois suivant son rôle
principal. Les occurrences purement Core et les fixtures de tests ne sont pas
comptées comme des dépendances legacy exécutables.

Répartition des 26 usages fonctionnels analysés :

| Catégorie | Nombre | Sens |
| --- | ---: | --- |
| A — lecteur passif migratable | 2 | aucun restant dans le rendu courant ; deux points de sélection/résolution isolés |
| B — projection UI / cache | 4 | création, clonage et rattachement de la projection |
| C — collage manuel / fusion | 1 | concaténation des chaînes au relâchement souris |
| D — ordre historique de chaîne | 3 | simulation et convention de parcours/insertion |
| E — XML / compatibilité | 2 | sérialisation et relecture du contrat v4 |
| F — mutation legacy résiduelle | 9 | maintien de la projection et résultats d'interactions |
| G — suspect / oubli à migrer | 5 | membres déterminés depuis legacy dans des flux actifs |

## Distinction impérative des `group_id`

### `group_id` Core — moderne, autoritaire

Dans `assembleur_core.py`, `TopologyGroup.group_id`, `TopologyWorld.groups`,
`find_group()`, `group_members()`, `get_group_of_element()`, les caches de
frontière et les API d'attachement utilisent l'identifiant DSU canonique. Ces
usages sont valides : ils portent les `element_ids`, les attachments et la
connectivité. Ils ne relèvent pas de cet audit legacy.

### `group_id` UI — historique, non autoritaire

Dans `assembleur_tk.py`, `last_drawn[*]["group_id"]`, `self.groups[gid]` et
`_next_group_id` forment une identité de projection locale. Elle est encore :

* produite par le placement manuel, les scénarios automatiques et le XML ;
* lue par des helpers de compatibilité, le collage et certains fallbacks ;
* comparée par le validateur MIG-GEO-001 au groupe Core.

Elle ne doit pas décider de l'appartenance métier. Les méthodes MIG-GEO-010
résolvent désormais cette appartenance avec le `topoElementId` de l'entrée
projetée, puis `TopologyWorld.get_group_of_element()`.

## Inventaire exhaustif des usages legacy fonctionnels

| # | Fichier, fonction/famille | Catégorie | Producteur / consommateur | Ordre de chaîne ? | Remplaçable par Core + `_last_drawn` ? |
| ---: | --- | --- | --- | --- | --- |
| 1 | `assembleur_tk.py`, `_get_active_topology_group_for_ui_gid()` | B | lit seulement `topoGroupId` de la projection pour joindre le groupe Core | non | nécessaire comme pont transitoire ; oui à terme avec une clé Core dans l'état UI |
| 2 | `TriangleViewerManual.__init__` et rattachement `manual.groups` | B | initialise/partage le cache UI historique | non | oui, mais pas nécessaire à MIG-011 |
| 3 | `_convertActiveAutoToManualSnapshot()` | B | clone `groups` et la séquence `nodes` lors d'un snapshot manuel | oui, conservation | partiellement ; dépend du contrat de snapshot historique |
| 4 | `_set_active_scenario()` | F | réconcilie `groups -> last_drawn.group_id`, bbox et `topoGroupId` | non pour les membres, oui pour le cache | oui pour les membres ; mutation de compatibilité à isoler |
| 5 | `_draw_group_outlines()` | B | énumère les clés UI pour dessiner des frontières déjà Core | non | déjà migré pour les membres et le contour |
| 6 | `_ctx_get_ui_group_from_triangle_index()` | A | retrouve un groupe UI en parcourant `nodes` avant contexte clic droit | non | oui : `topoElementId -> groupe Core`, puis projection éventuelle |
| 7 | `_ctx_capture_chemin_context()` | A | consomme le résultat précédent pour obtenir le groupe Core d'un chemin | non | oui ; le détour UI est évitable |
| 8 | `_on_ctrl_down()`, mode `move_group` | G | rotation d'alignement applique la pose à tous les `nodes` | non | oui : membres Core déjà mémorisables dans `_sel` ; risque de divergence |
| 9 | `_group_nodes()` | B | façade de lecture du contrat de chaîne historique | oui selon l'appelant | non à supprimer tant que C/D/E existent |
| 10 | placement manuel d'un singleton (`_place...`) | F | crée `self.groups[gid]`, un node singleton et écrit `group_id` | non | oui pour l'appartenance ; projection UI reste nécessaire transitoirement |
| 11 | `_apply_group_meta_after_split_()` | F | recopie `group_id` et bbox après une mutation UI | non | oui pour bbox ; le nom et le rôle doivent disparaître avec les mutations UI |
| 12 | `_degrouperGroupScreenBBox()` | B | bbox écran de la projection issue de dégrouper | non | déjà migré : Core + éléments projetés |
| 13 | `_degrouperTranslateGroupByScreen()` | F | translate les triangles listés dans `nodes` pendant le placement UI post-dégrouper | non | oui, avec les éléments de la composante Core ; à ne pas confondre avec une rupture topologique |
| 14 | `_applyDegrouperResultToTk()` | F | reconstruit les groupes UI depuis les composantes renvoyées par `degrouperAtNode()` | ordre partiellement conservé | oui pour membres ; contrat d'ordre de projection à fixer |
| 15 | `_ctx_delete_group()` | F | supprime/remappe groupes, nodes et `group_id` après suppression d'éléments Core | non | partiellement ; Core est modifié, cache UI est ensuite réparé |
| 16 | `_sync_group_elements_pose_to_core()` | G | choisit les éléments dont la pose UI est persistée vers Core en lisant `nodes` | non | oui, impérativement ; source de divergence si UI/Core diffèrent |
| 17 | `_prepare_mig_geo_operation_members()` | G | construit `LegacyMembers` pour diagnostic et fallback MOVE/ROTATE/FLIP | non | oui ; le chemin primaire est déjà Core |
| 18 | `_move_group_world()` fallback | G | déplace les `nodes` si les membres Core ne sont pas fournis | non | oui ; fallback legacy à caractériser puis supprimer séparément |
| 19 | `_on_canvas_left_up()` | C | fusionne groupes, calcule ancres, insère/concatène `nodes`, écrit edges et ids UI | oui : insertion avant/après ancre | partiellement ; attachment Core oui, ordre de chaîne non défini dans le Core |
| 20 | `assembleur_sim.py`, constructeurs de scénarios automatiques | D | créent une chaîne `nodes` et initialisent `group_id` | oui : ordre de génération | partiellement ; connectivité Core oui, narration/séquence historique non |
| 21 | `AlgoQuadrisParPaires._fill_group_vkeys_from_geometry()` | D | enrichit les voisins consécutifs avec `edge_in/out` | oui : voisins de liste | non sans définir un parcours Core explicite |
| 22 | `assembleur_sim.py`, formes de contrôle issues de chaîne | D | emploie la chaîne pour initialiser/contrôler les scénarios simulés | oui | oui pour les ensembles ; ordre requis par l'algorithme historique |
| 23 | `assembleur_io.py`, `saveScenarioXml()` | E | écrit `<groups>/<group>/<node>` et `triangle@group` | oui : ordre XML | partiellement ; Core peut fournir membres, pas le round-trip de chaîne actuel |
| 24 | `assembleur_io.py`, `loadScenarioXml()` | E | lit nodes/edges, restaure `group_id`, puis vérifie le lien Core | oui : ordre XML | indispensable tant que les fichiers v4 sont supportés |
| 25 | validateur MIG-GEO-001 et export F8 | F | compare `group_id` UI aux membres Core, sans muter | non | supprimable lorsque la projection legacy ne sera plus produite |
| 26 | tests `test_mig_geo001_group_linking.py` | F | fixtures et contrôles d'équivalence Core/UI | non | à adapter après suppression de la projection legacy |

Les occurrences dans `assembleur_core.py` de `self.groups` sont explicitement
exclues de ce tableau : elles appartiennent au `TopologyWorld`, pas à
`TriangleViewerManual.groups`.

## Mutations legacy restantes et risque de divergence

| Déclencheur | Mutation UI legacy | Moment de mutation Core | Risque |
| --- | --- | --- | --- |
| placement manuel | crée singleton, node et `group_id` | Core crée le groupe/élément puis projection | faible si création atomique ; identifiants distincts |
| activation scénario | remplit `last_drawn.group_id`, bbox et `topoGroupId` depuis `scen.groups` | aucun | moyen : une chaîne XML incomplète peut dégrader le cache UI |
| dégrouper | reconstruit/sépare les nodes après résultat Core | Core d'abord via `degrouperAtNode()` | moyen : ordre UI non défini par le résultat Core |
| translation post-dégrouper / CTRL | transforme les triangles énumérés dans `nodes` | synchronisation ultérieure ou absente selon le flux | fort : ensemble UI possiblement différent du groupe Core |
| MOVE/ROTATE/FLIP fallback | lit `nodes` si résolution Core absente | Core normalement déjà utilisé | moyen : masque une rupture du lien Core/UI |
| collage manuel | concatène chaînes et écrit `edge_in/out` / `group_id` | attachment Core créé pendant le workflow | fort : ordre et liaison UI peuvent diverger du Core |
| XML load/save | restaure ou écrit ordre, ids et edges | snapshot Core chargé et contrôlé après lecture | fort pour compatibilité et round-trip |

## Matrice de validation MIG-GEO-010

| Domaine | Migré vers Core + projection | Legacy encore lu | Commentaire |
| --- | --- | --- | --- |
| bbox | oui | non pour `_recompute_group_bbox`; oui dans la maintenance XML/dégrouper | bbox monde et écran passifs migrés |
| centroïde | oui | non | `_group_centroid()` résout les éléments Core |
| contour | oui | non | frontière Core et projection indexée |
| collision | oui | non dans l'assistance courante | `_update_edge_highlights()` utilise les éléments projetés |
| exclusion snap | oui | non | comparaison des groupes Core des éléments |
| sélection groupe | partiellement | oui | `_ctx_get_ui_group_from_triangle_index()` parcourt encore nodes |
| sélection sommet | oui | non | MIG-GEO-008 : nœud DSU -> groupe Core -> membres |
| déplacement | oui, chemin primaire | oui | fallback `_move_group_world()` et synchronisation de pose |
| rotation | oui, menu principal | oui | `_on_ctrl_down()` conserve une rotation d'alignement par nodes |
| flip | oui, chemin primaire | oui | préparation conserve `LegacyMembers` de diagnostic/fallback |
| surbrillance | oui | non | contours et aides utilisent les éléments projetés/Core |
| autres lecteurs passifs | oui | non trouvé | le reste est mutation, ordre, collage ou XML |

## Évaluation de MIG-GEO-011

### Ce qui ne doit pas entrer dans MIG-GEO-011

* `_on_canvas_left_up()` et toute concaténation/insertion de chaîne ;
* `edge_in`, `edge_out`, `vkey_in`, `vkey_out` ;
* `saveScenarioXml()` et `loadScenarioXml()` ;
* changement de format ou de compatibilité XML ;
* définition de l'ordre de parcours des scénarios automatiques.

### Périmètre réduit recommandé

1. Remplacer `_ctx_get_ui_group_from_triangle_index()` et son appelant par une
   résolution directe du `topoElementId` vers le groupe Core ; conserver une
   clé UI seulement si le menu doit adresser une projection historique.
2. Faire en sorte que `_sync_group_elements_pose_to_core()` reçoive les
   éléments projetés du groupe Core, non les `nodes` UI.
3. Remplacer la boucle de rotation CTRL du mode `move_group` par les membres
   Core déjà résolus au début du geste ; vérifier la synchronisation de poses.
4. Isoler les fallbacks de `_prepare_mig_geo_operation_members()` et
   `_move_group_world()` derrière une caractérisation explicite, sans les
   supprimer mécaniquement dans le même chantier.

Tests nécessaires : un test de groupe Core dont la chaîne UI est incomplète
pour le contexte clic droit ; un test de rotation CTRL ; un test de
synchronisation de poses ; un test démontrant que le fallback legacy ne masque
pas une projection Core manquante. Les exclusions ci-dessus restent absolues.

## Conclusion et prochain chantier

MIG-GEO-010 a absorbé la migration des lecteurs passifs prévue. MIG-GEO-011
ne doit donc pas être une seconde migration de contours/bbox/snap. Il doit être
un **micro-chantier de suppression des dernières résolutions de membres par la
chaîne UI dans les flux actifs de synchronisation et d'interaction**.

Les trois risques principaux sont :

1. une pose Core partiellement synchronisée si `_sync_group_elements_pose_to_core()`
   suit une chaîne UI désynchronisée ;
2. une rotation CTRL appliquée à un ensemble différent de celui sélectionné par
   le groupe Core ;
3. une régression de compatibilité XML ou de collage si l'ordre historique est
   assimilé à tort à une simple liste de membres.

Le chantier suivant recommandé est donc :

```text
MIG-GEO-011 — Résolution Core des membres dans les flux actifs UI
```

Il doit exclure explicitement collage manuel et XML historique.
