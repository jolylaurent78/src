# MIG-CACHE-001 — Audit des champs de last_drawn

## Objet et méthode

Audit documentaire sans modification de code. Il complète le document
MIG_CACHE_001_AUDIT_RECONSTRUCTION_LAST_DRAWN.md en prenant chaque champ
d'une entrée projetée comme unité d'analyse. Les recherches couvrent src/, les
tests, le simulateur, XML, exports et mixins. Les lignes sont indicatives de
l'état audité.

Dans le scénario actif, _last_drawn est l'alias de
CanvasObjectsCollection.entries (src/assembleur_tk.py:575-587). La collection
maintient des index sur id et topoElementId
(src/canvas_objects_collection.py:78-195). Les scénarios non actifs portent
encore une liste historique.

## Inventaire exhaustif du contrat observé

Les entrées normales observées ont les clés suivantes :

    id: int
    labels: tuple[str, str, str]
    pts: {O: point2d, B: point2d, L: point2d}
    mirrored: bool
    topoElementId: str
    topoGroupId: str

orient est une clé historique optionnelle de pose manuelle. _pick_poly et
_pick_pts sont des caches écran ajoutés aux mêmes dictionnaires. Il n'existe
pas de champ normal nommé vertex_local_xy, local_points, local_vertices,
local_coords, rotation, angle ou world_points. world_pts appartient à
self._drag, pas au contrat persistant d'une entrée ; la synchronisation de
pose l'accepte seulement comme repli défensif.

| Champ | Type | Rôle | Catégorie | Équivalent Core | Décision provisoire |
|---|---|---|---|---|---|
| id | int | identifiant catalogue/UI | UI pur | aucun équivalent 1:1 (element_id est distinct) | conserver |
| labels | tuple/list O/B/L | affichage et choix d'arête | UI pur, duplication probable | TopologyElement.vertex_labels | migrable sous conditions |
| pts | points monde O/B/L | géométrie, rendu, interaction | donnée Core dupliquée + cache | local + pose Core | blocage principal |
| mirrored | bool | flip et rendu | donnée Core dupliquée | TopologyPose2D.mirrored | migrable sous conditions |
| topoElementId | str | jointure projection/Core | référence Core | TopologyElement.element_id | à conserver |
| topoGroupId | str | projection du groupe Core | duplication probable | get_group_of_element() | supprimable après lecteurs |
| orient | str | reliquat catalogue | duplication/legacy | orientation locale mesurable | candidat rapide |
| _pick_poly | liste de points écran | hit-test | cache graphique | pts + viewport | régénérable |
| _pick_pts | O/B/L écran | hit-test sommet/arête | cache graphique | pts + viewport | régénérable |
| world_pts | O/B/L monde | état de drag | UI temporaire | pose/pts | hors contrat persistant |

## Lecteurs et écrivains, champ par champ

### id

**Écrit par** : pose manuelle (assembleur_tk.py:5634, 7070-7147),
PlacedTriangle.toLegacyDict() et branches automatiques
(assembleur_sim.py:47-67, 633-1185), chargeur XML
(assembleur_io.py:774-794). Les clones auto (4365-4445) le transportent.

**Lu par** : index de CanvasObjectsCollection (110-115, 177-195),
listbox/rendu/statut (assembleur_tk.py:5580, 5654, 5936-6020, 9281),
dictionnaire (assembleur_tk_mixin_dictionary.py:726-767), sauvegarde XML
(assembleur_io.py:492-505), debug/PDF et
PlacedTriangles.findByTriangleId() (assembleur_sim.py:96-109).

**Conclusion.** Ce n'est pas une donnée géométrique ni un doublon de groupe.
Le catalogue, les listes et le dictionnaire l'utilisent : il ne doit pas être
supprimé par MIG-CACHE-001.

### labels

**Écrit par** : pose manuelle depuis la ligne catalogue
(assembleur_tk.py:5634, 7082, 7122-7133), PlacedTriangle et le simulateur
(assembleur_sim.py:47-67, 390-426, 789-1185), enrichissement après XML
(assembleur_io.py:800-815).

**Lu par** : rendu (assembleur_tk.py:5936-6020, 6167-6245, 10176-10190),
contexte dictionnaire (7369-7371 et mixin), et EdgeChoice
(assembleur_edgechoice.py:243-258, 450-457) qui compare les labels aux
extrémités d'arêtes ; aussi export PDF/debug.

**Équivalent Core.** TopologyElement.vertex_labels, sérialisé dans les
snapshots (assembleur_core.py:426-467, 4043-4121).

**Suppression.** Possible seulement après adaptation d'EdgeChoice, du rendu,
du PDF et du fallback XML à une lecture par topoElementId ou catalogue.

### pts

**Type.** Dictionnaire de trois points monde 2D (principalement ndarray, mais
tuples/listes sont tolérés en XML/tests).

**Écrit ou modifié par** :

| Fichier:ligne | Fonction/flux | Écriture |
|---|---|---|
| assembleur_tk.py:5634, 7070-7147 | _place_dragged_triangle() | crée les O/B/L monde |
| assembleur_tk.py:7265-7291 | rollback | restaure les copies de points |
| assembleur_tk.py:8818-8898 | rotation | tourne directement O/B/L |
| assembleur_tk.py:9009-9058 | _ctx_flip_selected() | réfléchit les points |
| assembleur_tk.py:9064-9095 | _move_group_world() | translate les points |
| assembleur_tk.py:9469-9487, 9507-9696 | drag/snap/commit | modifie les points durant l'interaction |
| assembleur_tk.py:4365-4445 | auto local/monde | applique auto_geom_state et remplace les points monde |
| assembleur_sim.py:47-67, 136-162, 633-1185 | simulation | projette PlacedTriangle.points |
| assembleur_io.py:774-794 | XML | reconstruit O/B/L depuis le fichier |

**Lu par** :

| Fichier:ligne | Fonction/flux | Utilisation |
|---|---|---|
| assembleur_tk.py:747-821 | _sync_group_elements_pose_to_core() | ajuste la pose Core (Kabsch) depuis O/B/L |
| assembleur_tk.py:4982-5010, 7515-7555 | pick-cache, sélection | projection écran et hit-test |
| assembleur_tk.py:5936-6020 | _redraw_from() | dessin |
| assembleur_tk.py:6327-6436, 6694-6703, 6745-7058 | collage/snap/frontière | distances, ancres, candidats |
| assembleur_tk.py:7179-7730, 8267-8355, 8718-9696, 10176-10202 | bbox, dégroupage, guides, transformations, export | géométrie UI |
| assembleur_edgechoice.py:246-288, 399-517 | EdgeChoice automatique | géométrie des triangles candidats |
| assembleur_tk_mixin_bg.py:427, 505 | fond | bornes monde |
| assembleur_debug.py:64-92 | diagnostic | polygone/orientation |
| assembleur_io.py:492-505 | XML | persistance O/B/L |
| assembleur_core.py:35 | helper historique | forme géométrique depuis projection |

**Équivalent Core.**

    TopologyElement.vertex_local_xy + ElementPose(R, T, mirrored)
        -> TopologyWorld.elementLocalToWorld(element_id, local_point)
        -> O/B/L monde

Les éléments existent (assembleur_core.py:426-547, 3196-3203, 3288-3302).

**Conclusion.** pts est à la fois cache et autorité de fait : l'UI l'écrit,
puis _sync_group_elements_pose_to_core() recopie cette géométrie vers le Core.
Il est donc le blocage principal de la migration.

### mirrored

**Écrit par** : initialisation manuelle à False (assembleur_tk.py:7119-7120),
toggle de flip (9009-9058), PlacedTriangle/simulateur
(assembleur_sim.py:47-67, 501-525) et XML (assembleur_io.py:774-780).

**Lu par** : synchronisation de pose (assembleur_tk.py:747-821), rendu
(5936-6020), flip, sauvegarde XML (assembleur_io.py:492-505) et diagnostic
F11 GEO-ORIENT (assembleur_tk.py:921-1006).

**Équivalent Core.** TopologyPose2D.mirrored, accessible via getElementPose()
et écrit par setElementPose().

**Conclusion.** Donnée Core dupliquée. Elle doit devenir une valeur de cache
dérivée de la pose Core ; elle ne peut pas disparaître avant que flip et rendu
soient Core-first.

### topoElementId

**Écrit par** : création manuelle (assembleur_tk.py:7142-7147), simulateur
(assembleur_sim.py:47-67, 410-434, 633-1185) et XML
(assembleur_io.py:783-794, avec fallback temporaire id -> Txx). La projection
auto vérifie son existence au Core (136-162).

**Lu par** : indexes de collection
(canvas_objects_collection.py:98-108, 185-195), résolution de
groupes/collage/opérations Tk (assembleur_tk.py:687-821, 6811-6923,
7551-7555, 8718-9158), horloge/guides, XML, simulateur et debug.

**Conclusion.** C'est la référence Core minimale d'une projection
reconstructible. Elle doit rester dans last_drawn même lorsqu'il devient un
cache pur.

### topoGroupId

**Écrit par** : projection automatique par world.get_group_of_element()
(assembleur_sim.py:136-162), création manuelle (assembleur_tk.py:7146-7147)
et XML (assembleur_io.py:783-794).

**Lu par** : F11/GEO-ORIENT (assembleur_tk.py:921-1006), traces d'horloge
(assembleur_tk_mixin_clockarc.py:49-98) et tests historiques. Les autres
topoGroupId lus par guides/clock (assembleur_tk.py:7805-8397,
assembleur_io.py:458-604) appartiennent à leurs propres dictionnaires, non à
une entrée de last_drawn.

**Équivalent Core.** TopologyWorld.get_group_of_element(topoElementId).

**Conclusion.** Duplication probable : supprimable des entrées une fois F11,
les traces qui liraient une entrée, et les tests convertis à la dérivation
Core.

### orient

**Écrit par** _place_dragged_triangle() (assembleur_tk.py:7119). Le seul
lecteur d'entrée actuel est le diagnostic F11 GEO-ORIENT (995), qui tolère son
absence. Les paramètres orient du simulateur ne constituent pas un lecteur
persistant de last_drawn["orient"].

**Équivalent.** Pas de champ homonyme Core : l'orientation mesurable relève de
vertex_local_xy ; la réflexion relève de mirrored.

**Conclusion.** Reliquat sans consommateur fonctionnel identifié ; supprimable
après adaptation du seul diagnostic.

### _pick_poly, _pick_pts, world_pts et données locales

_rebuild_pick_cache() (assembleur_tk.py:4982-5010) écrit _pick_poly et
_pick_pts depuis pts, zoom et offset. La sélection lit _pick_pts (7515-7555)
et le hit-test consomme le cache voisin. Ce sont des caches graphiques
régénérables : ils doivent rester hors Core et ne doivent pas être persistés.

world_pts est seulement écrit dans self._drag (6625, 7072-7075, 9577-9588).
_sync_group_elements_pose_to_core() l'accepte comme repli (786), mais il n'est
pas un champ persistant de projection.

| Notion demandée | Emplacement réel |
|---|---|
| vertex_local_xy | TopologyElement (assembleur_core.py:426-547) |
| local_points | variable locale du dump F11 (assembleur_tk.py:930-963) |
| local_vertices, local_coords | aucun champ last_drawn |
| rotation/translation | TopologyPose2D Core |
| angle | outil UI d'arc, non lié à une entrée |
| world_points | aucun champ ; la convention présente est pts |

## Analyse de _sync_group_elements_pose_to_core()

La fonction (assembleur_tk.py:747-821) résout les membres par
getGroupElementIds(), retrouve les projections par topoElementId, lit
pts/mirrored, lit les locaux Core, ajuste (R,T) par Kabsch, puis appelle
world.setElementPose(). Elle est appelée par l'auto-sync (829-842), la pose
manuelle, move, rotate, flip, collage et dégroupage.

La résolution des membres est déjà Core-first. La fuite d'autorité est
strictement géométrique : la projection est encore source de la pose.

## Proposition de suppression et prérequis

| Champ | Peut être supprimé ? | Pourquoi | Prérequis |
|---|---|---|---|
| id | Non dans ce chantier | catalogue/UI, sans équivalent 1:1 | migration distincte de la couche catalogue |
| labels | Oui, sous conditions | copie de vertex_labels | rendre EdgeChoice/rendu/PDF Core-aware |
| pts | Oui à terme, pas maintenant | déductible depuis local + pose | Core-first pour drag/move/rotate/flip/auto/XML, puis rebuild |
| mirrored | Oui, sous conditions | copie de pose Core | flip Core-first, rendu dérivé |
| topoElementId | Non | jointure indispensable projection/Core | aucun |
| topoGroupId | Oui | dérivable immédiatement | migrer F11/traces/tests concernés |
| orient | Oui | diagnostic seul | adapter le dump F11 |
| _pick_* | Non comme caches, oui comme persistance | caches UI nécessaires mais régénérables | rebuild de pick après projection |

## Points bloquants

1. **Double autorité de pts** : l'UI écrit les points, puis les synchronise dans le Core. Un rebuild immédiat les écraserait tant que ce flux subsiste.
2. **Projection auto incomplète** : buildLegacyLastDrawnFromTopology() ne recalcule pas les points depuis le Core ; il ne dérive actuellement que topoGroupId.
3. **last_drawn_local + auto_geom_state** : les scénarios automatiques portent encore une géométrie locale et une transformation globale hors Core (assembleur_tk.py:4365-4445).
4. **XML v4** : le chargeur restaure le snapshot Core, mais adopte ensuite les points XML ; la SPEC devra fixer leur rôle de compatibilité.
5. **Lecteurs nombreux de pts** : rendu, hit-test, snap, EdgeChoice, bbox, fond, PDF et debug. Une première projection reconstruite peut conserver leur contrat de lecture avant de les migrer un par un.

## Conclusion

L'infrastructure de collection est prête : CanvasObjectsCollection peut
contenir un cache reconstruisible et topoElementId est la jointure canonique.
Ce qui empêche encore last_drawn d'être un cache pur est pts (et,
secondairement, mirrored) : ces champs sont directement modifiés par l'UI,
puis propagés vers le Core.

La migration fonctionnelle devra donc inverser le flux :

    commande UI -> pose/flip Core -> projection O/B/L + mirrored -> last_drawn
                     -> caches écran _pick_* régénérés

Après cette inversion, pts et mirrored seront des valeurs de cache,
topoGroupId et orient pourront être retirés, et topoElementId restera la
référence minimale de chaque objet affiché.

