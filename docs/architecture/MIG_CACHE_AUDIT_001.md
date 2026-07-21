# MIG-CACHE-AUDIT-001 — Audit de `last_drawn`

## 1. Objet, périmètre et conclusion

Cet audit décrit l'état **effectif** des entrées de `last_drawn` après les
migrations Core-first déjà réalisées. Il porte sur les dictionnaires de la
collection active `CanvasObjectsCollection.entries`, encore exposés par
compatibilité sous l'alias `TriangleViewerManual._last_drawn`, ainsi que sur
les listes `ScenarioAssemblage.last_drawn` des scénarios inactifs.

Architecture constatée :

```text
TopologyWorld ──projection──> last_drawn / CanvasObjectsCollection ──> Canvas
```

La géométrie autoritaire est désormais dans les poses des `TopologyElement`.
Les chemins MOVE, ROTATE, FLIP, SPLIT et création manuelle projettent le Core
après leur commit. Le cache reste toutefois lu par le rendu, le hit-test, le
XML et les aperçus temporaires.

**Conclusion principale.** Une entrée normale possède aujourd'hui six clés
métier/visuelles : `id`, `labels`, `pts`, `mirrored`, `topoElementId` et
`topoGroupId`. Deux clés techniques peuvent être ajoutées à la demande :
`_pick_poly` et `_pick_pts`. `orient` est une clé tolérée par le mode debug et
par certains scénarios/tests historiques, mais elle n'est plus produite par la
projection automatique normale. Les champs `group_id`, `edge_in`, `edge_out`,
`vkey_in`, `vkey_out`, `group_pos`, `bbox`, `rotation`, `local_*` et
`world_pts` ne sont **pas** des champs d'entrée `last_drawn` produits par le
runtime actuel ; ils appartiennent soit aux anciens groupes, soit au catalogue,
soit à l'état de drag.

Le cache ne peut donc pas encore être réduit à `pts + topoElementId` sans
migrer les lecteurs XML, le rendu des libellés et la compatibilité miroir.
Il n'existe en revanche plus de raison architecturale de conserver
`topoGroupId` ou `mirrored` comme copies persistantes dans l'entrée.

## 2. Méthode et structures examinées

Les recherches ont couvert tous les fichiers Python de `src/` et `tests/`,
avec examen des producteurs, lecteurs directs (`get`, indexation), projections
et chaînes d'appel. Les numéros ci-dessous sont des références du checkout
audité ; ils servent de points d'entrée, non de contrat d'API.

| Structure | Localisation | Rôle |
|---|---|---|
| `CanvasObjectsCollection.entries` | `src/canvas_objects_collection.py:77-286` | Liste réelle de la projection active ; maintient les index `topoElementId` et `id`. |
| `self._last_drawn` | `src/assembleur_tk.py:580, 681` | Alias de compatibilité vers `canvas_objects.entries`. |
| `ScenarioAssemblage.last_drawn` | `src/assembleur_core.py:69-91` | Projection stockée avec un scénario, ensuite liée à la collection lors de son activation. |
| `PlacedTriangle.toLegacyDict()` | `src/assembleur_sim.py:38-68` | Producteur de la forme dictionnaire lors de la finalisation automatique. |
| `buildLegacyLastDrawnFromTopology()` | `src/assembleur_sim.py:136-161` | Projection auto : ajoute `topoGroupId` depuis `TopologyWorld`. Le nom est historique ; sa sortie est la projection actuelle. |
| Chargeur XML | `src/assembleur_io.py:767-815` | Produit des entrées réhydratées depuis XML et le snapshot Core. |
| Création manuelle | `src/assembleur_tk.py:7196-7293` | Crée le squelette de l'entrée, puis appelle la projection Core. |
| Projection Core -> cache | `src/assembleur_tk.py:837-914` | `_project_core_element_to_last_drawn`, `_project_core_group_to_last_drawn`, `_project_core_group_to_collection`. |

### 2.1 Cycle de vie structurel

- La simulation construit des `PlacedTriangle`, puis les projette vers les
  dictionnaires de scénario (`assembleur_sim.py:136-161, 814, 1322`).
- Le changement de scénario lie la liste du scénario actif à
  `canvas_objects.entries` (`assembleur_tk.py:4633` et écritures associées).
- Les mutations de structure passent par `CanvasObjectsCollection` ; ses
  index reposent seulement sur `id` et `topoElementId`.
- Les transformations Core-first remplacent les `pts` du cache par une
  projection (`assembleur_tk.py:837-914`, et appels MOVE/ROTATE/FLIP vers
  `9200-10200`). Elles ne rendent pas `pts` autoritaire.
- Le chargeur XML importe d'abord le snapshot Core, puis valide chaque
  `topoElementId` avant de créer la projection (`assembleur_io.py:783-795`).

## 3. Inventaire exhaustif des attributs d'entrée

### 3.1 Tableau de synthèse

| Champ | Type / forme | Classe | État d'accès | Equivalent Core | Décision d'architecture |
|---|---|---|---|---|---|
| `pts` | `dict[O|B|L, ndarray/coordonnée]` | A — Cache UI | Lecture/écriture | `TopologyElement.local_frame` + `pose` | Conserver comme projection monde, non autoritaire. |
| `topoElementId` | `str` | B — Référence Core | Lecture/écriture à la création | `TopologyElement.element_id` | Conserver : lien minimal cache -> Core. |
| `id` | `int` (ID catalogue) | A — Cache UI | Lecture/écriture | `TopologyElement.meta["triId"]`/catalogue | Conserver provisoirement pour UI, liste, XML et index ; son origine doit rester catalogue, pas un second ID Core. |
| `labels` | tuple/liste `(O,B,L)` | C — Donnée dupliquée du Core | Lecture/écriture | `TopologyElement.vertex_labels` | Conserver comme cache de rendu à court terme ; migrer le rendu/chargement vers le Core avant suppression. |
| `mirrored` | `bool` | C — Donnée dupliquée du Core | Lecture/écriture | `TopologyElement.pose.mirrored` | Candidat à suppression après XML et toute lecture résiduelle ; le Canvas est déjà Core-first pour le suffixe `S`. |
| `topoGroupId` | `str` | C — Donnée dupliquée du Core | Lecture/écriture | `world.get_group_of_element(topoElementId)` | Candidat fort à suppression : ne doit jamais être une autorité. |
| `orient` | `str` (`CW`/`CCW`) facultatif | C — Donnée dupliquée du Core | Lecture rare / écriture historique | `TopologyElement.meta["orient"]`, coordonnées locales | Ne pas réintroduire en projection normale ; conserver seulement tant que le diagnostic F11 le tolère. |
| `_pick_pts` | dict écran `O/B/L` | A — Cache UI | Lecture/écriture | aucune donnée Core à conserver | Conserver cache UI, invalider/reconstruire après changement de vue/projection. |
| `_pick_poly` | liste de points écran | A — Cache UI | Lecture/écriture | aucune | Conserver cache UI, dérivé de `pts` et de la vue. |
| `group_id` | absent des entrées courantes | D — Legacy | lecture/écriture historique/tests | groupe Core | Supprimé de la production ; interdire sa réapparition. |

### 3.2 `pts`

**Description.** Dictionnaire des trois points monde affichés : `O`, `B`, `L`.
Ce sont les coordonnées employées directement par le Canvas et les sorties
visuelles.

**Création et modifications.**

- `PlacedTriangle.toLegacyDict()` le crée depuis `PlacedTriangle.points`
  (`assembleur_sim.py:58-68`), puis la finalisation auto l'associe au groupe
  Core (`136-161`).
- `loadScenarioXml()` le lit depuis les noeuds XML `O/B/L` et crée l'entrée
  (`assembleur_io.py:769-795`).
- `_place_dragged_triangle()` crée d'abord une entrée sans `pts`, puis
  `_project_core_element_to_last_drawn()` l'écrit depuis le Core
  (`assembleur_tk.py:7196-7293`, `837-914`).
- Les projections de groupe/élément écrasent `pts` après les commits
  Core-first. Les aperçus manuels de MOVE et ROTATE peuvent temporairement
  modifier cette même clé pendant le drag (`assembleur_tk.py:9398-9510`) ; le
  commit ou l'annulation reprojette ensuite le Core.

**Lecteurs fonctionnels.**

- Rendu Canvas/PDF : `_draw_triangle_screen` et les boucles de redraw
  (`assembleur_tk.py:6306`, `10677+`).
- Hit-test et recherche de sommet : `_rebuild_pick_cache`, `_hit_vertex`,
  opérations Canvas (`5117-5145`, `7649+`).
- Aperçus drag/snap et EdgeChoice : handlers souris et
  `assembleur_edgechoice.py` reçoivent les points projetés.
- XML : `saveScenarioXml()` sérialise O/B/L (`assembleur_io.py:490-505`).
- Horloge/ancre : `_clock_restore_anchor_world` consulte ponctuellement
  `pts` (`assembleur_tk.py:8402-8406`).

**Rôle et proposition.** `pts` est indispensable au cache graphique actuel,
mais sa source est le Core. Il n'est pas supprimable : le Canvas, le PDF et le
format XML actuel ont besoin d'une projection monde. Une future projection
globale doit simplement garantir qu'aucun commit métier ne le modifie
directement.

### 3.3 `topoElementId`

**Description.** Identifiant canonique de l'élément Core, par exemple `T28`.

**Création / modifications.**

- Assigné au `PlacedTriangle` dès la création Core dans la simulation, puis
  exporté par `toLegacyDict()` (`assembleur_sim.py:38-68, 643-652`).
- Créé manuellement à partir de l'ID attribué par
  `world.add_element_as_new_group()` (`assembleur_tk.py:7247-7272`).
- Lu depuis XML ; les XML v4 sans attribut utilisent uniquement le pont
  temporaire `28 -> T28`, qui est ensuite validé dans `world.elements`
  (`assembleur_io.py:783-795`).
- Il n'est pas transformé par MOVE/ROTATE/FLIP ; une suppression/reconstruction
  remplace structurellement l'entrée, non l'identité Core.

**Lecteurs.**

- Index de `CanvasObjectsCollection` (`canvas_objects_collection.py:98-120,
  188-191, 279-281`).
- Résolution sélection -> Core et groupe (`assembleur_tk.py:697-711`,
  `7683-7687`, `8853-8861`, `9200-9205`, `9277-9286`).
- Sauvegarde XML obligatoire (`assembleur_io.py:493-498`).
- Diagnostics GEO-ORIENT / F11 (`assembleur_tk.py:1043-1122`).

**Rôle et proposition.** C'est la seule référence Core strictement nécessaire
dans une entrée de projection. Elle doit rester, avec l'invariant : une entrée
persistante a un ID non vide appartenant à `world.elements`.

### 3.4 `id`

**Description.** Identifiant du triangle dans le catalogue, entier affichable
et utilisé par la liste UI. Il ne faut pas le confondre avec `topoElementId` :
un élément Core est une instance topologique, `id` est l'identité catalogue
historique et la clé de nombreuses données de présentation.

**Création / modifications.**

- `PlacedTriangle.triangleId -> toLegacyDict()["id"]`
  (`assembleur_sim.py:38-68`).
- Création manuelle depuis le catalogue (`assembleur_tk.py:7196-7272`).
- XML (`assembleur_io.py:769-795`).
- Pas de mutation géométrique constatée ; changement seulement par
  remplacement d'entrée ou suppression.

**Lecteurs.**

- Index catalogue de `CanvasObjectsCollection` et `_placed_ids`
  (`canvas_objects_collection.py:114-137, 181-191`,
  `assembleur_io.py:818`).
- Texte de rendu, status, listbox, filtre automatique et suppression de
  contexte (`assembleur_tk.py:9079-9082`, `9707`, plus les boucles de dessin).
- XML (`assembleur_io.py:493-505`).

**Rôle et proposition.** Ce champ est nécessaire aux conventions de catalogue
et à l'UI, même s'il peut être retrouvé via les métadonnées de
`TopologyElement`. Il reste un cache UI jusqu'à la migration explicite des
lecteurs vers cette métadonnée. Il n'est ni un identifiant de groupe ni une
source topologique.

### 3.5 `labels`

**Description.** Libellés visibles de sommets, généralement `(Bourges, B, L)`.

**Création / modifications.**

- Produit dans `PlacedTriangle.toLegacyDict()` depuis la branche de simulation
  (`assembleur_sim.py:58-68`).
- Créé manuellement depuis le triangle catalogue
  (`assembleur_tk.py:7196-7272`).
- Après chargement XML, complété depuis le DataFrame catalogue, ou par des
  libellés vides en mode dégradé (`assembleur_io.py:801-815`).

**Lecteurs.**

- Rendu de triangle : `_draw_triangle_screen(..., labels=...)`
  (`assembleur_tk.py:6306-6365`, redraw `10677+`).
- Simulation et EdgeChoice consultent les libellés des triangles placés, mais
  ceux-ci peuvent venir directement du catalogue/Core avant la projection.

**Rôle et proposition.** Les mêmes labels sont portés par
`TopologyElement.vertex_labels`. Le champ est une copie de confort pour le
rendu ; il est donc C, pas A. Sa suppression exige que le rendu et le chargeur
résolvent les labels à partir de `topoElementId` + Core/catalogue, y compris le
mode XML dégradé.

### 3.6 `mirrored`

**Description.** Indicateur de miroir historique dans l'entrée projetée.

**Création / modifications.**

- Produit par `PlacedTriangle.toLegacyDict()` et lors du chargement XML
  (`assembleur_sim.py:58-68`, `assembleur_io.py:769-781`).
- Création manuelle à `False` avant projection (`assembleur_tk.py:7268-7272`).
- Les opérations FLIP Core-first n'écrivent volontairement plus ce champ ; la
  pose `TopologyElement.pose.mirrored` est l'autorité.

**Lecteurs.**

- `saveScenarioXml()` écrit encore l'attribut XML `mirrored`
  (`assembleur_io.py:490-505`).
- `PlacedTriangle.fromLegacyDict()` accepte la valeur historique.
- Le diagnostic GEO-ORIENT F11 l'affiche (`assembleur_tk.py:1104-1122`).
- Le suffixe Canvas/PDF est désormais lu via le helper Core, non via le cache.

**Rôle et proposition.** C'est un doublon Core. Il n'est plus requis au rendu
normal ; il ne peut pas être retiré immédiatement car le writer XML et la
compatibilité de lecture l'utilisent. La migration doit d'abord faire écrire
le XML depuis `TopologyElement.pose.mirrored` et faire de l'ancien attribut
XML une entrée de compatibilité uniquement.

### 3.7 `topoGroupId`

**Description.** Copie du groupe canonique contenant l'élément projeté.

**Création / modifications.**

- Ajouté à chaque projection auto à partir de
  `topologyWorld.get_group_of_element(element_id)`
  (`assembleur_sim.py:136-161`).
- Ajouté par le chargeur XML à partir du même Core (`assembleur_io.py:783-795`).
- Ajouté lors de la création manuelle (`assembleur_tk.py:7268-7272`).
- Modifié indirectement lorsqu'une projection complète remplace l'entrée après
  fusion/split ; aucune mutation de groupe ne doit en faire l'autorité.

**Lecteurs.**

- Diagnostic GEO-ORIENT seulement dans l'entrée elle-même
  (`assembleur_tk.py:1043-1122`).
- Tests de cohérence de projection (`tests/test_topology_element_ids.py`).
- Les données d'horloge emploient aussi une clé nommée `topoGroupId`, mais ce
  sont des dictionnaires de guides/traits, **pas des entrées `last_drawn`**.

**Rôle et proposition.** C'est une redondance exacte de
`world.get_group_of_element(entry["topoElementId"])`. Les chemins métier
modernes résolvent déjà le groupe depuis l'élément. Il est supprimable après :
(1) migration du writer XML/diagnostic, (2) adaptation des tests pour vérifier
la projection Core directement, et (3) interdiction de nouveaux lecteurs.

### 3.8 `orient`

**Description.** Orientation catalogue `CW`/`CCW`, parfois injectée par des
fixtures ou anciennes entrées. La géométrie locale Core tient aujourd'hui
compte de `TopologyElement.meta["orient"]` (`assembleur_core.py:537-542`).

**Création / modifications.**

- La création manuelle peut fournir l'orientation catalogue dans le squelette
  de projection (`assembleur_tk.py:7196-7272`).
- Les entrées automatiques normales ne l'ajoutent pas dans
  `PlacedTriangle.toLegacyDict()`.
- Le chargeur XML actuel ne la lit ni ne l'écrit.

**Lecteurs.**

- Le dump GEO-ORIENT F11 l'affiche comme diagnostic facultatif
  (`assembleur_tk.py:1104-1122`).
- Tests XML de diagnostic injectent explicitement cette clé.

**Rôle et proposition.** Aucun consommateur fonctionnel Canvas/Core n'a été
trouvé. C'est un doublon Core, toléré uniquement pour le diagnostic. Ne pas le
mettre dans le contrat cible ; le dump peut interroger `element.meta` à la
place. Il est supprimable après adaptation du diagnostic et des fixtures.

### 3.9 `_pick_pts` et `_pick_poly`

**Description.** Coordonnées écran dérivées, respectivement dictionnaire des
sommets et polygone écran.

**Création / modifications.** `_rebuild_pick_cache()` les écrit pour chaque
entrée depuis `pts` et la transformation monde -> écran ; il écrit
respectivement `{}`/`None` lorsque les trois sommets sont absents
(`assembleur_tk.py:5117-5145`).

**Lecteurs.** `_hit_vertex` et les interactions souris utilisent
`_pick_pts` (`assembleur_tk.py:7649+`); le polygone participe au hit-test de
triangle. Aucun export XML ou Core ne les lit.

**Rôle et proposition.** Ce sont de vrais caches UI A. Ils ne doivent jamais
être sauvegardés, ni utilisés pour reconstruire une pose. Une évolution
possible serait de les déplacer dans une structure de pick séparée indexée par
`topoElementId`, mais cela n'est pas nécessaire pour MIG-CACHE-CLEANUP-001.

### 3.10 `group_id` historique

`group_id` n'est plus produit par la simulation, le dépôt manuel ou le
chargeur : les tests actuels vérifient explicitement son absence
(`tests/test_topology_element_ids.py:171, 338, 367, 505`). Les occurrences
restantes sont de compatibilité, documentation, tests de non-régression ou
structures hors `last_drawn`. C'est du legacy D : aucune migration de lecteur
du cache ne doit le restaurer.

## 4. Matrice dépendances : écrivains et lecteurs

| Champ | Fonctions qui écrivent / créent | Fonctions qui lisent | Impact de suppression |
|---|---|---|---|
| `pts` | `toLegacyDict`; `loadScenarioXml`; `_project_core_element_to_last_drawn`; aperçus MOVE/ROTATE | redraw/PDF, pick, snap/EdgeChoice, XML, ancre horloge | Bloquant : le Canvas et le XML requièrent une projection monde. |
| `topoElementId` | simulateur, loader XML, dépôt manuel | collection indexes, sélection, transformations Core, XML, debug | Bloquant : référence Core minimale. |
| `id` | simulateur, loader, dépôt manuel | listbox, status, filtres, index catalogue, XML | Migration catalogue/métadonnées nécessaire. |
| `labels` | simulateur, dépôt manuel, enrichissement post-load | rendu texte, calculs de simulation avant projection | Rendu + mode XML dégradé à migrer. |
| `mirrored` | simulateur, loader, dépôt manuel | writer XML, lecture legacy, debug | Writer/loader XML à basculer vers pose Core. |
| `topoGroupId` | projection auto, loader, dépôt manuel | debug et assertions/tests de projection | Faible : diagnostic/tests à adapter. |
| `orient` | ancienne création manuelle/fixtures | F11 GEO-ORIENT | Faible : debug à dériver du Core. |
| `_pick_pts` | `_rebuild_pick_cache` | hit-test sommet | Aucun impact Core ; conserver le cache. |
| `_pick_poly` | `_rebuild_pick_cache` | hit-test triangle | Aucun impact Core ; conserver le cache. |

## 5. Données hors entrée à ne pas confondre

### 5.1 `ScenarioAssemblage.last_drawn_local`

`last_drawn_local` est un attribut **du scénario**, non une clé de chacune de
ses entrées. `_autoEnsureLocalGeometry()` construit une copie locale et
`_autoRebuildWorldGeometryScenario()` peut encore reconstruire les points monde
par transformation (`assembleur_tk.py:4497-4572`). Les transformations AUTO
Core-first récentes l'appellent encore comme filet de compatibilité autour de
`auto_geom_state` (`8917+`, `8978+`, `9253+`).

Classification : D — legacy. Il contredit le contrat cible s'il est utilisé
pour rendre ou réécrire la géométrie. Sa suppression doit être un chantier
spécifique : elle exige de remplacer ces derniers initialisateurs/aperçus par
des snapshots de poses Core, pas de supprimer simplement l'attribut.

### 5.2 Champs signalés dans les audits précédents mais absents de l'entrée

| Nom | Emplacement réel | Conclusion |
|---|---|---|
| `group_id`, `edge_in`, `edge_out`, `vkey_in`, `vkey_out`, `group_pos` | Anciennes projections de groupe, XML historique, tests/docs | Pas un contrat `last_drawn` courant. |
| `bbox` | caches/objets de groupe et calculs de vue | Pas une clé de triangle projeté. |
| `rotation`, `angle`, `vertex_local_xy`, `local_points`, `local_vertices`, `local_coords` | `TopologyElementPose2D`, `TopologyElement.local_frame`, catalogue | Données Core ; ne pas les ajouter au cache. |
| `world_pts` | `self._drag` temporaire (`assembleur_tk.py:10030-10041`) | État de drag, pas `last_drawn`. |
| couleurs, sélection, handles | état Canvas/`self._sel` et IDs Tk | Pas stockés par entrée. |

## 6. Anomalies de cycle de vie

### 6.1 Écrit mais sans lecteur fonctionnel identifié

- `topoGroupId` : les lectures de l'entrée sont de diagnostic et de tests ;
  les vrais groupes sont résolus depuis le Core.
- `orient` : seulement diagnostic/F11 et fixtures, hors fonctionnalités de
  dessin ou de topologie.

### 6.2 Lu mais non produit de manière universelle

- `labels` : XML les complète après le chargement. Toute entrée créée par un
  chemin non standard doit donc être capable de rendre des labels absents.
- `orient` : précisément facultatif ; les diagnostics utilisent `<absent>`.
- `_pick_*` : créés paresseusement, jamais requis par une entrée fraîche.

### 6.3 Lecture/écriture transitoire acceptable

`pts` est écrit pendant les aperçus manuels. Ceci est acceptable seulement si
la fin de transaction est une projection du Core et si ESC rétablit la
projection Core ; ce contrat est déjà appliqué par les chemins manuels
Core-first. Il faut empêcher tout nouveau commit `pts -> pose`.

## 7. Champs supprimables et migrations préalables

### 7.1 Supprimables immédiatement du contrat de production

| Champ / état | Condition | Action future |
|---|---|---|
| `orient` | Mettre à jour F11 GEO-ORIENT et fixtures pour lire `element.meta["orient"]`. | Retirer les écritures manuelles résiduelles. |
| `topoGroupId` | Adapter diagnostic/tests pour appeler `world.get_group_of_element(topoElementId)`. | Retirer la projection de cette clé. |

Ces deux suppressions restent des modifications de code : elles ne sont pas
réalisées par le présent audit.

### 7.2 Nécessitant une migration préalable

| Champ | Prérequis exact |
|---|---|
| `mirrored` | Writer XML et toute compatibilité doivent dériver la valeur de la pose Core ; tests de round-trip à conserver. |
| `labels` | Rendu et chargement dégradé doivent connaître une API catalogue/Core à partir de `topoElementId` ou du `id` catalogue. |
| `id` | Remplacer listbox, mots, filtres et XML par une consultation des métadonnées catalogue du `TopologyElement`. |
| `pts` | Aucune suppression tant qu'il existe Canvas/PDF/hit-test/XML ; le but est de le traiter comme cache reconstruisible. |
| `last_drawn_local` | Remplacer les derniers aperçus AUTO par snapshots/projections Core. |

## 8. Fonctions bloquantes à traiter avant un nettoyage complet

1. `saveScenarioXml()` (`assembleur_io.py:391-505`) : persiste `id`,
   `mirrored`, `topoElementId` et `pts`. Il doit devenir un export de la
   projection Core, ou recevoir explicitement une projection validée.
2. `loadScenarioXml()` (`assembleur_io.py:522-815`) : reconstruit encore
   `pts` et `mirrored` depuis XML pour la compatibilité v4. Il devra reprojeter
   le Core une fois la compatibilité de fichier changée.
3. `_draw_triangle_screen()` et le redraw (`assembleur_tk.py:6306-6365,
   10677+`) : exige `pts` et `labels` dans le cache.
4. `_rebuild_pick_cache()` / hit-test (`5117-5145`, `7649+`) : exige `pts` ;
   il s'agit d'un consommateur légitime de cache UI, pas d'une violation Core.
5. `_autoEnsureLocalGeometry()` et `_autoRebuildWorldGeometryScenario()`
   (`4497-4572`) : reliquat `last_drawn_local` qui doit être isolé avant que
   la projection automatique soit exclusivement Core -> UI.
6. Les aperçus de drag/rotation (`9398-9510`) : écrire temporairement `pts`
   est légitime visuellement, mais ils doivent continuer à être encadrés par
   projection/annulation Core.

## 9. Architecture cible recommandée

### 9.1 Contrat minimal de l'entrée

Le contrat cible réaliste, après les migrations préalables, est :

```python
{
    # Référence stable vers l'autorité.
    "topoElementId": "T28",

    # Projection monde pour Canvas/PDF/hit-test — jamais une autorité métier.
    "pts": {"O": np.ndarray(...), "B": np.ndarray(...), "L": np.ndarray(...)},

    # Métadonnée de présentation provisoire, si le renderer n'interroge pas
    # encore le catalogue/Core directement.
    "id": 28,
    "labels": ("Bourges", "Bourges", "Lyon"),

    # Caches écran non persistés, idéalement hors dictionnaire à terme.
    "_pick_pts": {"O": (x, y), "B": (x, y), "L": (x, y)},
    "_pick_poly": [(x, y), (x, y), (x, y)],
}
```

`mirrored`, `topoGroupId` et `orient` n'appartiennent pas à ce contrat : ils
doivent être lus du Core lorsque nécessaires. `id` et `labels` peuvent ensuite
être retirés du dictionnaire seulement lorsque les consommateurs UI savent les
projeter depuis `TopologyElement`/catalogue.

### 9.2 Responsabilités

```text
TopologyWorld / TopologyElement
  ├─ identité élément, labels, orientation locale
  ├─ pose (R, T, mirrored)
  └─ groupe canonique et attachments
          │
          ▼
Projection unique
  ├─ points monde O/B/L
  ├─ métadonnées UI nécessaires
  └─ mise à jour atomique de CanvasObjectsCollection
          │
          ▼
CanvasObjectsCollection
  ├─ index topoElementId / id
  └─ cache de pick écran
```

## 10. Ordre recommandé pour MIG-CACHE-CLEANUP-001 et suivants

1. **CLEANUP-001A — retirer `orient` et `topoGroupId` des entrées.** Risque
   faible ; migrer F11 et tests avant les producteurs.
2. **CLEANUP-001B — rendre `mirrored` XML/Core-first.** Risque moyen, car le
   round-trip XML doit rester compatible.
3. **CLEANUP-002 — isoler/supprimer `last_drawn_local`.** Risque élevé :
   concerne les scénarios AUTO et leurs aperçus.
4. **CLEANUP-003 — réduire `labels` et `id` à une projection de catalogue.**
   Risque moyen à élevé : listbox, dictionnaire, XML et mode dégradé.
5. **CLEANUP-004 — formaliser une unique `rebuild_last_drawn_from_core()`.**
   Cette étape remplace les derniers producteurs ponctuels de `pts` hors
   aperçu UI et rend le cache entièrement reconstructible.

## 11. Décision de référence

Le projet peut maintenant imposer la règle suivante : **aucune écriture de
`last_drawn` ne modifie une vérité topologique ou une pose métier**. Les seules
écritures admises sont la projection depuis `TopologyWorld`, les caches écran,
et les aperçus strictement transactionnels qui sont ensuite abandonnés ou
écrasés par cette projection.

Cet audit ne modifie aucun code, aucun format XML et aucun test.

> Mise à jour MIG-CACHE-CLEANUP-001 : les champs `orient`, `topoGroupId` et
> `mirrored` ont été retirés des entrées runtime. Leurs valeurs éventuelles
> sont désormais résolues depuis `TopologyElement` / `TopologyWorld`; le
> format XML conserve seulement l'attribut historique `mirrored`, lu et
> validé contre la pose Core.
>
> Mise à jour MIG-CACHE-CLEANUP-002 : `labels` a également été retiré des
> entrées runtime. Le Canvas et le PDF lisent désormais
> `TopologyElement.vertex_labels` via `topoElementId`.
