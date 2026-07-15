# Stabilisation UI/Core — Phase 1

## Référentiel de l'existant aligné sur l'architecture cible topologique

**Projet :** AssembleurTriangles / La Chouette  
**Statut :** document de référence pour la stabilisation de l'existant  
**Référence conceptuelle cible :** [ARCHITECTURE_CIBLE_TOPOLOGIQUE.md](ARCHITECTURE_CIBLE_TOPOLOGIQUE.md)  
**Nature :** analyse, cartographie, critères d'observation et priorisation documentaire ; aucune modification de code n'est prescrite ici.

---

## 1. Rôle, périmètre et règles de lecture

Ce document remplace la première version de `STABILISATION_UI_CORE_PHASE1.md`. Il conserve son objectif : rendre observable et stabilisable l'architecture actuelle. Il l'interprète désormais à la lumière de la décision conceptuelle cible selon laquelle la topologie du scénario est la source de vérité, la géométrie monde une projection calculée, `_last_drawn` un cache UI et le canvas une vue.

Ce document ne dit donc pas que l'application actuelle respecte déjà la cible. Il distingue systématiquement :

| Marqueur | Sens |
|---|---|
| **Établi par le code actuel** | lecture directe des modules, appels et tests actuels |
| **Cible de référence** | règle issue de `ARCHITECTURE_CIBLE_TOPOLOGIQUE.md`; non supposée implémentée |
| **Écart à stabiliser** | différence constatée ou absence de garantie entre actuel et cible |
| **Risque plausible** | conséquence possible d'un flux ou d'une redondance ; pas un bug reproduit |
| **Question ouverte** | sujet intentionnellement non tranché |

### 1.1 Périmètre technique analysé

| Zone | Fichiers principaux | Rôle dans le chantier |
|---|---|---|
| UI et interactions | `src/assembleur_tk.py`, mixins Tk | points monde historiques, groupes UI, canvas, scénarios et callbacks |
| Topologie et géométrie Core | `src/assembleur_core.py` | éléments, poses, DSU, attaches, caches, frontières, chemins |
| Collage et simulation | `src/assembleur_edgechoice.py`, `src/assembleur_sim.py` | intention géométrique, attaches et scénarios auto |
| Persistance | `src/assembleur_io.py` | XML v4 hybride : Core + projection UI |
| Données externes | `src/assembleur_balises.py`, Excel/CSV | catalogue actuel, balises, calibration indirecte |
| Vérification | `tests/` | preuves actuelles, principalement non-GUI |

Les documents utilisés sont :

- [ANALYSE_FONCTIONNELLE_ET_ARCHITECTURE.md](ANALYSE_FONCTIONNELLE_ET_ARCHITECTURE.md), état des lieux global ;
- [ARCHITECTURE_CIBLE_TOPOLOGIQUE.md](ARCHITECTURE_CIBLE_TOPOLOGIQUE.md), référence conceptuelle cible ;
- le présent document, qui concentre les risques et préconditions de stabilisation de l'existant.

### 1.2 Ce que ce document apporte par rapport à l'ancienne Phase 1

| Partie de l'ancienne analyse | Toujours valide | Enrichissement imposé par la cible |
|---|---|---|
| Modèle hybride UI/Core | oui | ne pas le confondre avec le modèle souhaité |
| Inventaire `last_drawn`/groupes/DSU/XML | oui | qualifier ces structures : autorité actuelle, cache cible, compatibilité historique |
| Risques de synchronisation | oui | les ordonner autour de TG-RISK-001 et du validateur transversal |
| Opérations sensibles | oui | ajouter pour chacune le comportement cible, sans le mettre en œuvre |
| Transactions et rollback | oui | les relier à la publication d'une projection dérivée après validation |
| Sources de vérité | incomplète | séparer nettement source actuelle, source cible et données de transition |
| Catalogue, variantes et ancre | absent ou peu développé | les garder comme prérequis conceptuels, sans présumer de leur code |

---

# Architecture actuelle vs Architecture cible

## 2.1 Architecture actuelle : hybride et historiquement géométrique

**Établi par le code actuel.** L'assemblage existe sous plusieurs représentations mutables :

```text
Catalogue Excel courant
        │
        ├─ UI géométrique historique
        │    viewer._last_drawn / ScenarioAssemblage.last_drawn
        │    points monde O,B,L ; group_id/group_pos ; groups ; mirrored
        │                     │
        │                     │ _sync_group_elements_pose_to_core()
        │                     ▼
        └─ TopologyWorld
             TopologyElement locales + pose ; DSU ; TopologyGroup ; attachments
                                      │
                                      ▼
                            cache conceptuel → frontières → chemins → triplets

Canvas Tk = projection écran recréée depuis l'UI et certains dérivés Core
XML v4 = snapshot Core + triangles UI + groupes UI + état de vue
```

Les opérations manuelles commencent souvent par modifier les points monde UI, puis ajustent les poses Core : `_sync_group_elements_pose_to_core` (`src/assembleur_tk.py:686-765`). Lors du collage, des groupes UI sont fusionnés après l'application d'attaches Core (`src/assembleur_tk.py:10056-10240`). Lors du dégroupage, le Core produit d'abord les nouvelles composantes, puis l'UI les reconstruit (`src/assembleur_tk.py:7997-8159`). Le chargement XML restaure deux descriptions et relie ensuite leurs identités (`src/assembleur_io.py:720-965`).

L'état actuel n'est donc pas une architecture purement graphique, ni une architecture où le Core projette déjà l'UI de manière descendante. C'est un modèle hybride à synchronisations explicites et partielles.

## 2.2 Architecture cible : topologie autoritaire, projection descendante

**Cible de référence.** La définition durable d'un assemblage est :

```text
Catalogue de TriangleSlot et TriangleVariant
        +
Scénario topologique
  - variantes sélectionnées
  - attaches et orientations
  - ancre(s) persistante(s)
        ↓
Reconstruction géométrique déterministe
        ↓
Projection géométrique monde
        ↓
_last_drawn : cache de travail UI
        ↓
Canvas : vue
```

Dans ce modèle :

- la topologie définit les triangles logiques présents, les variantes sélectionnées, les attaches, leurs orientations et l'ancre ;
- la géométrie locale appartient aux variantes du catalogue ;
- les poses et coordonnées monde sont calculées ;
- `_last_drawn` peut être supprimé et régénéré sans perte de décision métier ;
- le canvas ne détient ni identité métier, ni attache, ni pose persistante autonome ;
- en cas de divergence, la topologie gagne et la projection est reconstruite ; aucune réconciliation symétrique n'est la règle cible.

## 2.3 Matrice de contraste

| Sujet | Architecture actuelle | Architecture cible | Écart à stabiliser avant migration |
|---|---|---|---|
| Autorité de géométrie | `pts` UI est modifié directement puis recopié vers pose Core | pose/points issus de la reconstruction | mesurer et rendre visible l'écart UI ↔ Core |
| `_last_drawn` | donnée de rendu et de manipulation, avec autorité opérationnelle implicite | cache calculé, mutable seulement comme aperçu | empêcher sa promotion implicite en vérité durable |
| Canvas | affichage recréé, mais interactions modifient immédiatement la géométrie UI | vue recevant des intentions | distinguer preview, commande et publication |
| Attaches | générées depuis le choix géométrique UI ; Core les stocke | relations stables entre slots/features | préserver la nomenclature, ne pas déduire les attaches du cache |
| Groupes | groupes UI ordonnés et groupes DSU liés par `topoGroupId` | composantes dérivées et/ou groupes métier à clarifier | ne pas simplifier avant d'avoir séparé les sémantiques |
| XML | snapshot Core et projection UI simultanément | scénario topologique autoritaire, géométrie cache optionnel | définir une hiérarchie de lecture et de validation |
| Déplacement | translation de points UI puis sync des poses | modification de l'ancre puis reconstruction | caractériser les gestes historiques et leurs intentions métier |
| Variantes/catalogue | catalogue Excel de triangles unitaires | slots, variantes et sélection de scénario | prérequis non encore implémenté |
| Validation | `validate_world` Core et relink XML ponctuel | validateur catalogue/scénario/reconstruction/UI/caches | TG-RISK-001 prioritaire |

## 2.4 Règle de non-confusion

La cible sert à interpréter et prioriser la stabilisation ; elle ne permet pas de déclarer automatiquement erroné tout comportement de l'existant. Par exemple, `TopologyWorld` porte aujourd'hui à la fois géométrie locale, pose, attaches, groupes, frontière et chemin. La cible distingue conceptuellement catalogue, scénario, reconstruction et caches, mais ne décide pas encore s'il faut éclater ou conserver `TopologyWorld` comme agrégat.

---

# Pourquoi la stabilisation reste nécessaire

Connaître la cible ne rend pas une refonte immédiate sûre. Le code actuel porte des dépendances fonctionnelles qui ne sont ni toutes explicitées ni toutes couvertes par les tests.

## 3.1 Dette technique à préserver avant de la réduire

| Dette / héritage | Observation | Danger d'une refonte directe |
|---|---|---|
| Écritures UI directes | les handlers modifient `last_drawn[*]['pts']` dans les mouvements, rotations et flips | perdre les subtilités d'interaction, de snapping et d'annulation |
| Groupes redondants | `self.groups` et groupes DSU n'ont ni même ID ni même ordre | simplifier sans sémantique connue peut supprimer une fonctionnalité cachée |
| Persistance double | XML v4 écrit snapshot Core et triangles/groupes UI | changer le format sans caractérisation casse les scénarios existants |
| Cache conceptuel/chemin | invalidations et recalculs ciblés | modifier les poses sans observer contours et triplets produit des erreurs tardives |
| Scénarios auto | vue/carte/géométrie auto possèdent des règles partagées | généraliser le manuel aux auto sans test peut introduire des effets croisés |
| Tolérances multiples | UI, attaches, contour, sérialisation utilisent des seuils de nature différente | une normalisation prématurée peut invalider un collage accepté historiquement |

## 3.2 Risques de régression

Une migration brutale peut modifier simultanément le geste utilisateur, les groupes, les attaches, les poses, les contours, les chemins et le format de fichier. Or l'application ne possède pas actuellement une couverture de non-régression intégrée pour : collage UI-Core, dégroupage complet, suppression/remapping `tid`, XML contradictoire, changement de scénario et isolations des scénarios auto.

La stabilisation consiste donc d'abord à observer, comparer et caractériser. Elle ne consiste pas à remplacer immédiatement les synchronisations par le moteur de reconstruction cible.

## 3.3 Dépendances implicites à documenter avant intervention

| Dépendance | Preuve actuelle | Ce qui doit être observé |
|---|---|---|
| index `tid` → groupe UI | `groups[*].nodes[*]['tid']` indexe `last_drawn`; suppression remappe tous les indices | conservation de l'ordre, rechargement XML et références temporaires |
| ID métier → ID Core | `TopologyWorld.format_element_id(id)` et `topoElementId` | collisions, manques et récupération sur fichier dégradé |
| points UI → pose Core | SVD dans `_sync_group_elements_pose_to_core` | erreur numérique et convention miroir |
| pose Core → frontière | `setElementPose` invalide la géométrie conceptuelle | rafraîchissement effectif du contour et du chemin |
| attache → composante | DSU et `rebuild_from_attachments` | translation des composantes vers les groupes UI |
| scénario actif → alias viewer | `_set_active_scenario` réaffecte `_last_drawn/groups` | références conservées par callbacks et états de sélection |
| balise → triplets | `_syncBalisesToWorld` avant création/recalcul | fraîcheur de la mesure et changement de calibration |

---

# Source de vérité

## 4.1 Décision de référence

**Cible de référence : la topologie du scénario est la source de vérité.** Elle doit porter les décisions métier : identité logique des triangles, variante active, attaches, orientation relative, ancre et métadonnées. La géométrie monde ne doit plus être considérée comme une autorité concurrente : elle est le résultat de ces décisions.

Cette règle est prospective. **Établi par le code actuel :** `TopologyWorld` porte déjà les éléments, locales, poses et attaches, mais l'UI reste autoritaire pendant plusieurs interactions. La stabilisation ne doit pas prétendre que la règle est déjà vraie partout ; elle doit rendre mesurable la transition entre les deux états.

## 4.2 Pourquoi la géométrie n'est pas une autorité métier cible

Une même topologie peut être projetée différemment en cas de variante, d'ancre, de calibration ou de reconstruction. Les coordonnées ne préservent pas à elles seules l'intention « telle arête de tel triangle est attachée à telle autre avec telle orientation ». Elles dépendent aussi d'arrondis, de transformations successives, du cache UI et du canvas.

Par conséquent, la géométrie peut prouver ou invalider une projection, mais elle ne doit pas redéfinir les attaches, l'identité de feature, la variante choisie ou l'ancre persistante.

## 4.3 Pourquoi `_last_drawn` est un cache cible

`_last_drawn` est indispensable pour l'ergonomie actuelle : dessin, picking, preview de drag, bbox, sélection, tooltip et compatibilité avec les callbacks Tk. Son futur statut de cache ne signifie pas qu'il doit disparaître ni qu'il est inutile. Cela signifie :

- son contenu doit être reconstructible depuis les données métier ;
- ses mutations transitoires doivent être distinguées d'une décision durable ;
- un redraw ou une régénération ne doivent pas perdre une attache, une variante ou une ancre ;
- une divergence constatée doit être diagnostiquée puis résolue par reconstruction descendante, pas par promotion silencieuse du cache.

## 4.4 Conséquences par couche

| Couche | Source actuelle | Statut cible | Conséquence de stabilisation |
|---|---|---|---|
| UI | points `pts`, groupes UI et sélection | vue + cache + intentions | documenter chaque écriture qui devient durable |
| Core | topologie, poses, frontières et chemins | scénario topologique + données dérivées à séparer conceptuellement | distinguer ce qui est autorité, projection ou cache interne |
| XML | Core + projection UI | persistance de l'autorité, cache facultatif et vérifié | rendre l'ordre de confiance explicite avant migration |
| Synchronisations | UI→Core et Core→UI selon l'opération | commandes domaine → reconstruction → cache | ne supprimer aucun flux avant caractérisation et validation |

---

# 5. Représentations actuelles et propriété des données

## 5.1 Canvas et état graphique Tkinter

`TriangleViewerManual._redraw_from` (`src/assembleur_tk.py:6003-6084`) efface puis recrée le canvas à partir des triangles UI, des layers et de certains dérivés Core. Les items Canvas, leurs IDs, les coordonnées écran, les surbrillances, le tooltip, les previews de drag et les guides sont donc recalculables. Le zoom et le pan agissent sur la projection monde-écran : `_world_to_screen` et `_screen_to_world` appliquent `zoom`, `offset` et l'inversion de l'axe Y écran (`src/assembleur_tk.py:5970-5973`).

| Donnée graphique | Porteur actuel | Autorité actuelle | Statut cible |
|---|---|---|---|
| polygones/lignes/textes | `tk.Canvas` | aucune autorité métier | vue jetable |
| IDs Canvas | widgets Tk | aucune | identifiants temporaires |
| cache de pick | `_pick_*` dans items UI | accélérateur | cache invalidable |
| `_sel`, `_drag`, `_edge_choice` | viewer | interaction en cours | état transitoire UI |
| aides de snapping/tooltip | viewer + Canvas | affichage | projection / diagnostic |
| zoom, offset, layers | viewer/scénario | état de vue | préférence de présentation, non topologique |

**Règle de stabilisation :** une erreur d'affichage n'implique pas automatiquement une erreur topologique ; réciproquement, un affichage correct ne prouve pas une attache ou une pose Core correcte.

## 5.2 Géométrie UI historique

Chaque entrée de `ScenarioAssemblage.last_drawn` contient la forme rendue et manipulée. Le scénario manuel partage directement cette liste et `groups` avec le viewer (`ScenarioAssemblage.__init__`, `src/assembleur_core.py:56-112`; `TriangleViewerManual.__init__`, `src/assembleur_tk.py:376-678`).

| Champ / structure | Usage établi | Reconstructible dans la cible ? | Vigilance |
|---|---|---|---|
| `id` | rang catalogue et identité pratique | oui, depuis slot/variant futur | aujourd'hui lié à `Txx` et au texte listbox |
| `labels` | affichage O/B/L | oui depuis catalogue | XML dégradé peut utiliser des labels vides |
| `pts` | sommets monde de rendu et interaction | oui, depuis reconstruction | actuellement muté avant Core |
| `orient` | orientation intrinsèque catalogue | oui, depuis variante | distinct du flip utilisateur |
| `mirrored` | miroir de pose utilisateur | oui, si relation/ancre/reconstruction l'exprime | ne pas le confondre avec `orient` |
| `group_id`, `group_pos` | organisation UI, interaction, ordre | à déterminer | `tid` mutable et ordre non représenté dans Core |
| `topoElementId`, `topoGroupId` | ponts d'identité | oui, projection du scénario | cohérence à valider |
| `_placed_ids` | filtre UI/listbox | oui, dérivé des éléments projetés | ne doit pas décider seul de l'existence métier |
| `self.groups` | nœuds `tid`, edges UI, bbox | partiellement, selon sémantique groupe | ne pas réduire à une composante sans audit |

## 5.3 Core actuel

`TopologyWorld` réunit aujourd'hui plusieurs responsabilités : éléments physiques, coordonnées locales, poses, nodes/edges, union-find, attachements, groupes, cache conceptuel, frontières, balises et chemin unique (`src/assembleur_core.py:1554-1615`).

| Objet Core | Donnée portée | Classification par rapport à la cible |
|---|---|---|
| `TopologyElement` | locales O/B/L, longueurs, labels/types, `meta`, pose | mélange actuel de catalogue/élément projeté ; les locales sont conceptuellement catalogue, la pose reconstruction |
| `TopologyAttachment` | features, `t`, `edgeFrom`, `mapping`, source | préfigure l'attache de scénario ; identité encore fondée sur `element_id` actuel |
| DSU / `TopologyGroup` | composantes, membres, `attachment_ids` | composante dérivée possible ; pas automatiquement groupe métier cible |
| `ConceptGroupCache` | occurrences, géométrie conceptuelle, frontières | cache dérivé |
| `TopologyBoundaries` | cycle/segments/orientation | cache dérivé de topologie et géométrie Core |
| `TopologyChemins` | snapshot de contour, masque, triplets | dérivé ou semi-métier à clarifier |
| `balisesWorld` | points monde de référence | donnée dérivée de balises et calibration |

`TopologyElement._try_build_default_local_coords` construit des locales à partir des longueurs et de `orient`, et `TopologyElementPose2D` applique la pose locale-vers-monde (`src/assembleur_core.py:354-534`). Le Core possède ainsi déjà les briques nécessaires pour ne pas dépendre structurellement du canvas. Il ne porte pas encore explicitement le catalogue de slots/variantes, la sélection de variante ou une ancre de scénario au sens cible.

## 5.4 Sources actuelles, sources cibles et transition

| Concept | Autorité observée maintenant | Source cible | Cache/projection autorisé | Écart critique |
|---|---|---|---|---|
| identité triangle | `id` Excel puis `Txx` Core | `TriangleSlot` stable | `id` de rendu | IDs fondés sur rang et `tid` mutable |
| forme locale | DataFrame, simulation et Core dupliquent la construction | `TriangleVariant` | locales Core de projection | variantes non modélisées |
| pose monde | `pts` UI puis pose Core | reconstruction depuis ancre et attaches | poses/points UI | double écriture actuelle |
| attache | `TopologyAttachment` après choix UI | scénario topologique | aide/aperçu UI | orientation cible peut être incomplètement explicite |
| groupe | groupes UI + DSU | composante dérivée et/ou groupe métier à trancher | structures UI | deux identités, deux ordres |
| frontière/contour | cache Core, rendu UI | dérivé | cache / dessin | invalidation non uniformisée |
| chemin/triplets | `TopologyChemins` | dérivé ou métadonnée à décider | Treeview/export | fraîcheur après mutation |
| ancre | positionnement UI, auto state et carte selon cas | scénario | drag temporaire | pas d'abstraction d'ancre unifiée |
| XML | deux vérités persistées | scénario autoritaire + cache identifié | cache vérifiable | ordre de priorité incomplet |

---

# Positionnement du validateur transversal

## 6.1 TG-RISK-001 devient le risque principal

**TG-RISK-001 — absence de validation UI/Core** est le risque principal, car il transforme tous les autres risques en divergences silencieuses et difficiles à localiser. Sans mesure transversale, il est impossible de savoir si un correctif local, une nouvelle interaction ou un chargement XML a laissé :

- un triangle UI sans élément Core ;
- un élément Core non projeté ;
- des points UI incompatibles avec la pose Core ;
- un groupe UI couvrant plusieurs composantes Core ;
- un `topoGroupId` périmé ;
- une frontière, un chemin ou un cache de pick calculé sur un état ancien ;
- une duplication de scénario partageant encore une référence inattendue.

Le validateur est donc l'instrument qui permet de stabiliser l'hybride avant d'en réduire les doubles représentations. Il n'est ni une implémentation du moteur cible, ni une réconciliation automatique, ni une manière de faire gagner `_last_drawn` sur le scénario.

## 6.2 Ce qu'il doit mesurer

| Niveau | Mesures attendues | Existant réutilisable | Lacune actuelle |
|---|---|---|---|
| Topologie Core | références, attachements, DSU, couvertures, caches | `TopologyWorld.validate_world` (`src/assembleur_core.py:4210`) | ne lit pas l'UI |
| Identité UI/Core | bijection item UI ↔ `TopologyElement`, unicité, éléments orphelins | relink de `loadScenarioXml` | validation seulement au chargement |
| Géométrie UI/Core | locales + pose Core → sommets attendus ; comparaison avec `pts` | `localToWorld`, sync SVD | aucune comparaison persistante |
| Groupes | `groups[*].nodes`, `topoGroupId`, DSU canonique, ordre/UI metadata | contrôles XML | aucune vérification après chaque opération |
| Projection | cache de pick, Canvas, bbox et IDs cohérents avec `last_drawn` | `_rebuild_pick_cache`, `_redraw_from` | aucun rapport unifié |
| Dérivés | frontière, chemin, triplets, balises et génération/invalidation | invariants de `TopologyChemins` | fraîcheur et dépendances non centralisées |
| Cible future | catalogue, slots, variantes, arbre, ancre, reconstruction | aucun modèle complet | critères conceptuels seulement |

### 6.2.1 Mesure géométrique minimale dans le modèle actuel

Pour chaque item UI ayant `topoElementId` :

1. récupérer les trois coordonnées locales de l'élément Core ;
2. les transformer avec la pose Core, miroir compris ;
3. comparer les résultats à `last_drawn[*]['pts']['O'/'B'/'L']` ;
4. rapporter l'écart par sommet et la tolérance utilisée ;
5. ne modifier ni l'UI ni le Core.

Cette mesure ne présume pas encore de la représentation cible des variantes. Elle caractérise précisément le contrat implicite de `_sync_group_elements_pose_to_core`.

### 6.2.2 Classes de tolérances à séparer

| Classe | Objet | Ne doit pas être confondue avec |
|---|---|---|
| pose/reconstruction | égalité coordonnées UI-Core, matrices et locales | hit-test pixels |
| attache | coïncidence ou orientation de features reliées | arrondi XML |
| affichage | picking, snap, épaisseur de traits | intégrité topologique |
| sérialisation | comparaison après écriture/lecture | tolérance de collage |
| dégénérescence | longueurs nulles, NaN, aire nulle | tolérance ergonomique |

## 6.3 Ce qu'il ne doit pas faire

Le validateur ne doit pas :

- modifier `pts`, poses, groupes, attaches, XML ou canvas pour « réparer » ;
- décider seul quelle représentation gagne dans le modèle actuel ;
- convertir automatiquement une coïncidence graphique en attache ;
- déduire des relations métier de la seule proximité de points ;
- introduire la future classe de catalogue ou le moteur de reconstruction comme effet de bord ;
- masquer une divergence derrière un recalcul silencieux ;
- dépendre d'IDs Canvas comme clés métier ;
- être utilisé comme substitut de tests de caractérisation.

Il doit produire un rapport stable : code, sévérité, couche, scénario, IDs concernés, valeurs observées, valeurs attendues, tolérance, contexte d'opération et nature de la divergence. Son résultat doit pouvoir être lu par un humain, exploité dans un test, ou attaché à une investigation.

## 6.4 Modes d'observation à prévoir conceptuellement

| Moment | But | Attente de stabilisation |
|---|---|---|
| après opération sensible | localiser l'introduction éventuelle de divergence | observation sans correction |
| après chargement XML | vérifier relink, poses et caches | compléter les contrôles XML existants |
| avant sauvegarde | signaler une projection non cohérente | ne pas changer le fichier automatiquement |
| après changement de scénario | détecter fuite de références ou cache de scénario précédent | isoler manuel/auto |
| dans les tests | caractériser le comportement actuel | construire une base avant migration |
| diagnostic manuel | comprendre un cas utilisateur | rapport détaillé et reproductible |

---

# Cartographie des flux de synchronisation

## 7.1 Vue d'ensemble

```text
Flux actuels observés

UI points/groupes ──────────────► Core poses / attaches / DSU
Core dégroupage / frontières ───► UI groupes / dessin
XML UI + XML Core ──────────────► UI + Core + relink croisé
Core snapshot + UI cache ───────► XML v4
Core/caches ────────────────────► Canvas via last_drawn et redraw

Flux cibles attendus

Commande UI ─► scénario topologique ─► reconstruction ─► last_drawn ─► canvas
```

## 7.2 Inventaire détaillé

| Flux | Fonctions / données | Nécessité actuelle | Statut par rapport à la cible | Surveillance requise |
|---|---|---|---|---|
| UI → Core (poses) | `_sync_group_elements_pose_to_core`, `setElementPose` | nécessaire pour l'hybride | flux de transition ; deviendrait reconstruction inversement | écart sommets UI/Core |
| UI → Core (attaches) | `EdgeChoiceEpts.createTopologyAttachments`, `apply_attachments` | nécessaire après collage | la commande métier d'attache reste nécessaire, mais pas la déduction depuis cache | mapping features/orientation et atomicité |
| UI → Core (suppression) | `_ctx_delete_group`, `removeElementsAndRebuild` | nécessaire | future commande scénario | ordre UI avant Core à surveiller |
| UI → Core (balises) | `_syncBalisesToWorld` | nécessaire aux triplets | donnée externe dérivée | calibration et fraîcheur |
| Core → UI (dégroupage) | `degrouperAtNode`, `_applyDegrouperResultToTk` | nécessaire | future projection reconstruite | absence de rollback UI |
| Core → UI (contours) | `computeBoundary`, `_group_outline_segments_topo` | nécessaire au rendu | projection/cache | invalidation après pose/attache |
| Core → UI (chemin) | `TopologyChemins`, `refreshCheminTreeView` | nécessaire | dérivé projeté | fraîcheur et groupe touché |
| `_last_drawn` → Core | points fit par SVD | fréquent | suspect à long terme : cache ne doit pas définir le domaine | mesure de dérive et mutations durables |
| Core → `_last_drawn` | import XML, auto rebuilds, résultat de dégroupage indirect | partiel | flux cible dominant | définir projection complète avant suppression de flux inverse |
| XML → Core | `_importPhysicalSnapshot` | autorité Core actuelle du snapshot | compatible si snapshot devient scénario autoritaire | validité/versions |
| XML → UI | triangles, groupes, listbox, carte | compatibilité historique | cache à vérifier/régénérer | double vérité |
| Core → XML | `_exportPhysicalSnapshot`, chemin | persistance | scénario cible | données dérivées explicitement marquées |
| `_last_drawn` → XML | points et groupes UI | compatibilité historique | cache optionnel seulement | ne doit pas gagner au load |

## 7.3 Flux nécessaires, suspects et à surveiller

| Catégorie | Flux | Justification |
|---|---|---|
| Nécessaire maintenant | UI→Core pose, UI→Core attache, Core→UI dégroupage | le modèle hybride ne peut pas fonctionner sans eux |
| Nécessaire dans la cible | commande UI→scénario, scénario→reconstruction, projection→UI/Canvas | flux asymétrique de référence |
| Suspect | `_last_drawn`→Core comme source générale, XML projection→UI sans comparaison | risque de promotion de cache |
| À surveiller | groupes UI↔DSU, XML hybride, auto states partagés, chemin après mutation | frontières d'instabilité majeures |

---

# Analyse de toutes les opérations sensibles

Les rubriques « futur comportement cible » décrivent le résultat architectural attendu, sans constituer une correction ni une séquence de modification de code.

## 8.1 Collage et création d'attaches

| Aspect | État actuel observé | Cible de référence | Risque de divergence |
|---|---|---|---|
| Déclencheur | drag sur canvas, aide de snap, relâché `_on_canvas_left_up` | intention UI « attacher ces features » | le geste peut dépendre de cache/picking obsolète |
| Géométrie UI | transform rigide appliqué d'abord aux `pts` du groupe mobile | aperçu temporaire possible, non autoritaire | l'UI peut être modifiée avant validation Core |
| Core | sync des poses, création de `TopologyAttachment`, transaction Core, DSU/fusion | attache scénario validée avec orientation explicite | plusieurs écritures séquentielles sans rollback de couches |
| Groupes | fusion manuelle de `groups`, `group_id`, `group_pos`, `topoGroupId` | composante/relation dérivée ou groupe métier explicite | ordre UI et composante Core peuvent diverger |
| Dérivés | cache conceptuel, contour, redraw | reconstruction complète ou ciblée, puis caches | chemin/contour ne sont pas systématiquement revalidés |

**Fonctions actuelles :** `TriangleViewerManual._on_canvas_left_up` (`src/assembleur_tk.py:9923-10248`), `EdgeChoiceEpts.createTopologyAttachments` (`src/assembleur_edgechoice.py`), `TopologyWorld.apply_attachments` (`src/assembleur_core.py:3237`).

**Observation prioritaire :** avant et après collage, comparer identités UI/Core, poses, attachements effectivement créés, groupe DSU canonique, membres des groupes UI, frontière et chemin éventuel. La coïncidence visible ne doit pas être confondue avec une attache persistée.

## 8.2 Détachement / suppression d'attache

Le code actuel offre un dégroupage ciblé par nœud, pas une abstraction générale explicitement nommée « suppression d'attache ». `TopologyWorld.degrouperAtNode` sélectionne des attaches incidentes, les retire, reconstruit les DSU et retourne les composantes (`src/assembleur_core.py:3734-3876`).

| Aspect | État actuel observé | Cible de référence | Risque de divergence |
|---|---|---|---|
| Intention | clic droit sur nœud, option conditionnelle | commande explicite de suppression d'attache(s) | le terme « dégroupage » mélange plusieurs sémantiques |
| Core | suppression d'un sous-ensemble d'attaches + rebuild complet | suppression relationnelle, recalcul des composantes | résultat Core correct mais UI non reconstruite |
| UI | `_applyDegrouperResultToTk` répartit les éléments en groupes UI et les décale visuellement | projection régénérée ; groupes UI distingués des composantes | déplacement visuel renvoyé vers poses Core |
| Ancrage | non modélisé explicitement | chaque composante doit avoir une politique de placement | composante devenue indépendante sans règle stable |
| Chemin | supprimé si le groupe touché portait le chemin | invalider/reconstruire selon dépendances | validité des triplets et caches non uniformisée |

**Question ouverte conservée :** le futur « détachement » doit-il désigner uniquement la suppression d'une attache, la séparation d'une composante, ou la dissociation d'un groupe UI ? Aucun de ces sens ne doit être assimilé aux autres sans décision métier.

## 8.3 Suppression d'un triangle ou d'un groupe

`_ctx_delete_group` (`src/assembleur_tk.py:9123-9208`) supprime l'ensemble du groupe UI ciblé. Il capture d'abord les IDs Core, réinsère les triangles dans la listbox, filtre et remappe `_last_drawn`, puis appelle `TopologyWorld.removeElementsAndRebuild`.

| Aspect | État actuel observé | Cible de référence | Risque de divergence |
|---|---|---|---|
| Entité supprimée | groupe UI et ses triangles | présence d'un slot/élément dans un scénario, plus attaches affectées | le groupe UI n'est pas nécessairement le bon agrégat métier |
| IDs | remapping des `tid` après filtration | identifiants stables de slot/élément ; cache re-généré | liens UI/XML indexés par position |
| Core | purge des éléments, nœuds, attaches et groupes orphelins | modification atomique du scénario, puis reconstruction | UI est modifiée avant le rebuild Core |
| Chemin | supprimé si son groupe Core est supprimé | dérivé invalidé | autres caches dépendants à confirmer |
| Retour arrière | pas de rollback UI/Core commun | publication seulement après validation | état partiel si exception Core |

**Validation à exiger lors de l'observation :** aucune attache ne référence un élément supprimé ; aucune entrée UI/groupe ne référence un ancien `tid`; aucun élément Core actif ne reste sans projection attendue.

## 8.4 Dégroupage

Le dégroupage est la frontière Core→UI la plus révélatrice : le Core reconstruit ses composantes, mais l'UI doit traduire ce résultat dans une structure ordonnée, graphique et manipulable.

| Étape actuelle | Données touchées | Cible de référence | Risque |
|---|---|---|---|
| collecter attaches à retirer | `attachments`, nœud cible, éléments du groupe | commande relationnelle sur scénario | sélection d'attaches implicite difficile à caractériser |
| rebuild Core | DSU, groupes, coverages, cache | recalcul de composantes | succès Core avant UI |
| construire résultat normatif | `mainGroupId`, `newGroupIds`, `movedElementIdsByGroup` | projection de composantes ou groupes métier | mapping elementId→tid manquant |
| reconstruire UI | `groups`, `group_id`, `group_pos`, bbox | reconstruction de cache UI | ordre et `edge_in/out` ne sont pas déduits du Core |
| décaler visuellement | points UI puis sync des poses | règle d'ancrage/composantes explicite | séparation graphique devient une décision géométrique durable |

Le futur comportement cible ne doit pas être déduit de la seule technique actuelle de décalage en pixels. Si plusieurs composantes sont autorisées, la politique d'ancrage et de position de chaque composante reste une question ouverte.

## 8.5 Chargement XML

`loadScenarioXml` valide l'enveloppe v4 et le snapshot, restaure l'UI historique, importe un nouveau monde Core, puis effectue un relink strict des éléments et groupes (`src/assembleur_io.py:540-965`).

| Élément | État actuel | Cible de référence | Stabilisation nécessaire |
|---|---|---|---|
| source Excel | peut être chargée, sinon mode dégradé | catalogue référencé/versionné ou contenu minimal | documenter l'autorité si catalogue absent |
| triangles UI | points, miroir, groupes lus avant Core | cache à régénérer ou cache explicitement vérifié | comparer points UI et poses Core |
| Core | `_importPhysicalSnapshot` crée éléments, poses, attaches et cache | scénario topologique autoritaire | versionner l'autorité du snapshot |
| relink | valide `topoElementId` et groupe canonique | validation transversale de niveau identité/groupe | compléter par mesure géométrique |
| chemin/guides | restaurés avec références Core | dérivés ou métadonnées explicitement définis | vérifier les dépendances après reconstruction |

Le chargement est un cas de référence pour le validateur : il contient déjà un contrôle identitaire utile mais ne résout pas l'ambiguïté de deux géométries persistées.

## 8.6 Sauvegarde XML

`saveScenarioXml` (`src/assembleur_io.py:389-539`) écrit un snapshot physique Core, le chemin, la source Excel, la carte, le compas, les guides, la listbox, les groupes UI, les triangles UI et les associations de mots.

| Question | État actuel | Cible de référence |
|---|---|---|
| Quelle donnée est durable ? | Core et projection UI sont tous deux écrits | scénario topologique et références de catalogue |
| Quelle donnée est cache ? | non marquée explicitement | géométrie monde/UI facultative et vérifiable |
| Quelle donnée gagne à l'ouverture ? | relink identitaire, pas de comparaison de pose | topologie/scénario gagne, cache régénéré ou rejeté |
| Quel contrat de compatibilité ? | version v4 stricte | version de schéma + stratégie de migration à décider |

La stabilisation ne doit pas modifier le format. Elle doit fournir les éléments permettant de distinguer, dans chaque fichier, le snapshot d'autorité, les données de compatibilité et les caches qui devront être contrôlés.

## 8.7 Déplacement

| Aspect | État actuel | Cible de référence | Risque |
|---|---|---|---|
| geste | handlers souris modifient directement les points monde | intention « déplacer l'assemblage/composante » | intention métier non explicitée |
| points | translation appliquée aux membres UI | projection temporaire possible | points deviennent autorité de fait |
| poses | sync par élément Core | poses obtenues par reconstruction depuis ancre | divergence entre deux syncs |
| auto | `auto_geom_state` partagé, rebuild/sync de mondes auto | ancre/scénario explicitement défini | effets inter-scénarios auto |
| vue | zoom/pan restent purement visuels | inchangé | ne pas confondre déplacement de vue et déplacement métier |

Dans la cible, le déplacement global modifie la pose de l'ancre, puis reconstruit la composante. Une transformation de cache pendant le drag reste possible, mais ne devient durable qu'après traduction en donnée métier explicite et validation.

## 8.8 Miroir / retournement

`_ctx_flip_selected` reflète les points de tous les membres du groupe UI, bascule `mirrored`, puis synchronise chaque pose Core (`src/assembleur_tk.py:9471-9529`). `TopologyWorld.flipGroup` existe également (`src/assembleur_core.py:3087-3132`) mais n'est pas le flux UI principal relevé.

| Sujet | État actuel | Cible de référence | Risque |
|---|---|---|---|
| `orient` | orientation intrinsèque de catalogue / locales | propriété de la variante | ne pas le confondre avec une pose |
| `mirrored` | transformation utilisateur portée par UI puis pose Core | orientation/attache/ancre explicitement déterminée | double réflexion ou perte de convention |
| groupe | flip de tous les membres | règle de transformation de composante | sémantique groupe non tranchée |
| validation | aucune comparaison systématique UI/Core | reconstruction et validation géométrique | erreur discrète de signe/déterminant |

## 8.9 Rotation

La rotation manuelle `_ctx_rotate_selected` prépare un snapshot des points du groupe puis `_on_canvas_motion_update_drag` recalcule la rotation à partir de ce snapshot afin de limiter les erreurs cumulatives. L'UI recalcule bbox, synchronise poses et redessine. En auto, elle modifie `auto_geom_state`, reconstruit tous les mondes auto et synchronise leurs poses.

La cible distingue : rotation de vue (sans effet métier), aperçu de rotation (cache), et changement durable de l'ancre ou de la relation qui définit la pose. La stabilisation doit d'abord caractériser quelle de ces intentions est réellement portée par chaque action actuelle.

## 8.10 Changement de scénario

`_set_active_scenario` (`src/assembleur_tk.py:4502-4641`) sauvegarde/restaure vue et carte, échange les alias viewer vers `last_drawn/groups`, réconcilie certaines métadonnées auto, reconstitue `_placed_ids`, invalide le cache de pick, redessine et rafraîchit les widgets.

| Risque actuel | Pourquoi | Cible de référence |
|---|---|---|
| référence de scénario précédent | callbacks/états viewer pointent vers listes remplacées | projection strictement attachée au scénario actif |
| confusion manuel/auto | manuel partage les objets viewer ; auto comporte des états partagés | propriété de chaque donnée explicitée |
| cache UI périmé | Canvas, pick, sélection, guides, chemin | caches invalidés et régénérés à l'activation |
| collision d'identité | IDs et mappings reconstruits selon scénario | slots/attaches stables et portés par scénario |

Le changement de scénario n'est pas une opération topologique cible ; c'est une publication d'un autre état autoritaire et de sa projection. Il doit donc devenir un cas de validation et de caractérisation, pas seulement de rendu.

---

# 9. Contrats de cohérence à observer dans l'existant

Les contrats suivants ne sont pas tous garantis par le code. Ils forment le registre que le validateur transversal devra pouvoir constater avant toute réduction de l'autorité UI.

| ID | Contrat actuel à mesurer | Contrôle existant | Cible associée | Criticité |
|---|---|---|---|---|
| TG-INV-001 | tout item UI placé référence un élément Core existant | relink XML, contrôles locaux | projection bijective de l'état autoritaire | élevée |
| TG-INV-002 | un élément Core actif est projeté au plus une fois dans l'UI | aucun global | cache complet sans duplication | critique |
| TG-INV-003 | les sommets UI égalent les locales Core transformées par la pose | sync unidirectionnelle, pas de mesure | géométrie reconstructible | critique |
| TG-INV-004 | `mirrored` UI et pose Core sont cohérents ; `orient` reste intrinsèque | logique locale | variante ≠ pose | élevée |
| TG-INV-005 | groupe UI non vide correspond à une seule composante Core canonique | relink XML | composante/groupe explicitement distingués | élevée |
| TG-INV-006 | chaque attache référence des features valides d'éléments existants | `_validate_attachment_p2`, `validate_world` | attache de scénario stable | critique |
| TG-INV-007 | aucune attache survivante ne référence un élément supprimé | `removeElementsAndRebuild` | suppression atomique de scénario | critique |
| TG-INV-008 | pose ou attache invalide les caches conceptuels concernés | APIs Core ciblées | projection régénérable | élevée |
| TG-INV-009 | frontière/chemin/triplets sont frais pour les dépendances dont ils dépendent | invariants internes chemin | dérivés invalidables | élevée |
| TG-INV-010 | `tid`/groupes UI restent cohérents après suppression/remap | code de suppression, validation XML | cache sans ID durable positionnel | élevée |
| TG-INV-011 | un scénario actif n'expose aucun cache du scénario précédent | invalidations partielles | publication isolée d'une projection | moyenne |
| TG-INV-012 | les valeurs numériques sont finies et les tolérances explicites | non global | reconstruction déterministe | moyenne |

---

# 10. Groupes, composantes, frontières et chemins

## 10.1 Le mot « groupe » doit rester non résolu

Le code actuel emploie simultanément :

1. un **groupe UI** : dictionnaire indexé par `ui_gid`, ordonné, contenant des `tid`, une bbox et des métadonnées d'arêtes ;
2. un **groupe Core** : composante maintenue par DSU et attachements, avec `element_ids` et `attachment_ids` ;
3. une **forme de groupe** utilisée pour déplacement, overlap et contour ;
4. possiblement une notion fonctionnelle héritée, non prouvée par la seule connexité.

La cible ne permet pas de supprimer cette ambiguïté par décret. Elle pose une hypothèse forte : une composante connexe peut devenir un dérivé des attaches si les groupes ne portent aucune autre sémantique. Cette hypothèse doit rester séparée de l'existant tant que les rôles de `group_pos`, `edge_in/out`, ordre de `nodes`, sélection et ancrage n'ont pas été caractérisés.

| Question | État actuel | Cible / décision non prise |
|---|---|---|
| groupe UI = composante Core ? | imposé au load XML pour un groupe non vide ; non validé partout | non présumer de l'équivalence |
| groupe Core = composante connexe ? | oui, DSU/rebuild d'attaches | cible : composante dérivée possible |
| ordre des membres | présent uniquement côté UI | peut être cache, ou sémantique métier à conserver |
| `edge_in/out` | utilisé dans groupes UI/simulation et XML | statut métier ou cache non décidé |
| ancre de groupe | absente comme concept unifié | cible : ancre de scénario/composante à définir |
| groupe métier | non identifiable avec certitude | ne pas l'éliminer avant audit |

## 10.2 Frontières et contours

La frontière Core est calculée depuis les éléments, poses, attachements et graphe conceptuel : `TopologyWorld.computeBoundary` assure la géométrie conceptuelle puis délègue à `TopologyBoundaries.compute` (`src/assembleur_core.py:1979-2015`). `_group_outline_segments_topo` construit ensuite des segments affichables en reliant éléments UI et Core (`src/assembleur_tk.py:6816-6870`).

| État | Source actuelle | Invalidation observée | Cible |
|---|---|---|---|
| cache conceptuel | Core éléments/poses/attaches | `invalidateConceptGraph/Geom`, commit et `setElementPose` | projection dérivée | 
| frontière | cache conceptuel | `computeBoundary` explicite ou recompute ciblé | cache déterministe |
| contour UI | Core + groupes/points UI | redraw | vue de projection |
| contour-only | option UI | redraw seulement | présentation sans autorité |

**Risque à surveiller :** si les points UI sont déplacés sans synchronisation, la frontière Core peut être géométriquement correcte pour le Core mais visuellement différente. Si une pose Core change sans reconstruction du cache UI, l'écart inverse est possible. TG-INV-003 doit rendre cette situation visible.

## 10.3 Chemins et triplets

`TopologyChemins` est propre à chaque `TopologyWorld`. Il conserve un snapshot de contour, un masque, un ordre de nœuds et des triplets ; `creerDepuisGroupe`, `appliquerEdition` et `recalculerChemin` contrôlent ses invariants internes (`src/assembleur_core.py:990-1533`).

| Dépendance du chemin | État actuel | Cible de référence |
|---|---|---|
| attachements / composante | frontière et cycle Core | dérivé après scénario/reconstruction |
| poses et géométrie | triplets calculent des mesures monde | projection dont la fraîcheur doit être suivie |
| balise | injectée par `_syncBalisesToWorld` | entrée externe explicitement versionnée/calibrée |
| édition UI | masque et orientation commandent le Core | commande métier ou édition de dérivé à clarifier |
| persistance | chemin XML sauvegardé avec snapshot | statut durable à préciser |

La stabilisation doit déterminer, par observation, quels changements de pose, d'attache, de variante ou d'ancre rendent le chemin inutilisable. Elle ne doit pas changer sa politique d'invalidation dans cette phase.

---

# 11. Persistance, restauration et compatibilité

## 11.1 Ce qui est actuellement persisté

Le XML v4 combine plusieurs niveaux :

| Donnée XML actuelle | Provenance | Statut actuel | Statut cible souhaité |
|---|---|---|---|
| `topoSnapshot` JSON | `TopologyWorld._exportPhysicalSnapshot` | Core physique, éléments/locales/poses/attaches | scénario autoritaire ou transition vers lui |
| `<chemins>` | `TopologyChemins._saveToXml` | chemin Core | dérivé/métadonnée à qualifier |
| `<triangles>` | `_last_drawn` et points O/B/L | projection UI persistée | cache de compatibilité/diagnostic |
| `<groups>` | `viewer.groups` et `tid` | organisation UI redondante | cache ou données métier à séparer |
| `<listbox>` | liste UI disponible | état de présentation/catalogue local | dérivé de scénario/catalogue |
| carte/clock/guides | viewer/scénario | vue et guidage | séparer présentation et référence métier |
| source Excel | chemin externe | catalogue historique | futur contrat catalogue/version à définir |

## 11.2 Séquence de restauration actuelle

```text
XML v4
  ├─ validation racine/version/snapshot
  ├─ restauration Excel éventuelle
  ├─ restauration vue/carte/horloge/listbox
  ├─ restauration _last_drawn et groupes UI
  ├─ import du nouveau TopologyWorld depuis topoSnapshot
  ├─ restauration chemin/guides
  ├─ relink strict item UI ↔ élément Core et groupe UI ↔ DSU
  └─ redraw + pick cache + widgets
```

Le relink constitue une base utile : il vérifie éléments manquants et cohérence du `topoGroupId` canonique. Il ne mesure pas l'égalité entre les coordonnées `pts` chargées et les points produits par poses/locales Core. Il ne qualifie pas non plus les données UI comme cache ou autorité.

## 11.3 Règle de lecture cible à documenter, non appliquée ici

La cible impose que les données autoritaires soient chargées d'abord : catalogue/références, scénario topologique, attaches, orientations et ancre. Une projection UI éventuellement conservée doit être explicitement identifiée comme dérivée, comparée si présente, puis régénérable. Cette règle est une grille d'analyse pour les futurs choix de schéma ; elle ne modifie pas le lecteur XML v4 existant.

## 11.4 Compatibilité et reproductibilité : questions de stabilisation

- Comment identifier durablement un triangle quand le catalogue passera d'un rang Excel à un slot/variant ?
- Quel niveau de version de catalogue doit être associé à un scénario pour pouvoir le rouvrir ?
- Les poses historiques doivent-elles être conservées pour compatibilité, diagnostic ou migration ?
- Comment signaler un XML dont snapshot Core et points UI sont identitaires mais géométriquement divergents ?
- Une reconstruction partielle peut-elle réutiliser des caches XML sans les promouvoir ?

Ces sujets ne sont pas résolus dans le présent document ; ils forment des préconditions de décision avant toute évolution de persistance.

---

# 12. Scénarios, variantes et ancrage

## 12.1 Ce que le code actuel garantit

Chaque `ScenarioAssemblage` possède son `TopologyWorld`, `last_drawn`, `groups`, `view_state`, `map_state`, statut et références de compas/guides (`src/assembleur_core.py:56-112`). Le manuel partage ses listes runtime avec le viewer ; les scénarios automatiques portent leurs structures mais certains états de vue, carte et géométrie auto sont communs au viewer. `_scenario_duplicate` réalise un `deepcopy` des listes/groupes et un `clonePhysicalState` du monde (`src/assembleur_tk.py:4695-4736`).

## 12.2 Ce que la cible introduit sans l'implémenter

La cible ajoute les notions de :

- `TriangleSlot` : identité métier stable ;
- `TriangleVariant` : hypothèse de géométrie locale d'un slot ;
- sélection d'une variante par slot et par scénario ;
- attaches entre features stables, indépendantes des variantes ;
- ancre persistante de scénario ;
- composantes éventuellement multiples avec une politique d'ancrage à décider.

Ces notions expliquent pourquoi la stabilisation actuelle doit éviter d'ancrer durablement les décisions métier dans `last_drawn`, `tid`, IDs Canvas ou indices d'arêtes calculés.

## 12.3 Questions spécifiques de scénario

| Sujet | Fait actuel | Cible / question ouverte |
|---|---|---|
| variantes | non modélisées explicitement | une variante par slot doit être sélectionnée par scénario |
| ancres | transformations UI/auto et carte, pas d'objet ancre unifié | ancre persistante de scénario ; granularité à trancher |
| composantes | groupes DSU possibles, placement UI ponctuel | une règle par composante doit être définie |
| duplication | structures principales clonées | réutilisation de slot/variant à rendre stable |
| auto | états globaux à certains niveaux | séparer scénarios autoritaires et préférences de vue |
| undo/redo | pas de journal de commandes établi | question ouverte, pas une demande immédiate |

---

# 13. Registre de risques de stabilisation

| ID | Intitulé | Nature | Représentations concernées | Déclencheur | Conséquence | Détectabilité actuelle | Lien cible |
|---|---|---|---|---|---|---|---|
| TG-RISK-001 | absence de validation UI/Core | fait observé | UI, Core, groupes, caches | toute opération ou load | divergence silencieuse | faible | validateur transversal |
| TG-RISK-002 | pose synchronisée sans structure | fait observé | `pts`, poses, groupes, attaches | mouvement/rotation/flip | contour Core différent de l'UI | faible | topologie autoritaire / projection |
| TG-RISK-003 | collage multi-couches non atomique | fait observé et risque plausible | UI points, Core, DSU, groupes UI | exception pendant collage | état hybride partiel | faible | commande atomique + publication |
| TG-RISK-004 | suppression UI avant rebuild Core | risque plausible | `last_drawn`, `tid`, Core | exception Core | UI supprimée/Core restant | faible | scénario atomique |
| TG-RISK-005 | dégroupage Core avant reconstruction UI | risque plausible | composantes, groupes UI, poses | mapping UI incomplet | groupes désaccordés | partielle | composantes/reconstruction |
| TG-RISK-006 | XML à double vérité | fait observé | snapshot, points UI, groupes | fichier ancien/externe | réouverture incohérente | identité seulement | persistance autoritaire |
| TG-RISK-007 | `tid` mutable utilisé comme lien durable | fait observé | groupes UI, XML, listbox | suppression/remap | mauvais membre ou groupe | contrôles de bornes | slots stables |
| TG-RISK-008 | attachements appliqués séquentiellement | fait observé | attaches/DSU/coverages | deuxième attache invalide | Core partiellement muté | validation par attache | atomicité domaine |
| TG-RISK-009 | dérivés chemin/frontière périmés | risque plausible | poses, attaches, caches, triplets | mutation sans recalcul | mesures obsolètes | faible | invalidation/reconstruction |
| TG-RISK-010 | conventions `orient` / `mirrored` | risque plausible | locales, pose, UI | flip/load/catalogue CW | miroir double ou inversé | faible | variante vs pose séparées |
| TG-RISK-011 | scénarios auto partagent certains états | fait observé | vue, carte, auto geometry | action auto | effet croisé inattendu | faible | ownership par couche |
| TG-RISK-012 | confusion Canvas / métier | faiblesse architecturale | Canvas, cache, UI | correctif local | correction perdue ou cache incohérent | visible au redraw | Canvas vue |
| TG-RISK-013 | groupe assimilé trop tôt à composante | risque plausible | groupes UI/Core | refactoring conceptuel | perte de sémantique | inconnue | question groupes |
| TG-RISK-014 | ancre implicite dans interaction | risque plausible | drag, poses, auto state | déplacement/changement de variante | reconstruction ambiguë | absente | ancre persistante |
| TG-RISK-015 | reconstruction partielle prématurée | risque cible | caches/poses/composantes | optimisation hâtive | caches divergents | absente | complète avant incrémentale |

TG-RISK-001 reste prioritaire : les risques 002 à 015 sont plus difficiles à reproduire, classer et traiter tant qu'aucun rapport transversal ne rend l'état hybride observable.

---

# 14. Tests existants et besoin de caractérisation

## 14.1 Couverture actuellement observée

| Domaine | Tests actuels | Valeur | Lacune de stabilisation |
|---|---|---|---|
| catalogue Excel/CSV | `test_triangle_excel_generation.py` | validation I/O et schéma | pas de slots/variantes ni pose UI |
| balises | `test_balises_manager.py`, `test_chemins_balise_ref_core.py` | lecture/projection et usage par triplet | pas de synchronisation viewer→world |
| chemin/triplets | `test_triplets_smoke.py`, tests balise | invariants géométriques ciblés | pas de cycle complet contour→UI→Core |
| dictionnaire/décryptage | plusieurs tests moteur | hors cœur du chantier | ne prouve pas la fraîcheur des triplets |
| runtime | `test_assembleur_engine_runtime.py` | contrôle moteur | sans rapport UI/Core |
| collage/attache | aucun test dédié trouvé | — | frontière la plus critique non caractérisée |
| déplacement/rotation/miroir | aucun test UI-Core dédié trouvé | — | pas d'écart pose/points mesuré |
| dégroupage/suppression | aucun test intégration identifié | — | pas de remap ou rollback caractérisé |
| XML hybride | aucun round-trip dédié trouvé | — | pas de fichier contradictoire ni comparaison pose/points |
| scénarios/duplication | aucun test GUI d'isolation identifié | — | manuel/auto et références non caractérisés |

## 14.2 Ce que les tests de caractérisation devront démontrer, sans définir la solution

Les futurs tests devront figer le comportement observé avant changement d'architecture. Ils ne doivent pas présumer que le résultat actuel est la cible définitive.

| Opération | Observation à capturer | Rapport transversal utile |
|---|---|---|
| pose initiale | item UI, élément Core, IDs, pose et points | TG-INV-001 à 004 |
| collage | points avant/après, attaches, DSU, groupes UI | TG-INV-003, 005 à 009 |
| détachement | attaches retirées, composantes, groupes UI, chemin | TG-INV-005 à 010 |
| suppression | `tid` remappés, Core purgé, attaches restantes | TG-INV-001, 006, 007, 010 |
| déplacement/rotation/miroir | écart UI/Core au départ, pendant/après, annulation | TG-INV-003, 004, 008 |
| chargement XML | source Core/UI, relink, écarts géométriques | TG-INV-001, 003, 005 |
| sauvegarde XML | propriétés durable/cache écrites | TG-RISK-006 |
| scénario/duplication | isolation des listes, mondes, caches et vue | TG-INV-011 |
| contour/chemin | invalidation et recalcul selon mutation | TG-INV-009 |

---

# Priorisation de stabilisation

Cette priorisation est une **feuille de route d'analyse et de préparation**, pas un plan d'implémentation. Elle ordonne les préconditions documentaires, les observations et les critères qui permettront ensuite de décider des travaux techniques.

## Phase A — Observation

| Aspect | Contenu |
|---|---|
| Objectif | rendre le modèle hybride visible sans le modifier |
| Périmètre | opérations sensibles, structures UI/Core, XML, scénarios, caches |
| Bénéfices | localisation factuelle des divergences, vocabulaire commun, meilleure reproductibilité des incidents |
| Risques | instrumentation future trop intrusive ou confondue avec une correction |
| Prérequis | cartographie actuelle, scénarios de démonstration, tolérances documentées |

La Phase A traite prioritairement TG-RISK-001 : identifier quand les deux représentations cessent de décrire le même assemblage.

## Phase B — Validation

| Aspect | Contenu |
|---|---|
| Objectif | définir la lecture transversale de l'état UI/Core sans réparation |
| Périmètre | identité, pose/points, groupes, attaches, caches, XML, scénarios |
| Bénéfices | rapport stable, diagnostic partageable, précondition de tests fiables |
| Risques | fausse précision numérique, règles qui présument une cible non implémentée |
| Prérequis | Phase A, classes de tolérances, distinction actuel/cible |

Le validateur doit mesurer le contrat actuel et annoncer les écarts ; il ne doit pas introduire la reconstruction cible ni modifier la source de vérité au runtime.

## Phase C — Tests de caractérisation

| Aspect | Contenu |
|---|---|
| Objectif | fixer les comportements de référence avant migration |
| Périmètre | pose, collage, détachement, suppression, rotation, miroir, XML, scénarios, contour/chemin |
| Bénéfices | détection de régression, base pour comparer futur moteur et existant |
| Risques | transformer un comportement accidentel en exigence ; tests GUI instables |
| Prérequis | observations et rapport du validateur, scénarios minimaux reproductibles |

Les tests doivent nommer le statut de chaque attente : comportement historique à préserver temporairement, invariant durable, ou ambiguïté à décider.

## Phase D — Contrats de synchronisation

| Aspect | Contenu |
|---|---|
| Objectif | formaliser ce qui est autoritaire, dérivé, invalidé et publié pour chaque opération |
| Périmètre | flux de la section 7 et opérations de la section 8 |
| Bénéfices | réduction des synchronisations implicites, préparation de l'asymétrie cible |
| Risques | imposer trop tôt une sémantique aux groupes, chemins ou ancres |
| Prérequis | Phases A à C, décisions explicites sur ambiguïtés observées |

Le produit attendu de cette phase est documentaire : une matrice de contrats par opération, pas leur mise en œuvre.

## Phase E — Préparation du catalogue

| Aspect | Contenu |
|---|---|
| Objectif | préparer l'introduction future de slots, variantes et identifiants stables |
| Périmètre | identité, nomenclature O/B/L, forme locale, versions, références de scénario/XML |
| Bénéfices | séparation identité/géométrie, persistance plus durable, attaches indépendantes des hypothèses |
| Risques | migration prématurée du DataFrame/rang `id`, perte de compatibilité XML |
| Prérequis | contrats d'identité existants caractérisés, stratégie de version/catalogue décidée |

Cette phase ne demande pas de créer les classes cible. Elle prépare les décisions de compatibilité et les contraintes de migration.

## Phase F — Préparation du moteur de reconstruction

| Aspect | Contenu |
|---|---|
| Objectif | rendre spécifiable un calcul complet et déterministe depuis topologie, variantes et ancre |
| Périmètre | graphe, orientations, composantes, ancres, poses, projection, invalidation |
| Bénéfices | permet de réduire l'autorité UI sur preuve plutôt que par refactoring |
| Risques | optimiser une reconstruction partielle avant d'avoir défini une reconstruction complète ; cycles ou ancres multiples non résolus |
| Prérequis | catalogue conceptuel, règles d'attache/orientation, décision composantes/ancrage, tests de caractérisation |

La première question de cette phase n'est pas « comment coder le moteur ? », mais « quelles entrées autoritaires permettent déjà une reconstruction complète et déterministe ? ».

---

# Questions ouvertes

## 16.1 Groupes et composantes

- Une composante connexe dérivée des attaches suffit-elle à représenter tous les groupes actuels ?
- Quel rôle durable portent l'ordre `group_pos`, `edge_in/out`, la bbox et l'identité `ui_gid` ?
- Un groupe métier distinct de la composante existe-t-il dans les usages ?
- Une composante peut-elle posséder sa propre ancre ou une sélection durable ?

## 16.2 XML cible et compatibilité

- Quel sous-ensemble minimal permet de reconstruire un scénario sans coordonnées monde historiques ?
- Quel statut de compatibilité accorder aux points UI, groupes UI et poses individuelles persistés aujourd'hui ?
- Comment associer version de catalogue, slot et variante à un fichier historique ?
- Quelle politique adopter face à un fichier présentant deux projections incompatibles ?

## 16.3 Variantes et catalogue

- Quelles variantes sont autorisées par slot, et comment sont-elles identifiées/versionnées ?
- Toutes les variantes respectent-elles strictement O/B/L et OB/BL/LO ?
- Un slot peut-il être absent ou non résolu dans un scénario ?
- Comment éviter qu'un rang de DataFrame reste l'identité métier durable ?

## 16.4 Reconstruction partielle et multi-composantes

- Les scénarios actuels autorisent-ils effectivement plusieurs composantes déconnectées ?
- Une ancre globale est-elle suffisante, ou faut-il une règle par composante ?
- Quelle partie de l'arbre doit être recalculée après un changement de variante ou une suppression d'attache ?
- Quand une optimisation partielle serait-elle légitime après une reconstruction complète fiable ?

## 16.5 Stratégie d'ancrage

- L'ancre cible est-elle une pose de triangle, un sommet, une arête, un point externe ou plusieurs formes ?
- Quelle relation existe entre ancre métier, balise géographique, carte et état de drag ?
- Comment exprimer le déplacement global sans transformer un aperçu UI en donnée durable implicite ?

## 16.6 Undo/redo

- Quelles opérations futures doivent être journalisées comme commandes métier ?
- Un changement de variante, d'attache, d'ancre ou de groupe métier doit-il être annulable ?
- Quel cache doit être reconstruit à l'annulation ?
- Les snapshots actuels Core peuvent-ils seulement servir au diagnostic, ou aussi à une politique future de rollback ?

---

# 17. Guide d'usage pour les futures sessions GPT et Codex

Avant toute demande qui touche topologie, géométrie, scénario, XML ou canvas :

1. Identifier l'opération dans la section 8 et son flux actuel.
2. Distinguer l'état actuel, la cible et l'écart ; ne pas citer la cible comme si elle était déjà codée.
3. Localiser les représentations touchées dans les sections 5 et 7.
4. Vérifier les invariants applicables de la section 9 et les risques de la section 13.
5. Déterminer si le sujet dépend d'une question ouverte de la section 16.
6. Lire les tests existants et signaler explicitement la lacune d'intégration éventuelle.
7. Pour un futur changement, demander ou produire d'abord la preuve de caractérisation et le rapport du validateur, plutôt qu'une synchronisation additionnelle non justifiée.

Règles de prudence :

- ne jamais traiter `_last_drawn` ou le Canvas comme source d'identité métier future ;
- ne jamais convertir automatiquement une coïncidence géométrique en attache persistante ;
- ne jamais déduire une orientation cible seulement depuis l'affichage courant ;
- ne jamais supprimer les groupes avant d'avoir séparé composante, groupe métier, groupe UI et sélection ;
- ne jamais faire gagner silencieusement une projection XML sur le snapshot topologique ;
- ne jamais optimiser la reconstruction partielle avant que la reconstruction complète soit définie et caractérisée ;
- ne jamais présenter une question ouverte comme un bug ou une décision prise.

---

# 18. Conclusion de stabilisation

## 18.1 Faits établis

- L'architecture actuelle est hybride : les points monde UI, les poses et attaches Core, les groupes UI/DSU et le XML redondant coexistent.
- `TopologyWorld` possède déjà une géométrie locale et des poses par élément, en plus de la topologie.
- Les synchronisations actuelles sont majoritairement UI→Core pour les gestes manuels, avec des retours Core→UI pour dégroupage, contour et chargement.
- Le relink XML vérifie les identités et groupes, mais pas l'équivalence géométrique points UI/poses Core.
- `TopologyWorld.validate_world` valide le Core mais n'est pas un validateur UI/Core.
- Les tests actuels ne couvrent pas le cycle intégré de collage, suppression, dégroupage, XML hybride et changement de scénario.

## 18.2 Position de stabilisation

La cible est connue : topologie/scénario autoritaires, reconstruction géométrique déterministe, `_last_drawn` cache et Canvas vue. L'existant ne doit pas être forcé à respecter cette cible par une réécriture immédiate. La stabilisation doit d'abord mesurer l'écart, caractériser les comportements, formaliser les contrats, puis seulement préparer catalogue et reconstruction.

## 18.3 Décisions à prendre ultérieurement, non prises ici

- sémantique et persistance des groupes ;
- modèle de composantes et d'ancres multiples ;
- identité et cycle de vie des variantes ;
- autorité et versionnement du XML cible ;
- statut durable des chemins et de leurs métadonnées ;
- stratégie de rollback/undo-redo ;
- trajectoire de `TopologyWorld` comme agrégat conservé ou responsabilités éclatées.

## Annexe A — références de code principales

| Sujet | Références |
|---|---|
| création/pose UI-Core | `src/assembleur_tk.py:7247 _place_dragged_triangle`; `src/assembleur_core.py:2990 add_element_as_new_group` |
| sync de pose | `src/assembleur_tk.py:686 _sync_group_elements_pose_to_core`; `src/assembleur_core.py:3048 setElementPose` |
| collage/attaches | `src/assembleur_tk.py:9923 _on_canvas_left_up`; `src/assembleur_edgechoice.py:EdgeChoiceEpts`; `src/assembleur_core.py:3237 apply_attachments` |
| dégroupage | `src/assembleur_tk.py:7934 _ctx_degrouper`, `7997 _applyDegrouperResultToTk`; `src/assembleur_core.py:3734 degrouperAtNode` |
| suppression | `src/assembleur_tk.py:9123 _ctx_delete_group`; `src/assembleur_core.py:3524 removeElementsAndRebuild` |
| scénarios | `src/assembleur_tk.py:4502 _set_active_scenario`, `4695 _scenario_duplicate`; `src/assembleur_core.py:56 ScenarioAssemblage` |
| rendu/cache | `src/assembleur_tk.py:6003 _redraw_from`, `5038 _ensure_pick_cache` |
| frontière/chemin | `src/assembleur_core.py:1979 computeBoundary`, `990 TopologyChemins` |
| XML | `src/assembleur_io.py:389 saveScenarioXml`, `540 loadScenarioXml` |
| validation Core | `src/assembleur_core.py:4210 validate_world` |

## Annexe B — contrôle documentaire de cette révision

Cette révision a relu intégralement l'ancienne Phase 1 et `ARCHITECTURE_CIBLE_TOPOLOGIQUE.md`. Elle remplace le document Phase 1, ne modifie aucun fichier source, ne corrige aucun bug et ne constitue pas un plan d'implémentation. Les phases A à F constituent une priorisation de stabilisation et de préparation documentaire, à valider avant tout travail technique.
