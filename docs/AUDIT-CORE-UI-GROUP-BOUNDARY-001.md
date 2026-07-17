# AUDIT-CORE-UI-GROUP-BOUNDARY-001 — Frontière des groupes Core / UI

**Statut :** audit uniquement.  
**Date :** 2026-07-17.  
**Périmètre :** tous les accès à `find_group()` et à `TopologyWorld.groups` hors de `src/assembleur_core.py`, dans les sources applicatives et les tests.  
**Hors périmètre :** toute modification du DSU, de `TopologyWorld`, de l'UI, des diagnostics ou des tests.

## Résumé exécutif

Le Core porte légitimement les identités atomiques, les alias DSU, `find_group()` et `world.groups`. Cette analyse ne remet pas en cause ce modèle.

En revanche, la frontière est actuellement incomplète : les couches Tk, simulation et debug reconstruisent parfois elles-mêmes la liste des groupes canoniques vivants. Cela les oblige à connaître les alias et à appeler `find_group()`, ce qui est précisément une fuite de l'implémentation DSU.

Relevé des sources de production hors Core :

| Recherche | Nombre | Observation |
|---|---:|---|
| `find_group()` dans `assembleur_core.py` | 52 | normal : mécanisme interne Core |
| `self.groups` dans `assembleur_core.py` | 30 | normal : registre interne Core |
| `find_group()` hors Core | 17 | 12 Tk, 3 debug, 2 simulation |
| `world.groups` / `topoWorld.groups` hors Core | 19 | 13 Tk, 3 simulation, 2 debug, 1 IO |
| Appels directs de `find_group()` dans les tests | 4 | caractérisation volontaire du DSU |
| Accès directs à `world.groups` dans les tests | 8 | caractérisation / fixtures ; pas du code applicatif |

Les 36 occurrences de production hors Core ne correspondent pas toutes à une violation de même niveau :

- 10 sont des lectures de groupe après une canonicalisation externe ; elles révèlent l'absence d'un accès Core encapsulé au groupe canonique ;
- 9 appartiennent à de véritables parcours / filtrages / déduplications de groupes vivants ; ce sont les violations les plus importantes ;
- 5 supposent qu'un `core_group_id` porté par l'UI est déjà canonique ; elles doivent être contractualisées ou protégées par une API Core ;
- 12 sont des outils de debug ou tests ; leur impact fonctionnel est faible, mais ils violent la frontière cible et propagent la connaissance du DSU.

La réponse à la question finale est donc :

> **Oui, on peut imposer que tout `group_id` manipulé hors Core soit déjà canonique.**
>
> Ce n'est pas encore possible de façon fiable car il manque une API Core pour obtenir un groupe canonique et pour itérer les seuls groupes vivants. Les parcours externes de `world.groups` et les appels externes à `find_group()` constituent les obstacles actuels.

## 1. Contrat architectural audité

### 1.1 Core : responsabilités autorisées

Dans `assembleur_core.py`, les éléments suivants sont internes et légitimes :

```text
group_id atomique / historique
        ↓
_group_parent, _group_members, _group_created_order
        ↓
find_group()
        ↓
world.groups
```

Le Core est le seul endroit autorisé à :

- recevoir un identifiant possiblement atomique ou alias ;
- le résoudre avec `find_group()` ;
- parcourir `world.groups` en connaissant les alias ;
- décider quelles entrées sont des représentants DSU ;
- reconstruire ou dédupliquer les groupes vivants.

### 1.2 Couches externes : contrat cible

Les couches `assembleur_tk.py`, `assembleur_sim.py`, `assembleur_debug.py`, les validateurs et outils doivent manipuler exclusivement :

```text
core_group_id canonique
        ↓
API métier Core
        ↓
TopologyGroup canonique / boundary / éléments / projection
```

Elles ne doivent pas dériver le `core_group_id` en parcourant le registre ni savoir qu'un alias existe.

## 2. APIs Core déjà disponibles

| API | Retour / effet | Canonique ? | Usage externe approprié |
|---|---|---:|---|
| `get_group_of_element(element_id)` | gid du groupe de l'élément | oui, car elle appelle `find_group()` dans le Core | point d'entrée normal depuis un `topoElementId` |
| `getGroupIdFromConceptNode(nodeId)` | gid du groupe du triangle porteur du nœud | oui, via `get_group_of_element()` | chemin, boundary, outils conceptuels |
| `getBoundarySegments(group_id)` | segments de frontière | oui en interne | contour UI ; il n'est pas nécessaire de canoniser avant l'appel |
| `computeBoundary(group_id)` | calcule la frontière | oui en interne | préparation du contexte de chemin / rendu |
| `ensureConceptGraph(group_id)` / `ensureConceptGeom(group_id)` | cache conceptuel du groupe | oui en interne | diagnostics graphiques ciblés |
| `simulateOverlapTopologique(groupA, groupB, ...)` | test d'overlap | reçoit des objets `TopologyGroup` | simulation, une fois l'obtention de groupe encapsulée |
| `degrouperAtNode(groupId, nodeId)` | résultat de dégroupage | oui en interne | UI, avec un `core_group_id` déjà fourni |

Les APIs manquantes ne concernent donc pas la canonicalisation d'un élément ou d'un nœud : elles existent déjà. Elles concernent l'accès à un objet groupe et l'énumération des représentants vivants.

## 3. Occurrences de `find_group()` hors Core

### A. Tableau exhaustif — sources de production

| Fichier | Fonction | Ligne | Usage | Justifié ? | Remplacement possible |
|---|---|---:|---|---|---|
| `assembleur_tk.py` | `_get_active_topology_group_for_ui_gid` | 789 | Canonise le `topoGroupId` lu dans un groupe UI avant de lire l'objet Core | partiellement ; adaptateur de compatibilité UI | API Core de lecture d'un groupe à partir d'un id ; ou garantir que le lien UI porte déjà un gid canonique |
| `assembleur_tk.py` | `_get_active_topology_group_for_entry` | 803 | Recanonise le résultat de `get_group_of_element()` | non ; l'API Core retourne déjà un gid canonique | utiliser directement le gid retourné, puis une API de groupe canonique |
| `assembleur_tk.py` | `_get_core_group_id_for_triangle_index` | 821 | Recanonise le résultat de `get_group_of_element()` | non | retourner directement `get_group_of_element()` |
| `assembleur_tk.py` | `_get_projected_elements_for_core_group` | 833 | Canonise un paramètre nommé `core_group_id` avant de lire le groupe | non selon le contrat cible | API Core de lecture par gid canonique ; contrat d'entrée canonique |
| `assembleur_tk.py` | `get_last_drawn_entries_for_core_group` | 872 | Même pattern pour naviguer groupe → projection | non selon le contrat cible | même API de lecture ; conserver seulement la projection UI |
| `assembleur_tk.py` | `_sync_group_elements_pose_to_core` | 983 | Même pattern lors de synchronisation des poses | non selon le contrat cible | API de groupe canonique / contrat canonique |
| `assembleur_tk.py` | `_autoSyncAllTopoPoses` | 1063 | Filtre les entrées `world.groups` pour ne garder que les représentants | non ; reconstruction externe des groupes vivants | itérateur Core des groupes vivants |
| `assembleur_tk.py` | `_collect_mig_geo001_audit` | 1174 | Détecte les représentants pendant l'audit F8 | non ; même si le but est diagnostique | itérateur Core de groupes vivants, enrichi des diagnostics nécessaires |
| `assembleur_tk.py` | `_draw_group_outlines` | 6560 | Canonise chaque clé brute pour construire un set de groupes vivants | non ; violation directe du principe cible | itérateur Core de groupes vivants |
| `assembleur_tk.py` | `_group_outline_segments_topo` | 7259 | Recanonise un paramètre déjà nommé `core_group_id` | non | `getBoundarySegments(core_group_id)` suffit : le Core canonise déjà son entrée |
| `assembleur_tk.py` | `_ctx_capture_chemin_context` | 8214 | Recanonise le résultat de `get_group_of_element()` | non ; redondant depuis MIG-GEO-011-A | utiliser directement le résultat Core |
| `assembleur_tk.py` | `_applyDegrouperResultToTk` | 8352 | Compare un lien UI possiblement ancien au `mainCoreGid` | partiellement ; adaptation UI legacy | rendre le résultat Core / la table UI explicitement canonique avant cette phase |
| `assembleur_sim.py` | `AssembleurEngine.run` | 1018 | Canonise `gidDest` avant d'obtenir l'objet groupe | partiellement ; `get_group_of_element()` est déjà canonique | API de lecture du groupe canonique |
| `assembleur_sim.py` | `AssembleurEngine.run` | 1019 | même chose pour le groupe mobile | partiellement | même API |
| `assembleur_debug.py` | `listTopoGroups` | 35 | Canonise chaque clé pour l'afficher | non au regard de la frontière, faible impact car debug | itérateur Core de groupes vivants et données de diagnostic associées |
| `assembleur_debug.py` | `plotTopoWorldBoundaries` | 128 | Canonise la liste par défaut issue de `groups.keys()` | non ; debug seulement | itérateur / ids canoniques Core |
| `assembleur_debug.py` | `plotConceptGraph` | 191 | Canonise le groupe reçu avant un appel Core | non | les APIs `ensureConceptGraph()` acceptent déjà et canonisent leur entrée ; contrat canonique externe |

### B. Tests

Les quatre appels directs dans `tests/test_audit_group_cleanup.py` caractérisent explicitement les aliases DSU. Ils sont justifiés par l'objet même de l'audit de nettoyage. Les stubs `find_group()` de `test_mig_geo001_group_linking.py` et `test_topology_comparison.py` ne sont pas des appels applicatifs : ils modélisent le contrat de leurs fixtures.

Ils ne bloquent pas la frontière de production, mais doivent rester limités aux tests qui vérifient le Core.

## 4. Occurrences de `world.groups` hors Core

### A. Tableau exhaustif — sources de production

| Fichier | Fonction | Ligne | Usage | Justifié ? | Remplacement possible |
|---|---|---:|---|---|---|
| `assembleur_tk.py` | `_get_active_topology_group_for_ui_gid` | 789 | lit l'objet après résolution d'un lien UI | partiellement | API Core `get_live_group(core_group_id)` ; adaptateur UI séparé |
| `assembleur_tk.py` | `_get_active_topology_group_for_entry` | 803 | lit l'objet du groupe de l'élément projeté | non comme accès direct | API Core de lecture de groupe |
| `assembleur_tk.py` | `_get_projected_elements_for_core_group` | 833 | lit l'objet pour `get_projected_elements()` | non comme accès direct | API Core de lecture de groupe, puis helper de projection existant |
| `assembleur_tk.py` | `get_last_drawn_entries_for_core_group` | 873 | lit les `element_ids` pour résoudre `_last_drawn` | non comme accès direct | API Core groupe canonique ou API éléments d'un groupe |
| `assembleur_tk.py` | `_sync_group_elements_pose_to_core` | 984 | lit le groupe pour synchroniser les poses | non comme accès direct | API Core groupe canonique |
| `assembleur_tk.py` | `_autoSyncAllTopoPoses` | 1062 | parcours de toutes les clés, puis filtrage DSU | non | itérateur Core de groupes vivants |
| `assembleur_tk.py` | `_collect_mig_geo001_audit` | 1172 | audit de toutes les composantes Core | non ; diagnostic ne doit pas connaître les alias | itérateur Core de groupes vivants / données de validation |
| `assembleur_tk.py` | `_draw_group_outlines` | 6559, 6561 | parcours + validation de la présence du représentant + déduplication externe | non, criticité forte | itérateur Core de groupes vivants |
| `assembleur_tk.py` | `_group_outline_segments_topo` | 7260 | vérifie que le représentant est dans le registre | non | laisser `getBoundarySegments()` gérer le contrat ou API de groupe |
| `assembleur_tk.py` | `_update_edge_highlights` | 7545, 7546 | récupère deux groupes depuis `topoGroupId` UI | partiellement ; les ids UI sont censés être canoniques | API Core de groupe ; résolution depuis `topoElementId` à terme |
| `assembleur_tk.py` | `_applyDegrouperResultToTk` | 8378 | fallback pour retrouver les éléments du groupe principal | partiellement ; `mainCoreGid` provient du résultat Core | API Core de groupe / API éléments de groupe |
| `assembleur_sim.py` | `AssembleurEngine.run` | 982 | itère tous les objets pour préparer les caches conceptuels | non ; traite aussi les alias | itérateur Core de groupes vivants |
| `assembleur_sim.py` | `AssembleurEngine.run` | 1018, 1019 | indexe les deux groupes participants | non comme accès direct | API Core de groupe canonique |
| `assembleur_io.py` | `loadScenarioXml` | 936 | vérifie que le gid canonique existe dans le registre | partiellement ; validation de chargement | API Core `has_live_group` ou groupe obtenu depuis l'élément |
| `assembleur_debug.py` | `listTopoGroups` | 34, 37 | parcourt toutes les clés et relit le représentant | non ; debug, faible impact | itérateur Core de groupes vivants |

### B. Tests

Huit accès directs apparaissent dans les tests :

- six dans `test_audit_group_cleanup.py`, qui vérifie expressément la coexistence entre alias DSU et objets `TopologyGroup` ;
- un dans `test_mig_geo001_group_linking.py`, qui obtient un groupe de fixture ;
- un snapshot / comparaison dans `test_topology_comparison.py`.

Ils sont légitimes comme tests de structure interne. Ils ne doivent pas servir de précédent à l'usage applicatif.

## 5. Violations de frontière et criticité

| Criticité | Emplacements | Violation | Pourquoi |
|---|---|---|---|
| **Forte** | `_draw_group_outlines` lignes 6558–6564 | UI reconstruit les groupes vivants, canonise les aliases et les déduplique | le pattern interdit est présent littéralement ; le rendu dépend de la forme interne du DSU |
| **Forte** | `_autoSyncAllTopoPoses` lignes 1062–1064 | UI filtre les représentants dans `world.groups` | synchronisation globale importante ; risque de traiter les aliases ou de changer de comportement après évolution Core |
| **Forte** | `_collect_mig_geo001_audit` lignes 1171–1179 | validateur F8 reconstitue les canoniques | le validateur transversal doit vérifier un contrat, non réimplémenter le modèle qu'il mesure |
| **Moyenne** | `AssembleurEngine.run` ligne 982 | simulation itère les groupes internes pour construire les caches | peut calculer plusieurs fois les caches d'une même composante ; fuite de DSU dans la simulation |
| **Moyenne** | `listTopoGroups`, `plotTopoWorldBoundaries` | outils debug parcourent / canonisent | faible impact fonctionnel, mais exemples faciles à recopier et diagnostics incohérents avec la cible |
| **Moyenne** | helpers Tk de lecture (`_get_*`, projection, pose, outline) | chaque helper appelle le DSU pour valider son entrée | le paramètre prétend déjà être `core_group_id` ; le contrat n'est pas tenu |
| **Moyenne** | `_update_edge_highlights`, `_applyDegrouperResultToTk` | dépendance à `topoGroupId` UI puis accès direct au registre | le lien UI peut être obsolète ou non canonique ; la couche UI fait l'adaptation elle-même |
| **Faible** | `_ctx_capture_chemin_context`, `_get_core_group_id_for_triangle_index`, `plotConceptGraph` | canonicalisation redondante après une API Core déjà canonique | pas de divergence probable, mais bruit architectural et mauvais précédent |
| **Faible** | `loadScenarioXml` | test d'existence direct après gid déjà canonique | validation correcte dans l'intention ; manque seulement une API de présence / lecture |

## 6. APIs Core manquantes ou insuffisamment exposées

Cette liste ne propose aucune implémentation dans le présent chantier. Elle identifie seulement le contrat que les usages externes tentent de recréer.

| Besoin externe | Symptomatique actuel | API Core envisageable | Consommateurs concernés |
|---|---|---|---|
| Itérer les représentants vivants | boucle `world.groups` + `find_group` + set | `iter_live_groups()` ou `iter_live_group_ids()` | contours, sync global, F8, simulation, debug |
| Obtenir un groupe métier depuis un gid canonique | `world.groups.get(...)` | `get_live_group(core_group_id)` | projection, poses, collision, simulation, IO |
| Vérifier l'existence d'un groupe vivant | `core_gid in world.groups` | `has_live_group(core_group_id)` | IO, outline |
| Lire les éléments d'un groupe vivant | lecture externe de `TopologyGroup.element_ids` | `get_group_element_ids(core_group_id)` si l'encapsulation doit être stricte | projection, dégroupage UI, synchronisation |
| Obtenir le groupe d'un élément | déjà fourni par `get_group_of_element()` | aucune nouvelle API nécessaire | triangles projetés, MOVE/ROTATE/FLIP, chemin |
| Obtenir le groupe d'un nœud | déjà fourni par `getGroupIdFromConceptNode()` | aucune nouvelle API nécessaire | chemin / interaction sur sommet |

### Point important sur les API existantes

`get_group_of_element()` est l'API de référence pour franchir la frontière depuis une entrée UI :

```text
_last_drawn[idx].topoElementId
        ↓
TopologyWorld.get_group_of_element(...)
        ↓
core_group_id canonique
```

Les appels externes qui appliquent immédiatement `find_group()` au résultat de cette méthode sont redondants et doivent être classés comme candidats simples à migration.

## 7. Parcours de groupes vivants à encapsuler

Les cinq parcours suivants ne peuvent pas être remplacés proprement par les APIs actuelles, car aucune API ne retourne la collection des seuls représentants :

| Parcours | Finalité | API nécessaire | Priorité |
|---|---|---|---|
| `_draw_group_outlines` | rendu des contours de toutes les composantes | `iter_live_group_ids()` | critique |
| `_autoSyncAllTopoPoses` | synchronisation de pose de tous les groupes auto | `iter_live_group_ids()` | critique |
| `_collect_mig_geo001_audit` | audit transversal Core/UI | `iter_live_groups()` ou vue de validation Core | critique |
| `AssembleurEngine.run` ligne 982 | préparation des caches conceptuels | `iter_live_group_ids()` | moyenne |
| outils `assembleur_debug.py` | liste / boundaries globales | `iter_live_groups()` | faible |

Ces cas sont la justification principale d'une future API Core. Tant qu'elle n'existe pas, les couches externes ne peuvent respecter pleinement la règle sans perdre la capacité de parcourir l'ensemble des composantes.

## 8. Priorisation MIG-GEO proposée

### MIG-GEO-016 — API de lecture des groupes canoniques

| Élément | Évaluation |
|---|---|
| Objectif | Définir, dans le Core, la lecture d'un groupe canonique et l'itération des groupes vivants, sans exposer la mécanique DSU aux appelants. |
| Complexité | moyenne |
| Risque | moyen : contrat à valider avec clone, import, dégroupage et groupes vides. |
| Bénéfice | débloque toutes les migrations externes sans modifier le DSU. |

### MIG-GEO-017 — Migration des parcours externes de groupes vivants

| Élément | Évaluation |
|---|---|
| Objectif | Faire migrer contours, auto-sync, F8, simulation de pré-calcul et debug vers l'itération Core. |
| Complexité | moyenne |
| Risque | moyen : ces flux itèrent plusieurs groupes et doivent conserver leur ordre / couverture. |
| Bénéfice | élimine les violations fortes et les déduplications externes. |

### MIG-GEO-018 — Migration des lecteurs unitaires de `TopologyGroup`

| Élément | Évaluation |
|---|---|
| Objectif | Remplacer les accès directs `world.groups.get(core_group_id)` des helpers de projection, pose, collision, simulation et IO. |
| Complexité | faible à moyenne |
| Risque | faible si les paramètres sont déjà canoniques ; moyen pour les liens issus des groupes UI. |
| Bénéfice | rend le contrat `core_group_id` explicite et réduit fortement le couplage UI/Core. |

### MIG-GEO-019 — Assainissement des adaptateurs UI `topoGroupId`

| Élément | Évaluation |
|---|---|
| Objectif | Caractériser puis garantir que les liens UI résiduels vers le Core stockent exclusivement un id canonique, sans que Tk le recanonise. |
| Complexité | moyenne |
| Risque | moyen à fort : collage manuel, dégroupage et XML restent dans la zone legacy. |
| Bénéfice | permet d'imposer strictement que tout id hors Core soit canonique. |

### MIG-GEO-020 — Durcissement des validateurs et diagnostics

| Élément | Évaluation |
|---|---|
| Objectif | Faire du validateur F8 un consommateur de vues Core, et non un duplicateur de `find_group()`. |
| Complexité | faible |
| Risque | faible ; outil non métier mais essentiel à la confiance des migrations. |
| Bénéfice | le validateur mesure la frontière au lieu de la contourner. |

## 9. Réponses explicites aux questions de conclusion

### Peut-on imposer que tout `group_id` manipulé hors Core soit déjà canonique ?

**Oui, comme règle cible.** Les opérations déjà migrées montrent le chemin : une entrée UI est résolue par `topoElementId`, puis `get_group_of_element()` donne un `core_group_id` canonique. Les APIs de boundary acceptent déjà cet identifiant sans nécessiter que l'UI appelle le DSU.

### Quels sont les points qui empêchent encore cette règle aujourd'hui ?

1. Il n'existe pas d'API Core d'énumération des groupes vivants ; l'UI, la simulation, F8 et le debug reconstruisent donc les représentants eux-mêmes.
2. Il n'existe pas d'API Core de lecture d'un `TopologyGroup` canonique ; les appelants indexent directement `world.groups`.
3. Certains liens `topoGroupId` subsistent dans la structure UI legacy. Ils sont normalement mis à jour vers un représentant, mais ce contrat n'est pas universellement garanti ni isolé derrière un adaptateur unique.
4. Des helpers externes portant déjà le nom `core_group_id` continuent de recanoniser leurs paramètres. Cela masque l'absence de contrat et rend les frontières impossibles à vérifier.
5. Les outils debug et le validateur F8 utilisent encore `world.groups` comme source d'énumération. Ils doivent être migrés après l'API d'itération, sans toucher aux mécanismes internes du Core.

## 10. Validation de l'audit

Méthode : recherches statiques ciblées dans `src/` et `tests/`, suivies d'une lecture des call-sites et des APIs Core pertinentes.

Aucun fichier de production n'a été modifié. Aucun test n'a été ajouté ou changé : ce livrable est exclusivement documentaire.
