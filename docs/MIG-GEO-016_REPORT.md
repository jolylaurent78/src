# MIG-GEO-016 — Encapsulation des groupes Core vivants

## Objet

Cette migration applique le contrat strict Core/UI : hors de
`assembleur_core.py`, le code de production ne canonise plus de groupe et
n'accède plus au registre interne `TopologyWorld.groups`.

Le DSU, les aliases, `find_group()` et le stockage interne de `groups` ne sont
pas modifiés. Ils restent entièrement dans le Core.

## APIs Core ajoutées

| API | Contrat |
|---|---|
| `getLiveGroupIds() -> list[str]` | retourne les IDs canoniques, uniques et triés des groupes enregistrés ; les groupes canoniques vides restent volontairement inclus |
| `hasLiveGroup(core_group_id) -> bool` | vérifie, dans le Core, qu'un ID canonique désigne un groupe enregistré |
| `getGroupElementIds(core_group_id) -> list[str]` | retourne une copie des `element_ids` d'un groupe canonique ; retourne `[]` si le groupe n'est pas exploitable |

`simulateOverlapTopologique()` accepte maintenant également des IDs de groupes
canoniques. La résolution éventuelle demeure interne au Core ; aucun objet
`TopologyGroup` n'est exposé à l'UI ou à la simulation.

## Parcours et lecteurs migrés

| Fichier | Fonction | Migration |
|---|---|---|
| `assembleur_tk.py` | `_draw_group_outlines()` | `getLiveGroupIds()` remplace la reconstruction locale des représentants |
| `assembleur_tk.py` | `_autoSyncAllTopoPoses()` | `getLiveGroupIds()` remplace le filtre `find_group(id) == id` |
| `assembleur_tk.py` | `_collect_mig_geo001_audit()` | IDs vivants et membres lus par APIs Core |
| `assembleur_tk.py` | helpers de projection et de pose | `getGroupElementIds()` remplace la lecture de `TopologyGroup.element_ids` |
| `assembleur_tk.py` | `_group_outline_segments_topo()` | appelle directement `getBoundarySegments(core_group_id)` |
| `assembleur_tk.py` | contexte de chemin / dégroupage / snap | suppression des canonicalisations externes et lectures directes du registre |
| `assembleur_sim.py` | pré-calcul de caches, collision | IDs vivants et IDs canoniques passés au Core |
| `assembleur_debug.py` | liste, boundaries, graphe conceptuel | APIs publiques Core ; aucune déduplication DSU locale |
| `assembleur_io.py` | validation de chargement | `hasLiveGroup()` remplace le test de présence dans le registre |

## Patterns supprimés

Les patterns suivants n'existent plus dans le code de production hors Core :

```python
world.find_group(...)
topoWorld.find_group(...)
world.groups[...]
world.groups.get(...)
world.groups.keys()
world.groups.values()
```

## Reliquats `groups` hors Core

La recherche élargie de `.groups[...]`, `.groups.get(...)`, `.groups.keys()` et
`.groups.values()` ne retourne plus aucun accès à `TopologyWorld.groups`.

Les reliquats concernent exclusivement la structure UI historique
`TriangleViewerManual.groups` / `ScenarioAssemblage.groups` : collage manuel,
dégroupage UI, XML UI historique, sélection et projection de chaîne. Ils ne
portent pas une référence `world` / `topoWorld`, ne parcourent pas le registre
Core et restent explicitement hors périmètre de MIG-GEO-016.

| Zone | Nature | Statut |
|---|---|---|
| `assembleur_tk.py` | `self.groups` UI legacy | hors périmètre : collage, dégroupage, snap, UI |
| `assembleur_io.py` | `viewer.groups` UI historique | hors périmètre : XML historique |

## Tests ajoutés ou adaptés

- Contrat `getLiveGroupIds()` : monde vide, singleton, groupes indépendants,
  fusion, chaîne d'aliases, alias physique présent et groupe canonique vide.
- `hasLiveGroup()` et `getGroupElementIds()` : groupe réel et groupe absent.
- `_get_core_group_id_for_triangle_index()` avec un faux monde ne possédant ni
  `find_group` ni `groups`.
- `_get_projected_elements_for_core_group()` avec un faux monde n'exposant que
  `getGroupElementIds()`.
- `_group_outline_segments_topo()` avec un faux monde n'exposant que
  `getBoundarySegments()`.
- Rendu des contours et audit F8 : confirmation de l'utilisation de
  `getLiveGroupIds()`.

## Validation

| Vérification | Résultat |
|---|---|
| `py_compile` des fichiers modifiés | OK ; avertissement préexistant d'échappement dans `assembleur_debug.py` |
| Tests Core/MIG ciblés | 37 passed |
| Recherche statique `find_group(` hors Core | 0 occurrence de production |
| Recherche statique `world.groups` / `topoWorld.groups` hors Core | 0 occurrence de production |
| Suite complète | 132 passed, 6 échecs préexistants hors MIG-GEO |
| `git diff --check` | OK |

Les six échecs de la suite complète concernent le cadence runtime et les tests
de décryptage (`angle180` / comptages de solutions). Ils sont identiques aux
échecs préexistants observés avant cette correction et n'impliquent ni
`TopologyWorld` ni MIG-GEO.

## Sujets reportés

- suppression des structures UI legacy `self.groups` / `groups[*]["nodes"]` ;
- collage manuel, dégroupage UI et XML historique ;
- règles Core de nettoyage des groupes vides et objets alias ;
- toute API qui exposerait directement `TopologyGroup`.
