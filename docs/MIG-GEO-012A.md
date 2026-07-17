# MIG-GEO-012A — Bbox pilotée par Core Group ID

## A. Fonction Core créée

`_recompute_coregroup_bbox(core_gid, scen=None)` calcule et retourne la bbox
d'un groupe Core. Elle résout le groupe canonique dans `TopologyWorld`, obtient
ses `element_ids`, construit/réutilise l'index de projection et agrège les
points O/B/L des entrées `_last_drawn` correspondantes. Elle ne lit ni
`self.groups`, ni `nodes`, ni un identifiant UI. Elle retourne `None` si le
groupe ou sa projection est absent.

`_recompute_group_bbox(gid)` est conservée comme wrapper legacy : elle lit
uniquement `topoGroupId` du cache UI, appelle l'API Core, puis met à jour le
cache `bbox` du groupe UI si une bbox a été calculée.

## B. Appelants migrés

| Appelant | Catégorie | Ancien flux | Nouveau flux |
| --- | --- | --- | --- |
| Reconstruction de géométrie de scénario actif | scénario | `self.groups.keys() -> gid` | `scen.topoWorld.groups -> core_gid` |
| Activation/réconciliation d'un scénario automatique | scénario | `scen.groups.keys() -> gid` | `scen.topoWorld.groups -> core_gid` |
| Création d'un singleton Core | création de groupe | `core_gid -> gid UI -> wrapper` | retour de `add_element_as_new_group() -> _recompute_coregroup_bbox(core_gid)` |
| Orientation manuelle de groupe | triangle/élément | `gid UI -> wrapper` | `rotate_members[core_group_id] -> API Core` |

## C. Appelants non migrés

Les autres appels à `_recompute_group_bbox(gid)` sont conservés pour éviter de
modifier les catégories B : suppression/remappage de chaînes UI, projection du
résultat de dégrouper, collage manuel, fallback MOVE et maintenance de cache.
Ces flux reçoivent souvent seulement un `gid` de projection ou manipulent
directement `nodes`; obtenir un `core_gid` exigerait de modifier leur contrat
legacy ou de réintroduire une résolution UI vers Core. Le wrapper est donc le
choix sûr jusqu'aux chantiers dédiés dégrouper/collage/projection.

## D. État final

* Appelants initiaux de `_recompute_group_bbox(gid)` : 15.
* Appelants migrés vers l'API Core : 4.
* Appelants conservés via wrapper legacy : 11.
* Usages de la nouvelle API Core : 5, dont le wrapper.

Validation exécutée : `python -m compileall -q src` et les tests ciblés
MIG-GEO/topology (`23 passed`). Prochaine étape : migrer séparément les
call-sites de sélection/MOVE/ROTATE/FLIP qui possèdent déjà un
`core_group_id`, sans toucher aux flux de chaîne UI.
