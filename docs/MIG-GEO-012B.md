# MIG-GEO-012B — Bbox Core dans MOVE et ROTATE

## Migrations réalisées

| Fonction / cas | Ancien appel | Nouveau appel | Origine `core_gid` | Cache UI bbox |
| --- | --- | --- | --- | --- |
| `_on_ctrl_down()` / `move_group` | `_recompute_group_bbox(gid)` | `_recompute_coregroup_bbox(core_gid)` | `self._sel["core_group_id"]` | non requis immédiatement ; le bloc redessine et met à jour l'assistance |
| `_on_canvas_motion_update_drag()` / `rotate_group` | `_recompute_group_bbox(gid)` | `_recompute_coregroup_bbox(core_gid)` | `sel["core_group_id"]` | non requis immédiatement ; synchronisation Core puis redraw |
| `_on_escape_key()` / rollback `rotate_group` | `_recompute_group_bbox(gid)` | `_recompute_coregroup_bbox(core_gid)` | `self._sel["core_group_id"]`, avant remise à `None` | non requis avant redraw |
| `_on_escape_key()` / rollback `move_group` | `_recompute_group_bbox(gid)` | `_recompute_coregroup_bbox(core_gid)` | `self._sel["core_group_id"]`, avant remise à `None` | non requis avant redraw |

Dans les quatre blocs, `gid` est conservé seulement pour les utilisations UI
existantes (aide de snap, état de sélection ou conditions legacy). Il ne pilote
plus le recalcul géométrique de bbox.

## État des appels

Après MIG-GEO-012A, 11 appelants utilisaient encore le wrapper
`_recompute_group_bbox(gid)`. MIG-GEO-012B en migre 4. Il reste **7 appels** :

* `_apply_group_meta_after_split_()` ;
* les rollbacks legacy de sélection ;
* `_degrouperTranslateGroupByScreen()` ;
* `_applyDegrouperResultToTk()` ;
* `_ctx_delete_group()` ;
* `_move_group_world()` ;
* le collage manuel dans `_on_canvas_left_up()`.

Ils sont conservés car ils manipulent directement une projection UI, des
chaînes `nodes`, ou un `gid` sans `core_group_id` déjà disponible. Leur
migration relève des chantiers dégrouper, fallback MOVE et collage.

## Validation

Exécuté :

```text
python -m compileall -q src
python -m pytest tests/test_mig_geo001_group_linking.py tests/test_topology_comparison.py tests/test_chemins_balise_ref_core.py -q
```

Résultat : `26 passed`.

Les scénarios GUI manuels MOVE/CTRL/ROTATE/ESC restent à exécuter dans
l'application ; cette migration ne modifie ni les coordonnées, ni les
rollbacks, ni la logique AUTO.
