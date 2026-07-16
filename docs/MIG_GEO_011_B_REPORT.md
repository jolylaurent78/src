# MIG-GEO-011-B — Synchronisation des poses depuis les groupes Core

Date : 17 juillet 2026

## Analyse préalable

La signature initiale était :

```python
_sync_group_elements_pose_to_core(ui_gid, scen=None)
```

La méthode lisait `self.groups[ui_gid]`, puis
`groups[ui_gid]["nodes"]`, exclusivement pour énumérer les `tid` des triangles
à synchroniser. Elle ne consultait ni `edge_in`, ni `edge_out`, ni
`vkey_in/out`, ni l'ordre de chaîne pour le calcul de pose.

Les onze call-sites analysés étaient : synchronisation automatique, suivi de
rotation, placement singleton, résultat de dégrouper, orientation, flip,
déplacement, validation de rotation, collage avant attachement et dépôt final.

## Migration réalisée

La nouvelle signature est :

```python
_sync_group_elements_pose_to_core(core_group_id, scen=None)
```

Le flux est désormais :

```text
core_group_id
  -> TopologyWorld.find_group()
  -> TopologyGroup.element_ids
  -> build_last_drawn_index(scen.last_drawn)
  -> get_projected_elements(...)
  -> setElementPose(...)
```

La méthode ne lit plus `self.groups`, `group_id` UI ni
`groups[*]["nodes"]`. Le calcul Kabsch, le traitement de `mirrored` et
l'appel à `setElementPose(...)` sont inchangés.

Le parcours automatique itère maintenant les groupes DSU canoniques du monde.
Les appels MOVE, ROTATE et FLIP transmettent le `core_group_id` déjà résolu ;
le collage avant attachement le résout depuis l'élément mobile. Le fallback
MOVE conserve son comportement de découverte si l'appelant ne fournit pas
encore l'identifiant Core.

## Validation automatisée

Un test a été ajouté dans `tests/test_mig_geo001_group_linking.py`. Il vide
`viewer.groups` avant la synchronisation et vérifie que l'élément du groupe
Core est néanmoins envoyé à `setElementPose`.

Commandes exécutées :

```text
python -m compileall -q src
python -m pytest tests/test_mig_geo001_group_linking.py tests/test_topology_comparison.py tests/test_chemins_balise_ref_core.py -q
```

Résultat : `26 passed`.

Les validations GUI demandées (déplacement simple/fusionné, XML, chemins,
comparaison) restent manuelles : la suite ne simule pas le canvas. La migration
ne modifie ni géométrie ni topologie, seulement la résolution des membres.

## Occurrences legacy restantes

Après cette migration, il reste **19 usages fonctionnels** de la chaîne legacy
`groups[*]["nodes"]` dans le code de production : 13 dans l'UI Tk, 3 dans le
XML et 3 dans la simulation. Ce nombre regroupe les blocs fonctionnels, pas
les répétitions de boucles, commentaires ou widgets nommés `nodes`.

Ils concernent collage manuel et concaténation ordonnée, création/suppression
de projections UI, résultat visuel de dégrouper, fallback temporaire des
opérations, XML v4 et génération automatique avec `edge_in/out`. Aucun n'a été
modifié par MIG-GEO-011-B.
