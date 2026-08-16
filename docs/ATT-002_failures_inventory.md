# Inventaire pytest en échec — ATT-002B

Exécution observée : `360 passed, 38 failed, 1 warning`.

Les 31 échecs de migration ont la même cause terminale :
`TopologyWorld.apply_attachment()` refuse explicitement l'ancien
`TopologyAttachment`, conformément au contrat V2. Les sept autres échecs ne
passent pas par ce contrat et sont classés préexistants.

| # | Fichier de test | Test | Exception | Première frame pertinente dans `src/` | Fonction Core appelée | Cause immédiate | Classification |
|---:|---|---|---|---|---|---|---|
| 1 | `tests/test_assembleur_engine_runtime.py` | `test_checkpointpolicy_progress_cadence` | `AssertionError: False is not True` | `assembleur_engine_runtime.py:138`, `CheckpointPolicy.onCheckpoint` | Aucune | La politique n'émet pas la progression attendue. | AUTRE / PREEXISTANT |
| 2 | `tests/test_audit_group_cleanup.py` | `test_get_live_group_ids_hides_fusion_alias_still_present_in_groups` | `TopologyAttachmentValidationError` | `assembleur_core.py:4796`, `TopologyWorld.apply_attachment` | `apply_attachment` | La fixture crée un `TopologyAttachment` legacy. | DEGROUPAGE/SUPPRESSION LEGACY |
| 3 | `tests/test_audit_group_cleanup.py` | `test_rebuild_keeps_fusion_alias_object_then_empties_it` | `TopologyAttachmentValidationError` | `assembleur_core.py:4796`, `TopologyWorld.apply_attachment` | `apply_attachment` | La fixture crée un `TopologyAttachment` legacy. | DEGROUPAGE/SUPPRESSION LEGACY |
| 4 | `tests/test_audit_group_cleanup.py` | `test_aggressive_alias_object_delete_is_readable_but_not_clone_stable` | `TopologyAttachmentValidationError` | `assembleur_core.py:4796`, `TopologyWorld.apply_attachment` | `apply_attachment` | La fixture crée un `TopologyAttachment` legacy. | DEGROUPAGE/SUPPRESSION LEGACY |
| 5 | `tests/test_deformation_simulation.py` | `test_deformation_is_pure_deterministic_and_preserves_attachments_and_anchor` | `TopologyAttachmentValidationError` | `assembleur_core.py:4796`, `TopologyWorld.apply_attachment` | `apply_attachment` | Fixture DEFORM en modèle legacy. | DEFORM LEGACY |
| 6 | `tests/test_deformation_simulation.py` | `test_deformation_rejects_a_degenerate_candidate_without_mutating_source` | `TopologyAttachmentValidationError` | `assembleur_core.py:4796`, `TopologyWorld.apply_attachment` | `apply_attachment` | Fixture DEFORM en modèle legacy. | DEFORM LEGACY |
| 7 | `tests/test_deformation_simulation.py` | `test_deformation_keeps_attachments_frozen_with_base_and_light_overrides` | `TopologyAttachmentValidationError` | `assembleur_core.py:4796`, `TopologyWorld.apply_attachment` | `apply_attachment` | Fixture DEFORM en modèle legacy. | DEFORM LEGACY |
| 8 | `tests/test_deformation_simulation.py` | `test_deformation_rematerializes_atomic_vertex_edge_with_current_edge_ratio[44.0-1.01-source]` | `TopologyAttachmentValidationError` | `assembleur_core.py:4531`, `TopologyWorld.apply_attachments` | `apply_attachments → apply_attachment` | `materialize_vertex_edge_attachments()` renvoie du legacy. | DEFORM LEGACY |
| 9 | `tests/test_deformation_simulation.py` | `test_deformation_rematerializes_atomic_vertex_edge_with_current_edge_ratio[45.5-1.01-destination]` | `TopologyAttachmentValidationError` | `assembleur_core.py:4531`, `TopologyWorld.apply_attachments` | `apply_attachments → apply_attachment` | `materialize_vertex_edge_attachments()` renvoie du legacy. | DEFORM LEGACY |
| 10 | `tests/test_deformation_simulation.py` | `test_deformation_rematerializes_atomic_vertex_edge_with_current_edge_ratio[44.0-0.99-destination]` | `TopologyAttachmentValidationError` | `assembleur_core.py:4531`, `TopologyWorld.apply_attachments` | `apply_attachments → apply_attachment` | `materialize_vertex_edge_attachments()` renvoie du legacy. | DEFORM LEGACY |
| 11 | `tests/test_deformation_simulation.py` | `test_deformation_ignores_an_overlapping_independent_group` | `TopologyAttachmentValidationError` | `assembleur_core.py:4796`, `TopologyWorld.apply_attachment` | `apply_attachment` | Fixture DEFORM en modèle legacy. | DEFORM LEGACY |
| 12 | `tests/test_deformation_simulation.py` | `test_deformation_requires_one_resolvable_group_anchor` | `TopologyAttachmentValidationError` | `assembleur_core.py:4796`, `TopologyWorld.apply_attachment` | `apply_attachment` | Fixture DEFORM en modèle legacy. | DEFORM LEGACY |
| 13 | `tests/test_deformation_simulation.py` | `test_deformation_replays_frozen_point_attachments[vertex-vertex-params0]` | `TopologyAttachmentValidationError` | `assembleur_core.py:4796`, `TopologyWorld.apply_attachment` | `apply_attachment` | Le test injecte un attachment point legacy. | DEFORM LEGACY |
| 14 | `tests/test_deformation_simulation.py` | `test_deformation_replays_frozen_point_attachments[vertex-edge-params1]` | `TopologyAttachmentValidationError` | `assembleur_core.py:4796`, `TopologyWorld.apply_attachment` | `apply_attachment` | Le test injecte un attachment point legacy. | DEFORM LEGACY |
| 15 | `tests/test_deformation_simulation.py` | `test_deformation_reanchors_a_group_when_anchor_is_remote_from_pilot` | `TopologyAttachmentValidationError` | `assembleur_core.py:4796`, `TopologyWorld.apply_attachment` | `apply_attachment` | Le test construit un attachment legacy. | DEFORM LEGACY |
| 16 | `tests/test_deformation_simulation.py` | `test_deformation_replays_vertex_vertex_and_vertex_edge_as_one_rigid_link` | `TopologyAttachmentValidationError` | `assembleur_core.py:4796`, `TopologyWorld.apply_attachment` | `apply_attachment` | Le test construit des attachments legacy. | DEFORM LEGACY |
| 17 | `tests/test_deformation_simulation.py` | `test_deformation_propagates_across_pairs_linked_by_an_atomic_point_link` | `TopologyAttachmentValidationError` | `assembleur_core.py:4796`, `TopologyWorld.apply_attachment` | `apply_attachment` | Le test construit des attachments legacy. | DEFORM LEGACY |
| 18 | `tests/test_deformation_simulation.py` | `test_deformation_keeps_reverse_edge_mapping_frozen` | `TopologyAttachmentValidationError` | `assembleur_core.py:4796`, `TopologyWorld.apply_attachment` | `apply_attachment` | Le test dépend de `mapping` legacy. | DEFORM LEGACY |
| 19 | `tests/test_deformation_simulation.py` | `test_deformation_accepts_a_world_produced_by_the_auto_simulator` | `TopologyAttachmentValidationError` | `assembleur_sim.py:791`, `AlgoQuadrisParPaires.run` | `apply_attachments → apply_attachment` | Le bootstrap AUTO produit des attachments legacy. | DEFORM LEGACY |
| 20 | `tests/test_edgechoice_boundary_rejections.py` | `test_boundary_owner_without_physical_anchor_is_rejected_without_exception` | `TopologyAttachmentValidationError` | `assembleur_core.py:4796`, `TopologyWorld.apply_attachment` | `apply_attachment` | Fixture edge-choice en vertex-edge legacy. | EDGECHOICE/MANUEL LEGACY |
| 21 | `tests/test_edgechoice_boundary_rejections.py` | `test_representable_boundary_owner_still_builds_edge_choice` | `TopologyAttachmentValidationError` | `assembleur_core.py:4796`, `TopologyWorld.apply_attachment` | `apply_attachment` | Fixture edge-choice en vertex-edge legacy. | EDGECHOICE/MANUEL LEGACY |
| 22 | `tests/test_engine_angle180_matching.py` | `test_angle180_matching_positive` | `AssertionError: False is not True` | Aucune frame `src/` dans la trace courte | Aucune | Le matching angulaire ne renvoie pas le résultat attendu. | AUTRE / PREEXISTANT |
| 23 | `tests/test_engine_run.py` | `test_engine_abs_mirroring_total_6_solutions` | `AssertionError: 8 == 6` | Aucune frame `src/` dans la trace courte | Aucune | Le nombre de solutions du moteur diffère de l'attendu. | AUTRE / PREEXISTANT |
| 24 | `tests/test_engine_run.py` | `test_engine_abs_extended_total_15_solutions` | `AssertionError: 16 == 15` | Aucune frame `src/` dans la trace courte | Aucune | Le nombre de solutions du moteur diffère de l'attendu. | AUTRE / PREEXISTANT |
| 25 | `tests/test_engine_run.py` | `test_engine_abs_extended_total_joker_solutions` | `AssertionError: 12 == 9` | Aucune frame `src/` dans la trace courte | Aucune | Le nombre de solutions du moteur diffère de l'attendu. | AUTRE / PREEXISTANT |
| 26 | `tests/test_engine_run.py` | `test_engine_rel_extended_total_count_smoke` | `AssertionError: 9 == 6` | Aucune frame `src/` dans la trace courte | Aucune | Le nombre de solutions du moteur diffère de l'attendu. | AUTRE / PREEXISTANT |
| 27 | `tests/test_scenario_auto_catalogue_runtime.py` | `test_auto_simulation_uses_catalogue_ids_and_clones_the_source_hypothesis` | `TopologyAttachmentValidationError` | `assembleur_sim.py:791`, `AlgoQuadrisParPaires.run` | `apply_attachments → apply_attachment` | Le bootstrap AUTO produit des attachments legacy. | AUTO LEGACY |
| 28 | `tests/test_scenario_hypothesis_topology_apply.py` | `test_apply_replaces_materialized_triangle_on_a_clone_and_replays_ob_connection` | `TopologyAttachmentValidationError` | `assembleur_core.py:4531`, `TopologyWorld.apply_attachments` | `apply_attachments → apply_attachment` | Helper de rejeu OB construit un edge-edge legacy. | POSE/REPLAY LEGACY |
| 29 | `tests/test_scenario_hypothesis_topology_apply.py` | `test_apply_replays_a_changed_pair_only_once` | `TopologyAttachmentValidationError` | `assembleur_core.py:4531`, `TopologyWorld.apply_attachments` | `apply_attachments → apply_attachment` | Helper de rejeu OB construit un edge-edge legacy. | POSE/REPLAY LEGACY |
| 30 | `tests/test_scenario_hypothesis_topology_apply.py` | `test_apply_replays_independent_neighbour_connections_and_preserves_anchor` | `TopologyAttachmentValidationError` | `assembleur_core.py:4531`, `TopologyWorld.apply_attachments` | `apply_attachments → apply_attachment` | Helper de rejeu OB construit un edge-edge legacy. | POSE/REPLAY LEGACY |
| 31 | `tests/test_topology_boundary_services.py` | `test_shared_edge_is_never_returned_by_boundary_services` | `TopologyAttachmentValidationError` | `assembleur_core.py:4796`, `TopologyWorld.apply_attachment` | `apply_attachment` | Fixture Boundary fondée sur un edge-edge legacy. | EDGECHOICE/MANUEL LEGACY |
| 32 | `tests/test_topology_boundary_services.py` | `test_boundary_segment_order_and_physical_metadata_are_stable` | `TopologyAttachmentValidationError` | `assembleur_core.py:4796`, `TopologyWorld.apply_attachment` | `apply_attachment` | Fixture Boundary fondée sur un edge-edge legacy. | EDGECHOICE/MANUEL LEGACY |
| 33 | `tests/test_topology_element_ids.py` | `test_simulation_uses_core_ids_instead_of_catalog_triangle_ids` | `TopologyAttachmentValidationError` | `assembleur_sim.py:469`, `createTopoQuadrilateral` | `apply_attachments → apply_attachment` | La simulation matérialise des attachments legacy. | AUTO LEGACY |
| 34 | `tests/test_topology_element_ids.py` | `test_auto_scenarios_keep_an_independent_ordered_element_history_per_branch` | `TopologyAttachmentValidationError` | `assembleur_sim.py:911`, `AlgoQuadrisParPaires.run` | `apply_attachments → apply_attachment` | Le bootstrap AUTO produit des attachments legacy. | AUTO LEGACY |
| 35 | `tests/test_topology_element_ids.py` | `test_auto_scenarios_reconstruct_connections_from_core_attachments` | `TopologyAttachmentValidationError` | `assembleur_sim.py:911`, `AlgoQuadrisParPaires.run` | `apply_attachments → apply_attachment` | Le bootstrap AUTO produit des attachments legacy. | AUTO LEGACY |
| 36 | `tests/test_topology_element_ids.py` | `test_auto_two_triangle_scenario_exposes_core_projection_dicts` | `TopologyAttachmentValidationError` | `assembleur_sim.py:791`, `AlgoQuadrisParPaires.run` | `apply_attachments → apply_attachment` | Le bootstrap AUTO produit des attachments legacy. | AUTO LEGACY |
| 37 | `tests/test_topology_element_ids.py` | `test_auto_reference_orientation_rotates_the_first_group_without_changing_mirror` | `TopologyAttachmentValidationError` | `assembleur_sim.py:791`, `AlgoQuadrisParPaires.run` | `apply_attachments → apply_attachment` | Le bootstrap AUTO produit des attachments legacy. | AUTO LEGACY |
| 38 | `tests/test_xml_core_first.py` | `test_xml_core_first_rejects_repository_v4_scenario_without_topology_ids` | `FileNotFoundError` | `assembleur_io.py:275`, `loadScenarioXml` | Aucune | Le fixture XML `Scenario_FrontiereActuelle v6.xml` est absent. | AUTRE / PREEXISTANT |

## Regroupement par catégorie

| Catégorie | Nombre | Tests |
|---|---:|---|
| DEFORM LEGACY | 15 | 5–19 |
| AUTO LEGACY | 6 | 27, 33–37 |
| EDGECHOICE/MANUEL LEGACY | 4 | 20–21, 31–32 |
| DEGROUPAGE/SUPPRESSION LEGACY | 3 | 2–4 |
| POSE/REPLAY LEGACY | 3 | 28–30 |
| AUTRE / PREEXISTANT | 7 | 1, 22–26, 38 |
| OVERLAP LEGACY | 0 | — |
| IO/SNAPSHOT LEGACY | 0 | — |
| TOPOLOGY_COMPARISON LEGACY | 0 | — |

La catégorie `AUTRE / PREEXISTANT` inclut le test XML : il est relié au
chargement IO, mais sa cause immédiate est l'absence préexistante du fichier de
fixture, non une incompatibilité d'attachments.

## Warning

Le warning unique est un `SyntaxWarning` préexistant dans
`src/assembleur_debug.py:288` (`invalid escape sequence '\\D'`).
