"""Caract\u00e9risation read-only du cycle de vie des groupes DSU.

Ces tests ne proposent pas le nettoyage cible : ils figent le comportement
constat\u00e9 par AUDIT-GROUP-CLEANUP-001, notamment la coexistence actuelle
d'un alias DSU et de son objet ``TopologyGroup`` historique.
"""

from src.assembleur_core import (
    TopologyAttachment,
    TopologyElement,
    TopologyFeatureRef,
    TopologyFeatureType,
    TopologyNodeType,
    TopologyWorld,
)


def _element(element_id: str) -> TopologyElement:
    return TopologyElement(
        element_id=element_id,
        name=element_id,
        vertex_labels=["O", "B", "L"],
        vertex_types=[
            TopologyNodeType.OUVERTURE,
            TopologyNodeType.BASE,
            TopologyNodeType.LUMIERE,
        ],
        edge_lengths_km=[3.0, 5.0, 4.0],
    )


def _fused_world() -> tuple[TopologyWorld, str, str]:
    world = TopologyWorld()
    group_a = world.add_element_as_new_group(_element("T01"))
    group_b = world.add_element_as_new_group(_element("T02"))
    attachment = TopologyAttachment(
        "A001",
        "vertex-vertex",
        TopologyFeatureRef(TopologyFeatureType.VERTEX, "T01", 0),
        TopologyFeatureRef(TopologyFeatureType.VERTEX, "T02", 0),
    )
    assert world.apply_attachment(attachment) == group_b
    return world, group_a, group_b


def test_get_live_group_ids_returns_empty_list_for_empty_world():
    assert TopologyWorld().getLiveGroupIds() == []


def test_get_live_group_ids_returns_one_atomic_group_once():
    world = TopologyWorld()
    group_id = world.add_element_as_new_group(_element("T01"))

    assert world.getLiveGroupIds() == [group_id]
    assert world.hasLiveGroup(group_id) is True
    assert world.getGroupElementIds(group_id) == ["T01"]
    assert world.getGroupElementIds("G-UNKNOWN") == []


def test_get_live_group_ids_is_deterministic_for_independent_groups():
    world = TopologyWorld()
    first = world.add_element_as_new_group(_element("T01"))
    second = world.add_element_as_new_group(_element("T02"))

    assert world.getLiveGroupIds() == [first, second]


def test_get_live_group_ids_hides_fusion_alias_still_present_in_groups():
    world, alias_group_id, canonical_group_id = _fused_world()

    assert alias_group_id in world.groups
    assert world.getLiveGroupIds() == [canonical_group_id]


def test_get_live_group_ids_hides_a_chain_of_aliases():
    world = TopologyWorld()
    first = world.add_element_as_new_group(_element("T01"))
    second = world.add_element_as_new_group(_element("T02"))
    third = world.add_element_as_new_group(_element("T03"))

    intermediate = world.union_groups(first, second)
    root = world.union_groups(intermediate, third)

    assert root == third
    assert world.getLiveGroupIds() == [root]


def test_get_live_group_ids_preserves_registered_empty_canonical_group():
    world = TopologyWorld()
    empty_group_id = world.create_group_atomic()

    # MIG-GEO-016 preserve la semantique des parcours existants : un
    # representant vide encore enregistre reste "live" pour cette API.
    assert world.getLiveGroupIds() == [empty_group_id]


def test_rebuild_keeps_fusion_alias_object_then_empties_it():
    world, alias_group_id, canonical_group_id = _fused_world()

    assert world.find_group(alias_group_id) == canonical_group_id
    assert alias_group_id in world.groups

    world.rebuild_from_attachments(list(world.attachments.values()))
    world.rebuildGroupElementLists()

    assert world.find_group(alias_group_id) == canonical_group_id
    assert world.groups[alias_group_id].element_ids == []
    assert world.groups[canonical_group_id].element_ids == ["T01", "T02"]


def test_aggressive_alias_object_delete_is_readable_but_not_clone_stable():
    world, alias_group_id, canonical_group_id = _fused_world()

    # Simulation demandee par l'audit : le lien DSU est conserve, seul l'objet
    # historique de l'alias est retire du registre des groupes.
    del world.groups[alias_group_id]

    assert world.find_group(alias_group_id) == canonical_group_id
    assert world.get_group_of_element("T01") == canonical_group_id
    world.rebuildGroupElementLists()
    assert world.groups[canonical_group_id].element_ids == ["T01", "T02"]

    # Le snapshot physique conserve les identites atomiques element->group :
    # son import recree donc l'objet de l'ancien identifiant avant de rejouer
    # l'attachment. Une purge directe n'est pas un invariant stable aujourd'hui.
    clone = world.clonePhysicalState()
    assert alias_group_id in clone.groups
    assert clone.find_group(alias_group_id) == canonical_group_id
