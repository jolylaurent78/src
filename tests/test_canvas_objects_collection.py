import pytest

from src.canvas_objects_collection import CanvasObjectsCollection


def _entry(triangle_id, topology_id=None):
    entry = {"id": triangle_id}
    if topology_id is not None:
        entry["topoElementId"] = topology_id
    return entry


def test_collection_preserves_list_compatibility_and_indexes():
    first = _entry(1, "T03")
    second = _entry(2, "T04")
    collection = CanvasObjectsCollection([first, second])

    assert len(collection) == 2
    assert collection[0] is first
    assert list(collection) == [first, second]
    assert collection.entries[1] is second
    assert collection.get_by_topology_id("T04") is second
    assert collection.get_index_by_topology_id("T03") == 0
    assert collection.get_by_triangle_id(2) is second


def test_collection_writes_and_legacy_alias_keep_indexes_coherent():
    collection = CanvasObjectsCollection()
    alias = collection.entries
    first = _entry(1, "T01")
    second = _entry(2, "T02")

    assert collection.add(first) is first
    alias.append(second)  # compatibilite: ecriture legacy via _last_drawn
    assert collection.get_index_by_topology_id("T02") == 1

    assert collection.remove_at(0) is first
    assert collection.get_by_triangle_id(1) is None

    collection.replace_all([_entry(3, "T03")])
    assert alias == [{"id": 3, "topoElementId": "T03"}]
    assert collection.get_by_topology_id("T03") is alias[0]

    alias[0]["topoElementId"] = "T04"  # mutation historique d'une entree
    assert collection.get_by_topology_id("T03") is None
    assert collection.get_index_by_topology_id("T04") == 0

    collection.remove(alias[0])
    assert len(collection) == 0
    collection.clear()


def test_triangle_index_lookup_tracks_structural_writes_and_reordering():
    first = _entry(1, "T01")
    second = _entry(2, "T02")
    collection = CanvasObjectsCollection([first, second])

    assert collection.get_index_by_triangle_id(1) == 0
    assert collection.get_index_by_triangle_id("2") == 1
    assert collection.get_index_by_triangle_id(99) is None

    collection.add(_entry(3, "T03"))
    assert collection.get_index_by_triangle_id(3) == 2

    collection.remove_at(1)
    assert collection.get_index_by_triangle_id(2) is None
    assert collection.get_index_by_triangle_id(3) == 1

    collection.replace_all([_entry(3, "T03"), _entry(1, "T01")])
    assert collection.get_index_by_triangle_id(3) == 0
    assert collection.get_index_by_triangle_id(1) == 1


def test_multiple_lookups_preserve_order_duplicates_and_strict_contract():
    first = _entry(10, "T10")
    second = _entry(20, "T20")
    collection = CanvasObjectsCollection([first, second])

    assert collection.get_many_by_topology_ids(None) == ()
    assert collection.get_many_by_topology_ids([]) == ()
    assert collection.get_many_by_topology_ids(["T20", "missing", "T10", "T20"]) == (
        second, first, second,
    )
    assert collection.get_indexed_by_topology_ids(["T20", "missing", "T10", "T20"]) == (
        (1, second), (0, first), (1, second),
    )
    assert collection.get_many_by_triangle_ids(None) == ()
    assert collection.get_many_by_triangle_ids([20, 999, 10, 20]) == (
        second, first, second,
    )

    with pytest.raises(KeyError, match="missing"):
        collection.get_many_by_topology_ids(["T10", "missing"], strict=True)
    with pytest.raises(KeyError, match="missing"):
        collection.get_indexed_by_topology_ids(["missing"], strict=True)
    with pytest.raises(KeyError, match="999"):
        collection.get_many_by_triangle_ids([10, 999], strict=True)


def test_iter_indexed_and_id_inventories_preserve_projection_order():
    first = _entry(10, "T10")
    second = {"id": None, "topoElementId": ""}
    third = _entry("10", "T10")
    fourth = {"topoElementId": "  T20  "}
    collection = CanvasObjectsCollection([first, second, third, fourth])

    assert list(CanvasObjectsCollection().iter_indexed()) == []
    assert list(collection.iter_indexed()) == [
        (0, first), (1, second), (2, third), (3, fourth),
    ]
    assert list(collection.iter_indexed(reverse=True)) == [
        (3, fourth), (2, third), (1, second), (0, first),
    ]
    assert collection.triangle_ids() == (10, "10")
    assert collection.topology_ids() == ("T10", "T10", "T20")


def test_collection_rejects_non_dictionary_entries():
    collection = CanvasObjectsCollection()
    with pytest.raises(TypeError, match="dictionnaire"):
        collection.add("not-an-entry")


def test_remove_many_preserves_requested_removal_order_and_old_to_new_mapping():
    entries = [_entry(10, "T10"), _entry(20, "T20"), _entry(30, "T30"), _entry(40, "T40")]
    collection = CanvasObjectsCollection(entries)

    result = collection.remove_many([3, 1, 3])

    assert result.removed_entries == [entries[3], entries[1]]
    assert result.old_to_new == {0: 0, 2: 1}
    assert collection.entries == [entries[0], entries[2]]
    assert collection.get_index_by_topology_id("T30") == 1


def test_dump_emits_entries_and_indexes_without_raising():
    class _Logger:
        def __init__(self):
            self.messages = []

        def debug(self, message, *args):
            self.messages.append(message % args if args else message)

    logger = _Logger()
    collection = CanvasObjectsCollection([_entry(15, "T42"), _entry(7, "T43")])
    collection.dump(logger, "test")

    assert any("[test] CanvasObjectsCollection entries=2" in message for message in logger.messages)
    assert "[000] triangle=15 topo=T42" in logger.messages
    assert "Topology index" in logger.messages
    assert "T42 -> 0" in logger.messages
    assert "Triangle index" in logger.messages
    assert "15 -> 0" in logger.messages
