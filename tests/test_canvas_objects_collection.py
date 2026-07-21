import pytest

from src.canvas_objects_collection import CanvasObjectsCollection


def _entry(topology_id):
    return {"topoElementId": topology_id}


def test_collection_preserves_list_compatibility_and_topology_index():
    first = _entry("T03")
    second = _entry("T04")
    collection = CanvasObjectsCollection([first, second])

    assert len(collection) == 2
    assert collection[0] is first
    assert list(collection) == [first, second]
    assert collection.entries[1] is second
    assert collection.get_by_topology_id("T04") is second
    assert collection.get_index_by_topology_id("T03") == 0


def test_collection_writes_keep_topology_index_coherent():
    collection = CanvasObjectsCollection()
    alias = collection.entries
    first = _entry("T01")
    second = _entry("T02")

    assert collection.add(first) is first
    alias.append(second)
    assert collection.get_index_by_topology_id("T02") == 1

    assert collection.remove_at(0) is first
    assert collection.get_by_topology_id("T01") is None

    collection.replace_all([_entry("T03")])
    assert alias == [{"topoElementId": "T03"}]
    assert collection.get_by_topology_id("T03") is alias[0]

    alias[0]["topoElementId"] = "T04"
    assert collection.get_by_topology_id("T03") is None
    assert collection.get_index_by_topology_id("T04") == 0

    collection.remove(alias[0])
    assert len(collection) == 0
    collection.clear()


def test_multiple_topology_lookups_preserve_order_duplicates_and_strict_contract():
    first = _entry("T10")
    second = _entry("T20")
    collection = CanvasObjectsCollection([first, second])

    assert collection.get_many_by_topology_ids(None) == ()
    assert collection.get_many_by_topology_ids([]) == ()
    assert collection.get_many_by_topology_ids(["T20", "missing", "T10", "T20"]) == (
        second, first, second,
    )
    assert collection.get_indexed_by_topology_ids(["T20", "missing", "T10", "T20"]) == (
        (1, second), (0, first), (1, second),
    )

    with pytest.raises(KeyError, match="missing"):
        collection.get_many_by_topology_ids(["T10", "missing"], strict=True)
    with pytest.raises(KeyError, match="missing"):
        collection.get_indexed_by_topology_ids(["missing"], strict=True)


def test_iter_indexed_and_topology_inventory_preserve_projection_order():
    first = _entry("T10")
    second = _entry("T20")
    collection = CanvasObjectsCollection([first, second])

    assert list(CanvasObjectsCollection().iter_indexed()) == []
    assert list(collection.iter_indexed()) == [(0, first), (1, second)]
    assert list(collection.iter_indexed(reverse=True)) == [(1, second), (0, first)]
    assert collection.topology_ids() == ("T10", "T20")


def test_collection_rejects_non_dictionary_entries_legacy_ids_and_duplicate_topology_ids():
    collection = CanvasObjectsCollection()
    with pytest.raises(TypeError, match="dictionnaire"):
        collection.add("not-an-entry")
    with pytest.raises(ValueError, match="champs legacy interdits"):
        collection.add({"id": 12, "topoElementId": "T12"})
    with pytest.raises(ValueError, match="obligatoire"):
        collection.add({})
    with pytest.raises(ValueError, match="dupliqué"):
        collection.replace_all([_entry("T01"), _entry("T01")])


def test_remove_many_preserves_requested_removal_order_and_old_to_new_mapping():
    entries = [_entry("T10"), _entry("T20"), _entry("T30"), _entry("T40")]
    collection = CanvasObjectsCollection(entries)

    result = collection.remove_many([3, 1, 3])

    assert result.removed_entries == [entries[3], entries[1]]
    assert result.old_to_new == {0: 0, 2: 1}
    assert collection.entries == [entries[0], entries[2]]
    assert collection.get_index_by_topology_id("T30") == 1


def test_collection_validates_a_minimal_core_projection_contract():
    world = type("World", (), {"elements": {"T01": object()}})()
    collection = CanvasObjectsCollection([{
        "topoElementId": "T01",
        "pts": {"O": (0.0, 0.0), "B": (1.0, 0.0), "L": (0.0, 1.0)},
    }])

    collection.validate_against_world(world)

    collection.entries[0]["pts"]["L"] = (float("nan"), 1.0)
    with pytest.raises(ValueError, match="point L invalide"):
        collection.validate_against_world(world)


def test_dump_emits_entries_and_topology_index_without_raising():
    class _Logger:
        def __init__(self):
            self.messages = []

        def debug(self, message, *args):
            self.messages.append(message % args if args else message)

    logger = _Logger()
    collection = CanvasObjectsCollection([_entry("T42"), _entry("T43")])
    collection.dump(logger, "test")

    assert any("[test] CanvasObjectsCollection entries=2" in message for message in logger.messages)
    assert "[000] topo=T42" in logger.messages
    assert "Topology index" in logger.messages
    assert "T42 -> 0" in logger.messages
