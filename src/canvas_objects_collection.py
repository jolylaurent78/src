"""Collection structurelle des objets graphiques projetés sur le canvas.

Ce module ne porte aucune logique métier ou géométrique. Il encapsule seulement
la liste projetée et ses index de navigation.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, Iterable, Iterator, Optional


@dataclass(frozen=True)
class CanvasObjectsRemovalResult:
    """Resultat structurel d'une suppression multiple."""

    removed_entries: list[dict]
    old_to_new: Dict[int, int]


class _ObservableEntries(list):
    """Liste compatible qui notifie sa collection lors des mutations structurelles."""

    def __init__(self, owner: "CanvasObjectsCollection", entries: Iterable[dict] = ()):
        self._owner = owner
        super().__init__(entries)

    def _changed(self) -> None:
        self._owner._rebuild_indexes()

    def _validate_replacement(self, replacement) -> list:
        candidate = list(replacement)
        self._owner._validate_entries(candidate)
        return candidate

    def append(self, entry) -> None:
        self._validate_replacement([*self, entry])
        super().append(entry)
        self._changed()

    def extend(self, entries) -> None:
        added = list(entries)
        self._validate_replacement([*self, *added])
        super().extend(added)
        self._changed()

    def insert(self, index, entry) -> None:
        candidate = list(self)
        candidate.insert(index, entry)
        self._validate_replacement(candidate)
        super().insert(index, entry)
        self._changed()

    def clear(self) -> None:
        super().clear()
        self._changed()

    def pop(self, index: int = -1):
        entry = super().pop(index)
        self._changed()
        return entry

    def remove(self, entry) -> None:
        super().remove(entry)
        self._changed()

    def __setitem__(self, index, value) -> None:
        candidate = list(self)
        candidate[index] = value
        self._validate_replacement(candidate)
        super().__setitem__(index, value)
        self._changed()

    def __delitem__(self, index) -> None:
        super().__delitem__(index)
        self._changed()

    def __iadd__(self, entries):
        added = list(entries)
        self._validate_replacement([*self, *added])
        result = super().__iadd__(added)
        self._changed()
        return result


class CanvasObjectsCollection:
    """Propriétaire de la collection projetée et de ses index de recherche.

    Les entrées restent des dictionnaires afin de préserver intégralement le
    contrat historique de ``_last_drawn`` durant la migration progressive.
    """

    def __init__(self, entries: Optional[Iterable[dict]] = None):
        self._by_topology_id: Dict[str, tuple[int, dict]] = {}
        self._revision = 0
        initial_entries = list(entries or ())
        self._validate_entries(initial_entries)
        self._entries = _ObservableEntries(self, initial_entries)
        self._rebuild_indexes()

    @property
    def entries(self) -> list:
        """Liste réelle, exposée temporairement pour l'alias ``_last_drawn``."""
        return self._entries

    def __len__(self) -> int:
        return len(self._entries)

    def __iter__(self) -> Iterator[dict]:
        return iter(self._entries)

    def __getitem__(self, index):
        return self._entries[index]

    def get_by_topology_id(self, topology_id: str) -> dict | None:
        self._rebuild_indexes()
        item = self._by_topology_id.get(str(topology_id or "").strip())
        return None if item is None else item[1]

    def get_index_by_topology_id(self, topology_id: str) -> int | None:
        self._rebuild_indexes()
        item = self._by_topology_id.get(str(topology_id or "").strip())
        return None if item is None else item[0]

    def get_many_by_topology_ids(
        self,
        topology_ids,
        *,
        strict: bool = False,
    ) -> tuple[dict, ...]:
        """Resout plusieurs IDs topologiques dans l'ordre demande."""
        return tuple(entry for _index, entry in self.get_indexed_by_topology_ids(
            topology_ids,
            strict=strict,
        ))

    def get_indexed_by_topology_ids(
        self,
        topology_ids,
        *,
        strict: bool = False,
    ) -> tuple[tuple[int, dict], ...]:
        """Resout plusieurs IDs topologiques en conservant ordre et doublons."""
        self._rebuild_indexes()
        indexed_entries = []
        for topology_id in topology_ids or ():
            key = str(topology_id or "").strip()
            item = self._by_topology_id.get(key)
            if item is None:
                if strict:
                    raise KeyError(f"topology_id absent: {topology_id!r}")
                continue
            indexed_entries.append(item)
        return tuple(indexed_entries)

    def iter_indexed(self, *, reverse: bool = False) -> Iterator[tuple[int, dict]]:
        """Itere sur ``(index, entree)`` sans copier la collection."""
        indices = range(len(self._entries) - 1, -1, -1) if reverse else range(len(self._entries))
        for index in indices:
            yield index, self._entries[index]

    def topology_ids(self) -> tuple[str, ...]:
        """Retourne les IDs topologiques normalises, dans l'ordre de projection."""
        topology_ids = []
        for entry in self._entries:
            if not isinstance(entry, dict):
                continue
            topology_id = str(entry.get("topoElementId", "") or "").strip()
            if topology_id:
                topology_ids.append(topology_id)
        return tuple(topology_ids)

    def add(self, entry: dict) -> dict:
        self._validate(entry)
        self._entries.append(entry)
        return entry

    def remove(self, entry: dict) -> None:
        self._entries.remove(entry)

    def remove_at(self, index: int) -> dict:
        return self._entries.pop(index)

    def remove_many(self, indices: Iterable[int]) -> CanvasObjectsRemovalResult:
        """Supprime plusieurs indices en preservant l'ordre des autres entrees."""
        requested_indices = []
        seen_indices = set()
        for index in indices:
            normalized_index = int(index)
            if normalized_index in seen_indices:
                continue
            seen_indices.add(normalized_index)
            requested_indices.append(normalized_index)

        removed_entries = [
            self._entries[index]
            for index in requested_indices
            if 0 <= index < len(self._entries)
        ]
        removed_indices = {
            index for index in requested_indices if 0 <= index < len(self._entries)
        }
        kept_entries = []
        old_to_new = {}
        for old_index, entry in enumerate(self._entries):
            if old_index in removed_indices:
                continue
            old_to_new[old_index] = len(kept_entries)
            kept_entries.append(entry)
        self.replace_all(kept_entries)
        return CanvasObjectsRemovalResult(
            removed_entries=removed_entries,
            old_to_new=old_to_new,
        )

    def replace_all(self, entries: Iterable[dict]) -> None:
        replacement = list(entries)
        self._validate_entries(replacement)
        self._entries[:] = replacement

    def clear(self) -> None:
        self._entries.clear()

    def dump(self, logger, label: str | None = None) -> None:
        """Emet un diagnostic lisible de la collection, sans effet fonctionnel."""
        try:
            self._rebuild_indexes()
            prefix = f"[{label}] " if label else ""
            logger.debug(
                "%sCanvasObjectsCollection entries=%s revision=%s",
                prefix,
                len(self._entries),
                self._revision,
            )
            for index, entry in enumerate(self._entries):
                if isinstance(entry, dict):
                    logger.debug(
                        "[%03d] topo=%s",
                        index,
                        entry.get("topoElementId"),
                    )
                else:
                    logger.debug("[%03d] entry=%r", index, entry)
            logger.debug("Topology index")
            for topology_id, (index, _entry) in self._by_topology_id.items():
                logger.debug("%s -> %s", topology_id, index)
        except Exception:
            # Instrumentation pure : un dump ne doit jamais interrompre l'UI.
            return

    def validate_against_world(self, topology_world) -> None:
        """Valide le contrat minimal d'une projection contre son monde Core.

        Ce contrôle explicite est destiné aux tests et aux diagnostics : il ne
        modifie ni la collection ni le monde.
        """
        self._validate_entries(self._entries)
        elements = getattr(topology_world, "elements", None)
        if not isinstance(elements, dict):
            raise TypeError("CanvasObjectsCollection: TopologyWorld invalide")
        for entry in self._entries:
            topology_id = str(entry["topoElementId"]).strip()
            if topology_id not in elements:
                raise ValueError(
                    "CanvasObjectsCollection: topoElementId absent du Core: "
                    f"{topology_id!r}"
                )
            points = entry.get("pts")
            if not isinstance(points, dict) or set(points) != {"O", "B", "L"}:
                raise ValueError(
                    "CanvasObjectsCollection: pts doit contenir exactement O, B et L"
                )
            for vertex_name in ("O", "B", "L"):
                try:
                    coordinates = tuple(points[vertex_name])
                except TypeError as exc:
                    raise ValueError(
                        f"CanvasObjectsCollection: point {vertex_name} invalide"
                    ) from exc
                if len(coordinates) != 2 or not all(
                    math.isfinite(float(value)) for value in coordinates
                ):
                    raise ValueError(
                        f"CanvasObjectsCollection: point {vertex_name} invalide"
                    )

    def _rebuild_indexes(self) -> None:
        self._by_topology_id = {}
        for index, entry in enumerate(self._entries):
            if not isinstance(entry, dict):
                continue
            topology_id = str(entry.get("topoElementId", "") or "").strip()
            if topology_id:
                self._by_topology_id.setdefault(topology_id, (index, entry))
        self._revision += 1

    @staticmethod
    def _validate(entry: dict) -> None:
        if not isinstance(entry, dict):
            raise TypeError("CanvasObjectsCollection: une entree doit etre un dictionnaire")
        forbidden_fields = {
            "id", "orient", "mirrored", "group_id", "topoGroupId", "labels",
        }
        present_forbidden_fields = forbidden_fields.intersection(entry)
        if present_forbidden_fields:
            raise ValueError(
                "CanvasObjectsCollection: champs legacy interdits: "
                f"{sorted(present_forbidden_fields)!r}"
            )
        topology_id = str(entry.get("topoElementId", "") or "").strip()
        if not topology_id:
            raise ValueError("CanvasObjectsCollection: topoElementId obligatoire")

    @classmethod
    def _validate_entries(cls, entries: Iterable[dict]) -> None:
        seen_topology_ids = set()
        for entry in entries:
            cls._validate(entry)
            topology_id = str(entry["topoElementId"]).strip()
            if topology_id in seen_topology_ids:
                raise ValueError(
                    f"CanvasObjectsCollection: topoElementId dupliqué: {topology_id!r}"
                )
            seen_topology_ids.add(topology_id)
