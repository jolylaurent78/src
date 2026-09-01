"""Contrat d'identite des objets persistants du Catalogue."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import os
from pathlib import Path
import re
import uuid
from typing import Mapping, MutableMapping, TYPE_CHECKING

if TYPE_CHECKING:
    from src.assembleur_catalogue import Catalogue


CATALOGUE_ID_KINDS = frozenset({"city", "beacon", "triangle", "template"})
CATALOGUE_ID_KIND_ORDER = ("city", "beacon", "triangle", "template")
_PREFIX_BY_KIND = {
    "city": "CITY",
    "beacon": "BEA",
    "triangle": "TRI",
    "template": "TPL",
}
_KIND_BY_PREFIX = {prefix: kind for kind, prefix in _PREFIX_BY_KIND.items()}
_SYSTEM_ID_RE = re.compile(r"^(CITY|BEA|TRI|TPL)-SYS-(\d{6,})$")
_USER_ID_RE = re.compile(
    r"^(CITY|BEA|TRI|TPL)-USR-"
    r"([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})$"
)
_ASSEMBLEUR_MODE_ENV = "ASSEMBLEUR_MODE"


def load_project_dotenv(
    *,
    environ: MutableMapping[str, str] | None = None,
    env_path: str | Path | None = None,
) -> bool:
    """Charge uniquement ASSEMBLEUR_MODE depuis le .env de développement.

    L'environnement du processus reste prioritaire et l'absence de .env est
    volontairement sans effet pour les distributions normales.
    """
    target_environment = os.environ if environ is None else environ
    target_path = (
        Path(__file__).resolve().parent.parent / ".env"
        if env_path is None else Path(env_path)
    )
    if not target_path.is_file():
        return False
    for raw_line in target_path.read_text(encoding="utf-8").splitlines():
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        if key.strip() == _ASSEMBLEUR_MODE_ENV:
            target_environment.setdefault(_ASSEMBLEUR_MODE_ENV, value.strip())
    return True


def _require_kind(kind: str) -> str:
    if kind not in CATALOGUE_ID_KINDS:
        raise ValueError(f"Type d'identite Catalogue inconnu : {kind!r}")
    return kind


def _matches_kind(value: object, kind: str, pattern: re.Pattern[str]) -> bool:
    if not isinstance(value, str):
        return False
    match = pattern.fullmatch(value)
    return match is not None and _KIND_BY_PREFIX[match.group(1)] == kind


def is_system_catalogue_id(value: object) -> bool:
    if not isinstance(value, str):
        return False
    match = _SYSTEM_ID_RE.fullmatch(value)
    return match is not None and int(match.group(2)) > 0


def get_system_catalogue_id_number(value: object, kind: str | None = None) -> int:
    """Retourne le numero d'un identifiant SYS Catalogue contractuel.

    Cette extraction est reservee aux controles de coherence de persistance ;
    elle ne participe pas a l'allocation des identifiants.
    """
    if kind is not None:
        _require_kind(kind)
    if not isinstance(value, str):
        raise ValueError(f"Identifiant SYS Catalogue invalide : {value!r}")
    match = _SYSTEM_ID_RE.fullmatch(value)
    if match is None or int(match.group(2)) <= 0:
        raise ValueError(f"Identifiant SYS Catalogue invalide : {value!r}")
    if kind is not None and _KIND_BY_PREFIX[match.group(1)] != kind:
        raise ValueError(
            f"Identifiant SYS Catalogue de type {kind!r} attendu : {value!r}"
        )
    return int(match.group(2))


def is_user_catalogue_id(value: object) -> bool:
    if not isinstance(value, str):
        return False
    match = _USER_ID_RE.fullmatch(value)
    if match is None:
        return False
    parsed = uuid.UUID(match.group(2))
    return parsed.version == 4 and str(parsed) == match.group(2)


def is_catalogue_id(value: object, kind: str) -> bool:
    _require_kind(kind)
    return (
        _matches_kind(value, kind, _SYSTEM_ID_RE) and is_system_catalogue_id(value)
    ) or (
        _matches_kind(value, kind, _USER_ID_RE) and is_user_catalogue_id(value)
    )


def is_catalogue_city_id(value: object) -> bool:
    return is_catalogue_id(value, "city")


def is_catalogue_beacon_id(value: object) -> bool:
    return is_catalogue_id(value, "beacon")


def is_catalogue_triangle_id(value: object) -> bool:
    return is_catalogue_id(value, "triangle")


def is_catalogue_template_id(value: object) -> bool:
    return is_catalogue_id(value, "template")


class CatalogueIdProvider(ABC):
    """Fabrique les identites Catalogue selon un contexte deja resolu."""

    @abstractmethod
    def new_id(self, catalogue: "Catalogue", kind: str) -> str:
        raise NotImplementedError

    def new_city_id(self, catalogue: "Catalogue") -> str:
        return self.new_id(catalogue, "city")

    def new_beacon_id(self, catalogue: "Catalogue") -> str:
        return self.new_id(catalogue, "beacon")

    def new_triangle_id(self, catalogue: "Catalogue") -> str:
        return self.new_id(catalogue, "triangle")

    def new_template_id(self, catalogue: "Catalogue") -> str:
        return self.new_id(catalogue, "template")


class SystemCatalogueIdProvider(CatalogueIdProvider):
    def new_id(self, catalogue: "Catalogue", kind: str) -> str:
        kind = _require_kind(kind)
        return f"{_PREFIX_BY_KIND[kind]}-SYS-{catalogue._allocate_system_id_number(kind):06d}"


class UserCatalogueIdProvider(CatalogueIdProvider):
    def new_id(self, catalogue: "Catalogue", kind: str) -> str:
        kind = _require_kind(kind)
        return f"{_PREFIX_BY_KIND[kind]}-USR-{uuid.uuid4()}"


@dataclass(frozen=True)
class ApplicationContext:
    mode: str
    catalogue_id_provider: CatalogueIdProvider

    @classmethod
    def from_environment(cls, environ: Mapping[str, str] | None = None) -> "ApplicationContext":
        source = os.environ if environ is None else environ
        mode = str(source.get(_ASSEMBLEUR_MODE_ENV, "USER")).strip().upper()
        if mode == "SYS":
            return cls(mode="SYS", catalogue_id_provider=SystemCatalogueIdProvider())
        if mode == "USER":
            return cls(mode="USER", catalogue_id_provider=UserCatalogueIdProvider())
        raise ValueError(
            "ASSEMBLEUR_MODE invalide : "
            f"{mode!r} (valeurs acceptees : SYS, USER)."
        )
