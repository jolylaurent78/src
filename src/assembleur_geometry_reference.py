"""R\u00e9f\u00e9rentiel g\u00e9om\u00e9trique local d'un sc\u00e9nario et r\u00e9solution unifi\u00e9e.

Le module ne modifie ni le Catalogue ni le r\u00e9f\u00e9rentiel local. Il fournit
uniquement des d\u00e9finitions r\u00e9solues et leur mat\u00e9rialisation Core.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import re

from pyproj import Transformer

from src.assembleur_catalogue import Catalogue
from src.assembleur_core import TopologyElement, build_topology_element_from_catalogue_triangle


@dataclass
class ScenarioCity:
    """Ville propre \u00e0 un sc\u00e9nario, distincte d'une ville Catalogue."""

    city_ref_id: str
    name: str
    latitude: float
    longitude: float
    catalogue_source_city_id: str | None = None


@dataclass
class ScenarioTriangle:
    """Triangle propre \u00e0 un sc\u00e9nario, pouvant r\u00e9f\u00e9rencer CITY ou SCITY."""

    triangle_ref_id: str
    note: str
    opening_city_ref_id: str
    base_city_ref_id: str
    light_city_ref_id: str
    catalogue_source_triangle_id: str | None = None


@dataclass(frozen=True)
class ResolvedCityDefinition:
    """Vue de lecture ind\u00e9pendante de l'origine d'une ville."""

    ref_id: str
    name: str
    latitude: float
    longitude: float
    origin: str
    catalogue_source_city_id: str | None = None


@dataclass(frozen=True)
class ResolvedTriangleDefinition:
    """Vue de lecture ind\u00e9pendante de l'origine d'un triangle."""

    ref_id: str
    note: str
    opening_city_ref_id: str
    base_city_ref_id: str
    light_city_ref_id: str
    origin: str
    catalogue_source_triangle_id: str | None = None


class ScenarioReference:
    """Overlay local minimal d'un sc\u00e9nario sur le Catalogue global."""

    _CITY_ID_PATTERN = re.compile(r"SCITY-\d{4,}$")
    _TRIANGLE_ID_PATTERN = re.compile(r"STRI-\d{4,}$")

    def __init__(self) -> None:
        self.cities: dict[str, ScenarioCity] = {}
        self.triangles: dict[str, ScenarioTriangle] = {}

    @staticmethod
    def _require_name(name: str, label: str) -> str:
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"Le nom {label} ne peut pas \u00eatre vide.")
        return name.strip()

    @staticmethod
    def _require_coordinate(value: float, minimum: float, maximum: float, label: str) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{label} doit \u00eatre un nombre.")
        normalized = float(value)
        if not minimum <= normalized <= maximum:
            raise ValueError(f"{label} doit \u00eatre compris entre {minimum} et {maximum}.")
        return normalized

    @staticmethod
    def _require_optional_catalogue_source(value: str | None, label: str) -> str | None:
        if value is None:
            return None
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{label} doit \u00eatre une cha\u00eene non vide ou None.")
        return value.strip()

    @staticmethod
    def _next_id(prefix: str, items: Mapping[str, object], pattern: re.Pattern[str]) -> str:
        numbers = [
            int(item_id.split("-", 1)[1])
            for item_id in items
            if pattern.fullmatch(item_id)
        ]
        return f"{prefix}-{max(numbers, default=0) + 1:04d}"

    def next_city_ref_id(self) -> str:
        return self._next_id("SCITY", self.cities, self._CITY_ID_PATTERN)

    def next_triangle_ref_id(self) -> str:
        return self._next_id("STRI", self.triangles, self._TRIANGLE_ID_PATTERN)

    def add_city(self, city: ScenarioCity) -> ScenarioCity:
        if not isinstance(city, ScenarioCity):
            raise TypeError("ScenarioReference.add_city attend une ScenarioCity")
        city_id = str(city.city_ref_id).strip()
        if not self._CITY_ID_PATTERN.fullmatch(city_id):
            raise ValueError(f"Identifiant de ville locale invalide : {city_id!r}")
        if city_id in self.cities:
            raise ValueError(f"Ville locale d\u00e9j\u00e0 d\u00e9finie : {city_id}")
        city.city_ref_id = city_id
        city.name = self._require_name(city.name, "de ville locale")
        city.latitude = self._require_coordinate(city.latitude, -90, 90, "La latitude")
        city.longitude = self._require_coordinate(city.longitude, -180, 180, "La longitude")
        city.catalogue_source_city_id = self._require_optional_catalogue_source(
            city.catalogue_source_city_id, "catalogue_source_city_id"
        )
        self.cities[city_id] = city
        return city

    def create_city(
        self,
        name: str,
        latitude: float,
        longitude: float,
        *,
        catalogue_source_city_id: str | None = None,
    ) -> ScenarioCity:
        return self.add_city(
            ScenarioCity(
                self.next_city_ref_id(),
                name,
                latitude,
                longitude,
                catalogue_source_city_id,
            )
        )

    def rename_city(self, city_ref_id: str, name: str) -> ScenarioCity:
        """Renomme une ville locale sans modifier son identite ni sa provenance."""
        city_id = str(city_ref_id).strip()
        try:
            city = self.cities[city_id]
        except KeyError as exc:
            raise KeyError(f"Ville locale inconnue : {city_id}") from exc
        city.name = self._require_name(name, "de ville locale")
        return city

    def add_triangle(self, triangle: ScenarioTriangle) -> ScenarioTriangle:
        if not isinstance(triangle, ScenarioTriangle):
            raise TypeError("ScenarioReference.add_triangle attend un ScenarioTriangle")
        triangle_id = str(triangle.triangle_ref_id).strip()
        if not self._TRIANGLE_ID_PATTERN.fullmatch(triangle_id):
            raise ValueError(f"Identifiant de triangle local invalide : {triangle_id!r}")
        if triangle_id in self.triangles:
            raise ValueError(f"Triangle local d\u00e9j\u00e0 d\u00e9fini : {triangle_id}")
        city_refs = (
            triangle.opening_city_ref_id,
            triangle.base_city_ref_id,
            triangle.light_city_ref_id,
        )
        if any(not isinstance(ref_id, str) or not ref_id.strip() for ref_id in city_refs):
            raise ValueError("Les trois r\u00e9f\u00e9rences de ville locale doivent \u00eatre non vides.")
        if len({ref_id.strip() for ref_id in city_refs}) != 3:
            raise ValueError("Les trois villes d'un triangle local doivent \u00eatre distinctes.")
        triangle.triangle_ref_id = triangle_id
        triangle.note = self._require_name(triangle.note, "de triangle local")
        (
            triangle.opening_city_ref_id,
            triangle.base_city_ref_id,
            triangle.light_city_ref_id,
        ) = tuple(ref_id.strip() for ref_id in city_refs)
        triangle.catalogue_source_triangle_id = self._require_optional_catalogue_source(
            triangle.catalogue_source_triangle_id, "catalogue_source_triangle_id"
        )
        self.triangles[triangle_id] = triangle
        return triangle

    def create_triangle(
        self,
        note: str,
        opening_city_ref_id: str,
        base_city_ref_id: str,
        light_city_ref_id: str,
        *,
        catalogue_source_triangle_id: str | None = None,
    ) -> ScenarioTriangle:
        return self.add_triangle(
            ScenarioTriangle(
                self.next_triangle_ref_id(),
                note,
                opening_city_ref_id,
                base_city_ref_id,
                light_city_ref_id,
                catalogue_source_triangle_id,
            )
        )

    def clone(self) -> "ScenarioReference":
        cloned = ScenarioReference()
        cloned.cities = {
            city_id: ScenarioCity(
                city.city_ref_id,
                city.name,
                city.latitude,
                city.longitude,
                city.catalogue_source_city_id,
            )
            for city_id, city in self.cities.items()
        }
        cloned.triangles = {
            triangle_id: ScenarioTriangle(
                triangle.triangle_ref_id,
                triangle.note,
                triangle.opening_city_ref_id,
                triangle.base_city_ref_id,
                triangle.light_city_ref_id,
                triangle.catalogue_source_triangle_id,
            )
            for triangle_id, triangle in self.triangles.items()
        }
        return cloned


class GeometryReferenceResolver:
    """R\u00e9sout les r\u00e9f\u00e9rences Catalogue et locales sans les modifier."""

    def __init__(self, catalogue: Catalogue, reference: ScenarioReference) -> None:
        if not isinstance(catalogue, Catalogue):
            raise TypeError("GeometryReferenceResolver attend un Catalogue")
        if not isinstance(reference, ScenarioReference):
            raise TypeError("GeometryReferenceResolver attend un ScenarioReference")
        self.catalogue = catalogue
        self.reference = reference
        self._lambert_transformer = Transformer.from_crs(
            "EPSG:4326", "EPSG:2154", always_xy=True
        )

    def resolve_city(self, ref_id: str) -> ResolvedCityDefinition:
        normalized = str(ref_id).strip()
        city = self.reference.cities.get(normalized)
        if city is not None:
            return ResolvedCityDefinition(
                ref_id=city.city_ref_id,
                name=city.name,
                latitude=city.latitude,
                longitude=city.longitude,
                origin="scenario",
                catalogue_source_city_id=city.catalogue_source_city_id,
            )
        city = self.catalogue.get_city(normalized)
        return ResolvedCityDefinition(
            ref_id=city.city_id,
            name=city.name,
            latitude=city.latitude,
            longitude=city.longitude,
            origin="catalogue",
            catalogue_source_city_id=city.city_id,
        )

    def resolve_triangle(self, ref_id: str) -> ResolvedTriangleDefinition:
        normalized = str(ref_id).strip()
        triangle = self.reference.triangles.get(normalized)
        if triangle is not None:
            return ResolvedTriangleDefinition(
                ref_id=triangle.triangle_ref_id,
                note=triangle.note,
                opening_city_ref_id=triangle.opening_city_ref_id,
                base_city_ref_id=triangle.base_city_ref_id,
                light_city_ref_id=triangle.light_city_ref_id,
                origin="scenario",
                catalogue_source_triangle_id=triangle.catalogue_source_triangle_id,
            )
        triangle = self.catalogue.get_triangle(normalized)
        return ResolvedTriangleDefinition(
            ref_id=triangle.triangle_id,
            note=triangle.note,
            opening_city_ref_id=triangle.opening_city_id,
            base_city_ref_id=triangle.base_city_id,
            light_city_ref_id=triangle.light_city_id,
            origin="catalogue",
            catalogue_source_triangle_id=triangle.triangle_id,
        )

    def get_catalogue_source_triangle_id(self, triangle_ref_id: str) -> str:
        """Retourne le TRI Catalogue canonique d'une reference TRI ou STRI."""
        try:
            triangle = self.resolve_triangle(triangle_ref_id)
        except KeyError as exc:
            raise ValueError(
                "Reference de triangle inconnue : "
                f"{triangle_ref_id!r}"
            ) from exc
        source_triangle_id = triangle.catalogue_source_triangle_id
        if not isinstance(source_triangle_id, str) or not source_triangle_id.strip():
            raise ValueError(
                "Triangle sans provenance Catalogue exploitable : "
                f"{triangle_ref_id!r}"
            )
        source_triangle_id = source_triangle_id.strip()
        try:
            self.catalogue.get_triangle(source_triangle_id)
        except KeyError as exc:
            raise ValueError(
                "Provenance Catalogue de triangle invalide : "
                f"{triangle_ref_id!r} -> {source_triangle_id!r}"
            ) from exc
        return source_triangle_id

    def get_city_lambert(self, city_ref_id: str) -> tuple[float, float]:
        city = self.resolve_city(city_ref_id)
        if city.origin == "catalogue":
            return self.catalogue.get_city_lambert(city.ref_id)
        point = self._lambert_transformer.transform(city.longitude, city.latitude)
        return (float(point[0]), float(point[1]))

    def lambert_to_geographic(self, lambert_x_m: float, lambert_y_m: float) -> tuple[float, float]:
        """Convertit un point Lambert vers (latitude, longitude) sans dépendre de Tk."""
        transformer = Transformer.from_crs("EPSG:2154", "EPSG:4326", always_xy=True)
        longitude, latitude = transformer.transform(float(lambert_x_m), float(lambert_y_m))
        return (float(latitude), float(longitude))

    def city_ref_ids_by_role(self, triangle_ref_id: str) -> dict[str, str]:
        triangle = self.resolve_triangle(triangle_ref_id)
        return {
            "O": triangle.opening_city_ref_id,
            "B": triangle.base_city_ref_id,
            "L": triangle.light_city_ref_id,
        }

    def materialize_triangle(
        self,
        triangle_ref_id: str,
        *,
        vertex_lambert_overrides: Mapping[str, tuple[float, float]] | None = None,
    ) -> TopologyElement:
        if vertex_lambert_overrides is None:
            overrides: dict[str, tuple[float, float]] = {}
        elif not isinstance(vertex_lambert_overrides, Mapping):
            raise ValueError("Les overrides Lambert doivent former une mapping O/B/L")
        else:
            overrides = dict(vertex_lambert_overrides)
        unknown_roles = set(overrides) - {"O", "B", "L"}
        if unknown_roles:
            raise ValueError(
                "R\u00f4le d'override Lambert inconnu: "
                + ", ".join(sorted(str(role) for role in unknown_roles))
            )

        triangle = self.resolve_triangle(triangle_ref_id)
        city_refs = self.city_ref_ids_by_role(triangle.ref_id)
        cities = {role: self.resolve_city(city_ref_id) for role, city_ref_id in city_refs.items()}
        lambert_by_role = {
            role: self.get_city_lambert(city_ref_id)
            for role, city_ref_id in city_refs.items()
        }
        lambert_by_role.update(overrides)
        return build_topology_element_from_catalogue_triangle(
            triangle_id=triangle.ref_id,
            opening_name=cities["O"].name,
            base_name=cities["B"].name,
            light_name=cities["L"].name,
            opening_lambert_xy=lambert_by_role["O"],
            base_lambert_xy=lambert_by_role["B"],
            light_lambert_xy=lambert_by_role["L"],
            opening_city_ref_id=city_refs["O"],
            base_city_ref_id=city_refs["B"],
            light_city_ref_id=city_refs["L"],
        )
