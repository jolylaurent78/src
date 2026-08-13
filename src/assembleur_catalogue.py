"""Modèle métier autonome du Catalogue, sans dépendance à l'interface ni au stockage."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import re

from pyproj import Transformer


@dataclass
class CatalogueCity:
    city_id: str
    name: str
    latitude: float
    longitude: float
    archived: bool = False


@dataclass
class CatalogueTriangle:
    triangle_id: str
    note: str
    opening_city_id: str
    base_city_id: str
    light_city_id: str
    archived: bool = False


@dataclass
class HypothesisTemplate:
    template_id: str
    name: str
    description: str = ""
    archived: bool = False
    triangle_ids_by_rank: list[str | None] = field(default_factory=lambda: [None] * 32)


@dataclass(frozen=True)
class TriangleGeometry:
    distance_ob_km: float
    distance_ol_km: float
    distance_bl_km: float
    angle_o_deg: float
    angle_b_deg: float
    angle_l_deg: float
    orientation: str


@dataclass(frozen=True)
class TemplateValidationStatus:
    state: str
    filled_ranks: int
    message: str | None = None


class Catalogue:
    """Agrégat racine du catalogue persistant futur."""

    version = 1
    _ID_PATTERNS = {
        "CITY": re.compile(r"CITY-\d{4,}$"),
        "TRI": re.compile(r"TRI-\d{4,}$"),
        "TPL": re.compile(r"TPL-\d{4,}$"),
    }
    _MIN_EDGE_LENGTH_M = 1e-6
    _MIN_DOUBLE_AREA_M2 = 1e-6

    def __init__(self) -> None:
        self.version = 1
        self.cities: dict[str, CatalogueCity] = {}
        self.triangles: dict[str, CatalogueTriangle] = {}
        self.templates: dict[str, HypothesisTemplate] = {}
        self.default_template_id: str | None = None
        self._city_lambert_cache: dict[str, tuple[float, float]] = {}
        self._lambert_transformer = Transformer.from_crs("EPSG:4326", "EPSG:2154", always_xy=True)

    def clone(self) -> "Catalogue":
        """Copie l'état métier sans propager les caches runtime."""
        cloned = Catalogue()
        cloned.version = self.version
        cloned.cities = {
            city_id: CatalogueCity(city.city_id, city.name, city.latitude, city.longitude, city.archived)
            for city_id, city in self.cities.items()
        }
        cloned.triangles = {
            triangle_id: CatalogueTriangle(
                triangle.triangle_id,
                triangle.note,
                triangle.opening_city_id,
                triangle.base_city_id,
                triangle.light_city_id,
                triangle.archived,
            )
            for triangle_id, triangle in self.triangles.items()
        }
        cloned.templates = {
            template_id: HypothesisTemplate(
                template.template_id,
                template.name,
                template.description,
                template.archived,
                list(template.triangle_ids_by_rank),
            )
            for template_id, template in self.templates.items()
        }
        cloned.default_template_id = self.default_template_id
        return cloned

    def _next_id(self, prefix: str, items: dict[str, object]) -> str:
        numbers = [int(item_id.split("-", 1)[1]) for item_id in items if self._ID_PATTERNS[prefix].fullmatch(item_id)]
        return f"{prefix}-{max(numbers, default=0) + 1:04d}"

    def _next_city_id(self) -> str:
        return self._next_id("CITY", self.cities)

    def _next_triangle_id(self) -> str:
        return self._next_id("TRI", self.triangles)

    def _next_template_id(self) -> str:
        return self._next_id("TPL", self.templates)

    def get_city(self, city_id: str) -> CatalogueCity:
        try:
            return self.cities[city_id]
        except KeyError as exc:
            raise KeyError(f"Ville inconnue : {city_id}") from exc

    def get_triangle(self, triangle_id: str) -> CatalogueTriangle:
        try:
            return self.triangles[triangle_id]
        except KeyError as exc:
            raise KeyError(f"Triangle inconnu : {triangle_id}") from exc

    def get_template(self, template_id: str) -> HypothesisTemplate:
        try:
            return self.templates[template_id]
        except KeyError as exc:
            raise KeyError(f"Template inconnu : {template_id}") from exc

    def iter_cities(self) -> tuple[CatalogueCity, ...]:
        return tuple(self.cities[item_id] for item_id in sorted(self.cities))

    def iter_triangles(self) -> tuple[CatalogueTriangle, ...]:
        return tuple(self.triangles[item_id] for item_id in sorted(self.triangles))

    def iter_templates(self) -> tuple[HypothesisTemplate, ...]:
        return tuple(self.templates[item_id] for item_id in sorted(self.templates))

    @staticmethod
    def _validate_name(name: str, label: str) -> str:
        cleaned = name.strip()
        if not cleaned:
            raise ValueError(f"Le nom {label} ne peut pas être vide.")
        return cleaned

    @staticmethod
    def _validate_coordinate(value: float, minimum: float, maximum: float, label: str) -> None:
        if not minimum <= value <= maximum:
            raise ValueError(f"{label} doit être compris entre {minimum} et {maximum}.")

    @staticmethod
    def _validate_note(note: str) -> str:
        cleaned = note.strip()
        if not cleaned:
            raise ValueError("La note du triangle ne peut pas être vide.")
        return cleaned

    @staticmethod
    def _ensure_unique_name(name: str, items: dict[str, object], current_id: str | None, attribute: str) -> None:
        if any(item_id != current_id and getattr(item, attribute).casefold() == name.casefold() for item_id, item in items.items()):
            raise ValueError(f"Un objet porte déjà le nom « {name} ».")

    def add_city(self, name: str, latitude: float, longitude: float, *, archived: bool = False) -> CatalogueCity:
        name = self._validate_name(name, "de ville")
        self._validate_coordinate(latitude, -90, 90, "La latitude")
        self._validate_coordinate(longitude, -180, 180, "La longitude")
        self._ensure_unique_name(name, self.cities, None, "name")
        city = CatalogueCity(self._next_city_id(), name, latitude, longitude, archived)
        self.cities[city.city_id] = city
        return city

    def update_city(self, city_id: str, *, name: str | None = None, latitude: float | None = None,
                    longitude: float | None = None, archived: bool | None = None) -> CatalogueCity:
        city = self.get_city(city_id)
        final_name = self._validate_name(name, "de ville") if name is not None else city.name
        final_latitude = latitude if latitude is not None else city.latitude
        final_longitude = longitude if longitude is not None else city.longitude
        self._validate_coordinate(final_latitude, -90, 90, "La latitude")
        self._validate_coordinate(final_longitude, -180, 180, "La longitude")
        self._ensure_unique_name(final_name, self.cities, city_id, "name")
        coordinates_changed = (final_latitude, final_longitude) != (city.latitude, city.longitude)
        city.name, city.latitude, city.longitude = final_name, final_latitude, final_longitude
        if archived is not None:
            city.archived = archived
        if coordinates_changed:
            self._city_lambert_cache.pop(city_id, None)
        return city

    def get_triangles_referencing_city(self, city_id: str) -> tuple[CatalogueTriangle, ...]:
        self.get_city(city_id)
        return tuple(triangle for triangle in self.iter_triangles() if city_id in (
            triangle.opening_city_id, triangle.base_city_id, triangle.light_city_id))

    def delete_city(self, city_id: str) -> None:
        references = self.get_triangles_referencing_city(city_id)
        if references:
            raise ValueError(f"Impossible de supprimer {city_id} : la ville est référencée par {len(references)} triangle(s).")
        del self.cities[city_id]
        self._city_lambert_cache.pop(city_id, None)

    def _validate_triangle_references(self, opening_city_id: str, base_city_id: str, light_city_id: str,
                                      *, reject_archived_new: set[str] | None = None) -> None:
        city_ids = (opening_city_id, base_city_id, light_city_id)
        if len(set(city_ids)) != 3:
            raise ValueError("Les trois villes d'un triangle doivent être distinctes.")
        for city_id in city_ids:
            city = self.get_city(city_id)
            if reject_archived_new is not None and city_id in reject_archived_new and city.archived:
                raise ValueError(f"La ville archivée {city_id} ne peut pas être sélectionnée.")

    def _ensure_unique_triplet(self, opening_city_id: str, base_city_id: str, light_city_id: str,
                               current_id: str | None = None) -> None:
        triplet = (opening_city_id, base_city_id, light_city_id)
        if any(item_id != current_id and (triangle.opening_city_id, triangle.base_city_id, triangle.light_city_id) == triplet
               for item_id, triangle in self.triangles.items()):
            raise ValueError("Un triangle possède déjà ce triplet Ouverture / Base / Lumière.")

    def add_triangle(self, note: str, opening_city_id: str, base_city_id: str, light_city_id: str,
                     *, archived: bool = False) -> CatalogueTriangle:
        note = self._validate_note(note)
        self._validate_triangle_references(opening_city_id, base_city_id, light_city_id,
                                           reject_archived_new={opening_city_id, base_city_id, light_city_id})
        self._ensure_unique_triplet(opening_city_id, base_city_id, light_city_id)
        triangle = CatalogueTriangle(self._next_triangle_id(), note, opening_city_id, base_city_id, light_city_id, archived)
        self.triangles[triangle.triangle_id] = triangle
        return triangle

    def update_triangle(self, triangle_id: str, *, note: str | None = None, opening_city_id: str | None = None,
                        base_city_id: str | None = None, light_city_id: str | None = None,
                        archived: bool | None = None) -> CatalogueTriangle:
        triangle = self.get_triangle(triangle_id)
        final_note = self._validate_note(note) if note is not None else triangle.note
        final_ids = (
            opening_city_id if opening_city_id is not None else triangle.opening_city_id,
            base_city_id if base_city_id is not None else triangle.base_city_id,
            light_city_id if light_city_id is not None else triangle.light_city_id,
        )
        changed_ids = {new for new, old in zip(final_ids, (triangle.opening_city_id, triangle.base_city_id, triangle.light_city_id)) if new != old}
        self._validate_triangle_references(*final_ids, reject_archived_new=changed_ids)
        self._ensure_unique_triplet(*final_ids, current_id=triangle_id)
        triangle.note = final_note
        triangle.opening_city_id, triangle.base_city_id, triangle.light_city_id = final_ids
        if archived is not None:
            triangle.archived = archived
        return triangle

    def get_templates_referencing_triangle(self, triangle_id: str) -> tuple[HypothesisTemplate, ...]:
        self.get_triangle(triangle_id)
        return tuple(template for template in self.iter_templates() if triangle_id in template.triangle_ids_by_rank)

    def delete_triangle(self, triangle_id: str) -> None:
        references = self.get_templates_referencing_triangle(triangle_id)
        if references:
            raise ValueError(f"Impossible de supprimer {triangle_id} : le triangle est référencé par {len(references)} template(s).")
        del self.triangles[triangle_id]

    def add_template(self, name: str, description: str = "", *, archived: bool = False) -> HypothesisTemplate:
        name = self._validate_name(name, "de template")
        self._ensure_unique_name(name, self.templates, None, "name")
        template = HypothesisTemplate(self._next_template_id(), name, description, archived)
        self.templates[template.template_id] = template
        if self.default_template_id is None:
            self.default_template_id = template.template_id
        return template

    def update_template(self, template_id: str, *, name: str | None = None, description: str | None = None,
                        archived: bool | None = None) -> HypothesisTemplate:
        template = self.get_template(template_id)
        final_name = self._validate_name(name, "de template") if name is not None else template.name
        self._ensure_unique_name(final_name, self.templates, template_id, "name")
        template.name = final_name
        if description is not None:
            template.description = description
        if archived is not None:
            template.archived = archived
        return template

    def _validate_template_ranks(self, ranks: list[str | None]) -> str | None:
        if len(ranks) != 32:
            return "Un template doit contenir exactement 32 rangs."
        seen: set[str] = set()
        for triangle_id in ranks:
            if triangle_id is None:
                continue
            if triangle_id not in self.triangles:
                return f"Triangle inconnu : {triangle_id}."
            if triangle_id in seen:
                return "Un triangle ne peut pas être utilisé dans plusieurs rangs."
            seen.add(triangle_id)
        for index in range(0, 32, 2):
            first, second = ranks[index], ranks[index + 1]
            if first is not None and second is not None and self.triangles[first].base_city_id != self.triangles[second].base_city_id:
                return f"Les rangs {index + 1} et {index + 2} doivent utiliser la même base."
        return None

    def set_template_ranks(self, template_id: str, triangle_ids_by_rank: list[str | None]) -> None:
        """Remplace atomiquement les 32 rangs d'un template par l'état final fourni."""
        template = self.get_template(template_id)
        preview = list(triangle_ids_by_rank)
        message = self._validate_template_ranks(preview)
        if message:
            raise ValueError(message)
        current_ids = {triangle_id for triangle_id in template.triangle_ids_by_rank if triangle_id is not None}
        for triangle_id in preview:
            if triangle_id is not None and self.triangles[triangle_id].archived and triangle_id not in current_ids:
                raise ValueError(f"Le triangle archivé {triangle_id} ne peut pas être affecté.")
        template.triangle_ids_by_rank[:] = preview

    def validate_template_ranks(self, template_id: str, triangle_ids_by_rank: list[str | None]) -> str | None:
        """Valide un aperçu de rangs sans modifier le catalogue."""
        template = self.get_template(template_id)
        preview = list(triangle_ids_by_rank)
        message = self._validate_template_ranks(preview)
        if message:
            return message
        current_ids = {item for item in template.triangle_ids_by_rank if item is not None}
        for triangle_id in preview:
            if triangle_id is not None and self.triangles[triangle_id].archived and triangle_id not in current_ids:
                return f"Le triangle archivé {triangle_id} ne peut pas être affecté."
        return None

    def set_template_rank(self, template_id: str, rank: int, triangle_id: str | None) -> None:
        if not 1 <= rank <= 32:
            raise ValueError("Le rang doit être compris entre 1 et 32.")
        template = self.get_template(template_id)
        preview = list(template.triangle_ids_by_rank)
        preview[rank - 1] = triangle_id
        self.set_template_ranks(template_id, preview)

    def get_template_validation_status(self, template_id: str) -> TemplateValidationStatus:
        template = self.get_template(template_id)
        filled_ranks = sum(triangle_id is not None for triangle_id in template.triangle_ids_by_rank)
        if filled_ranks < 32:
            return TemplateValidationStatus("Incomplet", filled_ranks)
        message = self._validate_template_ranks(template.triangle_ids_by_rank)
        return TemplateValidationStatus("Invalide", filled_ranks, message) if message else TemplateValidationStatus("Valide", filled_ranks)

    def set_default_template(self, template_id: str) -> None:
        self.get_template(template_id)
        self.default_template_id = template_id

    def delete_template(self, template_id: str) -> None:
        self.get_template(template_id)
        was_default = template_id == self.default_template_id
        del self.templates[template_id]
        if was_default:
            self.default_template_id = min(
                self.templates,
                key=lambda item_id: int(item_id.split("-", 1)[1]),
                default=None,
            )

    def get_default_template(self) -> HypothesisTemplate | None:
        if self.default_template_id is None:
            return None
        if self.default_template_id not in self.templates:
            raise RuntimeError(f"Le template par défaut {self.default_template_id} est absent du catalogue.")
        return self.templates[self.default_template_id]

    def require_valid_default_template(self) -> HypothesisTemplate:
        template = self.get_default_template()
        if template is None:
            raise ValueError("Aucun template par défaut n'est défini.")
        status = self.get_template_validation_status(template.template_id)
        if status.state != "Valide":
            raise ValueError(f"Le template par défaut est {status.state.lower()}.")
        return template

    def can_create_scenario(self) -> bool:
        template = self.get_default_template()
        return template is not None and self.get_template_validation_status(template.template_id).state == "Valide"

    def get_city_lambert(self, city_id: str) -> tuple[float, float]:
        if city_id in self._city_lambert_cache:
            return self._city_lambert_cache[city_id]
        city = self.get_city(city_id)
        point = tuple(float(value) for value in self._lambert_transformer.transform(city.longitude, city.latitude))
        self._city_lambert_cache[city_id] = point
        return point

    @staticmethod
    def _angle_degrees(vertex: tuple[float, float], first: tuple[float, float], second: tuple[float, float]) -> float:
        vector_a = (first[0] - vertex[0], first[1] - vertex[1])
        vector_b = (second[0] - vertex[0], second[1] - vertex[1])
        length_a, length_b = math.hypot(*vector_a), math.hypot(*vector_b)
        if length_a == 0 or length_b == 0:
            raise ValueError("Géométrie dégénérée.")
        cosine = (vector_a[0] * vector_b[0] + vector_a[1] * vector_b[1]) / (length_a * length_b)
        return math.degrees(math.acos(max(-1.0, min(1.0, cosine))))

    def get_triangle_geometry(self, triangle_id: str) -> TriangleGeometry:
        triangle = self.get_triangle(triangle_id)
        opening = self.get_city_lambert(triangle.opening_city_id)
        base = self.get_city_lambert(triangle.base_city_id)
        light = self.get_city_lambert(triangle.light_city_id)
        distance_ob_m = math.dist(opening, base)
        distance_ol_m = math.dist(opening, light)
        distance_bl_m = math.dist(base, light)
        # cross = 2 * signed area in Lambert m².
        cross = (base[0] - opening[0]) * (light[1] - opening[1]) - (base[1] - opening[1]) * (light[0] - opening[0])
        if distance_ob_m <= self._MIN_EDGE_LENGTH_M or distance_ol_m <= self._MIN_EDGE_LENGTH_M \
                or distance_bl_m <= self._MIN_EDGE_LENGTH_M or abs(cross) <= self._MIN_DOUBLE_AREA_M2:
            raise ValueError(f"Triangle {triangle_id} : géométrie dégénérée.")
        return TriangleGeometry(
            distance_ob_m / 1000.0,
            distance_ol_m / 1000.0,
            distance_bl_m / 1000.0,
            self._angle_degrees(opening, base, light),
            self._angle_degrees(base, opening, light),
            self._angle_degrees(light, opening, base),
            "CCW" if cross > 0 else "CW",
        )

    def validate(self) -> None:
        self._validate_collection(self.cities, "CITY", "city_id", "ville")
        self._validate_collection(self.triangles, "TRI", "triangle_id", "triangle")
        self._validate_collection(self.templates, "TPL", "template_id", "template")
        self._validate_unique_names(self.cities, "ville")
        self._validate_unique_names(self.templates, "template")
        triplets: set[tuple[str, str, str]] = set()
        for triangle in self.triangles.values():
            self._validate_note(triangle.note)
            self._validate_triangle_references(triangle.opening_city_id, triangle.base_city_id, triangle.light_city_id)
            triplet = (triangle.opening_city_id, triangle.base_city_id, triangle.light_city_id)
            if triplet in triplets:
                raise ValueError("Deux triangles possèdent le même triplet de villes.")
            triplets.add(triplet)
        for template in self.templates.values():
            self._validate_name(template.name, "de template")
            message = self._validate_template_ranks(template.triangle_ids_by_rank)
            if message:
                raise ValueError(f"Template {template.template_id} : {message}")
        if self.templates and self.default_template_id is None:
            raise ValueError("Un catalogue contenant des templates doit définir un template par défaut.")
        if self.default_template_id is not None and self.default_template_id not in self.templates:
            raise ValueError(f"Le template par défaut {self.default_template_id} est absent.")

    def _validate_collection(self, collection: dict[str, object], prefix: str, id_attribute: str, label: str) -> None:
        for item_id, item in collection.items():
            if item_id != getattr(item, id_attribute):
                raise ValueError(f"Clé {label} incohérente : {item_id}.")
            if not self._ID_PATTERNS[prefix].fullmatch(item_id):
                raise ValueError(f"Identifiant {label} invalide : {item_id}.")
            if isinstance(item, CatalogueCity):
                self._validate_name(item.name, "de ville")
                self._validate_coordinate(item.latitude, -90, 90, "La latitude")
                self._validate_coordinate(item.longitude, -180, 180, "La longitude")

    @staticmethod
    def _validate_unique_names(items: dict[str, object], label: str) -> None:
        names = [item.name.casefold() for item in items.values()]
        if len(names) != len(set(names)):
            raise ValueError(f"Les noms de {label} doivent être uniques.")
