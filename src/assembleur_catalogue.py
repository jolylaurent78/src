"""Modèle métier autonome du Catalogue, sans dépendance à l'interface ni au stockage."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from pathlib import PurePosixPath, PureWindowsPath

from pyproj import Transformer
from src.assembleur_catalogue_identity import (
    CATALOGUE_ID_KINDS,
    CATALOGUE_ID_KIND_ORDER,
    CatalogueIdProvider,
    UserCatalogueIdProvider,
    get_system_catalogue_id_number,
    is_catalogue_id,
    is_catalogue_city_id,
    is_catalogue_map_id,
    is_system_catalogue_id,
)


@dataclass
class CatalogueCity:
    city_id: str
    name: str
    latitude: float
    longitude: float
    archived: bool = False


@dataclass
class CatalogueBeacon:
    beacon_id: str
    city_id: str
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


@dataclass
class WorldRect:
    """Pose rectangulaire d'une carte dans le repère monde du Catalogue."""

    x0: float
    y0: float
    w: float
    h: float


def centered_world_rect(width: float, height: float) -> WorldRect:
    """Construit le repère local centré d'une carte."""
    return WorldRect(-float(width) / 2.0, -float(height) / 2.0, float(width), float(height))


@dataclass
class CatalogueMap:
    """Définition persistante d'une carte Catalogue, indépendante des assets physiques."""

    map_id: str
    name: str
    image_file: str
    calibration_points_file: str | None
    calibration_file: str | None
    projection: str | None
    default_world_rect: WorldRect
    default_scale_factor: float
    archived: bool = False
    description: str = ""
    calibration_city_ids: list[str] = field(default_factory=list)


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

    version = 5
    _MIN_EDGE_LENGTH_M = 1e-6
    _MIN_DOUBLE_AREA_M2 = 1e-6

    def __init__(
        self,
        *,
        id_provider: CatalogueIdProvider | None = None,
        provider: CatalogueIdProvider | None = None,
    ) -> None:
        if id_provider is not None and provider is not None:
            raise ValueError("Fournisseur d'identité Catalogue fourni deux fois.")
        self.version = Catalogue.version
        resolved_provider = id_provider if id_provider is not None else provider
        self.id_provider = resolved_provider if resolved_provider is not None else UserCatalogueIdProvider()
        self.id_counters: dict[str, int] = {kind: 0 for kind in CATALOGUE_ID_KIND_ORDER}
        self.cities: dict[str, CatalogueCity] = {}
        self.beacons: dict[str, CatalogueBeacon] = {}
        self.triangles: dict[str, CatalogueTriangle] = {}
        self.templates: dict[str, HypothesisTemplate] = {}
        self.maps: dict[str, CatalogueMap] = {}
        self.default_template_id: str | None = None
        self.default_map_id: str | None = None
        self.catalogue_reference_map_id: str | None = None
        self._city_lambert_cache: dict[str, tuple[float, float]] = {}
        self._lambert_transformer = Transformer.from_crs("EPSG:4326", "EPSG:2154", always_xy=True)

    def clone(self) -> "Catalogue":
        """Copie l'état métier sans propager les caches runtime."""
        cloned = Catalogue(id_provider=self.id_provider)
        cloned.version = self.version
        cloned.id_counters = dict(self.id_counters)
        cloned.cities = {
            city_id: CatalogueCity(city.city_id, city.name, city.latitude, city.longitude, city.archived)
            for city_id, city in self.cities.items()
        }
        cloned.beacons = {
            beacon_id: CatalogueBeacon(beacon.beacon_id, beacon.city_id, beacon.archived)
            for beacon_id, beacon in self.beacons.items()
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
        cloned.maps = {
            map_id: CatalogueMap(
                catalogue_map.map_id,
                catalogue_map.name,
                catalogue_map.image_file,
                catalogue_map.calibration_points_file,
                catalogue_map.calibration_file,
                catalogue_map.projection,
                WorldRect(
                    catalogue_map.default_world_rect.x0,
                    catalogue_map.default_world_rect.y0,
                    catalogue_map.default_world_rect.w,
                    catalogue_map.default_world_rect.h,
                ),
                catalogue_map.default_scale_factor,
                catalogue_map.archived,
                catalogue_map.description,
                list(catalogue_map.calibration_city_ids),
            )
            for map_id, catalogue_map in self.maps.items()
        }
        cloned.default_template_id = self.default_template_id
        cloned.default_map_id = self.default_map_id
        cloned.catalogue_reference_map_id = self.catalogue_reference_map_id
        return cloned

    def _allocate_system_id_number(self, kind: str) -> int:
        if kind not in CATALOGUE_ID_KINDS:
            raise ValueError(f"Type de compteur Catalogue inconnu : {kind!r}")
        self.id_counters[kind] += 1
        return self.id_counters[kind]

    def get_city(self, city_id: str) -> CatalogueCity:
        try:
            return self.cities[city_id]
        except KeyError as exc:
            raise KeyError(f"Ville inconnue : {city_id}") from exc

    def get_beacon(self, beacon_id: str) -> CatalogueBeacon:
        try:
            return self.beacons[beacon_id]
        except KeyError as exc:
            raise KeyError(f"Balise inconnue : {beacon_id}") from exc

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

    def get_map(self, map_id: str) -> CatalogueMap:
        try:
            return self.maps[map_id]
        except KeyError as exc:
            raise KeyError(f"Carte inconnue : {map_id}") from exc

    def iter_cities(self) -> tuple[CatalogueCity, ...]:
        return tuple(self.cities[item_id] for item_id in sorted(self.cities))

    def iter_beacons(self) -> tuple[CatalogueBeacon, ...]:
        return tuple(self.beacons[item_id] for item_id in sorted(self.beacons))

    def iter_triangles(self) -> tuple[CatalogueTriangle, ...]:
        return tuple(self.triangles[item_id] for item_id in sorted(self.triangles))

    def iter_templates(self) -> tuple[HypothesisTemplate, ...]:
        return tuple(self.templates[item_id] for item_id in sorted(self.templates))

    def iter_maps(self) -> tuple[CatalogueMap, ...]:
        return tuple(self.maps[item_id] for item_id in sorted(self.maps))

    @staticmethod
    def _validate_name(name: str, label: str) -> str:
        if not isinstance(name, str):
            raise ValueError(f"Le nom {label} doit être une chaîne.")
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
        city = CatalogueCity(self.id_provider.new_city_id(self), name, latitude, longitude, archived)
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

    @staticmethod
    def _validate_archived(value: bool, label: str) -> None:
        if not isinstance(value, bool):
            raise ValueError(f"{label} archived doit être un booléen.")

    def _validate_beacon_city_id(self, city_id: str, context_label: str) -> str:
        if not isinstance(city_id, str):
            raise ValueError(f"{context_label} : ville Catalogue introuvable {city_id!r}")
        if not city_id or city_id not in self.cities:
            raise ValueError(
                f"{context_label} : ville Catalogue introuvable {city_id}"
            )
        return city_id

    def _ensure_beacon_city_is_unique(self, city_id: str, current_beacon_id: str | None = None) -> None:
        if any(
            beacon_id != current_beacon_id and beacon.city_id == city_id
            for beacon_id, beacon in self.beacons.items()
        ):
            raise ValueError(f"La ville {city_id} possède déjà une balise.")

    def add_beacon(self, city_id: str) -> CatalogueBeacon:
        final_city_id = self._validate_beacon_city_id(city_id, "Nouvelle balise")
        self._ensure_beacon_city_is_unique(final_city_id)
        beacon_id = self.id_provider.new_beacon_id(self)
        beacon = CatalogueBeacon(beacon_id, final_city_id)
        self.beacons[beacon.beacon_id] = beacon
        return beacon

    def update_beacon(self, beacon_id: str, *, city_id: str | None = None,
                      archived: bool | None = None) -> CatalogueBeacon:
        beacon = self.get_beacon(beacon_id)
        final_city_id = (
            self._validate_beacon_city_id(city_id, beacon_id)
            if city_id is not None else beacon.city_id
        )
        self._ensure_beacon_city_is_unique(final_city_id, beacon_id)
        if archived is not None:
            self._validate_archived(archived, f"Balise {beacon_id}")
        beacon.city_id = final_city_id
        if archived is not None:
            beacon.archived = archived
        return beacon

    def delete_beacon(self, beacon_id: str) -> None:
        self.get_beacon(beacon_id)
        del self.beacons[beacon_id]

    def get_beacons_referencing_city(self, city_id: str) -> tuple[CatalogueBeacon, ...]:
        self.get_city(city_id)
        return tuple(beacon for beacon in self.iter_beacons() if beacon.city_id == city_id)

    def get_triangles_referencing_city(self, city_id: str) -> tuple[CatalogueTriangle, ...]:
        self.get_city(city_id)
        return tuple(triangle for triangle in self.iter_triangles() if city_id in (
            triangle.opening_city_id, triangle.base_city_id, triangle.light_city_id))

    def delete_city(self, city_id: str) -> None:
        beacon_references = self.get_beacons_referencing_city(city_id)
        if beacon_references:
            raise ValueError(
                f"Impossible de supprimer {city_id} : la ville est référencée par "
                f"{len(beacon_references)} balise(s)."
            )
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
        triangle = CatalogueTriangle(self.id_provider.new_triangle_id(self), note, opening_city_id, base_city_id, light_city_id, archived)
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
        template = HypothesisTemplate(self.id_provider.new_template_id(self), name, description, archived)
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
            replacement = next(iter(self.iter_templates()), None)
            self.default_template_id = replacement.template_id if replacement is not None else None

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

    @staticmethod
    def _validate_logical_asset_reference(value: object, label: str, *, required: bool) -> str | None:
        if value is None:
            if required:
                raise ValueError(f"{label} est obligatoire.")
            return None
        if not isinstance(value, str):
            raise ValueError(f"{label} doit être une chaîne ou null.")
        if not value or value != value.strip():
            raise ValueError(f"{label} ne peut pas être vide ou contenir des espaces de bord.")
        if "\\" in value:
            raise ValueError(f"{label} doit utiliser une référence logique avec des séparateurs '/'.")
        posix_path = PurePosixPath(value)
        windows_path = PureWindowsPath(value)
        raw_parts = value.split("/")
        if (
            posix_path.is_absolute()
            or windows_path.is_absolute()
            or windows_path.drive
            or any(part in ("", ".", "..") for part in raw_parts)
            or ":" in value
        ):
            raise ValueError(f"{label} doit être une référence relative sûre.")
        return value

    @staticmethod
    def _validate_map_number(value: object, label: str, *, strictly_positive: bool = False) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{label} doit être un nombre.")
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"{label} doit être fini.")
        if strictly_positive and number <= 0:
            raise ValueError(f"{label} doit être strictement positif.")
        return number

    @classmethod
    def _validate_world_rect(cls, value: object) -> WorldRect:
        if not isinstance(value, WorldRect):
            raise ValueError("default_world_rect doit être un WorldRect.")
        x0 = cls._validate_map_number(value.x0, "default_world_rect.x0")
        y0 = cls._validate_map_number(value.y0, "default_world_rect.y0")
        width = cls._validate_map_number(value.w, "default_world_rect.w", strictly_positive=True)
        height = cls._validate_map_number(value.h, "default_world_rect.h", strictly_positive=True)
        return WorldRect(x0, y0, width, height)

    @classmethod
    def _validate_catalogue_map_fields(
        cls,
        *,
        map_id: object,
        name: object,
        image_file: object,
        calibration_points_file: object,
        calibration_file: object,
        projection: object,
        default_world_rect: object,
        default_scale_factor: object,
        archived: object,
        description: object,
        calibration_city_ids: object,
    ) -> tuple[str, str, str, str | None, str | None, str | None, WorldRect, float, bool, str, list[str]]:
        if not isinstance(map_id, str) or not is_catalogue_id(map_id, "map"):
            raise ValueError(f"Identifiant carte invalide : {map_id!r}.")
        cleaned_name = cls._validate_name(name, "de carte")
        image = cls._validate_logical_asset_reference(image_file, "image_file", required=True)
        points = cls._validate_logical_asset_reference(
            calibration_points_file, "calibration_points_file", required=False
        )
        calibration = cls._validate_logical_asset_reference(
            calibration_file, "calibration_file", required=False
        )
        if calibration is None:
            if projection is not None:
                raise ValueError("projection doit être null sans calibration_file.")
            final_projection = None
        else:
            if projection not in (None, "EPSG:2154"):
                raise ValueError("projection doit être null ou 'EPSG:2154' avec calibration_file.")
            final_projection = projection
        rect = cls._validate_world_rect(default_world_rect)
        scale = cls._validate_map_number(
            default_scale_factor, "default_scale_factor", strictly_positive=True
        )
        if scale > 20.0:
            raise ValueError("default_scale_factor doit etre inferieur ou egal a 20.")
        if not isinstance(archived, bool):
            raise ValueError("Carte archived doit être un booléen.")
        if not isinstance(description, str):
            raise ValueError("description doit être une chaîne.")
        if not isinstance(calibration_city_ids, list):
            raise ValueError("calibration_city_ids doit être une liste.")
        final_city_ids: list[str] = []
        seen_city_ids: set[str] = set()
        for city_id in calibration_city_ids:
            if not is_catalogue_city_id(city_id):
                raise ValueError(f"Identifiant ville de calibration invalide : {city_id!r}.")
            if city_id in seen_city_ids:
                raise ValueError(f"Ville de calibration dupliquée : {city_id}.")
            seen_city_ids.add(city_id)
            final_city_ids.append(city_id)
        return (
            map_id,
            cleaned_name,
            image,
            points,
            calibration,
            final_projection,
            rect,
            scale,
            archived,
            description,
            final_city_ids,
        )

    def _validate_catalogue_map(self, catalogue_map: CatalogueMap) -> None:
        self._validate_catalogue_map_fields(
            map_id=catalogue_map.map_id,
            name=catalogue_map.name,
            image_file=catalogue_map.image_file,
            calibration_points_file=catalogue_map.calibration_points_file,
            calibration_file=catalogue_map.calibration_file,
            projection=catalogue_map.projection,
            default_world_rect=catalogue_map.default_world_rect,
            default_scale_factor=catalogue_map.default_scale_factor,
            archived=catalogue_map.archived,
            description=catalogue_map.description,
            calibration_city_ids=catalogue_map.calibration_city_ids,
        )

    def add_map(
        self,
        *,
        name: str,
        image_file: str,
        calibration_points_file: str | None = None,
        calibration_file: str | None = None,
        projection: str | None = None,
        default_world_rect: WorldRect,
        default_scale_factor: float,
        archived: bool = False,
        description: str = "",
        calibration_city_ids: list[str] | None = None,
    ) -> str:
        """Crée une carte après validation complète, avant toute allocation SYS."""
        # L'identité provisoire ne sert qu'à valider les autres champs avant
        # l'allocation réelle, qui reste exclusivement du ressort du provider.
        provisional_id = "MAP-SYS-000001"
        (
            _map_id,
            final_name,
            final_image,
            final_points,
            final_calibration,
            final_projection,
            final_rect,
            final_scale,
            final_archived,
            final_description,
            final_city_ids,
        ) = self._validate_catalogue_map_fields(
            map_id=provisional_id,
            name=name,
            image_file=image_file,
            calibration_points_file=calibration_points_file,
            calibration_file=calibration_file,
            projection=projection,
            default_world_rect=default_world_rect,
            default_scale_factor=default_scale_factor,
            archived=archived,
            description=description,
            calibration_city_ids=[] if calibration_city_ids is None else calibration_city_ids,
        )
        self._validate_calibration_city_ids_exist(final_city_ids)
        self._ensure_unique_name(final_name, self.maps, None, "name")
        map_id = self.id_provider.new_map_id(self)
        catalogue_map = CatalogueMap(
            map_id,
            final_name,
            final_image,
            final_points,
            final_calibration,
            final_projection,
            final_rect,
            final_scale,
            final_archived,
            final_description,
            final_city_ids,
        )
        self.maps[map_id] = catalogue_map
        return map_id

    def update_map(
        self,
        map_id: str,
        *,
        name: str | None = None,
        image_file: str | None = None,
        calibration_points_file: str | None | object = ...,
        calibration_file: str | None | object = ...,
        projection: str | None | object = ...,
        default_world_rect: WorldRect | None = None,
        default_scale_factor: float | None = None,
        archived: bool | None = None,
        description: str | None = None,
        calibration_city_ids: list[str] | None = None,
    ) -> CatalogueMap:
        catalogue_map = self.get_map(map_id)
        final_name = catalogue_map.name if name is None else name
        final_image = catalogue_map.image_file if image_file is None else image_file
        final_points = catalogue_map.calibration_points_file if calibration_points_file is ... else calibration_points_file
        final_calibration = catalogue_map.calibration_file if calibration_file is ... else calibration_file
        final_projection = catalogue_map.projection if projection is ... else projection
        final_rect = catalogue_map.default_world_rect if default_world_rect is None else default_world_rect
        final_scale = catalogue_map.default_scale_factor if default_scale_factor is None else default_scale_factor
        final_archived = catalogue_map.archived if archived is None else archived
        final_description = catalogue_map.description if description is None else description
        final_city_ids = catalogue_map.calibration_city_ids if calibration_city_ids is None else calibration_city_ids
        (
            _validated_id,
            validated_name,
            validated_image,
            validated_points,
            validated_calibration,
            validated_projection,
            validated_rect,
            validated_scale,
            validated_archived,
            validated_description,
            validated_city_ids,
        ) = self._validate_catalogue_map_fields(
            map_id=map_id,
            name=final_name,
            image_file=final_image,
            calibration_points_file=final_points,
            calibration_file=final_calibration,
            projection=final_projection,
            default_world_rect=final_rect,
            default_scale_factor=final_scale,
            archived=final_archived,
            description=final_description,
            calibration_city_ids=final_city_ids,
        )
        self._validate_calibration_city_ids_exist(validated_city_ids)
        self._ensure_unique_name(validated_name, self.maps, map_id, "name")
        catalogue_map.name = validated_name
        catalogue_map.image_file = validated_image
        catalogue_map.calibration_points_file = validated_points
        catalogue_map.calibration_file = validated_calibration
        catalogue_map.projection = validated_projection
        catalogue_map.default_world_rect = validated_rect
        catalogue_map.default_scale_factor = validated_scale
        catalogue_map.archived = validated_archived
        catalogue_map.description = validated_description
        catalogue_map.calibration_city_ids = validated_city_ids
        if validated_archived and self.default_map_id == map_id:
            self.default_map_id = None
        if validated_archived and self.catalogue_reference_map_id == map_id:
            self.catalogue_reference_map_id = None
        return catalogue_map

    def archive_map(self, map_id: str) -> CatalogueMap:
        return self.update_map(map_id, archived=True)

    def _validate_calibration_city_ids_exist(self, city_ids: list[str]) -> None:
        for city_id in city_ids:
            if city_id not in self.cities:
                raise ValueError(f"La ville de calibration {city_id} est absente.")

    def delete_map(self, map_id: str) -> None:
        self.get_map(map_id)
        if map_id == self.default_map_id:
            raise ValueError("La carte par défaut ne peut pas être supprimée.")
        if map_id == self.catalogue_reference_map_id:
            raise ValueError("La carte de référence Catalogue ne peut pas être supprimée.")
        del self.maps[map_id]

    def set_default_map(self, map_id: str) -> None:
        catalogue_map = self.get_map(map_id)
        if catalogue_map.archived:
            raise ValueError(f"La carte archivée {map_id} ne peut pas être la carte par défaut.")
        self.default_map_id = map_id

    def set_catalogue_reference_map(self, map_id: str) -> None:
        catalogue_map = self.get_map(map_id)
        if catalogue_map.calibration_file is None or catalogue_map.projection is None:
            raise ValueError(f"La carte de référence Catalogue {map_id} doit être calibrée.")
        self.catalogue_reference_map_id = map_id

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
        self._validate_collection(self.cities, "city", "city_id", "ville")
        self._validate_collection(self.beacons, "beacon", "beacon_id", "balise")
        self._validate_collection(self.triangles, "triangle", "triangle_id", "triangle")
        self._validate_collection(self.templates, "template", "template_id", "template")
        self._validate_collection(self.maps, "map", "map_id", "carte")
        self._validate_unique_names(self.cities, "ville")
        self._validate_unique_names(self.templates, "template")
        self._validate_unique_names(self.maps, "carte")
        beacon_city_ids: set[str] = set()
        for beacon in self.beacons.values():
            city_id = self._validate_beacon_city_id(beacon.city_id, beacon.beacon_id)
            if city_id in beacon_city_ids:
                raise ValueError(f"Plusieurs balises référencent la ville {city_id}.")
            beacon_city_ids.add(city_id)
            self._validate_archived(beacon.archived, f"Balise {beacon.beacon_id}")
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
        for catalogue_map in self.maps.values():
            self._validate_catalogue_map(catalogue_map)
            for city_id in catalogue_map.calibration_city_ids:
                if city_id not in self.cities:
                    raise ValueError(
                        f"La ville de calibration {city_id} de la carte {catalogue_map.map_id} est absente."
                    )
        if self.default_map_id is not None:
            if not isinstance(self.default_map_id, str):
                raise ValueError("defaultMapId doit être une chaîne ou null.")
            if not is_catalogue_map_id(self.default_map_id):
                raise ValueError(f"defaultMapId invalide : {self.default_map_id!r}.")
            if self.default_map_id not in self.maps:
                raise ValueError(f"La carte par défaut {self.default_map_id} est absente.")
            default_map = self.maps[self.default_map_id]
            if default_map.archived:
                raise ValueError(f"La carte par défaut {self.default_map_id} est archivée.")
        if self.catalogue_reference_map_id is not None:
            if not isinstance(self.catalogue_reference_map_id, str):
                raise ValueError("catalogueReferenceMapId doit être une chaîne ou null.")
            if not is_catalogue_map_id(self.catalogue_reference_map_id):
                raise ValueError(f"catalogueReferenceMapId invalide : {self.catalogue_reference_map_id!r}.")
            try:
                reference_map = self.maps[self.catalogue_reference_map_id]
            except KeyError as exc:
                raise ValueError(
                    f"La carte de référence Catalogue {self.catalogue_reference_map_id} est absente."
                ) from exc
            if reference_map.calibration_file is None or reference_map.projection is None:
                raise ValueError(
                    f"La carte de référence Catalogue {self.catalogue_reference_map_id} doit être calibrée."
                )

        self._validate_id_counters()

    def _validate_id_counters(self) -> None:
        if not isinstance(self.id_counters, dict):
            raise ValueError("Les compteurs d'identifiants Catalogue doivent former un dictionnaire.")
        counter_keys = set(self.id_counters)
        expected_keys = set(CATALOGUE_ID_KIND_ORDER)
        if counter_keys != expected_keys:
            missing = sorted(expected_keys - counter_keys)
            unexpected = sorted(counter_keys - expected_keys)
            details = []
            if missing:
                details.append(f"clés manquantes : {', '.join(missing)}")
            if unexpected:
                details.append(f"clés inconnues : {', '.join(unexpected)}")
            raise ValueError(
                "Les compteurs d'identifiants Catalogue sont incomplets ou invalides"
                f" ({'; '.join(details)})."
            )
        collections = {
            "city": self.cities,
            "beacon": self.beacons,
            "triangle": self.triangles,
            "template": self.templates,
            "map": self.maps,
        }
        for kind in CATALOGUE_ID_KIND_ORDER:
            counter = self.id_counters[kind]
            if isinstance(counter, bool) or not isinstance(counter, int) or counter < 0:
                raise ValueError(
                    f"Compteur d'identifiants Catalogue invalide pour {kind} : {counter!r}."
                )
            for item_id in collections[kind]:
                if is_system_catalogue_id(item_id):
                    number = get_system_catalogue_id_number(item_id, kind)
                    if counter < number:
                        raise ValueError(
                            f"Compteur d'identifiants Catalogue incohérent pour {kind} : "
                            f"{counter} est inférieur à l'identifiant SYS présent {item_id}."
                        )

    def _validate_collection(self, collection: dict[str, object], prefix: str, id_attribute: str, label: str) -> None:
        for item_id, item in collection.items():
            if item_id != getattr(item, id_attribute):
                raise ValueError(f"Clé {label} incohérente : {item_id}.")
            if not is_catalogue_id(item_id, prefix):
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
