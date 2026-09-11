"""Lecture stricte, indépendante de l'UI, des documents AlgoSimulator Traces V2."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from types import MappingProxyType
from typing import Mapping


_GEOMETRIES = {
    "point": ("x_l93", "y_l93"),
    "symbol": ("x_l93", "y_l93", "source"),
    "circle": ("center_x_l93", "center_y_l93", "radius_km"),
    "arc": ("center_x_l93", "center_y_l93", "radius_km", "start_azimuth_deg", "rotation_deg"),
    "line_between_points": ("x1_l93", "y1_l93", "x2_l93", "y2_l93"),
    "line_image_azimuth": ("x_l93", "y_l93", "azimuth_deg"),
    "vertical_image_line": ("x_l93", "y_l93"),
    "horizontal_image_line": ("x_l93", "y_l93"),
    "segment": ("x1_l93", "y1_l93", "x2_l93", "y2_l93"),
}
_GRAPHICS_KEYS = {"name", "color_bgr", "width", "style", "show_name", "visible", "tags", "tooltips", "scenario_tooltips"}


@dataclass(frozen=True)
class GeometricLayerGraphics:
    name: str
    color_bgr: tuple[float, float, float]
    width: float
    style: str
    show_name: bool | None
    visible: bool
    tags: Mapping[str, str]
    tooltips: tuple[str, ...]
    scenario_tooltips: tuple[str, ...]


@dataclass(frozen=True)
class GeometricLayerTrace:
    geometry_type: str
    geometry: Mapping[str, object]
    graphics: GeometricLayerGraphics


@dataclass(frozen=True)
class GeometricLayerModule:
    module_id: str
    label: str
    traces: tuple[GeometricLayerTrace, ...]


@dataclass(frozen=True)
class GeometricLayerDocument:
    source: str
    algorithm: str
    segment: str | int | float
    scope: Mapping[str, str]
    modules: tuple[GeometricLayerModule, ...]


def _mapping(value: object, label: str) -> dict:
    if not isinstance(value, dict):
        raise ValueError(f"{label} doit être un objet.")
    return value


def _exact_keys(value: dict, label: str, expected: set[str]) -> None:
    if set(value) != expected:
        missing = sorted(expected - set(value))
        extra = sorted(set(value) - expected)
        details = []
        if missing:
            details.append(f"champs absents : {', '.join(missing)}")
        if extra:
            details.append(f"champs inconnus : {', '.join(extra)}")
        raise ValueError(f"{label} a une structure invalide ({'; '.join(details)}).")


def _string(value: object, label: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} doit être une chaîne.")
    return value


def _number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{label} doit être un nombre fini.")
    return float(value)


def _string_list(value: object, label: str) -> tuple[str, ...]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ValueError(f"{label} doit être une liste de chaînes.")
    return tuple(value)


def _graphics_from_dict(value: object, label: str) -> GeometricLayerGraphics:
    raw = _mapping(value, label)
    _exact_keys(raw, label, _GRAPHICS_KEYS)
    color = raw["color_bgr"]
    if not isinstance(color, list) or len(color) != 3:
        raise ValueError(f"{label}.color_bgr doit contenir exactement trois nombres finis.")
    tags = raw["tags"]
    if not isinstance(tags, dict) or any(not isinstance(key, str) or not isinstance(item, str) for key, item in tags.items()):
        raise ValueError(f"{label}.tags doit être un objet de chaînes.")
    if raw["show_name"] is not None and not isinstance(raw["show_name"], bool):
        raise ValueError(f"{label}.show_name doit être un booléen ou null.")
    if not isinstance(raw["visible"], bool):
        raise ValueError(f"{label}.visible doit être un booléen.")
    return GeometricLayerGraphics(
        _string(raw["name"], f"{label}.name"),
        tuple(_number(item, f"{label}.color_bgr[{index}]") for index, item in enumerate(color)),
        _number(raw["width"], f"{label}.width"),
        _string(raw["style"], f"{label}.style"), raw["show_name"], raw["visible"],
        MappingProxyType(dict(tags)), _string_list(raw["tooltips"], f"{label}.tooltips"),
        _string_list(raw["scenario_tooltips"], f"{label}.scenario_tooltips"),
    )


def _trace_from_dict(value: object, label: str) -> GeometricLayerTrace:
    raw = _mapping(value, label)
    _exact_keys(raw, label, {"geometry", "graphics"})
    geometry = _mapping(raw["geometry"], f"{label}.geometry")
    geometry_type = _string(geometry.get("type"), f"{label}.geometry.type")
    fields = _GEOMETRIES.get(geometry_type)
    if fields is None:
        raise ValueError(f"{label}.geometry.type inconnu : {geometry_type!r}.")
    _exact_keys(geometry, f"{label}.geometry", {"type", *fields})
    converted: dict[str, object] = {"type": geometry_type}
    for field in fields:
        if geometry_type == "symbol" and field == "source":
            converted[field] = _string(geometry[field], f"{label}.geometry.{field}")
        else:
            converted[field] = _number(geometry[field], f"{label}.geometry.{field}")
    return GeometricLayerTrace(geometry_type, MappingProxyType(converted), _graphics_from_dict(raw["graphics"], f"{label}.graphics"))


def geometric_layer_document_from_dict(data: object) -> GeometricLayerDocument:
    root = _mapping(data, "document Traces")
    _exact_keys(root, "document Traces", {"schema_version", "source", "algorithm", "segment", "scope", "modules"})
    if root["schema_version"] != 2 or isinstance(root["schema_version"], bool):
        raise ValueError("document Traces.schema_version doit être 2.")
    segment = root["segment"]
    if isinstance(segment, bool) or not isinstance(segment, (str, int, float)) or (isinstance(segment, float) and not math.isfinite(segment)):
        raise ValueError("document Traces.segment doit être une chaîne ou un nombre fini.")
    scope = _mapping(root["scope"], "document Traces.scope")
    scope_type = _string(scope.get("type"), "document Traces.scope.type")
    expected_scope = {"type", "scenario"} if scope_type == "scenario" else {"type"} if scope_type == "automatic_aggregation" else None
    if expected_scope is None:
        raise ValueError(f"document Traces.scope.type inconnu : {scope_type!r}.")
    _exact_keys(scope, "document Traces.scope", expected_scope)
    if "scenario" in scope:
        _string(scope["scenario"], "document Traces.scope.scenario")
    modules = root["modules"]
    if not isinstance(modules, list):
        raise ValueError("document Traces.modules doit être une liste.")
    parsed_modules = []
    for index, raw_module in enumerate(modules):
        label = f"modules[{index}]"
        module = _mapping(raw_module, label)
        _exact_keys(module, label, {"id", "label", "traces"})
        traces = module["traces"]
        if not isinstance(traces, list):
            raise ValueError(f"{label}.traces doit être une liste.")
        parsed_modules.append(GeometricLayerModule(
            _string(module["id"], f"{label}.id"), _string(module["label"], f"{label}.label"),
            tuple(_trace_from_dict(trace, f"{label}.traces[{trace_index}]") for trace_index, trace in enumerate(traces)),
        ))
    return GeometricLayerDocument(
        _string(root["source"], "document Traces.source"), _string(root["algorithm"], "document Traces.algorithm"),
        segment, MappingProxyType(dict(scope)), tuple(parsed_modules),
    )


def load_geometric_layer_document(path: str | Path) -> GeometricLayerDocument:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Fichier Traces introuvable : {source}")
    try:
        data = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"JSON Traces invalide : {exc.msg}.") from exc
    return geometric_layer_document_from_dict(data)
