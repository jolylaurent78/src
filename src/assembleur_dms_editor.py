"""Éditeur DMS générique pour coordonnées géographiques décimales."""

from __future__ import annotations

import re
import tkinter as tk
from tkinter import ttk
from typing import Callable, Literal


CoordinateType = Literal["latitude", "longitude"]


class DmsCoordinateEditor(ttk.Frame):
    """Édite une coordonnée DMS bornée et expose uniquement une valeur décimale."""

    _HEMISPHERE_ALIASES = {
        "n": "N",
        "nord": "N",
        "s": "S",
        "sud": "S",
        "e": "E",
        "est": "E",
        "w": "W",
        "o": "W",
        "ouest": "W",
    }
    _HEMISPHERE_RE = r"(?:nord|sud|ouest|est|[NSEWO])\b"
    _DMS_RE = re.compile(
        r"(?P<degrees>\d{1,3})\s*[°º]\s*"
        r"(?P<minutes>\d{1,2})\s*['′]\s*"
        r"(?P<seconds>\d{1,2}(?:[.,]\d+)?)\s*[\"″]?\s*"
        rf"(?P<hemisphere>{_HEMISPHERE_RE})",
        re.IGNORECASE,
    )
    _SPACE_DMS_RE = re.compile(
        r"(?P<degrees>\d{1,3})\s+(?P<minutes>\d{1,2})\s+"
        r"(?P<seconds>\d{1,2}(?:[.,]\d+)?)\s*(?P<hemisphere>"
        + _HEMISPHERE_RE
        + r")",
        re.IGNORECASE,
    )
    _DECIMAL_PAIR_RE = re.compile(
        r"^\s*(?P<latitude>[+-]?\d+(?:[.,]\d+)?)\s*[,;]\s*"
        r"(?P<longitude>[+-]?\d+(?:[.,]\d+)?)\s*$"
    )

    def __init__(
        self,
        parent,
        *,
        coordinate_type: CoordinateType,
        on_change: Callable[[], None] | None = None,
        **kwargs,
    ):
        super().__init__(parent, **kwargs)
        if coordinate_type not in ("latitude", "longitude"):
            raise ValueError("coordinate_type doit être 'latitude' ou 'longitude'.")
        self.coordinate_type = coordinate_type
        self._on_change = on_change
        self._is_setting = False
        self._maximum_degrees = 90 if coordinate_type == "latitude" else 180
        hemispheres = ("N", "S") if coordinate_type == "latitude" else ("E", "W")

        self._hemisphere_var = tk.StringVar(value=hemispheres[0])
        self._degrees_var = tk.StringVar(value="0")
        self._minutes_var = tk.StringVar(value="0")
        self._seconds_var = tk.StringVar(value="0")
        self._last_valid_decimal = 0.0

        self._hemisphere = ttk.Combobox(self, values=hemispheres, width=3, state="readonly",
                                        textvariable=self._hemisphere_var)
        self._hemisphere.grid(row=0, column=0, sticky="w")
        self._degrees = ttk.Spinbox(self, from_=0, to=self._maximum_degrees, width=4,
                                    textvariable=self._degrees_var, wrap=False, command=self._commit_user_edit)
        self._degrees.grid(row=0, column=1, padx=(6, 0))
        ttk.Label(self, text="°").grid(row=0, column=2, sticky="w")
        self._minutes = ttk.Spinbox(self, from_=0, to=59, width=3,
                                    textvariable=self._minutes_var, wrap=False, command=self._commit_user_edit)
        self._minutes.grid(row=0, column=3, padx=(4, 0))
        ttk.Label(self, text="'").grid(row=0, column=4, sticky="w")
        self._seconds = ttk.Spinbox(self, from_=0, to=59, width=3,
                                    textvariable=self._seconds_var, wrap=False, command=self._commit_user_edit)
        self._seconds.grid(row=0, column=5, padx=(4, 0))
        ttk.Label(self, text='"').grid(row=0, column=6, sticky="w")

        self._hemisphere.bind("<<ComboboxSelected>>", lambda _event: self._commit_user_edit())
        for spinbox in (self._degrees, self._minutes, self._seconds):
            spinbox.bind("<FocusOut>", lambda _event: self._commit_user_edit())
            spinbox.bind("<Return>", lambda _event: self._commit_user_edit())

    def set_decimal(self, value: float) -> None:
        """Affiche une valeur décimale en DMS, avec secondes tronquées."""
        numeric = self._validated_decimal(value, self.coordinate_type)
        positive, negative = self._hemispheres_for(self.coordinate_type)
        total_seconds = int(abs(numeric) * 3600)  # troncature obligatoire
        degrees, remainder = divmod(total_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        self._is_setting = True
        try:
            self._hemisphere_var.set(positive if numeric >= 0 else negative)
            self._degrees_var.set(str(degrees))
            self._minutes_var.set(str(minutes))
            self._seconds_var.set(str(seconds))
            self._last_valid_decimal = numeric
        finally:
            self._is_setting = False

    def get_decimal(self) -> float:
        try:
            return self._current_decimal()
        except ValueError:
            return self._last_valid_decimal

    @classmethod
    def parse_coordinate(cls, text: str, coordinate_type: CoordinateType) -> float:
        """Convertit une coordonnée DMS ou décimale en float, avec secondes tronquées."""
        raw = str(text or "").strip()
        match = cls._DMS_RE.fullmatch(raw) or cls._SPACE_DMS_RE.fullmatch(raw)
        if match:
            parts = match.groupdict()
            return cls._dms_parts_to_decimal(
                parts["degrees"], parts["minutes"], parts["seconds"], parts["hemisphere"], coordinate_type
            )
        try:
            return cls._validated_decimal(float(raw.replace(",", ".")), coordinate_type)
        except ValueError as exc:
            raise ValueError(f"Coordonnée {coordinate_type} invalide : {text!r}") from exc

    @classmethod
    def parse_coordinate_pair(cls, text: str) -> tuple[float, float]:
        """Reconnaît une paire Wikipédia/ASCII DMS ou une paire décimale lat, lon."""
        raw = str(text or "").strip()
        decimal_match = cls._DECIMAL_PAIR_RE.fullmatch(raw)
        if decimal_match:
            return (
                cls._validated_decimal(float(decimal_match["latitude"].replace(",", ".")), "latitude"),
                cls._validated_decimal(float(decimal_match["longitude"].replace(",", ".")), "longitude"),
            )
        matches = list(cls._DMS_RE.finditer(raw)) or list(cls._SPACE_DMS_RE.finditer(raw))
        if len(matches) != 2:
            raise ValueError("Formats acceptés : DMS (N/S, E/W) ou latitude, longitude décimales.")
        values: dict[str, float] = {}
        for match in matches:
            parts = match.groupdict()
            hemisphere = cls._normalize_hemisphere(parts["hemisphere"])
            coordinate_type: CoordinateType = "latitude" if hemisphere in ("N", "S") else "longitude"
            if coordinate_type in values:
                raise ValueError("Une latitude et une longitude sont attendues.")
            values[coordinate_type] = cls._dms_parts_to_decimal(
                parts["degrees"], parts["minutes"], parts["seconds"], hemisphere, coordinate_type
            )
        if set(values) != {"latitude", "longitude"}:
            raise ValueError("Une latitude et une longitude sont attendues.")
        return values["latitude"], values["longitude"]

    @classmethod
    def _dms_parts_to_decimal(
        cls, degrees_raw: str, minutes_raw: str, seconds_raw: str, hemisphere: str, coordinate_type: CoordinateType
    ) -> float:
        degrees, minutes = int(degrees_raw), int(minutes_raw)
        seconds = int(float(seconds_raw.replace(",", ".")))  # troncature, jamais arrondi
        positive, negative = cls._hemispheres_for(coordinate_type)
        hemisphere = cls._normalize_hemisphere(hemisphere)
        maximum = 90 if coordinate_type == "latitude" else 180
        if (
            hemisphere not in (positive, negative)
            or not 0 <= degrees <= maximum
            or not 0 <= minutes <= 59
            or not 0 <= seconds <= 59
            or (degrees == maximum and (minutes != 0 or seconds != 0))
        ):
            raise ValueError("Coordonnée DMS hors limites.")
        sign = 1 if hemisphere == positive else -1
        return sign * (degrees + minutes / 60 + seconds / 3600)

    @staticmethod
    def _hemispheres_for(coordinate_type: CoordinateType) -> tuple[str, str]:
        return ("N", "S") if coordinate_type == "latitude" else ("E", "W")

    @classmethod
    def _normalize_hemisphere(cls, hemisphere: str) -> str:
        return cls._HEMISPHERE_ALIASES.get(hemisphere.lower(), "")

    @staticmethod
    def _validated_decimal(value: float, coordinate_type: CoordinateType) -> float:
        maximum = 90 if coordinate_type == "latitude" else 180
        numeric = float(value)
        if not -maximum <= numeric <= maximum:
            raise ValueError(f"Coordonnée {coordinate_type} hors limites.")
        return numeric

    def _current_decimal(self) -> float:
        degrees = int(self._degrees_var.get())
        minutes = int(self._minutes_var.get())
        seconds = int(self._seconds_var.get())
        positive, negative = self._hemispheres_for(self.coordinate_type)
        hemisphere = self._hemisphere_var.get()
        if (
            hemisphere not in (positive, negative)
            or not 0 <= degrees <= self._maximum_degrees
            or not 0 <= minutes <= 59
            or not 0 <= seconds <= 59
            or (degrees == self._maximum_degrees and (minutes != 0 or seconds != 0))
        ):
            raise ValueError("Coordonnée DMS hors limites.")
        sign = 1 if hemisphere == positive else -1
        return sign * (degrees + minutes / 60 + seconds / 3600)

    def _commit_user_edit(self) -> None:
        if self._is_setting:
            return
        try:
            decimal = self._current_decimal()
        except (TypeError, ValueError):
            self.set_decimal(self._last_valid_decimal)
            return
        self._last_valid_decimal = decimal
        if self._on_change is not None:
            self._on_change()
