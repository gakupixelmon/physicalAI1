"""Validated application settings for the physicalAI runtime.

The YAML file is intentionally kept at the application boundary.  Runtime
components receive typed settings objects instead of independently reading
possibly different keys from ``settings.yaml``.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.parse import urlparse

import yaml


class SettingsError(ValueError):
    """Raised when the YAML file does not match the public settings schema."""


def _section(data: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = data.get(name)
    if not isinstance(value, Mapping):
        raise SettingsError(f"settings.{name} must be a mapping")
    return value


def _number(value: Any, key: str, *, integer: bool = False) -> float | int:
    if isinstance(value, bool):
        raise SettingsError(f"settings.{key} must be numeric")
    try:
        converted = int(value) if integer else float(value)
    except (TypeError, ValueError) as exc:
        raise SettingsError(f"settings.{key} must be numeric") from exc
    if not isfinite(float(converted)):
        raise SettingsError(f"settings.{key} must be finite")
    return converted


def _positive(value: Any, key: str, *, allow_zero: bool = False) -> float:
    converted = float(_number(value, key))
    if converted < 0 or (converted == 0 and not allow_zero):
        qualifier = "non-negative" if allow_zero else "positive"
        raise SettingsError(f"settings.{key} must be {qualifier}")
    return converted


def _port(value: Any, key: str) -> int:
    converted = int(_number(value, key, integer=True))
    if not 1 <= converted <= 65535:
        raise SettingsError(f"settings.{key} must be in the range 1..65535")
    return converted


def _point(value: Any, key: str) -> tuple[float, float]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise SettingsError(f"settings.{key} must be a 2D sequence")
    if len(value) != 2:
        raise SettingsError(f"settings.{key} must be a 2D point")
    return (float(_number(value[0], f"{key}[0]")), float(_number(value[1], f"{key}[1]")))


def _string(value: Any, key: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SettingsError(f"settings.{key} must be a non-empty string")
    return value.strip()


def _bool(value: Any, key: str) -> bool:
    if not isinstance(value, bool):
        raise SettingsError(f"settings.{key} must be a boolean")
    return value


def _dictionary_setting(value: Any) -> str | int:
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise SettingsError("settings.camera.aruco_dictionary must be a name or integer")
    if isinstance(value, str) and not value.strip():
        raise SettingsError("settings.camera.aruco_dictionary must not be empty")
    return value.strip() if isinstance(value, str) else int(value)


@dataclass(frozen=True)
class CameraSettings:
    stream_url: str | None = None
    device_index: int | None = None
    avfoundation_device: str | None = None
    avfoundation_framerate: float = 30.0
    height_m: float = 2.0
    aruco_dictionary: str | int = "DICT_4X4_50"
    field_marker_ids: tuple[int, ...] = (0, 1, 2, 3)
    field_size_m: float = 2.0
    marker_size_m: float = 0.15
    field_marker_positions_m: dict[int, tuple[float, float]] | None = None
    camera_center_floor_m: tuple[float, float] | None = None
    drone_marker_id: int = 10
    target_model: str = "yolov8s-worldv2.pt"
    target_classes: tuple[str, ...] = ("start point", "delivery destination")
    target_confidence: float = 0.25
    target_aggregation_frames: int = 1
    required_target_classes: tuple[str, ...] = ()
    start_class: str = "start point"
    start_label: str = "start"
    start_from_drone_marker: bool = False
    start_marker_height_m: float = 0.0
    max_frame_reads: int = 4

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "CameraSettings":
        stream_value = data.get("stream_url")
        stream_url: str | None = None
        if stream_value is not None:
            stream_url = _string(stream_value, "camera.stream_url")
            parsed = urlparse(stream_url)
            if parsed.scheme not in {"http", "https"} or not parsed.netloc:
                raise SettingsError("settings.camera.stream_url must be an HTTP(S) URL")

        device_value = data.get("device_index")
        device_index: int | None = None
        if device_value is not None:
            device_index = int(_number(device_value, "camera.device_index", integer=True))
            if device_index < 0:
                raise SettingsError("settings.camera.device_index must be non-negative")

        avfoundation_value = data.get("avfoundation_device")
        avfoundation_device = (
            None
            if avfoundation_value is None
            else _string(avfoundation_value, "camera.avfoundation_device")
        )
        avfoundation_framerate = _positive(
            data.get("avfoundation_framerate", 30.0),
            "camera.avfoundation_framerate",
        )
        if (
            stream_url is None
            and device_index is None
            and avfoundation_device is None
        ):
            raise SettingsError(
                "one of settings.camera.stream_url, settings.camera.device_index, or "
                "settings.camera.avfoundation_device is required"
            )

        marker_ids_value = data.get("field_marker_ids", [0, 1, 2, 3])
        if not isinstance(marker_ids_value, Sequence) or isinstance(marker_ids_value, (str, bytes)):
            raise SettingsError("settings.camera.field_marker_ids must be a sequence")
        marker_ids = tuple(int(_number(value, "camera.field_marker_ids", integer=True)) for value in marker_ids_value)
        if len(marker_ids) != 4 or len(set(marker_ids)) != 4:
            raise SettingsError("settings.camera.field_marker_ids must contain four distinct IDs")

        field_positions = data.get("field_marker_positions_m")
        parsed_positions: dict[int, tuple[float, float]] | None = None
        if field_positions is not None:
            if not isinstance(field_positions, Mapping):
                raise SettingsError("settings.camera.field_marker_positions_m must be a mapping")
            parsed_positions = {}
            for marker_id in marker_ids:
                raw = field_positions.get(marker_id, field_positions.get(str(marker_id)))
                if raw is None:
                    raise SettingsError(
                        f"settings.camera.field_marker_positions_m is missing marker {marker_id}"
                    )
                parsed_positions[marker_id] = _point(
                    raw, f"camera.field_marker_positions_m[{marker_id}]"
                )

        target_classes_value = data.get(
            "target_classes", ["start point", "delivery destination"]
        )
        if not isinstance(target_classes_value, Sequence) or isinstance(
            target_classes_value, (str, bytes)
        ):
            raise SettingsError("settings.camera.target_classes must be a sequence")
        target_classes = tuple(
            _string(value, "camera.target_classes") for value in target_classes_value
        )
        if not target_classes:
            raise SettingsError("settings.camera.target_classes must not be empty")

        confidence = float(
            _number(data.get("target_confidence", 0.25), "camera.target_confidence")
        )
        if not 0.0 <= confidence <= 1.0:
            raise SettingsError("settings.camera.target_confidence must be in the range 0..1")

        aggregation_frames = int(
            _number(
                data.get("target_aggregation_frames", 1),
                "camera.target_aggregation_frames",
                integer=True,
            )
        )
        if aggregation_frames < 1:
            raise SettingsError("settings.camera.target_aggregation_frames must be positive")
        required_classes_value = data.get("required_target_classes", [])
        if not isinstance(required_classes_value, Sequence) or isinstance(
            required_classes_value, (str, bytes)
        ):
            raise SettingsError("settings.camera.required_target_classes must be a sequence")
        required_classes = tuple(
            _string(value, "camera.required_target_classes")
            for value in required_classes_value
        )

        center_value = data.get("camera_center_floor_m")
        camera_center_floor = (
            None
            if center_value is None
            else _point(center_value, "camera.camera_center_floor_m")
        )
        drone_marker_id = int(
            _number(data.get("drone_marker_id", 10), "camera.drone_marker_id", integer=True)
        )
        if drone_marker_id in marker_ids:
            raise SettingsError("settings.camera.drone_marker_id must not be a field marker ID")

        return cls(
            stream_url=stream_url,
            device_index=device_index,
            avfoundation_device=avfoundation_device,
            avfoundation_framerate=avfoundation_framerate,
            height_m=_positive(data.get("height_m", 2.0), "camera.height_m"),
            aruco_dictionary=_dictionary_setting(
                data.get("aruco_dictionary", "DICT_4X4_50")
            ),
            field_marker_ids=marker_ids,
            field_size_m=_positive(data.get("field_size_m", 2.0), "camera.field_size_m"),
            marker_size_m=_positive(data.get("marker_size_m", 0.15), "camera.marker_size_m"),
            field_marker_positions_m=parsed_positions,
            camera_center_floor_m=camera_center_floor,
            drone_marker_id=drone_marker_id,
            target_model=_string(
                data.get("target_model", "yolov8s-worldv2.pt"), "camera.target_model"
            ),
            target_classes=target_classes,
            target_confidence=confidence,
            target_aggregation_frames=aggregation_frames,
            required_target_classes=required_classes,
            start_class=_string(data.get("start_class", "start point"), "camera.start_class"),
            start_label=_string(data.get("start_label", "start"), "camera.start_label"),
            start_from_drone_marker=_bool(
                data.get("start_from_drone_marker", False),
                "camera.start_from_drone_marker",
            ),
            start_marker_height_m=_positive(
                data.get("start_marker_height_m", 0.0),
                "camera.start_marker_height_m",
                allow_zero=True,
            ),
            max_frame_reads=int(
                _number(data.get("max_frame_reads", 4), "camera.max_frame_reads", integer=True)
            ),
        )


@dataclass(frozen=True)
class FlightSettings:
    altitude_m: float = 0.5
    hover_seconds: float = 2.0
    speed_mps: float = 0.3

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "FlightSettings":
        return cls(
            altitude_m=_positive(data.get("altitude_m", 0.5), "flight.altitude_m"),
            hover_seconds=_positive(
                data.get("hover_seconds", 2.0), "flight.hover_seconds", allow_zero=True
            ),
            speed_mps=_positive(data.get("speed_mps", 0.3), "flight.speed_mps"),
        )


@dataclass(frozen=True)
class RouteSettings:
    penalty: float = 2.0
    reps: int = 2
    shots: int = 4096
    seed: int = 1234
    maxiter: int = 60

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "RouteSettings":
        penalty = _positive(data.get("penalty", 2.0), "route.penalty")
        reps = int(_number(data.get("reps", 2), "route.reps", integer=True))
        shots = int(_number(data.get("shots", 4096), "route.shots", integer=True))
        maxiter = int(_number(data.get("maxiter", 60), "route.maxiter", integer=True))
        if reps < 1 or shots < 1 or maxiter < 1:
            raise SettingsError("route.reps, route.shots, and route.maxiter must be positive")
        return cls(
            penalty=penalty,
            reps=reps,
            shots=shots,
            seed=int(_number(data.get("seed", 1234), "route.seed", integer=True)),
            maxiter=maxiter,
        )


@dataclass(frozen=True)
class DroneSettings:
    mode: str = "mock"
    host: str = "CHANGE_ME_LOCAL_STAMPFLY_HOST"
    command_port: int = 8889
    state_port: int = 8890
    timeout_s: float = 1.0
    retries: int = 2
    go_min_cm: int = 10
    go_max_cm: int = 300
    speed_min_cm_s: int = 10
    speed_max_cm_s: int = 100

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any], *, require_host: bool = False) -> "DroneSettings":
        mode = _string(data.get("mode", "mock"), "drone.mode").lower()
        if mode not in {"mock", "udp", "stampfly_udp"}:
            raise SettingsError("settings.drone.mode must be mock, udp, or stampfly_udp")
        host = str(data.get("host", "")).strip()
        if require_host or mode != "mock":
            if not host or host.startswith("CHANGE_ME"):
                raise SettingsError("settings.drone.host must be set for StampFly mode")
        command_port = _port(data.get("command_port", 8889), "drone.command_port")
        state_port = _port(data.get("state_port", 8890), "drone.state_port")
        timeout_s = _positive(data.get("timeout_s", 1.0), "drone.timeout_s")
        retries = int(_number(data.get("retries", 2), "drone.retries", integer=True))
        if retries < 0:
            raise SettingsError("settings.drone.retries must be non-negative")
        go_min = int(_number(data.get("go_min_cm", 10), "drone.go_min_cm", integer=True))
        go_max = int(_number(data.get("go_max_cm", 300), "drone.go_max_cm", integer=True))
        speed_min = int(
            _number(data.get("speed_min_cm_s", 10), "drone.speed_min_cm_s", integer=True)
        )
        speed_max = int(
            _number(data.get("speed_max_cm_s", 100), "drone.speed_max_cm_s", integer=True)
        )
        if not 1 <= go_min <= go_max or go_max <= 0:
            raise SettingsError("settings.drone.go_min_cm/go_max_cm are invalid")
        if not 1 <= speed_min <= speed_max or speed_max <= 0:
            raise SettingsError("settings.drone.speed_min_cm_s/speed_max_cm_s are invalid")
        return cls(
            mode=mode,
            host=host,
            command_port=command_port,
            state_port=state_port,
            timeout_s=timeout_s,
            retries=retries,
            go_min_cm=go_min,
            go_max_cm=go_max,
            speed_min_cm_s=speed_min,
            speed_max_cm_s=speed_max,
        )


@dataclass(frozen=True)
class AppSettings:
    camera: CameraSettings
    flight: FlightSettings
    route: RouteSettings
    drone: DroneSettings


def default_settings() -> AppSettings:
    """Return safe defaults used when mock mode has no local YAML file."""

    return AppSettings(
        camera=CameraSettings(stream_url="http://camera.example.invalid/stream"),
        flight=FlightSettings(),
        route=RouteSettings(),
        drone=DroneSettings(),
    )


def load_settings(
    path: str | Path,
    *,
    allow_missing: bool = False,
    require_drone_host: bool = False,
) -> AppSettings:
    """Load and validate one settings file without logging its values."""

    settings_path = Path(path)
    if not settings_path.exists():
        if allow_missing:
            return default_settings()
        raise FileNotFoundError(
            f"settings file not found: {settings_path}; copy settings.example.yaml first"
        )

    try:
        raw = yaml.safe_load(settings_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise SettingsError("settings file contains invalid YAML") from exc
    if not isinstance(raw, Mapping):
        raise SettingsError("settings file must contain a YAML mapping")

    camera = CameraSettings.from_mapping(_section(raw, "camera"))
    flight = FlightSettings.from_mapping(_section(raw, "flight"))
    route = RouteSettings.from_mapping(_section(raw, "route"))
    drone = DroneSettings.from_mapping(
        _section(raw, "drone"), require_host=require_drone_host
    )
    speed_cm_s = flight.speed_mps * 100.0
    if not drone.speed_min_cm_s <= speed_cm_s <= drone.speed_max_cm_s:
        raise SettingsError(
            "settings.flight.speed_mps must be within the configured StampFly speed range"
        )
    if flight.altitude_m > camera.height_m:
        raise SettingsError("settings.flight.altitude_m must not exceed camera.height_m")
    if camera.start_marker_height_m > camera.height_m:
        raise SettingsError("settings.camera.start_marker_height_m must not exceed camera.height_m")
    if camera.marker_size_m >= camera.field_size_m:
        raise SettingsError("settings.camera.marker_size_m must be smaller than field_size_m")
    if camera.max_frame_reads < 1:
        raise SettingsError("settings.camera.max_frame_reads must be positive")
    return AppSettings(camera=camera, flight=flight, route=route, drone=drone)
