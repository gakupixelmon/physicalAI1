"""Mission orchestration from recognition through return and landing."""

from __future__ import annotations

import itertools
import json
import logging
import select
import shutil
import subprocess
import sys
import threading
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import cv2

try:
    from .drone_link import DroneLink, MockDroneLink
except ImportError:  # Support ``python src/mission.py`` style imports.
    from drone_link import DroneLink, MockDroneLink


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class LocationObservation:
    """A recognized location in floor coordinates measured in metres."""

    label: str
    position: tuple[float, float]

    def as_dict(self) -> dict[str, Any]:
        return {"label": self.label, "position_m": list(self.position)}


class CameraInput(ABC):
    """Replaceable camera-side interface used by :class:`Mission`."""

    @abstractmethod
    def recognize_targets(self) -> list[LocationObservation]:
        """Return the start location followed by recognized destinations."""

    @abstractmethod
    def drone_marker_position(self) -> np.ndarray:
        """Return the current drone marker position in floor x/y metres."""

    def close(self) -> None:
        """Release camera resources; mock cameras have nothing to release."""


class OpenCVCameraInput(CameraInput):
    """Camera adapter for the configured MJPEG stream and perception stack.

    The adapter deliberately opens the stream lazily.  Constructing a real
    mission object therefore does not contact the camera or download YOLO
    weights.  ``read_latest_frame`` performs several reads and keeps only the
    newest successful frame so an OpenCV stream buffer does not drive the
    mission from stale images.
    """

    def __init__(
        self,
        source: str | int | None,
        mapper: Any,
        detector_factory: Callable[[], Any],
        *,
        camera_height_m: float,
        drone_height_m: float,
        target_aggregation_frames: int = 1,
        required_target_classes: Sequence[str] = (),
        start_class: str = "start point",
        start_label: str = "start",
        start_from_drone_marker: bool = False,
        start_marker_height_m: float = 0.0,
        max_frame_reads: int = 4,
        avfoundation_device: str | None = None,
        avfoundation_framerate: float = 30.0,
        ffmpeg_binary: str = "ffmpeg",
    ) -> None:
        if source is None and avfoundation_device is None:
            raise ValueError("camera source must not be empty")
        if isinstance(source, str) and not source:
            raise ValueError("camera source must not be empty")
        if source is not None and (
            not isinstance(source, (str, int)) or isinstance(source, bool)
        ):
            raise TypeError("camera source must be an HTTP(S) URL or a device index")
        if isinstance(source, int) and source < 0:
            raise ValueError("camera device index must be non-negative")
        if avfoundation_device is not None and (
            not isinstance(avfoundation_device, str) or not avfoundation_device.strip()
        ):
            raise ValueError("AVFoundation device name must be a non-empty string")
        if avfoundation_framerate <= 0:
            raise ValueError("AVFoundation framerate must be positive")
        if camera_height_m <= 0 or drone_height_m <= 0:
            raise ValueError("camera and drone heights must be positive")
        if drone_height_m > camera_height_m:
            raise ValueError("drone height must not exceed camera height")
        if start_marker_height_m < 0 or start_marker_height_m > camera_height_m:
            raise ValueError("start marker height must be in the camera height range")
        if target_aggregation_frames < 1:
            raise ValueError("target aggregation frames must be positive")
        if any(not isinstance(label, str) or not label.strip() for label in required_target_classes):
            raise ValueError("required target classes must be non-empty strings")
        if max_frame_reads < 1:
            raise ValueError("max_frame_reads must be positive")
        self.source = source
        self.avfoundation_device = (
            None if avfoundation_device is None else avfoundation_device.strip()
        )
        self.avfoundation_framerate = float(avfoundation_framerate)
        self.ffmpeg_binary = ffmpeg_binary
        self.mapper = mapper
        self._detector_factory = detector_factory
        self.camera_height_m = float(camera_height_m)
        self.drone_height_m = float(drone_height_m)
        self.target_aggregation_frames = int(target_aggregation_frames)
        self.required_target_classes = tuple(
            label.casefold().strip() for label in required_target_classes
        )
        self.start_class = start_class.casefold().strip()
        self.start_label = start_label
        self.start_from_drone_marker = bool(start_from_drone_marker)
        self.start_marker_height_m = float(start_marker_height_m)
        self.max_frame_reads = int(max_frame_reads)
        self._capture: cv2.VideoCapture | None = None
        self._detector: Any | None = None

    @property
    def opened(self) -> bool:
        return self.avfoundation_device is not None or (
            self._capture is not None and self._capture.isOpened()
        )

    def _ensure_capture(self) -> cv2.VideoCapture:
        if self._capture is None:
            if isinstance(self.source, int):
                backend = getattr(cv2, "CAP_AVFOUNDATION", cv2.CAP_ANY)
                capture = cv2.VideoCapture(self.source, backend)
            else:
                capture = cv2.VideoCapture(self.source)
            # Not all backends support this property, so it is only a hint.
            capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            if not capture.isOpened():
                capture.release()
                raise RuntimeError("camera stream could not be opened")
            self._capture = capture
        return self._capture

    def read_latest_frame(self) -> np.ndarray:
        """Read several frames and return the newest successful frame."""

        if self.avfoundation_device is not None:
            return self._read_avfoundation_frame()
        capture = self._ensure_capture()
        latest: np.ndarray | None = None
        for _ in range(self.max_frame_reads):
            ok, frame = capture.read()
            if ok and frame is not None:
                latest = frame
        if latest is None:
            raise RuntimeError("camera stream did not provide a frame")
        return latest

    def _read_avfoundation_frame(self) -> np.ndarray:
        """Capture one current frame from a named macOS AVFoundation device.

        OpenCV camera indexes are not stable when multiple macOS cameras are
        attached.  FFmpeg accepts the exact device name instead, avoiding a
        silent switch to a different camera.  A fresh process also avoids a
        buffered old frame.
        """

        if shutil.which(self.ffmpeg_binary) is None:
            raise RuntimeError(
                f"FFmpeg executable {self.ffmpeg_binary!r} is required for "
                "camera.avfoundation_device"
            )
        command = [
            self.ffmpeg_binary,
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "avfoundation",
            "-framerate",
            f"{self.avfoundation_framerate:g}",
            "-i",
            f"{self.avfoundation_device}:none",
            "-frames:v",
            "1",
            "-f",
            "image2pipe",
            "-vcodec",
            "mjpeg",
            "pipe:1",
        ]
        try:
            result = subprocess.run(
                command,
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=10.0,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError("timed out while reading AVFoundation camera frame") from exc

        if result.returncode != 0 or not result.stdout:
            detail = result.stderr.decode("utf-8", errors="replace").strip()
            raise RuntimeError(
                "AVFoundation camera did not provide a frame"
                + (f": {detail}" if detail else "")
            )
        encoded = np.frombuffer(result.stdout, dtype=np.uint8)
        frame = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if frame is None:
            raise RuntimeError("could not decode AVFoundation camera frame")
        return frame

    def calibrate(self) -> np.ndarray:
        """Read a current frame and calibrate the configured floor mapper."""

        frame = self.read_latest_frame()
        self.mapper.calibrate(frame)
        return frame

    def _ensure_mapper(self, frame: np.ndarray) -> None:
        if not self.mapper.calibrated:
            self.mapper.calibrate(frame)

    def _drone_position_with_retries(
        self,
        *,
        height_m: float,
        initial_frame: np.ndarray | None = None,
    ) -> np.ndarray | None:
        """Read fresh frames until the small drone marker is found."""

        for attempt in range(self.max_frame_reads):
            frame = (
                initial_frame
                if attempt == 0 and initial_frame is not None
                else self.read_latest_frame()
            )
            self._ensure_mapper(frame)
            position = self.mapper.drone_position(frame, h=height_m)
            if position is not None:
                return np.asarray(position, dtype=float)
        return None

    def recognize_targets(self) -> list[LocationObservation]:
        frame = self.read_latest_frame()
        self._ensure_mapper(frame)
        if self._detector is None:
            self._detector = self._detector_factory()
        if self.target_aggregation_frames > 1:
            return self._recognize_targets_aggregated(frame)
        return self._locations_from_detections(self._detector.detect(frame), frame)

    def _locations_from_detections(
        self, detections: Sequence[Any], frame: np.ndarray
    ) -> list[LocationObservation]:
        """Convert detections from one frame into floor observations."""

        observations: list[LocationObservation] = []
        class_counts: dict[str, int] = {}
        start_count = 0
        for detection in detections:
            class_name = str(detection.label).strip()
            if not class_name:
                continue
            if class_name.casefold() == self.start_class:
                start_count += 1
                if start_count > 1:
                    raise ValueError("more than one configured start target was detected")
                label = self.start_label
            else:
                class_counts[class_name] = class_counts.get(class_name, 0) + 1
                label = f"{class_name}_{class_counts[class_name]}"
            floor_position = self.mapper.to_floor(detection.center_px)
            observations.append(
                LocationObservation(
                    label=label,
                    position=(float(floor_position[0]), float(floor_position[1])),
                )
            )
        if start_count == 0 and self.start_from_drone_marker:
            start_position = self._drone_position_with_retries(
                height_m=self.start_marker_height_m,
                initial_frame=frame,
            )
            if start_position is not None:
                observations.append(
                    LocationObservation(
                        label=self.start_label,
                        position=(float(start_position[0]), float(start_position[1])),
                    )
                )
                start_count = 1
        if start_count != 1:
            raise ValueError(
                "configured start target was not detected; check camera.target_classes "
                "and camera.start_class, or place the drone marker at START"
            )
        return observations

    def _recognize_targets_aggregated(
        self, initial_frame: np.ndarray
    ) -> list[LocationObservation]:
        """Keep the strongest instance of each target class across fresh frames."""

        best_targets: dict[str, tuple[str, float, tuple[float, float]]] = {}
        best_start: tuple[float, tuple[float, float]] | None = None
        for frame_index in range(self.target_aggregation_frames):
            frame = initial_frame if frame_index == 0 else self.read_latest_frame()
            self._ensure_mapper(frame)
            for detection in self._detector.detect(frame):
                class_name = str(detection.label).strip()
                if not class_name:
                    continue
                confidence = float(detection.confidence)
                floor_position = self.mapper.to_floor(detection.center_px)
                position = (float(floor_position[0]), float(floor_position[1]))
                if class_name.casefold() == self.start_class:
                    if best_start is None or confidence > best_start[0]:
                        best_start = (confidence, position)
                    continue
                key = class_name.casefold()
                previous = best_targets.get(key)
                if previous is None or confidence > previous[1]:
                    best_targets[key] = (class_name, confidence, position)

        missing = [
            class_name
            for class_name in self.required_target_classes
            if class_name not in best_targets
        ]
        if missing:
            raise ValueError(
                "required delivery targets were not detected: " + ", ".join(missing)
            )

        observations = [
            LocationObservation(label=f"{class_name}_1", position=position)
            for class_name, _confidence, position in best_targets.values()
        ]
        if best_start is not None:
            observations.append(LocationObservation(self.start_label, best_start[1]))
        elif self.start_from_drone_marker:
            start_position = self._drone_position_with_retries(
                height_m=self.start_marker_height_m,
                initial_frame=initial_frame,
            )
            if start_position is not None:
                observations.append(
                    LocationObservation(
                        label=self.start_label,
                        position=(float(start_position[0]), float(start_position[1])),
                    )
                )
        if not any(item.label == self.start_label for item in observations):
            raise ValueError(
                "configured start target was not detected; check camera.target_classes "
                "and camera.start_class, or place the drone marker at START"
            )
        return observations

    def drone_marker_position(self) -> np.ndarray:
        position = self._drone_position_with_retries(height_m=self.drone_height_m)
        if position is None:
            raise RuntimeError(
                "drone ArUco marker was not detected in the latest camera frames"
            )
        return position

    def close(self) -> None:
        if self._capture is not None:
            self._capture.release()
        self._capture = None


class EmergencyStopRequested(RuntimeError):
    """Raised by the mission loop after the emergency key is pressed."""


class EmergencyStopMonitor:
    """Watch stdin for a single emergency key followed by Enter."""

    def __init__(self, drone_link: DroneLink, key: str = "e") -> None:
        if len(key) != 1:
            raise ValueError("emergency key must be one character")
        self.drone_link = drone_link
        self.key = key.lower()
        self.triggered = threading.Event()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None:
            return
        print(
            f"Emergency stop enabled: type {self.key!r} then Enter to stop motors.",
            file=sys.stderr,
            flush=True,
        )
        self._thread = threading.Thread(target=self._watch, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=0.5)
        self._thread = None

    def check(self) -> None:
        if self.triggered.is_set():
            raise EmergencyStopRequested("emergency stop key was pressed")

    def _watch(self) -> None:
        while not self._stop.is_set():
            try:
                ready, _, _ = select.select([sys.stdin], [], [], 0.1)
            except (OSError, ValueError):
                return
            if not ready:
                continue
            line = sys.stdin.readline().strip().lower()
            if line == self.key:
                try:
                    self.drone_link.emergency()
                finally:
                    self.triggered.set()
                return


class MockCamera(CameraInput):
    """Virtual camera backed by a ``MockDroneLink`` position."""

    def __init__(
        self,
        drone_link: MockDroneLink,
        locations: Sequence[LocationObservation],
        *,
        marker_noise_std: float = 0.0,
        seed: int | None = 3,
    ) -> None:
        if marker_noise_std < 0:
            raise ValueError("marker_noise_std must be non-negative")
        self.drone_link = drone_link
        self.locations = list(locations)
        self.marker_noise_std = float(marker_noise_std)
        self._rng = np.random.default_rng(seed)

    def recognize_targets(self) -> list[LocationObservation]:
        return list(self.locations)

    def drone_marker_position(self) -> np.ndarray:
        position = self.drone_link.position[:2]
        if self.marker_noise_std:
            position = position + self._rng.normal(0.0, self.marker_noise_std, size=2)
        return np.asarray(position, dtype=float)


@dataclass(frozen=True)
class MissionConfig:
    start_label: str = "start"
    takeoff_height_m: float = 0.5
    speed_mps: float = 0.3
    hover_seconds: float = 0.0
    correction_threshold_m: float = 0.10
    max_corrections: int = 2
    penalty: float = 2.0
    qaoa_reps: int = 2
    qaoa_shots: int = 512
    qaoa_maxiter: int = 60
    qaoa_seed: int = 1234
    enable_emergency_key: bool = False


DEFAULT_MOCK_LOCATIONS = (
    LocationObservation("start", (0.20, 0.20)),
    LocationObservation("delivery_1", (1.60, 0.30)),
    LocationObservation("delivery_2", (1.70, 1.60)),
    LocationObservation("delivery_3", (0.30, 1.70)),
)


def _distance_matrix(locations: Sequence[LocationObservation]) -> np.ndarray:
    points = np.asarray([location.position for location in locations], dtype=float)
    differences = points[:, None, :] - points[None, :, :]
    return np.linalg.norm(differences, axis=2)


def _route_length(route: Sequence[int], distances: np.ndarray) -> float:
    return float(sum(distances[a, b] for a, b in zip(route, route[1:])))


def _exact_route(distances: np.ndarray) -> tuple[tuple[int, ...], float]:
    best_route: tuple[int, ...] | None = None
    best_length = float("inf")
    for permutation in itertools.permutations(range(1, len(distances))):
        candidate = (0, *permutation, 0)
        length = _route_length(candidate, distances)
        if length < best_length:
            best_route = candidate
            best_length = length
    assert best_route is not None
    return best_route, best_length


def _point(value: Sequence[float] | np.ndarray) -> list[float]:
    return [float(component) for component in np.asarray(value, dtype=float)[:2]]


class Mission:
    """Run recognition, route planning, flight, correction, and landing."""

    def __init__(
        self,
        drone_link: DroneLink,
        camera: CameraInput,
        *,
        config: MissionConfig | None = None,
    ) -> None:
        self.drone_link = drone_link
        self.camera = camera
        self.config = config or MissionConfig()

    def run(self, log_path: str | Path = "results/run_log.json") -> dict[str, Any]:
        locations = self._recognized_locations()
        distances = _distance_matrix(locations)
        plan = self._plan_route(distances)
        route = tuple(plan["route_indices"])

        connected = False
        landed = False
        arrivals: list[dict[str, Any]] = []
        started = time.perf_counter()
        emergency_monitor = (
            EmergencyStopMonitor(self.drone_link)
            if self.config.enable_emergency_key
            else None
        )
        try:
            if emergency_monitor is not None:
                emergency_monitor.start()
            self.drone_link.connect()
            connected = True
            self.drone_link.takeoff(self.config.takeoff_height_m)

            commanded_xy = np.asarray(locations[0].position, dtype=float)
            for route_position, location_index in enumerate(route[1:], start=1):
                if emergency_monitor is not None:
                    emergency_monitor.check()
                target = np.asarray(locations[location_index].position, dtype=float)
                is_return = location_index == 0 and route_position == len(route) - 1
                displacement = target - commanded_xy
                command_result = self.drone_link.go(
                    float(displacement[0]),
                    float(displacement[1]),
                    0.0,
                    self.config.speed_mps,
                )
                result_array = np.asarray(command_result, dtype=float)
                if result_array.shape == (3,):
                    commanded_xy = result_array[:2]
                else:
                    commanded_xy = target.copy()

                if is_return:
                    return_position = self.camera.drone_marker_position()
                    arrivals.append(
                        {
                            "label": locations[location_index].label,
                            "target_position_m": _point(target),
                            "measured_position_m": _point(return_position),
                            "return": True,
                        }
                    )
                    continue

                measured_before = self.camera.drone_marker_position()
                error_before = float(np.linalg.norm(target - measured_before))
                actual_before = getattr(self.drone_link, "position", None)
                correction_attempts: list[dict[str, Any]] = []
                measured_after = measured_before.copy()

                for attempt in range(1, self.config.max_corrections + 1):
                    current_error = float(np.linalg.norm(target - measured_after))
                    if current_error <= self.config.correction_threshold_m:
                        break
                    correction = target - measured_after
                    command_result = self.drone_link.go(
                        float(correction[0]),
                        float(correction[1]),
                        0.0,
                        self.config.speed_mps,
                    )
                    result_array = np.asarray(command_result, dtype=float)
                    if result_array.shape == (3,):
                        commanded_xy = result_array[:2]
                    measured_after = self.camera.drone_marker_position()
                    if emergency_monitor is not None:
                        emergency_monitor.check()
                    correction_attempts.append(
                        {
                            "attempt": attempt,
                            "measured_position_m": _point(measured_after),
                            "error_m": float(np.linalg.norm(target - measured_after)),
                        }
                    )

                if self.config.hover_seconds > 0:
                    time.sleep(self.config.hover_seconds)
                error_after = float(np.linalg.norm(target - measured_after))
                actual_after = getattr(self.drone_link, "position", None)
                arrivals.append(
                    {
                        "label": locations[location_index].label,
                        "target_position_m": _point(target),
                        "measured_position_before_correction_m": _point(measured_before),
                        "measured_position_m": _point(measured_after),
                        "actual_position_before_correction_m": _point(
                            measured_before if actual_before is None else actual_before
                        ),
                        "actual_position_m": _point(
                            measured_after if actual_after is None else actual_after
                        ),
                        "error_without_correction_m": error_before,
                        "error_after_correction_m": error_after,
                        "correction_attempts": correction_attempts,
                        "corrected": bool(correction_attempts),
                    }
                )

            self.drone_link.land()
            landed = True
        finally:
            if emergency_monitor is not None:
                emergency_monitor.stop()
            if connected and not landed:
                try:
                    self.drone_link.land()
                except Exception:
                    LOGGER.exception("finally-land command failed")
            try:
                self.camera.close()
            except Exception:
                LOGGER.exception("camera close failed")

        before_errors = [item["error_without_correction_m"] for item in arrivals if not item.get("return")]
        after_errors = [item["error_after_correction_m"] for item in arrivals if not item.get("return")]
        before_mean = float(np.mean(before_errors)) if before_errors else 0.0
        after_mean = float(np.mean(after_errors)) if after_errors else 0.0
        elapsed = time.perf_counter() - started

        qaoa_log = dict(plan["qaoa"])
        qaoa_log["route_indices"] = list(route)
        qaoa_log["route_labels"] = [locations[index].label for index in route]
        qaoa_log["exact_route_indices"] = list(plan["exact_route_indices"])
        qaoa_log["exact_route_labels"] = [
            locations[index].label for index in plan["exact_route_indices"]
        ]

        result: dict[str, Any] = {
            "status": "completed",
            "completed": True,
            "mode": "mock" if isinstance(self.drone_link, MockDroneLink) else "custom",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "locations": [location.as_dict() for location in locations],
            "qaoa": qaoa_log,
            "flight": {
                "arrivals": arrivals,
                "landed": landed,
                "elapsed_seconds": float(elapsed),
            },
            "correction_summary": {
                "mean_error_without_correction_m": before_mean,
                "mean_error_after_correction_m": after_mean,
                "improved": bool(after_mean < before_mean),
                "threshold_m": self.config.correction_threshold_m,
                "max_corrections": self.config.max_corrections,
            },
            "classical_fallback": bool(qaoa_log.get("used_fallback", False)),
        }
        if isinstance(self.drone_link, MockDroneLink):
            result["mock_drone_history"] = self.drone_link.history
        self._write_log(log_path, result)
        return result

    def _recognized_locations(self) -> list[LocationObservation]:
        recognized = self.camera.recognize_targets()
        if len(recognized) < 2:
            raise ValueError("at least a start and one destination are required")
        labels = [location.label for location in recognized]
        if len(labels) != len(set(labels)):
            raise ValueError("recognized location labels must be unique")
        start_index = labels.index(self.config.start_label)
        if start_index:
            recognized = [recognized[start_index], *recognized[:start_index], *recognized[start_index + 1 :]]
        return list(recognized)

    def _plan_route(self, distances: np.ndarray) -> dict[str, Any]:
        exact_route, exact_length = _exact_route(distances)
        started = time.perf_counter()
        try:
            try:
                from .qaoa_route import solve_qaoa
            except ImportError:
                from qaoa_route import solve_qaoa
            qaoa_result = solve_qaoa(
                distances,
                penalty=self.config.penalty,
                reps=self.config.qaoa_reps,
                shots=self.config.qaoa_shots,
                seed=self.config.qaoa_seed,
                maxiter=self.config.qaoa_maxiter,
            )
            route = tuple(int(index) for index in qaoa_result["route"])
            if route[0] != 0 or route[-1] != 0 or len(route) != len(distances) + 1:
                raise ValueError("QAOA returned an invalid route")
            qaoa_log = {
                key: value
                for key, value in qaoa_result.items()
                if key not in {"counts", "exact_route", "route"}
            }
            qaoa_log.setdefault("used_fallback", False)
            qaoa_log.setdefault("runtime_seconds", time.perf_counter() - started)
            qaoa_log["exact_length"] = float(exact_length)
            return {
                "route_indices": route,
                "exact_route_indices": exact_route,
                "qaoa": qaoa_log,
            }
        except ImportError as exc:
            LOGGER.warning("Qiskit unavailable; using the exact classical route: %s", exc)
            return self._classical_plan(exact_route, exact_length, started, str(exc))

    @staticmethod
    def _classical_plan(
        exact_route: tuple[int, ...],
        exact_length: float,
        started: float,
        reason: str,
    ) -> dict[str, Any]:
        return {
            "route_indices": exact_route,
            "exact_route_indices": exact_route,
            "qaoa": {
                "route_length": float(exact_length),
                "exact_length": float(exact_length),
                "approximation_ratio": 1.0,
                "feasible_fraction": 0.0,
                "optimal_fraction": 0.0,
                "num_qubits": (len(exact_route) - 2) ** 2,
                "runtime_seconds": float(time.perf_counter() - started),
                "used_fallback": True,
                "fallback_reason": reason,
            },
        }

    @staticmethod
    def _write_log(path: str | Path, result: dict[str, Any]) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

