"""Floor-coordinate mapping, ArUco localization, and YOLO-World detection."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import ssl
import sys
from typing import Any, Mapping, Sequence

import cv2
import numpy as np


Point2D = Sequence[float] | np.ndarray


def _as_point_array(points: Point2D) -> tuple[np.ndarray, bool]:
    array = np.asarray(points, dtype=np.float32)
    single = array.shape == (2,)
    if single:
        return array.reshape(1, 2), True
    if array.ndim != 2 or array.shape[1] != 2:
        raise ValueError("points must have shape (2,) or (N, 2)")
    return array, False


def _image_to_gray(image: np.ndarray) -> np.ndarray:
    if not isinstance(image, np.ndarray) or image.ndim not in (2, 3):
        raise ValueError("image must be a grayscale or color numpy array")
    if image.ndim == 2:
        return image
    if image.shape[2] == 1:
        return image[:, :, 0]
    if image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    raise ValueError("color image must have 1, 3, or 4 channels")


def _dictionary_from_name(name: str | int):
    if isinstance(name, int):
        dictionary_id = name
    else:
        try:
            dictionary_id = getattr(cv2.aruco, name)
        except AttributeError as exc:
            raise ValueError(f"unknown ArUco dictionary: {name}") from exc
    return cv2.aruco.getPredefinedDictionary(dictionary_id)


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def _configure_macos_ssl_certificates() -> None:
    """Use macOS's trusted CA bundle when framework Python has none.

    Some python.org framework builds on macOS have no default CA file.  This
    affects the first download of YOLO-World/CLIP weights.  We only select the
    system trust bundle when the caller has not provided one and the Python
    default is missing; certificate verification remains enabled.
    """

    if sys.platform != "darwin" or os.environ.get("SSL_CERT_FILE"):
        return
    default_cafile = ssl.get_default_verify_paths().cafile
    if default_cafile and Path(default_cafile).is_file():
        return
    system_cafile = Path("/etc/ssl/cert.pem")
    if system_cafile.is_file():
        os.environ["SSL_CERT_FILE"] = str(system_cafile)


@dataclass(frozen=True)
class TargetDetection:
    """One YOLO-World detection in image coordinates."""

    label: str
    confidence: float
    bbox_xyxy: tuple[float, float, float, float]
    center_px: tuple[float, float]


class FloorMapper:
    """Map image pixels to a calibrated floor coordinate system.

    ``field_marker_positions`` contains the *centres* of the four field
    markers in metres.  The default coordinate system is a 2 m x 2 m square:
    ID 0=(0,0), ID 1=(2,0), ID 2=(2,2), ID 3=(0,2).  The y axis therefore
    points in the same direction as the image construction used by the
    synthetic tests.  Real installations may supply another mapping without
    changing the image-side detector.
    """

    def __init__(
        self,
        *,
        field_marker_ids: Sequence[int] = (0, 1, 2, 3),
        field_marker_positions: Mapping[int, Point2D] | None = None,
        field_size_m: float = 2.0,
        marker_size_m: float = 0.15,
        aruco_dictionary: str | int = "DICT_4X4_50",
        camera_height_m: float | None = None,
        drone_marker_id: int = 10,
        camera_center_floor: Point2D | None = None,
    ) -> None:
        if len(field_marker_ids) != 4 or len(set(field_marker_ids)) != 4:
            raise ValueError("exactly four distinct field marker IDs are required")
        if field_size_m <= 0 or marker_size_m <= 0:
            raise ValueError("field_size_m and marker_size_m must be positive")
        if camera_height_m is not None and camera_height_m <= 0:
            raise ValueError("camera_height_m must be positive")

        self.field_marker_ids = tuple(int(marker_id) for marker_id in field_marker_ids)
        self.marker_size_m = float(marker_size_m)
        self.camera_height_m = camera_height_m
        self.drone_marker_id = int(drone_marker_id)
        self.dictionary = _dictionary_from_name(aruco_dictionary)
        self.detector_parameters = cv2.aruco.DetectorParameters()
        self._homography: np.ndarray | None = None
        self._image_shape: tuple[int, int] | None = None

        if field_marker_positions is None:
            half = float(field_size_m)
            field_marker_positions = {
                self.field_marker_ids[0]: (0.0, 0.0),
                self.field_marker_ids[1]: (half, 0.0),
                self.field_marker_ids[2]: (half, half),
                self.field_marker_ids[3]: (0.0, half),
            }
        missing = set(self.field_marker_ids) - set(field_marker_positions)
        if missing:
            raise ValueError(f"missing floor positions for marker IDs: {sorted(missing)}")
        self.field_marker_positions = {
            marker_id: np.asarray(field_marker_positions[marker_id], dtype=np.float32)
            for marker_id in self.field_marker_ids
        }
        if any(point.shape != (2,) for point in self.field_marker_positions.values()):
            raise ValueError("field marker positions must be 2D points")

        if camera_center_floor is None:
            self.camera_center_floor: np.ndarray | None = None
        else:
            center = np.asarray(camera_center_floor, dtype=np.float32)
            if center.shape != (2,):
                raise ValueError("camera_center_floor must be a 2D point")
            self.camera_center_floor = center

    @property
    def homography(self) -> np.ndarray:
        if self._homography is None:
            raise RuntimeError("FloorMapper has not been calibrated")
        return self._homography.copy()

    @property
    def calibrated(self) -> bool:
        return self._homography is not None

    def _detect_markers(self, image: np.ndarray):
        gray = _image_to_gray(image)
        if hasattr(cv2.aruco, "ArucoDetector"):
            detector = cv2.aruco.ArucoDetector(self.dictionary, self.detector_parameters)
            return detector.detectMarkers(gray)[:2]
        return cv2.aruco.detectMarkers(
            gray,
            self.dictionary,
            parameters=self.detector_parameters,
        )[:2]

    def detect_markers(self, image: np.ndarray) -> dict[int, np.ndarray]:
        """Return detected marker corners keyed by integer marker ID."""

        corners, ids = self._detect_markers(image)
        if ids is None:
            return {}
        return {
            int(marker_id): np.asarray(marker_corners, dtype=np.float32).reshape(4, 2)
            for marker_corners, marker_id in zip(corners, ids.ravel())
        }

    def _detect_small_marker(
        self, image: np.ndarray, marker_id: int
    ) -> np.ndarray | None:
        """Retry one small marker with a higher-resolution detector pass.

        The drone marker is much smaller in the overhead image than the field
        markers. Resizing only for this fallback preserves normal field
        calibration behavior while making a 50 mm top marker more reliable.
        """

        scale = 2.0
        gray = _image_to_gray(image)
        enlarged = cv2.resize(
            gray,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_CUBIC,
        )
        parameters = cv2.aruco.DetectorParameters()
        parameters.adaptiveThreshWinSizeMin = 3
        parameters.adaptiveThreshWinSizeMax = 101
        parameters.adaptiveThreshWinSizeStep = 4
        parameters.minMarkerPerimeterRate = 0.01
        parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        if hasattr(cv2.aruco, "ArucoDetector"):
            corners, ids = cv2.aruco.ArucoDetector(
                self.dictionary, parameters
            ).detectMarkers(enlarged)[:2]
        else:
            corners, ids = cv2.aruco.detectMarkers(
                enlarged, self.dictionary, parameters=parameters
            )[:2]
        if ids is None:
            return None
        for candidate_corners, candidate_id in zip(corners, ids.ravel()):
            if int(candidate_id) == int(marker_id):
                return (
                    np.asarray(candidate_corners, dtype=np.float32)
                    .reshape(4, 2)
                    / scale
                )
        return None

    def calibrate(self, image: np.ndarray) -> np.ndarray:
        """Detect field markers and calculate the image-to-floor homography."""

        marker_corners = self.detect_markers(image)
        missing = [marker_id for marker_id in self.field_marker_ids if marker_id not in marker_corners]
        if missing:
            raise ValueError(f"field calibration markers not found: {missing}")

        image_points = np.array(
            [marker_corners[marker_id].mean(axis=0) for marker_id in self.field_marker_ids],
            dtype=np.float32,
        )
        floor_points = np.array(
            [self.field_marker_positions[marker_id] for marker_id in self.field_marker_ids],
            dtype=np.float32,
        )
        homography, _ = cv2.findHomography(image_points, floor_points, method=0)
        if homography is None or not np.all(np.isfinite(homography)):
            raise ValueError("could not calculate a floor homography")

        self._homography = homography.astype(np.float64)
        self._image_shape = image.shape[:2]
        if self.camera_center_floor is None:
            height, width = self._image_shape
            self.camera_center_floor = self.to_floor((width / 2.0, height / 2.0))
        return self.homography

    def to_floor(self, image_points: Point2D) -> np.ndarray:
        """Convert one pixel point or an ``(N,2)`` array to floor metres."""

        if self._homography is None:
            raise RuntimeError("FloorMapper has not been calibrated")
        points, single = _as_point_array(image_points)
        mapped = cv2.perspectiveTransform(points.reshape(-1, 1, 2), self._homography)
        mapped = mapped.reshape(-1, 2)
        return mapped[0] if single else mapped

    def to_image(self, floor_points: Point2D) -> np.ndarray:
        """Convert one floor point or an ``(N,2)`` array to image pixels."""

        if self._homography is None:
            raise RuntimeError("FloorMapper has not been calibrated")
        points, single = _as_point_array(floor_points)
        floor_to_image = np.linalg.inv(self._homography)
        mapped = cv2.perspectiveTransform(points.reshape(-1, 1, 2), floor_to_image)
        mapped = mapped.reshape(-1, 2)
        return mapped[0] if single else mapped

    def correct_parallax(
        self,
        measured_floor_point: Point2D,
        camera_center_floor: Point2D | None = None,
        h: float = 0.0,
        H: float | None = None,
    ) -> np.ndarray:
        """Correct a point measured on the floor plane for object height.

        The implemented equation is ``c + (1 - h/H) * (p_meas - c)``.  The
        method accepts ``h=0`` explicitly and leaves the measured point
        unchanged in that case.
        """

        if H is None:
            H = self.camera_height_m
        if H is None or H <= 0:
            raise ValueError("camera height H must be supplied and positive")
        if h < 0:
            raise ValueError("object height h must not be negative")
        if h > H:
            raise ValueError("object height h must not exceed camera height H")
        measured = np.asarray(measured_floor_point, dtype=np.float64)
        if measured.shape != (2,):
            raise ValueError("measured_floor_point must be a 2D point")
        if camera_center_floor is None:
            camera_center_floor = self.camera_center_floor
        if camera_center_floor is None:
            raise ValueError("camera_center_floor must be supplied or calibrated")
        center = np.asarray(camera_center_floor, dtype=np.float64)
        if center.shape != (2,):
            raise ValueError("camera_center_floor must be a 2D point")
        return center + (1.0 - float(h) / float(H)) * (measured - center)

    def drone_position(
        self,
        image: np.ndarray,
        *,
        h: float | None = None,
        drone_marker_id: int | None = None,
    ) -> np.ndarray | None:
        """Return the corrected floor position of the drone marker, if found."""

        if self._homography is None:
            raise RuntimeError("FloorMapper has not been calibrated")
        marker_id = self.drone_marker_id if drone_marker_id is None else int(drone_marker_id)
        marker_corners = self.detect_markers(image).get(marker_id)
        if marker_corners is None:
            marker_corners = self._detect_small_marker(image, marker_id)
        if marker_corners is None:
            return None
        measured = self.to_floor(marker_corners.mean(axis=0))
        if h is None:
            h = 0.0
        return self.correct_parallax(measured, h=h)


class TargetDetector:
    """Lazy YOLO-World detector for natural-language delivery targets."""

    def __init__(
        self,
        model_path: str | Path = "yolov8s-worldv2.pt",
        target_classes: Sequence[str] = ("delivery destination",),
        confidence: float = 0.25,
        *,
        model: Any | None = None,
    ) -> None:
        if not target_classes:
            raise ValueError("at least one target class is required")
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")
        self.model_path = str(model_path)
        self.target_classes = tuple(str(label) for label in target_classes)
        self.confidence = float(confidence)
        if model is None:
            try:
                _configure_macos_ssl_certificates()
                from ultralytics import YOLO
            except ImportError as exc:
                raise ImportError(
                    "TargetDetector requires ultralytics; install requirements.txt "
                    "before constructing the YOLO-World detector"
                ) from exc
            self.model = YOLO(self.model_path)
        else:
            self.model = model
        if hasattr(self.model, "set_classes"):
            self.model.set_classes(list(self.target_classes))

    def detect(self, image: np.ndarray) -> list[TargetDetection]:
        """Run YOLO-World and return normalized detection records."""

        _image_to_gray(image)  # validate the input without changing its channels
        if hasattr(self.model, "predict"):
            results = self.model.predict(
                source=image,
                conf=self.confidence,
                verbose=False,
            )
        else:
            results = self.model(image, conf=self.confidence, verbose=False)

        detections: list[TargetDetection] = []
        for result in results:
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue
            xyxy = _to_numpy(getattr(boxes, "xyxy", np.empty((0, 4))))
            confidences = _to_numpy(getattr(boxes, "conf", np.empty((0,))))
            class_indices = _to_numpy(getattr(boxes, "cls", np.empty((0,))))
            names = getattr(result, "names", getattr(self.model, "names", {}))
            for box, confidence, class_index in zip(xyxy, confidences, class_indices):
                score = float(confidence)
                if score < self.confidence or len(box) != 4:
                    continue
                coordinates = tuple(float(value) for value in box)
                center = (
                    (coordinates[0] + coordinates[2]) / 2.0,
                    (coordinates[1] + coordinates[3]) / 2.0,
                )
                class_number = int(class_index)
                if isinstance(names, Mapping):
                    label = str(names.get(class_number, self.target_classes[0]))
                elif class_number < len(names):
                    label = str(names[class_number])
                else:
                    label = self.target_classes[0]
                detections.append(
                    TargetDetection(
                        label=label,
                        confidence=score,
                        bbox_xyxy=coordinates,
                        center_px=center,
                    )
                )
        return detections

