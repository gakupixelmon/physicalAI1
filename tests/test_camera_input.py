import numpy as np
import cv2
from types import SimpleNamespace

from src.mission import OpenCVCameraInput


class _FakeCapture:
    def __init__(self, frames):
        self.frames = list(frames)
        self.released = False

    def set(self, *_args):
        return True

    def isOpened(self):
        return True

    def read(self):
        if self.frames:
            return True, self.frames.pop(0)
        return False, None

    def release(self):
        self.released = True


class _FakeMapper:
    calibrated = False

    def calibrate(self, frame):
        self.calibrated = True
        self.calibration_frame = frame


class _StartMarkerMapper:
    calibrated = True

    def to_floor(self, point):
        return np.asarray(point, dtype=float)

    def drone_position(self, _frame, *, h):
        assert h == 0.0
        return np.array([0.2, 0.3])


class _FakeDetector:
    def detect(self, _frame):
        return [
            type(
                "Detection",
                (),
                {"label": "apple", "center_px": (0.7, 0.8)},
            )()
        ]


class _AggregatingDetector:
    def detect(self, frame):
        if int(frame[0, 0, 0]) == 1:
            label, confidence, center = "apple", 0.4, (0.3, 0.4)
        else:
            label, confidence, center = "banana", 0.5, (0.7, 0.8)
        return [
            type(
                "Detection",
                (),
                {"label": label, "confidence": confidence, "center_px": center},
            )()
        ]


def test_opencv_camera_keeps_latest_frame(monkeypatch):
    frames = [
        np.full((2, 2, 3), value, dtype=np.uint8)
        for value in (1, 2, 3)
    ]
    capture = _FakeCapture(frames)
    monkeypatch.setattr("src.mission.cv2.VideoCapture", lambda _url: capture)
    camera = OpenCVCameraInput(
        "http://camera.test/stream",
        _FakeMapper(),
        lambda: None,
        camera_height_m=2.0,
        drone_height_m=0.5,
        max_frame_reads=3,
    )

    frame = camera.calibrate()

    assert int(frame[0, 0, 0]) == 3
    assert camera.mapper.calibrated is True
    camera.close()
    assert capture.released is True


def test_named_avfoundation_camera_uses_exact_device_name(monkeypatch):
    source_frame = np.full((2, 3, 3), 42, dtype=np.uint8)
    ok, encoded = cv2.imencode(".jpg", source_frame)
    assert ok
    calls = []

    def fake_run(command, **_kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=0, stdout=encoded.tobytes(), stderr=b"")

    monkeypatch.setattr("src.mission.shutil.which", lambda _binary: "/usr/local/bin/ffmpeg")
    monkeypatch.setattr("src.mission.subprocess.run", fake_run)
    camera = OpenCVCameraInput(
        None,
        _FakeMapper(),
        lambda: None,
        camera_height_m=2.0,
        drone_height_m=0.5,
        avfoundation_device="Android Webcam",
    )

    frame = camera.calibrate()

    assert frame.shape == source_frame.shape
    assert "Android Webcam:none" in calls[0]
    assert 0 not in calls[0]


def test_camera_can_use_parked_drone_marker_as_start(monkeypatch):
    capture = _FakeCapture([np.zeros((2, 2, 3), dtype=np.uint8)])
    monkeypatch.setattr("src.mission.cv2.VideoCapture", lambda _url: capture)
    camera = OpenCVCameraInput(
        "http://camera.test/stream",
        _StartMarkerMapper(),
        _FakeDetector,
        camera_height_m=2.0,
        drone_height_m=0.5,
        start_from_drone_marker=True,
        start_marker_height_m=0.0,
        max_frame_reads=1,
    )

    observations = camera.recognize_targets()

    assert [(item.label, item.position) for item in observations] == [
        ("apple_1", (0.7, 0.8)),
        ("start", (0.2, 0.3)),
    ]


def test_camera_aggregates_required_targets_across_frames(monkeypatch):
    capture = _FakeCapture(
        [
            np.full((2, 2, 3), 1, dtype=np.uint8),
            np.full((2, 2, 3), 2, dtype=np.uint8),
        ]
    )
    monkeypatch.setattr("src.mission.cv2.VideoCapture", lambda _url: capture)
    camera = OpenCVCameraInput(
        "http://camera.test/stream",
        _StartMarkerMapper(),
        _AggregatingDetector,
        camera_height_m=2.0,
        drone_height_m=0.5,
        target_aggregation_frames=2,
        required_target_classes=("apple", "banana"),
        start_from_drone_marker=True,
        start_marker_height_m=0.0,
        max_frame_reads=1,
    )

    observations = camera.recognize_targets()

    assert {(item.label, item.position) for item in observations} == {
        ("apple_1", (0.3, 0.4)),
        ("banana_1", (0.7, 0.8)),
        ("start", (0.2, 0.3)),
    }
