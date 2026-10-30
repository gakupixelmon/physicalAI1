import numpy as np
import pytest

from src.perception import TargetDetector


@pytest.mark.yolo
def test_yolo_world_detector_smoke():
    """Opt-in smoke test; constructing the detector may download its model."""

    detector = TargetDetector(
        model_path="yolov8s-worldv2.pt",
        target_classes=("delivery destination",),
    )
    detections = detector.detect(np.zeros((320, 320, 3), dtype=np.uint8))
    assert isinstance(detections, list)
