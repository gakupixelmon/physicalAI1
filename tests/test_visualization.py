import numpy as np

from src.visualization import draw_mission_overlay


class _IdentityMapper:
    def to_image(self, points):
        return np.asarray(points, dtype=float) * 20.0 + 10.0

    def to_floor(self, points):
        return (np.asarray(points, dtype=float) - 10.0) / 20.0


def test_mission_overlay_draws_route_boxes_and_coordinates():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    detections = [{"label": "delivery", "bbox_xyxy": (20, 20, 40, 40)}]
    route = [(0.2, 0.2), (2.0, 0.2), (2.0, 2.0), (0.2, 0.2)]

    output = draw_mission_overlay(
        frame,
        detections,
        _IdentityMapper(),
        route,
        current_floor_position=(1.0, 1.0),
        route_index=1,
    )

    assert output.shape == frame.shape
    assert np.count_nonzero(output) > 0

