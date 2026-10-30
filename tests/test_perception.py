import cv2
import numpy as np

from src.perception import FloorMapper


def _synthetic_floor_image():
    image_size = (900, 1200)  # height, width
    field_size = 2.0
    marker_size = 0.15
    marker_positions = {
        0: (0.20, 0.20),
        1: (1.80, 0.20),
        2: (1.80, 1.80),
        3: (0.20, 1.80),
    }
    floor_corners = np.array(
        [[0.0, 0.0], [field_size, 0.0], [field_size, field_size], [0.0, field_size]],
        dtype=np.float32,
    )
    image_corners = np.array(
        [[130.0, 90.0], [1060.0, 130.0], [1110.0, 790.0], [90.0, 730.0]],
        dtype=np.float32,
    )
    floor_to_image = cv2.getPerspectiveTransform(floor_corners, image_corners)
    image = np.full((*image_size, 3), 255, dtype=np.uint8)
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

    for marker_id, center in marker_positions.items():
        source = cv2.aruco.generateImageMarker(dictionary, marker_id, 320, borderBits=1)
        half = marker_size / 2.0
        world_square = np.array(
            [
                [center[0] - half, center[1] - half],
                [center[0] + half, center[1] - half],
                [center[0] + half, center[1] + half],
                [center[0] - half, center[1] + half],
            ],
            dtype=np.float32,
        )
        destination = cv2.perspectiveTransform(
            world_square.reshape(1, -1, 2), floor_to_image
        ).reshape(-1, 2)
        marker_to_image = cv2.getPerspectiveTransform(
            np.array([[0, 0], [319, 0], [319, 319], [0, 319]], dtype=np.float32),
            destination,
        )
        warped = cv2.warpPerspective(
            source,
            marker_to_image,
            (image_size[1], image_size[0]),
            borderValue=255,
        )
        mask = cv2.warpPerspective(
            np.full(source.shape, 255, dtype=np.uint8),
            marker_to_image,
            (image_size[1], image_size[0]),
        )
        region = mask > 0
        image[region] = cv2.cvtColor(warped, cv2.COLOR_GRAY2BGR)[region]
    return image, floor_to_image, marker_positions


def test_floor_mapper_synthetic_homography_is_within_one_centimetre():
    image, floor_to_image, marker_positions = _synthetic_floor_image()
    mapper = FloorMapper(
        field_marker_positions=marker_positions,
        marker_size_m=0.15,
    )
    mapper.calibrate(image)

    known_floor_point = np.array([1.05, 0.72], dtype=np.float32)
    pixel = cv2.perspectiveTransform(
        known_floor_point.reshape(1, 1, 2), floor_to_image
    ).reshape(2)
    estimated = mapper.to_floor(pixel)

    assert np.linalg.norm(estimated - known_floor_point) < 0.01


def test_correct_parallax_h_zero_does_not_change_point():
    mapper = FloorMapper(camera_height_m=2.0, camera_center_floor=(1.0, 1.0))
    measured = np.array([1.3, 0.4])
    corrected = mapper.correct_parallax(measured, h=0.0)
    np.testing.assert_allclose(corrected, measured)


def test_correct_parallax_camera_nadir_point_does_not_move():
    mapper = FloorMapper(camera_height_m=2.0, camera_center_floor=(1.0, 1.0))
    nadir = np.array([1.0, 1.0])
    corrected = mapper.correct_parallax(nadir, h=0.5)
    np.testing.assert_allclose(corrected, nadir)


def test_drone_position_uses_small_marker_fallback(monkeypatch):
    mapper = FloorMapper(camera_height_m=2.0, camera_center_floor=(1.0, 1.0))
    mapper._homography = np.eye(3, dtype=np.float64)
    marker_corners = np.array(
        [[0.4, 0.4], [0.6, 0.4], [0.6, 0.6], [0.4, 0.6]], dtype=np.float32
    )
    monkeypatch.setattr(mapper, "detect_markers", lambda _image: {})
    monkeypatch.setattr(
        mapper,
        "_detect_small_marker",
        lambda _image, marker_id: marker_corners if marker_id == 10 else None,
    )

    position = mapper.drone_position(np.zeros((2, 2), dtype=np.uint8), h=0.0)

    np.testing.assert_allclose(position, np.array([0.5, 0.5]))

