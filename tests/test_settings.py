import yaml
import pytest

from src.settings import SettingsError, load_settings


def _settings_data():
    return {
        "camera": {
            "stream_url": "http://camera.test/stream",
            "height_m": 2.5,
            "aruco_dictionary": "DICT_4X4_50",
            "field_marker_ids": [0, 1, 2, 3],
            "field_size_m": 2.0,
            "marker_size_m": 0.15,
            "drone_marker_id": 10,
            "target_model": "test-model.pt",
            "target_classes": ["start point", "delivery destination"],
            "target_confidence": 0.4,
            "start_class": "start point",
            "start_label": "start",
            "max_frame_reads": 5,
        },
        "flight": {"altitude_m": 0.5, "hover_seconds": 2.0, "speed_mps": 0.3},
        "route": {"penalty": 2.0, "reps": 3, "shots": 128, "seed": 9, "maxiter": 12},
        "drone": {
            "mode": "mock",
            "host": "CHANGE_ME_LOCAL_STAMPFLY_HOST",
            "command_port": 8889,
            "state_port": 8890,
            "timeout_s": 1.0,
            "retries": 2,
            "go_min_cm": 10,
            "go_max_cm": 300,
            "speed_min_cm_s": 10,
            "speed_max_cm_s": 100,
        },
    }


def test_load_settings_maps_all_runtime_sections(tmp_path):
    path = tmp_path / "settings.yaml"
    path.write_text(yaml.safe_dump(_settings_data()), encoding="utf-8")

    settings = load_settings(path)

    assert settings.camera.stream_url.endswith("/stream")
    assert settings.camera.height_m == 2.5
    assert settings.flight.speed_mps == 0.3
    assert settings.route.reps == 3
    assert settings.route.maxiter == 12
    assert settings.drone.state_port == 8890


def test_real_drone_validation_rejects_placeholder_host(tmp_path):
    path = tmp_path / "settings.yaml"
    path.write_text(yaml.safe_dump(_settings_data()), encoding="utf-8")

    with pytest.raises(SettingsError, match="drone.host"):
        load_settings(path, require_drone_host=True)


def test_missing_settings_can_use_safe_defaults_for_mock(tmp_path):
    settings = load_settings(tmp_path / "missing.yaml", allow_missing=True)

    assert settings.drone.mode == "mock"


def test_usb_camera_index_can_replace_stream_url(tmp_path):
    data = _settings_data()
    data["camera"].pop("stream_url")
    data["camera"]["device_index"] = 1
    path = tmp_path / "settings.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")

    settings = load_settings(path)

    assert settings.camera.stream_url is None
    assert settings.camera.device_index == 1


def test_named_avfoundation_camera_can_replace_stream_url(tmp_path):
    data = _settings_data()
    data["camera"].pop("stream_url")
    data["camera"]["avfoundation_device"] = "Android Webcam"
    path = tmp_path / "settings.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")

    settings = load_settings(path)

    assert settings.camera.stream_url is None
    assert settings.camera.avfoundation_device == "Android Webcam"
