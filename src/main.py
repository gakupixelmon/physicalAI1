"""Command-line entry point for the safe mock and supervised runtime."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

try:
    from .drone_link import MockDroneLink, StampFlyUdpLink
    from .mission import (
        DEFAULT_MOCK_LOCATIONS,
        Mission,
        MissionConfig,
        MockCamera,
        OpenCVCameraInput,
    )
    from .perception import FloorMapper, TargetDetector
    from .settings import AppSettings, load_settings
except ImportError:  # Required when invoked as ``python src/main.py``.
    from drone_link import MockDroneLink, StampFlyUdpLink
    from mission import (
        DEFAULT_MOCK_LOCATIONS,
        Mission,
        MissionConfig,
        MockCamera,
        OpenCVCameraInput,
    )
    from perception import FloorMapper, TargetDetector
    from settings import AppSettings, load_settings


LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--mock",
        action="store_true",
        help="run the complete mission against MockDroneLink (default)",
    )
    mode.add_argument(
        "--armed",
        action="store_true",
        help="select the supervised real-drone path; requires a literal yes confirmation",
    )
    parser.add_argument(
        "--settings",
        type=Path,
        default=PROJECT_ROOT / "config" / "settings.yaml",
        help="local settings file (default: config/settings.yaml)",
    )
    parser.add_argument(
        "--check-settings",
        action="store_true",
        help="validate settings without opening a camera or contacting a drone",
    )
    parser.add_argument(
        "--check-camera",
        action="store_true",
        help="open the configured stream and calibrate ArUco markers without flying",
    )
    parser.add_argument(
        "--log-path",
        type=Path,
        default=PROJECT_ROOT / "results" / "run_log.json",
        help="mission log path (default: results/run_log.json)",
    )
    parser.add_argument("--noise-std", type=float, default=0.04)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--qaoa-shots", type=int, default=None)
    parser.add_argument("--qaoa-maxiter", type=int, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if args.check_settings and args.armed:
        raise SystemExit("--check-settings cannot be combined with --armed")
    if args.check_camera and args.armed:
        raise SystemExit("--check-camera cannot be combined with --armed")

    settings = load_settings(
        args.settings,
        allow_missing=not (args.armed or args.check_settings or args.check_camera),
        require_drone_host=args.armed,
    )

    if args.check_settings:
        print("settings: valid")
        return 0

    if args.check_camera:
        _check_camera(settings)
        return 0

    if args.armed:
        confirmation = input(
            "REAL DRONE MODE: type 'yes' to acknowledge responsibility and continue: "
        ).strip().lower()
        if confirmation != "yes":
            print("real-drone mode cancelled")
            return 2
        if settings.drone.mode == "mock":
            raise ValueError("--armed requires settings.drone.mode to be udp or stampfly_udp")

        # Construction is intentionally side-effect free: no UDP datagram is
        # sent and the camera stream is opened only when Mission.run is called.
        link = _stampfly_link_from_settings(settings)
        camera = _camera_from_settings(settings)
        try:
            raise NotImplementedError(
                "real mission orchestration remains disabled until the camera and "
                "flight safety procedure are separately approved"
            )
        finally:
            camera.close()
            link.close()

    # Mock is deliberately the default, including when --mock is omitted.
    # Keep the historical deterministic mock-noise seed independent from the
    # QAOA seed in settings.  A CLI seed overrides both when supplied.
    seed = 3 if args.seed is None else args.seed
    link = MockDroneLink(
        initial_position=(*DEFAULT_MOCK_LOCATIONS[0].position, 0.0),
        noise_std=args.noise_std,
        seed=seed,
    )
    camera = MockCamera(link, DEFAULT_MOCK_LOCATIONS, seed=seed)
    config = _mission_config(settings, args)
    result = Mission(link, camera, config=config).run(args.log_path)
    print(
        json.dumps(
            {
                "status": result["status"],
                "log_path": str(args.log_path),
                "route": result["qaoa"]["route_labels"],
                "classical_fallback": result["classical_fallback"],
                "mean_error_without_correction_m": result["correction_summary"][
                    "mean_error_without_correction_m"
                ],
                "mean_error_after_correction_m": result["correction_summary"][
                    "mean_error_after_correction_m"
                ],
                "correction_improved": result["correction_summary"]["improved"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def _mission_config(settings: AppSettings, args: argparse.Namespace) -> MissionConfig:
    seed = settings.route.seed if args.seed is None else args.seed
    return MissionConfig(
        takeoff_height_m=settings.flight.altitude_m,
        speed_mps=settings.flight.speed_mps,
        hover_seconds=settings.flight.hover_seconds,
        penalty=settings.route.penalty,
        qaoa_reps=settings.route.reps,
        qaoa_shots=settings.route.shots if args.qaoa_shots is None else args.qaoa_shots,
        qaoa_maxiter=(
            settings.route.maxiter if args.qaoa_maxiter is None else args.qaoa_maxiter
        ),
        qaoa_seed=seed,
    )


def _camera_from_settings(settings: AppSettings) -> OpenCVCameraInput:
    camera_settings = settings.camera
    mapper = FloorMapper(
        field_marker_ids=camera_settings.field_marker_ids,
        field_marker_positions=camera_settings.field_marker_positions_m,
        field_size_m=camera_settings.field_size_m,
        marker_size_m=camera_settings.marker_size_m,
        aruco_dictionary=camera_settings.aruco_dictionary,
        camera_height_m=camera_settings.height_m,
        drone_marker_id=camera_settings.drone_marker_id,
        camera_center_floor=camera_settings.camera_center_floor_m,
    )

    def detector_factory() -> TargetDetector:
        return TargetDetector(
            model_path=camera_settings.target_model,
            target_classes=camera_settings.target_classes,
            confidence=camera_settings.target_confidence,
        )

    camera_source = None
    if camera_settings.avfoundation_device is None:
        camera_source = (
            camera_settings.device_index
            if camera_settings.device_index is not None
            else camera_settings.stream_url
        )
    if camera_source is None and camera_settings.avfoundation_device is None:
        raise ValueError("camera source is not configured")
    return OpenCVCameraInput(
        camera_source,
        mapper,
        detector_factory,
        camera_height_m=camera_settings.height_m,
        drone_height_m=settings.flight.altitude_m,
        target_aggregation_frames=camera_settings.target_aggregation_frames,
        required_target_classes=camera_settings.required_target_classes,
        start_class=camera_settings.start_class,
        start_label=camera_settings.start_label,
        start_from_drone_marker=camera_settings.start_from_drone_marker,
        start_marker_height_m=camera_settings.start_marker_height_m,
        max_frame_reads=camera_settings.max_frame_reads,
        avfoundation_device=camera_settings.avfoundation_device,
        avfoundation_framerate=camera_settings.avfoundation_framerate,
    )


def _check_camera(settings: AppSettings) -> None:
    camera = _camera_from_settings(settings)
    try:
        frame = camera.calibrate()
        print(
            json.dumps(
                {
                    "camera_check": "ok",
                    "frame_height": int(frame.shape[0]),
                    "frame_width": int(frame.shape[1]),
                    "aruco_calibrated": bool(camera.mapper.calibrated),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    finally:
        camera.close()


def _stampfly_link_from_settings(settings: AppSettings) -> StampFlyUdpLink:
    drone = settings.drone
    return StampFlyUdpLink(
        host=drone.host,
        command_port=drone.command_port,
        state_port=drone.state_port,
        timeout_s=drone.timeout_s,
        retries=drone.retries,
        go_min_cm=drone.go_min_cm,
        go_max_cm=drone.go_max_cm,
        speed_min_cm_s=drone.speed_min_cm_s,
        speed_max_cm_s=drone.speed_max_cm_s,
    )


if __name__ == "__main__":
    raise SystemExit(main())
