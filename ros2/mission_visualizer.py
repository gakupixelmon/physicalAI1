"""Replay a physicalAI mock mission as a 3D RViz scene."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time
from typing import Any, Sequence

import rclpy
from geometry_msgs.msg import Point
from rclpy.node import Node
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _point(values: Sequence[float]) -> tuple[float, float, float]:
    return (float(values[0]), float(values[1]), float(values[2]))


def _distance(a: Sequence[float], b: Sequence[float]) -> float:
    return math.sqrt(sum((float(x) - float(y)) ** 2 for x, y in zip(a, b)))


class MissionVisualizer(Node):
    """Publish planned and executed mock motion for RViz2."""

    def __init__(
        self,
        log_path: Path,
        *,
        playback_speed: float = 2.5,
        loop: bool = False,
    ) -> None:
        super().__init__("physicalai_mission_visualizer")
        if playback_speed <= 0:
            raise ValueError("playback_speed must be positive")

        self.data = json.loads(log_path.read_text(encoding="utf-8"))
        if self.data.get("mode") != "mock":
            raise ValueError("RViz playback requires a mock mission log")
        self.playback_speed = float(playback_speed)
        self.loop = loop
        self.locations = {
            item["label"]: _point((*item["position_m"], 0.0))
            for item in self.data["locations"]
        }
        self.route_labels = list(self.data["qaoa"]["route_labels"])
        self.route_z = self._takeoff_height()
        self.segments = self._build_segments()
        self.segment_ends: list[float] = []
        duration = 0.0
        for _, _, seconds, _ in self.segments:
            duration += seconds
            self.segment_ends.append(duration)
        self.duration = duration
        self.pause_after_landing = 2.0
        self.started_at = time.monotonic()
        self.trace: list[tuple[float, float, float]] = []
        self.last_trace_point: tuple[float, float, float] | None = None

        self.marker_pub = self.create_publisher(MarkerArray, "/physicalai/markers", 10)
        self.create_timer(0.1, self._publish_scene)
        self.get_logger().info(
            f"Loaded {log_path}; route: {' -> '.join(self.route_labels)}; "
            f"playback speed: {self.playback_speed:g}x"
        )

    def _takeoff_height(self) -> float:
        for event in self.data.get("mock_drone_history", []):
            if event.get("event") == "takeoff":
                return float(event["height"])
        return 0.5

    def _build_segments(
        self,
    ) -> list[tuple[tuple[float, float, float], tuple[float, float, float], float, str]]:
        start_label = self.route_labels[0]
        current = self.locations[start_label]
        history = self.data.get("mock_drone_history", [])
        arrivals = self.data.get("flight", {}).get("arrivals", [])
        move_labels: list[str] = []
        for arrival in arrivals:
            label = str(arrival.get("label", "target"))
            if arrival.get("return"):
                move_labels.append("Return to start")
            else:
                move_labels.append(f"Fly to {label}")
                move_labels.extend(
                    f"Position correction at {label}"
                    for _ in arrival.get("correction_attempts", [])
                )

        segments = []
        move_index = 0
        for event in history:
            kind = event.get("event")
            if kind == "takeoff":
                target = (current[0], current[1], float(event["height"]))
                label = "Takeoff"
                seconds = 1.0
            elif kind == "go":
                target = _point(event["position"])
                label = (
                    move_labels[move_index]
                    if move_index < len(move_labels)
                    else "Flight segment"
                )
                move_index += 1
                speed = max(float(event.get("speed", 0.3)), 0.05)
                seconds = max(0.7, _distance(current, target) / speed)
            elif kind == "land":
                target = _point(event["position"])
                label = "Landing"
                seconds = 1.0
            else:
                continue

            segments.append(
                (current, target, seconds / self.playback_speed, label)
            )
            current = target
        if not segments:
            raise ValueError("mock mission log has no simulated movement history")
        return segments

    def _position_at(
        self, elapsed: float
    ) -> tuple[tuple[float, float, float], str, bool]:
        if elapsed >= self.duration:
            return self.segments[-1][1], "Landed", True

        segment_start = 0.0
        for (start, end, duration, label), segment_end in zip(
            self.segments, self.segment_ends
        ):
            if elapsed <= segment_end:
                fraction = min(1.0, max(0.0, (elapsed - segment_start) / duration))
                position = tuple(
                    a + fraction * (b - a) for a, b in zip(start, end)
                )
                return position, label, False
            segment_start = segment_end
        return self.segments[-1][1], "Landed", True

    def _publish_scene(self) -> None:
        elapsed = time.monotonic() - self.started_at
        cycle_duration = self.duration + self.pause_after_landing
        if self.loop and elapsed >= cycle_duration:
            self.started_at = time.monotonic()
            self.trace.clear()
            self.last_trace_point = None
            elapsed = 0.0

        position, _, _ = self._position_at(elapsed)

        if (
            self.last_trace_point is None
            or _distance(position, self.last_trace_point) >= 0.025
        ):
            self.trace.append(position)
            self.last_trace_point = position

        now = self.get_clock().now().to_msg()
        array = MarkerArray()
        array.markers.extend(self._static_markers(now))
        array.markers.extend(self._drone_markers(now, position))
        array.markers.append(self._line_marker(
            now,
            "executed_trace",
            0,
            self.trace,
            width=0.035,
            color=(1.0, 0.42, 0.08, 1.0),
        ))
        self.marker_pub.publish(array)

    def _static_markers(self, stamp: Any) -> list[Marker]:
        points = [self.locations[label] for label in self.route_labels]
        all_xy = [point[:2] for point in self.locations.values()]
        min_x = min(point[0] for point in all_xy) - 0.18
        max_x = max(point[0] for point in all_xy) + 0.18
        min_y = min(point[1] for point in all_xy) - 0.18
        max_y = max(point[1] for point in all_xy) + 0.18
        floor = self._marker(
            stamp,
            "floor",
            0,
            Marker.CUBE,
            ((min_x + max_x) / 2, (min_y + max_y) / 2, -0.055),
            (max_x - min_x, max_y - min_y, 0.08),
            (0.22, 0.29, 0.37, 0.78),
        )
        route = self._line_marker(
            stamp,
            "planned_route",
            0,
            [(point[0], point[1], self.route_z) for point in points],
            width=0.035,
            color=(0.12, 0.78, 0.95, 1.0),
        )
        markers = [floor, route]
        for index, (label, point) in enumerate(self.locations.items()):
            is_start = label == self.route_labels[0]
            color = (1.0, 0.72, 0.12, 1.0) if is_start else (0.22, 0.9, 0.42, 1.0)
            markers.append(
                self._marker(
                    stamp,
                    "targets",
                    index,
                    Marker.SPHERE,
                    (point[0], point[1], 0.055),
                    (0.13, 0.13, 0.11),
                    color,
                )
            )
            markers.append(
                self._text_marker(
                    stamp,
                    "target_labels",
                    index,
                    label,
                    (1.0, 1.0, 1.0, 1.0),
                    (point[0], point[1], 0.24),
                    height=0.12,
                )
            )
        return markers

    def _drone_markers(
        self, stamp: Any, position: Sequence[float]
    ) -> list[Marker]:
        x, y, z = position
        body = self._marker(
            stamp,
            "drone",
            0,
            Marker.CUBE,
            position,
            (0.19, 0.14, 0.055),
            (0.2, 0.62, 1.0, 1.0),
        )
        arms = self._marker(
            stamp,
            "drone_arms",
            0,
            Marker.LINE_LIST,
            position,
            (0.025, 0.0, 0.0),
            (0.85, 0.9, 1.0, 1.0),
        )
        for dx, dy in ((0.15, 0.15), (0.15, -0.15), (-0.15, 0.15), (-0.15, -0.15)):
            arms.points.extend((self._ros_point((x, y, z)), self._ros_point((x + dx, y + dy, z))))
        rotors = []
        for index, (dx, dy) in enumerate(
            ((0.15, 0.15), (0.15, -0.15), (-0.15, 0.15), (-0.15, -0.15))
        ):
            rotors.append(
                self._marker(
                    stamp,
                    "rotors",
                    index,
                    Marker.SPHERE,
                    (x + dx, y + dy, z),
                    (0.075, 0.075, 0.035),
                    (0.98, 0.48, 0.16, 1.0),
                )
            )
        return [body, arms, *rotors]

    def _line_marker(
        self,
        stamp: Any,
        namespace: str,
        marker_id: int,
        points: Sequence[Sequence[float]],
        *,
        width: float,
        color: tuple[float, float, float, float],
    ) -> Marker:
        marker = self._marker(
            stamp,
            namespace,
            marker_id,
            Marker.LINE_STRIP,
            (0.0, 0.0, 0.0),
            (width, 0.0, 0.0),
            color,
        )
        marker.points = [self._ros_point(point) for point in points]
        return marker

    def _text_marker(
        self,
        stamp: Any,
        namespace: str,
        marker_id: int,
        text: str,
        color: tuple[float, float, float, float],
        position: Sequence[float],
        *,
        height: float,
    ) -> Marker:
        marker = self._marker(
            stamp,
            namespace,
            marker_id,
            Marker.TEXT_VIEW_FACING,
            position,
            (0.0, 0.0, height),
            color,
        )
        marker.text = text
        return marker

    def _marker(
        self,
        stamp: Any,
        namespace: str,
        marker_id: int,
        marker_type: int,
        position: Sequence[float],
        scale: Sequence[float],
        color: tuple[float, float, float, float],
    ) -> Marker:
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = stamp
        marker.ns = namespace
        marker.id = marker_id
        marker.type = marker_type
        marker.action = Marker.ADD
        marker.pose.position.x = float(position[0])
        marker.pose.position.y = float(position[1])
        marker.pose.position.z = float(position[2])
        marker.pose.orientation.w = 1.0
        marker.scale.x = float(scale[0])
        marker.scale.y = float(scale[1])
        marker.scale.z = float(scale[2])
        marker.color = ColorRGBA(r=color[0], g=color[1], b=color[2], a=color[3])
        marker.lifetime.sec = 0
        marker.lifetime.nanosec = 0
        return marker

    @staticmethod
    def _ros_point(position: Sequence[float]) -> Point:
        point = Point()
        point.x, point.y, point.z = (float(position[0]), float(position[1]), float(position[2]))
        return point


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--log",
        type=Path,
        default=PROJECT_ROOT / "results" / "ros2_simulation_log.json",
        help="mock mission JSON log to replay",
    )
    parser.add_argument("--playback-speed", type=float, default=2.5)
    parser.add_argument("--loop", action="store_true", help="repeat the playback")
    args, ros_args = parser.parse_known_args()

    if not args.log.is_file():
        parser.error(f"mission log does not exist: {args.log}")

    rclpy.init(args=ros_args)
    node = MissionVisualizer(
        args.log,
        playback_speed=args.playback_speed,
        loop=args.loop,
    )
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
