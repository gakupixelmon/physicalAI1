"""Visualization helpers for mission videos and floor-coordinate overlays."""

from __future__ import annotations

from typing import Any, Iterable, Sequence

import cv2
import numpy as np

from .perception import FloorMapper


def _detection_value(detection: Any, name: str, default: Any = None) -> Any:
    if isinstance(detection, dict):
        return detection.get(name, default)
    return getattr(detection, name, default)


def draw_mission_overlay(
    frame: np.ndarray,
    detections: Iterable[Any],
    mapper: FloorMapper,
    planned_route: Sequence[Sequence[float]],
    *,
    current_floor_position: Sequence[float] | None = None,
    route_index: int | None = None,
) -> np.ndarray:
    """Draw detections, floor coordinates, and a planned route on a frame.

    ``planned_route`` is a sequence of floor ``(x, y)`` points in metres.  The
    input frame is never modified; a new BGR frame is returned.  The function
    is intentionally independent of a video reader, so callers can use it in
    a live loop or while rendering recorded footage.
    """

    if not isinstance(frame, np.ndarray) or frame.ndim not in (2, 3):
        raise ValueError("frame must be a grayscale or color numpy array")
    output = frame.copy()
    if output.ndim == 2:
        output = cv2.cvtColor(output, cv2.COLOR_GRAY2BGR)

    height, width = output.shape[:2]
    route = np.asarray(planned_route, dtype=np.float32)
    if route.ndim != 2 or route.shape[1] != 2 or len(route) < 2:
        raise ValueError("planned_route must have shape (N, 2) with N >= 2")
    route_pixels = mapper.to_image(route)
    route_points = np.rint(route_pixels).astype(int)
    for start, end in zip(route_points, route_points[1:]):
        cv2.line(
            output,
            tuple(start),
            tuple(end),
            (255, 0, 0),
            2,
            cv2.LINE_AA,
        )
    for index, point in enumerate(route_points):
        color = (0, 255, 255) if route_index == index else (255, 0, 0)
        cv2.circle(output, tuple(point), 7, color, -1, cv2.LINE_AA)
        cv2.putText(
            output,
            str(index),
            (int(point[0]) + 8, int(point[1]) - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )

    for detection in detections:
        bbox = _detection_value(detection, "bbox_xyxy")
        if bbox is None or len(bbox) != 4:
            continue
        x1, y1, x2, y2 = [int(round(float(value))) for value in bbox]
        cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 0), 2)
        center = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
        floor = mapper.to_floor(center)
        label = str(_detection_value(detection, "label", "target"))
        text = f"{label} ({floor[0]:.2f}, {floor[1]:.2f})m"
        cv2.putText(
            output,
            text,
            (max(0, x1), max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    if current_floor_position is not None:
        current_pixel = np.rint(mapper.to_image(current_floor_position)).astype(int)
        cv2.drawMarker(
            output,
            tuple(current_pixel),
            (0, 0, 255),
            cv2.MARKER_CROSS,
            24,
            3,
        )
        current = np.asarray(current_floor_position, dtype=float)
        cv2.putText(
            output,
            f"drone ({current[0]:.2f}, {current[1]:.2f})m",
            (10, height - 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

    # Keep the frame dimensions explicit for callers that use video writers.
    if output.shape[1] != width or output.shape[0] != height:
        raise RuntimeError("overlay changed the frame dimensions")
    return output


__all__ = ["draw_mission_overlay"]
