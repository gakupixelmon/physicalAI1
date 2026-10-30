"""Generate printable ArUco markers and optical-flow textures.

Every output is an A4-sized PNG at 150 dpi.  The four field markers and the
drone marker are emitted as separate pages so that a 150 mm field marker can
be printed at its intended physical size.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from PIL import Image


A4_WIDTH_MM = 210.0
A4_HEIGHT_MM = 297.0
DEFAULT_DPI = 150
MARKER_DICTIONARY_NAME = "DICT_4X4_50"


def a4_pixels(dpi: int = DEFAULT_DPI) -> tuple[int, int]:
    """Return A4 width and height in pixels for ``dpi``."""

    if dpi <= 0:
        raise ValueError("dpi must be positive")
    return (
        round(A4_WIDTH_MM / 25.4 * dpi),
        round(A4_HEIGHT_MM / 25.4 * dpi),
    )


def mm_to_pixels(length_mm: float, dpi: int = DEFAULT_DPI) -> int:
    if length_mm <= 0:
        raise ValueError("length_mm must be positive")
    return round(length_mm / 25.4 * dpi)


def _dictionary(name: str = MARKER_DICTIONARY_NAME):
    try:
        dictionary_id = getattr(cv2.aruco, name)
    except AttributeError as exc:
        raise ValueError(f"unknown ArUco dictionary: {name}") from exc
    return cv2.aruco.getPredefinedDictionary(dictionary_id)


def make_aruco_page(
    marker_id: int,
    marker_size_mm: float,
    *,
    dpi: int = DEFAULT_DPI,
    dictionary_name: str = MARKER_DICTIONARY_NAME,
) -> np.ndarray:
    """Create a centered black-on-white ArUco marker page."""

    page_width, page_height = a4_pixels(dpi)
    marker_size_px = mm_to_pixels(marker_size_mm, dpi)
    if marker_size_px >= min(page_width, page_height):
        raise ValueError("marker does not fit on an A4 page")

    dictionary = _dictionary(dictionary_name)
    if not 0 <= marker_id < dictionary.bytesList.shape[0]:
        raise ValueError(f"marker_id {marker_id} is not in {dictionary_name}")

    if hasattr(cv2.aruco, "generateImageMarker"):
        marker = cv2.aruco.generateImageMarker(
            dictionary,
            marker_id,
            marker_size_px,
            borderBits=1,
        )
    else:  # OpenCV versions before the newer convenience function.
        marker = np.zeros((marker_size_px, marker_size_px), dtype=np.uint8)
        cv2.aruco.drawMarker(dictionary, marker_id, marker_size_px, marker, 1)

    page = np.full((page_height, page_width), 255, dtype=np.uint8)
    x0 = (page_width - marker_size_px) // 2
    y0 = (page_height - marker_size_px) // 2
    page[y0 : y0 + marker_size_px, x0 : x0 + marker_size_px] = marker
    return page


def make_flow_pattern(
    index: int,
    *,
    dpi: int = DEFAULT_DPI,
    seed: int = 20261026,
) -> np.ndarray:
    """Create one repeatable, high-contrast random texture page."""

    if index < 0:
        raise ValueError("index must be non-negative")
    width, height = a4_pixels(dpi)
    rng = np.random.default_rng(seed + index)
    page = np.full((height, width), 255, dtype=np.uint8)

    # A coarse random grid gives optical flow features at several spatial
    # scales without producing a fragile single-pixel noise pattern.
    margin = mm_to_pixels(12.0, dpi)
    inner_width = width - 2 * margin
    inner_height = height - 2 * margin
    coarse = rng.integers(0, 2, size=(44, 31), dtype=np.uint8) * 255
    texture = cv2.resize(
        coarse,
        (inner_width, inner_height),
        interpolation=cv2.INTER_NEAREST,
    )
    page[margin : margin + inner_height, margin : margin + inner_width] = texture

    # Add a few larger shapes so adjacent pages are not visually periodic.
    for _ in range(80):
        center = (
            int(rng.integers(margin, width - margin)),
            int(rng.integers(margin, height - margin)),
        )
        radius = int(rng.integers(mm_to_pixels(1.5, dpi), mm_to_pixels(7, dpi)))
        color = int(rng.choice([0, 255]))
        cv2.circle(page, center, radius, color, thickness=-1)
    return page


def _write(path: Path, image: np.ndarray, *, dpi: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(path, format="PNG", dpi=(dpi, dpi))
    return path


def generate_printables(
    output_dir: str | Path,
    *,
    dpi: int = DEFAULT_DPI,
    flow_count: int = 15,
    flow_seed: int = 20261026,
) -> list[Path]:
    """Generate all requested pages and return their paths."""

    if flow_count < 1:
        raise ValueError("flow_count must be positive")
    output = Path(output_dir)
    paths: list[Path] = []

    for marker_id in range(4):
        paths.append(
            _write(
                output / f"field_marker_id_{marker_id}_150mm.png",
                make_aruco_page(marker_id, 150.0, dpi=dpi),
                dpi=dpi,
            )
        )
    paths.append(
        _write(
            output / "drone_marker_id_10_50mm.png",
            make_aruco_page(10, 50.0, dpi=dpi),
            dpi=dpi,
        )
    )
    for index in range(flow_count):
        paths.append(
            _write(
                output / f"flow_pattern_{index + 1:02d}.png",
                make_flow_pattern(index, dpi=dpi, seed=flow_seed),
                dpi=dpi,
            )
        )
    return paths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/printables"),
        help="directory for generated PNG pages (default: results/printables)",
    )
    parser.add_argument("--dpi", type=int, default=DEFAULT_DPI)
    parser.add_argument("--flow-count", type=int, default=15)
    parser.add_argument("--seed", type=int, default=20261026)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    paths = generate_printables(
        args.output_dir,
        dpi=args.dpi,
        flow_count=args.flow_count,
        flow_seed=args.seed,
    )
    for path in paths:
        print(path)
    print(f"generated {len(paths)} PNG files at {args.dpi} dpi")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
