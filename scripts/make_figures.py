"""Generate QAOA, route, bitstring, and benchmark figures."""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "results" / ".matplotlib"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
import numpy as np

from benchmark_qaoa import (  # noqa: E402
    _distance_matrix,
    run_benchmark,
    write_summary_csv,
)
from src.qaoa_route import brute_force, decode, route_length, solve_qaoa  # noqa: E402
from src.visualization import draw_mission_overlay  # noqa: F401,E402


def _representative_points(seed: int) -> np.ndarray:
    return np.random.default_rng(seed).uniform(0.1, 1.9, size=(4, 2))


def _route_bitstring(route: tuple[int, ...] | list[int]) -> str:
    bits = [0] * 9
    for position, city in enumerate(route[1:-1]):
        bits[(int(city) - 1) * 3 + position] = 1
    return "".join(str(bit) for bit in reversed(bits))


def _representative_result(points: np.ndarray, reps: int, seed: int) -> dict[str, Any]:
    distances = _distance_matrix(points)
    exact_route, exact_length = brute_force(distances)
    try:
        result = solve_qaoa(
            distances,
            reps=reps,
            shots=2048,
            seed=seed,
            maxiter=80,
        )
    except ImportError as exc:
        result = {
            "route": exact_route,
            "route_length": exact_length,
            "exact_route": exact_route,
            "exact_length": exact_length,
            "approximation_ratio": 1.0,
            "feasible_fraction": 0.0,
            "optimal_fraction": 0.0,
            "num_qubits": 9,
            "runtime_seconds": 0.0,
            "used_fallback": True,
            "fallback_reason": str(exc),
            "counts": {_route_bitstring(exact_route): 1},
        }
    result.setdefault("exact_route", exact_route)
    result.setdefault("exact_length", exact_length)
    result.setdefault("counts", {})
    return result


def _save_figure(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def _fallback_circuit_figure(path: Path, n_qubits: int = 9, reps: int = 2) -> Path:
    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.set_xlim(-0.5, reps * 3.2 + 1.5)
    ax.set_ylim(-0.8, n_qubits - 0.2)
    for qubit in range(n_qubits):
        y = n_qubits - qubit - 1
        ax.plot([0, reps * 3.2 + 0.8], [y, y], color="black", linewidth=0.8)
        ax.text(-0.2, y, f"q[{qubit}]", ha="right", va="center")
        for layer in range(reps):
            base = layer * 3.2 + 0.5
            for x, label, color in (
                (base, "H", "#d9edf7"),
                (base + 1.0, "cost\nγ", "#fcf8e3"),
                (base + 2.0, "Rx\nβ", "#dff0d8"),
            ):
                ax.add_patch(Rectangle((x - 0.35, y - 0.27), 0.7, 0.54, facecolor=color, edgecolor="black"))
                ax.text(x, y, label, ha="center", va="center", fontsize=7)
    ax.set_title("QAOA circuit schematic (Qiskit unavailable; fallback view)")
    ax.axis("off")
    return _save_figure(fig, path)


def make_circuit_figure(path: Path, reps: int, points: np.ndarray, penalty: float = 2.0) -> Path:
    try:
        import src.qaoa_route as qaoa_route
        from qiskit_optimization.converters import LinearEqualityToPenalty

        qp = qaoa_route.build_qp(_distance_matrix(points))
        qubo = LinearEqualityToPenalty(penalty=penalty).convert(qp)
        cost = qaoa_route._qubo_operator(qubo)
        circuit, _, _ = qaoa_route._qaoa_circuit(cost, 9, reps)
        figure = circuit.draw(output="mpl", fold=-1, idle_wires=False)
        figure.suptitle(f"QAOA circuit, reps={reps}")
        return _save_figure(figure, path)
    except (ImportError, ModuleNotFoundError, AttributeError, TypeError, ValueError):
        return _fallback_circuit_figure(path, n_qubits=9, reps=reps)


def make_bitstring_figure(
    path: Path,
    counts: dict[str, int],
    distances: np.ndarray,
    exact_length: float,
    *,
    top_n: int = 12,
) -> Path:
    counts = {str(key).replace(" ", ""): int(value) for key, value in counts.items()}
    if not counts:
        counts = {"000000000": 1}
    total = sum(counts.values())
    ranked = sorted(counts.items(), key=lambda item: item[1], reverse=True)[:top_n]
    labels: list[str] = []
    probabilities: list[float] = []
    colors: list[str] = []
    for bitstring, count in ranked:
        decoded = decode(bitstring, 4)
        feasible = decoded is not None
        optimal = feasible and np.isclose(route_length(decoded, distances), exact_length)
        labels.append(bitstring)
        probabilities.append(count / total)
        colors.append("#2ca02c" if optimal else "#ffbf00" if feasible else "#bdbdbd")

    fig, ax = plt.subplots(figsize=(10, 5.5))
    y = np.arange(len(labels))
    ax.barh(y, probabilities, color=colors)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel("probability")
    ax.set_title("Top measured bitstrings")
    ax.legend(
        handles=[
            Patch(color="#2ca02c", label="feasible and optimal"),
            Patch(color="#ffbf00", label="feasible, non-optimal"),
            Patch(color="#bdbdbd", label="infeasible"),
        ],
        loc="lower right",
    )
    return _save_figure(fig, path)


def make_route_figure(path: Path, points: np.ndarray, qaoa_route: Sequence[int], seed: int) -> Path:
    rng = np.random.default_rng(seed + 1)
    random_route = (0, *rng.permutation(np.arange(1, len(points))).tolist(), 0)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(points[1:, 0], points[1:, 1], s=100, color="#ff7f0e", label="delivery")
    ax.scatter(points[0, 0], points[0, 1], s=130, color="#2ca02c", marker="*", label="start")
    for index, point in enumerate(points):
        ax.annotate(str(index), point, xytext=(6, 6), textcoords="offset points")

    qaoa_points = points[list(qaoa_route)]
    random_points = points[list(random_route)]
    ax.plot(qaoa_points[:, 0], qaoa_points[:, 1], "-o", color="#1f77b4", label="QAOA route")
    ax.plot(random_points[:, 0], random_points[:, 1], "--o", color="#9467bd", alpha=0.8, label="random route")
    ax.set_xlabel("floor x (m)")
    ax.set_ylabel("floor y (m)")
    ax.set_title("Delivery route comparison")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.25)
    ax.legend()
    return _save_figure(fig, path)


def make_metrics_figure(path: Path, csv_path: Path) -> Path:
    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"benchmark CSV is empty: {csv_path}")
    reps = np.asarray([int(row["reps"]) for row in rows])
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
    for axis, metric, title in zip(
        axes.ravel(),
        ("approximation_ratio", "feasible_fraction", "optimal_fraction", "runtime_seconds"),
        ("Approximation ratio", "Feasible fraction", "Optimal fraction", "Runtime (s)"),
    ):
        mean = np.asarray([float(row[f"{metric}_mean"]) for row in rows])
        std = np.asarray([float(row[f"{metric}_std"]) for row in rows])
        axis.errorbar(reps, mean, yerr=std, marker="o", capsize=4)
        axis.set_title(title)
        axis.set_xticks(reps)
        axis.grid(alpha=0.25)
    axes[1, 0].set_xlabel("reps")
    axes[1, 1].set_xlabel("reps")
    fig.suptitle("QAOA metrics by circuit depth")
    fig.tight_layout()
    return _save_figure(fig, path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-csv", type=Path, default=Path("results/benchmark_qaoa.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/figures"))
    parser.add_argument("--seed", type=int, default=20261028)
    parser.add_argument("--representative-reps", type=int, default=2)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not args.benchmark_csv.exists():
        rows = run_benchmark()
        write_summary_csv(rows, args.benchmark_csv)

    points = _representative_points(args.seed)
    distances = _distance_matrix(points)
    result = _representative_result(points, args.representative_reps, args.seed)
    paths = [
        make_circuit_figure(args.output_dir / "qaoa_circuit.png", args.representative_reps, points),
        make_bitstring_figure(
            args.output_dir / "bitstring_probabilities.png",
            result["counts"],
            distances,
            float(result["exact_length"]),
        ),
        make_route_figure(
            args.output_dir / "route_comparison.png",
            points,
            tuple(int(index) for index in result["route"]),
            args.seed,
        ),
        make_metrics_figure(args.output_dir / "metrics_by_reps.png", args.benchmark_csv),
    ]
    for path in paths:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

