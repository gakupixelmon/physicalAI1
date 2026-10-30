"""Benchmark QAOA route metrics for reps=1..4 over random layouts."""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.qaoa_route import brute_force, solve_qaoa  # noqa: E402


METRICS = (
    "approximation_ratio",
    "feasible_fraction",
    "optimal_fraction",
    "runtime_seconds",
)


def _distance_matrix(points: np.ndarray) -> np.ndarray:
    differences = points[:, None, :] - points[None, :, :]
    return np.linalg.norm(differences, axis=2)


def _fallback_result(distances: np.ndarray) -> dict[str, Any]:
    started = time.perf_counter()
    route, exact_length = brute_force(distances)
    return {
        "route": route,
        "route_length": exact_length,
        "exact_length": exact_length,
        "approximation_ratio": 1.0,
        "feasible_fraction": 0.0,
        "optimal_fraction": 0.0,
        "runtime_seconds": time.perf_counter() - started,
        "num_qubits": 9,
        "used_fallback": True,
    }


def generate_layouts(repeats: int, seed: int) -> list[np.ndarray]:
    if repeats < 1:
        raise ValueError("repeats must be positive")
    rng = np.random.default_rng(seed)
    return [rng.uniform(0.1, 1.9, size=(4, 2)) for _ in range(repeats)]


def run_benchmark(
    *,
    repeats: int = 20,
    reps_values: Iterable[int] = (1, 2, 3, 4),
    seed: int = 20261028,
    shots: int = 512,
    maxiter: int = 60,
) -> list[dict[str, Any]]:
    """Run all conditions and return one aggregate row per ``reps`` value."""

    layouts = generate_layouts(repeats, seed)
    summary: list[dict[str, Any]] = []
    for reps in reps_values:
        values = {metric: [] for metric in METRICS}
        fallback_count = 0
        for trial, points in enumerate(layouts):
            distances = _distance_matrix(points)
            try:
                result = solve_qaoa(
                    distances,
                    reps=int(reps),
                    shots=shots,
                    seed=seed + trial + int(reps) * 1000,
                    maxiter=maxiter,
                )
            except ImportError:
                result = _fallback_result(distances)
            for metric in METRICS:
                values[metric].append(float(result[metric]))
            fallback_count += int(bool(result.get("used_fallback", False)))

        row: dict[str, Any] = {
            "reps": int(reps),
            "repeats": int(repeats),
            "fallback_runs": fallback_count,
        }
        for metric in METRICS:
            observations = np.asarray(values[metric], dtype=float)
            row[f"{metric}_mean"] = float(np.mean(observations))
            row[f"{metric}_std"] = float(np.std(observations, ddof=1))
        summary.append(row)
    return summary


def write_summary_csv(rows: list[dict[str, Any]], output_path: str | Path) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = ["reps", "repeats", "fallback_runs"]
    for metric in METRICS:
        fields.extend([f"{metric}_mean", f"{metric}_std"])
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20261028)
    parser.add_argument("--shots", type=int, default=512)
    parser.add_argument("--maxiter", type=int, default=60)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/benchmark_qaoa.csv"),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    rows = run_benchmark(
        repeats=args.repeats,
        seed=args.seed,
        shots=args.shots,
        maxiter=args.maxiter,
    )
    output = write_summary_csv(rows, args.output)
    print(f"saved {output}")
    print(
        "reps  approximation_ratio(mean±std)  feasible_fraction(mean±std)  "
        "optimal_fraction(mean±std)  runtime_seconds(mean±std)"
    )
    for row in rows:
        print(
            f"{row['reps']:>4}  "
            f"{row['approximation_ratio_mean']:.6f}±{row['approximation_ratio_std']:.6f}  "
            f"{row['feasible_fraction_mean']:.6f}±{row['feasible_fraction_std']:.6f}  "
            f"{row['optimal_fraction_mean']:.6f}±{row['optimal_fraction_std']:.6f}  "
            f"{row['runtime_seconds_mean']:.6f}±{row['runtime_seconds_std']:.6f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

