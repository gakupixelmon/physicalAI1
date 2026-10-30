"""Run the route planner five times on a reproducible random 4-city problem."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.qaoa_route import solve_qaoa  # noqa: E402


def main() -> None:
    rng = np.random.default_rng(20261026)
    points = rng.uniform(0.0, 1.0, size=(4, 2))
    distances = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)

    print("random 4-location distance matrix")
    print(np.array2string(distances, precision=4, suppress_small=True))
    print("\nrun  route_length  approx_ratio  feasible_fraction  optimal_fraction  qubits  runtime_s  fallback")
    for run in range(1, 6):
        result = solve_qaoa(
            distances,
            reps=2,
            shots=2048,
            seed=1000 + run,
            maxiter=80,
        )
        print(
            f"{run:>3}  {result['route_length']:>12.6f}  "
            f"{result['approximation_ratio']:>13.6f}  "
            f"{result['feasible_fraction']:>17.6f}  "
            f"{result['optimal_fraction']:>16.6f}  "
            f"{result['num_qubits']:>6}  "
            f"{result['runtime_seconds']:>10.3f}  "
            f"{str(result['used_fallback']):>8}"
        )


if __name__ == "__main__":
    main()
