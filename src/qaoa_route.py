"""QAOA and exact solvers for a start-fixed travelling-salesperson route.

The binary variables are laid out as ``x(city, position)`` for every
non-start city and every non-start position.  For four locations this gives
3 * 3 = 9 qubits.  Qubit/variable index zero is the first variable in this
layout; a Qiskit measurement bit string is decoded from right to left because
the rightmost character represents qubit 0.
"""

from __future__ import annotations

import itertools
import logging
import time
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

try:
    from qiskit import QuantumCircuit
    from qiskit.circuit import ParameterVector
    from qiskit.circuit.library import PauliEvolutionGate
    from qiskit.primitives import StatevectorEstimator, StatevectorSampler
    from qiskit.quantum_info import SparsePauliOp
    from qiskit_optimization import QuadraticProgram
    from qiskit_optimization.converters import LinearEqualityToPenalty

    _QISKIT_AVAILABLE = True
    _QISKIT_IMPORT_ERROR: ImportError | None = None
except ImportError as exc:  # Keep exact helpers usable in a minimal environment.
    QuantumCircuit = None  # type: ignore[assignment]
    ParameterVector = None  # type: ignore[assignment]
    PauliEvolutionGate = None  # type: ignore[assignment]
    StatevectorEstimator = None  # type: ignore[assignment]
    StatevectorSampler = None  # type: ignore[assignment]
    SparsePauliOp = None  # type: ignore[assignment]
    QuadraticProgram = None  # type: ignore[assignment]
    LinearEqualityToPenalty = None  # type: ignore[assignment]
    _QISKIT_AVAILABLE = False
    _QISKIT_IMPORT_ERROR = exc


LOGGER = logging.getLogger(__name__)


def _require_qiskit() -> None:
    if not _QISKIT_AVAILABLE:
        raise ImportError(
            "QAOA requires qiskit>=2 and qiskit-optimization>=0.7; "
            "install requirements.txt first"
        ) from _QISKIT_IMPORT_ERROR


def _validate_distance_matrix(distance_matrix: Sequence[Sequence[float]]) -> np.ndarray:
    """Return a validated square floating-point distance matrix."""

    distances = np.asarray(distance_matrix, dtype=float)
    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square matrix")
    if distances.shape[0] < 2:
        raise ValueError("at least two locations are required")
    if not np.all(np.isfinite(distances)):
        raise ValueError("distance_matrix must contain only finite values")
    if np.any(distances < 0):
        raise ValueError("distance_matrix must not contain negative distances")
    return distances


def _normalise_distances(distances: np.ndarray) -> np.ndarray:
    """Normalize distances by their maximum value, preserving zero matrices."""

    maximum = float(np.max(distances))
    if maximum <= 0.0:
        return np.zeros_like(distances, dtype=float)
    return distances / maximum


def _non_start_cities(n_locations: int, start: int) -> list[int]:
    if not 0 <= start < n_locations:
        raise ValueError(f"start must be in [0, {n_locations})")
    return [city for city in range(n_locations) if city != start]


def _variable_name(city: int, position: int) -> str:
    return f"x_{city}_{position}"


def _variable_names(n_locations: int, start: int) -> list[str]:
    cities = _non_start_cities(n_locations, start)
    return [
        _variable_name(city, position)
        for city in cities
        for position in range(1, n_locations)
    ]


def build_qp(
    distance_matrix: Sequence[Sequence[float]],
    start: int = 0,
) -> QuadraticProgram:
    """Build the constrained start-fixed TSP quadratic program.

    The returned problem is the *original constrained problem*, not its QUBO
    penalty conversion.  Distances in its objective are normalized by the
    largest matrix entry.  There is one binary variable for each assignment
    of a non-start city to a non-start visit position.  The constraints enforce
    one city per position and one position per city.
    """

    _require_qiskit()
    distances = _validate_distance_matrix(distance_matrix)
    n_locations = distances.shape[0]
    cities = _non_start_cities(n_locations, start)
    positions = range(1, n_locations)
    normalized = _normalise_distances(distances)

    qp = QuadraticProgram(name="start_fixed_tsp")
    for city in cities:
        for position in positions:
            qp.binary_var(name=_variable_name(city, position))

    linear: dict[str, float] = {}
    quadratic: dict[tuple[str, str], float] = {}

    # Start -> first visited city.
    for city in cities:
        linear[_variable_name(city, 1)] = float(normalized[start, city])

    # Consecutive non-start positions.
    for position in range(1, n_locations - 1):
        next_position = position + 1
        for city in cities:
            for next_city in cities:
                coefficient = float(normalized[city, next_city])
                key = (
                    _variable_name(city, position),
                    _variable_name(next_city, next_position),
                )
                quadratic[key] = coefficient

    # Last visited city -> start.
    for city in cities:
        linear[_variable_name(city, n_locations - 1)] = float(
            linear.get(_variable_name(city, n_locations - 1), 0.0)
            + normalized[city, start]
        )

    qp.minimize(linear=linear, quadratic=quadratic)

    for city in cities:
        qp.linear_constraint(
            linear={_variable_name(city, position): 1.0 for position in positions},
            sense="==",
            rhs=1.0,
            name=f"city_{city}_once",
        )
    for position in positions:
        qp.linear_constraint(
            linear={_variable_name(city, position): 1.0 for city in cities},
            sense="==",
            rhs=1.0,
            name=f"position_{position}_once",
        )

    return qp


def decode(
    bitstring: str | Sequence[int] | Mapping[Any, int],
    n_locations: int,
    start: int = 0,
) -> tuple[int, ...] | None:
    """Decode an assignment into ``(start, ..., start)`` or return ``None``.

    A string is interpreted in Qiskit display order: its rightmost character
    is qubit 0.  A sequence is interpreted in qubit order (index 0 is qubit
    0).  Mapping inputs may use integer qubit indices.  Exactly one bit must
    be set in every city row and every position column.
    """

    if n_locations < 2:
        raise ValueError("at least two locations are required")
    cities = _non_start_cities(n_locations, start)
    n_qubits = (n_locations - 1) ** 2

    try:
        if isinstance(bitstring, str):
            cleaned = bitstring.replace(" ", "")
            if len(cleaned) != n_qubits or any(bit not in "01" for bit in cleaned):
                return None
            bits = [int(bit) for bit in reversed(cleaned)]
        elif isinstance(bitstring, Mapping):
            bits = [int(bitstring.get(index, 0)) for index in range(n_qubits)]
        else:
            bits = [int(bit) for bit in bitstring]
            if len(bits) != n_qubits or any(bit not in (0, 1) for bit in bits):
                return None
    except (TypeError, ValueError):
        return None

    if len(bits) != n_qubits or any(bit not in (0, 1) for bit in bits):
        return None

    ordered_cities: list[int] = []
    for position_index in range(n_locations - 1):
        selected = [
            cities[city_index]
            for city_index in range(n_locations - 1)
            if bits[city_index * (n_locations - 1) + position_index] == 1
        ]
        if len(selected) != 1:
            return None
        ordered_cities.append(selected[0])

    for city_index in range(n_locations - 1):
        row_sum = sum(
            bits[city_index * (n_locations - 1) + position_index]
            for position_index in range(n_locations - 1)
        )
        if row_sum != 1:
            return None

    return tuple([start, *ordered_cities, start])


def route_length(
    route: Sequence[int],
    distance_matrix: Sequence[Sequence[float]],
) -> float:
    """Return the length of a route, closing it if the final location differs."""

    distances = _validate_distance_matrix(distance_matrix)
    locations = [int(location) for location in route]
    if len(locations) < 2:
        raise ValueError("route must contain at least two locations")
    if locations[0] != locations[-1]:
        locations.append(locations[0])
    if any(location < 0 or location >= len(distances) for location in locations):
        raise ValueError("route contains an invalid location index")
    return float(sum(distances[a, b] for a, b in zip(locations, locations[1:])))


def brute_force(
    distance_matrix: Sequence[Sequence[float]],
    start: int = 0,
) -> tuple[tuple[int, ...], float]:
    """Return the exact shortest closed route and its length."""

    distances = _validate_distance_matrix(distance_matrix)
    cities = _non_start_cities(len(distances), start)
    best_route: tuple[int, ...] | None = None
    best_length = float("inf")
    for permutation in itertools.permutations(cities):
        candidate = (start, *permutation, start)
        length = route_length(candidate, distances)
        if length < best_length:
            best_route = candidate
            best_length = length
    assert best_route is not None
    return best_route, best_length


def _qubo_operator(
    qubo: QuadraticProgram,
) -> SparsePauliOp:
    """Convert an unconstrained binary QP objective to a Z Pauli operator."""

    n_qubits = qubo.get_num_vars()
    linear = np.asarray(qubo.objective.linear.to_array(), dtype=float)
    quadratic = np.asarray(
        qubo.objective.quadratic.to_array(symmetric=False), dtype=float
    )
    terms: dict[str, float] = {"I" * n_qubits: float(qubo.objective.constant)}

    def add_term(label: str, coefficient: float) -> None:
        terms[label] = terms.get(label, 0.0) + float(coefficient)

    def z_label(indices: Iterable[int]) -> str:
        label = ["I"] * n_qubits
        for index in indices:
            # Pauli labels are displayed q_(n-1) ... q_0.
            label[n_qubits - 1 - index] = "Z"
        return "".join(label)

    for index, coefficient in enumerate(linear):
        if coefficient:
            add_term("I" * n_qubits, coefficient / 2.0)
            add_term(z_label([index]), -coefficient / 2.0)

    # QuadraticProgram stores the upper triangle for x_i*x_j.  A diagonal
    # entry is x_i^2 and therefore contributes as a linear binary term.
    for i in range(n_qubits):
        diagonal = quadratic[i, i]
        if diagonal:
            add_term("I" * n_qubits, diagonal / 2.0)
            add_term(z_label([i]), -diagonal / 2.0)
        for j in range(i + 1, n_qubits):
            coefficient = quadratic[i, j]
            if coefficient:
                add_term("I" * n_qubits, coefficient / 4.0)
                add_term(z_label([i]), -coefficient / 4.0)
                add_term(z_label([j]), -coefficient / 4.0)
                add_term(z_label([i, j]), coefficient / 4.0)

    nonzero_terms = [(label, coefficient) for label, coefficient in terms.items() if coefficient]
    if not nonzero_terms:
        nonzero_terms = [("I" * n_qubits, 0.0)]
    return SparsePauliOp.from_list(nonzero_terms)


def _qaoa_circuit(
    cost_operator: SparsePauliOp,
    n_qubits: int,
    reps: int,
) -> tuple[QuantumCircuit, ParameterVector, ParameterVector]:
    gammas = ParameterVector("gamma", reps)
    betas = ParameterVector("beta", reps)
    circuit = QuantumCircuit(n_qubits)
    circuit.h(range(n_qubits))
    for layer in range(reps):
        circuit.append(
            PauliEvolutionGate(cost_operator, time=gammas[layer]),
            range(n_qubits),
        )
        for qubit in range(n_qubits):
            circuit.rx(2.0 * betas[layer], qubit)
    return circuit, gammas, betas


def _energy(
    estimator: StatevectorEstimator,
    circuit: QuantumCircuit,
    cost_operator: SparsePauliOp,
    gammas: ParameterVector,
    betas: ParameterVector,
    parameters: Sequence[float],
) -> float:
    reps = len(gammas)
    values = list(parameters)
    bindings = {
        **{gammas[index]: values[index] for index in range(reps)},
        **{betas[index]: values[reps + index] for index in range(reps)},
    }
    bound = circuit.assign_parameters(bindings)
    result = estimator.run([(bound, cost_operator)]).result()[0]
    return float(np.asarray(result.data.evs).reshape(-1)[0])


def _sample_counts(
    circuit: QuantumCircuit,
    gammas: ParameterVector,
    betas: ParameterVector,
    parameters: Sequence[float],
    shots: int,
    seed: int | None,
) -> Mapping[str, int]:
    reps = len(gammas)
    values = list(parameters)
    bindings = {
        **{gammas[index]: values[index] for index in range(reps)},
        **{betas[index]: values[reps + index] for index in range(reps)},
    }
    bound = circuit.assign_parameters(bindings)
    bound.measure_all()
    sampler = StatevectorSampler(default_shots=shots, seed=seed)
    result = sampler.run([bound], shots=shots).result()[0]
    # measure_all creates a classical register named ``meas`` in Qiskit 2.x.
    return result.data.meas.get_counts()


def solve_qaoa(
    distance_matrix: Sequence[Sequence[float]],
    *,
    start: int = 0,
    penalty: float = 2.0,
    reps: int = 2,
    shots: int = 4096,
    seed: int | None = 1234,
    maxiter: int = 100,
) -> dict[str, Any]:
    """Solve the route with QAOA and report route and evaluation metrics.

    ``penalty`` is applied only when converting the original constrained QP to
    an unconstrained QUBO.  Feasibility is checked against the original QP and
    then against :func:`decode`; the QUBO objective is never used as the
    feasibility test.
    """

    distances = _validate_distance_matrix(distance_matrix)
    if penalty <= 0:
        raise ValueError("penalty must be positive")
    if reps < 1:
        raise ValueError("reps must be at least 1")
    if shots < 1:
        raise ValueError("shots must be at least 1")

    started = time.perf_counter()
    qp = build_qp(distances, start=start)
    exact_route, exact_length = brute_force(distances, start=start)
    converter = LinearEqualityToPenalty(penalty=float(penalty))
    qubo = converter.convert(qp)
    cost_operator = _qubo_operator(qubo)
    n_qubits = qubo.get_num_vars()
    circuit, gammas, betas = _qaoa_circuit(cost_operator, n_qubits, reps)

    # scipy is intentionally imported here so the module can still be used for
    # build_qp/decode/brute_force in a minimal environment.
    from scipy.optimize import minimize

    rng = np.random.default_rng(seed)
    initial = np.concatenate(
        [rng.uniform(0.0, np.pi, reps), rng.uniform(0.0, np.pi / 2.0, reps)]
    )
    estimator = StatevectorEstimator()

    def objective(parameters: np.ndarray) -> float:
        return _energy(
            estimator,
            circuit,
            cost_operator,
            gammas,
            betas,
            parameters,
        )

    optimization = minimize(
        objective,
        initial,
        method="COBYLA",
        options={"maxiter": maxiter, "rhobeg": 0.5, "tol": 1e-4},
    )
    optimized_parameters = np.asarray(optimization.x, dtype=float)
    counts = _sample_counts(
        circuit,
        gammas,
        betas,
        optimized_parameters,
        shots,
        seed,
    )

    total_shots = int(sum(counts.values()))
    feasible_shots = 0
    optimal_shots = 0
    best_sampled_route: tuple[int, ...] | None = None
    best_sampled_length = float("inf")

    for raw_bitstring, count in counts.items():
        bitstring = str(raw_bitstring).replace(" ", "")
        bits = [int(bit) for bit in reversed(bitstring)] if bitstring else []
        is_original_qp_feasible = len(bits) == n_qubits and bool(
            qp.is_feasible(np.asarray(bits, dtype=float))
        )
        decoded = decode(bitstring, len(distances), start=start)
        if not is_original_qp_feasible or decoded is None:
            continue
        feasible_shots += int(count)
        sampled_length = route_length(decoded, distances)
        if np.isclose(sampled_length, exact_length, rtol=1e-9, atol=1e-9):
            optimal_shots += int(count)
        if sampled_length < best_sampled_length:
            best_sampled_route = decoded
            best_sampled_length = sampled_length

    used_fallback = best_sampled_route is None
    if used_fallback:
        LOGGER.warning(
            "QAOA produced no executable route; falling back to brute force exact solution."
        )
        selected_route = exact_route
        selected_length = exact_length
    else:
        selected_route = best_sampled_route
        selected_length = best_sampled_length

    elapsed = time.perf_counter() - started
    feasible_fraction = feasible_shots / total_shots if total_shots else 0.0
    optimal_fraction = optimal_shots / total_shots if total_shots else 0.0
    return {
        "route": selected_route,
        "route_length": float(selected_length),
        "exact_route": exact_route,
        "exact_length": float(exact_length),
        "approximation_ratio": float(exact_length / selected_length)
        if selected_length > 0
        else 1.0,
        "feasible_fraction": float(feasible_fraction),
        "optimal_fraction": float(optimal_fraction),
        "num_qubits": int(n_qubits),
        "runtime_seconds": float(elapsed),
        "used_fallback": bool(used_fallback),
        "counts": dict(counts),
        "qaoa_energy": float(optimization.fun),
        "optimizer_success": bool(optimization.success),
        "optimizer_message": str(optimization.message),
        "penalty": float(penalty),
        "reps": int(reps),
        "shots": int(shots),
        "seed": seed,
        "gammas": [str(parameter) for parameter in gammas],
        "betas": [str(parameter) for parameter in betas],
    }

