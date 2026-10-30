import numpy as np

from src.qaoa_route import brute_force, build_qp, decode, route_length, solve_qaoa


def _assignment_bits(route: tuple[int, ...], n_locations: int, start: int = 0) -> list[int]:
    cities = [city for city in range(n_locations) if city != start]
    bits = [0] * ((n_locations - 1) ** 2)
    for position_index, city in enumerate(route[1:-1]):
        city_index = cities.index(city)
        bits[city_index * (n_locations - 1) + position_index] = 1
    return bits


def test_decode_converts_valid_assignment_and_rejects_invalid_assignment():
    route = (0, 2, 1, 3, 0)
    bits = _assignment_bits(route, n_locations=4)
    # Qiskit displays q_(n-1) on the left and q0 on the right.
    displayed = "".join(str(bit) for bit in reversed(bits))

    assert decode(displayed, 4) == route
    assert decode("0" * 9, 4) is None
    assert decode(bits, 4) == route


def test_brute_force_returns_hand_calculable_route_length():
    distances = np.array(
        [
            [0, 1, 4],
            [1, 0, 2],
            [4, 2, 0],
        ],
        dtype=float,
    )
    route, length = brute_force(distances)

    assert length == 7.0
    assert route_length(route, distances) == 7.0


def test_four_locations_use_nine_qubits():
    distances = np.ones((4, 4), dtype=float) - np.eye(4)
    qp = build_qp(distances)
    assert qp.get_num_vars() == 9


def test_square_qaoa_matches_exact_route_length_with_fixed_seed():
    distances = np.array(
        [
            [0.0, 1.0, np.sqrt(2.0), 1.0],
            [1.0, 0.0, 1.0, np.sqrt(2.0)],
            [np.sqrt(2.0), 1.0, 0.0, 1.0],
            [1.0, np.sqrt(2.0), 1.0, 0.0],
        ]
    )
    _, exact_length = brute_force(distances)
    result = solve_qaoa(distances, reps=2, shots=2048, seed=7, maxiter=80)

    assert result["num_qubits"] == 9
    assert result["route_length"] == exact_length

