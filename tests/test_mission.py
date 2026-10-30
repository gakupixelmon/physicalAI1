import json

import numpy as np

from src.drone_link import MockDroneLink
from src.mission import (
    DEFAULT_MOCK_LOCATIONS,
    Mission,
    MissionConfig,
    MockCamera,
)


def test_mock_mission_completes_and_correction_improves_error(tmp_path, monkeypatch):
    def fake_solve_qaoa(distance_matrix, **kwargs):
        exact_length = float(
            sum(
                distance_matrix[a, b]
                for a, b in zip((0, 3, 2, 1, 0), (3, 2, 1, 0, 0))
            )
        )
        return {
            "route": (0, 3, 2, 1, 0),
            "route_length": exact_length,
            "exact_length": exact_length,
            "approximation_ratio": 1.0,
            "feasible_fraction": 1.0,
            "optimal_fraction": 1.0,
            "num_qubits": 9,
            "runtime_seconds": 0.001,
            "used_fallback": False,
        }

    import src.qaoa_route as qaoa_route

    monkeypatch.setattr(qaoa_route, "solve_qaoa", fake_solve_qaoa)
    link = MockDroneLink(
        initial_position=(*DEFAULT_MOCK_LOCATIONS[0].position, 0.0),
        noise_std=0.04,
        seed=3,
    )
    camera = MockCamera(link, DEFAULT_MOCK_LOCATIONS, seed=3)
    log_path = tmp_path / "run_log.json"

    result = Mission(
        link,
        camera,
        config=MissionConfig(qaoa_shots=1, qaoa_maxiter=1),
    ).run(log_path)

    assert result["completed"] is True
    assert result["flight"]["landed"] is True
    assert result["correction_summary"]["improved"] is True
    assert result["qaoa"]["route_labels"][0] == "start"
    assert log_path.exists()
    assert json.loads(log_path.read_text(encoding="utf-8"))["status"] == "completed"

