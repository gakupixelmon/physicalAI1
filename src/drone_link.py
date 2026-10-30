"""Safe drone-link abstractions and deterministic/mockable transports."""

from __future__ import annotations

from abc import ABC, abstractmethod
import socket
from typing import Any, Sequence

import numpy as np


class DroneLink(ABC):
    """Transport-independent interface used by the mission controller."""

    @abstractmethod
    def connect(self) -> None:
        """Connect to the drone transport."""

    @abstractmethod
    def takeoff(self, height: float = 0.5) -> None:
        """Take off to ``height`` metres."""

    @abstractmethod
    def land(self) -> None:
        """Land at the current horizontal position."""

    @abstractmethod
    def emergency(self) -> None:
        """Perform the transport's emergency action."""

    @abstractmethod
    def go(self, dx: float, dy: float, dz: float, speed: float) -> Sequence[float]:
        """Move by a relative displacement and return the new position."""


class DroneLinkError(RuntimeError):
    """A command was rejected or the configured transport failed."""


class MockDroneLink(DroneLink):
    """In-memory drone link with Gaussian per-command position drift."""

    def __init__(
        self,
        initial_position: Sequence[float] = (0.0, 0.0, 0.0),
        *,
        noise_std: float = 0.04,
        seed: int | None = 3,
    ) -> None:
        position = np.asarray(initial_position, dtype=float)
        if position.shape != (3,):
            raise ValueError("initial_position must be a 3D point")
        if not np.all(np.isfinite(position)):
            raise ValueError("initial_position must contain finite values")
        if noise_std < 0 or not np.isfinite(noise_std):
            raise ValueError("noise_std must be finite and non-negative")

        self._position = position.copy()
        self.noise_std = float(noise_std)
        self._rng = np.random.default_rng(seed)
        self._connected = False
        self._airborne = False
        self._history: list[dict[str, Any]] = []

    @property
    def position(self) -> np.ndarray:
        """Current simulated position as ``(x, y, z)`` in metres."""

        return self._position.copy()

    @property
    def connected(self) -> bool:
        return self._connected

    @property
    def airborne(self) -> bool:
        return self._airborne

    @property
    def history(self) -> list[dict[str, Any]]:
        return [dict(event) for event in self._history]

    def connect(self) -> None:
        self._connected = True
        self._history.append({"event": "connect"})

    def takeoff(self, height: float = 0.5) -> None:
        self._require_connected()
        if height <= 0 or not np.isfinite(height):
            raise ValueError("takeoff height must be positive")
        self._position[2] = float(height)
        self._airborne = True
        self._history.append({"event": "takeoff", "height": float(height)})

    def land(self) -> None:
        self._require_connected()
        self._position[2] = 0.0
        self._airborne = False
        self._history.append({"event": "land", "position": self.position.tolist()})

    def emergency(self) -> None:
        # The mock models emergency as an immediate landing and records it.
        self._position[2] = 0.0
        self._airborne = False
        self._history.append({"event": "emergency", "position": self.position.tolist()})

    def go(self, dx: float, dy: float, dz: float, speed: float) -> Sequence[float]:
        self._require_connected()
        if not self._airborne:
            raise RuntimeError("cannot go while the mock drone is not airborne")
        displacement = np.asarray([dx, dy, dz], dtype=float)
        if displacement.shape != (3,) or not np.all(np.isfinite(displacement)):
            raise ValueError("go displacement must contain three finite values")
        if speed <= 0 or not np.isfinite(speed):
            raise ValueError("speed must be positive")

        drift = self._rng.normal(0.0, self.noise_std, size=3)
        self._position += displacement + drift
        self._history.append(
            {
                "event": "go",
                "command": displacement.tolist(),
                "speed": float(speed),
                "drift": drift.tolist(),
                "position": self.position.tolist(),
            }
        )
        return self.position

    def _require_connected(self) -> None:
        if not self._connected:
            raise RuntimeError("mock drone is not connected")


class StampFlyUdpLink(DroneLink):
    """StampFly Ecosystem's Tello-compatible UDP command link.

    The host is deliberately required rather than hard-coded.  Put the local
    vehicle address in ``config/settings.yaml`` and pass it to this class.
    ``dx``, ``dy``, ``dz`` are metres at this project boundary and are converted
    to the API's centimetres; ``speed`` is metres per second and is converted
    to centimetres per second.
    """

    MIN_GO_CM = 10
    MAX_GO_CM = 300
    MIN_SPEED_CM_S = 10
    MAX_SPEED_CM_S = 100

    def __init__(
        self,
        host: str,
        *,
        command_port: int = 8889,
        state_port: int = 8890,
        timeout_s: float = 1.0,
        retries: int = 2,
        go_min_cm: int = MIN_GO_CM,
        go_max_cm: int = MAX_GO_CM,
        speed_min_cm_s: int = MIN_SPEED_CM_S,
        speed_max_cm_s: int = MAX_SPEED_CM_S,
        initial_position: Sequence[float] = (0.0, 0.0, 0.0),
        socket_factory: Any = socket.socket,
    ) -> None:
        if not host or host.startswith("CHANGE_ME"):
            raise ValueError("a local StampFly host is required")
        if not 1 <= command_port <= 65535:
            raise ValueError("command_port must be in the range 1..65535")
        if not 1 <= state_port <= 65535:
            raise ValueError("state_port must be in the range 1..65535")
        if timeout_s <= 0 or retries < 0:
            raise ValueError("timeout_s must be positive and retries non-negative")
        if not 1 <= go_min_cm <= go_max_cm:
            raise ValueError("go_min_cm/go_max_cm must be positive and ordered")
        if not 1 <= speed_min_cm_s <= speed_max_cm_s:
            raise ValueError("speed_min_cm_s/speed_max_cm_s must be positive and ordered")
        position = np.asarray(initial_position, dtype=float)
        if position.shape != (3,) or not np.all(np.isfinite(position)):
            raise ValueError("initial_position must be a finite 3D point")

        self.host = host
        self.command_port = int(command_port)
        self.state_port = int(state_port)
        self.timeout_s = float(timeout_s)
        self.retries = int(retries)
        self.go_min_cm = int(go_min_cm)
        self.go_max_cm = int(go_max_cm)
        self.speed_min_cm_s = int(speed_min_cm_s)
        self.speed_max_cm_s = int(speed_max_cm_s)
        self._position_estimate = position.copy()
        self._socket_factory = socket_factory
        self._socket: socket.socket | None = None
        self._connected = False
        self._airborne = False
        self.last_response: str | None = None

    @property
    def position(self) -> np.ndarray:
        """Open-loop position estimate; use the camera for measured position."""

        return self._position_estimate.copy()

    @property
    def connected(self) -> bool:
        return self._connected

    @property
    def airborne(self) -> bool:
        return self._airborne

    def connect(self) -> None:
        if self._connected:
            return
        sock = self._socket_factory(socket.AF_INET, socket.SOCK_DGRAM)
        sock.settimeout(self.timeout_s)
        self._socket = sock
        try:
            response = self._exchange("command")
            self._expect_ok("command", response)
            self._connected = True
        except Exception:
            self.close()
            raise

    def close(self) -> None:
        if self._socket is not None:
            self._socket.close()
        self._socket = None
        self._connected = False

    def takeoff(self, height: float = 0.5) -> None:
        self._require_connected()
        if height <= 0 or not np.isfinite(height):
            raise ValueError("takeoff height must be positive")
        # UDP 8889 is the Tello-compatible API. Its takeoff command has no
        # altitude argument; ``sf takeoff [alt]`` belongs to the separate
        # Wi-Fi CLI protocol and must not be sent to this UDP endpoint.
        response = self._exchange("takeoff")
        self._expect_ok("takeoff", response)
        self._position_estimate[2] = float(height)
        self._airborne = True

    def land(self) -> None:
        self._require_connected()
        response = self._exchange("land")
        self._expect_ok("land", response)
        self._position_estimate[2] = 0.0
        self._airborne = False

    def emergency(self) -> None:
        if self._socket is None:
            self._airborne = False
            return
        response = self._exchange("emergency")
        self._expect_ok("emergency", response)
        self._airborne = False

    def go(self, dx: float, dy: float, dz: float, speed: float) -> Sequence[float]:
        self._require_connected()
        if not self._airborne:
            raise RuntimeError("cannot go while the drone is not airborne")
        displacement_m = np.asarray([dx, dy, dz], dtype=float)
        if displacement_m.shape != (3,) or not np.all(np.isfinite(displacement_m)):
            raise ValueError("go displacement must contain three finite values")
        displacement_cm = np.rint(displacement_m * 100.0).astype(int)
        distance_cm = float(np.linalg.norm(displacement_cm))
        if not self.go_min_cm <= distance_cm <= self.go_max_cm:
            raise ValueError(
                f"go displacement must be {self.go_min_cm}..{self.go_max_cm} cm "
                f"in norm; got {distance_cm:.1f} cm"
            )
        speed_cm_s = int(round(float(speed) * 100.0))
        if not self.speed_min_cm_s <= speed_cm_s <= self.speed_max_cm_s:
            raise ValueError(
                f"speed must be {self.speed_min_cm_s}..{self.speed_max_cm_s} cm/s"
            )
        command = "go {} {} {} {}".format(*displacement_cm.tolist(), speed_cm_s)
        response = self._exchange(command)
        self._expect_ok(command, response)
        self._position_estimate += displacement_cm.astype(float) / 100.0
        return self.position

    def send_raw(self, command: str) -> str:
        """Send a documented text command, primarily for diagnostics/tests."""

        self._require_connected()
        response = self._exchange(command)
        self._expect_ok(command, response)
        return response

    def _exchange(self, command: str) -> str:
        if self._socket is None:
            raise RuntimeError("StampFly UDP link is not connected")
        payload = command.strip().encode("ascii")
        destination = (self.host, self.command_port)
        last_error: Exception | None = None
        for _ in range(self.retries + 1):
            try:
                self._socket.sendto(payload, destination)
                data, _ = self._socket.recvfrom(4096)
                response = data.decode("utf-8", errors="replace").strip()
                self.last_response = response
                return response
            except socket.timeout as exc:
                last_error = exc
        raise DroneLinkError(
            f"no response from StampFly UDP command endpoint after "
            f"{self.retries + 1} attempts: {command}"
        ) from last_error

    @staticmethod
    def _expect_ok(command: str, response: str) -> None:
        if response.lower().startswith("error"):
            raise DroneLinkError(f"StampFly rejected {command!r}: {response}")
        if response.lower() != "ok":
            raise DroneLinkError(f"unexpected response to {command!r}: {response!r}")

    def _require_connected(self) -> None:
        if not self._connected or self._socket is None:
            raise RuntimeError("StampFly UDP link is not connected")

