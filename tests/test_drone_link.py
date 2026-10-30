import numpy as np

import pytest

from src.drone_link import MockDroneLink, StampFlyUdpLink


def test_mock_drone_link_moves_and_records_gaussian_drift():
    link = MockDroneLink(noise_std=0.0, seed=1)
    link.connect()
    link.takeoff(0.5)
    position = np.asarray(link.go(1.0, -0.25, 0.0, 0.3))

    np.testing.assert_allclose(position, [1.0, -0.25, 0.5])
    assert [event["event"] for event in link.history] == ["connect", "takeoff", "go"]


class _FakeSocket:
    def __init__(self, *args):
        self.sent = []
        self.closed = False

    def settimeout(self, value):
        self.timeout = value

    def sendto(self, payload, destination):
        self.sent.append((payload.decode(), destination))

    def recvfrom(self, size):
        return b"ok\r\n", ("dummy", 8889)

    def close(self):
        self.closed = True


def test_stampfly_udp_link_formats_documented_commands_without_hardware():
    fake = _FakeSocket()
    link = StampFlyUdpLink(
        "local-stampfly-host",
        socket_factory=lambda *args: fake,
    )

    link.connect()
    link.takeoff(0.5)
    link.go(0.10, 0.0, 0.0, 0.10)
    link.land()

    assert [command for command, _ in fake.sent] == [
        "command",
        "takeoff",
        "go 10 0 0 10",
        "land",
    ]
    assert link.position.tolist() == [0.1, 0.0, 0.0]


def test_stampfly_udp_link_rejects_subminimum_motion():
    link = StampFlyUdpLink("local-stampfly-host", socket_factory=_FakeSocket)
    link.connect()
    link.takeoff()
    with pytest.raises(ValueError, match="10..300 cm"):
        link.go(0.05, 0.0, 0.0, 0.10)

