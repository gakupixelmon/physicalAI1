"""Display StampFly-compatible UDP commands without contacting hardware."""

from __future__ import annotations

import argparse
import socket
from typing import Iterable


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="localhost", help="local bind host")
    parser.add_argument("--port", type=int, default=8889)
    parser.add_argument(
        "--reply-ok",
        action="store_true",
        help="reply with ok so StampFlyUdpLink can be tested locally",
    )
    parser.add_argument("--once", action="store_true", help="exit after one datagram")
    return parser


def serve(
    host: str = "localhost",
    port: int = 8889,
    *,
    reply_ok: bool = False,
    once: bool = False,
) -> None:
    if not 1 <= port <= 65535:
        raise ValueError("port must be in the range 1..65535")
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as server:
        server.bind((host, port))
        print(f"dummy UDP server listening on {host}:{port}", flush=True)
        while True:
            data, address = server.recvfrom(4096)
            command = data.decode("utf-8", errors="replace").strip()
            print(f"{address[0]}:{address[1]} -> {command}", flush=True)
            if reply_ok:
                server.sendto(b"ok\r\n", address)
            if once:
                return


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    serve(args.host, args.port, reply_ok=args.reply_ok, once=args.once)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

