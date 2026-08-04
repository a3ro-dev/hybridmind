#!/usr/bin/env python3
"""Minimal Minecraft Java protocol maintenance endpoint.

It serves a status-ping MOTD and rejects login attempts while the real server
is intentionally offline.  It is deliberately dependency-free so it can run
on the existing Ubuntu host during a maintenance window.
"""
from __future__ import annotations

import json
import socketserver
import struct


MOTD = {"text": "§6§lUnder maintenance\n§7HybridMind benchmark in progress"}
KICK = {"text": "§6Under maintenance\n§7Please try again later."}


def read_varint(stream) -> int:
    value = shift = 0
    for _ in range(5):
        byte = stream.read(1)
        if not byte:
            raise EOFError
        value |= (byte[0] & 0x7F) << shift
        if not byte[0] & 0x80:
            return value
        shift += 7
    raise ValueError("VarInt too large")


def write_varint(value: int) -> bytes:
    result = bytearray()
    while True:
        byte = value & 0x7F
        value >>= 7
        result.append(byte | (0x80 if value else 0))
        if not value:
            return bytes(result)


def packet(packet_id: int, payload: bytes = b"") -> bytes:
    body = write_varint(packet_id) + payload
    return write_varint(len(body)) + body


def string(value: str) -> bytes:
    raw = value.encode("utf-8")
    return write_varint(len(raw)) + raw


class MaintenanceHandler(socketserver.StreamRequestHandler):
    def handle(self) -> None:
        try:
            _ = read_varint(self.rfile)  # handshake packet length
            if read_varint(self.rfile) != 0:
                return
            _ = read_varint(self.rfile)  # protocol version
            host_length = read_varint(self.rfile)
            self.rfile.read(host_length + 2)  # hostname and port
            next_state = read_varint(self.rfile)
            _ = read_varint(self.rfile)  # next packet length
            _ = read_varint(self.rfile)  # packet ID
            if next_state == 1:  # status ping
                status = json.dumps({"version": {"name": "Maintenance", "protocol": 767}, "players": {"max": 0, "online": 0}, "description": MOTD})
                self.wfile.write(packet(0, string(status)))
                self.wfile.flush()
                try:
                    _ = read_varint(self.rfile)
                    if read_varint(self.rfile) == 1:
                        self.wfile.write(packet(1, self.rfile.read(8)))
                except (EOFError, ValueError):
                    pass
            elif next_state == 2:  # login
                self.wfile.write(packet(0, string(json.dumps(KICK))))
            self.wfile.flush()
        except (EOFError, ValueError, OSError):
            return


class Server(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


if __name__ == "__main__":
    with Server(("0.0.0.0", 25565), MaintenanceHandler) as server:
        server.serve_forever()
