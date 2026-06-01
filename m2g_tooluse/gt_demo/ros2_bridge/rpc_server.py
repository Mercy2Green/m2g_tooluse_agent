from __future__ import annotations

import json
import queue
import socket
import socketserver
import threading
from dataclasses import asdict, is_dataclass
from typing import Any

from m2g_tooluse.gt_demo.types import SkillResult


def _json_default(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    return str(value)


class _GtDemoRpcHandler(socketserver.StreamRequestHandler):
    def handle(self) -> None:
        raw = self.rfile.readline().decode("utf-8").strip()
        if not raw:
            return
        try:
            payload = json.loads(raw)
            command = str(payload.get("command", "status"))
            response = self.server.owner.handle_rpc_command(command)  # type: ignore[attr-defined]
        except Exception as exc:
            response = {"success": False, "message": f"rpc error: {exc}", "state": "error"}
        self.wfile.write((json.dumps(response, default=_json_default) + "\n").encode("utf-8"))


class _ThreadedTcpServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True


class LocalGtDemoRpcServer:
    """Small localhost RPC server used between IsaacLab and the ROS2 node."""

    def __init__(self, host: str = "127.0.0.1", port: int = 8765):
        self.host = host
        self.port = int(port)
        self._queue: queue.Queue[str] = queue.Queue()
        self._lock = threading.Lock()
        self._busy = False
        self._last_result = SkillResult(success=True, message="gt_demo rpc server initialized")
        self._last_command = "idle"
        self._server = _ThreadedTcpServer((self.host, self.port), _GtDemoRpcHandler)
        self._server.owner = self  # type: ignore[attr-defined]
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    def start(self) -> None:
        self._thread.start()
        print(f"[GT_DEMO][RPC] listening on {self.host}:{self.port}")

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()

    def handle_rpc_command(self, command: str) -> dict[str, Any]:
        if command == "status":
            return self._status_response()
        with self._lock:
            self._queue.put(command)
            position = self._queue.qsize()
            busy = self._busy
        return {
            "success": True,
            "message": f"queued gt_demo command '{command}'",
            "state": "queued",
            "busy": busy,
            "queue_position": position,
            "last_command": self._last_command,
        }

    def pop_next_command(self) -> str | None:
        with self._lock:
            if self._busy:
                return None
            try:
                command = self._queue.get_nowait()
            except queue.Empty:
                return None
            self._busy = True
            self._last_command = command
            return command

    def finish_command(self, result: SkillResult) -> None:
        with self._lock:
            self._busy = False
            self._last_result = result

    def _status_response(self) -> dict[str, Any]:
        with self._lock:
            result = self._last_result
            return {
                "success": result.success,
                "message": result.message,
                "state": "busy" if self._busy else "idle",
                "busy": self._busy,
                "queued": self._queue.qsize(),
                "last_command": self._last_command,
                "data": result.data,
            }


def send_rpc_command(command: str, host: str = "127.0.0.1", port: int = 8765, timeout_s: float = 5.0) -> dict[str, Any]:
    payload = json.dumps({"command": command}).encode("utf-8") + b"\n"
    with socket.create_connection((host, int(port)), timeout=timeout_s) as sock:
        sock.sendall(payload)
        raw = sock.makefile("rb").readline().decode("utf-8")
    return json.loads(raw)
