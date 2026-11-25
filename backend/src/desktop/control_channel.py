"""Authenticated loopback TCP control channel for single-instance desktop signaling."""
from __future__ import annotations

import hmac
import json
import logging
import os
import secrets
import socket
import threading
from collections.abc import Callable
from typing import Any

from src.desktop.config import DesktopPaths
from src.desktop.io_utils import atomic_write_json

logger = logging.getLogger(__name__)


class ControlChannelServer:
    """Loopback TCP control server receiving authenticated `FOCUS` / `QUIT` commands."""

    def __init__(
        self,
        paths: DesktopPaths,
        on_focus: Callable[[], Any] | None = None,
        on_quit: Callable[[], Any] | None = None,
    ) -> None:
        self.paths = paths
        self.on_focus = on_focus
        self.on_quit = on_quit
        self.port: int | None = None
        self.token: str = ""
        self._socket: socket.socket | None = None
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()

    def start(self) -> int:
        """Bind `127.0.0.1:0`, publish `desktop_control.json`, and begin serving commands."""
        if self._socket is not None and self.port is not None:
            return self.port

        self._stop_event.clear()
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.bind(("127.0.0.1", 0))
        srv.listen(8)
        srv.settimeout(0.3)

        self.port = int(srv.getsockname()[1])
        self.token = secrets.token_hex(16)
        self._socket = srv

        atomic_write_json(
            self.paths.control_file_path,
            {"port": self.port, "token": self.token, "pid": os.getpid()},
        )

        self._thread = threading.Thread(
            target=self._serve_loop,
            name="desktop-control-channel",
            daemon=True,
        )
        self._thread.start()
        return self.port

    def _serve_loop(self) -> None:
        """Accept incoming loopback control connections until stopped."""
        srv = self._socket
        while not self._stop_event.is_set() and srv is not None:
            try:
                conn, _addr = srv.accept()
            except TimeoutError:
                continue
            except OSError:
                break

            with conn:
                self._handle_client(conn)

    def _handle_client(self, conn: socket.socket) -> None:
        """Read, authenticate, and dispatch a single control command."""
        conn.settimeout(2.0)
        buf = bytearray()
        parsed_payload: dict[str, Any] | None = None

        while len(buf) < 4096:
            try:
                chunk = conn.recv(1024)
            except (TimeoutError, OSError):
                break
            if not chunk:
                break
            buf.extend(chunk)
            text = buf.decode("utf-8", errors="replace").strip()
            if text:
                try:
                    candidate = json.loads(text)
                    if isinstance(candidate, dict):
                        parsed_payload = candidate
                        break
                except ValueError:
                    pass
            if b"\n" in buf:
                break

        if parsed_payload is None:
            try:
                candidate = json.loads(buf.decode("utf-8", errors="replace").strip())
                if isinstance(candidate, dict):
                    parsed_payload = candidate
            except ValueError:
                parsed_payload = None

        if parsed_payload is None:
            self._send_response(conn, {"ok": False, "error": "invalid_payload"})
            return

        provided_token = str(parsed_payload.get("token", ""))
        if not hmac.compare_digest(provided_token, self.token):
            self._send_response(conn, {"ok": False, "error": "unauthorized"})
            return

        command = str(parsed_payload.get("command", "")).strip().upper()
        if command == "FOCUS":
            if self.on_focus is not None:
                try:
                    self.on_focus()
                except Exception as exc:
                    logger.warning("Control channel on_focus callback failed: %s", exc)
            self._send_response(conn, {"ok": True, "command": "FOCUS"})
            return

        if command == "QUIT":
            if self.on_quit is not None:
                try:
                    self.on_quit()
                except Exception as exc:
                    logger.warning("Control channel on_quit callback failed: %s", exc)
            self._send_response(conn, {"ok": True, "command": "QUIT"})
            return

        self._send_response(conn, {"ok": False, "error": "unknown_command"})

    @staticmethod
    def _send_response(conn: socket.socket, body: dict[str, Any]) -> None:
        try:
            conn.sendall((json.dumps(body) + "\n").encode("utf-8"))
        except OSError:
            pass

    def stop(self) -> None:
        """Stop the control channel listener and delete `desktop_control.json`."""
        self._stop_event.set()
        srv = self._socket
        self._socket = None
        if srv is not None:
            try:
                srv.close()
            except OSError:
                pass

        if self._thread is not None and threading.current_thread() is not self._thread:
            self._thread.join(timeout=2.0)
        self._thread = None

        try:
            self.paths.control_file_path.unlink(missing_ok=True)
        except OSError:
            pass

    def __enter__(self) -> ControlChannelServer:
        self.start()
        return self

    def __exit__(self, *args: Any) -> None:
        self.stop()


def send_control_command(
    paths: DesktopPaths,
    command: str,
    timeout: float = 2.0,
    *,
    token_override: str | None = None,
) -> bool:
    """Send an authenticated `FOCUS` or `QUIT` command to a running instance."""
    ctl_path = paths.control_file_path
    if not ctl_path.is_file():
        return False

    try:
        meta = json.loads(ctl_path.read_text(encoding="utf-8"))
        port = int(meta["port"])
        token = str(meta["token"])
    except Exception:
        return False

    effective_token = token_override if token_override is not None else token
    payload = json.dumps({"command": command.strip().upper(), "token": effective_token}) + "\n"

    try:
        with socket.create_connection(("127.0.0.1", port), timeout=timeout) as sock:
            sock.settimeout(timeout)
            sock.sendall(payload.encode("utf-8"))
            raw = sock.recv(4096).decode("utf-8", errors="replace").strip()
            if not raw:
                return False
            resp = json.loads(raw)
            return bool(isinstance(resp, dict) and resp.get("ok"))
    except (OSError, TimeoutError, ValueError):
        return False
