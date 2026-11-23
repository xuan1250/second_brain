"""Win32 Job Object isolation, single-instance mutex, and authenticated loopback control channel."""
from __future__ import annotations

import ctypes
import hmac
import json
import logging
import os
import secrets
import socket
import subprocess
import sys
import tempfile
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

from src.desktop.config import DesktopPaths

logger = logging.getLogger(__name__)

DEFAULT_MUTEX_NAME = "Local\\SecondBrainDesktopMutex"
ERROR_ALREADY_EXISTS = 183
JOB_OBJECT_EXTENDED_LIMIT_INFORMATION_CLASS = 9
JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000
SYNCHRONIZE = 0x00100000


class JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
    """Win32 JOBOBJECT_BASIC_LIMIT_INFORMATION (64-bit ABI safe)."""

    _fields_ = [
        ("PerProcessUserTimeLimit", ctypes.c_int64),
        ("PerJobUserTimeLimit", ctypes.c_int64),
        ("LimitFlags", ctypes.c_uint32),
        ("MinimumWorkingSetSize", ctypes.c_size_t),
        ("MaximumWorkingSetSize", ctypes.c_size_t),
        ("ActiveProcessLimit", ctypes.c_uint32),
        ("Affinity", ctypes.c_size_t),
        ("PriorityClass", ctypes.c_uint32),
        ("SchedulingClass", ctypes.c_uint32),
    ]


class IO_COUNTERS(ctypes.Structure):
    """Win32 IO_COUNTERS (64-bit unsigned counters)."""

    _fields_ = [
        ("ReadOperationCount", ctypes.c_uint64),
        ("WriteOperationCount", ctypes.c_uint64),
        ("OtherOperationCount", ctypes.c_uint64),
        ("ReadTransferCount", ctypes.c_uint64),
        ("WriteTransferCount", ctypes.c_uint64),
        ("OtherTransferCount", ctypes.c_uint64),
    ]


class JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
    """Win32 JOBOBJECT_EXTENDED_LIMIT_INFORMATION (64-bit ABI safe)."""

    _fields_ = [
        ("BasicLimitInformation", JOBOBJECT_BASIC_LIMIT_INFORMATION),
        ("IoInfo", IO_COUNTERS),
        ("ProcessMemoryLimit", ctypes.c_size_t),
        ("JobMemoryLimit", ctypes.c_size_t),
        ("PeakProcessMemoryUsed", ctypes.c_size_t),
        ("PeakJobMemoryUsed", ctypes.c_size_t),
    ]


def _get_kernel32() -> ctypes.WinDLL:
    """Return kernel32 WinDLL configured with 64-bit-safe handle signatures."""
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

    kernel32.CreateJobObjectW.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p]
    kernel32.CreateJobObjectW.restype = ctypes.c_void_p

    kernel32.SetInformationJobObject.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_uint32,
    ]
    kernel32.SetInformationJobObject.restype = ctypes.c_int

    kernel32.AssignProcessToJobObject.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    kernel32.AssignProcessToJobObject.restype = ctypes.c_int

    kernel32.CreateMutexW.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_wchar_p]
    kernel32.CreateMutexW.restype = ctypes.c_void_p

    kernel32.OpenMutexW.argtypes = [ctypes.c_uint32, ctypes.c_int, ctypes.c_wchar_p]
    kernel32.OpenMutexW.restype = ctypes.c_void_p

    kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
    kernel32.CloseHandle.restype = ctypes.c_int

    return kernel32


class WindowsJobObject:
    """Win32 Job Object configured with `JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE`."""

    def __init__(self, name: str | None = None) -> None:
        self.name = name
        self._handle: ctypes.c_void_p | None = None
        self._kernel32: ctypes.WinDLL | None = None

        if sys.platform != "win32":
            return

        kernel32 = _get_kernel32()
        self._kernel32 = kernel32
        ctypes.set_last_error(0)
        raw_handle = kernel32.CreateJobObjectW(None, name)
        if not raw_handle:
            raise ctypes.WinError(ctypes.get_last_error())

        handle = ctypes.c_void_p(raw_handle)
        info = JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
        info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE

        ok = kernel32.SetInformationJobObject(
            handle,
            JOB_OBJECT_EXTENDED_LIMIT_INFORMATION_CLASS,
            ctypes.byref(info),
            ctypes.sizeof(info),
        )
        if not ok:
            err = ctypes.get_last_error()
            kernel32.CloseHandle(handle)
            raise ctypes.WinError(err)

        self._handle = handle

    @property
    def is_active(self) -> bool:
        """Return True when the underlying Win32 Job Object handle is open."""
        return self._handle is not None

    def assign_process(self, process_or_handle: subprocess.Popen[Any] | int) -> bool:
        """Assign a running process or OS handle to this Job Object."""
        if self._handle is None or self._kernel32 is None:
            return False

        handle_val: int | None = None
        if isinstance(process_or_handle, int) and not isinstance(process_or_handle, bool):
            handle_val = int(process_or_handle)
        else:
            raw = getattr(process_or_handle, "_handle", None)
            if isinstance(raw, int) and not isinstance(raw, bool):
                handle_val = int(raw)

        if handle_val is None or handle_val <= 0:
            return False

        ctypes.set_last_error(0)
        ok = self._kernel32.AssignProcessToJobObject(
            self._handle,
            ctypes.c_void_p(handle_val),
        )
        if not ok:
            err = ctypes.get_last_error()
            logger.warning("AssignProcessToJobObject failed (Win32 error %s)", err)
            return False
        return True

    def close(self) -> None:
        """Close the Job Object handle, terminating any remaining child processes."""
        if self._handle is not None and self._kernel32 is not None:
            try:
                self._kernel32.CloseHandle(self._handle)
            except Exception:
                pass
        self._handle = None

    def __enter__(self) -> WindowsJobObject:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


class SingleInstanceMutex:
    """Win32 named mutex guard preventing duplicate desktop launcher instances."""

    def __init__(self, name: str = DEFAULT_MUTEX_NAME) -> None:
        self.name = name
        self._handle: ctypes.c_void_p | None = None
        self._kernel32: ctypes.WinDLL | None = None
        self.already_exists: bool = False

    @property
    def is_primary(self) -> bool:
        """Return True when this instance currently owns the primary mutex lock."""
        return self._handle is not None and not self.already_exists

    def acquire(self) -> bool:
        """Attempt to acquire the named mutex; return False if already held elsewhere."""
        if self._handle is not None:
            return not self.already_exists

        if sys.platform != "win32":
            self.already_exists = False
            return True

        kernel32 = _get_kernel32()
        self._kernel32 = kernel32
        ctypes.set_last_error(0)
        raw_handle = kernel32.CreateMutexW(None, False, self.name)
        last_err = ctypes.get_last_error()
        if not raw_handle:
            raise ctypes.WinError(last_err)

        handle = ctypes.c_void_p(raw_handle)
        if last_err == ERROR_ALREADY_EXISTS:
            # Close duplicate handle immediately so secondary launchers do not extend mutex lifetime.
            kernel32.CloseHandle(handle)
            self.already_exists = True
            self._handle = None
            return False

        self.already_exists = False
        self._handle = handle
        return True

    def release(self) -> None:
        """Release the held Win32 mutex handle if owned by this instance."""
        if self._handle is not None and self._kernel32 is not None:
            try:
                self._kernel32.CloseHandle(self._handle)
            except Exception:
                pass
        self._handle = None

    def __enter__(self) -> SingleInstanceMutex:
        self.acquire()
        return self

    def __exit__(self, *args: Any) -> None:
        self.release()

    def __del__(self) -> None:
        try:
            self.release()
        except Exception:
            pass


def is_mutex_held(name: str = DEFAULT_MUTEX_NAME) -> bool:
    """Return True if the named Win32 mutex currently exists (used by `--quit` polling)."""
    if sys.platform != "win32":
        return False

    kernel32 = _get_kernel32()
    ctypes.set_last_error(0)
    raw_handle = kernel32.OpenMutexW(SYNCHRONIZE, False, name)
    if raw_handle:
        kernel32.CloseHandle(ctypes.c_void_p(raw_handle))
        return True
    return False


def _atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    """Atomically write JSON metadata into `path`."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(data, indent=2) + "\n"
    tmp_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=str(path.parent),
            prefix=f".{path.stem}_",
            suffix=".tmp",
            delete=False,
        ) as tmp_file:
            tmp_name = tmp_file.name
            tmp_file.write(payload)
            tmp_file.flush()
        os.replace(tmp_name, path)
    except Exception:
        if tmp_name:
            try:
                Path(tmp_name).unlink(missing_ok=True)
            except OSError:
                pass
        raise


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

        _atomic_write_json(
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
