"""Sidecar process spawning, pipe-reader streaming, HTTP readiness polling, and termination helpers."""
from __future__ import annotations

import logging
import subprocess
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from src.desktop.env import CREATE_NO_WINDOW
from src.desktop.job_object import WindowsJobObject


def _sup_mod():
    import src.desktop.supervisor as sup_mod

    return sup_mod


def spawn_pipe_reader(
    proc: subprocess.Popen[str],
    channel: str,
    target_logger: logging.Logger | None,
) -> threading.Thread | None:
    """Stream child stdout/stderr lines into a rotating file logger in a daemon thread."""
    pipe = getattr(proc, "stdout", None)
    if pipe is None:
        return None

    def _reader() -> None:
        try:
            for raw_line in pipe:
                line = str(raw_line).rstrip("\r\n")
                if line and target_logger is not None:
                    target_logger.info("%s", line)
        except Exception:
            pass
        finally:
            try:
                pipe.close()
            except Exception:
                pass

    thread = threading.Thread(
        target=_reader,
        name=f"desktop-log-{channel}",
        daemon=True,
    )
    thread.start()
    return thread


def wait_for_http_ready(
    url: str,
    proc: subprocess.Popen[Any],
    service_name: str,
    *,
    startup_timeout_sec: float,
    poll_interval_sec: float,
) -> None:
    """Poll `url` until HTTP 200 or fail early if child exits or timeout expires."""
    sup_mod = _sup_mod()
    deadline = time.monotonic() + startup_timeout_sec
    last_error: str = "no response yet"

    while time.monotonic() < deadline:
        exit_code = proc.poll()
        if exit_code is not None:
            raise RuntimeError(
                f"{service_name} exited prematurely with exit code {exit_code}"
            )

        try:
            with sup_mod.urllib.request.urlopen(url, timeout=1.5) as resp:
                status = getattr(resp, "status", None)
                if status is None:
                    status = resp.getcode()
                if status == 200:
                    return
                last_error = f"HTTP {status}"
        except (sup_mod.urllib.error.URLError, OSError, TimeoutError, ValueError) as exc:
            last_error = str(exc)

        time.sleep(poll_interval_sec)

    raise TimeoutError(
        f"{service_name} did not become ready at {url} within {startup_timeout_sec:.1f}s "
        f"(last error: {last_error})"
    )


def terminate_process(
    proc: subprocess.Popen[Any] | None,
    service_name: str,
    timeout_sec: float,
    *,
    reader_thread: threading.Thread | None = None,
    log_fn: Callable[..., None] | None = None,
) -> None:
    """Gracefully terminate a child process tree with timeout and hard-kill fallback."""
    if proc is None:
        return

    sup_mod = _sup_mod()

    if proc.poll() is None:
        if log_fn is not None:
            log_fn(logging.INFO, "Stopping %s (pid=%s)...", service_name, getattr(proc, "pid", "?"))
        if (
            sup_mod.sys.platform == "win32"
            and isinstance(proc, sup_mod._REAL_POPEN_TYPE)
            and isinstance(proc.pid, int)
            and not isinstance(proc.pid, bool)
            and proc.pid > 0
        ):
            sup_mod._kill_process_tree_win32(proc.pid)

        if proc.poll() is None:
            try:
                proc.terminate()
                proc.wait(timeout=timeout_sec)
            except sup_mod.subprocess.TimeoutExpired:
                if log_fn is not None:
                    log_fn(
                        logging.WARNING,
                        "%s did not exit within %.1fs; forcing kill.",
                        service_name,
                        timeout_sec,
                    )
                try:
                    proc.kill()
                    proc.wait(timeout=2.0)
                except (sup_mod.subprocess.TimeoutExpired, OSError):
                    pass
            except OSError:
                pass

    if reader_thread is not None and reader_thread.is_alive():
        reader_thread.join(timeout=1.5)

    # Only close stdout when the reader thread has exited; closing while blocked
    # in `for raw_line in pipe:` deadlocks on CPython's internal pipe lock.
    if reader_thread is None or not reader_thread.is_alive():
        pipe = getattr(proc, "stdout", None)
        if pipe is not None:
            try:
                pipe.close()
            except Exception:
                pass


def spawn_sidecar(
    *,
    cmd: list[str],
    cwd: Path,
    env: dict[str, str],
    channel: str,
    service_name: str,
    health_url: str,
    job_object: WindowsJobObject,
    target_logger: logging.Logger | None,
    startup_timeout_sec: float,
    poll_interval_sec: float,
    terminate_fn: Callable[..., None] | None = None,
) -> tuple[subprocess.Popen[str], threading.Thread | None]:
    """Spawn a sidecar process, attach Job Object and pipe logger, and wait for HTTP readiness."""
    sup_mod = _sup_mod()
    proc = sup_mod.subprocess.Popen(
        cmd,
        cwd=str(cwd),
        env=env,
        stdout=sup_mod.subprocess.PIPE,
        stderr=sup_mod.subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        creationflags=CREATE_NO_WINDOW,
    )
    job_object.assign_process(proc)
    reader = spawn_pipe_reader(proc, channel, target_logger)
    try:
        wait_for_http_ready(
            health_url,
            proc,
            service_name,
            startup_timeout_sec=startup_timeout_sec,
            poll_interval_sec=poll_interval_sec,
        )
    except Exception:
        if terminate_fn is not None:
            terminate_fn(proc, service_name, timeout_sec=5.0, reader_thread=reader)
        else:
            terminate_process(proc, service_name, 5.0, reader_thread=reader)
        raise
    return proc, reader
