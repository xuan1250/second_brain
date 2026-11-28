"""Win32 Job Object isolation and single-instance mutex."""
from __future__ import annotations

import ctypes
import logging
import subprocess
import sys
from typing import Any

from src.desktop.control_channel import ControlChannelServer, send_control_command

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


__all__ = [
    "ControlChannelServer",
    "DEFAULT_MUTEX_NAME",
    "ERROR_ALREADY_EXISTS",
    "IO_COUNTERS",
    "JOBOBJECT_BASIC_LIMIT_INFORMATION",
    "JOBOBJECT_EXTENDED_LIMIT_INFORMATION",
    "JOB_OBJECT_EXTENDED_LIMIT_INFORMATION_CLASS",
    "JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE",
    "SYNCHRONIZE",
    "SingleInstanceMutex",
    "WindowsJobObject",
    "is_mutex_held",
    "send_control_command",
]
