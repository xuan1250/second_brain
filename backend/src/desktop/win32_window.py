"""Win32 window icon, AppUserModelID, focus, Edge discovery, and asset resolution."""
from __future__ import annotations

import ctypes
import os
import shutil
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.desktop.config import DesktopPaths

DESKTOP_APP_USER_MODEL_ID = "SecondBrain.Desktop.App"
_SW_RESTORE = 9
_WM_SETICON = 0x0080
_ICON_SMALL = 0
_ICON_BIG = 1
_IMAGE_ICON = 1
_LR_LOADFROMFILE = 0x0010


def _set_app_user_model_id() -> None:
    """Assign an explicit Windows AppUserModelID for taskbar icon and grouping parity."""
    if sys.platform != "win32":
        return
    try:
        shell32 = ctypes.WinDLL("shell32", use_last_error=True)
        shell32.SetCurrentProcessExplicitAppUserModelID.argtypes = [ctypes.c_wchar_p]
        shell32.SetCurrentProcessExplicitAppUserModelID.restype = ctypes.c_long
        shell32.SetCurrentProcessExplicitAppUserModelID(DESKTOP_APP_USER_MODEL_ID)
    except Exception:
        pass


def _find_msedge_exe() -> Path | None:
    """Locate Microsoft Edge (`msedge.exe`) on PATH or standard Windows install paths."""
    for candidate_name in ("msedge.exe", "msedge"):
        found = shutil.which(candidate_name)
        if found:
            return Path(found)

    if sys.platform == "win32":
        roots = [
            os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"),
            os.environ.get("ProgramFiles", r"C:\Program Files"),
            os.environ.get("LOCALAPPDATA", ""),
        ]
        for root in roots:
            if not root:
                continue
            candidate = Path(root) / "Microsoft" / "Edge" / "Application" / "msedge.exe"
            if candidate.is_file():
                return candidate
    return None


def _focus_win32_window(title: str = "Second Brain") -> bool:
    """Best-effort Win32 FindWindowW + ShowWindow + SetForegroundWindow fallback."""
    if sys.platform != "win32":
        return False
    try:
        user32 = ctypes.WinDLL("user32", use_last_error=True)
        user32.FindWindowW.argtypes = [ctypes.c_wchar_p, ctypes.c_wchar_p]
        user32.FindWindowW.restype = ctypes.c_void_p
        user32.ShowWindow.argtypes = [ctypes.c_void_p, ctypes.c_int]
        user32.ShowWindow.restype = ctypes.c_int
        user32.SetForegroundWindow.argtypes = [ctypes.c_void_p]
        user32.SetForegroundWindow.restype = ctypes.c_int

        hwnd = user32.FindWindowW(None, title)
        if not hwnd:
            return False
        user32.ShowWindow(ctypes.c_void_p(hwnd), _SW_RESTORE)
        user32.SetForegroundWindow(ctypes.c_void_p(hwnd))
        return True
    except Exception:
        return False


def _apply_win32_window_icon(window_title: str, ico_path: Path) -> bool:
    """Apply small (16x16) and large (32x32) Win32 title-bar/taskbar icons from `ico_path`."""
    if sys.platform != "win32" or not ico_path.is_file():
        return False
    try:
        user32 = ctypes.WinDLL("user32", use_last_error=True)
        user32.FindWindowW.argtypes = [ctypes.c_wchar_p, ctypes.c_wchar_p]
        user32.FindWindowW.restype = ctypes.c_void_p
        user32.LoadImageW.argtypes = [
            ctypes.c_void_p,
            ctypes.c_wchar_p,
            ctypes.c_uint32,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_uint32,
        ]
        user32.LoadImageW.restype = ctypes.c_void_p
        user32.SendMessageW.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint32,
            ctypes.c_size_t,
            ctypes.c_void_p,
        ]
        user32.SendMessageW.restype = ctypes.c_ssize_t

        hwnd = user32.FindWindowW(None, window_title)
        if not hwnd:
            return False

        ico_str = str(ico_path)
        hicon_small = user32.LoadImageW(
            None, ico_str, _IMAGE_ICON, 16, 16, _LR_LOADFROMFILE
        )
        hicon_big = user32.LoadImageW(
            None, ico_str, _IMAGE_ICON, 32, 32, _LR_LOADFROMFILE
        )
        applied = False
        if hicon_small:
            user32.SendMessageW(
                ctypes.c_void_p(hwnd),
                _WM_SETICON,
                _ICON_SMALL,
                ctypes.c_void_p(hicon_small),
            )
            applied = True
        if hicon_big:
            user32.SendMessageW(
                ctypes.c_void_p(hwnd),
                _WM_SETICON,
                _ICON_BIG,
                ctypes.c_void_p(hicon_big),
            )
            applied = True
        return applied
    except Exception:
        return False


def resolve_app_icon_path(paths: DesktopPaths | None = None) -> Path | None:
    """Resolve `second-brain.ico` across packaged installer and dev repo layouts."""
    candidates: list[Path] = []
    if paths is not None:
        candidates.extend(
            [
                paths.app_dir / "app" / "installer" / "second-brain.ico",
                paths.app_dir / "installer" / "second-brain.ico",
            ]
        )

    module_parents = Path(__file__).resolve().parents
    if len(module_parents) > 3:
        candidates.append(module_parents[3] / "installer" / "second-brain.ico")

    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None
