"""Windows System Tray controller and programmatic icon rendering for Second Brain."""
from __future__ import annotations

import logging
import os
import subprocess
import sys
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

from src.desktop.config import DesktopConfig, DesktopPaths, load_desktop_config

logger = logging.getLogger(__name__)


def build_tray_icon_image(size: int = 64) -> Any | None:
    """Load the brand `.ico` emblem if present, or render the Pillow fallback icon."""
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        logger.debug("Pillow is not installed; skipping tray icon image generation.")
        return None

    target_size = max(16, int(size))
    resampling = getattr(getattr(Image, "Resampling", Image), "LANCZOS", Image.BICUBIC)
    ico_path = Path(__file__).resolve().parents[3] / "installer" / "second-brain.ico"
    if ico_path.is_file():
        try:
            with Image.open(ico_path) as ico_img:
                rgba = ico_img.convert("RGBA")
                if rgba.size == (target_size, target_size):
                    return rgba
                return rgba.resize((target_size, target_size), resample=resampling)
        except Exception as exc:
            logger.debug("Failed to load tray .ico '%s', using fallback: %s", ico_path, exc)

    scale = 2
    canvas_size = target_size * scale

    image = Image.new("RGBA", (canvas_size, canvas_size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)

    pad = 4 * scale
    radius = 14 * scale
    draw.rounded_rectangle(
        [(pad, pad), (canvas_size - pad, canvas_size - pad)],
        radius=radius,
        fill=(15, 23, 42, 255),       # #0f172a dark slate
        outline=(56, 189, 248, 220),  # #38bdf8 sky border
        width=2 * scale,
    )

    # Geometric neural triad representing linked knowledge chunks
    nodes = [
        (int(canvas_size * 0.30), int(canvas_size * 0.35), (56, 189, 248, 255)),   # #38bdf8
        (int(canvas_size * 0.70), int(canvas_size * 0.35), (129, 140, 248, 255)),  # #818cf8
        (int(canvas_size * 0.50), int(canvas_size * 0.69), (16, 185, 129, 255)),   # #10b981
    ]
    edge_width = max(2, 3 * scale)
    draw.line([(nodes[0][0], nodes[0][1]), (nodes[1][0], nodes[1][1])], fill=(56, 189, 248, 190), width=edge_width)
    draw.line([(nodes[0][0], nodes[0][1]), (nodes[2][0], nodes[2][1])], fill=(56, 189, 248, 190), width=edge_width)
    draw.line([(nodes[1][0], nodes[1][1]), (nodes[2][0], nodes[2][1])], fill=(129, 140, 248, 190), width=edge_width)

    center_x = int(canvas_size * 0.50)
    center_y = int(canvas_size * 0.47)
    center_r = max(3, int(4.5 * scale))
    draw.ellipse(
        [(center_x - center_r, center_y - center_r), (center_x + center_r, center_y + center_r)],
        fill=(248, 250, 252, 255),
    )

    node_r = max(4, int(5.5 * scale))
    for nx, ny, color in nodes:
        draw.ellipse(
            [(nx - node_r, ny - node_r), (nx + node_r, ny + node_r)],
            fill=color,
            outline=(248, 250, 252, 240),
            width=max(1, scale),
        )

    return image.resize((target_size, target_size), resample=resampling)


class SystemTrayController:
    """Manage the `pystray` system tray icon, menu actions, and Windows notifications."""

    def __init__(
        self,
        paths: DesktopPaths | None = None,
        config: DesktopConfig | None = None,
        *,
        get_config: Callable[[], DesktopConfig] | None = None,
        on_open: Callable[[], Any] | None = None,
        on_open_hardware_mode: Callable[[], Any] | None = None,
        on_reclaim_memory: Callable[[], Any] | None = None,
        on_open_data_folder: Callable[[], Any] | None = None,
        on_open_logs_folder: Callable[[], Any] | None = None,
        on_quit: Callable[[], Any] | None = None,
        pystray_module: Any | None = None,
        icon_image: Any | None = None,
    ) -> None:
        self.paths: DesktopPaths = paths or DesktopPaths()
        self._config: DesktopConfig | None = config
        self._get_config = get_config
        self.on_open = on_open
        self.on_open_hardware_mode = on_open_hardware_mode
        self.on_reclaim_memory = on_reclaim_memory
        self.on_open_data_folder = on_open_data_folder
        self.on_open_logs_folder = on_open_logs_folder
        self.on_quit = on_quit
        self._pystray_module = pystray_module
        self._icon_image = icon_image
        self._icon: Any | None = None
        self._running: bool = False

    @property
    def is_running(self) -> bool:
        """Return True while the detached tray icon loop is active."""
        return self._running and self._icon is not None

    @property
    def icon(self) -> Any | None:
        """Return the underlying `pystray.Icon` instance when active."""
        return self._icon

    def _resolve_config(self) -> DesktopConfig:
        if self._get_config is not None:
            return self._get_config()
        if self._config is not None:
            return self._config
        return load_desktop_config(self.paths)

    @staticmethod
    def _invoke_callback(callback: Callable[[], Any] | None, label: str) -> None:
        if callback is None:
            return
        try:
            callback()
        except Exception as exc:
            logger.warning("Tray action '%s' failed: %s", label, exc)

    @staticmethod
    def _open_folder(folder: Path) -> None:
        """Ensure `folder` exists and open it in the native file manager."""
        target = folder.expanduser().resolve()
        target.mkdir(parents=True, exist_ok=True)
        if sys.platform == "win32":
            os.startfile(str(target))  # type: ignore[attr-defined]
            return
        opener = "open" if sys.platform == "darwin" else "xdg-open"
        subprocess.Popen([opener, str(target)])

    def open_data_folder(self) -> None:
        """Open the active Second Brain data directory."""
        if self.on_open_data_folder is not None:
            self._invoke_callback(self.on_open_data_folder, "open_data_folder")
            return
        cfg = self._resolve_config()
        self._open_folder(self.paths.resolve_data_dir(cfg))

    def open_logs_folder(self) -> None:
        """Open the Second Brain desktop/sidecar logs directory."""
        if self.on_open_logs_folder is not None:
            self._invoke_callback(self.on_open_logs_folder, "open_logs_folder")
            return
        self._open_folder(self.paths.logs_dir)

    def _handle_open(self, _icon: Any = None, _item: Any = None) -> None:
        self._invoke_callback(self.on_open, "open")

    def _handle_open_hardware_mode(self, _icon: Any = None, _item: Any = None) -> None:
        self._invoke_callback(self.on_open_hardware_mode, "open_hardware_mode")

    def _handle_reclaim_memory(self, _icon: Any = None, _item: Any = None) -> None:
        self._invoke_callback(self.on_reclaim_memory, "reclaim_memory")

    def _handle_open_data_folder(self, _icon: Any = None, _item: Any = None) -> None:
        try:
            self.open_data_folder()
        except Exception as exc:
            logger.warning("Failed to open data folder: %s", exc)

    def _handle_open_logs_folder(self, _icon: Any = None, _item: Any = None) -> None:
        try:
            self.open_logs_folder()
        except Exception as exc:
            logger.warning("Failed to open logs folder: %s", exc)

    def _handle_quit(self, _icon: Any = None, _item: Any = None) -> None:
        self._invoke_callback(self.on_quit, "quit")

    def _resolve_pystray(self) -> Any | None:
        if self._pystray_module is not None:
            return self._pystray_module
        try:
            import pystray

            return pystray
        except Exception as exc:
            logger.warning("pystray is unavailable (%s); continuing without system tray.", exc)
            return None

    def _build_menu(self, pystray_mod: Any) -> Any:
        menu_cls = pystray_mod.Menu
        item_cls = pystray_mod.MenuItem
        return menu_cls(
            item_cls("Open Second Brain", self._handle_open, default=True),
            item_cls("Hardware & Compute Mode (CPU / CUDA)...", self._handle_open_hardware_mode),
            menu_cls.SEPARATOR,
            item_cls("Reclaim Memory / VRAM Now", self._handle_reclaim_memory),
            item_cls("Open Data Folder", self._handle_open_data_folder),
            item_cls("Open Logs Folder", self._handle_open_logs_folder),
            menu_cls.SEPARATOR,
            item_cls("Quit Second Brain", self._handle_quit),
        )

    def start(self) -> bool:
        """Create and start the detached `pystray.Icon` loop."""
        if self._running and self._icon is not None:
            return True

        pystray_mod = self._resolve_pystray()
        if pystray_mod is None:
            return False

        image = self._icon_image if self._icon_image is not None else build_tray_icon_image(64)
        if image is None:
            logger.warning("Tray icon image could not be created; skipping system tray.")
            return False

        try:
            menu = self._build_menu(pystray_mod)
            icon = pystray_mod.Icon(
                name="second_brain",
                icon=image,
                title="Second Brain",
                menu=menu,
            )
            icon.run_detached()
        except Exception as exc:
            logger.warning("Failed to start system tray icon (%s); continuing without tray.", exc)
            self._icon = None
            self._running = False
            return False

        self._icon = icon
        self._running = True
        return True

    def run_detached(self) -> bool:
        """Alias for `start()` matching `pystray.Icon.run_detached()` naming."""
        return self.start()

    def stop(self) -> None:
        """Stop the detached `pystray.Icon` if running."""
        icon = self._icon
        self._icon = None
        self._running = False
        if icon is None:
            return
        try:
            icon.stop()
            tray_thread = getattr(icon, "_thread", None)
            if (
                isinstance(tray_thread, threading.Thread)
                and tray_thread.is_alive()
                and tray_thread is not threading.current_thread()
            ):
                tray_thread.join(timeout=2.0)
        except Exception as exc:
            logger.debug("Error stopping system tray icon: %s", exc)

    def notify(self, message: str, title: str = "Second Brain") -> bool:
        """Display a Windows tray toast/balloon notification if supported and active."""
        icon = self._icon
        if icon is None:
            return False
        if not getattr(icon, "HAS_NOTIFICATION", True):
            return False
        try:
            icon.notify(message, title)
            return True
        except Exception as exc:
            logger.debug("Tray notification failed: %s", exc)
            return False
