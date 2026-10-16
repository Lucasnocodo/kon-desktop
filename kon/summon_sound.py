"""Preloaded, asynchronous native playback for the selected fox sound effect."""
from __future__ import annotations

from .paths import resource_root

import logging
from pathlib import Path
import sys

from PySide6.QtGui import QGuiApplication


DEFAULT_SOUND_PATH = resource_root() / "assets/sfx/yaohu-short.wav"
_LOG = logging.getLogger(__name__)


def _load_sound(path: Path):
    # Headless rendering does not open an audio device.
    if sys.platform != "darwin" or QGuiApplication.platformName() == "offscreen":
        return None
    try:
        from AppKit import NSSound
        sound = NSSound.alloc().initWithContentsOfFile_byReference_(str(path), False)
        if sound is None:
            _LOG.warning("Fox sound is unavailable: %s", path)
        return sound
    except Exception:
        # Audio availability must not interrupt the animation or file action.
        _LOG.warning("Could not load the fox sound", exc_info=True)
        return None


class SummonSound:
    """Keep the short WAV in memory so playback starts without a new process."""

    def __init__(self, path: str | Path = DEFAULT_SOUND_PATH):
        self.path = Path(path)
        self._sound = _load_sound(self.path)

    def play(self) -> None:
        if self._sound is not None:
            try:
                self._sound.stop()
                self._sound.play()
            except Exception:
                _LOG.warning("Could not play the fox sound", exc_info=True)

    def stop(self) -> None:
        if self._sound is not None:
            try:
                self._sound.stop()
            except Exception:
                _LOG.warning("Could not stop the fox sound", exc_info=True)
