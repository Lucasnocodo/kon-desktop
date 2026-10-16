"""Read-only application resources and per-user data for source and frozen runs."""
from __future__ import annotations

import os
from pathlib import Path
import sys


def resource_root() -> Path:
    if getattr(sys, "frozen", False):
        base = Path(sys._MEIPASS).resolve()
        resources = base.parent / "Resources"
        return resources.resolve() if base.name == "Frameworks" and resources.is_dir() else base
    return Path(__file__).resolve().parents[1]


def is_bundled_root(project_root=None) -> bool:
    return bool(getattr(sys, "frozen", False) and
                (project_root is None or Path(project_root).resolve() == resource_root()))


def state_root(project_root=None) -> Path:
    if is_bundled_root(project_root):
        override = os.environ.get("KON_DATA_DIR")
        return Path(override).expanduser().resolve() if override else (
            Path.home() / "Library" / "Application Support" / "KON Desktop").resolve()
    return Path(project_root or resource_root()).resolve() / ".kon"


def helper_path(name: str) -> Path:
    return resource_root() / "helpers" / name
