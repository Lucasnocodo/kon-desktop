"""Saved aiming profiles for a local machine/camera/display configuration."""
from __future__ import annotations

from .paths import state_root as user_state_dir

import hashlib
import json
import math
import os
import platform
from pathlib import Path

from PySide6.QtCore import QSysInfo

from .geometry import ScreenCalibration
from .personal_calibration import PersonalCalibration


def machine_key() -> str:
    identity = bytes(QSysInfo.machineUniqueId()) or platform.node().encode()
    return hashlib.sha256(identity).hexdigest()[:24]


def compatible_configuration(saved, current) -> bool:
    """Screen origin/name are informational; the model predicts local XY."""
    if not isinstance(saved, dict) or not isinstance(current, dict):
        return False
    # Retired acquisition metadata must never be interpreted as a profile
    # captured by the supported single-camera geometry.
    if any(config.get("acquisition_mode", "single") != "single" for config in (saved, current)):
        return False
    try:
        if (saved["eye"] != current["eye"] or saved["camera"] != current["camera"]
                or list(saved["screen"]) != list(current["screen"])):
            return False
        left, right = float(saved["fov"]), float(current["fov"])
        if not all(math.isfinite(value) for value in (left, right)) or abs(left - right) > .005:
            return False
        for key in ("machine_id", "screen_serial"):
            if saved.get(key) and current.get(key) and saved[key] != current[key]:
                return False
        return True
    except (KeyError, TypeError, ValueError, OverflowError):
        return False


def _paths(root: Path, config):
    identity = {key: config[key] for key in ("eye", "camera", "screen")}
    identity["fov"] = round(float(config["fov"]), 2)
    identity["machine_id"] = config.get("machine_id") or machine_key()
    identity["screen_serial"] = config.get("screen_serial") or ""
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:24]
    folder = user_state_dir(root) / 'aim-calibrations'
    return folder / f"{key}.json", folder / f"{key}.meta.json"


def _write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def save_calibration(root: Path, model, config):
    if not isinstance(config, dict) or config.get("acquisition_mode", "single") != "single":
        raise ValueError("Only single-camera aiming profiles can be saved.")
    metadata = dict(config, machine_id=config.get("machine_id") or machine_key())
    profile, profile_meta = _paths(root, metadata)
    model.save(profile)
    _write_json(profile_meta, metadata)
    model.save(user_state_dir(root) / 'calibration.json')
    _write_json(user_state_dir(root) / 'calibration-meta.json', metadata)


def load_calibration(root: Path, config):
    if not isinstance(config, dict) or config.get("acquisition_mode", "single") != "single":
        return None
    profile, profile_meta = _paths(root, config)
    candidates = [(profile, profile_meta),
                  (user_state_dir(root) / 'calibration.json', user_state_dir(root) / 'calibration-meta.json')]
    # A driver may stop reporting a serial without changing the display.
    # Retain previous profiles, and select only a compatible environment.
    folder = user_state_dir(root) / 'aim-calibrations'
    try:
        metadata_files = sorted(folder.glob("*.meta.json"), key=lambda path: path.stat().st_mtime, reverse=True)
    except OSError:
        metadata_files = []
    candidates.extend((path.with_name(path.name.replace(".meta.json", ".json")), path)
                      for path in metadata_files)
    for path, metadata_path in candidates:
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if not compatible_configuration(metadata, config):
                continue
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("calibration_model") == PersonalCalibration.calibration_model:
                model = PersonalCalibration.load(path)
                if list(model.screen_size) != list(config["screen"]):
                    continue
            else:
                model = ScreenCalibration.load(path)
            return model
        except (OSError, ValueError, TypeError, KeyError):
            continue
    return None


def gesture_profile_path(root: Path) -> Path:
    return user_state_dir(root) / 'gesture-calibrations' / f"{machine_key()}.json"
