"""Finite checks run by the distributed executable, without live devices/files."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import traceback
from unittest.mock import patch


def speech_check(root: Path) -> dict:
    """Use a public prerecorded fixture with the bundled CPU-only recognizer."""
    from .paths import helper_path
    from .speech import matches_keyword, whisper_command
    helper = helper_path("whisper-cli")
    model = root / "models/ggml-tiny.bin"
    fixture = root / "test-fixtures/jfk.wav"
    for path in (helper, model, fixture):
        if not path.is_file():
            raise FileNotFoundError(f"Missing bundled diagnostic input: {path.name}")
    transcript = ""
    for language in ("en", "zh"):
        command = whisper_command(helper, model, fixture, language=language)
        if language == "zh":
            command.extend(["--duration", "1000", "--best-of", "1", "--beam-size", "1"])
        completed = subprocess.run(command, capture_output=True, text=True, timeout=120)
        if completed.returncode:
            raise RuntimeError(f"Bundled CPU recognizer failed: {completed.stderr[-1000:]}")
        text = completed.stdout.strip()
        if "read_audio_data:" in text or "whisper_init" in text or (language == "en" and not text):
            raise RuntimeError("Recognizer did not return clean transcription on stdout")
        if language == "en":
            if "fellow americans" not in text.casefold() or matches_keyword(text):
                raise RuntimeError(f"Unexpected public fixture transcription: {text}")
            transcript = text
    return {"ok": True, "inference": "bundled local CPU", "microphone_opened": False,
            "english_transcript": transcript, "mandarin_initialization": True,
            "mandarin_accuracy_tested": False}


def runtime_check(root: Path, language: str, folder: Path) -> dict:
    import cv2
    import numpy as np
    import sounddevice
    from PySide6.QtGui import QImage
    from PySide6.QtWidgets import QApplication
    from . import desktop
    from .paths import state_root, helper_path
    from .i18n import get_language, set_language, translate
    checks = {}
    data = state_root()
    if data.is_relative_to(root):
        raise AssertionError("User state must be outside the application bundle")
    data.mkdir(parents=True, exist_ok=True)
    (data / "write-probe.txt").write_text("KON package verification", encoding="utf-8")
    checks["external_writable_user_data"] = True
    set_language(language)
    if get_language() != language:
        raise AssertionError("Requested application language was not selected")
    if translate("啟動鏡頭與麥克風") == "啟動鏡頭與麥克風" and language == "en":
        raise AssertionError("Bundled English catalog is missing")
    checks["locale"] = language
    for relative in ("assets/fox-heads.png", "assets/fox-dash/fox-dash-sheet.png",
                     "assets/fox-smoke/fox-smoke-sheet.png"):
        image = QImage(str(root / relative))
        if image.isNull() or not image.hasAlphaChannel():
            raise AssertionError(f"Missing transparent runtime artwork: {relative}")
    if not (root / "assets/sfx/yaohu-short.wav").is_file():
        raise AssertionError("Missing non-speech summon effect")
    checks["fox_artwork_and_sound"] = True
    def forbidden(*_args, **_kwargs):
        raise AssertionError("Package verification must not open devices or operate Finder")
    app = QApplication.instance() or QApplication([])
    with patch.object(cv2, "VideoCapture", forbidden), patch.object(sounddevice, "InputStream", forbidden), \
         patch.object(desktop, "scan_desktop", forbidden), patch.object(desktop, "trash_target", forbidden), \
         patch.object(desktop, "restore_receipt", forbidden):
        from .app import Window
        for mode in ("eye_ray", "hand"):
            window = Window(aim_mode=mode)
            try:
                window.show()
                app.processEvents()
                if window.camera is not None or window.speech is not None or window.running:
                    raise AssertionError("Opening the control room unexpectedly started devices")
                if window.mode.count() != 2 or window.aim_method.count() != 2:
                    raise AssertionError("Unexpected application modes in this release")
                if window.aim_mode != mode:
                    raise AssertionError("Requested single-camera aim mode was not selected")
                if language == "en" and any('\u3400' <= c <= '\u9fff' for c in window.start_button.text()):
                    raise AssertionError("English control-room action was not translated")
                if window.controls_scroll.widget().width() > window.controls_scroll.viewport().width():
                    raise AssertionError("Control-room actions are clipped horizontally")
                window.grab().save(str(folder / f"control-{language}-{mode}.png"))
                # Exercises real effect decoding and a virtual target only.
                window.preview_animation()
                window.overlay.timer.stop()
                target = window.overlay.eating
                if target is None or target.snapshot is not None or not window.overlay.practice:
                    raise AssertionError("Preview must use a virtual file")
                checks[f"control_room_{mode}"] = True
            finally:
                window.close()
                app.processEvents()
    checks["no_devices_or_finder_actions"] = True
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    blank = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.zeros((160, 160, 3), dtype=np.uint8))
    for name, options, task in (
        ("hand_landmarker.task", vision.HandLandmarkerOptions, vision.HandLandmarker),
        ("face_landmarker.task", vision.FaceLandmarkerOptions, vision.FaceLandmarker),
    ):
        with task.create_from_options(options(base_options=python.BaseOptions(
                model_asset_path=str(root / "models" / name), delegate=python.BaseOptions.Delegate.CPU))) as model:
            model.detect(blank)
        checks[name] = True
    from .speech import SpeechWorker
    worker = SpeechWorker()
    if worker.cli_path != helper_path("whisper-cli") or not worker.model_path.is_file():
        raise AssertionError("SpeechWorker must resolve bundled runtime files")
    completed = subprocess.run([str(worker.cli_path), "--help"], capture_output=True, timeout=30)
    if completed.returncode:
        raise RuntimeError(completed.stderr.decode(errors="replace")[-1000:])
    checks["bundled_cpu_helper"] = True
    return checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--package-check", action="store_true")
    group.add_argument("--speech-self-test", action="store_true")
    parser.add_argument("--language", choices=("en", "zh-TW"), default="en")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = {"ok": False, "camera_opened": False, "microphone_opened": False,
              "real_desktop_files_modified": False}
    with tempfile.TemporaryDirectory(prefix="kon-frozen-check-") as temporary:
        folder = Path(temporary)
        os.environ.update(KON_DATA_DIR=str(folder / "user-data"), MPLCONFIGDIR=str(folder / "matplotlib"),
                          QT_QPA_PLATFORM="offscreen")
        try:
            if not getattr(sys, "frozen", False):
                raise AssertionError("Run this diagnostic using the bundled KON executable")
            from .paths import resource_root
            root = resource_root()
            report["version"] = json.loads((root / "build-info.json").read_text())["version"]
            if args.speech_self_test:
                report.update(speech_check(root))
            else:
                report["checks"] = runtime_check(root, args.language, folder)
                report["ok"] = True
        except Exception:
            report["error"] = traceback.format_exc()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
