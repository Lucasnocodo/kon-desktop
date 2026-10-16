# Freeze only the explicitly staged runtime resources and core application.
from pathlib import Path
import tomllib
from PyInstaller.utils.hooks import collect_dynamic_libs, collect_data_files

root = Path(SPECPATH).parent
stage = root / "build/macos-arm64/stage"
version = tomllib.loads((root / "pyproject.toml").read_text())["project"]["version"]
a = Analysis(
    [str(root / "kon/entrypoint.py")], pathex=[str(root)],
    binaries=[(str(path), "helpers") for path in (stage / "helpers").iterdir() if path.is_file()]
             + collect_dynamic_libs("mediapipe") + collect_dynamic_libs("_sounddevice_data"),
    datas=[(str(stage / "resources"), ".")] + collect_data_files("mediapipe"),
    hiddenimports=["Cocoa", "Foundation", "AppKit", "objc", "_sounddevice", "sounddevice"],
    hookspath=[], runtime_hooks=[],
    excludes=["tkinter", "PyQt5", "PyQt6", "PySide2", "torch", "tensorflow", "jax", "IPython", "pytest"],
    noarchive=False,
)
# Reject accidental imports beyond the public app boundary even when an old
# local environment happens to contain those modules.
forbidden = ("kon.stereo", "kon.portal", "kon.pet", "kon.cat_pet", "kon.native_pet", "kon.photo", "kon.phone", "kon.calibration_voice")
assert not [name for name, *_ in a.pure if name.startswith(forbidden)], "Excluded application modules entered the bundle"
a.binaries = [entry for entry in a.binaries if not entry[0].lower().endswith(".dll")]
a.datas = [entry for entry in a.datas if not entry[0].lower().endswith(".dll")]
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name="KON", debug=False,
          bootloader_ignore_signals=False, strip=False, upx=False, console=False,
          target_arch="arm64", codesign_identity=None, entitlements_file=None)
collection = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False, name="KON")
app = BUNDLE(collection, name="KON.app", icon=str(root / "packaging/KON.icns"),
             bundle_identifier="io.github.lucasnocodo.kon-desktop", version=version,
             info_plist={
                 "CFBundleDisplayName": "KON", "CFBundleDevelopmentRegion": "en",
                 "CFBundleLocalizations": ["en", "zh-Hant"], "LSMinimumSystemVersion": "14.0",
                 "NSHighResolutionCapable": True,
                 "NSCameraUsageDescription": "KON uses the camera only after you start sensors, to recognize your hand and aim. Images are processed locally.",
                 "NSMicrophoneUsageDescription": "KON uses the microphone only after you start sensors or a voice test, to detect a short sound or recognize a local phrase.",
                 "NSAppleEventsUsageDescription": "Desktop mode reads Finder icon positions and moves only the file you select to Trash after the fox swallows it.",
             })
