#!/usr/bin/env python3
"""Build the allowlisted, offline arm64 macOS application with ad-hoc signing."""
from __future__ import annotations

import hashlib
from importlib.metadata import distributions, version
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import sysconfig
import tomllib

ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "build/macos-arm64"
STAGE = BUILD / "stage"
RESOURCES = STAGE / "resources"
MINIMUM_MACOS = "14.0"
WHISPER_COMMIT = "0c7be696371069e0baa2515ed7aa08d5a5473aa6"
RUNTIME_ASSETS = (
    "assets/fox-heads.png", "assets/fox-dash/fox-dash-sheet.png",
    "assets/fox-smoke/fox-smoke-sheet.png", "assets/sfx/yaohu-short.wav",
)
LOCALES = ("kon/locales/en.json", "kon/locales/diagnostic-fields.json")
MODEL_FILES = frozenset({"hand_landmarker.task", "face_landmarker.task", "ggml-tiny.bin"})


def run(*args, **kwargs):
    subprocess.run([str(arg) for arg in args], cwd=ROOT, check=True, **kwargs)


def digest(path):
    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def copy_file(source, destination):
    source, destination = Path(source), Path(destination)
    if not source.is_file():
        raise FileNotFoundError(f"Required build input is missing: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def is_license_text(source: Path, relative: Path) -> bool:
    """Recognize actual UTF-8 notices without collecting similarly named code."""
    if relative.is_absolute() or ".." in relative.parts:
        return False
    parts = [part.lower() for part in relative.parts]
    if any(part.endswith(".dsym") or part == "__pycache__" for part in parts):
        return False
    if source.suffix.lower() in {
        ".so", ".dylib", ".dll", ".pyd", ".exe", ".a", ".o", ".pyc",
        ".py", ".c", ".h", ".cpp", ".js", ".css", ".xml", ".png", ".jpg",
        ".jpeg", ".wav", ".zip", ".whl", ".pdf", ".mo",
    }:
        return False
    named_notice = re.match(r"^(license|licence|copying|notice|copyright)(?:$|[._-])", parts[-1])
    license_folder = any(part in {"license", "licenses", "licence", "licences"}
                         for part in parts[:-1])
    if not (named_notice or license_folder) or not source.is_file():
        return False
    try:
        content = source.read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        return False
    return bool(content.strip()) and not any(ord(char) < 32 and char not in "\n\r\t\f\b"
                                             for char in content)


def make_icon():
    """Lay out the existing closed-mouth fox in Apple's standard icon sizes."""
    icon = ROOT / "packaging/KON.icns"
    if icon.is_file():
        return icon
    from PySide6.QtCore import QRectF, Qt
    from PySide6.QtGui import QImage, QPainter
    source = QImage(str(ROOT / "assets/fox-heads.png"))
    if source.isNull():
        raise RuntimeError("Cannot load the existing fox icon artwork")
    source = source.copy(0, 0, source.width() // 2, source.height())
    iconset = BUILD / "KON.iconset"
    iconset.mkdir(parents=True, exist_ok=True)
    for size in (16, 32, 128, 256, 512):
        for factor in (1, 2):
            edge = size * factor
            image = QImage(edge, edge, QImage.Format_ARGB32)
            image.fill(Qt.transparent)
            painter = QPainter(image)
            painter.setRenderHint(QPainter.SmoothPixmapTransform)
            margin = edge * .055
            painter.drawImage(QRectF(margin, margin, edge - 2 * margin, edge - 2 * margin), source)
            painter.end()
            suffix = "@2x" if factor == 2 else ""
            if not image.save(str(iconset / f"icon_{size}x{size}{suffix}.png")):
                raise RuntimeError("Cannot encode application icon")
    run("iconutil", "--convert", "icns", "--output", icon, iconset)
    return icon


def stage_notices():
    notices = RESOURCES / "ThirdPartyNotices"
    notices.mkdir(parents=True)
    for relative in ("LICENSE", "NOTICE", "requirements.lock.txt", "UPSTREAMS.json", "models.expected.json"):
        copy_file(ROOT / relative, notices / relative)
    copy_file(ROOT / "assets/NOTICE.md", notices / "Media-NOTICE.md")
    copy_file(ROOT / "assets/provenance.json", notices / "Media-provenance.json")
    for folder in ("THIRD_PARTY_LICENSES", "packaging/licenses"):
        for path in sorted((ROOT / folder).rglob("*")):
            if path.is_file():
                copy_file(path, notices / Path(folder).name / path.relative_to(ROOT / folder))
    environment = {}
    for distribution in distributions():
        name = distribution.metadata.get("Name", "unknown")
        environment[name] = distribution.version
        for file in distribution.files or ():
            path = Path(str(file))
            source = Path(distribution.locate_file(file))
            if is_license_text(source, path):
                copy_file(source, notices / name / path)
    copy_file(Path(sysconfig.get_path("stdlib")) / "LICENSE.txt", notices / "Python-LICENSE.txt")
    copy_file(ROOT / "vendor/whisper.cpp/LICENSE", notices / "whisper.cpp-MIT.txt")
    (notices / "installed-components.json").write_text(json.dumps(environment, indent=2) + "\n")
    qt_version = version("PySide6_Essentials")
    qt_series = ".".join(qt_version.split(".")[:2])
    shiboken_version = version("shiboken6")
    (notices / "Runtime-NOTICE.txt").write_text(
        "KON Desktop includes Python, Qt/PySide6/Shiboken6, MediaPipe, OpenCV, NumPy, PyObjC, "
        "sounddevice/PortAudio, whisper.cpp and their runtime dependencies.\n"
        "Each component retains its own license; KON's MIT code license does not relicense these components "
        "or the image/audio assets. License texts and installed versions accompany this file.\n"
        "Qt/PySide6/Shiboken6 are dynamically linked, unmodified upstream wheels. You may replace compatible "
        "shared libraries and reverse-engineer the application to debug modifications to these libraries; "
        "re-sign the modified local bundle with an ad-hoc signature when needed.\n"
        "Corresponding source and build instructions: https://github.com/Lucasnocodo/kon-desktop\n"
        f"Installed Qt/PySide6 version: {qt_version}; Shiboken6 version: {shiboken_version}.\n"
        f"Qt sources: https://download.qt.io/archive/qt/{qt_series}/{qt_version}/submodules/\n"
        f"Qt for Python sources: https://code.qt.io/cgit/pyside/pyside-setup.git/tag/?h=v{qt_version}\n"
        "Qt licensing and component notices: https://doc.qt.io/qt-6/licenses-used-in-qt.html\n"
        "Qt for Python notices: https://doc.qt.io/qtforpython-6/licenses.html\n"
        "MediaPipe model sources: models.expected.json; downloaded model terms remain upstream.\n"
        "test-fixtures/jfk.wav is the unmodified diagnostic sample from the pinned whisper.cpp checkout; "
        "the upstream MIT notice is included. It is only used by explicit offline diagnostics.\n",
        encoding="utf-8")
    return environment


def stage_resources():
    if STAGE.exists():
        shutil.rmtree(STAGE)
    RESOURCES.mkdir(parents=True)
    expected = json.loads((ROOT / "models.expected.json").read_text())
    if set(expected) != MODEL_FILES:
        raise ValueError("The standalone release accepts only the three core runtime models")
    for relative in RUNTIME_ASSETS + LOCALES:
        copy_file(ROOT / relative, RESOURCES / relative)
    for name, record in expected.items():
        if Path(name).name != name:
            raise ValueError("Model manifest contains a non-local filename")
        path = ROOT / "models" / name
        if not path.is_file() or path.stat().st_size != record["bytes"] or digest(path) != record["sha256"]:
            raise RuntimeError(f"Unverified or missing model: {name}; run scripts/setup.sh")
        copy_file(path, RESOURCES / "models" / name)
    copy_file(ROOT / "vendor/whisper.cpp/samples/jfk.wav", RESOURCES / "test-fixtures/jfk.wav")
    helpers = STAGE / "helpers"
    helpers.mkdir()
    for path in sorted((BUILD / "whisper/bin").iterdir()):
        if path.is_file() and (path.name == "whisper-cli" or path.name.endswith(".dylib")):
            copy_file(path.resolve(), helpers / path.name)
    if not (helpers / "whisper-cli").is_file():
        raise RuntimeError("CPU recognizer build did not produce whisper-cli")
    environment = stage_notices()
    info = {"version": tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["version"],
            "architecture": "arm64", "minimum_macos_target": MINIMUM_MACOS,
            "signature": "ad-hoc", "apple_notarized": False, "languages": ["zh-TW", "en"],
            "whisper_commit": WHISPER_COMMIT}
    (RESOURCES / "build-info.json").write_text(json.dumps(info, indent=2) + "\n")
    inventory = {str(path.relative_to(RESOURCES)): digest(path)
                 for path in sorted(RESOURCES.rglob("*")) if path.is_file()}
    (BUILD / "payload-inventory.json").write_text(json.dumps({
        "resources": inventory, "helpers": sorted(path.name for path in helpers.iterdir()),
        "build_info": info}, indent=2) + "\n")
    (BUILD / "build-environment.json").write_text(json.dumps(environment, indent=2) + "\n")


def main():
    if sys.platform != "darwin" or platform.machine() != "arm64":
        raise RuntimeError("Build on an Apple Silicon Mac with Xcode Command Line Tools")
    for name, expected in (("pyinstaller", "6.22.3"), ("pyinstaller-hooks-contrib", "2026.8")):
        if version(name) != expected:
            raise RuntimeError("Install the pinned requirements-build.txt into this environment")
    run(sys.executable, ROOT / "scripts/check-public-scope.py")
    source = ROOT / "vendor/whisper.cpp"
    actual = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", str(source), "status", "--porcelain", "--untracked-files=no"], text=True)
    if actual != WHISPER_COMMIT or dirty:
        raise RuntimeError("whisper.cpp must match the clean pinned upstream checkout")
    BUILD.mkdir(parents=True, exist_ok=True)
    make_icon()
    run(ROOT / ".venv/bin/cmake", "-S", source, "-B", BUILD / "whisper", "-G", "Ninja",
        f"-DCMAKE_MAKE_PROGRAM={ROOT}/.venv/bin/ninja", "-DCMAKE_BUILD_TYPE=Release",
        f"-DCMAKE_OSX_DEPLOYMENT_TARGET={MINIMUM_MACOS}", "-DCMAKE_OSX_ARCHITECTURES=arm64",
        "-DWHISPER_BUILD_EXAMPLES=ON", "-DWHISPER_BUILD_TESTS=OFF", "-DGGML_NATIVE=OFF",
        "-DGGML_CPU_ARM_ARCH=armv8.2-a", "-DGGML_METAL=OFF", "-DGGML_OPENMP=OFF")
    run(ROOT / ".venv/bin/cmake", "--build", BUILD / "whisper", "--target", "whisper-cli", "--parallel",
        os.environ.get("KON_BUILD_JOBS", "4"))
    stage_resources()
    environment = dict(os.environ, MACOSX_DEPLOYMENT_TARGET=MINIMUM_MACOS,
                       PYINSTALLER_CONFIG_DIR=str(BUILD / "cache"), MPLCONFIGDIR=str(BUILD / "cache/matplotlib"))
    run(sys.executable, "-m", "PyInstaller", "--noconfirm", "--clean", "--distpath", ROOT / "dist/macos-arm64",
        "--workpath", BUILD / "pyinstaller", ROOT / "packaging/kon-macos.spec", env=environment)
    bundle = ROOT / "dist/macos-arm64/KON.app"
    for language in ("en", "zh-Hant"):
        copy_file(ROOT / f"packaging/{language}.lproj/InfoPlist.strings",
                  bundle / f"Contents/Resources/{language}.lproj/InfoPlist.strings")
    run("codesign", "--force", "--deep", "--sign", "-", bundle)
    run("codesign", "--verify", "--deep", "--strict", bundle)
    print(f"Built {bundle}; run .venv/bin/python scripts/verify-macos.py before packaging")


if __name__ == "__main__":
    main()
