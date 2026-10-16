#!/usr/bin/env python3
"""Archive only an unchanged bundle whose binary and frozen checks all passed."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import shutil
import subprocess
import tomllib

ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "build/macos-arm64"


def main():
    subprocess.run(["git", "diff", "--quiet", "HEAD"], cwd=ROOT, check=True)
    source_revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    version = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["version"]
    bundle = ROOT / "dist/macos-arm64/KON.app"
    spec = importlib.util.spec_from_file_location("kon_bundle_audit", ROOT / "scripts/verify-macos.py")
    verifier = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(verifier)
    fingerprint = verifier.bundle_fingerprint(bundle)
    reports = {}
    for name in ("binary-audit", "frozen-en", "frozen-zh-TW", "frozen-speech"):
        report = json.loads((BUILD / f"verification/{name}.json").read_text())
        if not report.get("ok") or report.get("bundle_fingerprint") != fingerprint or report.get("version") != version:
            raise RuntimeError(f"Run scripts/verify-macos.py for the exact current bundle: {name}")
        reports[name] = report
    destination = ROOT / f"dist/KON-{version}-macOS-arm64"
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)
    subprocess.run(["ditto", str(bundle), str(destination / "KON.app")], check=True)
    if verifier.bundle_fingerprint(destination / "KON.app") != fingerprint:
        raise RuntimeError("Application changed while preparing the distribution")
    for name in ("INSTALL-en.md", "開始使用-繁體中文.md"):
        shutil.copy2(ROOT / "packaging" / name, destination / name)
    shutil.copytree(bundle / "Contents/Resources/ThirdPartyNotices", destination / "ThirdPartyNotices")
    validation = destination / "Validation"
    validation.mkdir()
    for name in reports:
        shutil.copy2(BUILD / f"verification/{name}.json", validation / f"{name}.json")
    for language in ("en", "zh-TW"):
        shutil.copy2(BUILD / f"verification/control-{language}.png", validation / f"control-{language}.png")
    audit = reports["binary-audit"]
    release = {"version": version, "architecture": "arm64", "minimum_macos": audit["minimum_macos"],
               "source_repository": "https://github.com/Lucasnocodo/kon-desktop",
               "source_revision": source_revision,
               "tested_macos": platform.mac_ver()[0], "minimum_macos_hardware_tested": False,
               "apple_notarized": False, "signature": "ad-hoc", "languages": ["zh-TW", "en"],
               "user_data_directory": "~/Library/Application Support/KON Desktop/",
               "bundle_fingerprint": fingerprint, "macho_files_audited": audit["macho_files"],
               "camera_opened_during_automated_checks": False,
               "microphone_opened_during_automated_checks": False,
               "real_desktop_files_modified_during_automated_checks": False,
               "notes": ["Self-contained Python, verified models and CPU speech recognizer.",
                         "Binary deployment minimum is audited; oldest supported OS still needs hardware testing.",
                         "Image/audio media remain outside the MIT code license; see ThirdPartyNotices/Media-NOTICE.md."]}
    (destination / "Release.json").write_text(json.dumps(release, ensure_ascii=False, indent=2) + "\n")
    archive = ROOT / f"dist/KON-{version}-macOS-arm64.zip"
    subprocess.run(["ditto", "-c", "-k", "--sequesterRsrc", "--keepParent", str(destination), str(archive)], check=True)
    with archive.open("rb") as source:
        sha = hashlib.file_digest(source, "sha256").hexdigest()
    (ROOT / "dist/SHA256SUMS").write_text(f"{sha}  {archive.name}\n")
    print(json.dumps({"archive": str(archive), "bytes": archive.stat().st_size, "sha256": sha}, indent=2))


if __name__ == "__main__":
    main()
