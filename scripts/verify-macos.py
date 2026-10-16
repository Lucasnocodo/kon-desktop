#!/usr/bin/env python3
"""Audit the real bundle, then run finite checks using its own executable."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import plistlib
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "build/macos-arm64"
ARM64 = 16777228
MACHO_MAGIC = {bytes.fromhex(value) for value in ("cffaedfe", "feedfacf", "cafebabe", "bebafeca", "cafebabf", "bfbafeca")}
LOAD_COMMANDS = {12, 24, 0x80000018, 0x8000001F, 0x20, 0x80000023}


def digest(path):
    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def bundle_fingerprint(bundle: Path) -> str:
    """Bind every validation report to the same exact file/link inventory."""
    state = hashlib.sha256()
    for path in sorted(bundle.rglob("*")):
        relative = str(path.relative_to(bundle))
        if path.is_symlink():
            value = "link:" + os.readlink(path)
        elif path.is_file():
            value = digest(path)
        else:
            continue
        state.update(relative.encode() + b"\0" + value.encode() + b"\0")
    return state.hexdigest()


def decode_version(value):
    return [value >> 16, (value >> 8) & 255, value & 255]


def reference_issue(reference: str, binary: Path, bundle: Path) -> str | None:
    """Reject external absolute paths and escaped/missing relative load targets."""
    if reference.startswith(("/System/Library/", "/usr/lib/")):
        return None
    if reference.startswith("/"):
        return "external absolute load path"
    if reference.startswith("@loader_path/"):
        candidate = binary.parent / reference[len("@loader_path/"):]
    elif reference.startswith("@executable_path/"):
        candidate = bundle / "Contents/MacOS" / reference[len("@executable_path/"):]
    elif reference.startswith("@rpath/"):
        tail = reference[len("@rpath/"):]
        candidates = [binary.parent / tail, bundle / "Contents/Frameworks" / tail,
                      bundle / "Contents/MacOS" / tail]
        return None if any(path.resolve().is_relative_to(bundle) and path.exists() for path in candidates) else "unresolved bundled rpath"
    else:
        return "nonportable load path"
    return None if candidate.resolve().is_relative_to(bundle) and candidate.exists() else "escaped or missing bundled load target"


def audit_bundle(bundle: Path, inventory: dict) -> dict:
    from macholib.MachO import MachO
    from macholib.mach_o import LC_BUILD_VERSION, LC_VERSION_MIN_MACOSX
    failures, binaries, seen = [], [], set()
    target = [int(value) for value in inventory["build_info"]["minimum_macos_target"].split(".")]
    target += [0] * (3 - len(target))
    resources = bundle / "Contents/Resources"
    plist = plistlib.loads((bundle / "Contents/Info.plist").read_bytes())
    if plist.get("NSLocalNetworkUsageDescription"):
        failures.append({"info_plist": "unexpected local-network permission"})
    if plist.get("LSMinimumSystemVersion") != inventory["build_info"]["minimum_macos_target"]:
        failures.append({"info_plist": "deployment target differs from build inventory"})
    if set(plist.get("CFBundleLocalizations", [])) != {"en", "zh-Hant"}:
        failures.append({"info_plist": "missing bilingual application metadata"})
    for language in ("en", "zh-Hant"):
        if not (resources / f"{language}.lproj/InfoPlist.strings").is_file():
            failures.append({"localization": language})
    for path in bundle.rglob("*"):
        relative = str(path.relative_to(bundle))
        if path.is_symlink() and not path.resolve().is_relative_to(bundle):
            failures.append({"path": relative, "reason": "external symlink"})
            continue
        if not path.is_file() or path.resolve() in seen:
            continue
        seen.add(path.resolve())
        with path.open("rb") as source:
            magic = source.read(4)
        if magic not in MACHO_MAGIC:
            continue
        slices = []
        for header in MachO(str(path)).headers:
            minimum = None
            references = []
            for command, body, data in header.commands:
                if command.cmd in (LC_BUILD_VERSION, LC_VERSION_MIN_MACOSX):
                    minimum = decode_version(body.minos if command.cmd == LC_BUILD_VERSION else body.version)
                if command.cmd in LOAD_COMMANDS or command.cmd == 0x8000001C:
                    reference = data.rstrip(b"\0").decode(errors="replace")
                    if command.cmd == 0x8000001C:
                        if reference.startswith("/") and not reference.startswith(("/System/Library/", "/usr/lib/")):
                            failures.append({"path": relative, "reason": "external rpath", "reference": reference})
                    else:
                        references.append(reference)
                        issue = reference_issue(reference, path, bundle)
                        if issue:
                            failures.append({"path": relative, "reason": issue, "reference": reference})
            slices.append({"cpu_type": header.header.cputype, "minimum_macos": minimum})
            if minimum is None or minimum > target:
                failures.append({"path": relative, "reason": "unsupported or unknown deployment minimum", "slice": slices[-1]})
        if ARM64 not in {part["cpu_type"] for part in slices}:
            failures.append({"path": relative, "reason": "missing arm64 slice"})
        binaries.append({"path": relative, "slices": slices})
    expected = inventory["resources"]
    managed = ("assets", "models", "test-fixtures", "kon/locales", "ThirdPartyNotices")
    actual = {str(path.relative_to(resources)) for folder in managed
              for path in (resources / folder).rglob("*") if path.is_file()}
    actual.add("build-info.json")
    if set(expected) != actual:
        failures.append({"payload_inventory_mismatch": sorted(set(expected) ^ actual)})
    for relative, expected_digest in expected.items():
        path = resources / relative
        if not path.is_file() or digest(path) != expected_digest:
            failures.append({"path": relative, "reason": "payload hash mismatch"})
    actual_helpers = {path.name for path in (resources / "helpers").iterdir() if path.is_file()}
    if actual_helpers != set(inventory["helpers"]):
        failures.append({"helper_inventory_mismatch": sorted(actual_helpers ^ set(inventory["helpers"]))})
    signature = subprocess.run(["codesign", "--verify", "--deep", "--strict", str(bundle)], capture_output=True, text=True)
    if signature.returncode:
        failures.append({"signature": signature.stderr.strip()})
    metadata = subprocess.run(["codesign", "--display", "--verbose=4", str(bundle)], capture_output=True, text=True)
    if "Signature=adhoc" not in metadata.stderr:
        failures.append({"signature": "expected an explicit ad-hoc signature"})
    return {"ok": not failures, "version": inventory["build_info"]["version"], "architecture": "arm64",
            "minimum_macos": ".".join(map(str, target[:2])), "apple_notarized": False,
            "signature": "ad-hoc", "signature_valid": signature.returncode == 0,
            "macho_files": len(binaries), "resource_files": len(expected),
            "bundle_fingerprint": bundle_fingerprint(bundle), "failures": failures, "binaries": binaries}


def run_diagnostic(bundle, flag, output, language=None):
    command = [str(bundle / "Contents/MacOS/KON"), flag, "--output", str(output)]
    if language:
        command.extend(["--language", language])
    environment = standalone_environment()
    with tempfile.TemporaryDirectory(prefix="kon-independent-launch-") as temporary:
        process = subprocess.run(command, env=environment, cwd=temporary,
                                 capture_output=True, text=True, timeout=300)
    if not output.is_file():
        raise RuntimeError(f"Frozen diagnostic produced no report ({process.returncode}): {process.stderr[-2000:]}")
    report = json.loads(output.read_text())
    if process.returncode or not report.get("ok"):
        raise RuntimeError(f"Frozen diagnostic failed: {json.dumps(report, ensure_ascii=False)}")
    report["bundle_fingerprint"] = bundle_fingerprint(bundle)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    return report


def standalone_environment():
    """Keep diagnostics independent of the developer's Python and tool PATH."""
    environment = dict(os.environ, QT_QPA_PLATFORM="offscreen",
                       PATH="/usr/bin:/bin:/usr/sbin:/sbin")
    for name in ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV", "CONDA_PREFIX"):
        environment.pop(name, None)
    return environment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--app", type=Path, default=ROOT / "dist/macos-arm64/KON.app")
    parser.add_argument("--output-dir", type=Path, default=BUILD / "verification")
    parser.add_argument("--binary-only", action="store_true", help="Audit only; not sufficient to package a release")
    args = parser.parse_args()
    bundle = args.app.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    inventory = json.loads((BUILD / "payload-inventory.json").read_text())
    audit = audit_bundle(bundle, inventory)
    (args.output_dir / "binary-audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    if not audit["ok"]:
        print(json.dumps(audit["failures"], indent=2), file=sys.stderr)
        return 1
    if not args.binary_only:
        for language in ("en", "zh-TW"):
            run_diagnostic(bundle, "--package-check", args.output_dir / f"frozen-{language}.json", language)
        run_diagnostic(bundle, "--speech-self-test", args.output_dir / "frozen-speech.json")
        # A separate process exercises the public language flag and rendering path.
        with tempfile.TemporaryDirectory(prefix="kon-frozen-render-") as temporary:
            environment = dict(standalone_environment(), KON_DATA_DIR=temporary,
                               MPLCONFIGDIR=str(Path(temporary) / "matplotlib"))
            for language in ("en", "zh-TW"):
                subprocess.run([str(bundle / "Contents/MacOS/KON"), "--language", language, "--screenshot",
                                str((args.output_dir / f"control-{language}.png").resolve())],
                               env=environment, cwd=temporary, check=True, timeout=60)
        if bundle_fingerprint(bundle) != audit["bundle_fingerprint"]:
            raise RuntimeError("Hardware-free diagnostics modified the application bundle")
    print(json.dumps({key: audit[key] for key in ("ok", "version", "minimum_macos", "macho_files", "resource_files", "signature")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
