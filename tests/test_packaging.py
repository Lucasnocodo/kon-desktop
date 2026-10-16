"""Release audit regressions for external loads and exact validated bytes."""
import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/verify-macos.py"
spec = importlib.util.spec_from_file_location("bundle_verifier", SCRIPT)
verifier = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verifier)


def test_external_python_runtime_and_loader_escapes_are_rejected(tmp_path):
    bundle = tmp_path / "KON.app"
    binary = bundle / "Contents/Frameworks/example.dylib"
    binary.parent.mkdir(parents=True)
    binary.write_bytes(b"sample")
    assert verifier.reference_issue("/Library/Frameworks/Python.framework/Python", binary, bundle)
    assert verifier.reference_issue("@loader_path/../../../../private.dylib", binary, bundle)
    assert verifier.reference_issue("/System/Library/Frameworks/Cocoa.framework/Cocoa", binary, bundle) is None
    assert verifier.reference_issue("/usr/lib/libSystem.B.dylib", binary, bundle) is None


def test_relative_dylib_requires_an_existing_in_bundle_target(tmp_path):
    bundle = tmp_path / "KON.app"
    binary = bundle / "Contents/Frameworks/helpers/whisper-cli"
    binary.parent.mkdir(parents=True)
    binary.write_bytes(b"sample")
    assert verifier.reference_issue("@rpath/libwhisper.dylib", binary, bundle)
    library = binary.parent / "libwhisper.dylib"
    library.write_bytes(b"sample library")
    assert verifier.reference_issue("@rpath/libwhisper.dylib", binary, bundle) is None
    library.unlink()
    library.symlink_to(tmp_path / "outside.dylib")
    (tmp_path / "outside.dylib").write_bytes(b"external")
    assert verifier.reference_issue("@rpath/libwhisper.dylib", binary, bundle)


def test_validation_fingerprint_changes_with_resource_or_link_replacement(tmp_path):
    bundle = tmp_path / "KON.app"
    bundle.mkdir()
    resource = bundle / "resource.json"
    resource.write_text('{"language":"en"}')
    first = verifier.bundle_fingerprint(bundle)
    resource.write_text('{"language":"zh-TW"}')
    second = verifier.bundle_fingerprint(bundle)
    assert first != second
    alias = bundle / "alias"
    alias.symlink_to("resource.json")
    third = verifier.bundle_fingerprint(bundle)
    assert third != second
    alias.unlink()
    alias.symlink_to("other.json")
    assert verifier.bundle_fingerprint(bundle) != third


BUILD_SCRIPT = SCRIPT.with_name("build-macos.py")
build_spec = importlib.util.spec_from_file_location("bundle_builder", BUILD_SCRIPT)
builder = importlib.util.module_from_spec(build_spec)
build_spec.loader.exec_module(builder)


def test_notice_collection_rejects_copying_test_binaries_and_debug_symbols(tmp_path):
    for relative in (
        "PyObjCTest/copying.cpython-313-darwin.so",
        "PyObjCTest/copying.cpython-313-darwin.so.dSYM/Contents/Resources/DWARF/COPYING",
        "PyObjCTest/copying.py",
    ):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        # Even a text-like native test payload is not a license notice.
        path.write_text("not a distribution license")
        assert not builder.is_license_text(path, Path(relative))


def test_notice_collection_retains_metadata_and_legacy_text_licenses(tmp_path):
    for relative in (
        "example.dist-info/licenses/vendor/LICENSE",
        "example.dist-info/licenses/vendor/MIT",
        "example.dist-info/NOTICE.txt",
        "cv2/LICENSE-3RD-PARTY.txt",
        "fontTools/LICENSE.external",
    ):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("Copyright © contributors\nPermission is granted...", encoding="utf-8")
        assert builder.is_license_text(path, Path(relative))


def test_notice_collection_rejects_binary_payload_disguised_as_text(tmp_path):
    path = tmp_path / "LICENSE.txt"
    for content in (b"header\0binary", b"\xff\xfe\x89binary", b""):
        path.write_bytes(content)
        assert not builder.is_license_text(path, Path("example.dist-info/LICENSE.txt"))
