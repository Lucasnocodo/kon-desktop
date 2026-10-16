"""Download public model weights and verify committed size/SHA-256 expectations."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import ssl
import tempfile
import urllib.request

import certifi

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_PATH = ROOT / "models.expected.json"


def verify_model(path: Path, expected: dict) -> None:
    """Reject changed, truncated, or untrusted weights, including existing files."""
    size = path.stat().st_size
    with path.open("rb") as source:
        digest = hashlib.file_digest(source, "sha256").hexdigest()
    if size != expected["bytes"] or digest != expected["sha256"]:
        raise RuntimeError(
            f"Model integrity check failed for {path.name}: "
            f"expected {expected['bytes']} bytes / {expected['sha256']}, "
            f"got {size} bytes / {digest}. Remove the invalid file and rerun setup."
        )


def ensure_model(path: Path, expected: dict, *, context: ssl.SSLContext) -> None:
    if path.exists():
        verify_model(path, expected)
        return
    # A unique temporary file in the destination directory makes publication
    # atomic and prevents an interrupted transfer from becoming a ready model.
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".download", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as output:
            with urllib.request.urlopen(expected["url"], context=context, timeout=60) as response:
                while chunk := response.read(1024 * 1024):
                    output.write(chunk)
        verify_model(temporary, expected)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    expected_models = json.loads(EXPECTED_PATH.read_text(encoding="utf-8"))
    destination = ROOT / "models"
    destination.mkdir(exist_ok=True)
    context = ssl.create_default_context(cafile=certifi.where())
    for name, expected in expected_models.items():
        if Path(name).name != name:
            raise ValueError(f"Invalid model filename: {name}")
        ensure_model(destination / name, expected, context=context)
        print(f"Verified: {name} ({expected['bytes'] / 1024**2:.1f} MB)", flush=True)
    # This ignored receipt is informational; the tracked expectation is the
    # authority on every run, never a hash learned from downloaded bytes.
    (destination / "manifest.json").write_text(json.dumps(expected_models, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
