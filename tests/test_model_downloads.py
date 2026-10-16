"""Weights must be verified before use, including already-downloaded files."""
from __future__ import annotations

import hashlib
import importlib.util
import io
import json
from pathlib import Path
import ssl
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("kon_model_downloads", ROOT / "scripts/download-models.py")
downloads = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(downloads)


def expectation(data):
    return {"url": "https://example.invalid/model", "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest()}


def test_existing_verified_model_is_not_downloaded(tmp_path):
    path = tmp_path / "weights.bin"
    path.write_bytes(b"verified weights")
    with patch.object(downloads.urllib.request, "urlopen") as network:
        downloads.ensure_model(path, expectation(path.read_bytes()), context=ssl.create_default_context())
    network.assert_not_called()


def test_existing_same_size_modified_model_is_rejected_without_overwrite(tmp_path):
    path = tmp_path / "weights.bin"
    path.write_bytes(b"wrong")
    with patch.object(downloads.urllib.request, "urlopen") as network:
        with pytest.raises(RuntimeError, match="integrity check failed"):
            downloads.ensure_model(path, expectation(b"right"), context=ssl.create_default_context())
    assert path.read_bytes() == b"wrong"
    network.assert_not_called()


def test_download_is_published_only_after_verification(tmp_path):
    path = tmp_path / "weights.bin"
    data = b"model bytes"
    def open_remote(*args, **kwargs):
        assert not path.exists()
        return io.BytesIO(data)
    with patch.object(downloads.urllib.request, "urlopen", side_effect=open_remote):
        downloads.ensure_model(path, expectation(data), context=ssl.create_default_context())
    assert path.read_bytes() == data
    assert list(tmp_path.iterdir()) == [path]


@pytest.mark.parametrize("bad_data", [b"truncated", b"x" * len(b"expected model")])
def test_bad_download_never_becomes_model_and_removes_temporary_file(tmp_path, bad_data):
    path = tmp_path / "weights.bin"
    with patch.object(downloads.urllib.request, "urlopen", return_value=io.BytesIO(bad_data)):
        with pytest.raises(RuntimeError, match="integrity check failed"):
            downloads.ensure_model(path, expectation(b"expected model"), context=ssl.create_default_context())
    assert list(tmp_path.iterdir()) == []


def test_interrupted_response_removes_temporary_file(tmp_path):
    path = tmp_path / "weights.bin"
    response = io.BytesIO(b"partial")
    with patch.object(response, "read", side_effect=OSError("connection lost")):
        with patch.object(downloads.urllib.request, "urlopen", return_value=response):
            with pytest.raises(OSError, match="connection lost"):
                downloads.ensure_model(path, expectation(b"whole"), context=ssl.create_default_context())
    assert list(tmp_path.iterdir()) == []


def test_committed_model_expectations_are_not_download_receipts():
    expected = json.loads((ROOT / "models.expected.json").read_text())
    assert set(expected) == {"hand_landmarker.task", "face_landmarker.task", "ggml-tiny.bin"}
    assert all(item["url"].startswith("https://") and len(item["sha256"]) == 64
               and item["bytes"] > 1_000_000 for item in expected.values())
