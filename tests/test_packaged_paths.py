"""Frozen resources remain read-only; per-user state survives moving the App."""
import os
from pathlib import Path
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

from kon import paths


@pytest.fixture
def bundle(tmp_path, monkeypatch):
    contents = tmp_path / "分享 App.app/Contents"
    resources = contents / "Resources"
    frameworks = contents / "Frameworks"
    resources.mkdir(parents=True)
    frameworks.mkdir()
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "_MEIPASS", str(frameworks), raising=False)
    monkeypatch.setattr(sys, "executable", str(contents / "MacOS/KON"))
    data = tmp_path / "使用者資料"
    monkeypatch.setenv("KON_DATA_DIR", str(data))
    return resources, data


def test_packaged_root_uses_resources_instead_of_frameworks(bundle):
    resources, data = bundle
    assert paths.resource_root() == resources
    assert paths.state_root(resources) == data
    assert not data.exists()  # Merely inspecting paths does not create state.
    assert paths.helper_path("whisper-cli") == resources / "helpers/whisper-cli"


def test_explicit_dev_project_keeps_existing_state_layout(bundle, tmp_path):
    project = tmp_path / "開發專案"
    assert paths.state_root(project) == project / ".kon"


def test_default_packaged_data_location_does_not_depend_on_bundle_path(bundle, monkeypatch):
    monkeypatch.delenv("KON_DATA_DIR")
    expected = Path.home() / "Library/Application Support/KON"
    assert paths.state_root() == expected.resolve()
