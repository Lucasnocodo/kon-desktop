import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from kon.desktop import (
    DesktopTarget, FileSnapshot, pick_target, scan_desktop, TrashReceipt,
    restore_receipt,
)


@pytest.fixture
def desktop_scan(tmp_path, monkeypatch):
    desktop = tmp_path / "Desktop"
    desktop.mkdir()
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))

    def scan(rows):
        result = SimpleNamespace(returncode=0, stdout=json.dumps(rows), stderr="")
        monkeypatch.setattr("kon.desktop.subprocess.run", lambda *args, **kwargs: result)
        return scan_desktop((3440, 1440))

    return desktop, scan


@pytest.mark.parametrize("position", [{"x": 3374, "y": 630}, [3374, 630]])
def test_scan_accepts_jxa_point_object_and_legacy_pair(desktop_scan, position):
    desktop, scan = desktop_scan
    path = desktop / "桌面 file.txt"
    path.write_text("keep me")

    targets = scan([{"name": path.name, "url": path.as_uri(), "position": position}])

    assert len(targets) == 1
    assert targets[0].id == str(path)
    assert targets[0].name == path.name
    assert targets[0].center == (3374.0, 630.0)
    assert targets[0].snapshot == FileSnapshot.capture(path, desktop)
    assert path.read_text() == "keep me"


@pytest.mark.parametrize("position", [
    None, "12", [100], [100, 200, 300], {}, {"x": 100}, {"y": 200},
    {"x": "invalid", "y": 200}, {"x": None, "y": 200},
    {"x": True, "y": 200}, {"x": 100, "y": False}, [True, 200],
    {"x": float("inf"), "y": 200}, {"x": 100, "y": float("nan")},
    {"x": 10 ** 500, "y": 200},
])
def test_scan_ignores_invalid_positions_without_losing_valid_files(desktop_scan, position):
    desktop, scan = desktop_scan
    invalid = desktop / "invalid.txt"
    valid = desktop / "valid.txt"
    invalid.write_text("invalid position")
    valid.write_text("valid position")

    targets = scan([
        {"name": invalid.name, "url": invalid.as_uri(), "position": position},
        {"name": valid.name, "url": valid.as_uri(), "position": {"x": 200, "y": 300}},
    ])

    assert [target.name for target in targets] == [valid.name]
    assert invalid.exists() and valid.exists()


@pytest.mark.parametrize("position", [
    {"x": -1, "y": 100}, {"x": 3441, "y": 100},
    {"x": 100, "y": -1}, {"x": 100, "y": 1441},
])
def test_scan_excludes_icons_outside_primary_screen(desktop_scan, position):
    desktop, scan = desktop_scan
    path = desktop / "offscreen.txt"
    path.write_text("keep me")
    assert scan([{"name": path.name, "url": path.as_uri(), "position": position}]) == []
    assert path.exists()


def test_scan_keeps_file_safety_checks_for_jxa_point_objects(desktop_scan):
    desktop, scan = desktop_scan
    visible = desktop / "visible.txt"
    hidden = desktop / ".hidden.txt"
    outside = desktop.parent / "outside.txt"
    folder = desktop / "folder"
    link = desktop / "link.txt"
    for path in (visible, hidden, outside):
        path.write_text("keep me")
    folder.mkdir()
    link.symlink_to(visible)
    paths = [visible, hidden, outside, folder, link, desktop / "missing.txt"]
    rows = [{"name": path.name, "url": path.as_uri(), "position": {"x": 200, "y": 300}}
            for path in paths]
    rows += [
        {"name": "remote.txt", "url": "file://server" + str(visible), "position": {"x": 200, "y": 300}},
        {"name": "web.txt", "url": "https://example.com/file.txt", "position": {"x": 200, "y": 300}},
    ]

    assert [target.name for target in scan(rows)] == [visible.name]
    assert visible.exists() and hidden.exists() and outside.exists()
    assert folder.is_dir() and link.is_symlink()


def test_snapshot_refuses_outside_directory_symlink_and_folder(tmp_path):
    desktop = tmp_path / "Desktop"
    desktop.mkdir()
    file = desktop / "file.txt"
    file.write_text("gao")
    assert FileSnapshot.capture(file, desktop).path == file
    (desktop / "link").symlink_to(file)
    (desktop / "folder").mkdir()
    outside = tmp_path / "outside"
    outside.write_text("outside")
    for rejected in [outside, desktop / "link", desktop / "folder"]:
        with pytest.raises(ValueError):
            FileSnapshot.capture(rejected, desktop)


def test_replaced_or_edited_file_is_not_consumed(tmp_path):
    path = tmp_path / "file.txt"
    path.write_text("original")
    captured = FileSnapshot.capture(path, tmp_path)
    path.write_text("new")
    with pytest.raises(ValueError):
        captured.verify(tmp_path)
    captured = FileSnapshot.capture(path, tmp_path)
    replacement = tmp_path / "replacement.txt"
    replacement.write_text("new")
    os.replace(replacement, path)
    with pytest.raises(ValueError):
        captured.verify(tmp_path)


def test_overlapping_icons_have_no_target():
    a = DesktopTarget("a", "a", (100, 100))
    b = DesktopTarget("b", "b", (115, 100))
    assert pick_target([a, b], (110, 100)) is None
    assert pick_target([a], (100, 100)) is a
    assert pick_target([a], None) is None


def test_restore_uses_exclusive_rename_and_never_overwrites(tmp_path):
    trash = tmp_path / "trash.txt"
    original = tmp_path / "original.txt"
    trash.write_text("restore me")
    info = trash.stat()
    receipt = TrashReceipt(original, trash, info.st_dev, info.st_ino)
    original.write_text("keep me")
    with pytest.raises(ValueError):
        restore_receipt(receipt)
    assert original.read_text() == "keep me"
    original.unlink()
    restore_receipt(receipt)
    assert original.read_text() == "restore me"
    assert not trash.exists()


def test_icon_rearrangement_cancels_cached_target(monkeypatch):
    from kon.desktop import validate_target_position
    old = DesktopTarget("a", "a", (100, 100))
    moved = DesktopTarget("a", "a", (200, 100))
    monkeypatch.setattr("kon.desktop.scan_desktop", lambda _: [moved])
    with pytest.raises(ValueError):
        validate_target_position(old, (100, 100), (1000, 800))
    monkeypatch.setattr("kon.desktop.scan_desktop", lambda _: [old])
    assert validate_target_position(old, (100, 100), (1000, 800)) is old


def test_new_overlapping_icon_cancels_target(monkeypatch):
    from kon.desktop import validate_target_position
    old = DesktopTarget("a", "a", (100, 100))
    overlap = DesktopTarget("b", "b", (110, 100))
    monkeypatch.setattr("kon.desktop.scan_desktop", lambda _: [old, overlap])
    with pytest.raises(ValueError):
        validate_target_position(old, (100, 100), (1000, 800))
