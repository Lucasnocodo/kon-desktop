"""Finder icon positions and reversible, identity-checked Desktop trashing."""
from __future__ import annotations

import json
import math
import os
import stat
import subprocess
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote, urlparse


@dataclass(frozen=True)
class FileSnapshot:
    path: Path
    device: int
    inode: int
    size: int
    modified_ns: int

    @classmethod
    def capture(cls, path: Path, desktop: Path | None = None):
        desktop = (desktop or Path.home() / "Desktop").resolve()
        path = Path(os.path.abspath(path))
        if path.parent.resolve() != desktop or path.name.startswith("."):
            raise ValueError("只接受桌面第一層的可見檔案。")
        info = path.lstat()
        if not stat.S_ISREG(info.st_mode):
            raise ValueError("第一版僅接受一般檔案，不處理資料夾或符號連結。")
        return cls(path, info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns)

    def verify(self, desktop: Path | None = None):
        if FileSnapshot.capture(self.path, desktop) != self:
            raise ValueError("目標檔案已改變，請重新掃描桌面。")


@dataclass(frozen=True)
class DesktopTarget:
    id: str
    name: str
    center: tuple[float, float]
    snapshot: FileSnapshot | None = None
    radius: float = 40.0


@dataclass(frozen=True)
class TrashReceipt:
    original: Path
    trashed: Path
    device: int
    inode: int


def pick_target(targets: list[DesktopTarget], point: tuple[float, float] | None):
    if point is None:
        return None
    matches = [target for target in targets if math.dist(target.center, point) <= target.radius]
    # Overlapping/stacked icons must never resolve by arbitrary list order.
    return matches[0] if len(matches) == 1 else None


_FINDER_SCRIPT = '''
const finder = Application("Finder");
const rows = [];
for (const item of finder.desktop.items()) {
  try {
    rows.push({name:item.name(), url:item.url(), position:item.desktopPosition()});
  } catch (e) { /* inaccessible items are never targets */ }
}
JSON.stringify(rows);
'''


def _finder_position(value) -> tuple[float, float]:
    """JXA serializes Finder points as {x, y}; older scans used pairs."""
    if isinstance(value, dict):
        value = (value["x"], value["y"])
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError("Finder 圖示位置格式無效。")
    if any(isinstance(coordinate, bool) for coordinate in value):
        raise ValueError("Finder 圖示位置格式無效。")
    x, y = map(float, value)
    return x, y


def scan_desktop(screen_size: tuple[int, int]) -> list[DesktopTarget]:
    result = subprocess.run(
        ["/usr/bin/osascript", "-l", "JavaScript", "-e", _FINDER_SCRIPT],
        capture_output=True, text=True, timeout=20,
    )
    if result.returncode:
        raise RuntimeError("無法讀取 Finder 桌面位置。請允許啟動程式控制 Finder，並使用一般桌面圖示排列。\n" + result.stderr.strip())
    rows = json.loads(result.stdout)
    targets = []
    width, height = screen_size
    for row in rows:
        try:
            url = urlparse(row["url"])
            if url.scheme != "file" or url.netloc not in ("", "localhost"):
                continue
            snapshot = FileSnapshot.capture(Path(unquote(url.path)))
            x, y = _finder_position(row["position"])
            if not (math.isfinite(x) and math.isfinite(y) and 0 <= x <= width and 0 <= y <= height):
                continue
            targets.append(DesktopTarget(str(snapshot.path), row["name"], (x, y), snapshot))
        except (ValueError, OSError, TypeError, KeyError, OverflowError):
            continue
    return targets


def validate_target_position(target: DesktopTarget, aimed_point, screen_size) -> DesktopTarget:
    """Refresh Finder positions before committing an animation to a real file."""
    from .grid_selection import GridSelector
    current_targets = scan_desktop(screen_size)
    current = next((value for value in current_targets if value.id == target.id), None)
    chosen = pick_target(current_targets, aimed_point)
    grid_chosen = GridSelector().pick(current_targets, current.center, screen_size) if current else None
    if (current is None or current.snapshot != target.snapshot
            or math.dist(current.center, target.center) > 6.0
            or chosen is None or chosen.id != target.id
            or grid_chosen is None or grid_chosen.id != target.id):
        raise ValueError("桌面檔案或圖示位置已改變；已取消召喚，請重新讀取桌面。")
    return current


def trash_target(snapshot: FileSnapshot) -> TrashReceipt:
    from Foundation import NSFileManager, NSURL
    snapshot.verify()
    success, resulting_url, error = NSFileManager.defaultManager().trashItemAtURL_resultingItemURL_error_(
        NSURL.fileURLWithPath_(str(snapshot.path)), None, None
    )
    if not success:
        raise RuntimeError(str(error.localizedDescription()) if error else "移到垃圾桶失敗。")
    if resulting_url is None:
        raise RuntimeError("檔案已移到垃圾桶；系統未提供復原路徑，請從垃圾桶取回。")
    trashed = Path(str(resulting_url.path()))
    info = trashed.lstat()
    return TrashReceipt(snapshot.path, trashed, info.st_dev, info.st_ino)


def restore_receipt(receipt: TrashReceipt) -> None:
    info = receipt.trashed.lstat()
    if (info.st_dev, info.st_ino) != (receipt.device, receipt.inode) or not stat.S_ISREG(info.st_mode):
        raise ValueError("垃圾桶中的檔案已改變，請手動復原。")
    if os.path.lexists(receipt.original):
        raise ValueError("原位置已有同名檔案；不會覆寫，請手動復原。")
    # macOS RENAME_EXCL makes the no-overwrite check atomic.
    import ctypes
    library = ctypes.CDLL("/usr/lib/libSystem.B.dylib", use_errno=True)
    library.renamex_np.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
    library.renamex_np.restype = ctypes.c_int
    if library.renamex_np(os.fsencode(receipt.trashed), os.fsencode(receipt.original), 0x4):
        raise OSError(ctypes.get_errno(), "復原失敗；原位置不會被覆寫。")
