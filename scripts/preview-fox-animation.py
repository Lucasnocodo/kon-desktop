#!/usr/bin/env python3
"""Render KON's real Qt animation without sensors or desktop file operations.

Run with the project environment:
    .venv/bin/python scripts/preview-fox-animation.py

The default output includes every timeline frame, an annotated contact sheet,
and an animated PNG when Pillow is installed. Use --smoke-only to export the
same smoke effect by itself with a transparent background.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
from unittest.mock import patch

os.environ["QT_QPA_PLATFORM"] = "offscreen"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QFont, QImage, QLinearGradient, QPainter, QPen
from PySide6.QtWidgets import QApplication

from kon.desktop import DesktopTarget
import kon.overlay as overlay_module


def render_frame(overlay, elapsed, smoke_only=False):
    """Use the overlay's painter directly with a deterministic animation clock."""
    image = QImage(overlay.width(), overlay.height(), QImage.Format_ARGB32_Premultiplied)
    image.fill(Qt.transparent if smoke_only else QColor("#20242c"))
    painter = QPainter(image)
    try:
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setRenderHint(QPainter.SmoothPixmapTransform)
        if smoke_only:
            pose = overlay._eating_pose(elapsed)
            painter.translate(overlay.width(), 0)
            painter.scale(-1, 1)
            for behind in (True, False):
                overlay_module._draw_smoke(painter, pose, elapsed, overlay.reduce_motion, behind=behind)
            return image
        wallpaper = QLinearGradient(QPointF(0, 0), QPointF(image.width(), image.height()))
        wallpaper.setColorAt(0, QColor("#303440"))
        wallpaper.setColorAt(1, QColor("#171b23"))
        painter.fillRect(image.rect(), wallpaper)
        painter.setPen(QPen(QColor(255, 255, 255, 9), 1))
        for x in range(0, image.width(), 80):
            painter.drawLine(QPointF(x, 0), QPointF(x, image.height()))
        for y in range(0, image.height(), 80):
            painter.drawLine(QPointF(0, y), QPointF(image.width(), y))
        painter.fillRect(QRectF(0, 0, image.width(), 34), QColor(12, 15, 21, 120))
        painter.setPen(QColor("#e5e6e9"))
        painter.setFont(QFont("PingFang TC", 11))
        painter.drawText(QRectF(20, 2, image.width() - 40, 30), Qt.AlignVCenter,
                         "KON   妖狐吞噬動畫預覽")
        with patch.object(overlay_module.time, "monotonic", return_value=elapsed):
            overlay._paint_eating(painter)
    finally:
        painter.end()
    return image


def save_contact_sheet(overlay, output, smoke_only=False):
    bite = overlay_module.BITE_TIME
    smoke_start = overlay_module.SMOKE_START
    smoke_speed = overlay_module.SMOKE_SPEED
    # Preserve exact sprite start values so every authored pose is shown.
    timestamps = sorted(set(overlay_module.FRAME_TIMES + (
        overlay_module.SWALLOW_END, bite, bite + .02, smoke_start,
        smoke_start + .08 / smoke_speed, smoke_start + .18 / smoke_speed,
        smoke_start + .30 / smoke_speed, smoke_start + .45 / smoke_speed,
        smoke_start + .60 / smoke_speed, overlay_module.DURATION - .20 / smoke_speed,
        overlay_module.DURATION - .07 / smoke_speed, overlay_module.DURATION,
    )))
    if smoke_only:
        lifetime = overlay_module.DURATION - smoke_start
        timestamps = sorted(set([smoke_start + lifetime * index / 11 for index in range(12)]
                                + [smoke_start + .02 / smoke_speed, smoke_start + .04 / smoke_speed,
                                   smoke_start + .06 / smoke_speed,
                                   overlay_module.DURATION - .06 / smoke_speed]))
    thumb_width = 480
    thumb_height = round(thumb_width * overlay.height() / overlay.width())
    caption_height = 34
    columns = 4
    rows = math.ceil(len(timestamps) / columns)
    sheet = QImage(columns * thumb_width, rows * (thumb_height + caption_height),
                   QImage.Format_ARGB32_Premultiplied)
    sheet.fill(QColor("#11151c"))
    painter = QPainter(sheet)
    try:
        painter.setRenderHint(QPainter.SmoothPixmapTransform)
        painter.setPen(QColor("#e3e5eb"))
        painter.setFont(QFont("Menlo", 12))
        for index, elapsed in enumerate(timestamps):
            x = (index % columns) * thumb_width
            y = (index // columns) * (thumb_height + caption_height)
            frame = render_frame(overlay, elapsed, smoke_only)
            if smoke_only:
                for tile_y in range(0, thumb_height, 24):
                    for tile_x in range(0, thumb_width, 24):
                        color = "#dce0e5" if (tile_x // 24 + tile_y // 24) % 2 else "#b4bbc3"
                        painter.fillRect(QRectF(x + tile_x, y + tile_y,
                                               min(24, thumb_width - tile_x),
                                               min(24, thumb_height - tile_y)), QColor(color))
            painter.drawImage(QRectF(x, y, thumb_width, thumb_height), frame)
            pose = overlay_module.eating_pose(elapsed, overlay.eating.center,
                                             overlay.width(), overlay.height(),
                                             overlay.reduce_motion)
            caption = (f"{elapsed - smoke_start:.3f} s   |   smoke" if smoke_only
                       else f"{elapsed:.3f} s   |   frame {pose.sprite_frame + 1:02d}")
            painter.drawText(QRectF(x + 12, y + thumb_height, thumb_width - 24, caption_height),
                             Qt.AlignVCenter, caption)
    finally:
        painter.end()
    path = output / ("fox-smoke-timeline.png" if smoke_only else "fox-dash-timeline.png")
    if not sheet.save(str(path)):
        raise RuntimeError(f"Cannot save contact sheet: {path}")
    return path


def save_apng(frame_paths, output, duration, smoke_only=False):
    try:
        from PIL import Image
    except ImportError:
        return None
    images = []
    try:
        for path in frame_paths:
            with Image.open(path) as frame:
                images.append(frame.convert("RGBA"))
        total_ms = round(duration * 1000)
        # Distribute whole milliseconds to preserve the full animation duration.
        frame_ms = [round((i + 1) * total_ms / len(images)) - round(i * total_ms / len(images))
                    for i in range(len(images))]
        path = output / ("fox-smoke-preview.png" if smoke_only else "fox-dash-preview.png")
        images[0].save(path, format="PNG", save_all=True, append_images=images[1:],
                       duration=frame_ms, loop=0, disposal=0, blend=0, optimize=False)
        return path
    finally:
        for image in images:
            image.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=600)
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--reduced-motion", action="store_true")
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--no-apng", action="store_true")
    args = parser.parse_args()
    if args.width < 640 or args.height < 400 or not 1 <= args.fps <= 120:
        parser.error("Use a size of at least 640×400 and a frame rate between 1 and 120.")
    application = QApplication.instance() or QApplication([])
    frames = overlay_module._animation_sprites()
    if frames is None:
        parser.error(f"Animation sprites unavailable: {overlay_module.ANIMATION_ASSET_PATH}")
    overlay = overlay_module.ScreenOverlay(application.primaryScreen())
    overlay.resize(args.width, args.height)
    overlay.reduce_motion = args.reduced_motion
    target = ((args.width * .50, args.height * .72) if args.smoke_only
              else (args.width * .62, args.height * .54))
    overlay.eating = DesktopTarget("animation-preview", "練習檔案.txt", target)
    overlay.started = 0
    output = (args.output or ROOT / "output" / ("fox-smoke-preview" if args.smoke_only else "fox-dash-preview"))
    output = output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    start = overlay_module.SMOKE_START if args.smoke_only else 0
    duration = overlay_module.DURATION - start
    count = max(2, math.ceil(duration * args.fps))
    timestamps = [start + index * duration / (count - 1) for index in range(count)]
    frame_paths = []
    try:
        for index, elapsed in enumerate(timestamps):
            image = render_frame(overlay, elapsed, args.smoke_only)
            path = output / f"frame-{index:03d}.png"
            if not image.save(str(path)):
                raise RuntimeError(f"Cannot save animation frame: {path}")
            frame_paths.append(path)
        contact_sheet = save_contact_sheet(overlay, output, args.smoke_only)
        apng = None if args.no_apng else save_apng(frame_paths, output, duration, args.smoke_only)
        (output / "manifest.json").write_text(json.dumps({
            "renderer": ("kon.overlay._draw_smoke" if args.smoke_only
                         else "kon.overlay.ScreenOverlay._paint_eating"),
            "size": [args.width, args.height], "duration_seconds": duration,
            "frame_count": count, "requested_fps": args.fps,
            "timestamps_seconds": timestamps, "sprite_count": len(frames),
            "reduced_motion": args.reduced_motion,
            "smoke_only": args.smoke_only,
            "smoke_sprite_count": len(overlay_module._smoke_sprites() or ()),
            "smoke_speed": overlay_module.SMOKE_SPEED,
            "animation_scale": overlay_module.ANIMATION_SCALE,
            "direction": "right-to-left",
        }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"Rendered {len(frame_paths)} Qt frames to {output}")
        print(f"Contact sheet: {contact_sheet}")
        if apng:
            print(f"Animated PNG: {apng}")
        elif not args.no_apng:
            print("Animated PNG skipped: Pillow is unavailable.")
    finally:
        overlay.cancel()
        overlay.close()


if __name__ == "__main__":
    main()
