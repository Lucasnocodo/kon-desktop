"""Deterministic fox-head dash/cancellation checks; no files are trashed."""

import hashlib
import json
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6.QtGui import QColor, QFont, QFontMetrics, QIcon, QImage, QKeyEvent, QPainter, QPixmap, QRegion
from PySide6.QtCore import QEvent, QPoint, QPointF, QRectF, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from kon.desktop import DesktopTarget, FileSnapshot
from kon.grid_selection import GridSelector
import kon.overlay as overlay_module
from kon.overlay import (
    ANIMATION_SCALE, BITE_TIME, DASH_END, DURATION, REDUCED_BITE_TIME, REDUCED_DURATION,
    REFERENCE_DASH_DURATION, REFERENCE_SMOKE_DURATION, SMOKE_HEAD_GONE,
    SMOKE_SPEED, SMOKE_START, SWALLOW_END,
    ScreenOverlay, eating_pose,
)


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def stage(app):
    overlay = ScreenOverlay(app.primaryScreen())
    overlay.reduce_motion = False
    overlay.resize(1440, 900)
    yield overlay
    overlay.cancel()
    overlay.close()


def test_short_dash_approaches_target_monotonically_and_settles_before_swallow():
    entry = [eating_pose(DASH_END * fraction, (800, 450), 1440, 900)
             for fraction in (0, .2, .4, .6, .8, 1)]
    assert entry[0].opacity == 0
    assert all(a.head_rect.left() < b.head_rect.left() for a, b in zip(entry, entry[1:]))
    travel = entry[-1].head_rect.left() - entry[0].head_rect.left()
    size = entry[-1].head_rect.width()
    previous_landing = 800 - 74 - size * .66
    previous_full_run = previous_landing + size + 24
    assert travel == pytest.approx(previous_full_run / 3)
    assert entry[-1].head_rect.left() - entry[-2].head_rect.left() > travel * .1
    assert all(p.head_rect.top() == entry[0].head_rect.top() for p in entry)
    assert any(p.opacity > 0 for p in entry[1:-1])
    assert all(p.file_progress == 0 and not p.swallow_done for p in entry)
    assert all(p.file_position.x() == 800 and p.file_position.y() == 450 for p in entry)
    settled = [eating_pose(t, (800, 450), 1440, 900)
               for t in (DASH_END, DASH_END + .02, DASH_END + .05,
                         BITE_TIME, BITE_TIME + .14, DURATION - .1)]
    assert all(p.head_rect == settled[0].head_rect for p in settled)
    assert settled[1].file_progress > 0
    assert settled[2].opacity == 1


@pytest.mark.parametrize("width,height,target", [
    (1440, 900, (800, 450)),
    (960, 600, (595.2, 324)),
    (1920, 1080, (1900, 580)),
    (640, 400, (20, 240)),
])
def test_shortened_dash_preserves_original_movement_speed(width, height, target):
    start = eating_pose(0, target, width, height)
    landing = eating_pose(DASH_END, target, width, height)
    size = landing.head_rect.width()
    previous_landing = min(max(target[0] - 74 - size * .66, 14), width - size - 14)
    original_distance = previous_landing + size + 24
    short_distance = landing.head_rect.left() - start.head_rect.left()
    assert REFERENCE_DASH_DURATION == pytest.approx(.30)
    assert short_distance == pytest.approx(original_distance / 3)
    assert short_distance / DASH_END == pytest.approx(original_distance / REFERENCE_DASH_DURATION)
    early = eating_pose(DASH_END * .2, target, width, height)
    late = eating_pose(DASH_END * .8, target, width, height)
    sampled_speed = (late.head_rect.left() - early.head_rect.left()) / (DASH_END * .6)
    assert sampled_speed == pytest.approx(original_distance / REFERENCE_DASH_DURATION)


@pytest.mark.parametrize("reduce_motion", [False, True])
@pytest.mark.parametrize("elapsed", [0, DASH_END / 2, DASH_END, DASH_END + .025,
                                    SWALLOW_END, BITE_TIME, SMOKE_START + .12, DURATION])
def test_visual_scale_doubles_geometry_about_the_correct_mouth_anchor(elapsed, reduce_motion):
    base = eating_pose(elapsed, (800, 450), 1440, 900, reduce_motion)
    enlarged = overlay_module.visual_pose(base, reduce_motion)
    assert ANIMATION_SCALE == 2
    assert enlarged.head_rect.width() == pytest.approx(base.head_rect.width() * 2)
    assert enlarged.head_rect.height() == pytest.approx(base.head_rect.height() * 2)
    mouth = 1 if reduce_motion else base.mouth
    anchor = overlay_module.mouth_position(base.head_rect, mouth)
    visual_anchor = overlay_module.mouth_position(enlarged.head_rect, mouth)
    assert (visual_anchor.x(), visual_anchor.y()) == pytest.approx((anchor.x(), anchor.y()))
    for name in base.__dataclass_fields__:
        if name != "head_rect":
            assert getattr(enlarged, name) == getattr(base, name)


def test_visual_zoom_preserves_actual_dash_distance_and_speed():
    times = [DASH_END * fraction for fraction in (0, .2, .6, 1)]
    logical = [eating_pose(elapsed, (800, 450), 1440, 900) for elapsed in times]
    visual = [overlay_module.visual_pose(pose) for pose in logical]
    for index in range(1, len(times)):
        dt = times[index] - times[index - 1]
        base_speed = (logical[index].head_rect.left() - logical[index - 1].head_rect.left()) / dt
        zoomed_speed = (visual[index].head_rect.left() - visual[index - 1].head_rect.left()) / dt
        assert zoomed_speed == pytest.approx(base_speed)


def test_enlarged_impact_liquid_origin_and_smoke_extent_scale_together():
    base = eating_pose(BITE_TIME, (800, 450), 1440, 900)
    enlarged = overlay_module.visual_pose(base)
    anchor = overlay_module.mouth_position(base.head_rect, base.mouth)
    origins = []
    for pose in (base, enlarged):
        rect = pose.head_rect
        origins.append((rect.left() + rect.width() * overlay_module.SPRAY_MOUTH_UV[0],
                        rect.top() + rect.height() * overlay_module.SPRAY_MOUTH_UV[1]))
    assert origins[1][0] - anchor.x() == pytest.approx((origins[0][0] - anchor.x()) * 2)
    assert origins[1][1] - anchor.y() == pytest.approx((origins[0][1] - anchor.y()) * 2)
    assert overlay_module.smoke_rect(enlarged.head_rect).width() == pytest.approx(
        overlay_module.smoke_rect(base.head_rect).width() * 2)


def test_reduced_visual_scale_keeps_the_head_and_smoke_stationary():
    poses = [overlay_module.visual_pose(eating_pose(elapsed, (800, 450), 1440, 900, True), True)
             for elapsed in (DASH_END / 2, DASH_END, SWALLOW_END, BITE_TIME, SMOKE_START + .2)]
    assert all(pose.head_rect == poses[0].head_rect for pose in poses)
    assert all(overlay_module.smoke_rect(pose.head_rect) == overlay_module.smoke_rect(poses[0].head_rect)
               for pose in poses)


@pytest.mark.parametrize("reduce_motion", [False, True])
def test_native_painter_sends_one_enlarged_pose_to_all_effects_and_keeps_icon_native(stage, monkeypatch,
                                                                                  reduce_motion):
    stage.reduce_motion = reduce_motion
    elapsed = DASH_END + .05
    monkeypatch.setattr(overlay_module.time, "monotonic", lambda: elapsed)
    target = DesktopTarget("size-preview", "demo.txt", (800, 450))
    stage.eating, stage.started = target, 0
    mirrored_target = (stage.width() - target.center[0], target.center[1])
    base = eating_pose(elapsed, mirrored_target, stage.width(), stage.height(), reduce_motion)
    expected = overlay_module.visual_pose(base, reduce_motion)
    received = []
    def effect(painter, pose, *args, **kwargs):
        received.append(pose.head_rect)
    for name in ("_draw_dash", "_draw_smoke", "_draw_bite_impact", "_draw_liquid"):
        monkeypatch.setattr(overlay_module, name, effect)
    monkeypatch.setattr(overlay_module, "draw_fox",
                        lambda painter, rect, *args, **kwargs: received.append(rect))
    icons = []
    monkeypatch.setattr(stage, "_paint_file", lambda painter, center, scale:
                        icons.append((center, scale, painter.worldTransform().determinant())))
    image = QImage(stage.size(), QImage.Format_RGBA8888)
    image.fill(Qt.transparent)
    painter = QPainter(image)
    try:
        stage._paint_eating(painter)
    finally:
        painter.end()
    assert len(received) == 6 and all(rect == expected.head_rect for rect in received)
    screen_position = QPointF(stage.width() - base.file_position.x(), base.file_position.y())
    assert icons == [(screen_position, base.file_scale, 1.)]


@pytest.mark.parametrize("reduce_motion", [False, True])
@pytest.mark.parametrize("target", [(400, 450), (800, 450), (1100, 450)])
def test_native_fox_approaches_from_the_right_and_places_its_mouth_on_the_target(stage, monkeypatch,
                                                                             reduce_motion, target):
    stage.reduce_motion = reduce_motion
    stage.eating, stage.started = DesktopTarget("direction", "demo.txt", target), 0.
    clock = [0.]
    monkeypatch.setattr(overlay_module.time, "monotonic", lambda: clock[0])
    for name in ("_draw_dash", "_draw_smoke", "_draw_bite_impact", "_draw_liquid"):
        monkeypatch.setattr(overlay_module, name, lambda *args, **kwargs: None)
    monkeypatch.setattr(stage, "_paint_file", lambda *args: None)
    heads = []
    def head(painter, rect, mouth, **kwargs):
        transform = painter.worldTransform()
        heads.append((transform.mapRect(rect),
                      transform.map(overlay_module.mouth_position(rect, mouth)),
                      transform.determinant()))
    monkeypatch.setattr(overlay_module, "draw_fox", head)
    image = QImage(stage.size(), QImage.Format_RGBA8888)
    painter = QPainter(image)
    try:
        for fraction in (.15, .45, .75, 1.):
            clock[0] = DASH_END * fraction
            stage._paint_eating(painter)
    finally:
        painter.end()
    assert all(determinant < 0 for _, _, determinant in heads)
    if reduce_motion:
        assert all(rect == heads[0][0] for rect, _, _ in heads)
    else:
        assert all(before[0].left() > after[0].left() for before, after in zip(heads, heads[1:]))
    mouth = heads[-1][1]
    assert (mouth.x(), mouth.y()) == pytest.approx(target)


@pytest.mark.parametrize("source", ["animation", "head_sheet", "procedural"])
def test_native_fox_art_is_mirrored_to_face_left_for_every_asset_fallback(stage, monkeypatch, source):
    elapsed = DASH_END
    stage.eating, stage.started = DesktopTarget("facing", "demo.txt", (800, 450)), 0.
    monkeypatch.setattr(overlay_module.time, "monotonic", lambda: elapsed)
    if source != "animation":
        monkeypatch.setattr(overlay_module, "_animation_sprites", lambda: None)
    if source == "procedural":
        monkeypatch.setattr(overlay_module, "_head_sprites", lambda: None)
    for name in ("_draw_dash", "_draw_smoke", "_draw_bite_impact", "_draw_liquid"):
        monkeypatch.setattr(overlay_module, name, lambda *args, **kwargs: None)
    monkeypatch.setattr(stage, "_paint_file", lambda *args: None)
    actual = QImage(stage.size(), QImage.Format_RGBA8888)
    actual.fill(Qt.transparent)
    painter = QPainter(actual)
    try:
        stage._paint_eating(painter)
    finally:
        painter.end()
    canonical = QImage(stage.size(), QImage.Format_RGBA8888)
    canonical.fill(Qt.transparent)
    pose = stage._eating_pose(elapsed)
    painter = QPainter(canonical)
    try:
        painter.setOpacity(pose.opacity)
        overlay_module.draw_fox(painter, pose.head_rect, pose.mouth,
                                sprite_frame=pose.sprite_frame, facing_right=True)
    finally:
        painter.end()
    actual_pixels = np.frombuffer(actual.constBits(), dtype=np.uint8).reshape(stage.height(), stage.width(), 4)
    canonical_pixels = np.frombuffer(canonical.constBits(), dtype=np.uint8).reshape(stage.height(), stage.width(), 4)
    assert actual_pixels[:, :, 3].any()
    # Qt can round reflected raster samples by one alpha level. The complete
    # asymmetric silhouette still has to face the opposite way at every pixel.
    assert np.allclose(actual_pixels[:, :, 3], canonical_pixels[:, ::-1, 3], rtol=0, atol=1)


@pytest.mark.parametrize("reduce_motion", [False, True])
def test_head_appears_with_jaw_already_wide_open_and_closes_at_bite(reduce_motion):
    appearing = [eating_pose(t, (800, 450), 1440, 900, reduce_motion)
                 for t in (DASH_END * .03, DASH_END * .2, DASH_END * .6,
                           DASH_END * .8, DASH_END * .99)]
    assert all(p.mouth == 1 for p in appearing)
    arrival = eating_pose(DASH_END, (800, 450), 1440, 900, reduce_motion)
    if not reduce_motion:
        assert arrival.sprite_frame > appearing[-1].sprite_frame
    closing = [eating_pose(DASH_END + offset, (800, 450), 1440, 900, reduce_motion)
               for offset in (.01, .04, .08)]
    assert 1 > closing[0].mouth > closing[1].mouth > closing[2].mouth > 0
    if not reduce_motion:
        assert all(p.sprite_frame > arrival.sprite_frame for p in closing[1:])
    swallowed = eating_pose(BITE_TIME, (800, 450), 1440, 900, reduce_motion)
    assert swallowed.mouth == 0 and swallowed.swallow_done
    assert swallowed.sprite_frame > arrival.sprite_frame


def test_bite_impact_peaks_on_contact_then_settles_without_screen_translation():
    before = eating_pose(DASH_END, (800, 450), 1440, 900)
    approaching = eating_pose(BITE_TIME - .02, (800, 450), 1440, 900)
    contact = eating_pose(BITE_TIME, (800, 450), 1440, 900)
    recoiling = eating_pose(BITE_TIME + .06, (800, 450), 1440, 900)
    settled = eating_pose(BITE_TIME + .14, (800, 450), 1440, 900)
    assert before.impact == settled.impact == 0
    assert contact.impact > .9
    assert contact.impact > approaching.impact
    assert contact.impact > recoiling.impact > 0
    assert all(p.head_rect == before.head_rect for p in (approaching, contact, recoiling, settled))


@pytest.mark.parametrize("target", [(1, 1), (1439, 1), (1, 899), (1439, 899), (800, 450)])
def test_landing_and_final_mouth_stay_inside_screen_at_edge_targets(target):
    landing = eating_pose(DASH_END, target, 1440, 900)
    swallowed = eating_pose(BITE_TIME, target, 1440, 900)
    assert landing.head_rect == swallowed.head_rect
    assert 0 <= landing.head_rect.left() < landing.head_rect.right() <= 1440
    assert 0 <= landing.head_rect.top() < landing.head_rect.bottom() <= 900
    anchor = overlay_module.mouth_position(swallowed.head_rect)
    assert swallowed.file_position.x() == pytest.approx(anchor.x())
    assert swallowed.file_position.y() == pytest.approx(anchor.y())
    assert 0 <= anchor.x() <= 1440 and 0 <= anchor.y() <= 900


@pytest.mark.parametrize("reduce_motion", [False, True])
@pytest.mark.parametrize("width,height,target", [
    (1440, 900, (800, 450)),
    (960, 600, (595.2, 324)),
])
def test_arrival_centers_the_open_mouth_on_the_file(width, height, target, reduce_motion):
    arrival = eating_pose(DASH_END, target, width, height, reduce_motion)
    center = overlay_module.mouth_position(arrival.head_rect, 1)
    assert center.x() == pytest.approx(target[0])
    assert center.y() == pytest.approx(target[1])
    assert arrival.file_position.x() == pytest.approx(center.x())
    assert arrival.file_position.y() == pytest.approx(center.y())


def test_swallow_finishes_before_bite_then_head_dissolves_in_smoke():
    before = eating_pose(DASH_END + .001, (800, 450), 1440, 900)
    before_signal = eating_pose(BITE_TIME - .001, (800, 450), 1440, 900)
    swallowed = eating_pose(BITE_TIME, (800, 450), 1440, 900)
    dissolving = eating_pose(_smoke_time(.35), (800, 450), 1440, 900)
    finished = eating_pose(DURATION, (800, 450), 1440, 900)
    assert before.mouth > .8 and before.file_opacity > .9
    assert not before.swallow_done
    assert before_signal.swallow_done and before_signal.file_opacity == 0
    assert swallowed.swallow_done and swallowed.file_opacity == 0
    assert dissolving.smoke > .8 and dissolving.smoke_opacity > .9
    assert dissolving.opacity < .1
    assert finished.opacity == pytest.approx(0, abs=1e-12)
    assert finished.smoke_opacity == pytest.approx(0, abs=1e-12)


def test_short_dash_bite_timing_stays_fixed_while_smoke_duration_is_compressed():
    assert DASH_END == pytest.approx(REFERENCE_DASH_DURATION / 3)
    assert BITE_TIME == REDUCED_BITE_TIME
    assert BITE_TIME == pytest.approx(DASH_END + .10)
    assert SWALLOW_END == pytest.approx(DASH_END + .08)
    assert DURATION == REDUCED_DURATION
    assert SMOKE_START == pytest.approx(BITE_TIME + .10)
    assert SMOKE_SPEED == pytest.approx(1.21)
    assert DURATION == pytest.approx(SMOKE_START + REFERENCE_SMOKE_DURATION / SMOKE_SPEED)
    assert SMOKE_HEAD_GONE == pytest.approx(SMOKE_START + .42 / SMOKE_SPEED)
    assert not eating_pose(SWALLOW_END - .001, (800, 450), 1440, 900).swallow_done
    assert eating_pose(SWALLOW_END + .001, (800, 450), 1440, 900).swallow_done
    assert eating_pose(SWALLOW_END + .001, (800, 450), 1440, 900).file_opacity == 0
    assert eating_pose(DURATION - .05, (800, 450), 1440, 900).smoke_opacity > 0


def test_reduced_motion_has_no_translation_zoom_or_swirl():
    poses = [eating_pose(t, (800, 450), 1440, 900, True)
             for t in (DASH_END * .3, DASH_END, DASH_END + .06,
                       BITE_TIME, BITE_TIME + .02, BITE_TIME + .14, .9, DURATION - .05)]
    assert all(p.head_rect == poses[0].head_rect for p in poses)
    assert all(p.file_position == poses[0].file_position for p in poses)
    assert all(p.smoke == 0 for p in poses)
    assert all(p.dash == 0 for p in poses)
    assert all(p.impact == 0 for p in poses)
    assert all(p.file_scale == p.scale == 1 for p in poses)
    assert eating_pose(REDUCED_BITE_TIME, (800, 450), 1440, 900, True).swallow_done


def test_icon_finishes_inside_the_actual_sprite_mouth():
    swallowed = eating_pose(BITE_TIME, (800, 450), 1440, 900)
    anchor = overlay_module.mouth_position(swallowed.head_rect)
    assert swallowed.file_position.x() == pytest.approx(anchor.x())
    assert swallowed.file_position.y() == pytest.approx(anchor.y())


def test_icon_follows_rising_jaw_and_reaches_its_current_mouth_before_bite():
    closing = [eating_pose(t, (800, 450), 1440, 900)
               for t in (DASH_END + .025, DASH_END + .05, SWALLOW_END - .001)]
    assert all(a.file_position.y() > b.file_position.y() for a, b in zip(closing, closing[1:]))
    swallowed = eating_pose(SWALLOW_END, (800, 450), 1440, 900)
    anchor = overlay_module.mouth_position(swallowed.head_rect, swallowed.mouth)
    assert swallowed.swallow_done and swallowed.file_opacity == 0
    assert swallowed.file_position.x() == pytest.approx(anchor.x())
    assert swallowed.file_position.y() == pytest.approx(anchor.y())


def _smoke_time(reference_age):
    return SMOKE_START + reference_age / SMOKE_SPEED


def _smoke_image(elapsed, reduce_motion=False):
    image = QImage(960, 600, QImage.Format_RGBA8888)
    image.fill(Qt.transparent)
    painter = QPainter(image)
    try:
        painter.setRenderHint(QPainter.Antialiasing)
        pose = eating_pose(elapsed, (595, 324), 960, 600, reduce_motion)
        for behind in (True, False):
            overlay_module._draw_smoke(painter, pose, elapsed, reduce_motion, behind=behind)
    finally:
        painter.end()
    return image


@pytest.mark.parametrize("elapsed", [SMOKE_START - .001, SMOKE_START, DURATION, DURATION + .01])
def test_smoke_starts_after_impact_and_completely_disappears(elapsed, app):
    image = _smoke_image(elapsed)
    assert not any(bytes(image.constBits())[3::4])
    pose = eating_pose(elapsed, (595, 324), 960, 600)
    assert not overlay_module.smoke_puffs(elapsed, pose.head_rect)


def test_procedural_smoke_fallback_expands_and_rises_while_reduced_motion_stays_fixed():
    rect = eating_pose(DASH_END, (595, 324), 960, 600).head_rect
    early = overlay_module.smoke_puffs(_smoke_time(.09), rect)
    later = overlay_module.smoke_puffs(_smoke_time(.50), rect)
    assert len(early) == len(later) and len(early) > 1
    assert early == overlay_module.smoke_puffs(_smoke_time(.09), rect)
    assert all(after.center.y() < before.center.y() and after.radius > before.radius
               for before, after in zip(early, later))
    assert all(0 < puff.opacity <= 1 for puff in early + later)
    reduced_early = overlay_module.smoke_puffs(_smoke_time(.09), rect, True)
    reduced_later = overlay_module.smoke_puffs(_smoke_time(.50), rect, True)
    assert len(reduced_early) == len(reduced_later)
    assert all(before.center == after.center and before.radius == after.radius
               and before.rotation == after.rotation == 0
               for before, after in zip(reduced_early, reduced_later))


def test_dense_smoke_covers_the_head_and_is_gentler_when_motion_is_reduced(app):
    elapsed = _smoke_time(.20)
    normal = _smoke_image(elapsed)
    reduced = _smoke_image(elapsed, True)
    center = eating_pose(elapsed, (595, 324), 960, 600).head_rect.center().toPoint()
    assert normal.pixelColor(center).alpha() > 180
    assert sum(bytes(normal.constBits())[3::4]) > sum(bytes(reduced.constBits())[3::4]) > 0


def test_procedural_fallback_locally_evaporates_the_head_beneath_dense_smoke():
    rect = eating_pose(DASH_END, (595, 324), 960, 600).head_rect
    points = [rect.topLeft() + overlay_module.QPointF(rect.width() * x / 20, rect.height() * y / 20)
              for x in range(1, 20) for y in range(1, 20)]
    intact = overlay_module._head_smoke_clip(rect, SMOKE_START)
    wrapping = overlay_module._head_smoke_clip(rect, _smoke_time(.10))
    middle = overlay_module._head_smoke_clip(rect, _smoke_time(.20))
    later = overlay_module._head_smoke_clip(rect, _smoke_time(.30))
    assert all(intact.contains(point) for point in points)
    assert all(wrapping.contains(point) for point in points)
    middle_count = sum(middle.contains(point) for point in points)
    later_count = sum(later.contains(point) for point in points)
    assert 0 < middle_count < len(points)
    assert later_count < middle_count
    dense_foreground = [puff for puff in overlay_module.smoke_puffs(_smoke_time(.20), rect)
                        if not puff.behind and puff.opacity >= .85]
    removed = [point for point in points if not middle.contains(point)]
    assert all(any((point.x() - puff.center.x()) ** 2 + (point.y() - puff.center.y()) ** 2
                   <= (puff.radius * .7) ** 2 for puff in dense_foreground) for point in removed)
    assert eating_pose(_smoke_time(.20), (595, 324), 960, 600).opacity == 1
    assert overlay_module._head_smoke_clip(rect, SMOKE_HEAD_GONE).isEmpty()


def test_supernatural_smoke_sheet_has_twelve_registered_transparent_frames(app):
    frames = overlay_module._smoke_sprites()
    assert frames is not None and len(frames) == 12
    assert overlay_module._smoke_sprites() is frames
    hashes = set()
    for pixmap in frames:
        assert pixmap.width() == pixmap.height() == 512 and pixmap.hasAlphaChannel()
        rgba = pixmap.toImage().convertToFormat(QImage.Format_RGBA8888)
        pixels = bytes(rgba.constBits())
        hashes.add(hashlib.sha256(pixels).hexdigest())
        alpha = pixels[3::4]
        assert 0 in alpha and max(alpha) > 16
        assert max(alpha[:512] + alpha[-512:] + alpha[::512] + alpha[511::512]) <= 8
    assert len(hashes) == 12
    manifest = json.loads(overlay_module.SMOKE_ASSET_PATH.with_name("manifest.json").read_text())
    assert manifest["preserved_registration"] and manifest["pivot_uv"] == [.5, .62]
    assert manifest["source_generation"]["single_pass"]
    cell = manifest["source_cell_size"][0]
    assert manifest["source_cell_size"] == [cell, cell]
    assert manifest["shared_scale"] == pytest.approx(512 / cell)
    for index, record in enumerate(manifest["frame_records"]):
        x, y = index % 4 * cell, index // 4 * cell
        assert record["source_cell"] == [x, y, x + cell, y + cell]


def test_generated_smoke_advances_smoothly_and_reduced_motion_holds_one_pose(app):
    span = DURATION - SMOKE_START
    positions = [overlay_module.smoke_frame(SMOKE_START + span * fraction)
                 for fraction in (0, .2, .4, .6, .8, 1)]
    assert positions == pytest.approx([0, 2.2, 4.4, 6.6, 8.8, 11])
    first = _smoke_image(_smoke_time(.25))
    second = _smoke_image(_smoke_time(.45))
    assert bytes(first.constBits()) != bytes(second.constBits())
    reduced_first = _smoke_image(_smoke_time(.25), True)
    reduced_second = _smoke_image(_smoke_time(.45), True)
    assert bytes(reduced_first.constBits()) == bytes(reduced_second.constBits())


def test_smoke_frame_interpolation_preserves_opacity_without_density_dips(app):
    span = DURATION - SMOKE_START
    images = [_smoke_image(SMOKE_START + span * position / 11) for position in (3, 3.5, 4)]
    alpha_totals = [sum(bytes(image.constBits())[3::4]) for image in images]
    assert alpha_totals[1] >= (alpha_totals[0] + alpha_totals[2]) * .47


@pytest.mark.parametrize("malformed", [False, True])
def test_missing_or_malformed_smoke_sheet_falls_back_to_visible_smoke(app, monkeypatch, tmp_path, malformed):
    path = tmp_path / "smoke-sheet.png"
    if malformed:
        image = QImage(15, 16, QImage.Format_RGBA8888)
        image.fill(Qt.transparent)
        assert image.save(str(path))
    monkeypatch.setattr(overlay_module, "SMOKE_ASSET_PATH", path)
    assert overlay_module._smoke_sprites() is None
    fallback = _smoke_image(_smoke_time(.20))
    assert any(bytes(fallback.constBits())[3::4])


@pytest.mark.parametrize("reference_age", [.02, .12, .24, .47, .65, .81, REFERENCE_SMOKE_DURATION])
def test_faster_smoke_matches_the_original_effect_at_equivalent_phases(app, monkeypatch, reference_age):
    elapsed = _smoke_time(reference_age)
    reference_elapsed = SMOKE_START + reference_age
    assert overlay_module._smoke_clock(elapsed) == pytest.approx(reference_elapsed)
    fast_pose = eating_pose(elapsed, (595, 324), 960, 600)
    fast_image = _smoke_image(elapsed)
    fast_puffs = overlay_module.smoke_puffs(elapsed, fast_pose.head_rect)
    fast_position = overlay_module.smoke_frame(elapsed)
    with monkeypatch.context() as reference:
        reference.setattr(overlay_module, "SMOKE_SPEED", 1.0)
        reference.setattr(overlay_module, "DURATION", SMOKE_START + REFERENCE_SMOKE_DURATION)
        reference.setattr(overlay_module, "SMOKE_HEAD_GONE", SMOKE_START + .42)
        original_pose = eating_pose(reference_elapsed, (595, 324), 960, 600)
        original_image = _smoke_image(reference_elapsed)
        original_puffs = overlay_module.smoke_puffs(reference_elapsed, original_pose.head_rect)
        assert fast_position == pytest.approx(overlay_module.smoke_frame(reference_elapsed))
    assert (fast_pose.opacity, fast_pose.smoke, fast_pose.smoke_opacity) == pytest.approx((
        original_pose.opacity, original_pose.smoke, original_pose.smoke_opacity,
    ))
    assert len(fast_puffs) == len(original_puffs)
    for fast, original in zip(fast_puffs, original_puffs):
        assert (fast.center.x(), fast.center.y(), fast.radius, fast.opacity, fast.rotation) == pytest.approx((
            original.center.x(), original.center.y(), original.radius, original.opacity, original.rotation,
        ))
    faster_pixels, original_pixels = bytes(fast_image.constBits()), bytes(original_image.constBits())
    if faster_pixels != original_pixels:
        assert max(abs(a - b) for a, b in zip(faster_pixels, original_pixels)) <= 1


def test_smoke_speed_leaves_pre_onset_motion_bite_and_liquid_unchanged(app, monkeypatch):
    times = (0, DASH_END / 2, DASH_END, SWALLOW_END, BITE_TIME, SMOKE_START)
    current = [eating_pose(elapsed, (595, 324), 960, 600) for elapsed in times]
    assert all(overlay_module._smoke_clock(elapsed) == elapsed for elapsed in times)
    liquid_times = (BITE_TIME + .02, BITE_TIME + .10, BITE_TIME + .30, BITE_TIME + .50)
    liquid = [_liquid_pixels(elapsed) for elapsed in liquid_times]
    with monkeypatch.context() as reference:
        reference.setattr(overlay_module, "SMOKE_SPEED", 1.0)
        reference.setattr(overlay_module, "DURATION", SMOKE_START + REFERENCE_SMOKE_DURATION)
        reference.setattr(overlay_module, "SMOKE_HEAD_GONE", SMOKE_START + .42)
        assert current == [eating_pose(elapsed, (595, 324), 960, 600) for elapsed in times]
        assert liquid == [_liquid_pixels(elapsed) for elapsed in liquid_times]


def test_reference_sheet_splits_into_two_equal_square_cells(app):
    pair = overlay_module._head_sprites()
    assert pair is not None
    assert all(p.width() == p.height() == 887 for p in pair)
    assert all(p.hasAlphaChannel() for p in pair)


def test_animation_uses_twelve_distinct_transparent_sprite_frames(app):
    frames = overlay_module._animation_sprites()
    assert frames is not None and len(frames) == 12
    hashes = set()
    for pixmap in frames:
        assert pixmap.width() == pixmap.height() == 512
        assert pixmap.hasAlphaChannel()
        rgba = pixmap.toImage().convertToFormat(QImage.Format_RGBA8888)
        pixels = bytes(rgba.constBits())
        hashes.add(hashlib.sha256(pixels).hexdigest())
        alpha = pixels[3::4]
        assert 0 in alpha and max(alpha) == 255
    assert len(hashes) == len(frames)


def test_timeline_visits_all_sprite_frames_before_bite():
    poses = [eating_pose(index / 1000, (800, 450), 1440, 900)
             for index in range(round(BITE_TIME * 1000) + 1)]
    indices = [p.sprite_frame for p in poses]
    assert set(indices) == set(range(12))
    assert indices == sorted(indices)
    assert any(p.sprite_frame != poses[0].sprite_frame for p in poses if p.head_rect.right() < 800)
    assert indices[-1] == 11


def _liquid_pixels(elapsed, reduce_motion=False):
    image = QImage(960, 600, QImage.Format_RGBA8888)
    image.fill(Qt.transparent)
    painter = QPainter(image)
    try:
        painter.setRenderHint(QPainter.Antialiasing)
        pose = eating_pose(elapsed, (595, 324), 960, 600, reduce_motion)
        overlay_module._draw_liquid(painter, pose, elapsed, reduce_motion)
    finally:
        painter.end()
    return bytes(image.constBits())


@pytest.mark.parametrize("elapsed", [BITE_TIME - .01, BITE_TIME, BITE_TIME + .67, DURATION])
def test_liquid_is_transparent_before_bite_and_after_spray(elapsed, app):
    assert not any(_liquid_pixels(elapsed)[3::4])


def test_liquid_spray_is_red_deterministic_and_gentler_when_motion_is_reduced(app):
    first, later = BITE_TIME + .10, BITE_TIME + .19
    normal = _liquid_pixels(first)
    reduced = _liquid_pixels(first, True)
    assert normal == _liquid_pixels(first)
    assert normal != _liquid_pixels(later)
    assert reduced == _liquid_pixels(later, True)
    normal_alpha, reduced_alpha = normal[3::4], reduced[3::4]
    assert sum(normal_alpha) > sum(reduced_alpha) > 0
    opaque = [offset for offset in range(0, len(normal), 4) if normal[offset + 3] > 100]
    assert opaque
    assert all(normal[offset] > normal[offset + 1] * 2
               and normal[offset] > normal[offset + 2] * 2 for offset in opaque)


def test_connected_liquid_jet_starts_at_closed_lip_above_swallow_anchor(app):
    elapsed = BITE_TIME + .02
    pose = eating_pose(elapsed, (595, 324), 960, 600)
    pixels = _liquid_pixels(elapsed)
    lip_x = round(pose.head_rect.left() + pose.head_rect.width() * .78)
    lip_y = round(pose.head_rect.top() + pose.head_rect.height() * .61)
    swallow = overlay_module.mouth_position(pose.head_rect, 1)
    assert lip_y < swallow.y()
    assert any(pixels[(y * 960 + x) * 4 + 3] > 100
               for y in range(lip_y - 2, lip_y + 3)
               for x in range(lip_x - 2, lip_x + 3))
    assert pixels[(round(swallow.y()) * 960 + round(swallow.x())) * 4 + 3] == 0


@pytest.mark.parametrize("reduce_motion,bite_time,duration", [
    (False, BITE_TIME, DURATION), (True, REDUCED_BITE_TIME, REDUCED_DURATION),
])
def test_bite_and_completion_emit_once_in_order(stage, monkeypatch, reduce_motion, bite_time, duration):
    clock = [100.0]
    monkeypatch.setattr(overlay_module.time, "monotonic", lambda: clock[0])
    target = DesktopTarget("demo", "demo.txt", (800, 450))
    events = []
    stage.reduce_motion = reduce_motion
    stage.bite.connect(lambda value: events.append(("bite", value)))
    stage.animation_done.connect(lambda: events.append(("done", None)))
    stage.play(target)
    stage.timer.stop()
    clock[0] = 100 + bite_time - .001
    stage._tick()
    assert events == []
    clock[0] = 100 + bite_time + .001
    stage._tick()
    stage._tick()
    assert events == [("bite", target)]
    clock[0] = 100 + duration + .001
    stage._tick()
    stage._tick()
    assert events == [("bite", target), ("done", None)]
    assert stage.eating is None and not stage.timer.isActive()


def test_prepare_animation_decodes_all_static_effects_before_starting_a_clock(stage, monkeypatch):
    decoded = []
    monkeypatch.setattr(overlay_module, "_animation_sprites", lambda: decoded.append("fox") or ())
    monkeypatch.setattr(overlay_module, "_smoke_sprites", lambda: decoded.append("smoke") or ())
    monkeypatch.setattr(stage.summon_sound, "play", lambda: pytest.fail("Preparing art must stay silent"))
    stage.prepare_animation()
    assert decoded == ["fox", "smoke"]
    assert stage.eating is None and not stage.timer.isActive()


def test_first_play_starts_after_cold_sprite_decode_and_native_audio_initialization(stage, monkeypatch):
    clock = [100.]
    monkeypatch.setattr(overlay_module.time, "monotonic", lambda: clock[0])
    prepared = []
    def cold_load(name):
        prepared.append(name)
        clock[0] += .6
        return ()
    monkeypatch.setattr(overlay_module, "_animation_sprites", lambda: cold_load("fox"))
    monkeypatch.setattr(overlay_module, "_smoke_sprites", lambda: cold_load("smoke"))
    def first_audio_start():
        prepared.append("sound")
        clock[0] += 1.5
    monkeypatch.setattr(stage.summon_sound, "play", first_audio_start)
    target = DesktopTarget("cold-load", "demo.txt", (600., 420.))
    bitten = []
    stage.bite.connect(bitten.append)
    stage.play(target)
    stage.timer.stop()
    assert prepared == ["fox", "smoke", "sound"]
    assert stage.started == pytest.approx(clock[0])
    stage._tick()
    assert not stage.bitten and bitten == []
    assert stage.eating is target
    clock[0] += BITE_TIME + .001
    stage._tick()
    assert bitten == [target]


def test_cancel_during_native_playback_initialization_cannot_restart_an_old_animation(stage, monkeypatch):
    monkeypatch.setattr(stage.summon_sound, "play", lambda: stage.cancel())
    stage.play(DesktopTarget("cancel-before-start", "demo.txt", (800, 450)))
    assert stage.eating is None and not stage.timer.isActive()
    assert not stage.isVisible()


def test_cancel_removes_timers_and_prevents_late_bite(stage, monkeypatch):
    clock = [100.0]
    monkeypatch.setattr(overlay_module.time, "monotonic", lambda: clock[0])
    events = []
    stage.bite.connect(lambda target: events.append("bite"))
    stage.animation_done.connect(lambda: events.append("done"))
    stage.play(DesktopTarget("demo", "demo.txt", (800, 450)))
    stage.cancel()
    clock[0] += 100
    stage._tick()
    assert events == [] and stage.eating is None
    assert not stage.timer.isActive() and not stage.isVisible()


def test_calibration_cancel_then_play_restores_click_through(stage):
    stage.set_calibration((.5, .5), 3)
    assert stage.calibration_point is not None
    assert not stage.windowFlags() & Qt.WindowTransparentForInput
    stage.cancel()
    assert stage.calibration_point is None and not stage.isVisible()
    assert stage.windowFlags() & Qt.WindowTransparentForInput
    stage.play(DesktopTarget("demo", "demo.txt", (800, 450)))
    assert stage.windowFlags() & Qt.WindowTransparentForInput
    assert stage.calibration_point is None


def test_validation_overlay_uses_supplied_total_and_title(stage, app):
    stage.set_calibration((.5, .5), 2, total=7, title="驗證實際瞄準")
    app.processEvents()
    assert stage.calibration_total == 7
    assert stage._calibration_title.text() == "驗證實際瞄準 3 / 7"
    assert stage.windowFlags() & Qt.WindowType_Mask == Qt.Window
    stage.set_calibration((.5, .5), 8, total=9, title="驗證實際瞄準")
    assert stage._calibration_title.text() == "驗證實際瞄準 9 / 9"
    stage.cancel()
    stage.set_calibration((.5, .5), 0)
    assert stage._calibration_title.text() == "瞄準校正 1 / 9"


def test_synchronous_cancel_from_bite_handler_suppresses_completion(stage, monkeypatch):
    clock = [100.0]
    monkeypatch.setattr(overlay_module.time, "monotonic", lambda: clock[0])
    events = []
    def bite(target):
        events.append("bite")
        stage.cancel()
    stage.bite.connect(bite)
    stage.animation_done.connect(lambda: events.append("done"))
    stage.play(DesktopTarget("demo", "demo.txt", (800, 450)))
    stage.timer.stop()
    clock[0] += DURATION + 1
    stage._tick()
    assert events == ["bite"] and stage.eating is None


@pytest.mark.parametrize("ready", [False, True])
def test_calibration_space_return_and_button_are_single_actions(stage, app, ready):
    samples = []
    exits = []
    stage.sample_requested.connect(lambda: samples.append("sample"))
    stage.exit_requested.connect(lambda: exits.append("exit"))
    stage.set_calibration((.5, .5), 4)
    stage.set_calibration_state(6 if ready else 0, 6, "保持手孔瞄準圓點", ready)
    app.processEvents()
    assert stage.focusPolicy() == Qt.StrongFocus
    assert stage.windowFlags() & Qt.WindowType_Mask == Qt.Window
    assert stage.record_button.isEnabled()
    QTest.keyClick(stage, Qt.Key_Space)
    assert samples == ["sample"]
    QTest.mouseClick(stage.record_button, Qt.LeftButton)
    assert len(samples) == 2
    stage.record_button.setFocus()
    QTest.keyClick(stage.record_button, Qt.Key_Space)
    assert len(samples) == 3
    QTest.keyClick(stage.record_button, Qt.Key_Return)
    assert len(samples) == 4
    stage.cancel_calibration_button.setFocus()
    QTest.keyClick(stage.cancel_calibration_button, Qt.Key_Space)
    assert len(samples) == 5 and not exits
    QTest.keyClick(stage.cancel_calibration_button, Qt.Key_Escape)
    assert exits == ["exit"]
    QTest.mouseClick(stage.cancel_calibration_button, Qt.LeftButton)
    assert exits == ["exit", "exit"]


def test_calibration_autorepeat_cannot_emit_repeated_samples(stage, app):
    samples = []
    stage.sample_requested.connect(lambda: samples.append("sample"))
    stage.set_calibration((.5, .5), 0)
    app.processEvents()
    for receiver in (stage, stage.record_button, stage.cancel_calibration_button):
        repeated = QKeyEvent(QEvent.KeyPress, Qt.Key_Space, Qt.NoModifier, " ", True, 4)
        QApplication.sendEvent(receiver, repeated)
    assert samples == []


def test_calibration_feedback_is_independent_of_stream_status(stage):
    stage.set_calibration((.5, .5), 0)
    stage.calibration_feedback = "取樣失敗：手孔深度不穩，請保持位置。"
    stage.hint = "鏡頭更新中"
    stage.set_calibration_state(2, 6, "新的逐幀狀態", False)
    assert stage._calibration_feedback_label.text() == stage.calibration_feedback
    assert "新的逐幀狀態" in stage._calibration_state_label.text()
    assert stage.record_button.isEnabled()
    stage.calibration_feedback = ""
    assert "新的逐幀狀態" in stage._calibration_state_label.text()


def test_calibration_long_depth_error_and_timeout_feedback_fit_without_clipping(stage):
    reason = "手部投影對應不足，無法估計深度；請讓整隻手清楚入鏡"
    feedback = f"尚未記錄：{reason} 調整後再按空白鍵或點「記錄」。"
    stage.set_calibration((.82, .82), 8)
    stage.set_calibration_state(0, 6, reason, False)
    stage.calibration_feedback = feedback
    for label in (stage._calibration_state_label, stage._calibration_feedback_label):
        needed = label.fontMetrics().boundingRect(
            QRectF(0, 0, 276, 2000).toRect(), Qt.TextWordWrap, label.text()).height()
        assert label.height() >= needed
        assert label.textFormat() == Qt.PlainText
    assert stage.calibration_card.height() > 360
    assert stage.rect().contains(stage.calibration_card.geometry())
    assert stage._calibration_feedback_label.text() == feedback
    assert stage.record_button.isEnabled()


def test_calibration_card_never_covers_any_of_the_nine_target_rings(stage, app):
    for x in (.18, .5, .82):
        for y in (.18, .5, .82):
            stage.set_calibration((x, y), 0)
            app.processEvents()
            target_bounds = QRectF(x * stage.width() - 30, y * stage.height() - 30, 60, 60)
            assert not QRectF(stage.calibration_card.geometry()).intersects(target_bounds)
            assert stage.rect().contains(stage.calibration_card.geometry())


def test_calibration_preview_owns_pixels_and_leaves_desktop_unmasked(stage, app):
    stage.set_calibration((.5, .5), 0)
    preview = QImage(320, 180, QImage.Format_RGBA8888)
    preview.fill(QColor("#ef5d35"))
    stage.set_calibration_preview(preview)
    preview.fill(QColor("#00ff00"))
    assert stage._calibration_preview.pixelColor(20, 20).red() > 200
    assert not stage._camera_preview_label.pixmap().isNull()
    app.processEvents()
    rendered = QImage(stage.size(), QImage.Format_ARGB32)
    rendered.fill(Qt.transparent)
    painter = QPainter(rendered)
    try:
        stage.render(painter, QPoint(0, 0))
    finally:
        painter.end()
    assert rendered.pixelColor(20, 20).alpha() == 0
    assert rendered.pixelColor(stage.width() // 2, stage.height() // 2).alpha() == 0
    card = stage.calibration_card.geometry()
    assert rendered.pixelColor(card.center()).alpha() > 200


def test_replacement_started_during_bite_is_not_finished_by_old_tick(stage, monkeypatch):
    clock = [100.0]
    monkeypatch.setattr(overlay_module.time, "monotonic", lambda: clock[0])
    replacement = DesktopTarget("replacement", "second.txt", (600, 400))
    events = []
    def bite(target):
        events.append(target.id)
        stage.cancel()
        stage.play(replacement)
    stage.bite.connect(bite)
    stage.animation_done.connect(lambda: events.append("done"))
    stage.play(DesktopTarget("first", "first.txt", (800, 450)))
    stage.timer.stop()
    clock[0] += DURATION + 1
    stage._tick()
    assert events == ["first"]
    assert stage.eating is replacement and stage.timer.isActive()
    assert not stage.bitten


def test_real_target_icon_is_requested_without_any_file_mutation(stage, monkeypatch, tmp_path):
    requested = []
    class Provider:
        def icon(self, info):
            requested.append(info.absoluteFilePath())
            pixmap = QPixmap(24, 24)
            pixmap.fill(QColor("#ff0000"))
            return QIcon(pixmap)
    monkeypatch.setattr(overlay_module, "QFileIconProvider", Provider)
    path = tmp_path / "keep.txt"
    path.write_text("still here", encoding="utf-8")
    info = path.stat()
    snapshot = FileSnapshot(path, info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns)
    stage.play(DesktopTarget(str(path), path.name, (800, 450), snapshot))
    assert requested == [str(path)] and not stage._file_icon.isNull()
    stage.cancel()
    assert path.read_text(encoding="utf-8") == "still here"


def render_file_only(stage, monkeypatch, elapsed):
    monkeypatch.setattr(overlay_module.time, "monotonic", lambda: stage.started + elapsed)
    for name in ("_draw_dash", "_draw_smoke", "_draw_bite_impact", "_draw_liquid", "draw_fox"):
        monkeypatch.setattr(overlay_module, name, lambda *args, **kwargs: None)
    image = QImage(stage.size(), QImage.Format_ARGB32)
    image.fill(Qt.transparent)
    painter = QPainter(image)
    try:
        stage._paint_eating(painter)
    finally:
        painter.end()
    return image


@pytest.mark.parametrize("reduce_motion", [False, True])
@pytest.mark.parametrize("elapsed", [0., DASH_END + .035])
def test_right_to_left_hunt_keeps_asymmetric_targets_unmirrored(stage, monkeypatch,
                                                              reduce_motion, elapsed):
    stage.reduce_motion = reduce_motion
    red, blue = QColor("#ed1028"), QColor("#163eea")
    source = QPixmap(64, 64)
    source.fill(Qt.transparent)
    painter = QPainter(source)
    painter.fillRect(0, 0, 32, 64, red)
    painter.fillRect(32, 0, 32, 64, blue)
    painter.end()
    stage.play(DesktopTarget("asymmetric-file", "demo.txt", (800., 450.)))
    stage._file_icon = QIcon(source)
    sample_offset = 12.
    stage.timer.stop()
    pose = stage._eating_pose(elapsed)
    center = QPointF(stage.width() - pose.file_position.x(), pose.file_position.y())
    image = render_file_only(stage, monkeypatch, elapsed)
    left = image.pixelColor((center - QPointF(sample_offset * pose.file_scale, 0)).toPoint())
    right = image.pixelColor((center + QPointF(sample_offset * pose.file_scale, 0)).toPoint())
    assert left.alpha() > 0 and right.alpha() > 0
    assert left.red() > left.blue() * 3
    assert right.blue() > right.red() * 3
    if elapsed == 0.:
        assert (center.x(), center.y()) == pytest.approx(stage.eating.center)


def test_locked_banner_elides_long_file_name_but_preserves_speech_and_release_cues(app):
    metrics = QFontMetrics(QFont("PingFang TC", 11))
    name = "很長的桌面企劃檔案名稱" * 15 + "_final.pdf"
    lines = ScreenOverlay._banner_lines(
        f"已鎖定 {name} · 請說 kon\n張開手或移出鏡頭可解除", metrics, 530)
    assert len(lines) == 2
    assert lines[0].startswith("已鎖定 ") and lines[0].endswith("_final.pdf · 請說 kon")
    assert "…" in lines[0]
    assert lines[1] == "張開手或移出鏡頭可解除"
    assert all(metrics.horizontalAdvance(line) <= 530 for line in lines)


def test_paused_banner_retains_file_identity_and_fits_the_wait_reason(app):
    metrics = QFontMetrics(QFont("PingFang TC", 11))
    lines = ScreenOverlay._banner_lines(
        "已鎖定 草稿.txt · 請說 kon\n追蹤暫停；" + "請等待新的手部畫面，鎖定暫時保留。" * 8, metrics, 530)
    assert lines[0] == "已鎖定 草稿.txt · 請說 kon"
    assert lines[1].startswith("追蹤暫停") and lines[1].endswith("…")
    assert all(metrics.horizontalAdvance(line) <= 530 for line in lines)


def test_two_line_lock_banner_leaves_desktop_transparent_outside_local_card(stage, app):
    stage.targets = []
    stage.hint = "已鎖定 草稿.txt · 請說 kon\n張開手或移出鏡頭可解除"
    stage.show()
    app.processEvents()
    rendered = QImage(stage.size(), QImage.Format_ARGB32)
    rendered.fill(Qt.transparent)
    stage.render(rendered)
    center = stage.width() // 2
    assert rendered.pixelColor(center - 250, 85).alpha() > 200
    assert rendered.pixelColor(center, 94).alpha() == 0
    assert rendered.pixelColor(20, 20).alpha() == 0
    assert rendered.pixelColor(center, stage.height() // 2).alpha() == 0


def test_single_line_banner_keeps_existing_compact_height(stage, app):
    stage.targets = []
    stage.hint = "等待瞄準目標"
    stage.show()
    app.processEvents()
    rendered = QImage(stage.size(), QImage.Format_ARGB32)
    rendered.fill(Qt.transparent)
    stage.render(rendered)
    assert rendered.pixelColor(stage.width() // 2, 68).alpha() > 200
    assert rendered.pixelColor(stage.width() // 2, 78).alpha() == 0


def render_selection(stage):
    image = QImage(stage.size(), QImage.Format_ARGB32)
    image.fill(Qt.transparent)
    stage.render(image)
    return image


def test_grid_default_is_off_and_does_not_change_existing_transparency(stage):
    assert not stage.grid_enabled
    assert (stage.grid_columns, stage.grid_rows) == (30, 12)
    assert stage.grid_active_cell is None and stage.grid_ambiguous_cell is None
    image = render_selection(stage)
    assert image.pixelColor(48, 123).alpha() == 0
    assert image.pixelColor(20, 150).alpha() == 0


@pytest.mark.parametrize("width,height", [(3440, 1440), (800, 800), (1440, 900)])
def test_grid_bounds_use_column_row_and_cover_the_last_cell_exactly(stage, width, height):
    stage.resize(width, height)
    rect = stage._grid_cell_rect((29, 11))
    assert rect.x() == pytest.approx(29 * width / 30)
    assert rect.y() == pytest.approx(11 * height / 12)
    assert rect.width() == pytest.approx(width / 30)
    assert rect.height() == pytest.approx(height / 12)
    assert rect.right() == pytest.approx(width)
    assert rect.bottom() == pytest.approx(height)


@pytest.mark.parametrize("practice", [True, False])
def test_grid_selection_leaves_boundaries_and_interiors_completely_transparent(stage, practice):
    stage.grid_enabled = True
    stage.practice = practice
    image = render_selection(stage)
    assert image.pixelColor(48, 123).alpha() == 0
    assert image.pixelColor(20, 150).alpha() == 0
    assert image.pixelColor(20, 123).alpha() == 0
    assert image.pixelColor(125, 310).alpha() == 0


def test_active_and_ambiguous_cells_have_local_distinct_highlights(stage):
    stage.grid_enabled = True
    stage.grid_active_cell = (8, 6)
    stage.grid_ambiguous_cell = (10, 6)
    image = render_selection(stage)
    active = stage._grid_cell_rect(stage.grid_active_cell)
    ambiguous = stage._grid_cell_rect(stage.grid_ambiguous_cell)
    normal = stage._grid_cell_rect((12, 6))
    warm, red = image.pixelColor(active.center().toPoint()), image.pixelColor(ambiguous.center().toPoint())
    assert 0 < warm.alpha() < 50 and warm.red() > warm.green() > warm.blue()
    assert 0 < red.alpha() < 50 and red.red() > red.green() * 2
    assert image.pixelColor(normal.center().toPoint()).alpha() == 0
    assert image.pixelColor(round(active.left() + 1), round(active.center().y())).alpha() > 100
    assert image.pixelColor(round(ambiguous.left() + 1), round(ambiguous.center().y())).alpha() > 100


def test_grid_does_not_replace_or_move_the_smoothed_cursor(stage):
    stage.point = (195, 319)
    without_grid = render_selection(stage)
    stage.grid_enabled = True
    stage.grid_active_cell = (8, 6)
    with_grid = render_selection(stage)
    assert stage.point == (195, 319)
    for x, y in ((178, 319), (195, 309), (212, 319), (195, 329)):
        assert with_grid.pixelColor(x, y) == without_grid.pixelColor(x, y)


def test_grid_hidden_during_calibration_and_animation(stage, monkeypatch, app):
    stage.grid_enabled = True
    stage.set_calibration((.5, .5), 0)
    app.processEvents()
    image = render_selection(stage)
    assert image.pixelColor(48, 123).alpha() == 0
    stage.cancel()
    stage.eating = DesktopTarget("preview", "虛擬文件.txt", (800, 450))
    monkeypatch.setattr(stage, "_paint_eating", lambda painter: None)
    image = render_selection(stage)
    assert image.pixelColor(48, 123).alpha() == 0


@pytest.mark.parametrize("cell", [None, (-1, 0), (30, 0), (0, 12), (True, 0), (1.5, 2), (1,), "invalid"])
def test_invalid_grid_cell_never_draws_outside_the_matrix(stage, cell):
    assert stage._grid_cell_rect(cell) is None


def test_invalid_grid_dimensions_are_ignored_without_division_error(stage):
    stage.grid_enabled = True
    stage.grid_columns = 0
    stage.grid_active_cell = (0, 0)
    assert stage._grid_cell_rect((0, 0)) is None
    image = render_selection(stage)
    assert image.pixelColor(48, 123).alpha() == 0


@pytest.mark.parametrize("width,height", [(3440, 1440), (1920, 1080), (800, 800)])
@pytest.mark.parametrize("cell", [(7, 6), (29, 11)])
def test_practice_grid_tile_centers_and_all_content_fit_its_cell(stage, width, height, cell):
    stage.resize(width, height)
    stage.grid_enabled = True
    bounds = stage._grid_cell_rect(cell)
    target = DesktopTarget("demo", "非常長的演練檔案名稱" * 8 + ".pdf",
                           (bounds.center().x(), bounds.center().y()))
    layout = stage._practice_grid_layout(target)
    assert layout.cell == bounds
    assert layout.card.center() == bounds.center()
    assert bounds.contains(layout.card)
    assert layout.card.contains(layout.icon) and layout.card.contains(layout.label)
    assert layout.icon.bottom() < layout.label.top()
    assert bounds.contains(layout.highlight.adjusted(-1, -1, 1, 1))
    assert bounds.contains(layout.progress.adjusted(-1.5, -1.5, 1.5, 1.5))
    assert layout.icon_scale > 0 and layout.font.pixelSize() >= 9
    assert layout.title and "…" in layout.title
    assert QFontMetrics(layout.font).horizontalAdvance(layout.title) <= layout.label.width()


@pytest.mark.parametrize("width,height", [(3440, 1440), (1920, 1080), (800, 800)])
@pytest.mark.parametrize("progress", [0., 1.])
def test_rendered_practice_card_and_lock_marks_never_bleed_outside_cell(stage, width, height, progress):
    stage.resize(width, height)
    stage.grid_enabled = True
    stage.grid_active_cell = cell = (7, 6)
    stage.progress = progress
    bounds = stage._grid_cell_rect(cell)
    before = render_selection(stage)
    target = DesktopTarget("demo", "小格中的長文件名稱_final.pdf",
                           (bounds.center().x(), bounds.center().y()))
    stage.targets = [target]
    stage.target_id = target.id
    after = render_selection(stage)
    changed = 0
    for y in range(int(bounds.top()) - 8, int(bounds.bottom()) + 9):
        for x in range(int(bounds.left()) - 8, int(bounds.right()) + 9):
            if before.pixel(x, y) != after.pixel(x, y):
                changed += 1
                assert bounds.contains(QPoint(x, y))
    assert changed > 20
    layout = stage._practice_grid_layout(target)
    for area in (layout.icon, layout.label):
        assert any(before.pixel(x, y) != after.pixel(x, y)
                   for y in range(int(area.top()) + 1, int(area.bottom()))
                   for x in range(int(area.left()) + 1, int(area.right())))


def test_practice_grid_cell_assignment_matches_selector_at_exact_boundaries(stage):
    stage.grid_enabled = True
    point = (8 * stage.width() / 30, 6 * stage.height() / 12)
    target = DesktopTarget("demo", "邊界.txt", point)
    selector = GridSelector()
    assert stage._practice_grid_layout(target).cell == stage._grid_cell_rect(
        selector.cell_for(point, (stage.width(), stage.height())))


def test_adaptive_virtual_tile_is_unused_for_real_desktop_or_grid_off(stage):
    target = DesktopTarget("demo", "草稿.txt", (stage.width() / 2, stage.height() / 2))
    assert stage._practice_grid_layout(target) is None
    stage.grid_enabled = True
    stage.practice = False
    assert stage._practice_grid_layout(target) is None
    stage.targets = [target]
    stage.target_id = target.id
    stage.progress = 1
    image = render_selection(stage)
    # A real icon's original 98px lock arc is retained independently of cell size.
    x, y = map(round, target.center)
    assert image.pixelColor(x - 49, y).alpha() > 200


def test_invalid_practice_grid_target_does_not_fall_back_to_oversized_card(stage):
    stage.grid_enabled = True
    before = render_selection(stage)
    stage.targets = [DesktopTarget("outside", "越界.txt", (-10, stage.height() / 2))]
    stage.target_id = "outside"
    assert stage._practice_grid_layout(stage.targets[0]) is None
    assert render_selection(stage) == before


def test_static_selection_and_layout_are_reused_when_only_dynamic_state_changes(stage, monkeypatch):
    stage.grid_enabled = True
    target = DesktopTarget("demo", "草稿.txt", (360, 480))
    stage.targets = [target]
    calls = []
    original_tile = stage._paint_practice_tile
    def tile(*args):
        calls.append("tile")
        original_tile(*args)
    monkeypatch.setattr(stage, "_paint_practice_tile", tile)
    render_selection(stage)
    layer_key = stage._static_selection_cache.cacheKey()
    layout = stage._practice_grid_layout(target)
    stage.target_id = target.id
    stage.grid_active_cell = (7, 6)
    stage.grid_ambiguous_cell = (9, 4)
    stage.progress = .8
    stage.point = (710, 414)
    stage.hint = "已鎖定 草稿.txt · 請說 kon\n張開手可解除"
    render_selection(stage)
    assert calls == ["tile"]
    assert stage._static_selection_cache.cacheKey() == layer_key
    assert stage._practice_grid_layout(target) is layout


@pytest.mark.parametrize("change", ["resize", "rename", "move", "remove", "columns", "rows",
                                   "grid", "practice", "dpr"])
def test_static_selection_invalidates_for_every_visible_scene_change(stage, monkeypatch, change):
    stage.grid_enabled = True
    target = DesktopTarget("demo", "草稿.txt", (360, 480))
    stage.targets = [target]
    render_selection(stage)
    before = stage._static_selection_cache.cacheKey()
    before_key = stage._static_selection_key
    if change == "resize":
        stage.resize(1920, 1080)
    elif change == "rename":
        stage.targets = [overlay_module.replace(target, name="新檔名.pdf")]
    elif change == "move":
        stage.targets = [overlay_module.replace(target, center=(700, 510))]
    elif change == "remove":
        stage.targets = []
    elif change == "columns":
        stage.grid_columns = 15
    elif change == "rows":
        stage.grid_rows = 6
    elif change == "grid":
        stage.grid_enabled = False
    elif change == "practice":
        stage.practice = False
    else:
        monkeypatch.setattr(stage, "devicePixelRatioF", lambda: 2.)
    render_selection(stage)
    layer = stage._static_selection_cache
    assert layer.cacheKey() != before
    assert stage._static_selection_key != before_key
    assert layer.devicePixelRatioF() == stage.devicePixelRatioF()
    assert layer.width() == pytest.approx(stage.width() * stage.devicePixelRatioF())
    assert layer.height() == pytest.approx(stage.height() * stage.devicePixelRatioF())


@pytest.mark.parametrize("width,height,practice", [(3440, 1440, True), (1920, 1080, True),
                                                  (800, 800, True), (1440, 900, False)])
def test_cached_selection_is_pixel_identical_to_direct_static_drawing(stage, monkeypatch, width, height, practice):
    stage.resize(width, height)
    stage.grid_enabled = True
    stage.practice = practice
    bounds = stage._grid_cell_rect((7, 6))
    target = DesktopTarget("demo", "長文件名稱_final.pdf", (bounds.center().x(), bounds.center().y()))
    stage.targets = [target]
    stage.target_id = target.id
    stage.grid_active_cell = (7, 6)
    stage.grid_ambiguous_cell = (9, 4)
    stage.progress = .6
    stage.point = (width * .6 + .2, height * .6 + .7)
    cached = render_selection(stage)
    highlights = stage._paint_grid_highlights
    def direct(painter):
        stage._paint_practice_documents(painter)
        highlights(painter)
    monkeypatch.setattr(stage, "_static_selection_layer", lambda: QImage())
    monkeypatch.setattr(stage, "_paint_grid_highlights", direct)
    assert render_selection(stage) == cached


def test_animation_never_reuses_grid_or_pre_swallow_document_layer(stage, monkeypatch):
    clock = [100.]
    monkeypatch.setattr(overlay_module.time, "monotonic", lambda: clock[0])
    stage.grid_enabled = True
    bounds = stage._grid_cell_rect((7, 6))
    target = DesktopTarget("demo", "草稿.txt", (bounds.center().x(), bounds.center().y()))
    stage.targets = [target]
    before = render_selection(stage)
    def stale_layer():
        pytest.fail("animation must not draw the static selection layer")
    monkeypatch.setattr(stage, "_static_selection_layer", stale_layer)
    monkeypatch.setattr(stage, "_paint_eating", lambda painter: None)
    stage.eating = target
    stage.started = 100.
    clock[0] += overlay_module.PULL_START + .01
    during = render_selection(stage)
    assert during.pixelColor(48, 123).alpha() == 0
    icon = stage._practice_grid_layout(target).icon.center().toPoint()
    assert before.pixelColor(icon).red() > 200
    assert during.pixelColor(icon).red() < 50
    stage.bitten = True
    after = render_selection(stage)
    assert after.pixelColor(bounds.center().toPoint()).alpha() == 0


def test_cursor_presentation_updates_only_old_and_new_local_regions(stage, monkeypatch):
    regions = []
    monkeypatch.setattr(stage, "update", lambda region: regions.append(region))
    stage.set_cursor_point((100.25, 300.75))
    assert len(regions) == 1 and isinstance(regions[-1], QRegion)
    assert regions[-1].boundingRect().width() == 45
    stage.set_cursor_point((800.5, 600.25))
    moved = regions[-1]
    assert moved.contains(QPoint(100, 300)) and moved.contains(QPoint(800, 600))
    assert not moved.contains(QPoint(400, 450))
    stage.set_cursor_point((800.5, 600.25))
    assert len(regions) == 2
    stage.set_cursor_point(None)
    assert stage.point is None and len(regions) == 3
    assert regions[-1].contains(QPoint(800, 600))
    assert not regions[-1].contains(QPoint(100, 300))
    stage.set_cursor_point(None)
    assert len(regions) == 3 and not stage.timer.isActive()


@pytest.mark.parametrize("point", [(800.5, 500.25), None])
def test_cursor_dirty_region_covers_every_changed_pixel_and_leaves_static_layer_cached(stage, monkeypatch, point):
    stage.grid_enabled = True
    stage.targets = [DesktopTarget("demo", "草稿.txt", (360, 480))]
    stage.point = (301.25, 333.5)
    before = render_selection(stage)
    cache_key = stage._static_selection_cache.cacheKey()
    regions = []
    monkeypatch.setattr(stage, "update", lambda region: regions.append(region))
    stage.set_cursor_point(point)
    partial = before.copy()
    stage.render(partial, regions[0].boundingRect().topLeft(), regions[0])
    after = render_selection(stage)
    assert partial == after
    old_pixels = np.frombuffer(before.constBits(), dtype=np.uint8).reshape(stage.height(), stage.width(), 4)
    new_pixels = np.frombuffer(after.constBits(), dtype=np.uint8).reshape(stage.height(), stage.width(), 4)
    y, x = np.nonzero(np.any(old_pixels != new_pixels, axis=2))
    assert len(x) > 0 and len(regions) == 1
    assert all(regions[0].contains(QPoint(int(column), int(row))) for column, row in zip(x, y))
    assert stage._static_selection_cache.cacheKey() == cache_key


def test_cursor_dirty_region_is_clipped_and_invalid_sample_clears_old_cursor(stage, monkeypatch):
    regions = []
    monkeypatch.setattr(stage, "update", lambda region: regions.append(region))
    stage.set_cursor_point((0., 0.))
    assert stage.rect().contains(regions[-1].boundingRect())
    stage.set_cursor_point((float("nan"), 10.))
    assert stage.point is None and stage.rect().contains(regions[-1].boundingRect())
