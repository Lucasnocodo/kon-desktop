"""A fox head dashes from right to left, swallows its target, and dissolves.

Only a file icon replica is painted. The bite signal
asks the application to act AFTER that replica is visibly swallowed. The source
head sheet is unchanged; the animation uses a separate twelve-pose sheet.
"""
from __future__ import annotations
from .i18n import literal, get_language

from .paths import resource_root

from dataclasses import dataclass, replace
import math
from pathlib import Path
import time

from PySide6.QtCore import QEvent, QEasingCurve, QFileInfo, QPointF, QRect, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QFont, QIcon, QImage, QKeySequence, QLinearGradient, QPainterPath, QPen, QPixmap, QRadialGradient, QRegion, QShortcut
from .i18n import QFontMetrics, QPainter
from PySide6.QtWidgets import QFileIconProvider, QHBoxLayout, QVBoxLayout
from .i18n import QLabel, QPushButton, QWidget

from .desktop import DesktopTarget
from .effects import LIQUID_DROP_COUNT, LIQUID_DURATION, liquid_particles
from .summon_sound import SummonSound
from .grid_selection import GridSelector

PURPLE = QColor("#ff7848")  # Historical API name, now the reference's fiery accent.
ASSET_PATH = resource_root() / "assets" / "fox-heads.png"
ANIMATION_ASSET_PATH = ASSET_PATH.parent / "fox-dash" / "fox-dash-sheet.png"
SMOKE_ASSET_PATH = ASSET_PATH.parent / "fox-smoke" / "fox-smoke-sheet.png"
HEAD_MOUTH_UV = (.62, .70)
CLOSED_MOUTH_UV = (.69, .62)
SPRAY_MOUTH_UV = (.78, .61)
ANIMATION_SCALE = 2.0
DASH_TRAVEL_RATIO = 1 / 3
REFERENCE_DASH_DURATION = .30
DASH_END = REFERENCE_DASH_DURATION * DASH_TRAVEL_RATIO
PULL_START = DASH_END
SWALLOW_END = DASH_END + .08
CHOMP_END = DASH_END + .10
FRAME_TIMES = (tuple(t * DASH_TRAVEL_RATIO for t in (0, .045, .09, .135, .18, .24, .28))
               + tuple(DASH_END + offset for offset in (0, .025, .05, .07, .09)))
BITE_TIME = CHOMP_END
SPRAY_DURATION = LIQUID_DURATION
SPRAY_DROPLETS = LIQUID_DROP_COUNT
SMOKE_START = BITE_TIME + .10
SMOKE_SPEED = 1.21
REFERENCE_SMOKE_DURATION = .90
_SMOKE_REFERENCE_END = SMOKE_START + REFERENCE_SMOKE_DURATION
DURATION = SMOKE_START + REFERENCE_SMOKE_DURATION / SMOKE_SPEED
SMOKE_HEAD_GONE = SMOKE_START + .42 / SMOKE_SPEED
REDUCED_DURATION = DURATION
REDUCED_BITE_TIME = BITE_TIME
_SPRITE_PAIR = None
_SPRITE_SIGNATURE = None
_ANIMATION_SPRITES = None
_ANIMATION_SIGNATURE = None
_SMOKE_SPRITES = None
_SMOKE_SIGNATURE = None


def _curve(x1, y1, x2, y2):
    curve = QEasingCurve(QEasingCurve.Type.BezierSpline)
    curve.addCubicBezierSegment(QPointF(x1, y1), QPointF(x2, y2), QPointF(1, 1))
    return curve


_OUT = _curve(.23, 1, .32, 1)
_MOVE = _curve(.77, 0, .175, 1)


def _progress(elapsed, start, end, curve=None):
    value = min(max((elapsed - start) / (end - start), 0.0), 1.0)
    return curve.valueForProgress(value) if curve is not None else value


def _mix(a, b, fraction):
    return a + (b - a) * fraction


def _head_sprites():
    """Lazy, lossless runtime cell selection, preserving the original RGBA PNG."""
    global _SPRITE_PAIR, _SPRITE_SIGNATURE
    try:
        info = ASSET_PATH.stat()
        signature = (str(ASSET_PATH), info.st_mtime_ns, info.st_size)
    except OSError:
        return None
    if signature == _SPRITE_SIGNATURE:
        return _SPRITE_PAIR
    sheet = QPixmap(str(ASSET_PATH))
    if sheet.isNull() or sheet.width() % 2 or sheet.width() != 2 * sheet.height():
        _SPRITE_SIGNATURE, _SPRITE_PAIR = signature, None
        return None
    cell = sheet.width() // 2
    _SPRITE_PAIR = (sheet.copy(0, 0, cell, cell), sheet.copy(cell, 0, cell, cell))
    _SPRITE_SIGNATURE = signature
    return _SPRITE_PAIR


def _animation_sprites():
    """Select twelve discrete RGBA poses; cache decoding outside the paint loop."""
    global _ANIMATION_SPRITES, _ANIMATION_SIGNATURE
    try:
        info = ANIMATION_ASSET_PATH.stat()
        signature = (str(ANIMATION_ASSET_PATH), info.st_mtime_ns, info.st_size)
    except OSError:
        return None
    if signature == _ANIMATION_SIGNATURE:
        return _ANIMATION_SPRITES
    sheet = QPixmap(str(ANIMATION_ASSET_PATH))
    if (sheet.isNull() or sheet.width() % 4 or sheet.height() % 3
            or sheet.width() // 4 != sheet.height() // 3 or not sheet.hasAlphaChannel()):
        _ANIMATION_SIGNATURE, _ANIMATION_SPRITES = signature, None
        return None
    cell = sheet.width() // 4
    _ANIMATION_SPRITES = tuple(sheet.copy((i % 4) * cell, (i // 4) * cell, cell, cell)
                               for i in range(len(FRAME_TIMES)))
    _ANIMATION_SIGNATURE = signature
    return _ANIMATION_SPRITES


def _animation_frame(elapsed, reduce_motion=False):
    if reduce_motion:
        # Keep the jaw readable without playing the wind-swept dash poses.
        return 6 if elapsed < CHOMP_END else 11
    return max(i for i, start in enumerate(FRAME_TIMES) if elapsed >= start) if elapsed >= 0 else 0


def _smoke_sprites():
    """Decode the registered 4x3 supernatural smoke sheet once per asset change."""
    global _SMOKE_SPRITES, _SMOKE_SIGNATURE
    try:
        info = SMOKE_ASSET_PATH.stat()
        signature = (str(SMOKE_ASSET_PATH), info.st_mtime_ns, info.st_size)
    except OSError:
        return None
    if signature == _SMOKE_SIGNATURE:
        return _SMOKE_SPRITES
    sheet = QPixmap(str(SMOKE_ASSET_PATH))
    if (sheet.isNull() or sheet.width() % 4 or sheet.height() % 3
            or sheet.width() // 4 != sheet.height() // 3 or not sheet.hasAlphaChannel()):
        _SMOKE_SIGNATURE, _SMOKE_SPRITES = signature, None
        return None
    cell = sheet.width() // 4
    _SMOKE_SPRITES = tuple(sheet.copy(i % 4 * cell, i // 4 * cell, cell, cell) for i in range(12))
    _SMOKE_SIGNATURE = signature
    return _SMOKE_SPRITES


def smoke_frame(elapsed):
    """Continuous sheet position; interpolation preserves the authored curl."""
    return _progress(elapsed, SMOKE_START, DURATION) * 11


def _smoke_clock(elapsed):
    """Retiming applies only after smoke starts, preserving dash, bite and drops."""
    return elapsed if elapsed <= SMOKE_START else SMOKE_START + (elapsed - SMOKE_START) * SMOKE_SPEED


def smoke_rect(rect):
    size = rect.width() * 1.35
    anchor = QPointF(rect.center().x(), rect.top() + rect.height() * .57)
    return QRectF(anchor.x() - size * .50, anchor.y() - size * .62, size, size)


def _draw_spiral_eye(p, center, width, height, tilt=0):
    p.save()
    p.translate(center)
    p.rotate(tilt)
    glow = QRadialGradient(QPointF(0, 0), width * .8)
    glow.setColorAt(0, QColor(255, 100, 27, 120))
    glow.setColorAt(1, QColor(255, 78, 20, 0))
    p.setPen(Qt.NoPen)
    p.setBrush(glow)
    p.drawEllipse(QRectF(-width, -width * .6, width * 2, width * 1.2))
    eye = QPainterPath()
    eye.moveTo(-width / 2, height * .1)
    eye.quadTo(0, -height * .8, width / 2, -height * .05)
    eye.quadTo(width * .15, height * .6, -width / 2, height * .1)
    p.setPen(QPen(QColor("#251717"), 3))
    p.setBrush(QColor("#ef4a16"))
    p.drawPath(eye)
    p.setClipPath(eye, Qt.IntersectClip)
    spiral = QPainterPath()
    for step in range(240):
        angle = step * .17
        radius = width * .65 * step / 240
        point = QPointF(radius * math.cos(angle) - width * .1, radius * math.sin(angle))
        spiral.moveTo(point) if step == 0 else spiral.lineTo(point)
    p.setBrush(Qt.NoBrush)
    p.setPen(QPen(QColor("#ffd16a"), 1.3))
    p.drawPath(spiral)
    p.restore()


def _draw_fallback_head(p, rect, mouth):
    """Head only: white fur, black contours, four orange spiral eyes and fangs."""
    p.save()
    p.translate(rect.topLeft())
    p.scale(rect.width() / 300, rect.height() / 300)
    silhouette = QPainterPath()
    silhouette.moveTo(24, 168)
    silhouette.cubicTo(71, 161, 83, 148, 97, 115)
    for point in ((137, 49), (129, 85), (183, 18), (169, 69), (229, 27), (203, 91),
                  (279, 50), (250, 117), (291, 106), (271, 164), (299, 174),
                  (272, 191), (297, 215), (262, 222), (284, 257), (251, 248),
                  (244, 286), (218, 260), (201, 290), (172, 263), (137, 239)):
        silhouette.lineTo(*point)
    silhouette.cubicTo(102, 218, 43, 212, 26, 187)
    silhouette.closeSubpath()
    fur = QLinearGradient(QPointF(120, 40), QPointF(220, 270))
    fur.setColorAt(0, QColor("#f3f4ed"))
    fur.setColorAt(.55, QColor("#e7eae3"))
    fur.setColorAt(1, QColor("#969e9d"))
    p.setBrush(fur)
    p.setPen(QPen(QColor("#17191b"), 3.2))
    p.drawPath(silhouette)
    p.setPen(QPen(QColor("#7c8588"), 1.8))
    for a, b, c in (((86, 157), (122, 103), (170, 77)), ((127, 159), (170, 130), (221, 86)),
                    ((191, 175), (228, 173), (268, 166)), ((151, 227), (200, 250), (231, 271))):
        stroke = QPainterPath(QPointF(*a))
        stroke.quadTo(QPointF(*b), QPointF(*c))
        p.drawPath(stroke)
    gap = QPainterPath()
    gap.moveTo(33, 184)
    gap.quadTo(108, 183, 179, 149)
    gap.quadTo(221, 155, 215, 179)
    gap.cubicTo(202, 204 + 27 * mouth, 154, 218 + 52 * mouth, 107, 206 + 45 * mouth)
    gap.quadTo(57, 199 + 31 * mouth, 33, 184)
    p.setPen(QPen(QColor("#17191b"), 3))
    p.setBrush(QColor("#150a0a"))
    p.drawPath(gap)
    p.setPen(QPen(QColor("#707975"), 1))
    p.setBrush(QColor("#e8eae3"))
    for index in range(7):
        x = 61 + index * 21
        top = 186 - max(0, x - 82) * .25
        fang = QPainterPath()
        fang.moveTo(x - 8, top)
        fang.lineTo(x + 5, top + 13 + mouth * (21 if index % 2 else 29))
        fang.lineTo(x + 13, top - 6)
        fang.closeSubpath()
        p.drawPath(fang)
    p.setPen(Qt.NoPen)
    p.setBrush(QColor("#131719"))
    nose = QPainterPath()
    nose.moveTo(20, 165)
    nose.quadTo(33, 155, 49, 161)
    nose.lineTo(39, 185)
    nose.quadTo(22, 184, 20, 165)
    p.drawPath(nose)
    for center, width, height, tilt in ((QPointF(154, 131), 52, 31, -27),
                                        (QPointF(214, 139), 45, 24, -23),
                                        (QPointF(264, 158), 25, 16, -20),
                                        (QPointF(222, 239 + mouth * 8), 40, 20, 13)):
        _draw_spiral_eye(p, center, width, height, tilt)
    p.restore()


def draw_fox(p: QPainter, rect: QRectF, mouth: float = 0.0, *, sprite_frame=None, facing_right=False):
    """Draw ONLY the reference head, including in the application's tray/hero."""
    frames = _animation_sprites() if sprite_frame is not None else None
    if frames is not None:
        sprite = frames[min(max(sprite_frame, 0), len(frames) - 1)]
        p.save()
        p.setRenderHint(QPainter.SmoothPixmapTransform)
        p.drawPixmap(rect, sprite, QRectF(sprite.rect()))
        p.restore()
        return
    p.save()
    if facing_right:
        p.translate(rect.left() + rect.right(), 0)
        p.scale(-1, 1)
    sprites = _head_sprites()
    if sprites is None:
        _draw_fallback_head(p, rect, mouth)
        p.restore()
        return
    p.setRenderHint(QPainter.SmoothPixmapTransform)
    original_opacity = p.opacity()
    for sprite, weight in zip(sprites, (1 - mouth, mouth)):
        if weight > 0:
            p.setOpacity(original_opacity * weight)
            p.drawPixmap(rect, sprite, QRectF(sprite.rect()))
    p.restore()


@dataclass(frozen=True)
class EatingPose:
    head_rect: QRectF
    opacity: float
    scale: float
    mouth: float
    smoke: float
    smoke_opacity: float
    file_progress: float
    file_opacity: float
    file_scale: float
    file_position: QPointF
    swallow_done: bool
    sprite_frame: int = 0
    dash: float = 0.0
    impact: float = 0.0


def mouth_position(rect, mouth=0.0):
    """Track the mouth cavity as the lower jaw rises into the closing teeth."""
    uv = tuple(_mix(closed, opened, mouth)
               for closed, opened in zip(CLOSED_MOUTH_UV, HEAD_MOUTH_UV))
    return QPointF(rect.left() + rect.width() * uv[0],
                   rect.top() + rect.height() * uv[1])


def visual_pose(pose, reduce_motion=False):
    """Enlarge the entire VFX around the current bite cavity, keeping it planted."""
    rect = pose.head_rect
    # Reduced motion keeps the open pose stationary until the closed-frame swap.
    anchor = mouth_position(rect, 1.0 if reduce_motion else pose.mouth)
    enlarged = QRectF(anchor.x() + (rect.left() - anchor.x()) * ANIMATION_SCALE,
                      anchor.y() + (rect.top() - anchor.y()) * ANIMATION_SCALE,
                      rect.width() * ANIMATION_SCALE, rect.height() * ANIMATION_SCALE)
    return replace(pose, head_rect=enlarged)


def _dash_distance(target_x, size, width):
    """One third of the previous offscreen run-up, before contact realignment."""
    reference_landing = min(max(target_x - 74 - size * .66, 14), width - size - 14)
    return (reference_landing + size + 24) * DASH_TRAVEL_RATIO


def eating_pose(elapsed, target_center, width, height, reduce_motion=False):
    """Canonical right-facing pose, reflected by the renderer for a leftward bite."""
    x, y = target_center
    size = min(400.0, width * .40, height * .58)
    # Put the file between the upper and lower teeth at contact, removing the
    # previous gap. Transparent top padding can extend above the banner clearance.
    left = min(max(x - size * HEAD_MOUTH_UV[0], 14), width - size - 14)
    top = min(max(y - size * HEAD_MOUTH_UV[1], max(0, 82 - size * .10)), height - size - 20)
    rect = QRectF(left, top, size, size)
    smoke_elapsed = _smoke_clock(elapsed)
    if reduce_motion:
        opacity = _progress(elapsed, 0, .12, _OUT) * (1 - _progress(smoke_elapsed, BITE_TIME + .15, BITE_TIME + .65, _OUT))
        pull = _progress(elapsed, PULL_START, SWALLOW_END, _MOVE)
        mouth = 1 - _progress(elapsed, DASH_END, CHOMP_END)
        return EatingPose(rect, opacity, 1, mouth, 0,
                          _progress(smoke_elapsed, SMOKE_START, SMOKE_START + .20, _OUT) * (1 - _progress(smoke_elapsed, _SMOKE_REFERENCE_END - .35, _SMOKE_REFERENCE_END, _OUT)),
                          pull, 1 - pull, 1, QPointF(x, y), elapsed >= SWALLOW_END,
                          _animation_frame(elapsed, True))
    # Constant travel reaches the bite without a slow tail that reads as a pause.
    arrive = _progress(elapsed, 0, DASH_END)
    rect.moveLeft(_mix(left - _dash_distance(x, size, width), left, arrive))
    appear = _progress(elapsed, 0, .08 * DASH_TRAVEL_RATIO, _OUT)
    dissolve = _progress(smoke_elapsed, SMOKE_START + .28, SMOKE_START + .42, _OUT)
    mouth = 1 - _progress(elapsed, DASH_END, CHOMP_END)
    pull = _progress(elapsed, PULL_START, SWALLOW_END, _MOVE)
    destination = mouth_position(rect, mouth)
    position = QPointF(_mix(x, destination.x(), pull), _mix(y, destination.y(), pull))
    position.setY(position.y() - math.sin(math.pi * pull) * 6)
    dash = _progress(elapsed, 0, .04 * DASH_TRAVEL_RATIO, _OUT) * (1 - _progress(elapsed, .8 * DASH_END, DASH_END))
    smoke = _progress(smoke_elapsed, SMOKE_START, SMOKE_START + .42, _OUT)
    smoke_opacity = _progress(smoke_elapsed, SMOKE_START, SMOKE_START + .08, _OUT) * (1 - _progress(smoke_elapsed, _SMOKE_REFERENCE_END - .32, _SMOKE_REFERENCE_END))
    impact = _progress(elapsed, BITE_TIME - .025, BITE_TIME, _OUT) * (1 - _progress(elapsed, BITE_TIME, BITE_TIME + .14, _OUT))
    return EatingPose(rect, appear * (1 - dissolve), _mix(.96, 1, appear), mouth,
                      smoke, smoke_opacity, pull,
                      1 - _progress(elapsed, SWALLOW_END - .025, SWALLOW_END, _OUT), _mix(1, .18, pull),
                      position, elapsed >= SWALLOW_END, _animation_frame(elapsed), dash, impact)


def _draw_dash(p, pose):
    """Short tapered streaks behind the right-facing head during entry only."""
    if pose.dash <= 0:
        return
    p.save()
    p.setOpacity(p.opacity() * pose.dash * pose.opacity)
    p.setPen(Qt.NoPen)
    rect = pose.head_rect
    for offset, length, thickness in ((.28, .70, .016), (.42, 1.0, .012),
                                       (.57, .85, .019), (.70, .55, .01)):
        start = QPointF(rect.left() + rect.width() * .24, rect.top() + rect.height() * offset)
        tail = start.x() - rect.width() * length
        gradient = QLinearGradient(QPointF(tail, start.y()), start)
        gradient.setColorAt(0, QColor(245, 117, 63, 0))
        gradient.setColorAt(.72, QColor(245, 117, 63, 85))
        gradient.setColorAt(1, QColor(255, 231, 202, 150))
        streak = QPainterPath(start)
        streak.lineTo(tail, start.y() - rect.height() * thickness)
        streak.lineTo(tail, start.y() + rect.height() * thickness)
        streak.closeSubpath()
        p.setBrush(gradient)
        p.drawPath(streak)
    p.restore()


def _draw_liquid(p, pose, elapsed, reduce_motion=False):
    """Short fluid filaments break into beads with gravity and air resistance."""
    age = elapsed - BITE_TIME
    if age < 0 or age >= SPRAY_DURATION:
        return
    # The jaw is closed by this point; its lip is above the open-jaw swallow anchor.
    rect = pose.head_rect
    origin = QPointF(rect.left() + rect.width() * SPRAY_MOUTH_UV[0],
                     rect.top() + rect.height() * SPRAY_MOUTH_UV[1])
    size = pose.head_rect.width()
    opacity = _progress(age, 0, .018, _OUT) * (1 - _progress(age, .52, SPRAY_DURATION))
    p.save()
    p.setOpacity(p.opacity() * opacity)
    p.setPen(Qt.NoPen)
    if reduce_motion:
        p.setBrush(QColor("#d82932"))
        for i in range(5):
            point = origin + QPointF(size * .025 * i, size * .025 * math.sin(i * 2.4))
            p.drawEllipse(point, size * .009, size * .007)
        p.restore()
        return
    states = liquid_particles(age, size)
    # Join only young parcels from each stream. After emission ends, even the
    # trailing parcels leave the lip; the stream breaks instead of staying rigid.
    if age < .12:
        for jet in range(3):
            parcels = sorted((state for state in states if state.jet == jet), key=lambda s: s.age)
            if len(parcels) < 3:
                continue
            points = [origin + QPointF(state.x, state.y) for state in parcels]
            strand = QPainterPath(points[0])
            for index in range(1, len(points)):
                strand.quadTo(points[index - 1], (points[index - 1] + points[index]) / 2)
            strand.lineTo(points[-1])
            p.save()
            p.setOpacity(p.opacity() * (1 - _progress(age, .065, .12)))
            p.setPen(QPen(QColor("#c31c31"), size * (.018 - jet * .003), Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
            p.setBrush(Qt.NoBrush)
            p.drawPath(strand)
            p.restore()
    p.setPen(Qt.NoPen)
    for state in states:
        speed = math.hypot(state.vx, state.vy)
        # Stretch young fast beads along their velocity, then let them round up.
        # Reciprocal axes preserve the bead's projected area during deformation.
        stretch = 1 + min(speed * state.tail_seconds / max(state.radius, .1), 1.3) * (1 - _progress(state.age, .03, .18))
        axis = math.sqrt(stretch)
        radius = state.radius
        point = origin + QPointF(state.x, state.y)
        p.save()
        p.setOpacity(p.opacity() * state.opacity)
        p.translate(point)
        p.rotate(math.degrees(math.atan2(state.vy, state.vx)))
        p.setBrush(QColor("#d82335") if state.index % 3 else QColor("#a9142a"))
        p.drawEllipse(QRectF(-radius * axis, -radius / axis, radius * axis * 2, radius / axis * 2))
        if state.index % 3 == 1:
            p.setBrush(QColor("#ff6a70"))
            p.drawEllipse(QRectF(-radius * .7, -radius * .6, radius, radius * .55))
        p.restore()
    p.restore()


def _draw_bite_impact(p, pose, elapsed, reduce_motion=False):
    """A local sharp flash and expanding hit rays at the closed lip."""
    age = elapsed - BITE_TIME
    if reduce_motion or age < -.015 or age >= .12:
        return
    rect = pose.head_rect
    origin = QPointF(rect.left() + rect.width() * SPRAY_MOUTH_UV[0],
                     rect.top() + rect.height() * SPRAY_MOUTH_UV[1])
    size = rect.width()
    fade = 1 - _progress(elapsed, BITE_TIME, BITE_TIME + .12, _OUT)
    brightness = _progress(elapsed, BITE_TIME - .015, BITE_TIME, _OUT) * fade
    p.save()
    p.setOpacity(p.opacity() * brightness)
    radius = size * _mix(.035, .19, _progress(age, 0, .10, _OUT))
    p.setPen(Qt.NoPen)
    glow = QRadialGradient(origin, radius)
    glow.setColorAt(0, QColor(255, 245, 219, 220))
    glow.setColorAt(.25, QColor(255, 174, 103, 95))
    glow.setColorAt(1, QColor(255, 114, 70, 0))
    p.setBrush(glow)
    p.drawEllipse(origin, radius, radius)
    p.setBrush(QColor("#fff1d8"))
    for i in range(9):
        angle = i * math.tau / 9 + .12
        direction = QPointF(math.cos(angle), math.sin(angle))
        perpendicular = QPointF(-direction.y(), direction.x())
        inner = origin + direction * (radius * .30)
        tip = origin + direction * (radius * (1.0 if i % 2 else .78))
        ray = QPainterPath(inner + perpendicular * size * .007)
        ray.lineTo(tip)
        ray.lineTo(inner - perpendicular * size * .007)
        ray.closeSubpath()
        p.drawPath(ray)
    p.setBrush(Qt.NoBrush)
    p.setPen(QPen(QColor(255, 221, 180, 180), size * .005))
    p.drawEllipse(origin, radius * .63, radius * .63)
    p.restore()


@dataclass(frozen=True)
class SmokePuff:
    center: QPointF
    radius: float
    opacity: float
    rotation: float
    behind: bool


def smoke_puffs(elapsed, rect, reduce_motion=False):
    """Dense local billows expand, rise with buoyancy, and thin out together."""
    if elapsed <= SMOKE_START or elapsed >= DURATION:
        return ()
    elapsed = _smoke_clock(elapsed)
    size = rect.width()
    anchors = ((.69, .62), (.25, .58), (.47, .32), (.38, .74),
               (.65, .45), (.19, .39), (.56, .76), (.73, .67),
               (.31, .25), (.48, .53), (.24, .78), (.59, .29),
               (.34, .47), (.52, .65))
    puffs = []
    for index, (u, v) in enumerate(anchors):
        age = elapsed - SMOKE_START - (index % 5) * .018
        if age <= 0:
            continue
        opacity = _progress(age, 0, .08, _OUT) * (1 - _progress(elapsed, _SMOKE_REFERENCE_END - .32, _SMOKE_REFERENCE_END))
        if opacity <= 0:
            continue
        if reduce_motion:
            center = QPointF(rect.left() + size * u, rect.top() + size * v)
            radius, rotation = size * .18, 0
        else:
            drift = size * (.045 * math.sin(index * 2.4) + (u - .5) * .12)
            center = QPointF(rect.left() + size * u + drift * age + size * (u - .5) * .32 * age * age,
                             rect.top() + size * v - size * ((.18 + (index % 4) * .016) * age + .16 * age * age))
            radius = size * (.07 + .16 * _progress(age, 0, .38, _OUT)) * (1 + .10 * math.sin(index * 1.8))
            rotation = index * 37 + age * (18 if index % 2 else -14)
        puffs.append(SmokePuff(center, radius, opacity, rotation, index % 3 == 1))
    return tuple(puffs)


def _head_smoke_clip(rect, elapsed):
    """Irregular holes remove patches of fur underneath the billowing smoke."""
    visible = QPainterPath()
    if elapsed >= SMOKE_HEAD_GONE:
        return visible
    visible.addRect(rect)
    # Let the billows cover the face before eroding it; early thin haze cannot
    # conceal a hard alpha cutout convincingly.
    smoke_elapsed = _smoke_clock(elapsed)
    if smoke_elapsed <= SMOKE_START + .14:
        return visible
    progress = _progress(smoke_elapsed, SMOKE_START + .14, SMOKE_START + .42)
    holes = QPainterPath()
    holes.setFillRule(Qt.WindingFill)
    for index, puff in enumerate(smoke_puffs(elapsed, rect)):
        # Dissolve only underneath dense foreground clouds. Sparse patches of
        # smoke must not reveal isolated cutouts on the face or forehead.
        if puff.behind or puff.opacity < .85:
            continue
        radius = puff.radius * min(progress * 2, .60)
        holes.addEllipse(puff.center, radius * .70, radius * .62)
        for lobe in range(5):
            angle = lobe * math.tau / 5 + index * .3
            center = puff.center + QPointF(math.cos(angle) * radius * .35,
                                          math.sin(angle) * radius * .30)
            holes.addEllipse(center, radius * .48, radius * .43)
    return visible.subtracted(holes)


def _draw_smoke(p, pose, elapsed, reduce_motion=False, behind=False):
    sprites = _smoke_sprites()
    if sprites is not None:
        if behind or elapsed <= SMOKE_START or elapsed >= DURATION:
            return
        position = 4.0 if reduce_motion else smoke_frame(elapsed)
        index = min(int(position), len(sprites) - 1)
        fraction = position - index
        smoke_elapsed = _smoke_clock(elapsed)
        opacity = (pose.smoke_opacity * .48 if reduce_motion else
                   _progress(smoke_elapsed, SMOKE_START, SMOKE_START + .035, _OUT)
                   * (1 - _progress(smoke_elapsed, _SMOKE_REFERENCE_END - .12, _SMOKE_REFERENCE_END)))
        p.save()
        p.setRenderHint(QPainter.SmoothPixmapTransform)
        p.setOpacity(p.opacity() * opacity)
        rect = smoke_rect(pose.head_rect)
        if fraction > 0 and index + 1 < len(sprites):
            # Blend premultiplied pixels additively, preserving density instead
            # of producing an opacity dip between opaque smoke frames.
            blended = QImage(sprites[index].size(), QImage.Format_ARGB32_Premultiplied)
            blended.fill(Qt.transparent)
            blend_painter = QPainter(blended)
            blend_painter.setOpacity(1 - fraction)
            blend_painter.drawPixmap(0, 0, sprites[index])
            blend_painter.end()
            incoming = QImage(sprites[index + 1].size(), QImage.Format_ARGB32_Premultiplied)
            incoming.fill(Qt.transparent)
            incoming_painter = QPainter(incoming)
            incoming_painter.setOpacity(fraction)
            incoming_painter.drawPixmap(0, 0, sprites[index + 1])
            incoming_painter.end()
            # Weight each source first. Plus with global opacity would clamp
            # the unweighted sum before opacity is applied, thinning the smoke.
            blend_painter = QPainter(blended)
            blend_painter.setCompositionMode(QPainter.CompositionMode_Plus)
            blend_painter.drawImage(0, 0, incoming)
            blend_painter.end()
            p.drawImage(rect, blended)
        else:
            p.drawPixmap(rect, sprites[index], QRectF(sprites[index].rect()))
        p.restore()
        return
    puffs = smoke_puffs(elapsed, pose.head_rect, reduce_motion)
    if not puffs:
        return
    p.save()
    p.setPen(Qt.NoPen)
    for puff in puffs:
        if puff.behind != behind:
            continue
        p.save()
        p.setOpacity(p.opacity() * puff.opacity * (.46 if reduce_motion else .88))
        p.translate(puff.center)
        p.rotate(puff.rotation)
        radius = puff.radius
        # Soft haze around a shaped, denser cloud keeps it legible on a desktop.
        halo = QRadialGradient(QPointF(0, 0), radius * 1.20)
        halo.setColorAt(0, QColor(172, 180, 180, 65))
        halo.setColorAt(1, QColor(172, 180, 180, 0))
        p.setBrush(halo)
        p.drawEllipse(QPointF(0, 0), radius * 1.2, radius)
        cloud = QPainterPath()
        cloud.setFillRule(Qt.WindingFill)
        cloud.addEllipse(QPointF(0, 0), radius * .76, radius * .68)
        for lobe in range(6):
            angle = lobe * math.tau / 6
            offset = QPointF(math.cos(angle) * radius * .48, math.sin(angle) * radius * .39)
            lobe_radius = radius * (.47 + .07 * math.sin(lobe * 2.1))
            cloud.addEllipse(offset, lobe_radius, lobe_radius * .85)
        shade = QRadialGradient(QPointF(-radius * .20, -radius * .28), radius * 1.24)
        shade.setColorAt(0, QColor(224, 225, 218, 225))
        shade.setColorAt(.45, QColor(181, 188, 183, 220))
        shade.setColorAt(.82, QColor(111, 120, 119, 185))
        shade.setColorAt(1, QColor(99, 109, 109, 65))
        p.setBrush(shade)
        p.drawPath(cloud)
        p.restore()
    # Thin curling strands carry the last smoke upward as the dense core thins.
    if not reduce_motion and not behind:
        smoke_elapsed = _smoke_clock(elapsed)
        age = smoke_elapsed - SMOKE_START
        wisp_opacity = _progress(age, .22, .42, _OUT) * (1 - _progress(smoke_elapsed, _SMOKE_REFERENCE_END - .34, _SMOKE_REFERENCE_END))
        p.setOpacity(p.opacity() * wisp_opacity * .32)
        p.setBrush(Qt.NoBrush)
        rect = pose.head_rect
        size = rect.width()
        for index in range(5):
            x = rect.left() + size * (.25 + index * .09)
            y = rect.top() + size * (.35 - .20 * age - .16 * age * age)
            sway = size * .08 * math.sin(index * 1.8 + age * 2)
            curl = QPainterPath(QPointF(x, y))
            curl.cubicTo(QPointF(x + sway, y - size * .12),
                         QPointF(x - sway, y - size * .22),
                         QPointF(x + sway * .35, y - size * .33))
            p.setPen(QPen(QColor(194, 203, 197, 105), size * .011, Qt.SolidLine, Qt.RoundCap))
            p.drawPath(curl)
    p.restore()


class HeroWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(320, 320)

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        center = QPointF(self.width() / 2, self.height() / 2)
        size = min(380, self.width() - 22, self.height() - 14)
        glow = QRadialGradient(center + QPointF(38, 13), size * .55)
        glow.setColorAt(0, QColor(245, 104, 48, 24))
        glow.setColorAt(1, QColor(245, 104, 48, 0))
        p.setPen(Qt.NoPen)
        p.setBrush(glow)
        p.drawEllipse(center, size * .55, size * .55)
        draw_fox(p, QRectF(center.x() - size / 2, center.y() - size / 2, size, size))


@dataclass(frozen=True)
class _PracticeTileLayout:
    cell: QRectF
    card: QRectF
    icon: QRectF
    label: QRectF
    highlight: QRectF
    progress: QRectF
    icon_scale: float
    font: QFont
    title: str


class ScreenOverlay(QWidget):
    sample_requested = Signal()
    exit_requested = Signal()
    bite = Signal(object)
    animation_done = Signal()

    def __init__(self, screen):
        super().__init__()
        self.screen = screen
        self.setWindowFlags(Qt.Tool | Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.WindowTransparentForInput)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setGeometry(screen.geometry())
        self.targets: list[DesktopTarget] = []
        self.point = None
        self.target_id = None
        self.progress = 0.0
        self.grid_enabled = False
        self.grid_columns = 30
        self.grid_rows = 12
        self.grid_active_cell = None
        self.grid_ambiguous_cell = None
        self.practice = True
        self._static_selection_key = None
        self._static_selection_cache = QImage()
        self._practice_layout_viewport = None
        self._practice_layout_cache = {}
        self.calibration_point = None
        self.calibration_step = 0
        self.calibration_total = 9
        self.calibration_title = "瞄準校正"
        self._calibration_feedback = ""
        self._calibration_ready = False
        self._calibration_count = 0
        self._calibration_required = 6
        self._calibration_reason = "請讓慣用眼與手完整入鏡，再穿過手孔瞄準圓點。"
        self._calibration_preview = QImage()
        self.hint = ""
        self.eating = None
        self.bitten = False
        self.started = 0.0
        self.summon_sound = SummonSound()
        self._generation = 0
        self._file_icon = QIcon()
        self.reduce_motion = False
        try:
            from AppKit import NSWorkspace
            self.reduce_motion = bool(NSWorkspace.sharedWorkspace().accessibilityDisplayShouldReduceMotion())
        except ImportError:
            pass
        self.timer = QTimer(self)
        self.timer.setInterval(16)
        self.timer.timeout.connect(self._tick)
        self._build_calibration_card()
        self._calibration_shortcuts = []
        for key, callback in ((Qt.Key_Space, self._request_calibration_sample),
                              (Qt.Key_Return, self._request_calibration_sample),
                              (Qt.Key_Enter, self._request_calibration_sample),
                              (Qt.Key_Escape, self._request_calibration_exit)):
            shortcut = QShortcut(QKeySequence(key), self)
            shortcut.setContext(Qt.WidgetWithChildrenShortcut)
            shortcut.setAutoRepeat(False)
            shortcut.setEnabled(False)
            shortcut.activated.connect(callback)
            self._calibration_shortcuts.append(shortcut)

    def _build_calibration_card(self):
        self.calibration_card = QWidget(self)
        self.calibration_card.setObjectName("calibrationCard")
        self.calibration_card.setFixedSize(300, 360)
        self.calibration_card.setAttribute(Qt.WA_StyledBackground, True)
        self.calibration_card.setStyleSheet("""
            QWidget#calibrationCard { background: rgba(30, 25, 22, 247);
                border: 1px solid #986c4e; border-radius: 14px; }
            QLabel { color: #f3e7d9; background: transparent; border: none; }
            QPushButton { color: #f6e9dc; background: #49392e; border: 1px solid #84624c;
                border-radius: 7px; padding: 6px 7px; font-size: 11px; }
            QPushButton:hover { background: #634635; }
            QPushButton:focus { border: 2px solid #ffc595; }
            QPushButton#recordCalibration { background: #77432d; }
            QPushButton#recordCalibration[ready="true"] { background: #b04f2d; }
        """)
        layout = QVBoxLayout(self.calibration_card)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(6)
        self._calibration_title = QLabel("瞄準校正 1 / 9", self.calibration_card)
        self._calibration_title.setFont(QFont("PingFang TC", 13, QFont.DemiBold))
        self._calibration_title.setFixedHeight(23)
        layout.addWidget(self._calibration_title)
        self._camera_preview_label = QLabel("等待即時鏡頭畫面…", self.calibration_card)
        self._camera_preview_label.setAlignment(Qt.AlignCenter)
        self._camera_preview_label.setFixedSize(276, 155)
        self._camera_preview_label.setStyleSheet("background: #171512; color: #d1baa3; border-radius: 8px;")
        layout.addWidget(self._camera_preview_label)
        self._calibration_state_label = QLabel(self.calibration_card)
        self._calibration_state_label.setFont(QFont("PingFang TC", 11))
        self._calibration_state_label.setWordWrap(True)
        self._calibration_state_label.setFixedHeight(54)
        layout.addWidget(self._calibration_state_label)
        self._calibration_feedback_label = QLabel(self.calibration_card)
        self._calibration_feedback_label.setFont(QFont("PingFang TC", 10))
        self._calibration_feedback_label.setWordWrap(True)
        self._calibration_feedback_label.setFixedHeight(43)
        layout.addWidget(self._calibration_feedback_label)
        controls = QHBoxLayout()
        controls.setSpacing(7)
        self.record_button = QPushButton("開始取樣（空白鍵）", self.calibration_card)
        self.record_button.setObjectName("recordCalibration")
        self.record_button.setFixedHeight(34)
        self.record_button.setAutoRepeat(False)
        self.record_button.setFocusPolicy(Qt.StrongFocus)
        self.record_button.installEventFilter(self)
        self.record_button.clicked.connect(self._request_calibration_sample)
        self.cancel_calibration_button = QPushButton("取消校正（Esc）", self.calibration_card)
        self.cancel_calibration_button.setFixedHeight(34)
        self.cancel_calibration_button.setAutoRepeat(False)
        self.cancel_calibration_button.setFocusPolicy(Qt.StrongFocus)
        self.cancel_calibration_button.installEventFilter(self)
        self.cancel_calibration_button.clicked.connect(self._request_calibration_exit)
        controls.addWidget(self.record_button, 3)
        controls.addWidget(self.cancel_calibration_button, 2)
        layout.addLayout(controls)
        for label in (self._calibration_title, self._camera_preview_label,
                      self._calibration_state_label, self._calibration_feedback_label):
            label.setTextFormat(Qt.PlainText)
        self._refresh_calibration_state()
        self.calibration_card.hide()

    @property
    def calibration_feedback(self):
        return self._calibration_feedback

    @calibration_feedback.setter
    def calibration_feedback(self, value):
        self._calibration_feedback = str(value or "")
        self._refresh_calibration_state()

    @property
    def calibration_ready(self):
        return self._calibration_ready

    @calibration_ready.setter
    def calibration_ready(self, value):
        self._calibration_ready = bool(value)
        self._refresh_calibration_state()

    def set_calibration_state(self, count, required, reason="", ready=False):
        self._calibration_count = max(0, int(count))
        self._calibration_required = max(1, int(required))
        self._calibration_reason = str(reason or "")
        self._calibration_ready = bool(ready)
        self._refresh_calibration_state()

    def _refresh_calibration_state(self):
        if not hasattr(self, "record_button"):
            return
        ready = self._calibration_ready
        badge = "可記錄" if ready else "等待清晰畫面"
        state = f"清晰畫面 {self._calibration_count} / {self._calibration_required} · {badge}"
        self._calibration_state_label.setText(state + ("\n" + self._calibration_reason if self._calibration_reason else ""))
        self._calibration_state_label.setStyleSheet("color: #ffbc87;" if ready else "color: #ceb69f;")
        self._calibration_feedback_label.setText(self._calibration_feedback or "空白鍵、Enter 或按鈕開始取樣；Esc 返回。")
        # These real diagnostics wrap to more than two lines on macOS. Grow
        # the local card to fit the full reason and timeout feedback instead
        # of clipping either message or darkening the desktop behind it.
        extra_height = 0
        for label, minimum in ((self._calibration_state_label, 54),
                               (self._calibration_feedback_label, 43)):
            needed = label.fontMetrics().boundingRect(
                QRect(0, 0, self.calibration_card.width() - 24, 2000),
                Qt.TextWordWrap, label.text()).height() + 5
            label.setFixedHeight(max(minimum, needed))
            extra_height += label.height() - minimum
        self.calibration_card.setFixedHeight(360 + extra_height)
        self._position_calibration_card()
        self.record_button.setText("記錄這一點（空白鍵）" if ready else "開始取樣（空白鍵）")
        self.record_button.setEnabled(True)
        if self.record_button.property("ready") != ready:
            self.record_button.setProperty("ready", ready)
            self.record_button.style().unpolish(self.record_button)
            self.record_button.style().polish(self.record_button)
        self.calibration_card.update()

    def set_calibration_preview(self, image):
        self._calibration_preview = image.copy() if image is not None else QImage()
        if self._calibration_preview.isNull():
            self._camera_preview_label.clear()
            self._camera_preview_label.setText("等待即時鏡頭畫面…")
        else:
            pixmap = QPixmap.fromImage(self._calibration_preview).scaled(
                self._camera_preview_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self._camera_preview_label.setPixmap(pixmap)

    def _position_calibration_card(self):
        if self.calibration_point is None:
            return
        x, y = self.calibration_point
        target = QPointF(x * self.width(), y * self.height())
        width, height = self.calibration_card.width(), self.calibration_card.height()
        right = target.x() + 54
        left = target.x() - width - 54
        if right + width <= self.width() - 12:
            card_x = right
        elif left >= 12:
            card_x = left
        else:
            card_x = min(max(target.x() - width / 2, 12), self.width() - width - 12)
        card_y = min(max(target.y() - height / 2, 12), self.height() - height - 12)
        # On very narrow screens, use a vertical neighbor rather than cover
        # the aim ring. Standard Mac desktop widths use the left/right cases.
        if right + width > self.width() - 12 and left < 12:
            below = target.y() + 54
            above = target.y() - height - 54
            card_y = below if below + height <= self.height() - 12 else max(12, above)
        self.calibration_card.move(round(card_x), round(card_y))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, "calibration_card"):
            self._position_calibration_card()

    def _set_calibration_window_mode(self, interactive):
        flags = Qt.Window if interactive else Qt.Tool
        flags |= Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint
        if not interactive:
            flags |= Qt.WindowTransparentForInput
        if self.windowFlags() != flags:
            geometry = self.geometry()
            self.setWindowFlags(flags)
            self.setGeometry(geometry)
        for shortcut in self._calibration_shortcuts:
            shortcut.setEnabled(interactive)

    def _focus_calibration(self):
        if self.calibration_point is None or not self.isVisible():
            return
        self.raise_()
        self.activateWindow()
        if self.windowHandle() is not None:
            self.windowHandle().requestActivate()
        self.setFocus(Qt.OtherFocusReason)

    def _request_calibration_sample(self):
        if self.calibration_point is not None:
            # Always reach the application's pending-capture handler, even
            # before readiness. That handler provides visible rejection/status.
            self.sample_requested.emit()

    def _request_calibration_exit(self):
        if self.calibration_point is not None:
            self.exit_requested.emit()

    def set_calibration(self, point=None, step=0, *, total=9, title="瞄準校正"):
        if not isinstance(total, int) or isinstance(total, bool) or total < 1:
            raise ValueError("校正／驗證的總點數必須是正整數")
        if point is not None and self.eating is not None:
            self.cancel()
        self.calibration_point, self.calibration_step = point, step
        self.calibration_total, self.calibration_title = total, str(title)
        interactive = point is not None
        self._set_calibration_window_mode(interactive)
        self.calibration_card.setVisible(interactive)
        self._calibration_title.setText(f"{self.calibration_title} {step + 1} / {self.calibration_total}")
        self._position_calibration_card()
        self.show()
        if interactive:
            self._focus_calibration()
            QTimer.singleShot(0, self._focus_calibration)
        self.update()

    def keyPressEvent(self, event):
        if self.calibration_point is not None and event.isAutoRepeat() and event.key() in (
            Qt.Key_Space, Qt.Key_Return, Qt.Key_Enter, Qt.Key_Escape,
        ):
            event.accept()
            return
        # QShortcut owns all calibration actions; no duplicate key handler.
        super().keyPressEvent(event)

    def eventFilter(self, obj, event):
        if self.calibration_point is not None and event.type() == QEvent.KeyPress and event.isAutoRepeat():
            if event.key() in (Qt.Key_Space, Qt.Key_Return, Qt.Key_Enter, Qt.Key_Escape):
                return True
        return super().eventFilter(obj, event)

    def prepare_animation(self):
        """Decode static effects before the animation clock starts."""
        if _animation_sprites() is None:
            _head_sprites()
        _smoke_sprites()

    def play(self, target: DesktopTarget):
        if self.eating is not None:
            return
        generation = self._generation
        file_icon = QIcon()
        if target.snapshot is not None:
            file_icon = QFileIconProvider().icon(QFileInfo(str(target.snapshot.path)))
        self.prepare_animation()
        # Native audio can initialize its playback device synchronously on the
        # first call. Neither that work nor sprite decoding belongs to elapsed
        # animation time, or a first paint can already be past the entire bite.
        self.summon_sound.play()
        if self._generation != generation or self.eating is not None:
            return
        self._generation += 1
        self.eating, self.bitten = target, False
        self._file_icon = file_icon
        self.started = time.monotonic()
        self.show()
        self.timer.start()
        self.update()

    def cancel(self, *, stop_sound=True):
        self._generation += 1
        self.timer.stop()
        if stop_sound:
            self.summon_sound.stop()
        self.eating = None
        self.bitten = False
        self._file_icon = QIcon()
        self.hide()
        self.calibration_point = None
        self.calibration_step = 0
        self.calibration_total = 9
        self.calibration_title = "瞄準校正"
        self.calibration_card.hide()
        self.set_calibration_preview(None)
        self._set_calibration_window_mode(False)


    def _sync_animation_timer(self):
        if self.eating is not None:
            if not self.timer.isActive():
                self.timer.start()
        else:
            self.timer.stop()


    def _tick(self):
        now = time.monotonic()
        target = self.eating
        if target is None:
            self._sync_animation_timer()
            self.update()
            return
        generation = self._generation
        elapsed = now - self.started
        bite_time = REDUCED_BITE_TIME if self.reduce_motion else BITE_TIME
        duration = REDUCED_DURATION if self.reduce_motion else DURATION
        if elapsed >= bite_time and not self.bitten:
            self.bitten = True
            self.bite.emit(target)
            if self._generation != generation or self.eating is not target:
                return
        if elapsed >= duration:
            self.eating = None
            self._file_icon = QIcon()
            self.animation_done.emit()
        self._sync_animation_timer()
        self.update()

    def closeEvent(self, event):
        self.cancel()
        super().closeEvent(event)

    def set_cursor_point(self, point):
        """Present a cursor sample without invalidating the full-screen layer."""
        if point is not None:
            try:
                point = tuple(float(value) for value in point)
                if len(point) != 2 or not all(math.isfinite(value) for value in point):
                    point = None
            except (TypeError, ValueError, OverflowError):
                point = None
        old_point = self.point
        if point == old_point:
            return
        self.point = point
        region = QRegion()
        for value in (old_point, point):
            if value is not None:
                x, y = value
                region |= QRegion(QRectF(x - 22, y - 22, 44, 44).toAlignedRect())
        region &= QRegion(self.rect())
        if not region.isEmpty():
            self.update(region)

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        # Clear old overlay pixels to transparent, keeping the real desktop
        # completely visible outside the local marker and camera/control card.
        p.setCompositionMode(QPainter.CompositionMode_Source)
        p.fillRect(self.rect(), Qt.transparent)
        p.setCompositionMode(QPainter.CompositionMode_SourceOver)
        if self.calibration_point is not None:
            x, y = self.calibration_point
            point = QPointF(x * self.width(), y * self.height())
            p.setPen(QPen(PURPLE, 2))
            p.setBrush(Qt.NoBrush)
            p.drawEllipse(point, 24, 24)
            p.setPen(QPen(QColor("#fff6e8"), 1.3))
            p.drawEllipse(point, 8, 8)
            p.setPen(QColor("#ffb389"))
            p.setFont(QFont("PingFang TC", 11, QFont.DemiBold))
            p.drawText(QRectF(point.x() - 38, point.y() + 32, 76, 22), Qt.AlignCenter,
                       f"{self.calibration_step + 1} / {self.calibration_total}")
            return
        if self.eating is None:
            layer = self._static_selection_layer()
            if not layer.isNull():
                # The destination has just been cleared. Source preserves the
                # cached raster's colors, avoiding another alpha round trip.
                p.setCompositionMode(QPainter.CompositionMode_Source)
                p.drawImage(QPointF(0, 0), layer)
                p.setCompositionMode(QPainter.CompositionMode_SourceOver)
            if self.grid_enabled:
                self._paint_grid_highlights(p)
        else:
            # An eaten document's icon/card changes during the swallow. Reuse
            # geometry but never paint the full selection layer in this phase.
            self._paint_practice_documents(p)
        for target in self.targets:
            if target.id == self.target_id and self.eating is None:
                if self.grid_enabled and self.practice:
                    layout = self._practice_grid_layout(target)
                    if layout is not None:
                        p.save()
                        p.setClipRect(layout.cell)
                        p.setPen(QPen(PURPLE, 2))
                        p.setBrush(QColor(231, 118, 71, 25))
                        radius = min(12., layout.card.width() * .2)
                        p.drawRoundedRect(layout.highlight, radius, radius)
                        p.setPen(QPen(QColor("#ffe1c6"), 3))
                        p.drawArc(layout.progress, 90 * 16, -int(360 * 16 * self.progress))
                        p.restore()
                    continue
                x, y = target.center
                p.setPen(QPen(PURPLE, 2))
                p.setBrush(QColor(231, 118, 71, 25))
                p.drawRoundedRect(QRectF(x - 43, y - 43, 86, 86), 16, 16)
                p.setPen(QPen(QColor("#ffe1c6"), 3))
                p.drawArc(QRectF(x - 49, y - 49, 98, 98), 90 * 16, -int(360 * 16 * self.progress))
        if self.point is not None and self.eating is None:
            point = QPointF(*self.point)
            p.setPen(QPen(PURPLE, 1.5))
            p.setBrush(Qt.NoBrush)
            p.drawEllipse(point, 10, 10)
            for a, b in (((-17, 0), (-6, 0)), ((6, 0), (17, 0)), ((0, -17), (0, -6)), ((0, 6), (0, 17))):
                p.drawLine(point + QPointF(*a), point + QPointF(*b))
        p.setFont(QFont("PingFang TC", 11))
        banner = self.hint or ("演練模式 · 瞄準虛擬檔案，維持手勢並說觸發詞" if self.practice else "桌面模式 · 維持鎖定並說觸發詞 → 移到垃圾桶")
        lines = self._banner_lines(banner, p.fontMetrics(), 530)
        banner_height = 64 if len(lines) > 1 else 48
        p.setBrush(QColor(25, 19, 16, 232))
        p.setPen(QPen(QColor("#77543d"), 1))
        p.drawRoundedRect(QRectF(self.width() / 2 - 280, 24, 560, banner_height), 16, 16)
        p.setPen(QColor("#f1e4d6"))
        text_rect = QRectF(self.width() / 2 - 275, 24, 550, banner_height)
        if len(lines) == 1:
            p.drawText(text_rect, Qt.AlignCenter | Qt.TextSingleLine, lines[0])
        else:
            text_rect.adjust(0, 7, 0, -7)
            line_height = text_rect.height() / 2
            p.drawText(QRectF(text_rect.x(), text_rect.y(), text_rect.width(), line_height),
                       Qt.AlignCenter | Qt.TextSingleLine, lines[0])
            p.setPen(QColor("#dac6b6"))
            p.drawText(QRectF(text_rect.x(), text_rect.y() + line_height, text_rect.width(), line_height),
                       Qt.AlignCenter | Qt.TextSingleLine, lines[1])
        if self.eating is not None:
            self._paint_eating(p)

    def _selection_cache_key(self):
        return (self.width(), self.height(), self.devicePixelRatioF(),
                self.logicalDpiX(), self.logicalDpiY(), get_language(), self.grid_enabled,
                self.grid_columns, self.grid_rows, self.practice,
                tuple((target.id, target.name, target.center) for target in self.targets))

    def _static_selection_layer(self):
        key = self._selection_cache_key()
        if key == self._static_selection_key:
            return self._static_selection_cache
        self._practice_layout_cache.clear()
        layer = QImage()
        if self.grid_enabled or (self.practice and self.targets):
            ratio = self.devicePixelRatioF()
            layer = QImage(math.ceil(self.width() * ratio), math.ceil(self.height() * ratio), QImage.Format_ARGB32)
            layer.setDevicePixelRatio(ratio)
            layer.setDotsPerMeterX(round(self.logicalDpiX() / .0254))
            layer.setDotsPerMeterY(round(self.logicalDpiY() / .0254))
            layer.fill(Qt.transparent)
            painter = QPainter(layer)
            try:
                painter.setRenderHint(QPainter.Antialiasing)
                self._paint_practice_documents(painter)
            finally:
                painter.end()
        self._static_selection_cache = layer
        self._static_selection_key = key
        return layer

    def _paint_practice_documents(self, p):
        if not self.practice:
            return
        for target in self.targets:
            draw_original = True
            if self.eating is target:
                if self.bitten:
                    continue
                draw_original = time.monotonic() - self.started < PULL_START
            if self.grid_enabled:
                layout = self._practice_grid_layout(target)
                if layout is not None:
                    self._paint_practice_tile(p, layout, draw_original)
                continue
            x, y = target.center
            p.setPen(QPen(QColor("#80634e"), 1))
            p.setBrush(QColor(25, 20, 17, 235))
            p.drawRoundedRect(QRectF(x - 54, y - 56, 108, 117), 14, 14)
            if draw_original:
                self._document(p, QPointF(x, y - 8), 1)
            p.setPen(QColor("#eee4d8"))
            p.setFont(QFont("PingFang TC", 10))
            p.drawText(QRectF(x - 50, y + 30, 100, 24), Qt.AlignCenter, literal(target.name))

    def _practice_grid_layout(self, target):
        """Keep a virtual document and its selection marks inside its grid cell."""
        if not self.grid_enabled or not self.practice:
            return None
        viewport = (self.width(), self.height(), self.devicePixelRatioF(),
                    self.logicalDpiX(), self.logicalDpiY(), self.grid_columns, self.grid_rows)
        if viewport != self._practice_layout_viewport:
            self._practice_layout_cache.clear()
            self._practice_layout_viewport = viewport
        try:
            key = (target.id, target.name, tuple(target.center))
            if key in self._practice_layout_cache:
                return self._practice_layout_cache[key]
        except TypeError:
            return None
        try:
            selector = GridSelector(self.grid_columns, self.grid_rows)
        except ValueError:
            return None
        cell = self._grid_cell_rect(selector.cell_for(target.center, (self.width(), self.height())))
        if cell is None:
            return None
        padding = min(4., cell.width() * .12, cell.height() * .08)
        card = cell.adjusted(padding, padding, -padding, -padding)
        inset = min(5., card.width() * .1, card.height() * .06)
        inner = card.adjusted(inset, inset, -inset, -inset)
        font = QFont("PingFang TC")
        font.setPixelSize(13 if card.width() >= 90 else 11 if card.width() >= 45 else 9)
        metrics = QFontMetrics(font)
        label_height = min(inner.height() / 2, metrics.height() + 2)
        label = QRectF(inner.left(), inner.bottom() - label_height, inner.width(), label_height)
        gap = min(4., card.height() * .04)
        icon_area = QRectF(inner.left(), inner.top(), inner.width(), max(0., label.top() - gap - inner.top()))
        scale = min(1., icon_area.width() / 40., icon_area.height() / 50.)
        icon = QRectF(0, 0, 40 * scale, 50 * scale)
        icon.moveCenter(icon_area.center())
        highlight = card.adjusted(-1., -1., 1., 1.)
        arc_size = max(0., min(98., card.width() - 6., card.height() - 6.))
        progress = QRectF(0, 0, arc_size, arc_size)
        progress.moveCenter(card.center())
        title = metrics.elidedText(literal(target.name), Qt.ElideMiddle, max(0, int(label.width())))
        layout = _PracticeTileLayout(cell, card, icon, label, highlight, progress, scale, font, title)
        self._practice_layout_cache[key] = layout
        return layout

    def _paint_practice_tile(self, p, layout, draw_original):
        p.save()
        p.setClipRect(layout.cell)
        p.setPen(QPen(QColor("#80634e"), 1))
        p.setBrush(QColor(25, 20, 17, 235))
        radius = min(12., layout.card.width() * .2, layout.card.height() * .16)
        p.drawRoundedRect(layout.card, radius, radius)
        if draw_original:
            # The paper path's vertical center is -2 in its native coordinates.
            center = layout.icon.center() + QPointF(0, 2 * layout.icon_scale)
            self._document(p, center, layout.icon_scale)
        p.setClipRect(layout.label, Qt.IntersectClip)
        p.setPen(QColor("#eee4d8"))
        p.setFont(layout.font)
        p.drawText(layout.label, Qt.AlignCenter | Qt.TextSingleLine, literal(layout.title))
        p.restore()

    def _grid_cell_rect(self, cell):
        try:
            column, row = cell
        except (TypeError, ValueError):
            return None
        columns, rows = self.grid_columns, self.grid_rows
        if (any(not isinstance(value, int) or isinstance(value, bool) for value in (columns, rows, column, row))
                or columns < 1 or rows < 1 or not 0 <= column < columns or not 0 <= row < rows):
            return None
        return QRectF(column * self.width() / columns, row * self.height() / rows,
                      self.width() / columns, self.height() / rows)

    def _paint_grid_highlights(self, p):
        p.save()
        p.setRenderHint(QPainter.Antialiasing, True)
        active = self._grid_cell_rect(self.grid_active_cell)
        if active is not None:
            locked = self.progress >= 1.0
            p.setPen(QPen(QColor("#ffe1c6") if locked else QColor("#ffad7a"), 2.2 if locked else 1.7))
            p.setBrush(QColor(255, 164, 100, 24))
            p.drawRect(active.adjusted(1.2, 1.2, -1.2, -1.2))
        ambiguous = self._grid_cell_rect(self.grid_ambiguous_cell)
        if ambiguous is not None:
            p.setPen(QPen(QColor("#ff777c"), 2))
            p.setBrush(QColor(239, 87, 93, 22))
            p.drawRect(ambiguous.adjusted(1.2, 1.2, -1.2, -1.2))
        p.restore()

    @staticmethod
    def _banner_lines(text, metrics, width):
        """Fit local status text while retaining a locked file's speech cue."""
        lines = str(text).splitlines() or [""]
        first = lines[0]
        prefix, speech, suffix = first.rpartition(" · 請說 ")
        if first.startswith("已鎖定 ") and speech:
            heading = "已鎖定 "
            tail = speech + suffix
            name_width = width - metrics.horizontalAdvance(heading + tail)
            if name_width >= metrics.horizontalAdvance("…"):
                first = heading + metrics.elidedText(literal(prefix[len(heading):]), Qt.ElideMiddle, name_width) + tail
            else:
                first = metrics.elidedText(first, Qt.ElideMiddle, width)
        else:
            first = metrics.elidedText(first, Qt.ElideMiddle, width)
        if len(lines) == 1:
            return (first,)
        return first, metrics.elidedText(" · ".join(lines[1:]), Qt.ElideRight, width)

    @staticmethod
    def _document(p, center, scale):
        p.save()
        p.translate(center)
        p.scale(scale, scale)
        p.setPen(QPen(QColor("#d3bd9d"), 1))
        p.setBrush(QColor("#fff3df"))
        p.drawRoundedRect(QRectF(-19, -26, 38, 48), 5, 5)
        p.setPen(QPen(QColor("#ad8861"), 2))
        for y in (-8, 0, 8):
            p.drawLine(QPointF(-10, y), QPointF(10, y))
        p.restore()

    def _paint_file(self, p, center, scale):
        if self._file_icon.isNull():
            self._document(p, center, scale)
            return
        p.save()
        p.translate(center)
        p.scale(scale, scale)
        self._file_icon.paint(p, QRect(-32, -32, 64, 64))
        p.restore()

    def _eating_pose(self, elapsed):
        """Compute in mirrored screen coordinates, preserving all VFX anchors."""
        target = self.eating
        if target is None:
            return None
        center = (self.width() - target.center[0], target.center[1])
        return visual_pose(
            eating_pose(elapsed, center, self.width(), self.height(), self.reduce_motion),
            self.reduce_motion,
        )

    def _paint_eating(self, p):
        elapsed = time.monotonic() - self.started
        pose = self._eating_pose(elapsed)
        if pose is None:
            return
        original_transform = p.worldTransform()
        p.save()
        # Reflect the entire authored head, mouth and effects together. Merely
        # flipping the sprite would leave the swallow and spray on the wrong lip.
        p.translate(self.width(), 0)
        p.scale(-1, 1)
        _draw_dash(p, pose)
        _draw_smoke(p, pose, elapsed, self.reduce_motion, behind=True)
        p.save()
        p.setOpacity(p.opacity() * pose.opacity)
        center = pose.head_rect.center()
        p.translate(center)
        p.scale(pose.scale, pose.scale)
        p.translate(-center)
        # Keep the lip planted while the skull recoils and compresses at contact.
        lip = QPointF(pose.head_rect.left() + pose.head_rect.width() * SPRAY_MOUTH_UV[0],
                      pose.head_rect.top() + pose.head_rect.height() * SPRAY_MOUTH_UV[1])
        p.translate(lip)
        p.rotate(-2.8 * pose.impact)
        p.scale(1 + .09 * pose.impact, 1 - .07 * pose.impact)
        p.translate(-lip)
        if not self.reduce_motion and _smoke_sprites() is None:
            p.setClipPath(_head_smoke_clip(pose.head_rect, elapsed), Qt.IntersectClip)
        draw_fox(p, pose.head_rect, pose.mouth, sprite_frame=pose.sprite_frame, facing_right=True)
        p.restore()
        if pose.file_opacity > 0:
            p.save()
            # The target retains its own appearance and asymmetric alpha canvas;
            # only its position follows the reflected mouth into the bite.
            p.setWorldTransform(original_transform)
            p.setOpacity(p.opacity() * pose.file_opacity)
            self._paint_file(p, QPointF(self.width() - pose.file_position.x(),
                                       pose.file_position.y()), pose.file_scale)
            p.restore()
        _draw_smoke(p, pose, elapsed, self.reduce_motion)
        _draw_bite_impact(p, pose, elapsed, self.reduce_motion)
        _draw_liquid(p, pose, elapsed, self.reduce_motion)
        p.restore()
