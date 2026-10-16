"""Guarded interpolation of nine personal eye-through-hole calibration rays.

An exact interpolation residual is not an accuracy measurement. Live use
requires independent camera captures, including edge and interstitial tests,
and their frame-level pixel-error check. No camera or file action runs here.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
from typing import Sequence

import numpy as np

DEFAULT_TARGETS = tuple((x, y) for y in (.18, .5, .82) for x in (.18, .5, .82))
VALIDATION_TARGETS = ((.05, .05), (.95, .05), (.05, .95), (.95, .95),
                      (.34, .34), (.66, .34), (.34, .66), (.66, .66), (.5, .82))


def _points(values, count=None):
    try:
        array = np.asarray(values, dtype=float)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("校正點資料格式無效。") from error
    if array.ndim != 2 or array.shape[1:] != (2,) or (count is not None and len(array) != count) or not np.isfinite(array).all():
        raise ValueError("校正需要完整、有限的 XY 點。")
    return array


def _pairs(array):
    return tuple((float(point[0]), float(point[1])) for point in array)


def _cross(a, b):
    return float(a[0] * b[1] - a[1] * b[0])


def _segments_intersect(a, b, c, d):
    signs = [_cross(b-a, c-a), _cross(b-a, d-a), _cross(d-c, a-c), _cross(d-c, b-c)]
    if signs[0] * signs[1] < -1e-14 and signs[2] * signs[3] < -1e-14:
        return True
    for value, point, p, q in ((signs[0], c, a, b), (signs[1], d, a, b),
                               (signs[2], a, c, d), (signs[3], b, c, d)):
        if abs(value) <= 1e-10 and np.all(point >= np.minimum(p, q)-1e-10) and np.all(point <= np.maximum(p, q)+1e-10):
            return True
    return False


@dataclass(frozen=True)
class _Triangle:
    source: np.ndarray
    destination: np.ndarray
    inverse: np.ndarray
    condition: float
    gain: float
    pixel_gain_per_mm: float

    def map(self, point):
        uv = self.inverse @ (point - self.source[0])
        weights = np.array([1-uv.sum(), uv[0], uv[1]])
        if np.any(weights < -1e-8) or np.any(weights > 1+1e-8):
            return None
        return weights @ self.destination


def _mesh(raw, target, source_mean, source_scale, screen_size):
    """All cells convex, consistently oriented, with a simple outer boundary."""
    rows, cols, _ = raw.shape
    normalized = (raw-source_mean) / source_scale
    flat, destinations = normalized.reshape(-1, 2), target.reshape(-1, 2)
    boundary = ([col for col in range(cols)] + [row*cols+cols-1 for row in range(1, rows)]
                + [(rows-1)*cols+col for col in range(cols-2, -1, -1)]
                + [row*cols for row in range(rows-2, 0, -1)])
    for i in range(len(boundary)):
        for j in range(i+1, len(boundary)):
            if j == i+1 or i == 0 and j == len(boundary)-1:
                continue
            if _segments_intersect(flat[boundary[i]], flat[boundary[(i+1)%len(boundary)]],
                                   flat[boundary[j]], flat[boundary[(j+1)%len(boundary)]]):
                raise ValueError("校正邊界交叉；請重取位置混淆的點。")
    orientation = None
    result = []
    for row in range(rows-1):
        for col in range(cols-1):
            a, b = row*cols+col, row*cols+col+1
            c, d = (row+1)*cols+col, (row+1)*cols+col+1
            quad = flat[[a, b, d, c]]
            turns = [_cross(quad[(i+1)%4]-quad[i], quad[(i+2)%4]-quad[(i+1)%4]) for i in range(4)]
            if min(abs(turn) for turn in turns) < 1e-5 or not (all(turn > 0 for turn in turns) or all(turn < 0 for turn in turns)):
                raise ValueError("校正區域折疊或太窄；請重取相鄰位置的點。")
            sign = 1 if turns[0] > 0 else -1
            if orientation is not None and sign != orientation:
                raise ValueError("校正區域方向不一致；請重取錯位的點。")
            orientation = sign
            options = []
            for indices in (((a,b,d), (a,d,c)), ((a,b,c), (b,d,c))):
                triangles = []
                for triple in indices:
                    src, dst = flat[list(triple)], destinations[list(triple)]
                    matrix = np.column_stack((src[1]-src[0], src[2]-src[0]))
                    if abs(np.linalg.det(matrix)) < .002:
                        break
                    inverse = np.linalg.inv(matrix)
                    jacobian = np.column_stack((dst[1]-dst[0], dst[2]-dst[0])) @ inverse
                    gain = float(np.linalg.svd(jacobian, compute_uv=False)[0])
                    condition = max(float(np.linalg.cond(np.column_stack((src[(i+1)%3]-src[i], src[(i+2)%3]-src[i])))) for i in range(3))
                    physical = np.diag(screen_size) @ jacobian @ np.diag(1/source_scale)
                    pixel_gain = float(np.linalg.svd(physical, compute_uv=False)[0]/1000)
                    if condition > 25 or gain > 6:
                        break
                    triangles.append(_Triangle(src, dst, inverse, condition, gain, pixel_gain))
                if len(triangles) == 2:
                    options.append(triangles)
            if not options:
                raise ValueError("相鄰校正點太接近，定位噪聲會被過度放大；請重取這一區。")
            # Pick the safer diagonal using actual output pixels and metric input.
            result.extend(min(options, key=lambda option: (max(t.pixel_gain_per_mm for t in option), max(t.condition for t in option))))
    return tuple(result)


@dataclass(frozen=True)
class ValidationReport:
    status: str
    passed: bool
    reason: str
    sample_count: int
    target_points: tuple = ()
    predicted_points: tuple = ()
    point_errors: tuple = ()
    point_pixel_errors: tuple = ()
    point_p95_pixels: tuple = ()
    point_frame_counts: tuple = ()
    rms_error: float | None = None
    max_error: float | None = None
    rms_pixels: float | None = None
    max_pixels: float | None = None
    frame_samples_checked: bool = False
    max_rms_pixels: float = 30.
    max_point_pixels: float = 40.

    @property
    def pixel_errors(self):
        return self.point_pixel_errors

    def to_dict(self):
        return dict(self.__dict__)


class PersonalCalibration:
    calibration_model = "personal_piecewise_affine_v1"

    @classmethod
    def fit(cls, raw_points, screen_points=None, *, screen_size=(1, 1)):
        raw = _points(raw_points, 9)
        target = _points(DEFAULT_TARGETS if screen_points is None else screen_points, 9)
        size = np.asarray(screen_size, dtype=float)
        if size.shape != (2,) or not np.isfinite(size).all() or np.any(size <= 0):
            raise ValueError("螢幕尺寸無效。")
        grid = target.reshape(3,3,2)
        xs, ys = grid[0,:,0], grid[:,0,1]
        expected = np.array([(x,y) for y in ys for x in xs])
        if (not np.allclose(target, expected, atol=1e-8) or np.any(np.diff(xs) <= 0)
                or np.any(np.diff(ys) <= 0) or np.any(target < 0) or np.any(target > 1)
                or xs[-1]-xs[0] < .35 or ys[-1]-ys[0] < .35):
            raise ValueError("個人校正需要由左到右、由上到下排列的九點格線。")
        mean, scale = raw.mean(0), np.ptp(raw, axis=0)
        if not np.isfinite(mean).all() or not np.isfinite(scale).all() or np.any(scale < 1e-4):
            raise ValueError("瞄準線分布不足；請確認每個圓點都取得不同位置。")
        nraw = (raw-mean)/scale
        distances = np.linalg.norm(nraw[:,None]-nraw[None,:],axis=2)+np.eye(9)*100
        if distances.min() < .015:
            raise ValueError("兩個校正點幾乎重合；請重取這些位置。")
        core = _mesh(raw.reshape(3,3,2), grid, mean, scale, size)
        model = cls()
        model.raw_points, model.screen_points = _pairs(raw), _pairs(target)
        model.screen_size, model._mean, model._scale = tuple(float(v) for v in size), mean, scale
        model._triangles = core
        model.coverage_bounds = (float(xs[0]),float(ys[0]),float(xs[-1]),float(ys[-1]))
        model.full_screen_coverage = bool(np.allclose([xs[0],ys[0],xs[-1],ys[-1]],[0,0,1,1]))
        model.coverage_reason = "已取得完整螢幕的幾何覆蓋。" if model.full_screen_coverage else "目前只支持九個標記內的區域。"
        if not model.full_screen_coverage:
            extended = np.empty((5,5,2))
            extended[1:4,1:4] = raw.reshape(3,3,2)
            extended[1:4,0] = extended[1:4,1] + (0-xs[0])/(xs[1]-xs[0])*(extended[1:4,2]-extended[1:4,1])
            extended[1:4,4] = extended[1:4,3] + (1-xs[2])/(xs[2]-xs[1])*(extended[1:4,3]-extended[1:4,2])
            extended[0] = extended[1] + (0-ys[0])/(ys[1]-ys[0])*(extended[2]-extended[1])
            extended[4] = extended[3] + (1-ys[2])/(ys[2]-ys[1])*(extended[3]-extended[2])
            ghost_target = np.array([(x,y) for y in (0,*ys,1) for x in (0,*xs,1)]).reshape(5,5,2)
            try:
                ghost = _mesh(extended, ghost_target, mean, scale, size)
            except ValueError as error:
                model.coverage_reason = f"邊緣延伸未通過幾何檢查，保留中央演練：{error}"
            else:
                model._triangles = ghost
                model.coverage_bounds = (0.,0.,1.,1.)
                model.full_screen_coverage = True
                model.coverage_reason = "邊緣延伸通過幾何檢查；實際使用仍須新影格驗證。"
        model.sample_count = 9
        model.condition_number = max(t.condition for t in model._triangles)
        model.max_jacobian_gain = max(t.gain for t in model._triangles)
        model.max_noise_gain_pixels_per_mm = max(t.pixel_gain_per_mm for t in model._triangles)
        errors = [np.linalg.norm(np.array(model.predict(point))-target[index]) for index,point in enumerate(raw)]
        model.interpolation_rms_error = float(np.sqrt(np.mean(np.array(errors)**2)))
        model.interpolation_max_error = float(max(errors))
        model.rms_error = model.max_error = None
        model.validation = None
        model.validated = False
        model._validation_request = None
        return model

    @property
    def validation_targets(self):
        return VALIDATION_TARGETS

    @property
    def ready_for_live(self):
        left,top,right,bottom = self.coverage_bounds
        covers_checks = all(left <= x <= right and top <= y <= bottom for x,y in self.validation_targets)
        # Live actions are restricted by the app to this verified screen ROI.
        # A small uncalibrated margin does not imply unsafe extrapolation;
        # literal full-screen coverage remains a separate diagnostic flag.
        return bool(covers_checks and self.validated and self.validation
                    and self.validation.passed and self.validation.frame_samples_checked)

    def predict(self, point):
        try:
            raw = np.asarray(point,dtype=float)
        except (TypeError,ValueError,OverflowError):
            return None
        if raw.shape != (2,) or not np.isfinite(raw).all():
            return None
        normalized = (raw-self._mean)/self._scale
        for triangle in self._triangles:
            result = triangle.map(normalized)
            if result is not None:
                if np.any(result < -1e-8) or np.any(result > 1+1e-8):
                    return None
                return tuple(float(v) for v in np.clip(result,0,1))
        return None

    def clear_validation(self):
        """Retesting immediately withdraws previous live-use approval."""
        self.validation = None
        self.validated = False
        self.rms_error = self.max_error = None
        self._validation_request = None

    def validate(self, fresh_raw, fresh_screen, *, screen_size, max_rms_pixels=30.,
                 max_point_pixels=40., frame_samples=None):
        # Starting a new check revokes earlier verification even if malformed
        # data raises; an old pass must not survive a failed new verification.
        self.clear_validation()
        if (not math.isfinite(max_rms_pixels) or not math.isfinite(max_point_pixels)
                or not 0 < max_rms_pixels <= 30 or not 0 < max_point_pixels <= 40):
            raise ValueError("驗證門檻須維持平均 30 px、單點 40 px 以內。")
        raw, target = _points(fresh_raw), _points(fresh_screen)
        size = np.asarray(screen_size,dtype=float)
        if raw.shape != target.shape or size.shape != (2,) or not np.isfinite(size).all() or np.any(size <= 0):
            raise ValueError("驗證點或螢幕尺寸不成對。")
        if not np.array_equal(size,np.array(self.screen_size)):
            raise ValueError("驗證螢幕尺寸與建模時不同，請用目前螢幕尺寸重建個人校正。")
        self._validation_request = {"raw_points":_pairs(raw),"screen_points":_pairs(target),
            "screen_size":tuple(float(v) for v in size),"max_rms_pixels":max_rms_pixels,
            "max_point_pixels":max_point_pixels,"frame_samples":None}
        predictions = tuple(self.predict(point) for point in raw)
        errors = tuple(None if point is None else float(np.linalg.norm(np.array(point)-goal)) for point,goal in zip(predictions,target))
        pixels = tuple(None if point is None else float(np.linalg.norm((np.array(point)-goal)*size)) for point,goal in zip(predictions,target))
        common = dict(sample_count=len(raw),target_points=_pairs(target),predicted_points=predictions,
                      point_errors=errors,point_pixel_errors=pixels,max_rms_pixels=max_rms_pixels,max_point_pixels=max_point_pixels)
        status, reason = "passed", "獨立驗證通過，仍請以目前高亮檔案確認瞄準。"
        if len(raw) < 9:
            status,reason = "insufficient_samples","需完成九個新畫面的驗證位置。"
        elif any(point is None for point in predictions):
            status,reason = "outside_support","新驗證位置超出安全映射範圍；請重新取得涵蓋邊緣的九點校正。"
        else:
            rms = float(np.sqrt(np.mean(np.array(errors)**2)))
            rms_px = float(np.sqrt(np.mean(np.array(pixels)**2)))
            common.update(rms_error=rms,max_error=float(max(errors)),rms_pixels=rms_px,max_pixels=float(max(pixels)))
            # Require all nine designated independent locations, not repeats of one node.
            matched = {int(np.argmin(np.linalg.norm(np.array(VALIDATION_TARGETS)-point,axis=1))) for point in target}
            if matched != set(range(9)) or any(np.min(np.linalg.norm(np.array(VALIDATION_TARGETS)-point,axis=1)) > .005 for point in target):
                status,reason = "insufficient_coverage","需驗證四個螢幕角落、四個格線中間位置及下方中央點。"
            elif rms_px > max_rms_pixels or max(pixels) > max_point_pixels:
                status,reason = "high_error",f"新影格驗證偏差仍過大：平均 {rms_px:.1f} px、最大 {max(pixels):.1f} px。"
        if frame_samples is not None:
            if len(frame_samples) != len(raw):
                raise ValueError("連續影格驗證資料未與驗證點成對。")
            counts, p95, serialized = [], [], []
            for frames,goal in zip(frame_samples,target):
                values = _points(frames)
                serialized.append(_pairs(values))
                counts.append(len(values))
                mapped = [self.predict(value) for value in values]
                p95.append(None if not mapped or any(v is None for v in mapped) else float(np.percentile([np.linalg.norm((np.array(v)-goal)*size) for v in mapped],95)))
            self._validation_request["frame_samples"] = serialized
            common.update(point_frame_counts=tuple(counts),point_p95_pixels=tuple(p95),frame_samples_checked=True)
            if status == "passed" and any(count < 6 for count in counts):
                status,reason = "insufficient_frames","每個驗證位置至少需要六筆新影格，請保持瞄準片刻。"
            elif status == "passed" and any(value is None or value > max_point_pixels for value in p95):
                status,reason = "unstable_frames","連續影格的偏差或抖動超過 40 px；請穩定姿勢或重取敏感的校正區域。"
        elif status == "passed":
            reason = "中位數位置通過；實際使用仍需連續新影格的抖動驗證。"
        report = ValidationReport(status,status == "passed",reason,**common)
        self.validation,self.validated = report,report.passed
        self.rms_error,self.max_error = report.rms_error,report.max_error
        return report

    def save(self,path):
        destination = Path(path).expanduser()
        destination.parent.mkdir(parents=True,exist_ok=True)
        payload = {"version":1,"calibration_model":self.calibration_model,"raw_points":self.raw_points,
                   "screen_points":self.screen_points,"screen_size":self.screen_size,
                   "validation_request":self._validation_request,
                   "validation":None if self.validation is None else self.validation.to_dict()}
        temporary = destination.with_name(destination.name+".tmp")
        temporary.write_text(json.dumps(payload,indent=2,allow_nan=False),encoding="utf-8")
        os.replace(temporary,destination)

    @classmethod
    def load(cls,path):
        try:
            payload = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
            if payload.get("version") != 1 or payload.get("calibration_model") != cls.calibration_model:
                raise ValueError("個人校正檔版本不支援。")
            model = cls.fit(payload["raw_points"],payload["screen_points"],screen_size=payload["screen_size"])
            request = payload.get("validation_request")
            if request is not None:
                model.validate(request["raw_points"],request["screen_points"],screen_size=request["screen_size"],
                               max_rms_pixels=request["max_rms_pixels"],max_point_pixels=request["max_point_pixels"],
                               frame_samples=request.get("frame_samples"))
            return model
        except (KeyError,TypeError,json.JSONDecodeError) as error:
            raise ValueError("個人校正檔不完整。") from error
