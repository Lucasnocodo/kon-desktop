"""Measured stereo geometry for KON's two-camera eye-through-hand aiming.

All coordinates are UNMIRRORED. Intrinsics refer to the exact captured image
size, with no resize/crop guesses. Stereo extrinsics map primary camera points
into the secondary camera: ``X_secondary = rotation @ X_primary + translation``.
Distances, checkerboard square sizes, and triangulated coordinates use meters.
Read timestamps can reject stale frame pairs, but cannot prove exposure sync.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Sequence

import cv2
import numpy as np


MIN_CALIBRATION_PAIRS = 15
MAX_INTRINSIC_RMS_PX = 1.5
MAX_STEREO_RMS_PX = 2.0
_RMS_QUALITY_ERROR = "校正重投影誤差過大；請重新拍攝清晰、同步且角度多樣的棋盤"


class CalibrationQualityError(ValueError):
    """A rejected calibration with a serializable report, never a usable rig."""

    def __init__(self, message, diagnostics):
        super().__init__(message)
        self.diagnostics = json.loads(json.dumps(diagnostics, allow_nan=False))


class TriangulationQualityError(ValueError):
    """A failed depth gate with replayable pixel observations and numeric evidence."""

    def __init__(self, message, diagnostics):
        super().__init__(message)
        self.diagnostics = json.loads(json.dumps(diagnostics, allow_nan=False))


def _array(value, shape, label):
    try:
        result = np.array(value, dtype=np.float64, copy=True)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} 必須是有限數值") from exc
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError(f"{label} 的尺寸或數值無效")
    result.setflags(write=False)
    return result


def _size(value):
    if (not isinstance(value, (tuple, list)) or len(value) != 2
            or any(isinstance(item, bool) or not isinstance(item, (int, np.integer))
                   or item <= 0 for item in value)):
        raise ValueError("影像尺寸必須是正整數 (寬, 高)")
    return tuple(int(item) for item in value)


def _positive(value, label, *, allow_zero=False):
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} 無效") from exc
    if isinstance(value, bool) or not math.isfinite(result) or result < 0 or (not allow_zero and result == 0):
        raise ValueError(f"{label} 無效")
    return result


@dataclass(frozen=True)
class CameraCalibration:
    image_size: tuple[int, int]
    matrix: np.ndarray
    distortion: np.ndarray

    def __post_init__(self):
        size = _size(self.image_size)
        matrix = _array(self.matrix, (3, 3), "鏡頭內參")
        try:
            distortion = np.asarray(self.distortion, dtype=np.float64).reshape(-1)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("鏡頭畸變參數無效") from exc
        if distortion.size not in (4, 5, 8, 12, 14):
            raise ValueError("鏡頭畸變參數數量無效")
        distortion = _array(distortion, (distortion.size,), "鏡頭畸變參數")
        if (matrix[0, 0] <= 0 or matrix[1, 1] <= 0
                or not np.allclose(matrix[2], (0, 0, 1), atol=1e-9)
                or abs(matrix[0, 1]) > 1e-9 or abs(matrix[1, 0]) > 1e-9):
            raise ValueError("鏡頭內參不是有效的 OpenCV 針孔模型")
        if (not .1 * size[0] <= matrix[0, 0] <= 10 * size[0]
                or not .1 * size[1] <= matrix[1, 1] <= 10 * size[0]):
            raise ValueError("鏡頭焦距超出合理範圍；請加入更多棋盤傾斜角度重新校正")
        if not (-size[0] <= matrix[0, 2] <= size[0] * 2
                and -size[1] <= matrix[1, 2] <= size[1] * 2):
            raise ValueError("鏡頭主點超出合理範圍")
        object.__setattr__(self, "image_size", size)
        object.__setattr__(self, "matrix", matrix)
        object.__setattr__(self, "distortion", distortion)

    def to_dict(self):
        return {"image_size": list(self.image_size), "matrix": self.matrix.tolist(),
                "distortion": self.distortion.tolist()}

    @classmethod
    def from_dict(cls, data):
        try:
            return cls(data["image_size"], data["matrix"], data["distortion"])
        except (KeyError, TypeError) as exc:
            raise ValueError("鏡頭校正檔案格式無效") from exc


@dataclass(frozen=True)
class StereoCalibration:
    primary: CameraCalibration
    secondary: CameraCalibration
    rotation: np.ndarray
    translation: np.ndarray
    primary_source: int = 0
    secondary_source: int = 1
    primary_id: str = ""
    secondary_id: str = ""
    rms: float = 0.0
    primary_rms: float = 0.0
    secondary_rms: float = 0.0
    sample_count: int = 0

    def __post_init__(self):
        if not isinstance(self.primary, CameraCalibration) or not isinstance(self.secondary, CameraCalibration):
            raise ValueError("缺少兩台鏡頭的內參")
        rotation = _array(self.rotation, (3, 3), "雙鏡頭旋轉")
        try:
            translation = np.asarray(self.translation, dtype=np.float64).reshape(3)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("雙鏡頭平移無效") from exc
        translation = _array(translation, (3,), "雙鏡頭平移")
        if (not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5)
                or not math.isclose(float(np.linalg.det(rotation)), 1.0, abs_tol=1e-5)):
            raise ValueError("雙鏡頭旋轉不是正交旋轉矩陣")
        if not .04 <= np.linalg.norm(translation) <= 2.0:
            raise ValueError("兩鏡頭間距須介於 4 公分與 2 公尺；請確認棋盤格單位是公尺")
        for name in ("primary_source", "secondary_source"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 0:
                raise ValueError("攝影機索引無效")
            object.__setattr__(self, name, int(value))
        if self.primary_source == self.secondary_source:
            raise ValueError("雙鏡頭必須使用不同攝影機")
        for name in ("primary_id", "secondary_id"):
            if not isinstance(getattr(self, name), str):
                raise ValueError("攝影機識別資料無效")
        for name in ("rms", "primary_rms", "secondary_rms"):
            object.__setattr__(self, name, _positive(getattr(self, name), name, allow_zero=True))
        if self.rms > MAX_STEREO_RMS_PX or max(self.primary_rms, self.secondary_rms) > MAX_INTRINSIC_RMS_PX:
            raise ValueError(_RMS_QUALITY_ERROR)
        if isinstance(self.sample_count, bool) or not isinstance(self.sample_count, int) or self.sample_count < 0:
            raise ValueError("校正樣本數無效")
        object.__setattr__(self, "rotation", rotation)
        object.__setattr__(self, "translation", translation)

    @property
    def baseline_m(self):
        return float(np.linalg.norm(self.translation))

    def _geometry_dict(self):
        return {"primary": self.primary.to_dict(), "secondary": self.secondary.to_dict(),
                "rotation": self.rotation.tolist(), "translation": self.translation.tolist(),
                "primary_source": self.primary_source, "secondary_source": self.secondary_source,
                "primary_id": self.primary_id, "secondary_id": self.secondary_id}

    @property
    def calibration_id(self):
        payload = json.dumps(self._geometry_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]

    def validate_sizes(self, primary_size, secondary_size):
        """Reject changed capture modes; crop and FOV changes need recalibration."""
        if (_size(primary_size) != self.primary.image_size
                or _size(secondary_size) != self.secondary.image_size):
            raise ValueError("目前影像尺寸與雙鏡頭校正不符，請恢復原解析度或重新校正")

    def to_dict(self):
        return dict(self._geometry_dict(), schema_version=1, units="meters", rms=self.rms,
                    primary_rms=self.primary_rms, secondary_rms=self.secondary_rms,
                    sample_count=self.sample_count, calibration_id=self.calibration_id)

    @classmethod
    def from_dict(cls, data):
        try:
            if data["schema_version"] != 1 or data["units"] != "meters":
                raise ValueError("不支援的雙鏡頭校正版本或單位")
            result = cls(CameraCalibration.from_dict(data["primary"]),
                         CameraCalibration.from_dict(data["secondary"]), data["rotation"],
                         data["translation"], data["primary_source"], data["secondary_source"],
                         data.get("primary_id", ""), data.get("secondary_id", ""),
                         data["rms"], data["primary_rms"], data["secondary_rms"], data["sample_count"])
            if data.get("calibration_id") != result.calibration_id:
                raise ValueError("雙鏡頭校正資料指紋不符")
            return result
        except (KeyError, TypeError) as exc:
            raise ValueError("雙鏡頭校正檔案格式無效") from exc

    def save(self, path: Path | str):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # A unique temporary path makes simultaneous profile saves safe.
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                             prefix=path.name + ".", suffix=".tmp", delete=False) as stream:
                temporary = Path(stream.name)
                json.dump(self.to_dict(), stream, ensure_ascii=False, indent=2, allow_nan=False)
            os.replace(temporary, path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    @classmethod
    def load(cls, path: Path | str):
        try:
            return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
        except (json.JSONDecodeError, UnicodeError) as exc:
            raise ValueError("雙鏡頭校正檔案無法解析") from exc


def checkerboard_object_points(columns=9, rows=6, square_m=.025):
    """Return row-major INNER corners; measure a printed square edge in meters."""
    if (isinstance(columns, bool) or isinstance(rows, bool)
            or not isinstance(columns, int) or not isinstance(rows, int)
            or columns < 3 or rows < 3 or columns > 30 or rows > 30):
        raise ValueError("棋盤內角點數須介於 3 與 30")
    square_m = _positive(square_m, "棋盤格邊長")
    if not .005 <= square_m <= .2:
        raise ValueError("棋盤格邊長須介於 5 mm 與 200 mm，單位為公尺")
    points = np.zeros((columns * rows, 3), np.float32)
    points[:, :2] = np.mgrid[0:columns, 0:rows].T.reshape(-1, 2) * square_m
    return points


def detect_checkerboard(image, columns=9, rows=6):
    """Find subpixel inner corners in an original, unmirrored image."""
    checkerboard_object_points(columns, rows)
    image = np.asarray(image)
    if image.dtype != np.uint8 or image.ndim not in (2, 3) or min(image.shape[:2]) < 32:
        raise ValueError("棋盤影像須是有效的 uint8 灰階或 BGR 影像")
    if image.ndim == 3:
        if image.shape[2] not in (3, 4):
            raise ValueError("棋盤影像色彩通道無效")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY if image.shape[2] == 3 else cv2.COLOR_BGRA2GRAY)
    found, corners = cv2.findChessboardCornersSB(image, (columns, rows),
                                               flags=cv2.CALIB_CB_NORMALIZE_IMAGE)
    return corners.reshape(-1, 2).astype(np.float64) if found else None


def _pixels(value, label, size):
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} 影像座標無效") from exc
    if (result.ndim != 2 or result.shape[1] != 2 or len(result) == 0
            or not np.isfinite(result).all()):
        raise ValueError(f"{label} 須是有限的 Nx2 影像座標")
    if (np.any(result < 0) or np.any(result[:, 0] >= size[0])
            or np.any(result[:, 1] >= size[1])):
        raise ValueError(f"{label} 的特徵點超出拍攝影像")
    return result


def _coverage(views, size):
    normalized = [view / np.asarray(size) for view in views]
    centers = np.array([view.mean(axis=0) for view in normalized])
    spans = np.array([np.ptp(view, axis=0) for view in normalized])
    if np.any(np.ptp(centers, axis=0) < .12):
        raise ValueError("棋盤位置太集中，請在兩個畫面的上、下、左、右移動棋盤")
    if np.min(spans) < .045 or np.max(spans[:, 0]) < .2 or np.max(spans[:, 1]) < .15:
        raise ValueError("棋盤在影像中太小，請讓角點清楚並涵蓋更大的畫面範圍")
    areas = spans.prod(axis=1)
    if max(areas) / min(areas) < 1.3:
        raise ValueError("棋盤距離變化不足，請在多個距離拍攝")


def _report_number(value):
    value = float(value)
    return value if math.isfinite(value) else None


def _residual_stats(obj, observed, rotation, translation, matrix, distortion):
    projected, _ = cv2.projectPoints(obj, cv2.Rodrigues(rotation)[0], translation, matrix, distortion)
    residuals = np.linalg.norm(projected.reshape(-1, 2) - observed.reshape(-1, 2), axis=1)
    return {'rms': _report_number(np.sqrt(np.mean(residuals ** 2))),
            'max': _report_number(np.max(residuals))}


def _calibration_diagnostics(objects, primary, secondary, k1, d1, k2, d2,
                             primary_rvecs, primary_tvecs, secondary_rvecs, secondary_tvecs,
                             result, rms1, rms2):
    rms, _, _, _, _, rotation, translation = result[:7]
    joint_poses = len(result) == 12
    board_rvecs, board_tvecs = result[9:11] if joint_poses else (primary_rvecs, primary_tvecs)
    report = {
        'schema_version': 1, 'units': 'pixels', 'sample_count': len(objects),
        'primary_rms': _report_number(rms1), 'secondary_rms': _report_number(rms2),
        'stereo_rms': _report_number(rms), 'joint_board_poses': joint_poses,
        'limits': {'intrinsic_rms': MAX_INTRINSIC_RMS_PX, 'stereo_rms': MAX_STEREO_RMS_PX,
                   'view_rms': 3.0, 'corner_max': 8.0},
        'views': [], 'failures': [], 'failed_thresholds': [],
    }
    for index, (obj, first, second) in enumerate(zip(objects, primary, secondary)):
        mono_primary = _residual_stats(obj, first, cv2.Rodrigues(primary_rvecs[index])[0],
                                       primary_tvecs[index], k1, d1)
        mono_secondary = _residual_stats(obj, second, cv2.Rodrigues(secondary_rvecs[index])[0],
                                         secondary_tvecs[index], k2, d2)
        board_rotation = cv2.Rodrigues(board_rvecs[index])[0]
        board_translation = np.asarray(board_tvecs[index]).reshape(3)
        checked_primary = _residual_stats(obj, first, board_rotation, board_translation, k1, d1)
        checked_secondary = _residual_stats(obj, second, rotation @ board_rotation,
                                           rotation @ board_translation + translation.reshape(3), k2, d2)
        # Older Extended APIs expose joint RMS but not the jointly fitted poses,
        # so individual joint corner maxima cannot be recovered from that API.
        joint_primary, joint_secondary = checked_primary, checked_secondary
        if not joint_poses:
            per_view = np.asarray(result[-1]).reshape(len(objects), 2)
            joint_primary = {'rms': _report_number(per_view[index, 0]), 'max': None}
            joint_secondary = {'rms': _report_number(per_view[index, 1]), 'max': None}
        report['views'].append({
            'sample_id': index + 1, 'mono_primary': mono_primary, 'mono_secondary': mono_secondary,
            'joint_primary': joint_primary, 'joint_secondary': joint_secondary,
            'checked_primary': checked_primary, 'checked_secondary': checked_secondary,
        })
    return report


def _rms_summary(report):
    return '；'.join(f"{label} RMS {report[key]:.3f} px" for key, label in
                    (('primary_rms', '主鏡頭'), ('secondary_rms', '副鏡頭'), ('stereo_rms', '雙鏡頭')))


def solve_stereo_calibration(
    object_points_m, primary_points_px, secondary_points_px, primary_size, secondary_size,
    *, primary_source=0, secondary_source=1, primary_id="", secondary_id="",
):
    """Fit >=15 simultaneous diverse checkerboard views; reject poor fits.

    Each object point list must correspond to the SAME physical inner corner
    in both views. Hold the board still when collecting a pair: software read
    timestamps do not establish that wireless camera exposures were simultaneous.
    Different resolutions are supported because stereoCalibrate fixes the
    separately measured intrinsics; its imageSize is only an initialization hint.
    """
    primary_size, secondary_size = _size(primary_size), _size(secondary_size)
    if not (len(object_points_m) == len(primary_points_px) == len(secondary_points_px)
            and len(object_points_m) >= MIN_CALIBRATION_PAIRS):
        raise ValueError(f"至少需要 {MIN_CALIBRATION_PAIRS} 組同時可見、姿態多樣的棋盤配對")
    objects, primary, secondary = [], [], []
    for obj, first, second in zip(object_points_m, primary_points_px, secondary_points_px):
        first = _pixels(first, "主鏡頭", primary_size)
        second = _pixels(second, "第二鏡頭", secondary_size)
        obj = _array(obj, (len(first), 3), "棋盤公尺座標")
        if len(first) != len(second) or len(first) < 12 or np.linalg.matrix_rank(obj - obj.mean(axis=0)) != 2:
            raise ValueError("棋盤配對角點不完整或不是平面棋盤")
        if not .02 <= np.linalg.norm(np.ptp(obj, axis=0)) <= 2:
            raise ValueError("棋盤尺寸不合理；請使用公尺單位")
        objects.append(obj.astype(np.float32))
        primary.append(first.astype(np.float32).reshape(-1, 1, 2))
        secondary.append(second.astype(np.float32).reshape(-1, 1, 2))
    _coverage([p.reshape(-1, 2) for p in primary], primary_size)
    _coverage([p.reshape(-1, 2) for p in secondary], secondary_size)
    try:
        rms1, k1, d1, rvecs, tvecs = cv2.calibrateCamera(objects, primary, primary_size, None, None)
        rms2, k2, d2, secondary_rvecs, secondary_tvecs = cv2.calibrateCamera(
            objects, secondary, secondary_size, None, None)
        normals = np.array([cv2.Rodrigues(rvec)[0][:, 2] for rvec in rvecs])
        # Translating a frontoparallel board never constrains focal length well.
        if np.max(np.linalg.norm(normals - normals.mean(axis=0), axis=1)) < math.sin(math.radians(8)):
            raise ValueError("棋盤傾斜角度不足，請加入左右轉向及上下傾斜的姿態")
        result = cv2.stereoCalibrateExtended(objects, primary, secondary, k1, d1, k2, d2,
                                             primary_size, None, None,
                                             flags=cv2.CALIB_FIX_INTRINSIC,
                                             criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER,
                                                       100, 1e-7))
        rms, k1, d1, k2, d2, rotation, translation = result[:7]
        diagnostics = _calibration_diagnostics(objects, primary, secondary, k1, d1, k2, d2,
                                               rvecs, tvecs, secondary_rvecs, secondary_tvecs,
                                               result, rms1, rms2)
        try:
            calibration = StereoCalibration(CameraCalibration(primary_size, k1, d1),
                                             CameraCalibration(secondary_size, k2, d2),
                                             rotation, translation, primary_source, secondary_source,
                                             primary_id, secondary_id, float(rms), float(rms1),
                                             float(rms2), len(objects))
        except ValueError as error:
            if str(error) != _RMS_QUALITY_ERROR:
                raise
            failures = []
            for key, label, limit in (('primary_rms', '主鏡頭', MAX_INTRINSIC_RMS_PX),
                                      ('secondary_rms', '副鏡頭', MAX_INTRINSIC_RMS_PX),
                                      ('stereo_rms', '雙鏡頭', MAX_STEREO_RMS_PX)):
                if diagnostics[key] > limit:
                    failures.append({'metric': key, 'value': diagnostics[key], 'limit': limit,
                                     'label': label})
            diagnostics.update(failure_stage='overall_rms', failures=failures,
                               failed_thresholds=[failure['metric'] for failure in failures])
            failed = '；'.join(f"{failure['label']} RMS 超過 {failure['limit']:.3f} px"
                              for failure in failures)
            raise CalibrationQualityError(
                f"校正重投影誤差過大：{failed}。{_rms_summary(diagnostics)}", diagnostics) from error
        # The stereo fit jointly optimizes each board pose. Reusing the independent
        # primary-camera poses can falsely reject a good stereo fit, especially
        # when its small/noisy board is magnified in the higher-resolution camera.
        # Older OpenCV releases expose only per-view RMS: keep their conservative
        # pose check rather than silently dropping the individual-corner guard.
        # Guard a single bad/moving board pair hidden inside an aggregate RMS,
        # including bad primary observations which the secondary fit can outweigh.
        for view in diagnostics['views']:
            for camera, label in (('primary', '主鏡頭'), ('secondary', '副鏡頭')):
                stats = view[f'checked_{camera}']
                failures = [{'metric': metric, 'camera': camera, 'sample_id': view['sample_id'],
                             'value': stats[key], 'limit': limit}
                            for key, metric, limit in (('max', 'corner_max', 8), ('rms', 'view_rms', 3))
                            if stats[key] is not None and stats[key] > limit]
                if failures:
                    diagnostics.update(failure_stage='per_view', failures=failures,
                                       failed_thresholds=[failure['metric'] for failure in failures])
                    failed = '；'.join(f"{'最大角點' if failure['metric'] == 'corner_max' else '每組 RMS'} "
                                       f"{failure['value']:.3f} px > {failure['limit']:.3f} px"
                                       for failure in failures)
                    raise CalibrationQualityError(
                        f"部分棋盤配對誤差過大：第 {view['sample_id']} 組{label} {failed}。"
                        f"{_rms_summary(diagnostics)}", diagnostics)
        return calibration
    except cv2.error as exc:
        raise ValueError("OpenCV 無法完成雙鏡頭校正，請重新拍攝棋盤") from exc


def triangulate_points(
    calibration: StereoCalibration, primary_pixels, secondary_pixels, *,
    max_reprojection_error_px=3.0, min_angle_degrees=.5, min_depth_m=.05, max_depth_m=4.0,
    diagnostics=None,
):
    """Return Nx3 primary-camera meters, or raise if ANY pair is unreliable.

    An optional dict receives the same JSON-safe evidence on success or failure.
    Point indices are zero-based and preserve the caller's landmark ordering.
    Unreached measurements and nonfinite diagnostic numbers are represented by
    None; rejected depth is never returned as a usable point array.
    """
    def numbers(value):
        if isinstance(value, (list, tuple)):
            return [numbers(item) for item in value]
        number = float(value)
        return number if math.isfinite(number) else None

    def pixels_for_report(value):
        try:
            value = np.asarray(value, dtype=float)
            return numbers(value.tolist()) if value.ndim == 2 and value.shape[1] == 2 else None
        except (TypeError, ValueError, OverflowError):
            return None

    def limit_for_report(value):
        try:
            return numbers(value)
        except (TypeError, ValueError, OverflowError):
            return None

    report = diagnostics if diagnostics is not None else {}
    report.clear()
    report.update(
        schema_version=1, kind='kon.stereo_triangulation', calibration_id=calibration.calibration_id,
        primary_source_size=list(calibration.primary.image_size),
        secondary_source_size=list(calibration.secondary.image_size),
        primary_pixels=pixels_for_report(primary_pixels), secondary_pixels=pixels_for_report(secondary_pixels),
        limits={'max_reprojection_error_px': limit_for_report(max_reprojection_error_px),
                'min_angle_degrees': limit_for_report(min_angle_degrees),
                'min_depth_m': limit_for_report(min_depth_m), 'max_depth_m': limit_for_report(max_depth_m)},
        point_count=None, failing_gate=None, failed_point_indices=[],
        primary_depth_m=None, secondary_depth_m=None, parallax_degrees=None,
        primary_reprojection_error_px=None, secondary_reprojection_error_px=None,
        worst_point_index=None, worst_camera=None, worst_reprojection_error_px=None,
    )
    gate = 'input'
    try:
        first = _pixels(primary_pixels, "主鏡頭", calibration.primary.image_size)
        second = _pixels(secondary_pixels, "第二鏡頭", calibration.secondary.image_size)
        report['point_count'] = len(first)
        gate = 'point_count'
        if first.shape != second.shape:
            raise ValueError("雙鏡頭特徵點數量不一致")
        gate = 'thresholds'
        maximum_error = _positive(max_reprojection_error_px, "重投影誤差門檻")
        minimum_angle = _positive(min_angle_degrees, "三角定位夾角門檻")
        minimum_depth = _positive(min_depth_m, "最小深度")
        maximum_depth = _positive(max_depth_m, "最大深度")
        if minimum_angle >= 90 or minimum_depth >= maximum_depth:
            raise ValueError("三角定位品質門檻無效")
        gate = 'undistortion'
        norm1 = cv2.undistortPoints(first.reshape(-1, 1, 2), calibration.primary.matrix,
                                   calibration.primary.distortion).reshape(-1, 2)
        norm2 = cv2.undistortPoints(second.reshape(-1, 1, 2), calibration.secondary.matrix,
                                   calibration.secondary.distortion).reshape(-1, 2)
        projection1 = np.column_stack((np.eye(3), np.zeros(3)))
        projection2 = np.column_stack((calibration.rotation, calibration.translation))
        gate = 'homogeneous'
        homogeneous = cv2.triangulatePoints(projection1, projection2, norm1.T, norm2.T)
        if np.any(np.abs(homogeneous[3]) < 1e-10):
            report['failed_point_indices'] = np.flatnonzero(np.abs(homogeneous[3]) < 1e-10).tolist()
            raise ValueError("雙鏡頭視差不足，無法可靠估計深度")
        points = (homogeneous[:3] / homogeneous[3]).T
        secondary_points = points @ calibration.rotation.T + calibration.translation
        report['primary_depth_m'] = numbers(points[:, 2].tolist())
        report['secondary_depth_m'] = numbers(secondary_points[:, 2].tolist())
        gate = 'depth'
        if (not np.isfinite(points).all()
                or np.any(points[:, 2] < minimum_depth) or np.any(points[:, 2] > maximum_depth)
                or np.any(secondary_points[:, 2] < minimum_depth)
                or np.any(secondary_points[:, 2] > maximum_depth)):
            invalid = (~np.isfinite(points).all(axis=1) | (points[:, 2] < minimum_depth)
                       | (points[:, 2] > maximum_depth) | (secondary_points[:, 2] < minimum_depth)
                       | (secondary_points[:, 2] > maximum_depth))
            report['failed_point_indices'] = np.flatnonzero(invalid).tolist()
            raise ValueError("三維點位於鏡頭後方或超出可靠深度範圍")
        rays1 = np.column_stack((norm1, np.ones(len(norm1))))
        rays2 = np.column_stack((norm2, np.ones(len(norm2)))) @ calibration.rotation
        rays1 /= np.linalg.norm(rays1, axis=1)[:, None]
        rays2 /= np.linalg.norm(rays2, axis=1)[:, None]
        angles = np.degrees(np.arccos(np.clip(np.sum(rays1 * rays2, axis=1), -1, 1)))
        report['parallax_degrees'] = numbers(angles.tolist())
        gate = 'parallax'
        if np.any(angles < minimum_angle):
            report['failed_point_indices'] = np.flatnonzero(angles < minimum_angle).tolist()
            raise ValueError("雙鏡頭三角定位夾角太小，請調整鏡頭間距或靠近拍攝")
        gate = 'reprojection'
        projected1, _ = cv2.projectPoints(points, np.zeros(3), np.zeros(3),
                                         calibration.primary.matrix, calibration.primary.distortion)
        projected2, _ = cv2.projectPoints(points, cv2.Rodrigues(calibration.rotation)[0],
                                         calibration.translation, calibration.secondary.matrix,
                                         calibration.secondary.distortion)
        error1 = np.linalg.norm(projected1.reshape(-1, 2) - first, axis=1)
        error2 = np.linalg.norm(projected2.reshape(-1, 2) - second, axis=1)
        report['primary_reprojection_error_px'] = numbers(error1.tolist())
        report['secondary_reprojection_error_px'] = numbers(error2.tolist())
        errors = np.stack((error1, error2))
        worst_camera, worst_point = np.unravel_index(np.argmax(np.nan_to_num(errors, nan=-np.inf)), errors.shape)
        report.update(worst_point_index=int(worst_point), worst_camera=('primary', 'secondary')[worst_camera],
                      worst_reprojection_error_px=numbers(errors[worst_camera, worst_point]))
        if np.any(error1 > maximum_error) or np.any(error2 > maximum_error):
            report['failed_point_indices'] = np.flatnonzero((error1 > maximum_error) | (error2 > maximum_error)).tolist()
            raise ValueError("雙鏡頭特徵不一致或不同步，已暫停深度定位")
        return points
    except cv2.error as exc:
        report['failing_gate'] = gate
        raise TriangulationQualityError("OpenCV 無法三角定位這組特徵點", report) from exc
    except ValueError as exc:
        report['failing_gate'] = gate
        raise TriangulationQualityError(str(exc), report) from exc


@dataclass(frozen=True)
class ScreenPlane:
    """Physical screen plane in the PRIMARY camera's metric coordinate frame.

    right/down are as seen by a USER facing the display. In an unmirrored
    upright camera facing that user, display-right is camera-image-LEFT.
    Thus right/down/userward is a left-handed basis, and the userward front
    normal is ``-cross(right_unit, down_unit)``.
    """
    top_left_m: np.ndarray
    right_unit: np.ndarray
    down_unit: np.ndarray
    width_m: float
    height_m: float

    def __post_init__(self):
        top_left = _array(self.top_left_m, (3,), "螢幕左上角")
        right = _array(self.right_unit, (3,), "螢幕向右方向")
        down = _array(self.down_unit, (3,), "螢幕向下方向")
        if (not math.isclose(float(np.linalg.norm(right)), 1, abs_tol=1e-6)
                or not math.isclose(float(np.linalg.norm(down)), 1, abs_tol=1e-6)
                or abs(np.dot(right, down)) > 1e-6):
            raise ValueError("螢幕基底必須互相垂直且長度為一")
        width, height = _positive(self.width_m, "螢幕寬度"), _positive(self.height_m, "螢幕高度")
        if not (.1 <= width <= 3 and .1 <= height <= 2):
            raise ValueError("螢幕實體尺寸不合理，請使用公尺單位")
        object.__setattr__(self, "top_left_m", top_left)
        object.__setattr__(self, "right_unit", right)
        object.__setattr__(self, "down_unit", down)
        object.__setattr__(self, "width_m", width)
        object.__setattr__(self, "height_m", height)

    @classmethod
    def from_measurements(cls, width_m, height_m, camera_x_m, camera_y_m, camera_forward_m,
                          pitch_degrees=0, yaw_degrees=0, roll_degrees=0):
        """Lens center from display top-left, x USER-right/y down/z userward.

        All directions are as seen by the user facing the display. Positive
        pitch aims the camera DOWN; positive yaw aims toward display RIGHT;
        positive roll rotates the camera CLOCKWISE. Zero means upright,
        perpendicular to the display, facing the user. Combined angles apply
        pitch, then yaw, then roll, about these fixed display axes.

        Display right/down/userward is LEFT-handed, whereas OpenCV camera
        image-right/down/forward is right-handed. The camera-to-display map
        is Rz(roll) @ Ry(yaw) @ Rx(-pitch) @ diag(-1, 1, 1); the reflection is
        essential for off-center lens locations and tilted cameras.
        Re-measure after moving the display or primary camera.
        """
        center = _array((camera_x_m, camera_y_m, camera_forward_m), (3,), "鏡頭相對螢幕位置")
        angles = _array((pitch_degrees, yaw_degrees, roll_degrees), (3,), "鏡頭角度")
        if np.max(np.abs(center)) > 3 or not 0 <= center[2] <= .5 or np.max(np.abs(angles)) > 75:
            raise ValueError("鏡頭與螢幕的位置或角度超出桌面配置範圍")
        pitch, yaw, roll = np.radians(angles * [-1, 1, 1])
        cx, sx, cy, sy, cz, sz = (math.cos(pitch), math.sin(pitch), math.cos(yaw),
                                 math.sin(yaw), math.cos(roll), math.sin(roll))
        rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
        ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
        rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
        camera_to_screen = rz @ ry @ rx @ np.diag([-1., 1., 1.])
        return cls(camera_to_screen.T @ -center, camera_to_screen.T[:, 0],
                   camera_to_screen.T[:, 1], width_m, height_m)

    @property
    def front_unit(self):
        """Unit normal from the display toward its viewer, in camera axes."""
        normal = -np.cross(self.right_unit, self.down_unit)
        return normal / np.linalg.norm(normal)

    def to_dict(self):
        return {"top_left_m": self.top_left_m.tolist(), "right_unit": self.right_unit.tolist(),
                "down_unit": self.down_unit.tolist(), "width_m": self.width_m, "height_m": self.height_m}

    @property
    def plane_id(self):
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]

    def measurements(self):
        """Recover the keyword arguments accepted by ``from_measurements``."""
        camera_to_screen = np.stack((self.right_unit, self.down_unit, self.front_unit))
        center = -(camera_to_screen @ self.top_left_m)
        rotation = camera_to_screen @ np.diag([-1., 1., 1.])
        pitch = -math.atan2(rotation[2, 1], rotation[2, 2])
        yaw = math.asin(float(np.clip(-rotation[2, 0], -1, 1)))
        roll = math.atan2(rotation[1, 0], rotation[0, 0])
        return {"width_m": self.width_m, "height_m": self.height_m,
                "camera_x_m": float(center[0]), "camera_y_m": float(center[1]),
                "camera_forward_m": float(center[2]), "pitch_degrees": math.degrees(pitch),
                "yaw_degrees": math.degrees(yaw), "roll_degrees": math.degrees(roll)}

    @classmethod
    def from_dict(cls, data):
        try:
            return cls(data["top_left_m"], data["right_unit"], data["down_unit"],
                       data["width_m"], data["height_m"])
        except (KeyError, TypeError) as exc:
            raise ValueError("螢幕平面設定格式無效") from exc

    def ray_hit(self, eye, aperture):
        """Continue eye -> aperture to screen, returning screen-basis meters.

        Reject reversed/parallel rays and implausible distances. A hit may be
        outside the screen; the existing nine-point mapping handles user bias.
        """
        try:
            eye = _array(eye, (3,), "眼睛座標")
            aperture = _array(aperture, (3,), "手圈座標")
        except ValueError:
            return None
        normal = self.front_unit
        eye_distance = float(np.dot(eye - self.top_left_m, normal))
        aperture_distance = float(np.dot(aperture - self.top_left_m, normal))
        if not (.18 <= eye_distance <= 2.5 and .06 <= aperture_distance <= eye_distance - .035):
            return None
        direction = aperture - eye
        if not .04 <= np.linalg.norm(direction) <= 1.6:
            return None
        denominator = float(np.dot(direction, normal))
        if abs(denominator) < 1e-9:
            return None
        distance = -eye_distance / denominator
        if not 1 < distance <= 30:
            return None
        hit = eye + distance * direction - self.top_left_m
        return float(np.dot(hit, self.right_unit)), float(np.dot(hit, self.down_unit))
