"""Explain an accepted or rejected affine calibration without changing its limits.

Residuals are measured in normalized screen coordinates. A large residual
does not prove that a particular user sample is wrong: depth scale, head
position and estimated aperture location can produce systematic errors.
The suggested point is the largest residual, suitable for a targeted retry.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import math
from typing import Sequence

import numpy as np

from .geometry import ScreenCalibration

XY = tuple[float, float]


class FitStatus(str, Enum):
    PASSED = "passed"
    HIGH_RESIDUAL = "high_residual"
    INSUFFICIENT_SAMPLES = "insufficient_samples"
    INVALID_SAMPLES = "invalid_samples"
    SCREEN_COVERAGE = "screen_coverage"
    INSUFFICIENT_SPREAD = "insufficient_spread"
    COLLINEAR_RAYS = "collinear_rays"
    DEGENERATE_MAPPING = "degenerate_mapping"


@dataclass(frozen=True)
class FitDiagnostics:
    status: FitStatus
    reason: str
    sample_count: int = 0
    rms_error: float | None = None
    max_error: float | None = None
    condition_number: float | None = None
    predicted_points: tuple[XY, ...] = ()
    target_points: tuple[XY, ...] = ()
    residual_vectors: tuple[XY, ...] = ()
    point_errors: tuple[float, ...] = ()
    suggested_point_index: int | None = None
    raw_span: tuple[float, ...] = ()
    design_rank: int | None = None
    leave_one_out_errors: tuple[float | None, ...] = ()
    max_rms_error: float = .08
    max_point_error: float = .16
    calibration: ScreenCalibration | None = field(default=None, repr=False, compare=False)

    @property
    def passed(self) -> bool:
        return self.status is FitStatus.PASSED and self.calibration is not None

    @property
    def predictions(self) -> tuple[XY, ...]:
        return self.predicted_points

    @property
    def per_point_errors(self) -> tuple[float, ...]:
        return self.point_errors

    @property
    def residuals(self) -> tuple[XY, ...]:
        """Prediction minus target; draw an arrow from target to prediction."""
        return self.residual_vectors

    def to_dict(self) -> dict:
        """JSON-safe report; never serialize a rejected mapping as a live model."""
        return {
            "status": self.status.value, "reason": self.reason, "passed": self.passed,
            "sample_count": self.sample_count, "rms_error": self.rms_error,
            "max_error": self.max_error, "condition_number": self.condition_number,
            "predicted_points": self.predicted_points, "target_points": self.target_points,
            "residual_vectors": self.residual_vectors, "point_errors": self.point_errors,
            "suggested_point_index": self.suggested_point_index,
            "raw_span": self.raw_span, "design_rank": self.design_rank,
            "leave_one_out_errors": self.leave_one_out_errors,
            "max_rms_error": self.max_rms_error, "max_point_error": self.max_point_error,
        }


def _xy_tuples(points: np.ndarray) -> tuple[XY, ...]:
    return tuple((float(point[0]), float(point[1])) for point in points)


def analyze_calibration(
    raw_points: Sequence[Sequence[float]], screen_points: Sequence[Sequence[float]],
    *, max_rms_error: float = .08, max_point_error: float = .16,
) -> FitDiagnostics:
    """Compute the same normalized affine fit, retaining failed-fit residuals.

    Only a fit that also passes ScreenCalibration.fit at the unchanged
    supplied limits returns a calibration object. Leave-one-out numbers
    predict each omitted sample using the other samples; they are separate
    from training residuals and do not affect acceptance or remove points.
    """
    if (not math.isfinite(max_rms_error) or not math.isfinite(max_point_error)
            or max_rms_error <= 0 or max_point_error <= 0):
        raise ValueError("Calibration error limits must be finite and positive")
    common = {"max_rms_error": max_rms_error, "max_point_error": max_point_error}
    try:
        raw, target = np.asarray(raw_points, dtype=float), np.asarray(screen_points, dtype=float)
    except (TypeError, ValueError, OverflowError):
        return FitDiagnostics(FitStatus.INVALID_SAMPLES, "校正資料格式無效，請重新取得瞄準樣本。", **common)
    if raw.size == target.size == 0:
        return FitDiagnostics(FitStatus.INSUFFICIENT_SAMPLES, "尚未取得校正點，請先記錄螢幕上的標記。", **common)
    count = len(raw) if raw.ndim > 0 else 0
    common["sample_count"] = count
    if raw.ndim != 2 or raw.shape[1:] != (2,) or target.shape != raw.shape:
        return FitDiagnostics(FitStatus.INVALID_SAMPLES, "瞄準線與螢幕標記資料不成對，請重新記錄。", **common)
    if not np.isfinite(raw).all() or not np.isfinite(target).all():
        return FitDiagnostics(FitStatus.INVALID_SAMPLES, "校正包含無效的瞄準位置，請等待眼睛與手孔追蹤恢復。", **common)
    common["target_points"] = _xy_tuples(target)
    with np.errstate(over="ignore", invalid="ignore"):
        span = np.ptp(raw, axis=0)
    if not np.isfinite(span).all():
        return FitDiagnostics(FitStatus.INVALID_SAMPLES, "瞄準位置超出合理數值範圍，請重新取得追蹤資料。", **common)
    common["raw_span"] = tuple(float(value) for value in span)
    if count < 5:
        return FitDiagnostics(FitStatus.INSUFFICIENT_SAMPLES, "校正至少需要五個分散位置，請繼續記錄標記。", **common)
    if np.any(target < 0) or np.any(target > 1) or np.any(np.ptp(target, axis=0) < .35):
        return FitDiagnostics(FitStatus.SCREEN_COVERAGE, "標記未涵蓋螢幕的上下與左右範圍，請重新完成分散的校正點。", **common)
    with np.errstate(over="ignore", invalid="ignore"):
        mean, scale = raw.mean(axis=0), raw.std(axis=0)
    if not np.isfinite(mean).all() or not np.isfinite(scale).all():
        common["raw_span"] = ()
        return FitDiagnostics(FitStatus.INVALID_SAMPLES, "瞄準位置超出合理數值範圍，請重新取得追蹤資料。", **common)
    if np.any(scale < 1e-5):
        return FitDiagnostics(FitStatus.INSUFFICIENT_SPREAD,
                              "各標記的瞄準線幾乎相同；請確認鏡頭仍在更新，並穿過手圈瞄準每個新圓點。", **common)
    design = np.column_stack(((raw - mean) / scale, np.ones(count)))
    condition = float(np.linalg.cond(design))
    rank = int(np.linalg.matrix_rank(design))
    common["condition_number"] = condition if math.isfinite(condition) else None
    common["design_rank"] = rank
    if not math.isfinite(condition) or condition > 1000 or rank != 3:
        return FitDiagnostics(FitStatus.COLLINEAR_RAYS,
                              "瞄準樣本集中在一條線上，無法區分上下與左右；請重新記錄不同方向的標記。", **common)
    coefficients, _, fit_rank, _ = np.linalg.lstsq(design, target, rcond=None)
    if fit_rank != 3 or abs(np.linalg.det(coefficients[:2])) < 1e-6:
        return FitDiagnostics(FitStatus.DEGENERATE_MAPPING,
                              "瞄準線與標記無法形成可用對應；請確認每次記錄的是當前圓點，且眼睛與手孔定位一致。", **common)

    prediction = design @ coefficients
    residual = prediction - target
    errors = np.linalg.norm(residual, axis=1)
    rms, worst = float(np.sqrt(np.mean(errors ** 2))), float(errors.max())
    common.update(rms_error=rms, max_error=worst, predicted_points=_xy_tuples(prediction),
                  residual_vectors=_xy_tuples(residual), point_errors=tuple(float(error) for error in errors))
    held_out: list[float | None] = []
    if count >= 6:
        for index in range(count):
            keep = np.arange(count) != index
            reduced = design[keep]
            if np.linalg.matrix_rank(reduced) != 3 or np.linalg.cond(reduced) > 1000:
                held_out.append(None)
                continue
            loo_coefficients, _, _, _ = np.linalg.lstsq(reduced, target[keep], rcond=None)
            held_out.append(float(np.linalg.norm(design[index] @ loo_coefficients - target[index])))
    common["leave_one_out_errors"] = tuple(held_out)
    if rms > max_rms_error or worst > max_point_error:
        index = int(np.argmax(errors))
        return FitDiagnostics(
            FitStatus.HIGH_RESIDUAL,
            f"校正平均偏差 {rms * 100:.1f}%、最大偏差 {worst * 100:.1f}%，仍超出可用範圍。"
            f"第 {index + 1} 點偏差最大，可先重取這一點；若仍失敗，需檢查眼睛／手孔深度與固定坐姿。",
            suggested_point_index=index, **common,
        )
    # Trusted acceptance path retains all the geometry module's safeguards.
    calibration = ScreenCalibration.fit(raw, target, max_rms_error=max_rms_error, max_point_error=max_point_error)
    return FitDiagnostics(FitStatus.PASSED,
                          f"校正通過：平均偏差 {rms * 100:.1f}%、最大偏差 {worst * 100:.1f}%。請先在演練確認瞄準位置。",
                          calibration=calibration, **common)
