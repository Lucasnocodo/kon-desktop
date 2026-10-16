"""Local numerical calibration attempts; never store camera or audio media."""
from __future__ import annotations

from .paths import state_root as user_state_dir

from datetime import datetime, timezone
import json
from pathlib import Path
from uuid import uuid4


def save_attempt(root: Path, raw_points, screen_points, diagnostics, config, point_details=(),
                 *, personal=None, verification_details=()):
    """Keep every completed attempt and an atomic pointer to the most recent.

    The saved values explain failures without retaining any camera pixels,
    microphone audio, face landmarks or desktop file names.
    """
    created = datetime.now(timezone.utc)
    attempt_id = created.strftime("%Y%m%dT%H%M%S") + "-" + uuid4().hex[:8]
    payload = {
        "version": 1,
        "attempt_id": attempt_id,
        "created_at": created.isoformat(),
        "config": config,
        "raw_points": [list(point) for point in raw_points],
        "screen_points": [list(point) for point in screen_points],
        "point_details": list(point_details),
        "fit": {
            "status": diagnostics.status.value,
            "reason": diagnostics.reason,
            "rms_error": diagnostics.rms_error,
            "max_error": diagnostics.max_error,
            "condition_number": diagnostics.condition_number,
            "predicted_points": [list(point) for point in diagnostics.predicted_points],
            "point_errors": list(diagnostics.point_errors),
            "suggested_point_index": diagnostics.suggested_point_index,
            "raw_span": list(diagnostics.raw_span),
            "design_rank": diagnostics.design_rank,
            "leave_one_out_errors": list(diagnostics.leave_one_out_errors),
            "limits": {"rms_error": .08, "max_point_error": .16},
        },
    }
    if personal is not None:
        payload["fit"]["purpose"] = "reference_affine_not_acceptance"
        payload["personal_mapping"] = {
            "calibration_model": personal.calibration_model,
            "adopted_points": personal.sample_count,
            "coverage_bounds": personal.coverage_bounds,
            "full_screen_coverage": personal.full_screen_coverage,
            "coverage_reason": personal.coverage_reason,
            "max_noise_gain_pixels_per_mm": personal.max_noise_gain_pixels_per_mm,
            "ready_for_live": personal.ready_for_live,
            "validation": None if personal.validation is None else personal.validation.to_dict(),
            "verification_details": list(verification_details),
        }
    encoded = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False)
    folder = user_state_dir(root) / 'calibration-attempts'
    folder.mkdir(parents=True, exist_ok=True)
    destination = folder / f"{attempt_id}.json"
    destination.write_text(encoded, encoding="utf-8")
    last = user_state_dir(root) / 'calibration-last-attempt.json'
    temporary = last.with_suffix(".tmp")
    temporary.write_text(encoded, encoding="utf-8")
    temporary.replace(last)
    return destination


def load_last_attempt(root: Path, config):
    """Recover same-camera numerical samples for a targeted retry after restart."""
    try:
        payload = json.loads((user_state_dir(root) / 'calibration-last-attempt.json').read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or payload.get("version") != 1 or payload.get("config") != config:
            return None
        return payload
    except (OSError, ValueError, TypeError):
        return None
