from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

import numpy as np


@dataclass(frozen=True)
class AlignmentFit:
    classification: str
    latency_s: float
    max_abs_residual_s: float
    rms_residual_s: float
    missing_edge_index: int | None = None


def _as_float_array(values) -> np.ndarray:
    return np.asarray(values, dtype=float)


def _fit_alignment(reference: np.ndarray, observed: np.ndarray, classification: str, missing_edge_index: int | None = None) -> AlignmentFit:
    if reference.shape != observed.shape:
        raise ValueError("reference and observed must have the same shape")
    if observed.size == 0:
        return AlignmentFit(classification=classification, latency_s=0.0, max_abs_residual_s=0.0, rms_residual_s=0.0, missing_edge_index=missing_edge_index)

    residuals = observed - reference
    latency_s = float(np.median(residuals))
    centered = residuals - latency_s
    max_abs_residual_s = float(np.max(np.abs(centered))) if centered.size else 0.0
    rms_residual_s = float(sqrt(np.mean(np.square(centered)))) if centered.size else 0.0
    return AlignmentFit(
        classification=classification,
        latency_s=latency_s,
        max_abs_residual_s=max_abs_residual_s,
        rms_residual_s=rms_residual_s,
        missing_edge_index=missing_edge_index,
    )


def classify_missing_events(reference_times_s, observed_times_s, *, tolerance_s: float | None = None) -> dict[str, float | int | str | None]:
    reference = _as_float_array(reference_times_s)
    observed = _as_float_array(observed_times_s)

    if reference.ndim != 1 or observed.ndim != 1:
        raise ValueError("reference_times_s and observed_times_s must be 1-D")

    expected_count = int(reference.size)
    observed_count = int(observed.size)
    missing_count = expected_count - observed_count
    expected_span_s = float(reference[-1] - reference[0]) if expected_count >= 2 else 0.0
    observed_span_s = float(observed[-1] - observed[0]) if observed_count >= 2 else 0.0
    period_s = float(np.median(np.diff(reference))) if expected_count >= 2 else 0.0
    if tolerance_s is None:
        tolerance_s = max(period_s * 0.35, 1e-6)

    result: dict[str, float | int | str | None] = {
        "expected_count": expected_count,
        "observed_count": observed_count,
        "missing_count": missing_count,
        "expected_span_s": expected_span_s,
        "observed_span_s": observed_span_s,
        "span_error_s": observed_span_s - expected_span_s,
        "tolerance_s": float(tolerance_s),
        "classification": "unknown",
        "latency_s": None,
        "max_abs_residual_s": None,
        "rms_residual_s": None,
        "missing_edge_index": None,
        "missing_edge_number": None,
        "missing_edge_time_s": None,
    }

    if expected_count == 0:
        result["classification"] = "no_reference"
        return result
    if observed_count == 0:
        result["classification"] = "no_observed"
        return result
    if missing_count < 0:
        result["classification"] = "extra_observed"
        return result

    candidates: list[AlignmentFit] = []
    if missing_count == 0:
        candidates.append(_fit_alignment(reference, observed, "exact"))
    elif missing_count == 1:
        candidates.append(_fit_alignment(reference[1:], observed, "missing_first", missing_edge_index=0))
        candidates.append(_fit_alignment(reference[:-1], observed, "missing_last", missing_edge_index=expected_count - 1))
        for missing_edge_index in range(1, expected_count - 1):
            aligned_reference = np.concatenate((reference[:missing_edge_index], reference[missing_edge_index + 1 :]))
            candidates.append(
                _fit_alignment(
                    aligned_reference,
                    observed,
                    "missing_internal",
                    missing_edge_index=missing_edge_index,
                )
            )
    else:
        result["classification"] = "multiple_missing"
        return result

    ranked = sorted(
        candidates,
        key=lambda fit: (
            fit.max_abs_residual_s,
            0 if fit.classification == "missing_internal" else 1,
            0 if fit.classification == "exact" else 1,
        ),
    )
    best = ranked[0]

    if (
        missing_count == 1
        and len(ranked) >= 2
        and {ranked[0].classification, ranked[1].classification} == {"missing_first", "missing_last"}
        and abs(ranked[0].max_abs_residual_s - ranked[1].max_abs_residual_s) <= float(tolerance_s)
    ):
        result["classification"] = "missing_boundary"
        result["latency_s"] = best.latency_s
        result["max_abs_residual_s"] = best.max_abs_residual_s
        result["rms_residual_s"] = best.rms_residual_s
        return result

    result["classification"] = best.classification
    result["latency_s"] = best.latency_s
    result["max_abs_residual_s"] = best.max_abs_residual_s
    result["rms_residual_s"] = best.rms_residual_s
    if best.missing_edge_index is not None:
        result["missing_edge_index"] = int(best.missing_edge_index)
        result["missing_edge_number"] = int(best.missing_edge_index) + 1
        result["missing_edge_time_s"] = float(reference[best.missing_edge_index])

    if best.max_abs_residual_s > float(tolerance_s):
        result["classification"] = "ambiguous"

    return result


def compute_trigger_timing_budget(
    *,
    fps_hz: float,
    pulse_width_ms: float,
    exposure_us: float,
    trigger_delay_us: float = 0.0,
    overlap_mode: str | None = None,
) -> dict[str, float | str | bool | None]:
    if fps_hz <= 0:
        raise ValueError("fps_hz must be > 0")

    period_ms = 1000.0 / fps_hz
    exposure_ms = exposure_us / 1000.0
    trigger_delay_ms = trigger_delay_us / 1000.0
    active_window_ms = exposure_ms + trigger_delay_ms
    slack_to_period_ms = period_ms - active_window_ms

    return {
        "fps_hz": float(fps_hz),
        "period_ms": period_ms,
        "pulse_width_ms": float(pulse_width_ms),
        "exposure_ms": exposure_ms,
        "trigger_delay_ms": trigger_delay_ms,
        "active_window_ms": active_window_ms,
        "slack_to_period_ms": slack_to_period_ms,
        "pulse_duty_cycle_pct": (100.0 * pulse_width_ms / period_ms),
        "active_duty_cycle_pct": (100.0 * active_window_ms / period_ms),
        "overlap_mode": overlap_mode,
        "over_period": slack_to_period_ms < 0.0,
        "tight_timing": slack_to_period_ms < 0.5,
    }