from __future__ import annotations

import pytest

from multibios.blackfly.trigger_path_analysis import classify_missing_events, compute_trigger_timing_budget


def test_classify_missing_events_reports_boundary_ambiguity_for_first_side_too() -> None:
    reference = [0.000, 0.005, 0.010, 0.015, 0.020]
    observed = [0.006, 0.011, 0.016, 0.021]

    summary = classify_missing_events(reference, observed)

    assert summary["classification"] == "missing_boundary"
    assert summary["missing_edge_number"] is None


def test_classify_missing_events_reports_boundary_ambiguity() -> None:
    reference = [0.000, 0.005, 0.010, 0.015, 0.020]
    observed = [0.001, 0.006, 0.011, 0.016]

    summary = classify_missing_events(reference, observed)

    assert summary["classification"] == "missing_boundary"
    assert summary["missing_edge_number"] is None


def test_classify_missing_events_detects_missing_internal() -> None:
    reference = [0.000, 0.005, 0.010, 0.015, 0.020]
    observed = [0.001, 0.006, 0.016, 0.021]

    summary = classify_missing_events(reference, observed)

    assert summary["classification"] == "missing_internal"
    assert summary["missing_edge_number"] == 3
    assert summary["missing_edge_time_s"] == pytest.approx(0.010)


def test_compute_trigger_timing_budget_reports_headroom() -> None:
    budget = compute_trigger_timing_budget(
        fps_hz=200.0,
        pulse_width_ms=1.0,
        exposure_us=4500.0,
        trigger_delay_us=14.0,
        overlap_mode="ReadOut",
    )

    assert budget["period_ms"] == pytest.approx(5.0)
    assert budget["active_window_ms"] == pytest.approx(4.514)
    assert budget["slack_to_period_ms"] == pytest.approx(0.486)
    assert budget["tight_timing"] is True