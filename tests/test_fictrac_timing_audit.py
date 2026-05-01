from __future__ import annotations

from pathlib import Path

import numpy as np

from multibios.fictrac_client import FICTRAC_FRAME_DTYPE
from multibios.fictrac_timing_audit import compare_interval_summaries, summarize_fictrac_intervals


def _write_frames(run_dir: Path, *, wall_time_s: list[float], frame_cnt: list[int], alt_timestamp_ms: list[float]) -> None:
    frames = np.zeros(len(frame_cnt), dtype=FICTRAC_FRAME_DTYPE)
    frames["wall_time"] = np.asarray(wall_time_s, dtype=np.float64)
    frames["frame_cnt"] = np.asarray(frame_cnt, dtype=np.int64)
    frames["alt_timestamp"] = np.asarray(alt_timestamp_ms, dtype=np.float64)
    np.savez_compressed(run_dir / "fictrac_frames.npz", frames=frames)


def test_summarize_fictrac_intervals_reports_max_gap_position(tmp_path: Path) -> None:
    run_dir = tmp_path / "2026-05-01_12-00-00"
    run_dir.mkdir()
    _write_frames(
        run_dir,
        wall_time_s=[1.000, 1.005, 1.010, 1.019, 1.024],
        frame_cnt=[0, 1, 2, 3, 4],
        alt_timestamp_ms=[1000.0, 1005.0, 1010.0, 1019.0, 1024.0],
    )

    summary = summarize_fictrac_intervals(run_dir)

    assert summary["frame_count"] == 5
    assert summary["interval_count"] == 4
    assert summary["max_interval_ms"] == 9.0
    assert summary["max_interval_index"] == 2
    assert summary["max_interval_after_frame"] == 3
    assert summary["intervals_ms"] == [5.0, 5.0, 9.0, 5.0]


def test_compare_interval_summaries_reports_deltas(tmp_path: Path) -> None:
    baseline_dir = tmp_path / "2026-05-01_12-00-00"
    other_dir = tmp_path / "2026-05-01_12-01-00"
    baseline_dir.mkdir()
    other_dir.mkdir()

    _write_frames(
        baseline_dir,
        wall_time_s=[1.000, 1.005, 1.010, 1.015],
        frame_cnt=[0, 1, 2, 3],
        alt_timestamp_ms=[1000.0, 1005.0, 1010.0, 1015.0],
    )
    _write_frames(
        other_dir,
        wall_time_s=[1.000, 1.005, 1.013, 1.018],
        frame_cnt=[0, 1, 2, 3],
        alt_timestamp_ms=[1000.0, 1005.0, 1013.0, 1018.0],
    )

    summaries = [summarize_fictrac_intervals(baseline_dir), summarize_fictrac_intervals(other_dir)]
    comparisons = compare_interval_summaries(summaries)

    assert len(comparisons) == 1
    assert comparisons[0]["max_interval_delta_ms"] == 3.0
    assert comparisons[0]["max_interval_index_delta"] == 1
    assert comparisons[0]["max_interval_after_frame_delta"] == 1