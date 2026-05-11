from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from multibios.fictrac_client import FICTRAC_FRAME_DTYPE
from multibios.parity_audit import summarize_run_parity


def test_summarize_run_parity_reports_trigger_deltas(tmp_path: Path) -> None:
    run_dir = tmp_path / "2026-04-30_22-46-07"
    run_dir.mkdir()

    (run_dir / "digital_edges.csv").write_text(
        "line,edge_type,sample_idx,time_ms\n"
        "TRIG_CAMERA,rising,1,0.5\n"
        "TRIG_CAMERA,falling,2,1.0\n"
        "TRIG_CAMERA,rising,3,1.5\n",
        encoding="utf-8",
    )
    (run_dir / "fictrac_camera_recording.json").write_text(
        json.dumps({"saved_raw_frames": 1, "expected_frames": 2}),
        encoding="utf-8",
    )
    (run_dir / "fictrac_driver_diagnostics.json").write_text(
        json.dumps({"frame_cnt": 2, "final_returncode": 0, "stop_method": "ctrl_break"}),
        encoding="utf-8",
    )
    (run_dir / "blackfly_recording.json").write_text(
        json.dumps({"saved_frames": 2, "expected_frame_count": 2}),
        encoding="utf-8",
    )

    frames = np.zeros(2, dtype=FICTRAC_FRAME_DTYPE)
    np.savez_compressed(run_dir / "fictrac_frames.npz", frames=frames)

    summary = summarize_run_parity(run_dir)

    assert summary["counts"]["trigger_rising_edges"] == 2
    assert summary["counts"]["fictrac_saved_raw_frames"] == 1
    assert summary["counts"]["fictrac_udp_frame_cnt"] == 2
    assert summary["counts"]["fictrac_callback_frames"] == 2
    assert summary["counts"]["second_camera_saved_frames"] == 2
    assert summary["mismatches_vs_trigger"]["fictrac_saved_raw_frames"] == -1
    assert summary["mismatches_vs_trigger"]["fictrac_udp_frame_cnt"] == 0
    assert summary["exact_trigger_match"] is False


def test_summarize_run_parity_requires_all_counts_for_exact_match(tmp_path: Path) -> None:
    run_dir = tmp_path / "2026-04-30_22-46-08"
    run_dir.mkdir()

    (run_dir / "digital_edges.csv").write_text(
        "line,edge_type,sample_idx,time_ms\nTRIG_CAMERA,rising,1,0.5\n",
        encoding="utf-8",
    )
    (run_dir / "fictrac_camera_recording.json").write_text(
        json.dumps({"saved_raw_frames": 1, "expected_frames": 1}),
        encoding="utf-8",
    )
    (run_dir / "fictrac_driver_diagnostics.json").write_text(
        json.dumps({"frame_cnt": 1}),
        encoding="utf-8",
    )
    (run_dir / "blackfly_recording.json").write_text(
        json.dumps({"saved_frames": 1, "expected_frame_count": 1}),
        encoding="utf-8",
    )
    frames = np.zeros(1, dtype=FICTRAC_FRAME_DTYPE)
    np.savez_compressed(run_dir / "fictrac_frames.npz", frames=frames)

    summary = summarize_run_parity(run_dir)

    assert summary["exact_trigger_match"] is True


def test_summarize_run_parity_reads_structured_run_layout(tmp_path: Path) -> None:
    run_dir = tmp_path / "2026-05-05_10-04-02"
    (run_dir / "planned" / "daq" / "digital_outputs").mkdir(parents=True)
    (run_dir / "recorded" / "tracking" / "fictrac").mkdir(parents=True)
    (run_dir / "recorded" / "cameras" / "fictrac_camera").mkdir(parents=True)
    (run_dir / "recorded" / "cameras" / "secondary_camera").mkdir(parents=True)

    (run_dir / "planned" / "daq" / "digital_outputs" / "edge_table.csv").write_text(
        "line,edge_type,sample_idx,time_ms\n"
        "TRIG_CAMERA,rising,1,0.5\n"
        "TRIG_CAMERA,rising,2,1.0\n",
        encoding="utf-8",
    )
    (run_dir / "recorded" / "tracking" / "fictrac" / "session_record.json").write_text(
        json.dumps({"recording_summary": {"saved_raw_frames": 2, "expected_frames": 2}}),
        encoding="utf-8",
    )
    (run_dir / "recorded" / "tracking" / "fictrac" / "fictrac_driver_diagnostics.json").write_text(
        json.dumps({"frame_cnt": 2, "final_returncode": 0, "stop_method": "ctrl_break"}),
        encoding="utf-8",
    )
    (run_dir / "recorded" / "cameras" / "secondary_camera" / "recording_summary.json").write_text(
        json.dumps({"saved_frames": 2, "expected_frame_count": 2}),
        encoding="utf-8",
    )

    frames = np.zeros(2, dtype=FICTRAC_FRAME_DTYPE)
    np.savez_compressed(run_dir / "recorded" / "tracking" / "fictrac" / "frame_series.npz", frames=frames)

    summary = summarize_run_parity(run_dir)

    assert summary["counts"] == {
        "trigger_rising_edges": 2,
        "fictrac_saved_raw_frames": 2,
        "fictrac_udp_frame_cnt": 2,
        "fictrac_callback_frames": 2,
        "second_camera_saved_frames": 2,
    }
    assert summary["exact_trigger_match"] is True
    assert summary["fictrac_final_returncode"] == 0
    assert summary["fictrac_stop_method"] == "ctrl_break"


def test_summarize_run_parity_prefers_recording_overrides_before_structured_files_exist(tmp_path: Path) -> None:
    run_dir = tmp_path / "2026-05-05_10-12-49"
    (run_dir / "planned" / "daq" / "digital_outputs").mkdir(parents=True)
    (run_dir / "recorded" / "tracking" / "fictrac").mkdir(parents=True)

    (run_dir / "planned" / "daq" / "digital_outputs" / "edge_table.csv").write_text(
        "line,edge_type,sample_idx,time_ms\n"
        "TRIG_CAMERA,rising,1,0.5\n"
        "TRIG_CAMERA,rising,2,1.0\n",
        encoding="utf-8",
    )
    (run_dir / "recorded" / "tracking" / "fictrac" / "fictrac_driver_diagnostics.json").write_text(
        json.dumps({"frame_cnt": 2, "final_returncode": 0}),
        encoding="utf-8",
    )

    frames = np.zeros(2, dtype=FICTRAC_FRAME_DTYPE)
    np.savez_compressed(run_dir / "recorded" / "tracking" / "fictrac" / "frame_series.npz", frames=frames)

    summary = summarize_run_parity(
        run_dir,
        fictrac_recording_override={"saved_raw_frames": 2, "expected_frames": 2},
        blackfly_recording_override={"saved_frames": 2, "expected_frame_count": 2},
    )

    assert summary["counts"] == {
        "trigger_rising_edges": 2,
        "fictrac_saved_raw_frames": 2,
        "fictrac_udp_frame_cnt": 2,
        "fictrac_callback_frames": 2,
        "second_camera_saved_frames": 2,
    }
    assert summary["exact_trigger_match"] is True