from __future__ import annotations

import json
from pathlib import Path

from multibios.run_dataset import (RunDatasetLayout,
                                   build_cli_args_payload,
                                   build_hardware_snapshot_payload,
                                   build_placeholder_experiment_record,
                                   build_resolved_runtime_payload,
                                   build_run_manifest_payload,
                                   build_teensy_transcript_meta_payload,
                                   mirror_fictrac_camera_recording,
                                   mirror_secondary_camera_recording,
                                   build_software_environment_payload,
                                   build_source_snapshot_payload,
                                   build_timing_anchors_payload,
                                   normalize_run_relative_path)


def test_run_dataset_layout_uses_new_tree_paths(tmp_path: Path) -> None:
    run_dir = tmp_path / "2026-05-03_16-49-37"
    run_dir.mkdir()

    layout = RunDatasetLayout(run_dir)
    layout.ensure_directories()

    assert layout.protocol_copy_path == run_dir / "inputs" / "protocol.yaml"
    assert layout.hardware_copy_path == run_dir / "inputs" / "hardware.yaml"
    assert layout.timing_anchors_path == run_dir / "planned" / "timing_anchors.json"
    assert layout.experiment_record_path == run_dir / "experiment" / "record.json"
    assert layout.experiment_record_meta_path == run_dir / "experiment" / "record.meta.json"
    assert layout.parity_audit_path == run_dir / "derived" / "validation" / "parity_audit.json"
    assert layout.teensy_transcript_path == run_dir / "logs" / "primary" / "teensy_transcript.jsonl"
    assert layout.primary_logs_dir.is_dir()
    assert layout.diagnostic_logs_dir.is_dir()


def test_normalize_run_relative_path_returns_forward_slash_paths(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    artifact = run_dir / "planned" / "timing_anchors.json"
    artifact.parent.mkdir(parents=True)
    artifact.write_text("{}", encoding="utf-8")

    assert normalize_run_relative_path(run_dir, artifact) == "planned/timing_anchors.json"


def test_build_timing_anchors_payload_tracks_clock_domains() -> None:
    payload = build_timing_anchors_payload(
        sample_rate_hz=2000,
        t0_unix_seconds=1777846177.5,
        t0_perf_counter_seconds=123.456,
    )

    assert payload["schema_name"] == "multibios.timing_anchors"
    assert payload["daq_sample_rate_hz"] == 2000
    assert payload["t0_unix_seconds"] == 1777846177.5
    assert payload["t0_perf_counter_seconds"] == 123.456
    assert payload["t0_utc"].endswith("Z")
    assert payload["clock_domains"][0]["name"] == "daq_sample_clock"
    assert payload["conversion_rules"][0]["from_domain"] == "daq_sample_clock"


def test_build_placeholder_experiment_record_prefills_known_run_fields() -> None:
    record = build_placeholder_experiment_record(
        run_id="2026-05-03_16-49-37",
        run_uuid="1234",
        source_filename="short_protocol.yaml",
        protocol_name="Short Protocol",
        protocol_version="1.0",
        rig_id="Dev1",
        operator=None,
    )

    assert record["schema_name"] == "multibios.experiment_record"
    assert record["record_status"] == "draft_pre"
    assert record["pre_experiment"]["experiment_date"] == "2026-05-03"
    assert record["pre_experiment"]["source_filename"] == "short_protocol.yaml"
    assert record["pre_experiment"]["protocol_name"] == "Short Protocol"
    assert record["pre_experiment"]["rig_id"] == "Dev1"
    assert record["post_experiment"]["aborted"] is False
    assert record["amendments"] == []


def test_build_input_metadata_payloads_capture_core_context(tmp_path: Path) -> None:
    cli_payload = build_cli_args_payload(args={"yaml": "protocols/short_protocol.yaml", "dry_run": True})
    runtime_payload = build_resolved_runtime_payload(
        args={"dry_run": True},
        runtime_cfg={"capture_teensy_serial": False},
        sample_rate_hz=2000,
        duration_ms=5700.0,
        rng_seed=123,
    )
    hardware_payload = build_hardware_snapshot_payload(
        device="Dev1",
        digital_outputs={"TRIG_CAMERA": "Dev1/port0/line0"},
        analog_outputs={},
        analog_inputs={},
        digital_inputs={},
    )
    env_payload = build_software_environment_payload(package_versions={"numpy": "1.0"})
    source_payload = build_source_snapshot_payload(repo_root=tmp_path, entrypoint="python -m multibios.run_protocol")

    assert cli_payload["args"]["dry_run"] is True
    assert runtime_payload["sample_rate_hz"] == 2000
    assert runtime_payload["runtime_config"]["capture_teensy_serial"] is False
    assert hardware_payload["rig_id"] == "Dev1"
    assert env_payload["package_versions"]["numpy"] == "1.0"
    assert source_payload["entrypoint"] == "python -m multibios.run_protocol"


def test_build_run_manifest_payload_indexes_new_tree_files(tmp_path: Path) -> None:
    run_dir = tmp_path / "2026-05-03_16-49-37"
    run_dir.mkdir()
    layout = RunDatasetLayout(run_dir)
    layout.ensure_directories()
    layout.protocol_copy_path.write_text("protocol: {}\n", encoding="utf-8")
    layout.experiment_record_path.write_text("{}", encoding="utf-8")
    layout.experiment_record_meta_path.write_text("{}", encoding="utf-8")
    layout.timing_anchors_path.write_text("{}", encoding="utf-8")

    manifest = build_run_manifest_payload(
        layout=layout,
        run_id=run_dir.name,
        run_uuid="1234",
        status="completed",
        started_utc="2026-05-03T16:49:37Z",
        completed_utc="2026-05-03T16:49:45Z",
        rig_id="Dev1",
        operator=None,
        sample_rate_hz=2000,
    )

    indexed_paths = {entry["path"] for entry in manifest["artifact_index"]}
    assert manifest["schema_name"] == "multibios.run_manifest"
    assert manifest["timing_anchor_file"] == "planned/timing_anchors.json"
    assert manifest["experiment_record_path"] == "experiment/record.json"
    assert manifest["experiment_record_meta_path"] == "experiment/record.meta.json"
    assert manifest["metadata_status"]["record_present"] is False
    assert manifest["metadata_status"]["metadata_complete"] is False
    assert "inputs/protocol.yaml" in indexed_paths
    assert "planned/timing_anchors.json" in indexed_paths
    assert "experiment/record.json" in indexed_paths
    assert "experiment/record.meta.json" in indexed_paths


def test_build_run_manifest_payload_populates_checksums(tmp_path: Path) -> None:
    run_dir = tmp_path / "2026-05-03_16-49-37"
    run_dir.mkdir()
    layout = RunDatasetLayout(run_dir)
    layout.ensure_directories()
    layout.protocol_copy_path.write_text("protocol: {}\n", encoding="utf-8")

    manifest = build_run_manifest_payload(
        layout=layout,
        run_id=run_dir.name,
        run_uuid="1234",
        status="completed",
        started_utc="2026-05-03T16:49:37Z",
        completed_utc="2026-05-03T16:49:45Z",
        rig_id="Dev1",
        operator=None,
        sample_rate_hz=2000,
    )

    artifact = next(entry for entry in manifest["artifact_index"] if entry["path"] == "inputs/protocol.yaml")
    assert artifact["checksum_sha256"] is not None
    assert len(artifact["checksum_sha256"]) == 64


def test_build_teensy_transcript_meta_payload_marks_primary_log() -> None:
    payload = build_teensy_transcript_meta_payload(
        source_port="COM9",
        capture_start_utc="2026-05-03T16:49:37Z",
        capture_end_utc="2026-05-03T16:49:45Z",
    )

    assert payload["schema_name"] == "multibios.teensy_transcript_meta"
    assert payload["artifact_role"] == "primary_log"
    assert payload["source_port"] == "COM9"
    assert payload["line_schema"]["message_field"] == "line"


def test_mirror_secondary_camera_recording_copies_index_and_video(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    layout = RunDatasetLayout(run_dir)
    layout.ensure_directories()

    index_path = run_dir / "blackfly_cam0_frame_index.csv"
    index_path.write_text("frame_index,frame_id\n0,10\n", encoding="utf-8")
    bin_path = run_dir / "blackfly_cam0_frames.bin"
    bin_path.write_bytes(b"1234")
    video_path = run_dir / "blackfly_cam0_lossless.avi"
    video_path.write_bytes(b"avi")

    manifest = mirror_secondary_camera_recording(
        layout=layout,
        run_dir=run_dir,
        recording={
            "camera_index": 0,
            "model": "Blackfly S",
            "serial": "26021184",
            "format": "raw-mono8-stream",
            "dtype": "uint8",
            "frame_width": 4,
            "frame_height": 4,
            "configured_width": 4,
            "configured_height": 4,
            "configured_offset_x": 0,
            "configured_offset_y": 0,
            "binning": 1,
            "requested_exposure_us": 4500.0,
            "requested_gain_db": 0.0,
            "requested_gamma": 1.0,
            "configured_gain_db": 0.0,
            "configured_gamma": 1.0,
            "nominal_trigger_fps": 200.0,
            "started_at": "2026-05-04 12:00:00",
            "completed_at": "2026-05-04 12:00:05",
            "expected_frame_count": 1,
            "saved_frames": 1,
            "frame_index_path": str(index_path),
            "frame_bin_path": str(bin_path),
            "lossless_video": {"path": str(video_path), "fps": 200.0},
            "analysis": {"frame_count_matches_expected": True, "missing_frames_vs_expected": 0, "source_fps": 200.0, "saved_fps": 200.0},
            "no_dropped_frames": True,
            "raw_chunks_retained": False,
            "raw_chunk_cleanup": {"policy": "delete_after_parity", "applied": True, "deleted_chunk_paths": [str(bin_path)], "parity_summary_path": "parity_audit.json"},
        },
    )

    assert manifest["camera_name"] == "secondary_camera"
    assert manifest["frame_index_path"] == "recorded/cameras/secondary_camera/frame_index.csv"
    assert manifest["frame_stream_path"] == "recorded/cameras/secondary_camera/frame_stream.bin"
    assert manifest["lossless_video_path"] == "recorded/cameras/secondary_camera/lossless_video.avi"
    assert layout.secondary_camera_manifest_path.exists()


def test_mirror_fictrac_camera_recording_copies_index_and_video(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    layout = RunDatasetLayout(run_dir)
    layout.ensure_directories()

    source_manifest_path = run_dir / "fictrac-raw-20260504_120000.json"
    source_index_path = run_dir / "fictrac-raw-20260504_120000-index.csv"
    video_path = run_dir / "fictrac-raw-20260504_120000-lossless.avi"
    source_index_path.write_text("frame_index,log_frame,chunk_index,chunk_frame_index\n0,0,0,0\n", encoding="utf-8")
    video_path.write_bytes(b"avi")
    source_manifest_path.write_text(
        json.dumps(
            {
                "format": "raw-bgr8-chunks",
                "dtype": "uint8",
                "channels": 3,
                "frame_width": 4,
                "frame_height": 4,
                "frame_index_path": str(source_index_path),
            }
        ),
        encoding="utf-8",
    )

    manifest = mirror_fictrac_camera_recording(
        layout=layout,
        run_dir=run_dir,
        recording={
            "camera_index": 1,
            "camera_fps": 200.0,
            "raw_stream_manifest": str(source_manifest_path),
            "raw_stream_format": "raw-bgr8-chunks",
            "raw_stream_chunks": [],
            "saved_raw_frames": 1,
            "expected_frames": 1,
            "missing_frames_vs_expected": 0,
            "no_dropped_frames": True,
            "lossless_video": {"path": str(video_path), "fps": 200.0},
            "raw_chunks_retained": False,
            "raw_chunk_cleanup": {"policy": "delete_after_parity", "applied": True, "deleted_chunk_paths": [], "parity_summary_path": "parity_audit.json"},
        },
    )

    assert manifest["camera_name"] == "fictrac_camera"
    assert manifest["frame_index_path"] == "recorded/cameras/fictrac_camera/frame_index.csv"
    assert manifest["lossless_video_path"] == "recorded/cameras/fictrac_camera/lossless_video.avi"
    assert layout.fictrac_camera_manifest_path.exists()