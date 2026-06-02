from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys
from typing import Any


SCHEMA_VERSION = "1.0.0"


def _utc_iso(timestamp: float | None) -> str | None:
    if timestamp is None:
        return None
    return datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat().replace("+00:00", "Z")


def normalize_run_relative_path(run_dir: Path, artifact_path: Path) -> str:
    return artifact_path.resolve().relative_to(run_dir.resolve()).as_posix()


@dataclass(frozen=True)
class RunDatasetLayout:
    run_dir: Path

    @property
    def inputs_dir(self) -> Path:
        return self.run_dir / "inputs"

    @property
    def experiment_dir(self) -> Path:
        return self.run_dir / "experiment"

    @property
    def planned_dir(self) -> Path:
        return self.run_dir / "planned"

    @property
    def planned_daq_dir(self) -> Path:
        return self.planned_dir / "daq"

    @property
    def planned_digital_outputs_dir(self) -> Path:
        return self.planned_daq_dir / "digital_outputs"

    @property
    def planned_analog_outputs_dir(self) -> Path:
        return self.planned_daq_dir / "analog_outputs"

    @property
    def recorded_dir(self) -> Path:
        return self.run_dir / "recorded"

    @property
    def recorded_daq_dir(self) -> Path:
        return self.recorded_dir / "daq"

    @property
    def recorded_analog_inputs_dir(self) -> Path:
        return self.recorded_daq_dir / "analog_inputs"

    @property
    def recorded_digital_inputs_dir(self) -> Path:
        return self.recorded_daq_dir / "digital_inputs"

    @property
    def recorded_cameras_dir(self) -> Path:
        return self.recorded_dir / "cameras"

    @property
    def recorded_tracking_dir(self) -> Path:
        return self.recorded_dir / "tracking"

    @property
    def recorded_microscopy_dir(self) -> Path:
        return self.recorded_dir / "microscopy"

    @property
    def derived_dir(self) -> Path:
        return self.run_dir / "derived"

    @property
    def logs_dir(self) -> Path:
        return self.run_dir / "logs"

    @property
    def primary_logs_dir(self) -> Path:
        return self.logs_dir / "primary"

    @property
    def diagnostic_logs_dir(self) -> Path:
        return self.logs_dir / "diagnostics"

    @property
    def validation_dir(self) -> Path:
        return self.derived_dir / "validation"

    @property
    def previews_dir(self) -> Path:
        return self.derived_dir / "previews"

    @property
    def secondary_camera_dir(self) -> Path:
        return self.recorded_cameras_dir / "secondary_camera"

    @property
    def fictrac_camera_dir(self) -> Path:
        return self.recorded_cameras_dir / "fictrac_camera"

    @property
    def fictrac_tracking_dir(self) -> Path:
        return self.recorded_tracking_dir / "fictrac"

    @property
    def protocol_copy_path(self) -> Path:
        return self.inputs_dir / "protocol.yaml"

    @property
    def readme_path(self) -> Path:
        return self.run_dir / "README.md"

    @property
    def checksums_path(self) -> Path:
        return self.run_dir / "checksums.sha256"

    @property
    def notes_path(self) -> Path:
        return self.run_dir / "notes.md"

    @property
    def hardware_copy_path(self) -> Path:
        return self.inputs_dir / "hardware.yaml"

    @property
    def cli_args_path(self) -> Path:
        return self.inputs_dir / "cli_args.json"

    @property
    def resolved_runtime_path(self) -> Path:
        return self.inputs_dir / "resolved_runtime.json"

    @property
    def source_snapshot_path(self) -> Path:
        return self.inputs_dir / "source_snapshot.json"

    @property
    def software_environment_path(self) -> Path:
        return self.inputs_dir / "software_environment.json"

    @property
    def hardware_snapshot_path(self) -> Path:
        return self.inputs_dir / "hardware_snapshot.json"

    @property
    def timing_anchors_path(self) -> Path:
        return self.planned_dir / "timing_anchors.json"

    @property
    def compile_report_path(self) -> Path:
        return self.planned_dir / "compile_report.json"

    @property
    def control_plan_path(self) -> Path:
        return self.planned_dir / "control_plan.csv"

    @property
    def planned_do_channels_path(self) -> Path:
        return self.planned_digital_outputs_dir / "channels.json"

    @property
    def planned_do_signal_array_path(self) -> Path:
        return self.planned_digital_outputs_dir / "signal_array.npz"

    @property
    def planned_do_signal_meta_path(self) -> Path:
        return self.planned_digital_outputs_dir / "signal_array.meta.json"

    @property
    def planned_do_edge_table_path(self) -> Path:
        return self.planned_digital_outputs_dir / "edge_table.csv"

    @property
    def planned_do_edge_meta_path(self) -> Path:
        return self.planned_digital_outputs_dir / "edge_table.meta.json"

    @property
    def planned_do_commit_edge_table_path(self) -> Path:
        return self.planned_digital_outputs_dir / "commit_edge_table.csv"

    @property
    def planned_do_commit_edge_meta_path(self) -> Path:
        return self.planned_digital_outputs_dir / "commit_edge_table.meta.json"

    @property
    def planned_ao_channels_path(self) -> Path:
        return self.planned_analog_outputs_dir / "channels.json"

    @property
    def planned_ao_signal_array_path(self) -> Path:
        return self.planned_analog_outputs_dir / "signal_array.npz"

    @property
    def planned_ao_signal_meta_path(self) -> Path:
        return self.planned_analog_outputs_dir / "signal_array.meta.json"

    @property
    def experiment_record_path(self) -> Path:
        return self.experiment_dir / "record.json"

    @property
    def experiment_record_meta_path(self) -> Path:
        return self.experiment_dir / "record.meta.json"

    @property
    def run_manifest_path(self) -> Path:
        return self.run_dir / "run_manifest.json"

    @property
    def parity_audit_path(self) -> Path:
        return self.validation_dir / "parity_audit.json"

    @property
    def dataset_completeness_path(self) -> Path:
        return self.validation_dir / "dataset_completeness.json"

    @property
    def timing_alignment_path(self) -> Path:
        return self.validation_dir / "timing_alignment.json"

    @property
    def daq_capture_summary_path(self) -> Path:
        return self.validation_dir / "daq_capture_summary.json"

    @property
    def secondary_camera_integrity_path(self) -> Path:
        return self.validation_dir / "secondary_camera_integrity.json"

    @property
    def fictrac_integrity_path(self) -> Path:
        return self.validation_dir / "fictrac_integrity.json"

    @property
    def protocol_preview_path(self) -> Path:
        return self.previews_dir / "protocol_preview.html"

    @property
    def teensy_transcript_path(self) -> Path:
        return self.primary_logs_dir / "teensy_transcript.jsonl"

    @property
    def teensy_transcript_meta_path(self) -> Path:
        return self.primary_logs_dir / "teensy_transcript.meta.json"

    @property
    def warnings_path(self) -> Path:
        return self.diagnostic_logs_dir / "warnings.json"

    @property
    def run_log_path(self) -> Path:
        return self.diagnostic_logs_dir / "run_log.txt"

    @property
    def fictrac_driver_diagnostics_path(self) -> Path:
        return self.diagnostic_logs_dir / "fictrac_driver_diagnostics.json"

    @property
    def recorded_ai_channels_path(self) -> Path:
        return self.recorded_analog_inputs_dir / "channels.json"

    @property
    def recorded_ai_samples_path(self) -> Path:
        return self.recorded_analog_inputs_dir / "samples.npz"

    @property
    def recorded_ai_samples_meta_path(self) -> Path:
        return self.recorded_analog_inputs_dir / "samples.meta.json"

    @property
    def recorded_di_channels_path(self) -> Path:
        return self.recorded_digital_inputs_dir / "channels.json"

    @property
    def recorded_di_samples_path(self) -> Path:
        return self.recorded_digital_inputs_dir / "samples.npz"

    @property
    def recorded_di_samples_meta_path(self) -> Path:
        return self.recorded_digital_inputs_dir / "samples.meta.json"

    @property
    def recorded_di_edge_table_path(self) -> Path:
        return self.recorded_digital_inputs_dir / "edge_table.csv"

    @property
    def recorded_di_edge_meta_path(self) -> Path:
        return self.recorded_digital_inputs_dir / "edge_table.meta.json"

    @property
    def secondary_camera_manifest_path(self) -> Path:
        return self.secondary_camera_dir / "recording_manifest.json"

    @property
    def secondary_camera_recording_summary_path(self) -> Path:
        return self.secondary_camera_dir / "recording_summary.json"

    @property
    def secondary_camera_source_manifest_path(self) -> Path:
        return self.secondary_camera_dir / "source_manifest.json"

    @property
    def secondary_camera_analysis_path(self) -> Path:
        return self.secondary_camera_dir / "analysis.json"

    @property
    def secondary_camera_frame_index_meta_path(self) -> Path:
        return self.secondary_camera_dir / "frame_index.meta.json"

    @property
    def secondary_camera_frame_stream_meta_path(self) -> Path:
        return self.secondary_camera_dir / "frame_stream.meta.json"

    @property
    def secondary_camera_lossless_video_meta_path(self) -> Path:
        return self.secondary_camera_dir / "lossless_video.meta.json"

    @property
    def fictrac_camera_manifest_path(self) -> Path:
        return self.fictrac_camera_dir / "recording_manifest.json"

    @property
    def fictrac_camera_recording_summary_path(self) -> Path:
        return self.fictrac_camera_dir / "recording_summary.json"

    @property
    def fictrac_camera_raw_stream_manifest_path(self) -> Path:
        return self.fictrac_camera_dir / "raw_stream_manifest.json"

    @property
    def fictrac_camera_frame_index_meta_path(self) -> Path:
        return self.fictrac_camera_dir / "frame_index.meta.json"

    @property
    def fictrac_camera_frame_stream_meta_path(self) -> Path:
        return self.fictrac_camera_dir / "frame_stream.meta.json"

    @property
    def fictrac_camera_lossless_video_meta_path(self) -> Path:
        return self.fictrac_camera_dir / "lossless_video.meta.json"

    @property
    def fictrac_tracking_runtime_config_path(self) -> Path:
        return self.fictrac_tracking_dir / "runtime_config.txt"

    @property
    def fictrac_tracking_runtime_json_path(self) -> Path:
        return self.fictrac_tracking_dir / "runtime_config.json"

    @property
    def fictrac_tracking_session_record_path(self) -> Path:
        return self.fictrac_tracking_dir / "session_record.json"

    @property
    def fictrac_tracking_frame_series_path(self) -> Path:
        return self.fictrac_tracking_dir / "frame_series.npz"

    @property
    def fictrac_tracking_frame_series_meta_path(self) -> Path:
        return self.fictrac_tracking_dir / "frame_series.meta.json"

    @property
    def fictrac_tracking_output_dat_path(self) -> Path:
        return self.fictrac_tracking_dir / "tracker_output.dat"

    @property
    def fictrac_tracking_template_image_path(self) -> Path:
        return self.fictrac_tracking_dir / "template.png"

    def ensure_directories(self) -> None:
        for path in (
            self.inputs_dir,
            self.experiment_dir,
            self.planned_dir,
            self.planned_daq_dir,
            self.planned_digital_outputs_dir,
            self.planned_analog_outputs_dir,
            self.recorded_dir,
            self.recorded_daq_dir,
            self.recorded_analog_inputs_dir,
            self.recorded_digital_inputs_dir,
            self.recorded_cameras_dir,
            self.recorded_tracking_dir,
            self.recorded_microscopy_dir,
            self.derived_dir,
            self.validation_dir,
            self.previews_dir,
            self.primary_logs_dir,
            self.diagnostic_logs_dir,
            self.secondary_camera_dir,
            self.fictrac_camera_dir,
            self.fictrac_tracking_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)


def build_timing_anchors_payload(
    *,
    sample_rate_hz: float,
    t0_unix_seconds: float | None,
    t0_perf_counter_seconds: float | None,
) -> dict[str, Any]:
    return {
        "schema_name": "multibios.timing_anchors",
        "schema_version": SCHEMA_VERSION,
        "artifact_role": "planned_output",
        "daq_sample_rate_hz": sample_rate_hz,
        "t0_utc": _utc_iso(t0_unix_seconds),
        "t0_unix_seconds": t0_unix_seconds,
        "t0_perf_counter_seconds": t0_perf_counter_seconds,
        "clock_domains": [
            {
                "name": "daq_sample_clock",
                "unit": "sample_index",
                "description": "Primary DAQ hardware sample clock for the run.",
            },
            {
                "name": "utc_wall_clock",
                "unit": "utc_timestamp",
                "description": "UTC wall clock aligned to DO start.",
            },
            {
                "name": "system_perf_counter",
                "unit": "seconds",
                "description": "Process-local perf_counter aligned to DO start.",
            },
        ],
        "conversion_rules": [
            {
                "from_domain": "daq_sample_clock",
                "to_domain": "utc_wall_clock",
                "formula": "utc = t0_unix_seconds + sample_idx / daq_sample_rate_hz",
                "parameters": {
                    "t0_unix_seconds": t0_unix_seconds,
                    "daq_sample_rate_hz": sample_rate_hz,
                },
            },
            {
                "from_domain": "daq_sample_clock",
                "to_domain": "system_perf_counter",
                "formula": "perf_counter = t0_perf_counter_seconds + sample_idx / daq_sample_rate_hz",
                "parameters": {
                    "t0_perf_counter_seconds": t0_perf_counter_seconds,
                    "daq_sample_rate_hz": sample_rate_hz,
                },
            },
        ],
    }


def build_placeholder_experiment_record(
    *,
    run_id: str,
    run_uuid: str,
    source_filename: str,
    protocol_name: str,
    protocol_version: str | None,
    rig_id: str,
    operator: str | None,
) -> dict[str, Any]:
    experiment_date = run_id.split("_", 1)[0]
    return {
        "schema_name": "multibios.experiment_record",
        "schema_version": SCHEMA_VERSION,
        "artifact_role": "input_record",
        "record_status": "draft_pre",
        "run_id": run_id,
        "run_uuid": run_uuid,
        "entered_by": operator,
        "entered_started_utc": None,
        "entered_completed_utc": None,
        "ui_version": None,
        "pre_experiment": {
            "experiment_date": experiment_date,
            "source_filename": source_filename,
            "expected_imaging_periods": 0,
            "fly_id": None,
            "species": None,
            "genotype": None,
            "hemisphere": "unknown",
            "age": {"value": None, "unit": "unknown"},
            "starvation": {"value": None, "unit": "unknown"},
            "stimulus_modality": None,
            "rig_temperature_c": None,
            "humidity_percent": None,
            "protocol_name": protocol_name,
            "protocol_version": protocol_version,
            "rig_id": rig_id,
            "operator": operator,
        },
        "post_experiment": {
            "response": None,
            "notes": None,
            "duration_s": None,
            "completion_status": None,
            "aborted": False,
            "exclusion_reason": None,
            "imaging_dataset_source_path": None,
            "imaging_dataset_relative_path": None,
            "imaging_acquisition_type": None,
            "imaging_num_rois": None,
            "imaging_num_channels": None,
            "imaging_num_planes": None,
            "observed_anomalies": [],
            "quality_flags": [],
        },
        "custom_fields": {},
        "amendments": [],
    }


def build_cli_args_payload(*, args: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_name": "multibios.cli_args",
        "schema_version": SCHEMA_VERSION,
        "artifact_role": "input_record",
        "args": args,
    }


def build_resolved_runtime_payload(
    *,
    args: dict[str, Any],
    runtime_cfg: dict[str, Any],
    sample_rate_hz: float,
    duration_ms: float,
    rng_seed: int | None,
) -> dict[str, Any]:
    return {
        "schema_name": "multibios.resolved_runtime",
        "schema_version": SCHEMA_VERSION,
        "artifact_role": "input_record",
        "sample_rate_hz": sample_rate_hz,
        "duration_ms": duration_ms,
        "rng_seed": rng_seed,
        "cli_overrides": args,
        "runtime_config": runtime_cfg,
    }


def build_hardware_snapshot_payload(
    *,
    device: str,
    digital_outputs: dict[str, str],
    analog_outputs: dict[str, str],
    analog_inputs: dict[str, str],
    digital_inputs: dict[str, str],
) -> dict[str, Any]:
    return {
        "schema_name": "multibios.hardware_snapshot",
        "schema_version": SCHEMA_VERSION,
        "artifact_role": "input_record",
        "rig_id": device,
        "device": device,
        "digital_outputs": digital_outputs,
        "analog_outputs": analog_outputs,
        "analog_inputs": analog_inputs,
        "digital_inputs": digital_inputs,
    }


def build_channel_map_payload(
    *,
    schema_name: str,
    artifact_role: str,
    channels: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "schema_name": schema_name,
        "schema_version": SCHEMA_VERSION,
        "artifact_role": artifact_role,
        "channels": channels,
    }


def build_array_meta_payload(
    *,
    schema_name: str,
    artifact_role: str,
    data_path: str,
    dtype: str,
    shape: list[int],
    axis_order: list[str],
    clock_domain: str,
    sample_rate_hz: float | None,
    value_unit: str | None,
    channel_path: str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "schema_name": schema_name,
        "schema_version": SCHEMA_VERSION,
        "artifact_role": artifact_role,
        "data_path": data_path,
        "dtype": dtype,
        "shape": shape,
        "axis_order": axis_order,
        "clock_domain": clock_domain,
        "sample_rate_hz": sample_rate_hz,
        "value_unit": value_unit,
        "channel_path": channel_path,
    }
    if extra:
        payload.update(extra)
    return payload


def build_table_meta_payload(
    *,
    schema_name: str,
    artifact_role: str,
    table_path: str,
    columns: list[dict[str, Any]],
    clock_domain: str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "schema_name": schema_name,
        "schema_version": SCHEMA_VERSION,
        "artifact_role": artifact_role,
        "table_path": table_path,
        "columns": columns,
        "clock_domain": clock_domain,
    }
    if extra:
        payload.update(extra)
    return payload


def build_dataset_completeness_payload(
    *,
    manifest: dict[str, Any],
    required_paths: list[str],
) -> dict[str, Any]:
    indexed_paths = {entry.get("path") for entry in manifest.get("artifact_index", [])}
    missing_paths = [path for path in required_paths if path not in indexed_paths]
    metadata_status = manifest.get("metadata_status") or {}
    return {
        "schema_name": "multibios.dataset_completeness",
        "schema_version": SCHEMA_VERSION,
        "artifact_role": "derived_summary",
        "dataset_complete": not missing_paths and bool(metadata_status.get("metadata_complete")),
        "missing_required_artifacts": missing_paths,
        "metadata_status": metadata_status,
        "run_status": manifest.get("status"),
    }


def build_timing_alignment_payload(
    *,
    sample_rate_hz: float,
    t0_unix_seconds: float | None,
    t0_perf_counter_seconds: float | None,
    expected_camera_frames: int | None,
) -> dict[str, Any]:
    return {
        "schema_name": "multibios.timing_alignment",
        "schema_version": SCHEMA_VERSION,
        "artifact_role": "derived_summary",
        "clock_domains": ["daq_sample_clock", "utc_wall_clock", "system_perf_counter"],
        "daq_sample_rate_hz": sample_rate_hz,
        "t0_utc": _utc_iso(t0_unix_seconds),
        "t0_perf_counter_seconds": t0_perf_counter_seconds,
        "expected_camera_trigger_count": expected_camera_frames,
    }


def build_daq_capture_summary_payload(
    *,
    ai_names: list[str],
    ai_shape: list[int] | None,
    di_names: list[str],
    di_shape: list[int] | None,
    di_edge_count: int | None,
) -> dict[str, Any]:
    return {
        "schema_name": "multibios.daq_capture_summary",
        "schema_version": SCHEMA_VERSION,
        "artifact_role": "derived_summary",
        "analog_inputs": {
            "channel_names": ai_names,
            "shape": ai_shape,
        },
        "digital_inputs": {
            "channel_names": di_names,
            "shape": di_shape,
            "edge_count": di_edge_count,
        },
    }


def build_integrity_summary_payload(
    *,
    schema_name: str,
    recording_manifest: dict[str, Any] | None,
    recording_manifest_path: str | None = None,
) -> dict[str, Any]:
    return {
        "schema_name": schema_name,
        "schema_version": SCHEMA_VERSION,
        "artifact_role": "derived_summary",
        "recording_manifest_path": recording_manifest_path,
        "integrity_summary": (recording_manifest or {}).get("integrity_summary"),
        "retention_state": (recording_manifest or {}).get("retention_state"),
    }


def build_fictrac_session_record_payload(
    *,
    recording: dict[str, Any] | None,
    runtime_info: dict[str, Any] | None,
) -> dict[str, Any]:
    return {
        "schema_name": "multibios.fictrac_session_record",
        "schema_version": SCHEMA_VERSION,
        "artifact_role": "primary_evidence",
        "runtime_info": runtime_info or {},
        "recording_summary": recording or {},
    }


def build_notes_placeholder() -> str:
    return "# Run Notes\n\nAdd freeform operator notes here.\n"


def build_readme_text(
    *,
    run_id: str,
    status: str,
    rig_id: str,
    started_utc: str | None,
    protocol_name: str | None,
    metadata_status: dict[str, Any],
) -> str:
    lines = [
        f"# Run {run_id}",
        "",
        f"- Status: {status}",
        f"- Started UTC: {started_utc or 'unknown'}",
        f"- Rig ID: {rig_id}",
        f"- Protocol: {protocol_name or 'unknown'}",
        f"- Metadata complete: {metadata_status.get('metadata_complete')}",
        "",
        "## Key Locations",
        "",
        "- Inputs: inputs/",
        "- Experiment record: experiment/record.json",
        "- Planned outputs: planned/",
        "- Recorded evidence: recorded/",
        "- Validation outputs: derived/validation/",
        "- Logs: logs/",
        "",
    ]
    missing = metadata_status.get("missing_required_fields") or {}
    pre_missing = missing.get("pre_experiment") or []
    post_missing = missing.get("post_experiment") or []
    if pre_missing or post_missing:
        lines.extend([
            "## Incomplete Metadata",
            "",
            f"- Pre-experiment missing: {', '.join(pre_missing) if pre_missing else 'none'}",
            f"- Post-experiment missing: {', '.join(post_missing) if post_missing else 'none'}",
            "",
        ])
    return "\n".join(lines)


def compute_file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def build_checksums_payload(*, run_dir: Path, excluded_paths: set[Path] | None = None) -> list[tuple[str, str]]:
    excluded = {path.resolve() for path in (excluded_paths or set())}
    checksums: list[tuple[str, str]] = []
    for path in sorted(run_dir.rglob("*")):
        if not path.is_file():
            continue
        if path.resolve() in excluded:
            continue
        checksums.append((compute_file_sha256(path), normalize_run_relative_path(run_dir, path)))
    return checksums


def build_software_environment_payload(*, package_versions: dict[str, str | None]) -> dict[str, Any]:
    return {
        "schema_name": "multibios.software_environment",
        "schema_version": SCHEMA_VERSION,
        "artifact_role": "input_record",
        "python_version": sys.version,
        "python_executable": sys.executable,
        "python_prefix": sys.prefix,
        "platform": platform.platform(),
        "package_versions": package_versions,
    }


def build_source_snapshot_payload(*, repo_root: Path, entrypoint: str) -> dict[str, Any]:
    def _git_output(args: list[str]) -> str | None:
        try:
            completed = subprocess.run(
                args,
                cwd=repo_root,
                capture_output=True,
                text=True,
                check=True,
            )
        except Exception:
            return None
        value = completed.stdout.strip()
        return value or None

    commit = _git_output(["git", "rev-parse", "HEAD"])
    status_output = _git_output(["git", "status", "--porcelain"])
    return {
        "schema_name": "multibios.source_snapshot",
        "schema_version": SCHEMA_VERSION,
        "artifact_role": "input_record",
        "repository_root": str(repo_root),
        "entrypoint": entrypoint,
        "git_commit": commit,
        "git_dirty": bool(status_output) if status_output is not None else None,
    }


def build_run_manifest_payload(
    *,
    layout: RunDatasetLayout,
    run_id: str,
    run_uuid: str,
    status: str,
    started_utc: str | None,
    completed_utc: str | None,
    rig_id: str,
    operator: str | None,
    sample_rate_hz: float,
    warnings: list[dict[str, Any]] | None = None,
    missing_optional_artifacts: list[str] | None = None,
    retention_summary: dict[str, Any] | None = None,
    metadata_status: dict[str, Any] | None = None,
    checksum_lookup: dict[str, str] | None = None,
) -> dict[str, Any]:
    artifact_index: list[dict[str, Any]] = []
    root_level_roles = {
        "README.md": "derived_summary",
        "checksums.sha256": "derived_summary",
        "notes.md": "input_record",
    }
    for path in sorted(layout.run_dir.rglob("*")):
        if not path.is_file() or path == layout.run_manifest_path:
            continue
        rel = normalize_run_relative_path(layout.run_dir, path)
        top = Path(rel).parts[0]
        if top not in {"inputs", "experiment", "planned", "recorded", "derived", "logs"} and rel not in root_level_roles:
            continue
        artifact_role = root_level_roles.get(rel)
        if artifact_role is None:
            artifact_role = {
                "inputs": "input_record",
                "experiment": "input_record",
                "planned": "planned_output",
                "recorded": "primary_evidence",
                "derived": "derived_summary",
                "logs": "primary_log" if rel.startswith("logs/primary/") else "diagnostic_log",
            }[top]
        artifact_index.append(
            {
                "artifact_id": rel.replace("/", "__"),
                "path": rel,
                "artifact_role": artifact_role,
                "media_type": _guess_media_type(path),
                "produced_by": "multibios.run_protocol",
                "required_for_completeness": top in {"inputs", "experiment", "planned", "recorded"} or rel.startswith("logs/primary/") or rel in root_level_roles,
                "retained": True,
                "checksum_sha256": (checksum_lookup or {}).get(rel, compute_file_sha256(path)),
            }
        )

    return {
        "schema_name": "multibios.run_manifest",
        "schema_version": SCHEMA_VERSION,
        "dataset_kind": "run_protocol_dataset",
        "run_id": run_id,
        "run_uuid": run_uuid,
        "status": status,
        "started_utc": started_utc,
        "completed_utc": completed_utc,
        "rig_id": rig_id,
        "operator": operator,
        "primary_clock": {
            "name": "daq_sample_clock",
            "sample_rate_hz": sample_rate_hz,
            "tick_unit": "sample_index",
            "wall_time_anchor": normalize_run_relative_path(layout.run_dir, layout.timing_anchors_path),
        },
        "timing_anchor_file": normalize_run_relative_path(layout.run_dir, layout.timing_anchors_path),
        "experiment_record_path": normalize_run_relative_path(layout.run_dir, layout.experiment_record_path),
        "experiment_record_meta_path": normalize_run_relative_path(layout.run_dir, layout.experiment_record_meta_path),
        "metadata_status": metadata_status or {
            "record_present": False,
            "record_status": None,
            "pre_experiment_complete": False,
            "post_experiment_complete": False,
            "metadata_complete": False,
            "operator_recorded": bool(operator),
            "entered_by": operator,
            "entered_started_utc": None,
            "entered_completed_utc": None,
            "ui_version": None,
            "missing_required_fields": {
                "pre_experiment": [],
                "post_experiment": [],
            },
        },
        "artifact_index": artifact_index,
        "warnings": warnings or [],
        "missing_optional_artifacts": missing_optional_artifacts or [],
        "retention_summary": retention_summary or {
            "raw_chunks_deleted_after_validation": False,
            "deleted_artifact_count": 0,
            "validated_access_copies_present": False,
            "parity_audit_path": None,
        },
    }


def build_teensy_transcript_meta_payload(
    *,
    source_port: str | None,
    capture_start_utc: str | None,
    capture_end_utc: str | None,
) -> dict[str, Any]:
    return {
        "schema_name": "multibios.teensy_transcript_meta",
        "schema_version": SCHEMA_VERSION,
        "artifact_role": "primary_log",
        "line_format": "jsonl",
        "timestamp_basis": "system_perf_counter",
        "source_port": source_port,
        "source_device": "teensy",
        "line_schema": {
            "timestamp_field": "timestamp",
            "message_field": "line",
            "tag_field": "tag",
            "raw_line_field": None,
            "line_kind_values": ["tx", "rx", "info", "warning", "error"],
        },
        "capture_start_utc": capture_start_utc,
        "capture_end_utc": capture_end_utc,
    }


def resolve_run_artifact_path(run_dir: Path, path_value: Any) -> Path | None:
    if not path_value:
        return None
    path = Path(str(path_value))
    candidates: list[Path] = []
    if path.is_absolute():
        candidates.append(path)
    else:
        candidates.extend([path, run_dir / path.name, run_dir / path])
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return None


def mirror_artifact_file(run_dir: Path, path_value: Any, destination: Path) -> str | None:
    source = resolve_run_artifact_path(run_dir, path_value)
    if source is None:
        return None
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.resolve() != destination.resolve():
        shutil.copy2(source, destination)
    return normalize_run_relative_path(run_dir, destination)


def relocate_artifact_file(run_dir: Path, path_value: Any, destination: Path) -> str | None:
    source = resolve_run_artifact_path(run_dir, path_value)
    if source is None:
        return None
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.resolve() != destination.resolve():
        if destination.exists():
            destination.unlink()
        shutil.move(str(source), str(destination))
    return normalize_run_relative_path(run_dir, destination)


def mirror_secondary_camera_recording(
    *,
    layout: RunDatasetLayout,
    run_dir: Path,
    recording: dict[str, Any],
) -> dict[str, Any]:
    frame_index_path = mirror_artifact_file(
        run_dir,
        recording.get("frame_index_path"),
        layout.secondary_camera_dir / "frame_index.csv",
    )
    frame_stream_path = mirror_artifact_file(
        run_dir,
        recording.get("frame_bin_path"),
        layout.secondary_camera_dir / "frame_stream.bin",
    )
    lossless_video = recording.get("lossless_video") or {}
    lossless_path = None
    if lossless_video.get("path"):
        lossless_suffix = Path(str(lossless_video["path"])).suffix or ".avi"
        lossless_path = mirror_artifact_file(
            run_dir,
            lossless_video.get("path"),
            layout.secondary_camera_dir / f"lossless_video{lossless_suffix}",
        )

    artifacts: list[dict[str, Any]] = []
    for rel_path, role, description in (
        (frame_index_path, "primary_evidence", "Frame timing index"),
        (frame_stream_path, "primary_evidence", "Primary frame stream"),
        (lossless_path, "validated_access_copy", "Validated contiguous review video"),
    ):
        if rel_path:
            artifacts.append({"path": rel_path, "artifact_role": role, "retained": True, "description": description})

    cleanup = recording.get("raw_chunk_cleanup") or {}
    manifest = {
        "schema_name": "multibios.camera_recording_manifest",
        "schema_version": SCHEMA_VERSION,
        "artifact_role": "primary_evidence",
        "camera_name": "secondary_camera",
        "camera_index": recording.get("camera_index"),
        "camera_model": recording.get("model"),
        "camera_serial": recording.get("serial"),
        "pixel_format": recording.get("format"),
        "dtype": recording.get("dtype"),
        "channels": int(recording.get("channels", 1) or 1),
        "frame_width": recording.get("frame_width"),
        "frame_height": recording.get("frame_height"),
        "configured_roi": {
            "width": recording.get("configured_width") or recording.get("requested_roi_width"),
            "height": recording.get("configured_height") or recording.get("requested_roi_height"),
            "offset_x": recording.get("configured_offset_x"),
            "offset_y": recording.get("configured_offset_y"),
            "binning": recording.get("binning"),
        },
        "requested_settings": {
            "exposure_us": recording.get("requested_exposure_us"),
            "gain_db": recording.get("requested_gain_db"),
            "gamma": recording.get("requested_gamma"),
        },
        "actual_settings": {
            "exposure_us": recording.get("requested_exposure_us"),
            "gain_db": recording.get("configured_gain_db"),
            "gamma": recording.get("configured_gamma"),
        },
        "trigger_mode": "hardware_triggered",
        "nominal_trigger_fps": recording.get("nominal_trigger_fps"),
        "started_utc": recording.get("started_at"),
        "completed_utc": recording.get("completed_at"),
        "expected_frame_count": recording.get("expected_frame_count"),
        "saved_frame_count": recording.get("saved_frames"),
        "frame_index_path": frame_index_path,
        "frame_stream_path": frame_stream_path,
        "chunk_paths": [],
        "lossless_video_path": lossless_path,
        "artifacts": artifacts,
        "retention_state": {
            "raw_chunks_retained": bool(recording.get("raw_chunks_retained", False)),
            "cleanup_policy": cleanup.get("policy"),
            "cleanup_applied": bool(cleanup.get("applied", False)),
            "deleted_artifact_paths": [Path(str(path)).name for path in cleanup.get("deleted_chunk_paths", [])],
            "validated_by_parity_audit": cleanup.get("parity_summary_path") is not None,
            "parity_audit_path": "derived/validation/parity_audit.json" if cleanup.get("parity_summary_path") else None,
        },
        "integrity_summary": {
            "frame_count_matches_expected": (recording.get("analysis") or {}).get("frame_count_matches_expected"),
            "missing_frames_vs_expected": (recording.get("analysis") or {}).get("missing_frames_vs_expected"),
            "no_dropped_frames": recording.get("no_dropped_frames"),
            "source_fps": (recording.get("analysis") or {}).get("source_fps"),
            "saved_fps": (recording.get("analysis") or {}).get("saved_fps"),
        },
    }
    layout.secondary_camera_manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if frame_index_path:
        layout.secondary_camera_frame_index_meta_path.write_text(
            json.dumps(
                build_table_meta_payload(
                    schema_name="multibios.camera_frame_index_meta",
                    artifact_role="primary_evidence",
                    table_path=frame_index_path,
                    clock_domain="daq_sample_clock",
                    columns=[
                        {"name": "frame_index", "unit": "frame", "description": "Saved frame ordinal."},
                        {"name": "frame_id", "unit": "frame", "description": "Camera frame identifier when available."},
                    ],
                ),
                indent=2,
            ),
            encoding="utf-8",
        )
    if frame_stream_path:
        layout.secondary_camera_frame_stream_meta_path.write_text(
            json.dumps(
                build_array_meta_payload(
                    schema_name="multibios.camera_frame_stream_meta",
                    artifact_role="primary_evidence",
                    data_path=frame_stream_path,
                    dtype=str(recording.get("dtype") or "uint8"),
                    shape=[
                        int(recording.get("saved_frames") or 0),
                        int(recording.get("frame_height") or 0),
                        int(recording.get("frame_width") or 0),
                        int(recording.get("channels", 1) or 1),
                    ],
                    axis_order=["frame", "y", "x", "channel"],
                    clock_domain="daq_sample_clock",
                    sample_rate_hz=recording.get("nominal_trigger_fps"),
                    value_unit=None,
                ),
                indent=2,
            ),
            encoding="utf-8",
        )
    if lossless_path:
        layout.secondary_camera_lossless_video_meta_path.write_text(
            json.dumps(
                {
                    "schema_name": "multibios.camera_video_meta",
                    "schema_version": SCHEMA_VERSION,
                    "artifact_role": "validated_access_copy",
                    "video_path": lossless_path,
                    "nominal_fps": (lossless_video or {}).get("fps"),
                    "source_manifest_path": normalize_run_relative_path(run_dir, layout.secondary_camera_manifest_path),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    return manifest


def mirror_fictrac_camera_recording(
    *,
    layout: RunDatasetLayout,
    run_dir: Path,
    recording: dict[str, Any],
) -> dict[str, Any]:
    source_manifest_path = resolve_run_artifact_path(run_dir, recording.get("raw_stream_manifest"))
    source_manifest: dict[str, Any] = {}
    if source_manifest_path is not None:
        source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))

    frame_index_path = mirror_artifact_file(
        run_dir,
        source_manifest.get("frame_index_path"),
        layout.fictrac_camera_dir / "frame_index.csv",
    )
    frame_stream_path = None
    raw_chunks = recording.get("raw_stream_chunks") or []
    if len(raw_chunks) == 1:
        frame_stream_path = mirror_artifact_file(
            run_dir,
            raw_chunks[0],
            layout.fictrac_camera_dir / "frame_stream.bin",
        )
    lossless_video = recording.get("lossless_video") or {}
    lossless_path = None
    if lossless_video.get("path"):
        lossless_suffix = Path(str(lossless_video["path"])).suffix or ".avi"
        lossless_path = mirror_artifact_file(
            run_dir,
            lossless_video.get("path"),
            layout.fictrac_camera_dir / f"lossless_video{lossless_suffix}",
        )

    artifacts: list[dict[str, Any]] = []
    for rel_path, role, description in (
        (frame_index_path, "primary_evidence", "Frame timing index"),
        (frame_stream_path, "primary_evidence", "Primary frame stream"),
        (lossless_path, "validated_access_copy", "Validated contiguous review video"),
    ):
        if rel_path:
            artifacts.append({"path": rel_path, "artifact_role": role, "retained": True, "description": description})

    cleanup = recording.get("raw_chunk_cleanup") or {}
    manifest = {
        "schema_name": "multibios.camera_recording_manifest",
        "schema_version": SCHEMA_VERSION,
        "artifact_role": "primary_evidence",
        "camera_name": "fictrac_camera",
        "camera_index": recording.get("camera_index"),
        "camera_model": None,
        "camera_serial": recording.get("fictrac_camera_serial"),
        "pixel_format": source_manifest.get("format") or recording.get("raw_stream_format"),
        "dtype": source_manifest.get("dtype"),
        "channels": int(source_manifest.get("channels", 3) or 3),
        "frame_width": source_manifest.get("frame_width"),
        "frame_height": source_manifest.get("frame_height"),
        "configured_roi": {
            "width": source_manifest.get("frame_width"),
            "height": source_manifest.get("frame_height"),
            "offset_x": None,
            "offset_y": None,
            "binning": None,
        },
        "requested_settings": {
            "exposure_us": None,
            "gain_db": None,
            "gamma": None,
        },
        "actual_settings": {
            "exposure_us": None,
            "gain_db": None,
            "gamma": None,
        },
        "trigger_mode": "fictrac_raw_stream",
        "nominal_trigger_fps": recording.get("camera_fps"),
        "started_utc": None,
        "completed_utc": None,
        "expected_frame_count": recording.get("expected_frames"),
        "saved_frame_count": recording.get("saved_raw_frames"),
        "frame_index_path": frame_index_path,
        "frame_stream_path": frame_stream_path,
        "chunk_paths": [],
        "lossless_video_path": lossless_path,
        "artifacts": artifacts,
        "retention_state": {
            "raw_chunks_retained": bool(recording.get("raw_chunks_retained", False)),
            "cleanup_policy": cleanup.get("policy"),
            "cleanup_applied": bool(cleanup.get("applied", False)),
            "deleted_artifact_paths": [Path(str(path)).name for path in cleanup.get("deleted_chunk_paths", [])],
            "validated_by_parity_audit": cleanup.get("parity_summary_path") is not None,
            "parity_audit_path": "derived/validation/parity_audit.json" if cleanup.get("parity_summary_path") else None,
        },
        "integrity_summary": {
            "frame_count_matches_expected": recording.get("missing_frames_vs_expected") == 0 if recording.get("expected_frames") is not None else None,
            "missing_frames_vs_expected": recording.get("missing_frames_vs_expected"),
            "no_dropped_frames": recording.get("no_dropped_frames"),
            "source_fps": source_manifest.get("fps"),
            "saved_fps": (lossless_video or {}).get("fps"),
        },
    }
    layout.fictrac_camera_manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if frame_index_path:
        layout.fictrac_camera_frame_index_meta_path.write_text(
            json.dumps(
                build_table_meta_payload(
                    schema_name="multibios.camera_frame_index_meta",
                    artifact_role="primary_evidence",
                    table_path=frame_index_path,
                    clock_domain="daq_sample_clock",
                    columns=[
                        {"name": "frame_index", "unit": "frame", "description": "Saved frame ordinal."},
                        {"name": "log_frame", "unit": "frame", "description": "FicTrac log frame number when available."},
                        {"name": "chunk_index", "unit": "chunk", "description": "Raw chunk ordinal."},
                        {"name": "chunk_frame_index", "unit": "frame", "description": "Frame ordinal within the raw chunk."},
                    ],
                ),
                indent=2,
            ),
            encoding="utf-8",
        )
    if frame_stream_path:
        layout.fictrac_camera_frame_stream_meta_path.write_text(
            json.dumps(
                build_array_meta_payload(
                    schema_name="multibios.camera_frame_stream_meta",
                    artifact_role="primary_evidence",
                    data_path=frame_stream_path,
                    dtype=str(source_manifest.get("dtype") or "uint8"),
                    shape=[
                        int(recording.get("saved_raw_frames") or 0),
                        int(source_manifest.get("frame_height") or 0),
                        int(source_manifest.get("frame_width") or 0),
                        int(source_manifest.get("channels", 3) or 3),
                    ],
                    axis_order=["frame", "y", "x", "channel"],
                    clock_domain="daq_sample_clock",
                    sample_rate_hz=recording.get("camera_fps"),
                    value_unit=None,
                ),
                indent=2,
            ),
            encoding="utf-8",
        )
    if lossless_path:
        layout.fictrac_camera_lossless_video_meta_path.write_text(
            json.dumps(
                {
                    "schema_name": "multibios.camera_video_meta",
                    "schema_version": SCHEMA_VERSION,
                    "artifact_role": "validated_access_copy",
                    "video_path": lossless_path,
                    "nominal_fps": (lossless_video or {}).get("fps"),
                    "source_manifest_path": normalize_run_relative_path(run_dir, layout.fictrac_camera_manifest_path),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    return manifest


def _guess_media_type(path: Path) -> str:
    suffix = path.suffix.lower()
    return {
        ".json": "application/json",
        ".jsonl": "application/x-ndjson",
        ".yaml": "application/yaml",
        ".csv": "text/csv",
        ".txt": "text/plain",
        ".npz": "application/x-npz",
        ".bin": "application/octet-stream",
        ".avi": "video/x-msvideo",
        ".png": "image/png",
        ".dat": "text/plain",
        ".html": "text/html",
        ".md": "text/markdown",
        ".sha256": "text/plain",
    }.get(suffix, "application/octet-stream")