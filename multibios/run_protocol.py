#!/usr/bin/env python3
"""
Run hardware-clocked NI USB-6353 protocol and log MFC analog feedback + synchronized DI rails.

- DO (master): drives S bits, LOAD_REQ, RCK, triggers
- AO (slave): drives MFC setpoints
- AI (slave): records MFC feedback (0–10 V) locked to DO sample clock
- DI (slave): records synchronized digital input rails, locked to DO sample clock

Artifacts are written to data/runs/YYYY-MM-DD_HH-MM-SS/
- compiled_do.npz / compiled_ao.npz
- capture_ai.npz (MFC feedback, optional)
- capture_di.npz (digital input rails, optional)
- do_map.json / ao_map.json / di_map.json
- rck_edges.csv (planned commits)
- digital_edges.csv (rising/falling edges for all DO lines)
- di_edges.csv (rising/falling edges for DI lines, if present)
- preview.html (interactive Plotly: DO + AO + AI/DI overlays)
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import subprocess
import sys
import threading
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import cv2
import nidaqmx
import numpy as np
# Plotly
import plotly.graph_objects as go
import yaml
from nidaqmx.constants import AcquisitionType, Edge, LineGrouping
from nidaqmx.stream_readers import AnalogMultiChannelReader
from nidaqmx.stream_writers import AnalogMultiChannelWriter
from plotly.subplots import make_subplots

# FicTrac / camera helpers
from multibios.fictrac_client import (FICTRAC_FRAME_DTYPE, BaseFicTracCallback,
                                      FicTracDriver, FicTracFrame,
                                      FicTracFrameStore)
from multibios.fictrac_config import resolve_fictrac_config_path
from multibios.fictrac_consumer import ClosedLoopFrameConsumer
from multibios.fictrac_raw_recording import postprocess_fictrac_raw_recording
from multibios.fictrac_runtime import prepare_fictrac_runtime
from multibios.parity_audit import summarize_run_parity
from multibios.protocol.control_plan import (compile_control_plan,
                                             write_control_plan_csv)
# Compiler
from multibios.protocol.schema import (CompileError, ProtocolCompiler,
                                       TimingConfig)
from multibios.serial_line_monitor import SerialLineMonitor
# Visualization helpers
from multibios.viz_helpers import make_protocol_figure, write_edge_csv


RAW_CHUNK_RETENTION_POLICIES = {"keep", "delete_after_parity"}


def _read_yaml_text(path: Path) -> Any:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _warn_ignored_protocol_timing_key(key: str, hardware_key: str) -> None:
    warnings.warn(
        f"protocol.timing.{key} is ignored; use {hardware_key} in config/hardware.yaml",
        UserWarning,
        stacklevel=3,
    )


def _apply_hardware_owned_camera_timing(timing_block: dict[str, Any], cfg: RunProtocolConfig) -> None:
    if "camera_interval" in timing_block:
        _warn_ignored_protocol_timing_key("camera_interval", "camera_recording.trigger_fps_hz")
        timing_block.pop("camera_interval", None)
    if "camera_pulse_duration" in timing_block:
        _warn_ignored_protocol_timing_key("camera_pulse_duration", "camera_recording.trigger_pulse_ms")
        timing_block.pop("camera_pulse_duration", None)

    if cfg.camera_trigger_fps_hz is not None and cfg.camera_trigger_fps_hz > 0:
        timing_block["camera_interval"] = 1000.0 / cfg.camera_trigger_fps_hz
    else:
        timing_block["camera_interval"] = 0.0

    if cfg.camera_trigger_pulse_ms is not None and cfg.camera_trigger_pulse_ms > 0:
        timing_block["camera_pulse_duration"] = float(cfg.camera_trigger_pulse_ms)


@dataclass
class RunProtocolConfig:
    teensy_port: str = ""
    teensy_baud: int = 115_200
    capture_teensy_serial: bool = False
    fictrac_config: str = ""
    fictrac_bin: str = ""
    fictrac_console_out: str = "fictrac_output.txt"
    fictrac_camera_serial: str = ""
    fictrac_first_frame_timeout_ms: int = 0
    fictrac_arm_delay_s: float = 0.5
    fictrac_startup_timeout_s: float = 90.0
    fictrac_timeout_s: float = 5.0
    blackfly_exposure_us: float | None = None
    blackfly_roi_width: int | None = None
    blackfly_roi_height: int | None = None
    blackfly_binning: int = 1
    blackfly_gain_db: float | None = None
    blackfly_gamma: float | None = None
    save_fictrac_camera_video: bool = False
    fictrac_raw_video_codec: str = "raw"
    save_second_camera_video: bool = False
    camera_trigger_fps_hz: float | None = None
    camera_trigger_pulse_ms: int | None = None
    second_camera_index: int | None = None
    second_camera_serial: str = ""
    second_camera_timeout_ms: int = 250
    second_camera_queue_size: int = 512
    second_camera_stream_buffer_count: int = 256
    second_camera_exposure_us: float | None = None
    second_camera_roi_width: int | None = None
    second_camera_roi_height: int | None = None
    second_camera_binning: int = 1
    second_camera_gain_db: float | None = None
    second_camera_gamma: float | None = None
    verify_camera_recording: bool = True
    convert_second_camera_bin_to_lossless_mkv: bool = True
    raw_chunk_retention_policy: str = "keep"
    data_dir: str = "data/runs"


class ExperimentCallback(BaseFicTracCallback):
    """FicTrac callback with efficient frame retention for logging and control."""

    def __init__(self) -> None:
        self._store = FicTracFrameStore(chunk_size=8192, recent_capacity=2048)
        self._stop = threading.Event()

    @property
    def latest(self) -> Optional[FicTracFrame]:
        return self._store.latest

    @property
    def frame_count(self) -> int:
        return self._store.count

    def get_latest(self) -> tuple[int, Optional[FicTracFrame]]:
        return self._store.get_latest()

    def wait_for_next_frame(
        self, after_seq: int = -1, timeout: float | None = None
    ) -> tuple[int, Optional[FicTracFrame]]:
        return self._store.wait_for_next(after_seq=after_seq, timeout=timeout)

    def recent_frames(self, max_count: int | None = None) -> np.ndarray:
        return self._store.recent_array(max_count=max_count)

    def frame_array(self) -> np.ndarray:
        return self._store.frame_array()

    def save_npz(self, path: str | Path) -> int:
        return self._store.save_npz(path)

    def make_consumer(self, *, start_at_latest: bool = False) -> ClosedLoopFrameConsumer:
        return ClosedLoopFrameConsumer(self._store, start_at_latest=start_at_latest)

    def process_callback(self, track_state) -> bool:
        self._store.append(FicTracFrame.from_state(track_state, time.perf_counter()))
        return not self._stop.is_set()

    def request_stop(self) -> None:
        self._stop.set()

    def stop_requested(self) -> bool:
        return self._stop.is_set()


def _read_fictrac_config_values(path: str | Path) -> dict[str, str]:
    values: dict[str, str] = {}
    with open(path, encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#") or ":" not in line:
                continue
            key, _, value = line.partition(":")
            values[key.strip()] = value.strip()
    return values


def _upsert_fictrac_config_line(lines: list[str], key: str, value: str) -> None:
    rendered = f"{key:<17}: {value}\n"
    for idx, raw_line in enumerate(lines):
        line = raw_line.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        current_key, _, _ = line.partition(":")
        if current_key.strip() == key:
            lines[idx] = rendered
            return
    lines.append(rendered)


def _prepare_fictrac_runtime_config(
    source_config_path: str | Path,
    run_dir: str | Path,
    *,
    enable_raw_video: bool,
    camera_fps: float | None,
    video_codec: str,
    first_frame_timeout_ms: int,
    force_headless: bool = False,
    camera_index_override: int | None = None,
) -> tuple[Path, int | None, dict[str, Any]]:
    source_path = Path(source_config_path)
    target_dir = Path(run_dir)
    lines = source_path.read_text(encoding="utf-8").splitlines(keepends=True)
    values = _read_fictrac_config_values(source_path)

    fictrac_camera_index: int | None = None
    src_fn = values.get("src_fn")
    if src_fn is not None:
        try:
            fictrac_camera_index = int(src_fn)
        except ValueError:
            fictrac_camera_index = None

    if camera_index_override is not None:
        fictrac_camera_index = int(camera_index_override)
        _upsert_fictrac_config_line(lines, "src_fn", str(fictrac_camera_index))

    output_base = (target_dir.resolve() / "fictrac").as_posix()
    _upsert_fictrac_config_line(lines, "output_fn", output_base)
    _upsert_fictrac_config_line(lines, "src_first_frame_timeout_ms", str(int(first_frame_timeout_ms)))
    if enable_raw_video:
        _upsert_fictrac_config_line(lines, "save_raw", "y")
        _upsert_fictrac_config_line(lines, "vid_codec", video_codec)
        if camera_fps is not None and camera_fps > 0:
            _upsert_fictrac_config_line(lines, "src_fps", f"{camera_fps:.6f}")
    if force_headless:
        _upsert_fictrac_config_line(lines, "do_display", "n")

    runtime_path = target_dir / "fictrac_runtime_config.txt"
    runtime_path.write_text("".join(lines), encoding="utf-8")

    return runtime_path, fictrac_camera_index, {
        "source_config": str(source_path),
        "runtime_config": str(runtime_path),
        "output_base": output_base,
        "save_raw": bool(enable_raw_video),
        "video_codec": video_codec,
        "camera_fps": camera_fps,
        "first_frame_timeout_ms": int(first_frame_timeout_ms),
        "force_headless": bool(force_headless),
        "fictrac_camera_index": fictrac_camera_index,
    }


def _load_yaml_file(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    yaml_path = Path(path)
    if not yaml_path.exists():
        raise FileNotFoundError(f"YAML file not found: {yaml_path}")
    with open(yaml_path, encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML file must contain a mapping at top level: {yaml_path}")
    return data


def _yaml_section(raw: dict[str, Any], key: str) -> dict[str, Any]:
    value = raw.get(key, {})
    return value if isinstance(value, dict) else {}


def _value_or_fallback(primary: Any, fallback: Any) -> Any:
    return fallback if primary is None else primary


def _warn_deprecated_experiment_key(
    key: str,
    hardware_path: str | Path | None,
    target_block: str,
) -> None:
    target = str(hardware_path) if hardware_path is not None else "config/hardware.yaml"
    warnings.warn(
        f"experiment_config key '{key}' is deprecated; move it to the {target_block} block in {target}",
        DeprecationWarning,
        stacklevel=3,
    )


_EXPERIMENT_HARDWARE_OVERRIDE_TARGETS: dict[str, str] = {
    "teensy_port": "teensy.port",
    "teensy_baud": "teensy.baud",
    "capture_teensy_serial": "teensy.capture_serial",
    "fictrac_config": "fictrac.config",
    "fictrac_bin": "fictrac.bin",
    "fictrac_console_out": "fictrac.console_out",
    "fictrac_camera_serial": "fictrac.camera_serial",
    "fictrac_first_frame_timeout_ms": "fictrac.first_frame_timeout_ms",
    "fictrac_target_fps": "camera_recording.trigger_fps_hz",
    "fictrac_arm_delay_s": "fictrac.arm_delay_s",
    "fictrac_startup_timeout_s": "fictrac.startup_timeout_s",
    "fictrac_timeout_s": "fictrac.timeout_s",
    "blackfly_exposure_us": "blackfly_defaults.exposure_us",
    "blackfly_roi_width": "blackfly_defaults.roi_width",
    "blackfly_roi_height": "blackfly_defaults.roi_height",
    "blackfly_binning": "blackfly_defaults.binning",
    "blackfly_gain_db": "blackfly_defaults.gain_db",
    "blackfly_gamma": "blackfly_defaults.gamma",
    "save_fictrac_camera_video": "camera_recording.save_fictrac_camera_video",
    "save_camera_raw_video": "camera_recording.save_fictrac_camera_video",
    "fictrac_raw_video_codec": "camera_recording.fictrac_raw_video_codec",
    "save_second_camera_video": "camera_recording.save_second_camera_video",
    "camera_trigger_fps_hz": "camera_recording.trigger_fps_hz",
    "camera_trigger_pulse_ms": "camera_recording.trigger_pulse_ms",
    "second_camera_index": "camera_recording.second_camera_index",
    "second_camera_serial": "camera_recording.second_camera_serial",
    "second_camera_timeout_ms": "camera_recording.second_camera_timeout_ms",
    "other_camera_timeout_ms": "camera_recording.second_camera_timeout_ms",
    "second_camera_queue_size": "camera_recording.second_camera_queue_size",
    "other_camera_queue_size": "camera_recording.second_camera_queue_size",
    "second_camera_stream_buffer_count": "camera_recording.second_camera_stream_buffer_count",
    "other_camera_stream_buffer_count": "camera_recording.second_camera_stream_buffer_count",
    "second_camera_exposure_us": "camera_recording.second_camera_exposure_us or blackfly_defaults.exposure_us",
    "other_camera_exposure_us": "camera_recording.second_camera_exposure_us or blackfly_defaults.exposure_us",
    "second_camera_roi_width": "camera_recording.second_camera_roi_width or blackfly_defaults.roi_width",
    "other_camera_roi_width": "camera_recording.second_camera_roi_width or blackfly_defaults.roi_width",
    "second_camera_roi_height": "camera_recording.second_camera_roi_height or blackfly_defaults.roi_height",
    "other_camera_roi_height": "camera_recording.second_camera_roi_height or blackfly_defaults.roi_height",
    "second_camera_binning": "camera_recording.second_camera_binning or blackfly_defaults.binning",
    "other_camera_binning": "camera_recording.second_camera_binning or blackfly_defaults.binning",
    "second_camera_gain_db": "camera_recording.second_camera_gain_db or blackfly_defaults.gain_db",
    "other_camera_gain_db": "camera_recording.second_camera_gain_db or blackfly_defaults.gain_db",
    "second_camera_gamma": "camera_recording.second_camera_gamma or blackfly_defaults.gamma",
    "other_camera_gamma": "camera_recording.second_camera_gamma or blackfly_defaults.gamma",
    "verify_camera_recording": "camera_recording.verify_no_dropped_frames",
    "convert_second_camera_bin_to_lossless_mkv": "camera_recording.convert_second_camera_bin_to_lossless_mkv",
    "raw_chunk_retention_policy": "camera_recording.raw_chunk_retention_policy",
    "delete_raw_chunks_after_parity": "camera_recording.raw_chunk_retention_policy",
    "data_dir": "data_output.data_dir",
}


def _reject_experiment_hardware_overrides(
    raw: dict[str, Any],
    *,
    config_path: str | Path | None,
    hardware_path: str | Path | None,
) -> None:
    violations = [
        f"{key} -> {target}"
        for key, target in _EXPERIMENT_HARDWARE_OVERRIDE_TARGETS.items()
        if key in raw
    ]
    if not violations:
        return

    config_label = str(config_path) if config_path is not None else "experiment config"
    hardware_label = str(hardware_path) if hardware_path is not None else "config/hardware.yaml"
    raise ValueError(
        f"Experiment-level hardware overrides are not allowed in {config_label}. "
        f"Use {hardware_label} and config/config_camera.txt as the single source of truth. "
        f"Invalid keys: {', '.join(sorted(violations))}"
    )


def load_run_protocol_config(
    path: str | Path | None,
    hardware_path: str | Path | None = None,
) -> RunProtocolConfig:
    raw = _load_yaml_file(path)
    _reject_experiment_hardware_overrides(raw, config_path=path, hardware_path=hardware_path)
    hardware = _load_yaml_file(hardware_path)
    hardware_teensy = _yaml_section(hardware, "teensy")
    hardware_fictrac = _yaml_section(hardware, "fictrac")
    hardware_blackfly = _yaml_section(hardware, "blackfly_defaults")
    hardware_camera_recording = _yaml_section(hardware, "camera_recording")
    hardware_data_output = _yaml_section(hardware, "data_output")

    cfg = RunProtocolConfig()

    cfg.teensy_port = str(hardware_teensy.get("port", cfg.teensy_port))
    if "teensy_port" in raw:
        _warn_deprecated_experiment_key("teensy_port", hardware_path, "teensy")
        cfg.teensy_port = str(raw["teensy_port"])

    cfg.teensy_baud = int(hardware_teensy.get("baud", cfg.teensy_baud))
    if "teensy_baud" in raw:
        _warn_deprecated_experiment_key("teensy_baud", hardware_path, "teensy")
        cfg.teensy_baud = int(raw["teensy_baud"])

    cfg.capture_teensy_serial = bool(hardware_teensy.get("capture_serial", cfg.capture_teensy_serial))
    if "capture_teensy_serial" in raw:
        _warn_deprecated_experiment_key("capture_teensy_serial", hardware_path, "teensy")
        cfg.capture_teensy_serial = bool(raw["capture_teensy_serial"])

    cfg.fictrac_config = str(hardware_fictrac.get("config", cfg.fictrac_config))
    if "fictrac_config" in raw:
        _warn_deprecated_experiment_key("fictrac_config", hardware_path, "fictrac")
        cfg.fictrac_config = str(raw["fictrac_config"])
    cfg.fictrac_config = str(resolve_fictrac_config_path(cfg.fictrac_config, hardware_path=hardware_path))

    cfg.fictrac_bin = str(hardware_fictrac.get("bin", cfg.fictrac_bin))
    if "fictrac_bin" in raw:
        _warn_deprecated_experiment_key("fictrac_bin", hardware_path, "fictrac")
        cfg.fictrac_bin = str(raw["fictrac_bin"])

    cfg.fictrac_console_out = str(hardware_fictrac.get("console_out", cfg.fictrac_console_out))
    if "fictrac_console_out" in raw:
        _warn_deprecated_experiment_key("fictrac_console_out", hardware_path, "fictrac")
        cfg.fictrac_console_out = str(raw["fictrac_console_out"])

    cfg.fictrac_camera_serial = str(hardware_fictrac.get("camera_serial", cfg.fictrac_camera_serial) or "")
    if "fictrac_camera_serial" in raw:
        _warn_deprecated_experiment_key("fictrac_camera_serial", hardware_path, "fictrac")
        cfg.fictrac_camera_serial = str(raw["fictrac_camera_serial"])

    cfg.fictrac_first_frame_timeout_ms = int(
        hardware_fictrac.get("first_frame_timeout_ms", cfg.fictrac_first_frame_timeout_ms)
    )
    if "fictrac_first_frame_timeout_ms" in raw:
        _warn_deprecated_experiment_key("fictrac_first_frame_timeout_ms", hardware_path, "fictrac")
        cfg.fictrac_first_frame_timeout_ms = int(raw["fictrac_first_frame_timeout_ms"])

    if "target_fps" in hardware_fictrac:
        raise ValueError(
            f"fictrac.target_fps is no longer supported in {hardware_path}; "
            "use camera_recording.trigger_fps_hz as the single source of truth"
        )

    cfg.fictrac_arm_delay_s = float(hardware_fictrac.get("arm_delay_s", cfg.fictrac_arm_delay_s))
    if "fictrac_arm_delay_s" in raw:
        _warn_deprecated_experiment_key("fictrac_arm_delay_s", hardware_path, "fictrac")
        cfg.fictrac_arm_delay_s = float(raw["fictrac_arm_delay_s"])

    cfg.fictrac_startup_timeout_s = float(
        hardware_fictrac.get("startup_timeout_s", cfg.fictrac_startup_timeout_s)
    )
    if "fictrac_startup_timeout_s" in raw:
        _warn_deprecated_experiment_key("fictrac_startup_timeout_s", hardware_path, "fictrac")
        cfg.fictrac_startup_timeout_s = float(raw["fictrac_startup_timeout_s"])

    cfg.fictrac_timeout_s = float(hardware_fictrac.get("timeout_s", cfg.fictrac_timeout_s))
    if "fictrac_timeout_s" in raw:
        _warn_deprecated_experiment_key("fictrac_timeout_s", hardware_path, "fictrac")
        cfg.fictrac_timeout_s = float(raw["fictrac_timeout_s"])

    blackfly_exposure = hardware_blackfly.get("exposure_us", cfg.blackfly_exposure_us)
    cfg.blackfly_exposure_us = None if blackfly_exposure is None else float(blackfly_exposure)
    if "blackfly_exposure_us" in raw:
        _warn_deprecated_experiment_key("blackfly_exposure_us", hardware_path, "blackfly_defaults")
        cfg.blackfly_exposure_us = float(raw["blackfly_exposure_us"])

    blackfly_roi_width = hardware_blackfly.get("roi_width", cfg.blackfly_roi_width)
    cfg.blackfly_roi_width = None if blackfly_roi_width is None else int(blackfly_roi_width)
    if "blackfly_roi_width" in raw:
        _warn_deprecated_experiment_key("blackfly_roi_width", hardware_path, "blackfly_defaults")
        cfg.blackfly_roi_width = int(raw["blackfly_roi_width"])

    blackfly_roi_height = hardware_blackfly.get("roi_height", cfg.blackfly_roi_height)
    cfg.blackfly_roi_height = None if blackfly_roi_height is None else int(blackfly_roi_height)
    if "blackfly_roi_height" in raw:
        _warn_deprecated_experiment_key("blackfly_roi_height", hardware_path, "blackfly_defaults")
        cfg.blackfly_roi_height = int(raw["blackfly_roi_height"])

    cfg.blackfly_binning = int(hardware_blackfly.get("binning", cfg.blackfly_binning))
    if "blackfly_binning" in raw:
        _warn_deprecated_experiment_key("blackfly_binning", hardware_path, "blackfly_defaults")
        cfg.blackfly_binning = int(raw["blackfly_binning"])

    blackfly_gain = hardware_blackfly.get("gain_db", cfg.blackfly_gain_db)
    cfg.blackfly_gain_db = None if blackfly_gain is None else float(blackfly_gain)
    if "blackfly_gain_db" in raw:
        _warn_deprecated_experiment_key("blackfly_gain_db", hardware_path, "blackfly_defaults")
        cfg.blackfly_gain_db = float(raw["blackfly_gain_db"])

    blackfly_gamma = hardware_blackfly.get("gamma", cfg.blackfly_gamma)
    cfg.blackfly_gamma = None if blackfly_gamma is None else float(blackfly_gamma)
    if "blackfly_gamma" in raw:
        _warn_deprecated_experiment_key("blackfly_gamma", hardware_path, "blackfly_defaults")
        cfg.blackfly_gamma = float(raw["blackfly_gamma"])

    cfg.save_fictrac_camera_video = bool(
        hardware_camera_recording.get(
            "save_fictrac_camera_video",
            hardware_camera_recording.get("save_camera_raw_video", cfg.save_fictrac_camera_video),
        )
    )
    if "save_fictrac_camera_video" in raw:
        _warn_deprecated_experiment_key("save_fictrac_camera_video", hardware_path, "camera_recording")
        cfg.save_fictrac_camera_video = bool(raw["save_fictrac_camera_video"])
    elif "save_camera_raw_video" in raw:
        _warn_deprecated_experiment_key("save_camera_raw_video", hardware_path, "camera_recording")
        cfg.save_fictrac_camera_video = bool(raw["save_camera_raw_video"])

    cfg.fictrac_raw_video_codec = str(
        hardware_camera_recording.get("fictrac_raw_video_codec", cfg.fictrac_raw_video_codec)
    )
    if "fictrac_raw_video_codec" in raw:
        _warn_deprecated_experiment_key("fictrac_raw_video_codec", hardware_path, "camera_recording")
        cfg.fictrac_raw_video_codec = str(raw["fictrac_raw_video_codec"])

    cfg.save_second_camera_video = bool(
        hardware_camera_recording.get("save_second_camera_video", cfg.save_second_camera_video)
    )
    if "save_second_camera_video" in raw:
        _warn_deprecated_experiment_key("save_second_camera_video", hardware_path, "camera_recording")
        cfg.save_second_camera_video = bool(raw["save_second_camera_video"])

    camera_trigger_fps_hz = hardware_camera_recording.get("trigger_fps_hz", cfg.camera_trigger_fps_hz)
    cfg.camera_trigger_fps_hz = None if camera_trigger_fps_hz is None else float(camera_trigger_fps_hz)
    if "camera_trigger_fps_hz" in raw:
        _warn_deprecated_experiment_key("camera_trigger_fps_hz", hardware_path, "camera_recording")
        cfg.camera_trigger_fps_hz = float(raw["camera_trigger_fps_hz"])

    camera_trigger_pulse_ms = hardware_camera_recording.get("trigger_pulse_ms", cfg.camera_trigger_pulse_ms)
    cfg.camera_trigger_pulse_ms = None if camera_trigger_pulse_ms is None else int(camera_trigger_pulse_ms)
    if "camera_trigger_pulse_ms" in raw:
        _warn_deprecated_experiment_key("camera_trigger_pulse_ms", hardware_path, "camera_recording")
        cfg.camera_trigger_pulse_ms = int(raw["camera_trigger_pulse_ms"])

    second_camera_index = hardware_camera_recording.get("second_camera_index", cfg.second_camera_index)
    cfg.second_camera_index = None if second_camera_index is None else int(second_camera_index)
    if "second_camera_index" in raw:
        _warn_deprecated_experiment_key("second_camera_index", hardware_path, "camera_recording")
        cfg.second_camera_index = int(raw["second_camera_index"])

    cfg.second_camera_serial = str(hardware_camera_recording.get("second_camera_serial", cfg.second_camera_serial) or "")
    if "second_camera_serial" in raw:
        _warn_deprecated_experiment_key("second_camera_serial", hardware_path, "camera_recording")
        cfg.second_camera_serial = str(raw["second_camera_serial"])

    cfg.second_camera_timeout_ms = int(
        hardware_camera_recording.get("second_camera_timeout_ms", cfg.second_camera_timeout_ms)
    )
    if "second_camera_timeout_ms" in raw:
        _warn_deprecated_experiment_key("second_camera_timeout_ms", hardware_path, "camera_recording")
        cfg.second_camera_timeout_ms = int(raw["second_camera_timeout_ms"])
    elif "other_camera_timeout_ms" in raw:
        _warn_deprecated_experiment_key("other_camera_timeout_ms", hardware_path, "camera_recording")
        cfg.second_camera_timeout_ms = int(raw["other_camera_timeout_ms"])

    cfg.second_camera_queue_size = int(
        hardware_camera_recording.get("second_camera_queue_size", cfg.second_camera_queue_size)
    )
    if "second_camera_queue_size" in raw:
        _warn_deprecated_experiment_key("second_camera_queue_size", hardware_path, "camera_recording")
        cfg.second_camera_queue_size = int(raw["second_camera_queue_size"])
    elif "other_camera_queue_size" in raw:
        _warn_deprecated_experiment_key("other_camera_queue_size", hardware_path, "camera_recording")
        cfg.second_camera_queue_size = int(raw["other_camera_queue_size"])

    cfg.second_camera_stream_buffer_count = int(
        hardware_camera_recording.get(
            "second_camera_stream_buffer_count",
            cfg.second_camera_stream_buffer_count,
        )
    )
    if "second_camera_stream_buffer_count" in raw:
        _warn_deprecated_experiment_key("second_camera_stream_buffer_count", hardware_path, "camera_recording")
        cfg.second_camera_stream_buffer_count = int(raw["second_camera_stream_buffer_count"])
    elif "other_camera_stream_buffer_count" in raw:
        _warn_deprecated_experiment_key("other_camera_stream_buffer_count", hardware_path, "camera_recording")
        cfg.second_camera_stream_buffer_count = int(raw["other_camera_stream_buffer_count"])

    second_camera_exposure = _value_or_fallback(
        hardware_camera_recording.get("second_camera_exposure_us"),
        hardware_blackfly.get("exposure_us", cfg.second_camera_exposure_us),
    )
    cfg.second_camera_exposure_us = None if second_camera_exposure is None else float(second_camera_exposure)
    if "second_camera_exposure_us" in raw:
        _warn_deprecated_experiment_key("second_camera_exposure_us", hardware_path, "camera_recording")
        cfg.second_camera_exposure_us = float(raw["second_camera_exposure_us"])
    elif "other_camera_exposure_us" in raw:
        _warn_deprecated_experiment_key("other_camera_exposure_us", hardware_path, "camera_recording")
        cfg.second_camera_exposure_us = float(raw["other_camera_exposure_us"])

    second_camera_roi_width = _value_or_fallback(
        hardware_camera_recording.get("second_camera_roi_width"),
        hardware_blackfly.get("roi_width", cfg.second_camera_roi_width),
    )
    cfg.second_camera_roi_width = None if second_camera_roi_width is None else int(second_camera_roi_width)
    if "second_camera_roi_width" in raw:
        _warn_deprecated_experiment_key("second_camera_roi_width", hardware_path, "camera_recording")
        cfg.second_camera_roi_width = int(raw["second_camera_roi_width"])
    elif "other_camera_roi_width" in raw:
        _warn_deprecated_experiment_key("other_camera_roi_width", hardware_path, "camera_recording")
        cfg.second_camera_roi_width = int(raw["other_camera_roi_width"])

    second_camera_roi_height = _value_or_fallback(
        hardware_camera_recording.get("second_camera_roi_height"),
        hardware_blackfly.get("roi_height", cfg.second_camera_roi_height),
    )
    cfg.second_camera_roi_height = None if second_camera_roi_height is None else int(second_camera_roi_height)
    if "second_camera_roi_height" in raw:
        _warn_deprecated_experiment_key("second_camera_roi_height", hardware_path, "camera_recording")
        cfg.second_camera_roi_height = int(raw["second_camera_roi_height"])
    elif "other_camera_roi_height" in raw:
        _warn_deprecated_experiment_key("other_camera_roi_height", hardware_path, "camera_recording")
        cfg.second_camera_roi_height = int(raw["other_camera_roi_height"])

    cfg.second_camera_binning = int(
        hardware_camera_recording.get(
            "second_camera_binning",
            cfg.blackfly_binning,
        )
    )
    if "second_camera_binning" in raw:
        _warn_deprecated_experiment_key("second_camera_binning", hardware_path, "camera_recording")
        cfg.second_camera_binning = int(raw["second_camera_binning"])
    elif "other_camera_binning" in raw:
        _warn_deprecated_experiment_key("other_camera_binning", hardware_path, "camera_recording")
        cfg.second_camera_binning = int(raw["other_camera_binning"])

    second_camera_gain = _value_or_fallback(
        hardware_camera_recording.get("second_camera_gain_db"),
        hardware_blackfly.get("gain_db", cfg.second_camera_gain_db),
    )
    cfg.second_camera_gain_db = None if second_camera_gain is None else float(second_camera_gain)
    if "second_camera_gain_db" in raw:
        _warn_deprecated_experiment_key("second_camera_gain_db", hardware_path, "camera_recording")
        cfg.second_camera_gain_db = float(raw["second_camera_gain_db"])
    elif "other_camera_gain_db" in raw:
        _warn_deprecated_experiment_key("other_camera_gain_db", hardware_path, "camera_recording")
        cfg.second_camera_gain_db = float(raw["other_camera_gain_db"])

    second_camera_gamma = _value_or_fallback(
        hardware_camera_recording.get("second_camera_gamma"),
        hardware_blackfly.get("gamma", cfg.second_camera_gamma),
    )
    cfg.second_camera_gamma = None if second_camera_gamma is None else float(second_camera_gamma)
    if "second_camera_gamma" in raw:
        _warn_deprecated_experiment_key("second_camera_gamma", hardware_path, "camera_recording")
        cfg.second_camera_gamma = float(raw["second_camera_gamma"])
    elif "other_camera_gamma" in raw:
        _warn_deprecated_experiment_key("other_camera_gamma", hardware_path, "camera_recording")
        cfg.second_camera_gamma = float(raw["other_camera_gamma"])

    cfg.verify_camera_recording = bool(
        hardware_camera_recording.get("verify_no_dropped_frames", cfg.verify_camera_recording)
    )
    if "verify_camera_recording" in raw:
        _warn_deprecated_experiment_key("verify_camera_recording", hardware_path, "camera_recording")
        cfg.verify_camera_recording = bool(raw["verify_camera_recording"])

    cfg.convert_second_camera_bin_to_lossless_mkv = bool(
        hardware_camera_recording.get(
            "convert_second_camera_bin_to_lossless_mkv",
            cfg.convert_second_camera_bin_to_lossless_mkv,
        )
    )
    if "convert_second_camera_bin_to_lossless_mkv" in raw:
        _warn_deprecated_experiment_key(
            "convert_second_camera_bin_to_lossless_mkv",
            hardware_path,
            "camera_recording",
        )
        cfg.convert_second_camera_bin_to_lossless_mkv = bool(raw["convert_second_camera_bin_to_lossless_mkv"])

    raw_chunk_retention_policy = str(
        hardware_camera_recording.get("raw_chunk_retention_policy", cfg.raw_chunk_retention_policy) or cfg.raw_chunk_retention_policy
    ).strip().lower()
    if raw_chunk_retention_policy not in RAW_CHUNK_RETENTION_POLICIES:
        allowed = ", ".join(sorted(RAW_CHUNK_RETENTION_POLICIES))
        raise ValueError(
            f"camera_recording.raw_chunk_retention_policy must be one of: {allowed}; got {raw_chunk_retention_policy!r}"
        )
    cfg.raw_chunk_retention_policy = raw_chunk_retention_policy
    if "raw_chunk_retention_policy" in raw:
        _warn_deprecated_experiment_key("raw_chunk_retention_policy", hardware_path, "camera_recording")
        cfg.raw_chunk_retention_policy = str(raw["raw_chunk_retention_policy"]).strip().lower()
    if "delete_raw_chunks_after_parity" in raw:
        _warn_deprecated_experiment_key("delete_raw_chunks_after_parity", hardware_path, "camera_recording")
        cfg.raw_chunk_retention_policy = "delete_after_parity" if bool(raw["delete_raw_chunks_after_parity"]) else "keep"

    cfg.data_dir = str(hardware_data_output.get("data_dir", cfg.data_dir))
    if "data_dir" in raw:
        _warn_deprecated_experiment_key("data_dir", hardware_path, "data_output")
        cfg.data_dir = str(raw["data_dir"])

    return cfg


def _count_rising_edges(trace: np.ndarray | None) -> int | None:
    if trace is None:
        return None
    if trace.size == 0:
        return 0
    bool_trace = np.asarray(trace, dtype=np.bool_)
    return int(bool_trace[0]) + int(np.count_nonzero(~bool_trace[:-1] & bool_trace[1:]))


def _first_rising_edge_sample(trace: np.ndarray | None) -> int | None:
    if trace is None:
        return None
    if trace.size == 0:
        return None
    bool_trace = np.asarray(trace, dtype=np.bool_)
    if bool_trace[0]:
        return 0
    rising = np.flatnonzero(~bool_trace[:-1] & bool_trace[1:])
    if rising.size == 0:
        return None
    return int(rising[0] + 1)


def _compute_second_camera_startup_timeout_s(
    *,
    first_trigger_sample: int | None,
    sample_rate: int,
    arm_delay_s: float,
    recorder_timeout_ms: int,
) -> float:
    first_trigger_s = 0.0
    if first_trigger_sample is not None and sample_rate > 0:
        first_trigger_s = float(first_trigger_sample) / float(sample_rate)
    return max(
        2.0,
        float(arm_delay_s) + first_trigger_s + (float(recorder_timeout_ms) / 1000.0) + 0.5,
    )


def _compute_fictrac_drain_timeout_s(
    *,
    expected_frame_count: int | None,
    observed_frame_count: int,
    camera_fps: float | None,
) -> float:
    if expected_frame_count is None or expected_frame_count <= observed_frame_count:
        return 0.5

    remaining_frames = expected_frame_count - observed_frame_count
    if camera_fps is None or camera_fps <= 0:
        return max(1.0, 0.05 * remaining_frames)

    return max(0.5, remaining_frames / camera_fps + max(0.25, 8.0 / camera_fps))


def _discover_fictrac_raw_videos(run_dir: Path) -> list[str]:
    return sorted(str(path) for path in run_dir.glob("fictrac-raw-*"))


def _count_fictrac_saved_raw_frames(run_dir: Path) -> int | None:
    vidlog_paths = sorted(run_dir.glob("fictrac-vidLogFrames-*.txt"))
    if not vidlog_paths:
        return None

    lines = [line.strip() for line in vidlog_paths[-1].read_text(encoding="utf-8").splitlines() if line.strip()]
    return len(lines)


def _build_fictrac_recording_summary(
    *,
    run_dir: Path,
    runtime_info: dict[str, Any],
    frame_count: int | None,
    expected_frame_count: int | None,
) -> dict[str, Any]:
    return postprocess_fictrac_raw_recording(
        run_dir=run_dir,
        runtime_info=runtime_info,
        frame_count=frame_count,
        expected_frame_count=expected_frame_count,
        legacy_raw_videos=_discover_fictrac_raw_videos(run_dir),
        legacy_saved_raw_frames=_count_fictrac_saved_raw_frames(run_dir),
    )


def _write_parity_summary(run_dir: Path) -> tuple[dict[str, Any], Path]:
    summary = summarize_run_parity(run_dir)
    parity_path = run_dir / "parity_audit.json"
    parity_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary, parity_path


def _resolve_run_artifact_path(run_dir: Path, path_value: Any) -> Path | None:
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


def _video_reaches_frame(video_path: Path, frame_index: int) -> bool:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        capture.release()
        return False

    try:
        reported_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if reported_frames > 0 and reported_frames <= frame_index:
            return False
        if frame_index > 0 and not capture.set(cv2.CAP_PROP_POS_FRAMES, float(frame_index)):
            return False
        ok, _ = capture.read()
        return bool(ok)
    finally:
        capture.release()


def _recording_chunk_field_names(recording: dict[str, Any]) -> list[str]:
    names: list[str] = []
    if recording.get("raw_stream_chunks") is not None:
        names.append("raw_stream_chunks")
    if recording.get("chunk_paths") is not None:
        names.append("chunk_paths")
    return names


def _recording_chunk_paths(run_dir: Path, recording: dict[str, Any]) -> list[Path]:
    raw_paths: list[Any] = []
    raw_paths.extend(recording.get("raw_stream_chunks", []) or [])
    raw_paths.extend(recording.get("chunk_paths", []) or [])
    if not raw_paths and recording.get("frame_bin_path"):
        raw_paths.append(recording["frame_bin_path"])

    resolved: list[Path] = []
    seen: set[Path] = set()
    for path_value in raw_paths:
        resolved_path = _resolve_run_artifact_path(run_dir, path_value)
        if resolved_path is None or resolved_path in seen:
            continue
        seen.add(resolved_path)
        resolved.append(resolved_path)
    return resolved


def _recording_saved_frame_count(recording: dict[str, Any]) -> int | None:
    for key in ("saved_raw_frames", "saved_frames"):
        value = recording.get(key)
        if value is not None:
            return int(value)
    return None


def _recording_lossless_video_path(run_dir: Path, recording: dict[str, Any]) -> Path | None:
    lossless_video = recording.get("lossless_video") or {}
    return _resolve_run_artifact_path(run_dir, lossless_video.get("path"))


def _parity_summary_allows_chunk_cleanup(
    parity_summary: dict[str, Any],
    *,
    fictrac_recording: dict[str, Any] | None,
    blackfly_recording: dict[str, Any] | None,
) -> tuple[bool, str | None]:
    counts = parity_summary.get("counts", {})
    trigger_count = counts.get("trigger_rising_edges")
    if trigger_count is None:
        return False, "missing_trigger_count"

    required_keys: list[str] = []
    if fictrac_recording:
        required_keys.extend([
            "fictrac_saved_raw_frames",
            "fictrac_udp_frame_cnt",
            "fictrac_callback_frames",
        ])
    if blackfly_recording:
        required_keys.append("second_camera_saved_frames")

    for key in required_keys:
        value = counts.get(key)
        if value is None:
            return False, f"missing_{key}"
        if int(value) != int(trigger_count):
            return False, f"parity_mismatch_{key}"

    return True, None


def _build_chunk_cleanup_metadata(
    *,
    policy: str,
    parity_path: Path,
    applied: bool,
    reason: str | None,
    chunk_paths: list[Path],
    deleted_chunk_bytes: int,
    validated_video_path: Path | None,
    validated_frame_count: int | None,
) -> dict[str, Any]:
    return {
        "policy": policy,
        "applied": applied,
        "reason": reason,
        "deleted_chunk_count": len(chunk_paths) if applied else 0,
        "deleted_chunk_bytes": int(deleted_chunk_bytes) if applied else 0,
        "deleted_chunk_paths": [str(path) for path in chunk_paths] if applied else [],
        "parity_summary_path": str(parity_path),
        "validated_video_path": None if validated_video_path is None else str(validated_video_path),
        "validated_frame_count": validated_frame_count,
    }


def _annotate_recording_after_chunk_cleanup(
    recording: dict[str, Any],
    *,
    cleanup_info: dict[str, Any],
    applied: bool,
) -> dict[str, Any]:
    updated = dict(recording)
    updated["raw_chunk_cleanup"] = cleanup_info
    updated["raw_chunks_retained"] = not applied
    if applied:
        for field_name in _recording_chunk_field_names(updated):
            updated[field_name] = []
        if "frame_bin_path" in updated:
            updated["frame_bin_path"] = None
    return updated


def _annotate_manifest_after_chunk_cleanup(
    run_dir: Path,
    manifest_path_value: Any,
    *,
    cleanup_info: dict[str, Any],
    applied: bool,
) -> None:
    manifest_path = _resolve_run_artifact_path(run_dir, manifest_path_value)
    if manifest_path is None or not manifest_path.exists():
        return

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["raw_chunk_cleanup"] = cleanup_info
    manifest["raw_chunks_retained"] = not applied
    if applied:
        if "chunk_paths" in manifest:
            manifest["chunk_paths"] = []
        if "frame_bin_path" in manifest:
            manifest["frame_bin_path"] = None
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def _finalize_raw_chunk_retention(
    *,
    run_dir: Path,
    policy: str,
    parity_summary: dict[str, Any],
    parity_path: Path,
    fictrac_recording: dict[str, Any] | None,
    blackfly_recording: dict[str, Any] | None,
    logger: logging.Logger,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    if policy == "keep":
        return fictrac_recording, blackfly_recording

    parity_ok, parity_reason = _parity_summary_allows_chunk_cleanup(
        parity_summary,
        fictrac_recording=fictrac_recording,
        blackfly_recording=blackfly_recording,
    )

    def _finalize_one(
        recording: dict[str, Any] | None,
        *,
        system_name: str,
        manifest_key: str,
    ) -> dict[str, Any] | None:
        if recording is None:
            return None

        chunk_paths = _recording_chunk_paths(run_dir, recording)
        saved_frames = _recording_saved_frame_count(recording)
        video_path = _recording_lossless_video_path(run_dir, recording)
        chunk_bytes_before_delete = sum(path.stat().st_size for path in chunk_paths if path.exists())
        applied = False
        if not parity_ok:
            reason = parity_reason
        elif not chunk_paths:
            reason = "no_raw_chunks_found"
        elif saved_frames is None or saved_frames <= 0:
            reason = "missing_saved_frame_count"
        elif video_path is None or not video_path.exists():
            reason = "missing_lossless_video"
        elif not _video_reaches_frame(video_path, saved_frames - 1):
            reason = "lossless_video_failed_validation"
        else:
            reason = None
            applied = True
            for chunk_path in chunk_paths:
                if chunk_path.exists():
                    chunk_path.unlink(missing_ok=True)

        cleanup_info = _build_chunk_cleanup_metadata(
            policy=policy,
            parity_path=parity_path,
            applied=applied,
            reason=reason,
            chunk_paths=chunk_paths,
            deleted_chunk_bytes=chunk_bytes_before_delete,
            validated_video_path=video_path,
            validated_frame_count=saved_frames,
        )
        updated = _annotate_recording_after_chunk_cleanup(recording, cleanup_info=cleanup_info, applied=applied)
        _annotate_manifest_after_chunk_cleanup(
            run_dir,
            recording.get(manifest_key),
            cleanup_info=cleanup_info,
            applied=applied,
        )
        if applied:
            logger.info(
                "Deleted %s raw chunk(s) for %s after parity validation (%s bytes)",
                cleanup_info["deleted_chunk_count"],
                system_name,
                cleanup_info["deleted_chunk_bytes"],
            )
        else:
            logger.info("Retained %s raw chunks: %s", system_name, reason)
        return updated

    return (
        _finalize_one(fictrac_recording, system_name="FicTrac", manifest_key="raw_stream_manifest"),
        _finalize_one(blackfly_recording, system_name="second camera", manifest_key="manifest_path"),
    )


def _run_fictrac(
    driver: FicTracDriver,
    callback: ExperimentCallback,
    state: dict[str, Exception | None],
    logger: logging.Logger,
) -> None:
    try:
        driver.run()
        if not callback._stop.is_set():
            state["error"] = RuntimeError(
                "FicTrac process exited unexpectedly (no exception)"
            )
    except Exception as exc:
        state["error"] = exc
        logger.error("FicTrac thread error: %s", exc)


def _check_fictrac_health(
    cfg: RunProtocolConfig,
    callback: ExperimentCallback | None,
    fictrac_thread: threading.Thread | None,
    fictrac_state: dict[str, Exception | None],
    other_camera_recorder: Any,
) -> None:
    if other_camera_recorder is not None:
        other_camera_recorder.raise_if_failed()

    if callback is None:
        return

    if fictrac_thread is not None and not fictrac_thread.is_alive():
        if fictrac_state.get("error") is not None:
            raise RuntimeError(
                f"FicTrac thread crashed: {fictrac_state['error']}"
            ) from fictrac_state["error"]
        raise RuntimeError("FicTrac thread stopped unexpectedly")

    latest = callback.latest
    if latest is None:
        return

    stale_for_s = time.perf_counter() - latest.wall_time
    if stale_for_s > cfg.fictrac_timeout_s:
        raise RuntimeError(
            f"FicTrac stopped producing frames for {stale_for_s:.1f} s "
            f"(timeout={cfg.fictrac_timeout_s:.1f} s)"
        )


def _wait_for_second_camera_first_frame(
    *,
    recorder: Any,
    startup_timeout_s: float,
    logger: logging.Logger,
    health_check: Callable[[], None],
) -> None:
    logger.info("Waiting up to %.1f s for the second camera first frame...", startup_timeout_s)
    deadline = time.monotonic() + startup_timeout_s
    while recorder.frame_count <= 0 and time.monotonic() < deadline:
        health_check()
        time.sleep(0.1)

    health_check()
    if recorder.frame_count <= 0:
        raise RuntimeError(
            f"Second camera did not capture any frames within {startup_timeout_s:.1f} s of protocol start"
        )

    logger.info("Second camera connected (%s frame(s) captured)", recorder.frame_count)


def _is_fictrac_terminal_exit_error(exc: RuntimeError) -> bool:
    message = str(exc)
    return (
        "FicTrac process exited unexpectedly (no exception)" in message
        or "FicTrac thread stopped unexpectedly" in message
    )


def _wait_for_fictrac_frame_drain(
    *,
    callback: ExperimentCallback | None,
    expected_frame_count: int | None,
    camera_fps: float | None,
    logger: logging.Logger,
    health_check: Callable[[], None],
) -> None:
    if callback is None or expected_frame_count is None or expected_frame_count <= 0:
        return

    observed_frame_count = callback.frame_count
    if observed_frame_count >= expected_frame_count:
        return

    timeout_s = _compute_fictrac_drain_timeout_s(
        expected_frame_count=expected_frame_count,
        observed_frame_count=observed_frame_count,
        camera_fps=camera_fps,
    )
    logger.info(
        "Waiting up to %.2f s for FicTrac to drain final %s frame(s)...",
        timeout_s,
        expected_frame_count - observed_frame_count,
    )
    deadline = time.monotonic() + timeout_s
    while callback.frame_count < expected_frame_count and time.monotonic() < deadline:
        try:
            health_check()
        except RuntimeError as exc:
            if not _is_fictrac_terminal_exit_error(exc):
                raise
            logger.info(
                "FicTrac ended before drain completed at %s/%s frame(s).",
                callback.frame_count,
                expected_frame_count,
            )
            return
        time.sleep(0.05)

    try:
        health_check()
    except RuntimeError as exc:
        if not _is_fictrac_terminal_exit_error(exc):
            raise
        logger.info(
            "FicTrac ended before drain completed at %s/%s frame(s).",
            callback.frame_count,
            expected_frame_count,
        )
        return
    if callback.frame_count >= expected_frame_count:
        logger.info("FicTrac drained to expected frame count (%s).", callback.frame_count)
        return

    logger.warning(
        "FicTrac drain wait ended at %s/%s frame(s); proceeding with shutdown.",
        callback.frame_count,
        expected_frame_count,
    )


def _preconfigure_fictrac_camera_external(
    *,
    camera_index: int,
    runtime_cfg: RunProtocolConfig,
    logger: logging.Logger,
) -> None:
    command = [
        sys.executable,
        "-m",
        "multibios.blackfly.preconfigure_camera",
        "--camera-index",
        str(camera_index),
        "--binning",
        str(runtime_cfg.blackfly_binning),
    ]
    if runtime_cfg.blackfly_exposure_us is not None:
        command.extend(["--exposure-us", str(runtime_cfg.blackfly_exposure_us)])
    if runtime_cfg.blackfly_roi_width is not None:
        command.extend(["--roi-width", str(runtime_cfg.blackfly_roi_width)])
    if runtime_cfg.blackfly_roi_height is not None:
        command.extend(["--roi-height", str(runtime_cfg.blackfly_roi_height)])
    if runtime_cfg.blackfly_gain_db is not None:
        command.extend(["--gain-db", str(runtime_cfg.blackfly_gain_db)])
    if runtime_cfg.blackfly_gamma is not None:
        command.extend(["--gamma", str(runtime_cfg.blackfly_gamma)])

    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=30.0,
        cwd=str(Path(__file__).resolve().parent.parent),
    )
    stdout = completed.stdout.strip()
    stderr = completed.stderr.strip()
    if completed.returncode != 0:
        detail = stderr or stdout or f"return code {completed.returncode}"
        raise RuntimeError(f"external preconfigure helper failed: {detail}")

    if stdout:
        last_line = stdout.splitlines()[-1]
        try:
            geometry = json.loads(last_line)
        except json.JSONDecodeError:
            logger.info("FicTrac camera helper output:\n%s", stdout)
        else:
            logger.info(
                "FicTrac camera %s preconfigured externally: %sx%s at offset (%s, %s).",
                camera_index,
                geometry.get("width"),
                geometry.get("height"),
                geometry.get("offset_x"),
                geometry.get("offset_y"),
            )


def _list_blackfly_cameras_external() -> list[dict[str, Any]]:
    command = [
        sys.executable,
        "-m",
        "multibios.blackfly.preconfigure_camera",
        "--list-cameras",
    ]
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=30.0,
        cwd=str(Path(__file__).resolve().parent.parent),
    )
    stdout = completed.stdout.strip()
    stderr = completed.stderr.strip()
    if completed.returncode != 0:
        detail = stderr or stdout or f"return code {completed.returncode}"
        raise RuntimeError(f"camera enumeration helper failed: {detail}")
    if not stdout:
        raise RuntimeError("camera enumeration helper produced no output")
    payload = json.loads(stdout.splitlines()[-1])
    cameras = payload.get("cameras")
    if not isinstance(cameras, list):
        raise RuntimeError("camera enumeration helper returned no camera list")
    return [camera for camera in cameras if isinstance(camera, dict)]


def _resolve_camera_roles(runtime_cfg: RunProtocolConfig, logger: logging.Logger) -> dict[str, dict[str, Any] | None]:
    cameras = _list_blackfly_cameras_external()
    cameras_by_serial = {
        str(camera.get("serial")): camera
        for camera in cameras
        if camera.get("serial")
    }

    def _resolve(label: str, *, serial: str, index: int | None) -> dict[str, Any] | None:
        if serial:
            camera = cameras_by_serial.get(serial)
            if camera is None:
                raise RuntimeError(f"Configured {label} serial {serial} was not found among connected cameras.")
            return camera
        if index is None:
            return None
        for camera in cameras:
            if int(camera.get("camera_index", -1)) == int(index):
                return camera
        raise RuntimeError(f"Configured {label} index {index} was not found among connected cameras.")

    fictrac_camera = _resolve("FicTrac camera", serial=runtime_cfg.fictrac_camera_serial, index=None)
    second_camera = _resolve(
        "second camera",
        serial=runtime_cfg.second_camera_serial,
        index=runtime_cfg.second_camera_index,
    )

    if fictrac_camera is not None:
        logger.info(
            "Resolved FicTrac camera: index=%s serial=%s model=%s",
            fictrac_camera.get("camera_index"),
            fictrac_camera.get("serial"),
            fictrac_camera.get("model"),
        )
    if second_camera is not None:
        logger.info(
            "Resolved second camera: index=%s serial=%s model=%s",
            second_camera.get("camera_index"),
            second_camera.get("serial"),
            second_camera.get("model"),
        )
    if fictrac_camera is not None and second_camera is not None and fictrac_camera.get("serial") == second_camera.get("serial"):
        raise RuntimeError("FicTrac camera and second camera resolve to the same serial; roles must point to different cameras.")

    return {"fictrac": fictrac_camera, "second": second_camera}


def _reset_fictrac_camera_external(*, camera_index: int, logger: logging.Logger) -> None:
    command = [
        sys.executable,
        "-m",
        "multibios.blackfly.preconfigure_camera",
        "--camera-index",
        str(camera_index),
        "--reset-editable",
    ]
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=30.0,
        cwd=str(Path(__file__).resolve().parent.parent),
    )
    stdout = completed.stdout.strip()
    stderr = completed.stderr.strip()
    if completed.returncode != 0:
        detail = stderr or stdout or f"return code {completed.returncode}"
        raise RuntimeError(f"external reset helper failed: {detail}")
    if stdout:
        last_line = stdout.splitlines()[-1]
        try:
            payload = json.loads(last_line)
        except json.JSONDecodeError:
            logger.info("FicTrac camera reset helper output:\n%s", stdout)
            return
        logger.info("FicTrac camera %s reset externally to editable mode.", payload.get("camera_index", camera_index))


def _safe_stop_task(task: Any, logger: logging.Logger, label: str) -> None:
    if task is None:
        return

    try:
        task.stop()
        logger.debug("  %s task stopped", label)
    except Exception as exc:
        logger.debug("  %s task stop skipped: %s", label, exc)


def _stop_protocol_tasks(
    *,
    do_task: Any,
    ao_task: Any,
    ai_task: Any,
    di_task: Any,
    logger: logging.Logger,
) -> None:
    _safe_stop_task(di_task, logger, "DI slave")
    _safe_stop_task(ai_task, logger, "AI slave")
    _safe_stop_task(ao_task, logger, "AO slave")
    _safe_stop_task(do_task, logger, "DO master")


def _stop_fictrac(
    *,
    fictrac_driver: FicTracDriver | None,
    fictrac_callback: ExperimentCallback | None,
    fictrac_thread: threading.Thread | None,
    fictrac_camera_index: int | None,
    logger: logging.Logger,
) -> None:
    thread_exited_cleanly = True
    if fictrac_thread is not None:
        if fictrac_driver is not None and hasattr(fictrac_driver, "expect_terminal_drain"):
            fictrac_driver.expect_terminal_drain()
        if fictrac_thread.ident is None:
            logger.info("FicTrac thread was never started; skipping join.")
        else:
            logger.info("Waiting briefly for FicTrac to exit naturally...")
            fictrac_thread.join(timeout=15.0)
        if fictrac_thread.is_alive() and fictrac_callback is not None:
            logger.info("Stopping FicTrac...")
            fictrac_callback.request_stop()
            fictrac_thread.join(timeout=10.0)
        if fictrac_thread.is_alive():
            if fictrac_driver is not None:
                logger.warning("FicTrac thread did not exit after cooperative stop; requesting driver stop.")
                fictrac_driver.request_stop()
                fictrac_thread.join(timeout=10.0)
        if fictrac_thread.is_alive():
            thread_exited_cleanly = False
            logger.warning("FicTrac thread did not exit cleanly; skipping camera reset")
    elif fictrac_driver is not None:
        if fictrac_callback is not None:
            logger.info("Stopping FicTrac...")
            fictrac_callback.request_stop()
        fictrac_driver.request_stop()
    if thread_exited_cleanly and fictrac_camera_index is not None:
        try:
            logger.info("Resetting FicTrac camera %s to editable mode...", fictrac_camera_index)
            _reset_fictrac_camera_external(camera_index=fictrac_camera_index, logger=logger)
        except Exception as exc:
            logger.warning(
                "Failed to reset FicTrac camera %s after shutdown: %s",
                fictrac_camera_index,
                exc,
            )


def _stop_other_camera_recorder(
    *,
    recorder: Any,
    logger: logging.Logger,
) -> dict[str, Any]:
    logger.info("Stopping second Blackfly recorder...")
    try:
        return recorder.stop()
    except Exception as exc:
        logger.warning(f"Second Blackfly recorder reported an error during stop: {exc}")
        manifest_path = recorder.manifest_path
        if manifest_path.exists():
            try:
                recovered = json.loads(manifest_path.read_text(encoding="utf-8"))
                recovered["stop_warning"] = str(exc)
                logger.warning(
                    "Recovered second-camera recording metadata from manifest; continuing postprocess."
                )
                return recovered
            except Exception:
                raise exc
        raise exc


# ----------------------------- progress monitor -----------------------------
class ProtocolProgressMonitor:
    """Monitor and display protocol execution progress in real-time."""
    
    def __init__(
        self, 
        do_data: np.ndarray, 
        do_names: List[str],
        ao_data: np.ndarray,
        ao_names: List[str],
        dt_ms: float,
        sample_rate: int,
        logger: logging.Logger,
        update_interval_ms: int = 100
    ):
        """
        Initialize the progress monitor.
        
        Args:
            do_data: Digital output array (n_lines x n_samples)
            do_names: Names of digital output lines
            ao_data: Analog output array (n_channels x n_samples)
            ao_names: Names of analog output channels
            dt_ms: Time step per sample in milliseconds
            sample_rate: DAQ sample rate in Hz
            logger: Logger instance for output
            update_interval_ms: How often to update the display (milliseconds)
        """
        self.do_data = do_data.astype(bool)
        self.do_names = do_names
        self.ao_data = ao_data
        self.ao_names = ao_names
        self.dt_ms = dt_ms
        self.sample_rate = sample_rate
        self.logger = logger
        self.update_interval_ms = update_interval_ms
        
        self.n_samples = do_data.shape[1]
        self.duration_s = self.n_samples / sample_rate
        
        self.start_time = None
        self.stop_flag = threading.Event()
        self.monitor_thread = None
        
    def _format_state(self, sample_idx: int) -> str:
        """Format the current expected state as a compact readable string."""
        # Get current sample (or last if we're past the end)
        idx = min(sample_idx, self.n_samples - 1)
        elapsed_ms = sample_idx * self.dt_ms
        
        # Build compact state string
        parts = [f"{elapsed_ms:7.1f}ms"]
        
        # Digital outputs - show as indexed pattern
        if self.do_names:
            do_pattern = "".join("█" if self.do_data[i, idx] else "░" 
                                for i in range(len(self.do_names)))
            parts.append(f"DO:{do_pattern}")
        
        # Analog outputs - show significant non-zero values with indices
        if self.ao_names:
            active_ao = []
            for i in range(len(self.ao_names)):
                v = self.ao_data[i, idx]
                if abs(v) > 0.01:  # Only show non-zero channels
                    active_ao.append(f"{i}:{v:.2f}")
            
            if active_ao:
                parts.append(f"AO:{','.join(active_ao)}")
            else:
                parts.append("AO:---")
        
        return " | ".join(parts)
    
    def _monitor_loop(self):
        """Background thread that periodically displays expected state."""
        last_update_time = time.time()
        
        while not self.stop_flag.is_set():
            current_time = time.time()
            
            # Calculate expected sample based on elapsed time
            elapsed_s = current_time - self.start_time
            expected_sample = int(elapsed_s * self.sample_rate)
            
            # Check if it's time to update
            if (current_time - last_update_time) >= (self.update_interval_ms / 1000.0):
                if expected_sample < self.n_samples:
                    # Protocol still running
                    progress_pct = (expected_sample / self.n_samples) * 100
                    state_str = self._format_state(expected_sample)
                    # Compact format: [  5%] instead of [  5.0%]
                    self.logger.info(f"[{progress_pct:3.0f}%] {state_str}")
                    last_update_time = current_time
                elif expected_sample >= self.n_samples and elapsed_s <= self.duration_s + 1.0:
                    # Just finished
                    self.logger.info(f"[100%] Protocol execution complete")
                    break
            
            # Sleep briefly to avoid busy-waiting
            time.sleep(0.01)
    
    def start(self):
        """Start monitoring in a background thread."""
        self.start_time = time.time()
        self.stop_flag.clear()
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        
        # Print compact startup info
        self.logger.info(f"Progress monitor: {self.n_samples} samples @ {self.sample_rate}Hz, updates every {self.update_interval_ms}ms")
        
        # Print legend for channel indices
        if self.do_names:
            do_legend = " ".join(f"[{i}:{name}]" for i, name in enumerate(self.do_names))
            self.logger.info(f"DO Legend: {do_legend}")
        
        if self.ao_names:
            ao_legend = " ".join(f"[{i}:{name}]" for i, name in enumerate(self.ao_names))
            self.logger.info(f"AO Legend: {ao_legend}")
        
    def stop(self):
        """Stop monitoring."""
        self.stop_flag.set()
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)


# ----------------------------- hardware adapter -----------------------------
@dataclass
class HardwareMap:
    device: str
    digital_outputs: Dict[str, str]
    analog_outputs: Dict[str, str]
    analog_inputs: Dict[str, str]
    digital_inputs: Dict[str, str]  # Synchronized digital inputs (READY rails, camera returns, etc.)

    # adapter fields the compiler expects
    @property
    def do_lines(self) -> Dict[str, str]:
        return self.digital_outputs

    @property
    def ao_channels(self) -> Dict[str, str]:
        return self.analog_outputs


def load_hardware(path: Path) -> HardwareMap:
    """Load hardware configuration from YAML file with detailed logging."""
    logger = logging.getLogger(__name__)
    logger.debug(f"Loading hardware YAML from: {path}")
    
    try:
        y = _read_yaml_text(path)
        logger.debug(f"YAML keys found: {list(y.keys()) if isinstance(y, dict) else 'Not a dict'}")
        
        # Validate required fields
        if "device" not in y:
            raise ValueError("Missing required 'device' field in hardware YAML")
        
        hw_map = HardwareMap(
            device=y["device"],
            digital_outputs=y.get("digital_outputs", {}),
            analog_outputs=y.get("analog_outputs", {}),
            analog_inputs=y.get("analog_inputs", {}),
            digital_inputs=y.get("digital_inputs", {}),
        )
        
        logger.debug(f"Hardware map created successfully:")
        logger.debug(f"  Device: {hw_map.device}")
        logger.debug(f"  DO channels: {len(hw_map.digital_outputs)}")
        logger.debug(f"  AO channels: {len(hw_map.analog_outputs)}")
        logger.debug(f"  AI channels: {len(hw_map.analog_inputs)}")
        logger.debug(f"  DI channels: {len(hw_map.digital_inputs)}")
        
        return hw_map
        
    except Exception as e:
        logger.error(f"Failed to load hardware configuration: {e}")
        raise


# ----------------------------- logging utils --------------------------------
def setup_logging(verbose: bool = False, debug: bool = False) -> logging.Logger:
    """Set up logging configuration with appropriate verbosity level."""
    logger = logging.getLogger(__name__)
    
    # Clear any existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Set logging level
    if debug:
        level = logging.DEBUG
    elif verbose:
        level = logging.INFO
    else:
        level = logging.WARNING
    
    logger.setLevel(level)
    
    # Create console handler with formatting
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    
    # Create formatter
    formatter = logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    handler.setFormatter(formatter)
    
    logger.addHandler(handler)
    return logger


# ----------------------------- logging utils --------------------------------
def ensure_run_dir(root: Path) -> Path:
    """Create timestamped run directory with logging."""
    logger = logging.getLogger(__name__)
    
    ts = time.strftime("%Y-%m-%d_%H-%M-%S")
    d = root / ts
    
    logger.debug(f"Creating run directory: {d}")
    logger.debug(f"  Root directory: {root}")
    logger.debug(f"  Timestamp: {ts}")
    
    try:
        d.mkdir(parents=True, exist_ok=False)
        logger.debug(f"✓ Run directory created successfully")
    except FileExistsError:
        logger.warning(f"Run directory already exists (should be rare): {d}")
    except Exception as e:
        logger.error(f"Failed to create run directory: {e}")
        raise
        
    return d


# ----------------------------- main -----------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Run NI 6353 hardware-clocked protocol with AI/DI logging."
    )
    ap.add_argument(
        "--yaml", default="protocols/example_protocol.yaml", help="Protocol YAML"
    )
    ap.add_argument(
        "--hardware", default="config/hardware.yaml", help="Hardware map YAML"
    )
    ap.add_argument(
        "--experiment",
        help="Optional runtime config YAML for FicTrac/camera settings (hardware-owned values preferred in hardware.yaml)",
    )
    ap.add_argument("--device", help="Override device name (else from hardware.yaml)")
    ap.add_argument("--dry-run", action="store_true", help="Compile only; no hardware")
    ap.add_argument(
        "--interactive",
        action="store_true",
        help="Always save interactive HTML preview",
    )
    ap.add_argument("--out-root", help="Run folder root (defaults to hardware data_output.data_dir or data/runs)")
    ap.add_argument(
        "--verbose", "-v", 
        action="store_true", 
        help="Enable verbose logging and detailed progress output"
    )
    ap.add_argument(
        "--debug", 
        action="store_true", 
        help="Enable debug logging (even more verbose than --verbose)"
    )
    ap.add_argument(
        "--progress",
        action="store_true",
        help="Enable real-time progress monitor during protocol execution"
    )
    ap.add_argument(
        "--progress-interval",
        type=int,
        default=100,
        help="Progress update interval in milliseconds (default: 100)"
    )
    # Optional pulse tuning overrides (otherwise read from YAML)
    ap.add_argument("--preload-lead-ms", type=int)
    ap.add_argument("--load-req-ms", type=int)
    ap.add_argument("--rck-ms", type=int)
    ap.add_argument("--trig-ms", type=int)
    ap.add_argument(
        "--seed",
        type=int,
        help="Override protocol.timing.seed for reproducible randomization",
    )
    args = ap.parse_args()

    # Set up logging based on verbosity level
    logger = setup_logging(verbose=args.verbose, debug=args.debug)
    
    logger.info("=== MultiBiOS Protocol Runner Starting ===")
    logger.info(f"Command line arguments: {vars(args)}")
    logger.info(f"Python version: {sys.version}")
    logger.info(f"Working directory: {Path.cwd()}")
    
    # Validate input files
    proto_path = Path(args.yaml)
    hw_path = Path(args.hardware)
    
    logger.info(f"Protocol file path: {proto_path.absolute()}")
    logger.info(f"Hardware file path: {hw_path.absolute()}")
    
    if not proto_path.exists():
        logger.error(f"Protocol file not found: {proto_path}")
        raise SystemExit(f"Protocol file not found: {proto_path}")
    if not hw_path.exists():
        logger.error(f"Hardware file not found: {hw_path}")
        raise SystemExit(f"Hardware file not found: {hw_path}")
        
    logger.info("✓ Input files validated successfully")

    # Load hardware configuration with detailed logging
    logger.info("Loading hardware configuration...")
    logger.debug(f"Reading hardware YAML from: {hw_path}")
    
    hw = load_hardware(hw_path)
    logger.info(f"✓ Hardware configuration loaded successfully")
    logger.info(f"  Device: {hw.device}")
    logger.info(f"  Digital outputs: {len(hw.digital_outputs)} channels")
    for name, channel in hw.digital_outputs.items():
        logger.debug(f"    {name} -> {channel}")
    logger.info(f"  Analog outputs: {len(hw.analog_outputs)} channels")  
    for name, channel in hw.analog_outputs.items():
        logger.debug(f"    {name} -> {channel}")
    logger.info(f"  Analog inputs: {len(hw.analog_inputs)} channels")
    for name, channel in hw.analog_inputs.items():
        logger.debug(f"    {name} -> {channel}")
    logger.info(f"  Digital inputs: {len(hw.digital_inputs)} channels")
    for name, channel in hw.digital_inputs.items():
        logger.debug(f"    {name} -> {channel}")
    
    if args.device:
        logger.info(f"Overriding device name: {hw.device} -> {args.device}")
        hw.device = args.device

    runtime_cfg = load_run_protocol_config(args.experiment, hardware_path=hw_path)
    run_root = Path(args.out_root) if args.out_root else Path(runtime_cfg.data_dir)
    logger.info("Runtime capture settings:")
    logger.info(f"  Run output root: {run_root}")
    logger.info(f"  FicTrac enabled: {bool(runtime_cfg.fictrac_config)}")
    logger.info(f"  Second camera recording: {runtime_cfg.save_second_camera_video}")
    logger.info(f"  Raw chunk retention: {runtime_cfg.raw_chunk_retention_policy}")

    # Load and process protocol YAML
    logger.info("Loading protocol configuration...")
    logger.debug(f"Reading protocol YAML from: {proto_path}")
    
    y = _read_yaml_text(proto_path)
    logger.info("✓ Protocol YAML loaded successfully")
    logger.debug(f"Protocol keys: {list(y.keys()) if isinstance(y, dict) else 'Not a dict'}")
    
    if args.seed is not None:
        logger.info(f"Overriding protocol seed: {args.seed}")
        y.setdefault("protocol", {}).setdefault("timing", {})["seed"] = int(args.seed)
        
    # Process timing configuration with detailed logging
    logger.info("Processing timing configuration...")
    protocol_block = y.setdefault("protocol", {})
    if not isinstance(protocol_block, dict):
        raise ValueError("protocol YAML must contain a mapping under 'protocol'")
    timing_block = protocol_block.setdefault("timing", {})
    if not isinstance(timing_block, dict):
        raise ValueError("protocol YAML must contain a mapping under 'protocol.timing'")

    _apply_hardware_owned_camera_timing(timing_block, runtime_cfg)

    t = timing_block
    logger.debug(f"Raw timing config: {t}")
    
    tcfg = TimingConfig(
        base_unit=t.get("base_unit", "ms"),
        sample_rate=int(t.get("sample_rate", 1000)),
        camera_interval_ms=float(t.get("camera_interval", 0.0)),
        camera_pulse_ms=float(t.get("camera_pulse_duration", 5.0)),
        preload_lead_ms=int(
            args.preload_lead_ms
            if args.preload_lead_ms is not None
            else t.get("preload_lead_ms", 2)
        ),
        load_req_ms=int(
            args.load_req_ms
            if args.load_req_ms is not None
            else t.get("load_req_ms", 1)
        ),
        rck_pulse_ms=int(
            args.rck_ms if args.rck_ms is not None else t.get("rck_pulse_ms", 1)
        ),
        trig_pulse_ms=int(
            args.trig_ms if args.trig_ms is not None else t.get("trig_pulse_ms", 5)
        ),
        setup_hold_samples=int(t.get("setup_hold_samples", 5)),
    )
    
    logger.info("✓ Timing configuration processed")
    logger.info(f"  Sample rate: {tcfg.sample_rate} Hz")
    logger.info(f"  Base unit: {tcfg.base_unit}")
    logger.info(f"  Camera interval: {tcfg.camera_interval_ms} ms")
    if runtime_cfg.camera_trigger_fps_hz is not None:
        logger.info(f"  Camera trigger FPS (hardware-owned): {runtime_cfg.camera_trigger_fps_hz:.3f} Hz")
    if runtime_cfg.camera_trigger_pulse_ms is not None:
        logger.info(f"  Camera trigger pulse (hardware-owned): {runtime_cfg.camera_trigger_pulse_ms} ms")
    logger.info(f"  Camera pulse: {tcfg.camera_pulse_ms} ms")
    logger.info(f"  Preload lead: {tcfg.preload_lead_ms} ms")
    logger.info(f"  Load request: {tcfg.load_req_ms} ms")
    logger.info(f"  RCK pulse: {tcfg.rck_pulse_ms} ms")
    logger.info(f"  Trigger pulse: {tcfg.trig_pulse_ms} ms")
    logger.info(f"  Setup/hold samples: {tcfg.setup_hold_samples}")

    # Compile protocol with detailed progress logging
    logger.info("=== Starting Protocol Compilation ===")
    comp = ProtocolCompiler(hw, tcfg)
    logger.info("✓ Protocol compiler initialized")
    
    try:
        logger.info("Compiling protocol from YAML...")
        start_time = time.time()
        comp.compile_from_yaml(y)
        compile_time = time.time() - start_time
        
        logger.info(f"✓ Protocol compilation completed in {compile_time:.2f} seconds")
        logger.info(f"  Total samples: {comp.N}")
        logger.info(f"  Duration: {comp.N * comp.dt_ms:.1f} ms ({comp.N * comp.dt_ms / 1000:.2f} seconds)")
        logger.info(f"  Sample time step: {comp.dt_ms:.3f} ms")
        logger.info(f"  Digital output lines: {len(comp.line_order)}")
        logger.info(f"  Analog output channels: {len(comp.ao_order)}")
        logger.info(f"  RNG seed used: {getattr(comp, 'rng_seed', 'N/A')}")
        
        if hasattr(comp, 'rck_log') and comp.rck_log:
            logger.info(f"  RCK commit events: {len(comp.rck_log)}")
            logger.debug("  RCK event details:")
            for i, (sig, si, tms) in enumerate(comp.rck_log[:5]):  # Show first 5
                logger.debug(f"    {i+1}: {sig} at sample {si} ({tms:.3f} ms)")
            if len(comp.rck_log) > 5:
                logger.debug(f"    ... and {len(comp.rck_log) - 5} more")
                
    except CompileError as e:
        logger.error(f"Protocol compilation failed: {e}")
        raise SystemExit(f"[compile error] {e}")

    control_plan = compile_control_plan(y, seed=getattr(comp, "rng_seed", None))

    # Create run directory and save artifacts with detailed logging
    logger.info("=== Creating Run Directory and Artifacts ===")
    run_dir = ensure_run_dir(run_root)
    logger.info(f"✓ Run directory created: {run_dir}")
    logger.info(f"  Run timestamp: {run_dir.name}")
    
    # Save compilation report
    logger.info("Saving compilation report...")
    report_file = run_dir / "compile_report.json"
    report_file.write_text(json.dumps(comp.report, indent=2))
    logger.info(f"  ✓ Compilation report: {report_file}")
    logger.debug(f"    Report keys: {list(comp.report.keys()) if hasattr(comp, 'report') and comp.report else 'No report'}")
    
    # Save input files for reproducibility
    logger.info("Saving input files for reproducibility...")
    proto_copy = run_dir / "protocol.yaml"
    hw_copy = run_dir / "hardware.yaml"
    proto_copy.write_text(proto_path.read_text(encoding="utf-8"), encoding="utf-8")
    hw_copy.write_text(hw_path.read_text(encoding="utf-8"), encoding="utf-8")
    logger.info(f"  ✓ Protocol YAML copy: {proto_copy}")
    logger.info(f"  ✓ Hardware YAML copy: {hw_copy}")
    
    # Save metadata  (t0 anchors start as None; re-written after do_task.start)
    logger.info("Saving run metadata...")
    t0_utc: float | None = None
    t0_perf: float | None = None
    meta_data = {
        "device": hw.device,
        "sample_rate": comp.tcfg.sample_rate,
        "duration_ms": comp.N * comp.dt_ms,
        "rng_seed": getattr(comp, 'rng_seed', None),
        "args": vars(args),
        # Hardware clock anchors —————————————————————————————————————————
        # t0_utc: Unix timestamp (time.time()) recorded immediately before
        #   do_task.start().  Convert sample index to wall time:
        #   wall_time = t0_utc + sample_idx / sample_rate
        # t0_perf_counter: time.perf_counter() at the same instant, for
        #   aligning software events (serial, FicTrac) that were also
        #   stamped with perf_counter in the same process.
        "t0_utc": t0_utc,
        "t0_perf_counter": t0_perf,
    }
    meta_file = run_dir / "meta.json"
    meta_file.write_text(json.dumps(meta_data, indent=2))
    logger.info(f"  ✓ Metadata: {meta_file}")
    logger.debug(f"    Metadata: {meta_data}")
    
    # Save RCK edges log
    logger.info("Saving RCK edges log...")
    rck_file = run_dir / "rck_edges.csv"
    with rck_file.open("w") as f:
        f.write("signal,sample_idx,time_ms\n")
        for sig, si, tms in comp.rck_log:
            f.write(f"{sig},{si},{tms:.3f}\n")
    logger.info(f"  ✓ RCK edges: {rck_file} ({len(comp.rck_log)} events)")

    # Save channel mapping files
    logger.info("Saving channel mapping files...")
    do_names = comp.line_order
    ao_names = comp.ao_order
    
    do_map = {"names": do_names, "phys": [hw.digital_outputs[n] for n in do_names]}
    do_map_file = run_dir / "do_map.json"
    do_map_file.write_text(json.dumps(do_map, indent=2))
    logger.info(f"  ✓ DO mapping: {do_map_file} ({len(do_names)} lines)")
    
    ao_map = {"names": ao_names, "phys": [hw.analog_outputs[n] for n in ao_names]}
    ao_map_file = run_dir / "ao_map.json"
    ao_map_file.write_text(json.dumps(ao_map, indent=2))
    logger.info(f"  ✓ AO mapping: {ao_map_file} ({len(ao_names)} channels)")
    
    # DI map (synchronized digital inputs) — write even if empty for consistency
    di_names_cfg = list(hw.digital_inputs.keys())
    di_map = {"names": di_names_cfg, "phys": [hw.digital_inputs[n] for n in di_names_cfg]}
    di_map_file = run_dir / "di_map.json"
    di_map_file.write_text(json.dumps(di_map, indent=2))
    logger.info(f"  ✓ DI mapping: {di_map_file} ({len(di_names_cfg)} lines)")

    # Save compiled arrays
    logger.info("Saving compiled waveform arrays...")
    do_file = run_dir / "compiled_do.npz"
    ao_file = run_dir / "compiled_ao.npz"
    
    logger.debug(f"  DO array shape: {comp.do.shape}, dtype: {comp.do.dtype}")
    np.savez_compressed(do_file, data=comp.do.astype(np.bool_))
    logger.info(f"  ✓ Digital outputs: {do_file}")
    
    logger.debug(f"  AO array shape: {comp.ao.shape}, dtype: {comp.ao.dtype}")
    np.savez_compressed(ao_file, data=comp.ao.astype(np.float32))
    logger.info(f"  ✓ Analog outputs: {ao_file}")

    control_plan_file = run_dir / "control_plan.csv"
    write_control_plan_csv(control_plan_file, control_plan.timeline)
    logger.info(f"  ✓ Shared control plan: {control_plan_file} ({len(control_plan.timeline)} events)")

    # Digital edge log (super helpful to diff runs)
    logger.info("Computing and saving digital edge transitions...")
    edge_file = run_dir / "digital_edges.csv"
    write_edge_csv(edge_file, do_names, comp.do.astype(bool), comp.dt_ms)
    logger.info(f"  ✓ Digital edges: {edge_file}")
    
    # Count edges for summary
    do_bool = comp.do.astype(bool)
    total_edges = 0
    for i in range(len(do_names)):
        edges = np.sum(np.diff(do_bool[i, :].astype(int)) != 0)
        total_edges += edges
        logger.debug(f"    {do_names[i]}: {edges} transitions")
    logger.info(f"  Total edge transitions: {total_edges}")

    preview_file = run_dir / "preview.html"

    # Generate preview visualization
    if args.interactive:
        logger.info("Generating preview visualization...")
        t_ms = np.arange(comp.N) * comp.dt_ms
        fig = make_protocol_figure(
            t_ms,
            comp.do.astype(bool),
            do_names,
            comp.ao,
            ao_names,
            title="Preview (no DAQ)",
            rck_log=comp.rck_log,
        )
        fig.write_html(preview_file, include_plotlyjs="cdn")
        logger.info(f"  ✓ Preview visualization: {preview_file}")

    if args.dry_run:
        logger.info("=== DRY RUN COMPLETE ===")
        logger.info(f"All artifacts saved to: {run_dir}")
        if args.interactive:
            logger.info(f"Preview available at: {preview_file}")
            print(f"Dry-run complete. Preview: {preview_file}")
        else:
            logger.info("Preview not generated because --interactive was not requested")
            print(f"Dry-run complete. Artifacts: {run_dir}")
        return

    # --- DAQ execution: DO master, AO slave, AI slave (MFC feedback), DI slave (synchronized DI)
    logger.info("=== Starting DAQ Hardware Execution ===")
    N = comp.N
    rate = comp.tcfg.sample_rate
    
    logger.info(f"DAQ Configuration:")
    logger.info(f"  Device: {hw.device}")
    logger.info(f"  Sample rate: {rate} Hz")
    logger.info(f"  Total samples: {N}")
    logger.info(f"  Estimated duration: {N/rate:.2f} seconds")
    if args.progress:
        logger.info(f"  Real-time progress monitoring: ENABLED (interval: {args.progress_interval}ms)")
    else:
        logger.info(f"  Real-time progress monitoring: DISABLED (use --progress to enable)")

    # Prepare channel lists
    ai_names = list(hw.analog_inputs.keys())
    ai_phys = [hw.analog_inputs[n] for n in ai_names]
    di_names = list(hw.digital_inputs.keys())
    di_phys = [hw.digital_inputs[n] for n in di_names]
    
    logger.info(f"Channel Summary:")
    logger.info(f"  Digital outputs (DO): {len(do_names)} channels")
    for i, (name, phys) in enumerate(zip(do_names, [hw.digital_outputs[n] for n in do_names])):
        logger.debug(f"    DO[{i}]: {name} -> {phys}")
        
    logger.info(f"  Analog outputs (AO): {len(ao_names)} channels")
    for i, (name, phys) in enumerate(zip(ao_names, [hw.analog_outputs[n] for n in ao_names])):
        logger.debug(f"    AO[{i}]: {name} -> {phys}")
        
    logger.info(f"  Analog inputs (AI): {len(ai_names)} channels")
    for i, (name, phys) in enumerate(zip(ai_names, ai_phys)):
        logger.debug(f"    AI[{i}]: {name} -> {phys}")
        
    logger.info(f"  Digital inputs (DI): {len(di_names)} channels")
    for i, (name, phys) in enumerate(zip(di_names, di_phys)):
        logger.debug(f"    DI[{i}]: {name} -> {phys}")

    logger.info("Creating DAQ tasks...")
    fictrac_callback: ExperimentCallback | None = None
    fictrac_driver: FicTracDriver | None = None
    fictrac_thread: threading.Thread | None = None
    fictrac_state: dict[str, Exception | None] = {"error": None}
    fictrac_runtime_info: dict[str, Any] = {}
    fictrac_camera_index: int | None = None
    fictrac_camera_serial: str | None = None
    other_camera_recorder: Any = None
    other_camera_recording: dict[str, Any] = {}
    teensy_serial_monitor: SerialLineMonitor | None = None
    teensy_serial_transcript: list[dict[str, Any]] = []
    camera_trigger_trace = comp.do[do_names.index("TRIG_CAMERA")] if "TRIG_CAMERA" in do_names else None
    expected_camera_frames = _count_rising_edges(camera_trigger_trace)
    first_camera_trigger_sample = _first_rising_edge_sample(camera_trigger_trace)
    nominal_camera_fps = (1000.0 / comp.tcfg.camera_interval_ms) if comp.tcfg.camera_interval_ms > 0 else None
    long_high_rate_fictrac_run = bool(
        runtime_cfg.camera_trigger_fps_hz is not None
        and runtime_cfg.camera_trigger_fps_hz >= 120.0
        and expected_camera_frames is not None
        and expected_camera_frames >= 10_000
    )
    run_interrupted = False
    resolved_camera_roles: dict[str, dict[str, Any] | None] = {"fictrac": None, "second": None}

    if runtime_cfg.fictrac_camera_serial or runtime_cfg.second_camera_serial:
        resolved_camera_roles = _resolve_camera_roles(runtime_cfg, logger)

    if runtime_cfg.capture_teensy_serial:
        if not runtime_cfg.teensy_port:
            raise RuntimeError("capture_teensy_serial is enabled but no teensy.port is configured in hardware.yaml")
        logger.info("Starting Teensy serial monitor on %s...", runtime_cfg.teensy_port)
        teensy_serial_monitor = SerialLineMonitor(
            port=runtime_cfg.teensy_port,
            baudrate=runtime_cfg.teensy_baud,
            timeout=1.0,
            boot_delay_s=0.5,
            reset_input_buffer_on_open=True,
        )
        teensy_serial_monitor.open()
        logger.info("✓ Teensy serial monitor active")

    try:
        with (
            nidaqmx.Task("DO_MASTER") as do_task,
            nidaqmx.Task("AO_SLAVE") as ao_task,
            nidaqmx.Task("AI_SLAVE") as ai_task,
            nidaqmx.Task("DI_READY") as di_task,
        ):
            logger.info("✓ DAQ tasks created successfully")

            # DO master lines
            logger.info("Configuring DO master task...")
            for i, ch in enumerate([hw.digital_outputs[n] for n in do_names]):
                logger.debug(f"  Adding DO channel {i}: {ch}")
                do_task.do_channels.add_do_chan(
                    ch, line_grouping=LineGrouping.CHAN_PER_LINE
                )
            do_task.timing.cfg_samp_clk_timing(
                rate=rate,
                active_edge=Edge.RISING,
                sample_mode=AcquisitionType.FINITE,
                samps_per_chan=N,
            )
            logger.debug(f"  Writing DO data array shape: {comp.do.shape}")
            do_task.write(comp.do.astype(np.bool_))
            logger.info("✓ DO master task configured and data loaded")

            # AO slave
            if ao_names:
                logger.info("Configuring AO slave task...")
                ao_channel_str = ",".join([hw.analog_outputs[n] for n in ao_names])
                logger.debug(f"  AO channels: {ao_channel_str}")
                ao_task.ao_channels.add_ao_voltage_chan(
                    ao_channel_str,
                    min_val=0.0,
                    max_val=5.0,
                )
                ao_task.timing.cfg_samp_clk_timing(
                    rate=rate,
                    source=f"/{hw.device}/do/SampleClock",
                    active_edge=Edge.RISING,
                    sample_mode=AcquisitionType.FINITE,
                    samps_per_chan=N,
                )
                ao_task.triggers.start_trigger.cfg_dig_edge_start_trig(
                    f"/{hw.device}/do/StartTrigger"
                )
                logger.debug(f"  Writing AO data array shape: {comp.ao.shape}")
                AnalogMultiChannelWriter(ao_task.out_stream).write_many_sample(
                    comp.ao.astype(np.float64)
                )
                logger.info("✓ AO slave task configured and data loaded")
            else:
                logger.info("No AO channels configured, skipping AO task")

            # AI slave (MFC feedback)
            ai_buf = None
            if ai_phys:
                logger.info("Configuring AI slave task...")
                ai_channel_str = ",".join(ai_phys)
                logger.debug(f"  AI channels: {ai_channel_str}")
                ai_task.ai_channels.add_ai_voltage_chan(
                    ai_channel_str, min_val=0.0, max_val=10.0
                )
                ai_task.timing.cfg_samp_clk_timing(
                    rate=rate,
                    source=f"/{hw.device}/do/SampleClock",
                    active_edge=Edge.RISING,
                    sample_mode=AcquisitionType.FINITE,
                    samps_per_chan=N,
                )
                ai_task.triggers.start_trigger.cfg_dig_edge_start_trig(
                    f"/{hw.device}/do/StartTrigger"
                )
                ai_reader = AnalogMultiChannelReader(ai_task.in_stream)
                ai_buf = np.zeros((len(ai_phys), N), dtype=np.float64)
                logger.debug(f"  AI buffer shape: {ai_buf.shape}")
                logger.info("✓ AI slave task configured and buffer allocated")
            else:
                logger.info("No AI channels configured, skipping AI task")

            # DI slave (synchronized digital inputs such as READY rails and camera returns)
            if di_phys:
                logger.info("Configuring DI slave task...")
                for i, ch in enumerate(di_phys):
                    logger.debug(f"  Adding DI channel {i}: {ch}")
                    di_task.di_channels.add_di_chan(
                        ch, line_grouping=LineGrouping.CHAN_PER_LINE
                    )
                di_task.timing.cfg_samp_clk_timing(
                    rate=rate,
                    source=f"/{hw.device}/do/SampleClock",
                    active_edge=Edge.RISING,
                    sample_mode=AcquisitionType.FINITE,
                    samps_per_chan=N,
                )
                di_task.triggers.start_trigger.cfg_dig_edge_start_trig(
                    f"/{hw.device}/do/StartTrigger"
                )
                logger.info("✓ DI slave task configured successfully")
            else:
                logger.info("No DI channels configured, skipping DI task")

            if runtime_cfg.fictrac_config:
                logger.info("Preparing FicTrac runtime configuration...")
                if long_high_rate_fictrac_run:
                    logger.info(
                        "Long high-rate FicTrac run detected (%s expected camera frames at %.3f Hz); forcing headless runtime config.",
                        expected_camera_frames,
                        runtime_cfg.camera_trigger_fps_hz,
                    )
                fictrac_config_path, fictrac_camera_index, fictrac_runtime_info = _prepare_fictrac_runtime_config(
                    runtime_cfg.fictrac_config,
                    run_dir,
                    enable_raw_video=runtime_cfg.save_fictrac_camera_video,
                    camera_fps=runtime_cfg.camera_trigger_fps_hz,
                    video_codec=runtime_cfg.fictrac_raw_video_codec,
                    first_frame_timeout_ms=runtime_cfg.fictrac_first_frame_timeout_ms,
                    force_headless=long_high_rate_fictrac_run,
                    camera_index_override=(
                        int(resolved_camera_roles["fictrac"]["camera_index"])
                        if resolved_camera_roles["fictrac"] is not None
                        else None
                    ),
                )
                fictrac_camera_serial = (
                    str(resolved_camera_roles["fictrac"].get("serial"))
                    if resolved_camera_roles["fictrac"] is not None
                    else None
                )
                if fictrac_camera_serial:
                    fictrac_runtime_info["fictrac_camera_serial"] = fictrac_camera_serial
                if fictrac_camera_index is not None:
                    logger.info(
                        "FicTrac camera %s should use hardware.yaml image settings (ROI=%sx%s, exposure=%s us, binning=%s, gain=%s dB, gamma=%s).",
                        fictrac_camera_index,
                        runtime_cfg.blackfly_roi_width,
                        runtime_cfg.blackfly_roi_height,
                        runtime_cfg.blackfly_exposure_us,
                        runtime_cfg.blackfly_binning,
                        runtime_cfg.blackfly_gain_db,
                        runtime_cfg.blackfly_gamma,
                    )
                    try:
                        _preconfigure_fictrac_camera_external(
                            camera_index=fictrac_camera_index,
                            runtime_cfg=runtime_cfg,
                            logger=logger,
                        )
                    except Exception as exc:
                        logger.warning(
                            "Failed to preconfigure FicTrac camera %s externally; FicTrac may fall back to the camera's current ROI state: %s",
                            fictrac_camera_index,
                            exc,
                        )
                runtime_dirs = prepare_fictrac_runtime()
                if runtime_dirs:
                    logger.info("FicTrac runtime PATH prepared:")
                    for runtime_dir in runtime_dirs:
                        logger.info("  %s", runtime_dir)
                fictrac_callback = ExperimentCallback()
                fictrac_driver = FicTracDriver(
                    config_file=str(fictrac_config_path),
                    console_ouput_file=runtime_cfg.fictrac_console_out,
                    track_change_callback=fictrac_callback,
                    plot_on=False,
                    fic_trac_bin_path=runtime_cfg.fictrac_bin or None,
                )
                fictrac_thread = threading.Thread(
                    target=_run_fictrac,
                    args=(fictrac_driver, fictrac_callback, fictrac_state, logger),
                    name="FicTrac",
                    daemon=True,
                )
            else:
                fictrac_camera_index = None

            if runtime_cfg.save_second_camera_video:
                second_camera_startup_timeout_s = 0.0
                second_camera_index = (
                    int(resolved_camera_roles["second"]["camera_index"])
                    if resolved_camera_roles["second"] is not None
                    else runtime_cfg.second_camera_index
                )
                if second_camera_index is None and fictrac_camera_index in (0, 1):
                    second_camera_index = 1 - fictrac_camera_index
                if second_camera_index is None:
                    raise RuntimeError(
                        "save_second_camera_video requires camera_recording.second_camera_serial, camera_recording.second_camera_index, or a numeric FicTrac src_fn camera index of 0 or 1."
                    )
                if fictrac_camera_index is not None and second_camera_index == fictrac_camera_index:
                    raise RuntimeError(
                        "second_camera_index cannot match FicTrac's live camera index because the same Blackfly cannot be opened twice."
                    )

                from multibios.blackfly.triggered_camera_record import TriggeredCameraRecorder

                second_camera_timeout_ms = runtime_cfg.second_camera_timeout_ms
                if fictrac_thread is not None and runtime_cfg.fictrac_arm_delay_s > 0:
                    second_camera_timeout_ms += int(runtime_cfg.fictrac_arm_delay_s * 1000) + 100
                second_camera_startup_timeout_s = _compute_second_camera_startup_timeout_s(
                    first_trigger_sample=first_camera_trigger_sample,
                    sample_rate=rate,
                    arm_delay_s=runtime_cfg.fictrac_arm_delay_s if fictrac_thread is not None else 0.0,
                    recorder_timeout_ms=second_camera_timeout_ms,
                )
                second_camera_timeout_ms = max(
                    second_camera_timeout_ms,
                    int(np.ceil(second_camera_startup_timeout_s * 1000.0)),
                )

                logger.info("Recording Blackfly camera %s into the run directory...", second_camera_index)
                other_camera_recorder = TriggeredCameraRecorder(
                    camera_index=second_camera_index,
                    run_dir=run_dir,
                    timeout_ms=second_camera_timeout_ms,
                    queue_size=runtime_cfg.second_camera_queue_size,
                    stream_buffer_count=runtime_cfg.second_camera_stream_buffer_count,
                    exposure_us=runtime_cfg.second_camera_exposure_us,
                    roi_width=runtime_cfg.second_camera_roi_width,
                    roi_height=runtime_cfg.second_camera_roi_height,
                    binning=runtime_cfg.second_camera_binning,
                    gain_db=runtime_cfg.second_camera_gain_db,
                    gamma=runtime_cfg.second_camera_gamma,
                )
                other_camera_recording = other_camera_recorder.start()

            progress_monitor = None
            try:
                # Start tasks in proper sequence
                logger.info("Starting DAQ tasks...")
                if ao_names:
                    logger.debug("  Starting AO slave task...")
                    ao_task.start()
                if ai_phys:
                    logger.debug("  Starting AI slave task...")
                    ai_task.start()
                if di_phys:
                    logger.debug("  Starting DI slave task...")
                    di_task.start()

                logger.debug("  Starting DO master task...")
                start_time = time.time()

                if args.progress:
                    progress_monitor = ProtocolProgressMonitor(
                        do_data=comp.do,
                        do_names=do_names,
                        ao_data=comp.ao,
                        ao_names=ao_names,
                        dt_ms=comp.dt_ms,
                        sample_rate=comp.tcfg.sample_rate,
                        logger=logger,
                        update_interval_ms=args.progress_interval
                    )
                    progress_monitor.start()

                if fictrac_thread is not None:
                    fictrac_thread.start()
                    if runtime_cfg.fictrac_arm_delay_s > 0:
                        logger.info(
                            "Waiting %.3f s for FicTrac camera arm before starting DO...",
                            runtime_cfg.fictrac_arm_delay_s,
                        )
                        time.sleep(runtime_cfg.fictrac_arm_delay_s)
                        _check_fictrac_health(
                            runtime_cfg,
                            fictrac_callback,
                            fictrac_thread,
                            fictrac_state,
                            other_camera_recorder,
                        )

                # Capture hardware clock anchor: perf_counter and wall time as close as
                # possible to the DO start trigger so every sample_idx can be converted to
                # absolute time via:  t_utc = t0_utc + sample_idx / sample_rate
                t0_perf = time.perf_counter()
                t0_utc = time.time()
                do_task.start()

                meta_data["t0_utc"] = t0_utc
                meta_data["t0_perf_counter"] = t0_perf
                meta_file.write_text(json.dumps(meta_data, indent=2))
                logger.info("✓ All DAQ tasks started, protocol execution in progress...")

                if other_camera_recorder is not None:
                    _wait_for_second_camera_first_frame(
                        recorder=other_camera_recorder,
                        startup_timeout_s=second_camera_startup_timeout_s,
                        logger=logger,
                        health_check=lambda: _check_fictrac_health(
                            runtime_cfg,
                            fictrac_callback,
                            fictrac_thread,
                            fictrac_state,
                            other_camera_recorder,
                        ),
                    )

                if fictrac_callback is not None:
                    startup_timeout_s = runtime_cfg.fictrac_startup_timeout_s
                    if startup_timeout_s <= 0:
                        logger.info("Waiting indefinitely for FicTrac first frame...")
                        while fictrac_callback.latest is None:
                            time.sleep(0.5)
                            _check_fictrac_health(
                                runtime_cfg,
                                fictrac_callback,
                                fictrac_thread,
                                fictrac_state,
                                other_camera_recorder,
                            )
                    else:
                        logger.info("Waiting up to %.1f s for FicTrac first frame...", startup_timeout_s)
                        deadline = time.monotonic() + startup_timeout_s
                        while fictrac_callback.latest is None and time.monotonic() < deadline:
                            time.sleep(0.5)
                            _check_fictrac_health(
                                runtime_cfg,
                                fictrac_callback,
                                fictrac_thread,
                                fictrac_state,
                                other_camera_recorder,
                            )
                    if fictrac_callback.latest is None:
                        raise RuntimeError(
                            f"FicTrac did not produce any frames within {startup_timeout_s:.1f} s"
                        )
                    logger.info("FicTrac connected (frame %s)", fictrac_callback.latest.frame_cnt)

                timeout = max(10.0, N / rate + 5.0)
                logger.info(f"Waiting for protocol completion (timeout: {timeout:.1f}s)...")
                deadline = time.monotonic() + timeout
                while not do_task.is_task_done():
                    _check_fictrac_health(
                        runtime_cfg,
                        fictrac_callback,
                        fictrac_thread,
                        fictrac_state,
                        other_camera_recorder,
                    )
                    if time.monotonic() >= deadline:
                        raise TimeoutError(f"DO task did not finish within {timeout:.1f} s")
                    time.sleep(0.5)
                do_task.wait_until_done(timeout=1.0)
                execution_time = time.time() - start_time
                logger.info(f"✓ Protocol execution completed in {execution_time:.2f} seconds")

                _wait_for_fictrac_frame_drain(
                    callback=fictrac_callback,
                    expected_frame_count=expected_camera_frames if runtime_cfg.verify_camera_recording else None,
                    camera_fps=runtime_cfg.camera_trigger_fps_hz,
                    logger=logger,
                    health_check=lambda: _check_fictrac_health(
                        runtime_cfg,
                        fictrac_callback,
                        fictrac_thread,
                        fictrac_state,
                        other_camera_recorder,
                    ),
                )

                if other_camera_recorder is not None:
                    other_camera_recording = _stop_other_camera_recorder(
                        recorder=other_camera_recorder,
                        logger=logger,
                    )
                    other_camera_recorder = None

                _stop_fictrac(
                    fictrac_driver=fictrac_driver,
                    fictrac_callback=fictrac_callback,
                    fictrac_thread=fictrac_thread,
                    fictrac_camera_index=fictrac_camera_index,
                    logger=logger,
                )
                fictrac_driver = None
                fictrac_thread = None

                logger.info("Stopping tasks and reading data...")
                _safe_stop_task(do_task, logger, "DO master")
                if ao_names:
                    _safe_stop_task(ao_task, logger, "AO slave")

                if ai_phys:
                    logger.info("Reading AI data...")
                    try:
                        ai_reader.read_many_sample(
                            ai_buf,
                            number_of_samples_per_channel=N,
                            timeout=max(10.0, N / rate + 5.0),
                        )
                        _safe_stop_task(ai_task, logger, "AI slave")

                        ai_file = run_dir / "capture_ai.npz"
                        np.savez_compressed(
                            ai_file,
                            names=np.array(ai_names, dtype=object),
                            data=ai_buf.astype(np.float32),
                        )
                        logger.info(f"✓ AI data saved: {ai_file}")
                        logger.info(f"  AI data shape: {ai_buf.shape}")
                        for i, name in enumerate(ai_names):
                            min_val, max_val = ai_buf[i].min(), ai_buf[i].max()
                            mean_val = ai_buf[i].mean()
                            logger.debug(f"    {name}: min={min_val:.3f}V, max={max_val:.3f}V, mean={mean_val:.3f}V")
                    except Exception as e:
                        logger.error(f"Failed to read AI data: {e}")
                        raise

                if di_phys:
                    logger.info("Reading DI data...")
                    try:
                        di_data = di_task.read(
                            number_of_samples_per_channel=N,
                            timeout=max(10.0, N / rate + 5.0)
                        )
                        _safe_stop_task(di_task, logger, "DI slave")

                        di_file = run_dir / "capture_di.npz"
                        np.savez_compressed(
                            di_file,
                            names=np.array(di_names, dtype=object),
                            data=np.array(di_data).astype(np.bool_),
                        )
                        logger.info(f"✓ DI data saved: {di_file}")

                        di_bool = np.array(di_data).astype(bool)
                        logger.info(f"  DI data shape: {di_bool.shape}")
                        for i, name in enumerate(di_names):
                            high_count = np.sum(di_bool[i])
                            high_pct = high_count / N * 100
                            logger.debug(f"    {name}: {high_count}/{N} samples high ({high_pct:.1f}%)")
                    except Exception as e:
                        logger.error(f"Failed to read DI data: {e}")
                        raise
            except KeyboardInterrupt:
                run_interrupted = True
                logger.warning("Keyboard interrupt received; stopping protocol early...")
            except Exception as e:
                logger.error(f"Protocol execution failed: {e}")
                raise
            finally:
                if progress_monitor:
                    progress_monitor.stop()
                _stop_protocol_tasks(
                    do_task=do_task,
                    ao_task=ao_task,
                    ai_task=ai_task,
                    di_task=di_task,
                    logger=logger,
                )

            if run_interrupted:
                logger.warning("Run interrupted by user; skipping remaining acquisition and post-run artifact generation.")
                return

        logger.info("✓ All DAQ tasks completed and data acquired")
    finally:
        _stop_fictrac(
            fictrac_driver=fictrac_driver,
            fictrac_callback=fictrac_callback,
            fictrac_thread=fictrac_thread,
            fictrac_camera_index=fictrac_camera_index,
            logger=logger,
        )
        if other_camera_recorder is not None:
            other_camera_recording = _stop_other_camera_recorder(
                recorder=other_camera_recorder,
                logger=logger,
            )
        if teensy_serial_monitor is not None:
            teensy_serial_transcript = teensy_serial_monitor.get_transcript()
            teensy_serial_monitor.close()
            if teensy_serial_transcript:
                teensy_transcript_file = run_dir / "teensy_serial_transcript.jsonl"
                with open(teensy_transcript_file, "w", encoding="utf-8") as handle:
                    for entry in teensy_serial_transcript:
                        json.dump(entry, handle, default=str)
                        handle.write("\n")
                logger.info(f"✓ Teensy serial transcript saved: {teensy_transcript_file}")

    logger.info("✓ All DAQ tasks completed and data acquired")

    # Post-run interactive viz with AI/DI overlays (if recorded)
    logger.info("=== Generating Post-Run Visualization ===")
    
    di_names_overlay = di_data_overlay = None
    ai_names_overlay = ai_data_overlay = None
    
    # Load DI data if available
    di_file = run_dir / "capture_di.npz"
    if di_file.exists():
        logger.info("Loading DI data for visualization overlay...")
        npz_di = np.load(di_file, allow_pickle=True)
        di_names_overlay = list(npz_di["names"])
        di_data_overlay = npz_di["data"].astype(bool)
        logger.info(f"  ✓ DI overlay data loaded: {len(di_names_overlay)} channels")
        logger.debug(f"    DI overlay shape: {di_data_overlay.shape}")

    # Load AI data if available  
    ai_file = run_dir / "capture_ai.npz"
    if ai_file.exists():
        logger.info("Loading AI data for visualization overlay...")
        npz_ai = np.load(ai_file, allow_pickle=True)
        ai_names_overlay = list(npz_ai["names"])
        ai_data_overlay = npz_ai["data"]
        logger.info(f"  ✓ AI overlay data loaded: {len(ai_names_overlay)} channels")
        logger.debug(f"    AI overlay shape: {ai_data_overlay.shape}")

    if fictrac_callback is not None:
        fictrac_frames_file = run_dir / "fictrac_frames.npz"
        fictrac_callback.save_npz(fictrac_frames_file)
        logger.info(f"✓ FicTrac frames saved: {fictrac_frames_file}")

    if fictrac_runtime_info:
        fictrac_runtime_file = run_dir / "fictrac_runtime.json"
        fictrac_runtime_file.write_text(json.dumps(fictrac_runtime_info, indent=2), encoding="utf-8")
        logger.info(f"✓ FicTrac runtime info saved: {fictrac_runtime_file}")

    if fictrac_runtime_info.get("save_raw"):
        fictrac_recording = _build_fictrac_recording_summary(
            run_dir=run_dir,
            runtime_info=fictrac_runtime_info,
            frame_count=fictrac_callback.frame_count if fictrac_callback is not None else None,
            expected_frame_count=expected_camera_frames if runtime_cfg.verify_camera_recording else None,
        )
        fictrac_recording_file = run_dir / "fictrac_camera_recording.json"
        fictrac_recording_file.write_text(json.dumps(fictrac_recording, indent=2), encoding="utf-8")
        logger.info(f"✓ FicTrac recording summary saved: {fictrac_recording_file}")

    if other_camera_recording:
        from multibios.blackfly.triggered_camera_record import postprocess_triggered_camera_recording

        other_camera_recording = postprocess_triggered_camera_recording(
            other_camera_recording,
            expected_frame_count=expected_camera_frames if runtime_cfg.verify_camera_recording else None,
            nominal_fps=nominal_camera_fps,
            convert_to_lossless_mkv=runtime_cfg.convert_second_camera_bin_to_lossless_mkv,
        )
        blackfly_recording_file = run_dir / "blackfly_recording.json"
        blackfly_recording_file.write_text(json.dumps(other_camera_recording, indent=2), encoding="utf-8")
        logger.info(f"✓ Blackfly recording summary saved: {blackfly_recording_file}")

    if fictrac_runtime_info.get("save_raw") or other_camera_recording:
        parity_summary, parity_summary_path = _write_parity_summary(run_dir)
        logger.info(f"✓ Parity audit saved: {parity_summary_path}")

        fictrac_recording, other_camera_recording = _finalize_raw_chunk_retention(
            run_dir=run_dir,
            policy=runtime_cfg.raw_chunk_retention_policy,
            parity_summary=parity_summary,
            parity_path=parity_summary_path,
            fictrac_recording=fictrac_recording if fictrac_runtime_info.get("save_raw") else None,
            blackfly_recording=other_camera_recording,
            logger=logger,
        )

        if fictrac_runtime_info.get("save_raw") and fictrac_recording is not None:
            fictrac_recording_file = run_dir / "fictrac_camera_recording.json"
            fictrac_recording_file.write_text(json.dumps(fictrac_recording, indent=2), encoding="utf-8")
        if other_camera_recording is not None:
            blackfly_recording_file = run_dir / "blackfly_recording.json"
            blackfly_recording_file.write_text(json.dumps(other_camera_recording, indent=2), encoding="utf-8")

    # Generate comprehensive visualization
    if args.interactive:
        logger.info("Generating comprehensive interactive figure...")
        fig = make_protocol_figure(
            t_ms,
            comp.do.astype(bool),
            do_names,
            comp.ao,
            ao_names,
            ai=ai_data_overlay,
            ai_names=ai_names_overlay,
            di=di_data_overlay,
            di_names=di_names_overlay,
            rck_log=comp.rck_log,
            title="Protocol (DO/AO) + Digital Inputs (DI) + MFC Feedback (AI)",
        )
        
        final_preview = run_dir / "preview.html"
        logger.info("Writing final interactive HTML...")
        fig.write_html(final_preview, include_plotlyjs="cdn")
        logger.info(f"✓ Final visualization saved: {final_preview}")

    # Generate DI edge log if present
    if di_file.exists():
        logger.info("Computing DI line edge transitions...")
        di_edge_file = run_dir / "di_edges.csv"
        write_edge_csv(di_edge_file, di_names_overlay, di_data_overlay, comp.dt_ms)
        
        # Count DI edges for summary
        di_edges_total = 0
        for i in range(len(di_names_overlay)):
            edges = np.sum(np.diff(di_data_overlay[i, :].astype(int)) != 0)
            di_edges_total += edges
            logger.debug(f"    {di_names_overlay[i]}: {edges} DI transitions")
        logger.info(f"✓ DI edges saved: {di_edge_file} ({di_edges_total} total transitions)")

    # Final summary
    logger.info("=== RUN COMPLETION SUMMARY ===")
    logger.info(f"✓ Run directory: {run_dir}")
    logger.info(f"✓ Protocol duration: {comp.N * comp.dt_ms:.1f} ms ({comp.N * comp.dt_ms / 1000:.2f} seconds)")
    logger.info(f"✓ Total samples: {comp.N:,}")
    logger.info(f"✓ Sample rate: {comp.tcfg.sample_rate} Hz")
    logger.info(f"✓ Digital outputs: {len(do_names)} channels")
    logger.info(f"✓ Analog outputs: {len(ao_names)} channels")
    if ai_names_overlay:
        logger.info(f"✓ Analog inputs captured: {len(ai_names_overlay)} channels")
    if di_names_overlay:
        logger.info(f"✓ Digital inputs captured: {len(di_names_overlay)} channels")
    
    # List all generated files
    logger.info("Generated artifacts:")
    for file_path in sorted(run_dir.glob("*")):
        size_bytes = file_path.stat().st_size
        size_str = f"{size_bytes:,} bytes"
        if size_bytes > 1024:
            size_str = f"{size_bytes/1024:.1f} KB"
        if size_bytes > 1024*1024:
            size_str = f"{size_bytes/(1024*1024):.1f} MB"
        logger.info(f"  {file_path.name}: {size_str}")

    print(f"Run complete. See interactive preview: {run_dir/'preview.html'}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted by user.")
        raise SystemExit(130)
