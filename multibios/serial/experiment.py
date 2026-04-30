#!/usr/bin/env python3
"""
experiment.py — Open-loop experiment runner (computer-timebase + serial).

Parses the same YAML protocol format used by the NIDAQ-compiled approach,
but executes it via:

  * **Teensy v2 serial** for odor valve control
  * **AlicatManager serial** for MFC setpoints
  * **NIDAQ finite DO task** for camera/microscope triggers & latch pulses
    * **Internal FicTrac client** for ball tracking (recorded throughout)

Architecture
------------
::

    ┌────────────────────────────────────────────────┐
    │  Main thread                                   │
    │  1. Parse YAML -> build timeline + DAQ waveform │
    │  2. Start FicTrac (background thread)          │
    │  3. Optionally set MFC setpoints (async)       │
    │  4. Start NIDAQ trigger task                   │
    │  5. Walk timeline on time.perf_counter()       │
    │     -> serial commands to Teensy / Alicat       │
    │  6. Wait for NIDAQ to finish                   │
    │  7. Stop FicTrac, save data                    │
    └────────────────────────────────────────────────┘

Closed-loop readiness
---------------------
The ``ExperimentCallback`` exposes the latest FicTrac state via a thread-safe
property and supports waiting for the next received frame by sequence number.
A future closed-loop engine can read the newest frame without trying to process
every intermediate frame when the control loop runs slower than camera rate.

Usage
-----
::

    python -m multibios.experiment \\
        --protocol config/example_protocol.yaml \\
        --hardware config/hardware.yaml \\
        --experiment config/experiment_config.yaml \\
        --dry-run          # preview, no hardware
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import shutil
import sys
import threading
import time
import warnings
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple

import numpy as np
import yaml

# MultiBiOS imports
from multibios.fictrac_client import (FICTRAC_FRAME_DTYPE, BaseFicTracCallback,
                                      FicTracDriver, FicTracFrame,
                                      FicTracFrameStore)
from multibios.fictrac_consumer import ClosedLoopFrameConsumer
from multibios.protocol.control_plan import (TimelineEvent,
                                             compile_control_plan,
                                             write_control_plan_csv)
from multibios.fictrac_runtime import prepare_fictrac_runtime
from multibios.protocol.schema import (BIG_STATE_CODE, SMALL_STATE_CODE,
                                       CompileError)
from multibios.serial_line_monitor import SerialLineMonitor
from multibios.serial.daq_triggers import (DAQTriggerManager, TriggerConfig,
                                           build_trigger_waveform)
from multibios.serial.teensy_controller import TeensyController

@dataclass
class ExperimentConfig:
    """Configuration loaded from experiment_config.yaml."""
    # Teensy serial port
    teensy_port: str = "COM3"
    teensy_baud: int = 115_200

    # FicTrac settings
    fictrac_config: str = ""
    fictrac_bin: str = ""
    fictrac_console_out: str = "fictrac_output.txt"
    fictrac_first_frame_timeout_ms: int = 0
    fictrac_startup_timeout_s: float = 90.0
    fictrac_timeout_s: float = 5.0
    save_fictrac_camera_video: bool = False
    fictrac_raw_video_codec: str = "raw"
    save_second_camera_video: bool = False
    second_camera_index: int | None = None
    second_camera_timeout_ms: int = 250
    second_camera_queue_size: int = 512
    second_camera_stream_buffer_count: int = 256
    second_camera_exposure_us: float | None = None
    second_camera_roi_width: int | None = None
    second_camera_roi_height: int | None = None
    second_camera_binning: int = 1
    verify_camera_recording: bool = True
    convert_second_camera_bin_to_lossless_mkv: bool = True

    # MFC control mode: "alicat_serial" or "none" (skip MFC)
    mfc_mode: str = "alicat_serial"
    # Mapping from protocol YAML device names -> Alicat unit names
    # e.g. {"mfc.air_left_setpoint": "A", "mfc.odor_left_setpoint": "B", ...}
    mfc_device_map: Dict[str, str] = field(default_factory=dict)
    # Alicat scan options
    alicat_ports: List[str] = field(default_factory=list)
    alicat_baud: List[int] = field(default_factory=list)
    # Expected Alicat unit IDs — scan stops early once all are found; warns if any missing
    alicat_expected_ids: List[str] = field(default_factory=list)

    # DAQ trigger overrides (applied on top of protocol timing section)
    latch_interval_ms: float = 50.0

    # Output directory base
    data_dir: str = "data/runs"

    # Automatically open the explorer in a browser when the run finishes
    open_explorer: bool = True
    # Port the explorer server listens on
    explorer_port: int = 8050

    # Live MFC readout interval (seconds) during the experiment run.
    # A compact status line is printed every N seconds showing actual flow,
    # setpoint, and gas for every Alicat device.
    # Set to 0 to disable live readout (setpoint commands still work normally).
    mfc_live_interval_s: float = 1.0

# ═══════════════════════════════════════════════════════════════════════════
# FicTrac callback — records every frame, exposes latest state
# ═══════════════════════════════════════════════════════════════════════════

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

    output_base = (target_dir.resolve() / "fictrac").as_posix()
    _upsert_fictrac_config_line(lines, "output_fn", output_base)
    _upsert_fictrac_config_line(lines, "src_first_frame_timeout_ms", str(int(first_frame_timeout_ms)))
    if enable_raw_video:
        _upsert_fictrac_config_line(lines, "save_raw", "y")
        _upsert_fictrac_config_line(lines, "vid_codec", video_codec)
        if camera_fps is not None and camera_fps > 0:
            _upsert_fictrac_config_line(lines, "src_fps", f"{camera_fps:.6f}")

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
        "fictrac_camera_index": fictrac_camera_index,
    }


def _load_yaml_file(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    yaml_path = Path(path)
    if not yaml_path.exists():
        return {}
    with open(yaml_path, encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _yaml_section(raw: dict[str, Any], key: str) -> dict[str, Any]:
    value = raw.get(key, {})
    return value if isinstance(value, dict) else {}


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


def _count_rising_edges(trace: np.ndarray | None) -> int | None:
    if trace is None:
        return None
    if trace.size == 0:
        return 0
    bool_trace = np.asarray(trace, dtype=np.bool_)
    return int(bool_trace[0]) + int(np.count_nonzero(~bool_trace[:-1] & bool_trace[1:]))


def _discover_fictrac_raw_videos(run_dir: Path) -> list[str]:
    return sorted(str(path) for path in run_dir.glob("fictrac-raw-*"))


def _build_fictrac_recording_summary(
    *,
    run_dir: Path,
    runtime_info: dict[str, Any],
    frame_count: int | None,
    expected_frame_count: int | None,
) -> dict[str, Any]:
    actual_frames = None if frame_count is None else int(frame_count)
    missing_frames = None
    no_dropped_frames = None
    if expected_frame_count is not None and actual_frames is not None:
        missing_frames = max(int(expected_frame_count) - actual_frames, 0)
        no_dropped_frames = actual_frames == int(expected_frame_count)

    return {
        "camera_index": runtime_info.get("fictrac_camera_index"),
        "save_raw": bool(runtime_info.get("save_raw", False)),
        "video_codec": runtime_info.get("video_codec"),
        "camera_fps": runtime_info.get("camera_fps"),
        "output_base": runtime_info.get("output_base"),
        "raw_videos": _discover_fictrac_raw_videos(run_dir),
        "actual_frames": actual_frames,
        "expected_frames": expected_frame_count,
        "missing_frames_vs_expected": missing_frames,
        "no_dropped_frames": no_dropped_frames,
    }


def compile_timeline(
    protocol_yaml: Dict[str, Any],
    seed: Optional[int] = None,
) -> Tuple[List[TimelineEvent], List[float], List[Tuple[float, float]], float]:
    plan = compile_control_plan(protocol_yaml, seed=seed)
    return (
        plan.timeline,
        plan.microscope_times_ms,
        plan.camera_windows_ms,
        plan.total_duration_ms,
    )


# ═══════════════════════════════════════════════════════════════════════════
# MFC helper (Alicat serial)
# ═══════════════════════════════════════════════════════════════════════════

async def _apply_mfc_setpoints(
    mgr: Any,
    mfc_commands: Dict[str, float],
    mfc_device_map: Dict[str, str],
    alicat_ports: List[str],
    alicat_baud: List[int],
    alicat_expected_ids: Optional[List[str]] = None,
) -> None:
    """Apply MFC setpoints using an existing AlicatManager instance.

    ``mfc_commands`` maps protocol device names (e.g. ``mfc.air_left_setpoint``)
    to target flow rates. ``mfc_device_map`` may map those to either full
    Alicat manager device names (for example ``A@COM8``) or bare unit IDs
    (for example ``A``).
    """
    def _resolve_target_name(configured_name: str) -> Optional[str]:
        """Resolve a configured Alicat target to a cached device name."""
        raw = str(configured_name).strip()
        if not raw:
            return None

        if raw in mgr.device_map:
            return raw

        raw_upper = raw.upper()
        matches = [
            name for name, info in mgr.device_map.items()
            if name.upper() == raw_upper
            or info.get("unit", "").upper() == raw_upper
            or name.split("@", 1)[0].upper() == raw_upper
        ]
        if len(matches) == 1:
            return matches[0]

        controller_matches = [
            name for name in matches
            if mgr.device_map.get(name, {}).get("type") == "controller"
        ]
        if len(controller_matches) == 1:
            return controller_matches[0]

        if len(matches) > 1:
            print(
                f"  WARNING: Alicat target '{configured_name}' is ambiguous; "
                f"matches {matches}. Use a full device name like 'A@COM8'."
            )
        return None

    targets: Dict[str, float] = {}
    for dev_key, flow_rate in mfc_commands.items():
        configured_name = mfc_device_map.get(dev_key)
        alicat_name = _resolve_target_name(configured_name) if configured_name else None
        if configured_name and alicat_name is None:
            print(
                f"  WARNING: No cached Alicat device matches mapping "
                f"{dev_key} -> {configured_name}. Available: {mgr.names()}"
            )
        if alicat_name:
            targets[alicat_name] = flow_rate

    if targets:
        results = await mgr.set_all(targets)
        for name, err in results.items():
            if err:
                print(f"  WARNING: MFC {name}: set failed — {err}")
    elif mfc_commands:
        print("  WARNING: No MFC targets resolved; no Alicat setpoints were sent.")


async def _set_mfc_setpoints(
    mfc_commands: Dict[str, float],
    mfc_device_map: Dict[str, str],
    alicat_ports: List[str],
    alicat_baud: List[int],
    alicat_expected_ids: List[str] | None = None,
) -> None:
    """Apply MFC setpoints (creates its own AlicatManager).

    Kept for backward compatibility.  New code should use
    ``_MFCMonitor.set_setpoints()`` so all MFC I/O shares a single
    persistent event loop and AlicatManager instance.
    """
    try:
        from multibios.alicat_manager import AlicatManager  # legacy serial path
    except ImportError:
        raise RuntimeError(
            "AlicatManager (serial MFC control) has been moved to legacy/. "
            "MFC setpoints are now driven via NI-DAQ analog outputs — "
            "use run_protocol.py or tests/mfc_analog_test.py instead."
        )

    mgr = AlicatManager()
    if not mgr.device_map:
        scan_kw: Dict[str, Any] = {}
        if alicat_ports:
            scan_kw["ports"] = alicat_ports
        if alicat_baud:
            scan_kw["baudrates"] = alicat_baud
        if alicat_expected_ids:
            scan_kw["expected_ids"] = alicat_expected_ids
        await mgr.scan(**scan_kw)

    await _apply_mfc_setpoints(
        mgr, mfc_commands, mfc_device_map, alicat_ports, alicat_baud, alicat_expected_ids
    )


# ═══════════════════════════════════════════════════════════════════════════
# MFC monitor — persistent background asyncio loop
# ═══════════════════════════════════════════════════════════════════════════

class _MFCMonitor:
    """Background asyncio event loop for all Alicat MFC I/O during an experiment.

    Runs a single persistent asyncio event loop on a daemon thread so that
    all serial operations share one ``AlicatManager`` instance — avoiding the
    ``FlowMeter.open_ports`` class-cache collisions that arise when multiple
    ``asyncio.run()`` calls each create a fresh event loop.

    When *live_interval_s* > 0 a periodic poll task wakes every
    *live_interval_s* seconds, reads every MFC device, and prints a compact
    status line to stdout, interleaved naturally with the experiment output::

        [   12.0s] MFC: A@COM7 flow=10.001 setpt=10.0 gas=Air  |  B@COM7 flow=5.002 setpt=5.0 gas=Air

    Set *live_interval_s* = 0 to disable live readout (setpoint commands still
    use this monitor's shared loop).
    """

    def __init__(
        self,
        *,
        device_map: Dict[str, str],
        alicat_ports: List[str],
        alicat_baud: List[int],
        alicat_expected_ids: List[str],
        live_interval_s: float = 1.0,
    ) -> None:
        self._device_map = device_map
        self._alicat_ports = alicat_ports
        self._alicat_baud = alicat_baud
        self._alicat_expected_ids = alicat_expected_ids
        self._live_interval = live_interval_s

        self._loop: asyncio.AbstractEventLoop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._loop.run_forever, daemon=True, name="MFCMonitor"
        )
        self._mgr: Optional[Any] = None          # AlicatManager, set in start()
        self._poll_task: Optional[asyncio.Task] = None  # lives inside the loop
        self._t0: float = 0.0

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def start(self, t0: float = 0.0) -> None:
        """Start the background event loop and initialise AlicatManager."""
        try:
            from multibios.alicat_manager import AlicatManager  # legacy serial path
        except ImportError:
            raise RuntimeError(
                "AlicatManager (serial MFC control) has been moved to legacy/. "
                "MFC setpoints are now driven via NI-DAQ analog outputs — "
                "use run_protocol.py or tests/mfc_analog_test.py instead."
            )

        self._t0 = t0
        self._thread.start()

        async def _init() -> None:
            self._mgr = AlicatManager()
            if not self._mgr.device_map:
                kw: Dict[str, Any] = {}
                if self._alicat_ports:
                    kw["ports"] = self._alicat_ports
                if self._alicat_baud:
                    kw["baudrates"] = self._alicat_baud
                if self._alicat_expected_ids:
                    kw["expected_ids"] = self._alicat_expected_ids
                await self._mgr.scan(**kw)
            if self._live_interval > 0:
                self._poll_task = asyncio.create_task(self._poll_loop())

        asyncio.run_coroutine_threadsafe(_init(), self._loop).result(timeout=30.0)

    def set_t0(self, t0: float) -> None:
        """Update the experiment start reference time."""
        self._t0 = t0

    def stop(self) -> None:
        """Cancel the poll task and shut down the background event loop."""
        async def _cancel() -> None:
            if self._poll_task is not None:
                self._poll_task.cancel()
                try:
                    await self._poll_task
                except asyncio.CancelledError:
                    pass

        asyncio.run_coroutine_threadsafe(_cancel(), self._loop).result(timeout=5.0)
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=5.0)

    # ── Setpoints ─────────────────────────────────────────────────────────────

    def set_setpoints(self, mfc_commands: Dict[str, float]) -> None:
        """Send MFC setpoints (blocks the calling thread until complete)."""
        if self._mgr is None:
            raise RuntimeError("_MFCMonitor not started")
        future = asyncio.run_coroutine_threadsafe(
            _apply_mfc_setpoints(
                self._mgr, mfc_commands, self._device_map,
                self._alicat_ports, self._alicat_baud,
                self._alicat_expected_ids or None,
            ),
            self._loop,
        )
        try:
            future.result(timeout=30.0)
        except Exception as e:
            print(f"  WARNING: MFC set failed: {e}")

    def zero_all(self) -> None:
        """Zero all MFC controllers (blocks). Called during shutdown."""
        if self._mgr is None:
            return
        zero_cmds = {dev: 0.0 for dev in self._device_map}
        future = asyncio.run_coroutine_threadsafe(
            _apply_mfc_setpoints(
                self._mgr, zero_cmds, self._device_map,
                self._alicat_ports, self._alicat_baud,
                self._alicat_expected_ids or None,
            ),
            self._loop,
        )
        try:
            future.result(timeout=30.0)
        except Exception as e:
            print(f"    WARNING: MFC zeroing failed: {e}")

    # ── Live display ──────────────────────────────────────────────────────────

    async def _poll_loop(self) -> None:
        """Poll all MFC devices and print a status line every live_interval_s."""
        while True:
            await asyncio.sleep(self._live_interval)
            try:
                results = await self._mgr.get_all()
                elapsed = time.perf_counter() - self._t0
                self._display(results, elapsed)
            except asyncio.CancelledError:
                raise
            except Exception:
                pass

    def _display(self, results: Dict[str, Any], elapsed_s: float) -> None:
        if not results:
            return
        parts: List[str] = []
        for name in sorted(results):
            s = results[name]
            if isinstance(s, dict) and "error" not in s:
                flow = s.get("mass_flow", "?")
                setpt = s.get("setpoint", "?")
                gas = s.get("gas", "?")
                flow_str = f"{flow:.3f}" if isinstance(flow, (int, float)) else str(flow)
                parts.append(f"{name} flow={flow_str} setpt={setpt} gas={gas}")
            else:
                err = s.get("error", "?") if isinstance(s, dict) else str(s)
                parts.append(f"{name} ERR={err}")
        print(f"  [{elapsed_s:7.1f}s] MFC: " + "  |  ".join(parts))


# ═══════════════════════════════════════════════════════════════════════════
# Experiment runner
# ═══════════════════════════════════════════════════════════════════════════

class ExperimentRunner:
    """Orchestrates an open-loop experiment.

    1. Compiles the YAML protocol into a timeline
    2. Builds the NIDAQ trigger waveform
    3. Starts FicTrac
    4. Starts the NIDAQ trigger task
    5. Walks the timeline, sending serial commands
    6. Saves all data on completion
    """

    def __init__(
        self,
        protocol_path: str | Path,
        hardware_path: str | Path,
        experiment_cfg: ExperimentConfig,
        *,
        seed: Optional[int] = None,
        dry_run: bool = False,
        verbose: bool = False,
    ) -> None:
        self.protocol_path = Path(protocol_path)
        self.hardware_path = Path(hardware_path)
        self.cfg = experiment_cfg
        self.seed = seed
        self.dry_run = dry_run
        self.verbose = verbose

        # Loaded at prepare()
        self.protocol_yaml: Dict[str, Any] = {}
        self.timeline: List[TimelineEvent] = []
        self.microscope_times: List[float] = []
        self.camera_windows: List[Tuple[float, float]] = []
        self.total_duration_ms: float = 0.0
        self.trigger_waveform: Optional[np.ndarray] = None
        self.trigger_cfg: Optional[TriggerConfig] = None

        # Runtime
        self._teensy: Optional[TeensyController] = None
        self._daq: Optional[DAQTriggerManager] = None
        self._fictrac_callback: Optional[ExperimentCallback] = None
        self._fictrac_driver: Optional[FicTracDriver] = None
        self._fictrac_thread: Optional[threading.Thread] = None
        self._fictrac_error: Optional[Exception] = None   # set if thread crashes
        self._fictrac_runtime_info: Dict[str, Any] = {}
        self._mfc_monitor: Optional[_MFCMonitor] = None
        self._event_log: List[Dict[str, Any]] = []
        self._teensy_transcript: List[Dict[str, Any]] = []
        self._teensy_serial_monitor: Optional[SerialLineMonitor] = None
        self._run_dir: Optional[Path] = None
        self._t0: float = 0.0  # experiment start (perf_counter)
        self._other_camera_recorder: Any = None
        self._other_camera_recording: Dict[str, Any] = {}

    # ── Phase 1: Prepare ────────────────────────────────────────────────────

    def prepare(self) -> None:
        """Parse protocol, build timeline and DAQ waveform."""
        # Load protocol
        with open(self.protocol_path, encoding="utf-8") as f:
            self.protocol_yaml = yaml.safe_load(f)

        # Compile timeline
        self.timeline, self.microscope_times, self.camera_windows, self.total_duration_ms = \
            compile_timeline(self.protocol_yaml, seed=self.seed)

        if self.verbose:
            print(f"Protocol: {self.protocol_yaml.get('protocol', {}).get('name', '?')}")
            print(f"Total duration: {self.total_duration_ms / 1000:.1f} s")
            print(f"Timeline events: {len(self.timeline)}")
            print(f"Microscope triggers: {len(self.microscope_times)}")
            print(f"Camera windows: {len(self.camera_windows)}")

        # Build trigger config from protocol timing section
        timing = self.protocol_yaml.get("protocol", {}).get("timing", {})
        self.trigger_cfg = TriggerConfig(
            sample_rate=int(timing.get("sample_rate", 2000)),
            camera_interval_ms=float(timing.get("camera_interval", 100)),
            camera_pulse_ms=float(timing.get("camera_pulse_duration", 5)),
            trig_pulse_ms=float(timing.get("trig_pulse_ms", 5)),
            latch_interval_ms=self.cfg.latch_interval_ms,
            preload_lead_ms=float(timing.get("preload_lead_ms", 2)),
            load_req_ms=float(timing.get("load_req_ms", 1)),
            rck_pulse_ms=float(timing.get("rck_pulse_ms", 1)),
        )

        # Build waveform
        self.trigger_waveform = build_trigger_waveform(
            total_duration_ms=self.total_duration_ms,
            cfg=self.trigger_cfg,
            microscope_times_ms=self.microscope_times,
            camera_enable_windows=self.camera_windows if self.camera_windows else None,
        )

        if self.verbose:
            N = self.trigger_waveform.shape[1]
            print(f"DAQ waveform: {N} samples @ {self.trigger_cfg.sample_rate} Hz "
                  f"({N / self.trigger_cfg.sample_rate:.1f} s)")
            print(f"Latch interval: {self.trigger_cfg.latch_interval_ms} ms")

        # Create output directory
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self._run_dir = Path(self.cfg.data_dir) / ts
        if not self.dry_run:
            self._run_dir.mkdir(parents=True, exist_ok=True)

    # ── Phase 2: Preview ────────────────────────────────────────────────────

    def preview(self) -> None:
        """Print the timeline for preview / dry-run."""
        print(f"\n{'='*80}")
        print(f"EXPERIMENT TIMELINE PREVIEW")
        print(f"{'='*80}")
        print(f"Total duration: {self.total_duration_ms / 1000:.1f} s "
              f"({self.total_duration_ms / 60000:.1f} min)")
        print(f"{'='*80}\n")

        current_phase = ""
        for evt in self.timeline:
            if evt.phase != current_phase:
                current_phase = evt.phase
                print(f"\n--- Phase: {current_phase} ---")

            t_s = evt.time_ms / 1000.0
            if evt.action == "olfactometer":
                print(f"  [{t_s:8.2f}s] OLFACTOMETER {evt.side:5s} -> {evt.state}")
            elif evt.action == "switch_valve":
                print(f"  [{t_s:8.2f}s] SWITCH_VALVE {evt.side:5s} -> {evt.state}")
            elif evt.action == "mfc":
                print(f"  [{t_s:8.2f}s] MFC {evt.device:30s} -> {evt.value:.2f}")
            elif evt.action == "log_only":
                print(f"  [{t_s:8.2f}s] {evt.device:30s} -> {evt.state}")

        print(f"\n{'='*80}")
        print(f"Microscope triggers at: {[f'{t/1000:.1f}s' for t in self.microscope_times]}")
        print(f"Camera windows: {[(f'{s/1000:.1f}s', f'{e/1000:.1f}s') for s, e in self.camera_windows]}")
        print(f"{'='*80}\n")

    # ── Phase 3: Run ────────────────────────────────────────────────────────

    def run(self) -> None:
        """Execute the full experiment."""
        if self.dry_run:
            print("DRY RUN — no hardware interaction.")
            self._simulate_timeline()
            self._save_metadata()
            return

        try:
            self._start_hardware()
            self._execute_timeline()
        except KeyboardInterrupt:
            print("\n⚠ Experiment interrupted by user!")
            self._log_event("INTERRUPTED", "KeyboardInterrupt")
        except Exception as e:
            print(f"\n⚠ Experiment error: {e}")
            if self.verbose:
                import traceback
                traceback.print_exc()
            self._log_event("ERROR", str(e))
            raise
        finally:
            self._shutdown()
            self._save_data()

    def _start_hardware(self) -> None:
        """Initialize all hardware subsystems."""
        # 1. Open Teensy serial
        print(f"Opening Teensy on {self.cfg.teensy_port}...")
        self._teensy_serial_monitor = SerialLineMonitor(
            port=self.cfg.teensy_port,
            baudrate=self.cfg.teensy_baud,
            timeout=1.0,
            boot_delay_s=0.5,
            reset_input_buffer_on_open=True,
        )
        self._teensy = TeensyController(
            port=self.cfg.teensy_port,
            baudrate=self.cfg.teensy_baud,
            serial_monitor=self._teensy_serial_monitor,
        )
        self._teensy.open()
        resp = self._teensy.reset()
        print(f"  Teensy RESET: {resp}")

        # 2. Start MFC monitor (owns the background asyncio loop for all Alicat I/O)
        if self.cfg.mfc_mode == "alicat_serial" and self.cfg.mfc_device_map:
            live = self.cfg.mfc_live_interval_s
            print(
                f"  Starting MFC monitor "
                f"({'live readout every ' + str(live) + ' s' if live > 0 else 'live readout disabled'})..."
            )
            self._mfc_monitor = _MFCMonitor(
                device_map=self.cfg.mfc_device_map,
                alicat_ports=self.cfg.alicat_ports,
                alicat_baud=self.cfg.alicat_baud,
                alicat_expected_ids=self.cfg.alicat_expected_ids,
                live_interval_s=live,
            )
            self._mfc_monitor.start()
            self._apply_initial_mfcs()

        # 3. Start FicTrac
        if self.cfg.fictrac_config:
            if self._run_dir is None:
                raise RuntimeError("Run directory was not prepared before starting FicTrac.")

            print("Starting FicTrac...")
            nominal_camera_fps = None
            if self.trigger_cfg is not None and self.trigger_cfg.camera_interval_ms > 0:
                nominal_camera_fps = 1000.0 / self.trigger_cfg.camera_interval_ms

            fictrac_config_path, fictrac_camera_index, self._fictrac_runtime_info = _prepare_fictrac_runtime_config(
                self.cfg.fictrac_config,
                self._run_dir,
                enable_raw_video=self.cfg.save_fictrac_camera_video,
                camera_fps=nominal_camera_fps,
                video_codec=self.cfg.fictrac_raw_video_codec,
                first_frame_timeout_ms=self.cfg.fictrac_first_frame_timeout_ms,
            )

            if self.cfg.save_second_camera_video:
                second_camera_index = self.cfg.second_camera_index
                if second_camera_index is None and fictrac_camera_index in (0, 1):
                    second_camera_index = 1 - fictrac_camera_index

                if second_camera_index is None:
                    raise RuntimeError(
                        "save_second_camera_video requires camera_recording.second_camera_index or a numeric FicTrac src_fn camera index of 0 or 1."
                    )
                if fictrac_camera_index is not None and second_camera_index == fictrac_camera_index:
                    raise RuntimeError(
                        "second_camera_index cannot match FicTrac's live camera index because the same Blackfly cannot be opened twice."
                    )

                from multibios.blackfly.triggered_camera_record import TriggeredCameraRecorder

                print(f"  Recording Blackfly camera {second_camera_index} into the run directory...")
                self._other_camera_recorder = TriggeredCameraRecorder(
                    camera_index=second_camera_index,
                    run_dir=self._run_dir,
                    timeout_ms=self.cfg.second_camera_timeout_ms,
                    queue_size=self.cfg.second_camera_queue_size,
                    stream_buffer_count=self.cfg.second_camera_stream_buffer_count,
                    exposure_us=self.cfg.second_camera_exposure_us,
                    roi_width=self.cfg.second_camera_roi_width,
                    roi_height=self.cfg.second_camera_roi_height,
                    binning=self.cfg.second_camera_binning,
                )
                self._other_camera_recording = self._other_camera_recorder.start()

            runtime_dirs = prepare_fictrac_runtime()
            if runtime_dirs:
                print("  FicTrac runtime PATH prepared:")
                for runtime_dir in runtime_dirs:
                    print(f"    {runtime_dir}")
            self._fictrac_callback = ExperimentCallback()
            self._fictrac_driver = FicTracDriver(
                config_file=str(fictrac_config_path),
                console_ouput_file=self.cfg.fictrac_console_out,
                track_change_callback=self._fictrac_callback,
                plot_on=False,
                fic_trac_bin_path=self.cfg.fictrac_bin or None,
            )
            self._fictrac_thread = threading.Thread(
                target=self._run_fictrac,
                name="FicTrac",
                daemon=True,
            )

        if self._fictrac_thread is not None:
            self._fictrac_thread.start()

        if self._fictrac_callback is not None:
            startup_timeout_s = self.cfg.fictrac_startup_timeout_s
            if startup_timeout_s <= 0:
                print("  Waiting indefinitely for FicTrac first frame...")
                while self._fictrac_callback.latest is None:
                    time.sleep(0.5)
                    self._check_fictrac_health()
            else:
                print(f"  Waiting up to {startup_timeout_s:.1f} s for FicTrac first frame...")
                deadline = time.monotonic() + startup_timeout_s
                while self._fictrac_callback.latest is None and time.monotonic() < deadline:
                    time.sleep(0.5)
                    self._check_fictrac_health()
            if self._fictrac_callback.latest is None:
                raise RuntimeError(
                    f"FicTrac did not produce any frames within {startup_timeout_s:.1f} s"
                )
            print(f"  FicTrac connected (frame {self._fictrac_callback.latest.frame_cnt})")

        # 4. Start NIDAQ triggers
        print("Starting NIDAQ trigger task...")
        self._daq = DAQTriggerManager(
            hw_path=self.hardware_path,
            cfg=self.trigger_cfg,
            waveform=self.trigger_waveform,
        )
        self._daq.start()
        print(f"  NIDAQ running ({self._daq.duration_s:.1f} s finite task)")

    def _run_fictrac(self) -> None:
        """Target for the FicTrac background thread."""
        try:
            self._fictrac_driver.run()
            # run() returned normally — FicTrac exited without raising.
            # Record this so _check_fictrac_health can distinguish it from
            # a deliberate stop (where request_stop() is called first).
            if not self._fictrac_callback._stop.is_set():
                self._fictrac_error = RuntimeError(
                    "FicTrac process exited unexpectedly (no exception)"
                )
        except Exception as e:
            self._fictrac_error = e
            print(f"  FicTrac thread error: {e}")
            self._log_event("ERROR", f"fictrac: {e}")

    def _check_fictrac_health(self) -> None:
        """Raise if FicTrac has died or stopped producing frames."""
        if self._other_camera_recorder is not None:
            self._other_camera_recorder.raise_if_failed()

        if self._fictrac_callback is None:
            return

        if self._fictrac_thread is not None and not self._fictrac_thread.is_alive():
            if self._fictrac_error is not None:
                raise RuntimeError(
                    f"FicTrac thread crashed: {self._fictrac_error}"
                ) from self._fictrac_error
            raise RuntimeError("FicTrac thread stopped unexpectedly")

        latest = self._fictrac_callback.latest
        if latest is None:
            return

        stale_for_s = time.perf_counter() - latest.wall_time
        if stale_for_s > self.cfg.fictrac_timeout_s:
            raise RuntimeError(
                f"FicTrac stopped producing frames for {stale_for_s:.1f} s "
                f"(timeout={self.cfg.fictrac_timeout_s:.1f} s)"
            )

    def _sleep_until_event(self, target_wall: float) -> None:
        """Sleep until a wall-clock target while monitoring FicTrac health."""
        while True:
            now = time.perf_counter()
            remaining = target_wall - now
            if remaining <= 0:
                return
            time.sleep(min(remaining, 0.5))
            self._check_fictrac_health()

    def _apply_initial_mfcs(self) -> None:
        """Apply all MFC setpoints that appear in the first phase."""
        initial_mfcs: Dict[str, float] = {}
        for evt in self.timeline:
            if evt.action == "mfc":
                initial_mfcs[evt.device] = evt.value
                # Only take the first occurrence per device
                if len(initial_mfcs) >= 4:
                    break

        if initial_mfcs and self.cfg.mfc_device_map:
            print(f"  Setting initial MFC setpoints: {initial_mfcs}")
            try:
                if self._mfc_monitor is not None:
                    self._mfc_monitor.set_setpoints(initial_mfcs)
                else:
                    asyncio.run(_set_mfc_setpoints(
                        initial_mfcs,
                        self.cfg.mfc_device_map,
                        self.cfg.alicat_ports,
                        self.cfg.alicat_baud,
                        self.cfg.alicat_expected_ids or None,
                    ))
            except Exception as e:
                print(f"  WARNING: Failed to set MFC setpoints: {e}")

    def _execute_timeline(self) -> None:
        """Walk the timeline on computer timebase, sending serial commands."""
        print(f"\n{'='*60}")
        print("EXPERIMENT RUNNING")
        print(f"{'='*60}\n")

        self._t0 = time.perf_counter()
        self._log_event("START", f"t0={self._t0:.6f}")
        if self._mfc_monitor is not None:
            self._mfc_monitor.set_t0(self._t0)

        # Group MFC commands that share the same time for batch sending
        pending_mfcs: Dict[str, float] = {}
        last_mfc_time: float = -1.0

        for i, evt in enumerate(self.timeline):
            # Sleep until this event's time
            target_wall = self._t0 + evt.time_ms / 1000.0
            self._sleep_until_event(target_wall)

            self._check_fictrac_health()

            actual_t = time.perf_counter()
            actual_ms = (actual_t - self._t0) * 1000.0
            jitter_ms = actual_ms - evt.time_ms

            # Execute action
            if evt.action == "olfactometer":
                try:
                    self._teensy.set_olfactometer(
                        evt.side, evt.state, wait=False
                    )
                    self._log_event(
                        "OLFACTOMETER",
                        f"{evt.side} -> {evt.state}",
                        scheduled_ms=evt.time_ms,
                        actual_ms=actual_ms,
                        jitter_ms=jitter_ms,
                        phase=evt.phase,
                        repeat=evt.repeat_idx,
                        ev_side=evt.side,
                        ev_state=evt.state,
                    )
                    if self.verbose:
                        print(f"  [{actual_ms/1000:8.2f}s] OLF {evt.side:5s} -> {evt.state}"
                              f"  (jitter {jitter_ms:+.1f} ms)")
                except Exception as e:
                    self._log_event("ERROR", f"olfactometer: {e}")
                    print(f"  ERROR: olfactometer {evt.side} {evt.state}: {e}")

            elif evt.action == "switch_valve":
                try:
                    self._teensy.set_switch_valve(
                        evt.side, evt.state, wait=False
                    )
                    self._log_event(
                        "SWITCH_VALVE",
                        f"{evt.side} -> {evt.state}",
                        scheduled_ms=evt.time_ms,
                        actual_ms=actual_ms,
                        jitter_ms=jitter_ms,
                        phase=evt.phase,
                        repeat=evt.repeat_idx,
                        ev_side=evt.side,
                        ev_state=evt.state,
                    )
                    if self.verbose:
                        print(f"  [{actual_ms/1000:8.2f}s] SV  {evt.side:5s} -> {evt.state}"
                              f"  (jitter {jitter_ms:+.1f} ms)")
                except Exception as e:
                    self._log_event("ERROR", f"switch_valve: {e}")
                    print(f"  ERROR: switch_valve {evt.side} {evt.state}: {e}")

            elif evt.action == "mfc":
                # Batch MFC commands at the same time
                if evt.time_ms != last_mfc_time and pending_mfcs:
                    self._send_mfcs(pending_mfcs)
                    pending_mfcs = {}
                pending_mfcs[evt.device] = evt.value
                last_mfc_time = evt.time_ms
                self._log_event(
                    "MFC",
                    f"{evt.device} -> {evt.value:.3f}",
                    scheduled_ms=evt.time_ms,
                    actual_ms=actual_ms,
                    jitter_ms=jitter_ms,
                    phase=evt.phase,
                    repeat=evt.repeat_idx,
                    ev_device=evt.device,
                    ev_value=evt.value,
                )
                if self.verbose:
                    print(f"  [{actual_ms/1000:8.2f}s] MFC {evt.device} -> {evt.value:.2f}"
                          f"  (jitter {jitter_ms:+.1f} ms)")

            elif evt.action == "log_only":
                self._log_event(
                    "TRIGGER",
                    f"{evt.device} -> {evt.state}",
                    scheduled_ms=evt.time_ms,
                    actual_ms=actual_ms,
                    phase=evt.phase,
                    repeat=evt.repeat_idx,
                    ev_device=evt.device,
                    ev_state=evt.state,
                )
                if self.verbose:
                    print(f"  [{actual_ms/1000:8.2f}s] {evt.device} -> {evt.state}")

            # Print phase changes
            if i + 1 < len(self.timeline) and self.timeline[i+1].phase != evt.phase:
                print(f"\n{'─'*40}")
                print(f"  Entering phase: {self.timeline[i+1].phase}")
                print(f"{'─'*40}")

        # Flush remaining MFC commands
        if pending_mfcs:
            self._send_mfcs(pending_mfcs)

        # Wait for NIDAQ to finish
        elapsed = time.perf_counter() - self._t0
        remaining = (self.total_duration_ms / 1000.0) - elapsed
        if remaining > 0:
            print(f"\n  Timeline complete. Waiting {remaining:.1f} s for NIDAQ task to finish...")
            self._sleep_until_event(time.perf_counter() + remaining + 0.5)

        if self._daq is not None:
            self._daq.wait(timeout_s=10.0)

        total_elapsed = time.perf_counter() - self._t0
        self._log_event("COMPLETE", f"total={total_elapsed:.3f}s")
        print(f"\n{'='*60}")
        print(f"EXPERIMENT COMPLETE ({total_elapsed:.1f} s)")
        print(f"{'='*60}")

    def _send_mfcs(self, mfc_commands: Dict[str, float]) -> None:
        """Send MFC setpoints via Alicat serial (fire-and-forget)."""
        if self.cfg.mfc_mode != "alicat_serial" or not self.cfg.mfc_device_map:
            return
        try:
            if self._mfc_monitor is not None:
                self._mfc_monitor.set_setpoints(mfc_commands)
            else:
                asyncio.run(_set_mfc_setpoints(
                    mfc_commands,
                    self.cfg.mfc_device_map,
                    self.cfg.alicat_ports,
                    self.cfg.alicat_baud,
                    self.cfg.alicat_expected_ids or None,
                ))
        except Exception as e:
            print(f"  WARNING: MFC set failed: {e}")

    def _simulate_timeline(self) -> None:
        """Dry-run: print timeline without any hardware."""
        print("\n  Simulated execution (no hardware):")
        current_phase = ""
        for evt in self.timeline:
            if evt.phase != current_phase:
                current_phase = evt.phase
                print(f"\n  --- {current_phase} ---")
            t_s = evt.time_ms / 1000.0
            if evt.action == "olfactometer":
                print(f"  [{t_s:8.2f}s] -> Teensy: {evt.state} {evt.side}")
            elif evt.action == "switch_valve":
                print(f"  [{t_s:8.2f}s] -> Teensy: {evt.state} SV {evt.side}")
            elif evt.action == "mfc":
                print(f"  [{t_s:8.2f}s] -> Alicat: {evt.device} = {evt.value:.2f}")
            elif evt.action == "log_only":
                print(f"  [{t_s:8.2f}s] -> DAQ:    {evt.device} {evt.state}")

    # ── Shutdown ────────────────────────────────────────────────────────────

    def _shutdown(self) -> None:
        """Gracefully shut down all hardware."""
        print("\nShutting down...")

        # 1. Stop DAQ
        if self._daq is not None:
            print("  Stopping NIDAQ...")
            self._daq.stop()
            self._daq = None

        if self._other_camera_recorder is not None:
            print("  Stopping second Blackfly recorder...")
            try:
                self._other_camera_recording = self._other_camera_recorder.stop()
            except Exception as e:
                print(f"    WARNING: Blackfly recorder shutdown failed: {e}")
                self._log_event("ERROR", f"blackfly_recorder: {e}")
            finally:
                self._other_camera_recorder = None

        # 2. Reset Teensy (all valves off)
        if self._teensy is not None and self._teensy.is_open:
            print("  Resetting Teensy (all valves off)...")
            try:
                self._teensy.reset()
            except Exception as e:
                print(f"    WARNING: Teensy reset failed: {e}")
            if self._teensy_serial_monitor is not None:
                self._teensy_transcript = self._teensy_serial_monitor.get_transcript()
            self._teensy.close()
            self._teensy = None
            self._teensy_serial_monitor = None

        # 3. Stop FicTrac
        if self._fictrac_callback is not None:
            print("  Stopping FicTrac...")
            self._fictrac_callback.request_stop()
        if self._fictrac_thread is not None:
            self._fictrac_thread.join(timeout=10.0)
            if self._fictrac_thread.is_alive():
                print("    WARNING: FicTrac thread did not exit cleanly")

        # 4. Zero MFCs then stop the monitor
        if self._mfc_monitor is not None:
            print("  Zeroing MFCs...")
            self._mfc_monitor.zero_all()
            self._mfc_monitor.stop()
            self._mfc_monitor = None
        elif self.cfg.mfc_mode == "alicat_serial" and self.cfg.mfc_device_map:
            # Fallback: monitor was never created (e.g. no device map at launch)
            print("  Zeroing MFCs...")
            zero_cmds = {dev: 0.0 for dev in self.cfg.mfc_device_map}
            try:
                asyncio.run(_set_mfc_setpoints(
                    zero_cmds,
                    self.cfg.mfc_device_map,
                    self.cfg.alicat_ports,
                    self.cfg.alicat_baud,
                    self.cfg.alicat_expected_ids or None,
                ))
            except Exception as e:
                print(f"    WARNING: MFC zeroing failed: {e}")

        print("  Shutdown complete.")

    # ── Data saving ─────────────────────────────────────────────────────────

    def _log_event(self, event_type: str, detail: str, **kwargs) -> None:
        entry = {
            "wall_time": time.perf_counter(),
            "type": event_type,
            "detail": detail,
        }
        entry.update(kwargs)
        self._event_log.append(entry)

    def _save_data(self) -> None:
        """Save all experiment data to the run directory."""
        if self._run_dir is None:
            return

        print(f"\nSaving data to {self._run_dir}...")

        # 1. Event log (JSON — full fidelity)
        with open(self._run_dir / "event_log.json", "w") as f:
            json.dump(self._event_log, f, indent=2, default=str)

        # 2. Event log (CSV — easy to inspect)
        with open(self._run_dir / "event_log.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "wall_time", "experiment_time_s", "type", "detail",
                "scheduled_ms", "actual_ms", "jitter_ms",
                "phase", "repeat", "ev_side", "ev_state", "ev_device", "ev_value",
            ])
            for e in self._event_log:
                t0 = self._t0 if self._t0 else 0.0
                exp_t = e["wall_time"] - t0
                writer.writerow([
                    f"{e['wall_time']:.6f}",
                    f"{exp_t:.6f}",
                    e.get("type", ""),
                    e.get("detail", ""),
                    e.get("scheduled_ms", ""),
                    e.get("actual_ms", ""),
                    e.get("jitter_ms", ""),
                    e.get("phase", ""),
                    e.get("repeat", ""),
                    e.get("ev_side", ""),
                    e.get("ev_state", ""),
                    e.get("ev_device", ""),
                    e.get("ev_value", ""),
                ])

        # 2b. Raw Teensy serial transcript for firmware-level auditability.
        if self._teensy_transcript:
            with open(self._run_dir / "teensy_serial_transcript.jsonl", "w", encoding="utf-8") as f:
                for entry in self._teensy_transcript:
                    json.dump(entry, f, default=str)
                    f.write("\n")

        # 3. DAQ trigger waveform (compressed)
        if self.trigger_waveform is not None:
            np.savez_compressed(
                self._run_dir / "trigger_waveform.npz",
                waveform=self.trigger_waveform,
                line_names=np.array([
                    "GLOBAL_LOAD_REQ",
                    "RCK_OLFACTOMETER_LEFT", "RCK_SWITCHVALVE_LEFT",
                    "RCK_OLFACTOMETER_RIGHT", "RCK_SWITCHVALVE_RIGHT",
                    "TRIG_CAMERA", "TRIG_MICRO",
                ]),
            )

        # 4. Timeline (compiled protocol schedule — for reference)
        write_control_plan_csv(self._run_dir / "timeline.csv", self.timeline)

        # 5. Raw FicTrac frames (compact numeric store)
        if self._fictrac_callback is not None:
            self._fictrac_callback.save_npz(self._run_dir / "fictrac_frames.npz")

        # 6. Merged experiment CSV — primary analysis file
        n_frames = self._save_merged_csv()

        expected_camera_frames = None
        if self.trigger_waveform is not None:
            expected_camera_frames = _count_rising_edges(self.trigger_waveform[5])

        if self._fictrac_runtime_info.get("save_raw"):
            fictrac_recording = _build_fictrac_recording_summary(
                run_dir=self._run_dir,
                runtime_info=self._fictrac_runtime_info,
                frame_count=self._fictrac_callback.frame_count if self._fictrac_callback is not None else None,
                expected_frame_count=expected_camera_frames if self.cfg.verify_camera_recording else None,
            )
            with open(self._run_dir / "fictrac_camera_recording.json", "w", encoding="utf-8") as f:
                json.dump(fictrac_recording, f, indent=2)
            self._fictrac_runtime_info["recording_summary"] = fictrac_recording

        if self._other_camera_recording:
            from multibios.blackfly.triggered_camera_record import postprocess_triggered_camera_recording

            self._other_camera_recording = postprocess_triggered_camera_recording(
                self._other_camera_recording,
                expected_frame_count=expected_camera_frames if self.cfg.verify_camera_recording else None,
                nominal_fps=(1000.0 / self.trigger_cfg.camera_interval_ms)
                if self.trigger_cfg is not None and self.trigger_cfg.camera_interval_ms > 0
                else None,
                convert_to_lossless_mkv=self.cfg.convert_second_camera_bin_to_lossless_mkv,
            )
            with open(self._run_dir / "blackfly_recording.json", "w", encoding="utf-8") as f:
                json.dump(self._other_camera_recording, f, indent=2)

        print(f"  FicTrac frames: {n_frames}")
        if self._fictrac_runtime_info.get("recording_summary"):
            fictrac_recording = self._fictrac_runtime_info["recording_summary"]
            print(
                "  FicTrac camera raw frames: "
                f"{fictrac_recording.get('actual_frames')} "
                f"(expected: {fictrac_recording.get('expected_frames')}, "
                f"no_drop={fictrac_recording.get('no_dropped_frames')})"
            )
        if self._other_camera_recording:
            print(
                "  Second camera raw frames: "
                f"{self._other_camera_recording.get('saved_frames', 0)} "
                f"(expected: {self._other_camera_recording.get('expected_frame_count')}, "
                f"no_drop={self._other_camera_recording.get('no_dropped_frames')}, "
                f"manifest: {self._other_camera_recording.get('manifest_path')})"
            )
        print(f"  Data saved to:  {self._run_dir}")

        if self.cfg.open_explorer and not self.dry_run:
            self._launch_explorer()

    def _launch_explorer(self) -> None:
        """Start the explorer server (if not already running) and open the browser."""
        import socket
        import subprocess
        import threading
        import webbrowser

        port = self.cfg.explorer_port
        run_dir = str(self._run_dir.resolve()) if self._run_dir else ""

        # Check if explorer is already listening
        already_running = False
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                already_running = True
        except OSError:
            pass

        if not already_running:
            subprocess.Popen(
                [sys.executable, "-m", "multibios.apps.explorer", "--no-browser", "--port", str(port)],
                cwd=str(Path(__file__).parent.parent),
                creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
                | getattr(subprocess, "DETACHED_PROCESS", 0),
                close_fds=True,
            )
            print(f"  Explorer: started on http://127.0.0.1:{port}")
        else:
            print(f"  Explorer: already running on http://127.0.0.1:{port}")

        # Open browser after a brief pause for server startup
        url = f"http://127.0.0.1:{port}"
        def _open():
            import time
            time.sleep(2.0 if not already_running else 0.3)
            webbrowser.open(url)
        threading.Thread(target=_open, daemon=True).start()

    def _save_merged_csv(self) -> int:
        """Build the primary analysis CSV: one row per FicTrac frame with all
        forward-filled output states and hardware signals merged in.

        Columns
        -------
        experiment_time_s          wall_time − t0
        wall_time                  time.perf_counter() value
        frame_cnt                  FicTrac frame number
        del_rot_cam_*              camera-frame delta rotation vector
        del_rot_error              FicTrac delta-rotation fit error
        del_rot_lab_*              lab-frame delta rotation vector
        abs_ori_cam_*              camera-frame absolute orientation vector
        abs_ori_lab_*              lab-frame absolute orientation vector
        posx / posy                integrated ball position (ball radii)
        heading                    heading angle (radians)
        speed                      ball speed (ball radii / s)
        direction                  movement direction (radians)
        intx / inty                integrated X/Y (alternate FicTrac units)
        fictrac_timestamp          FicTrac internal frame timestamp
        seq_num                    FicTrac UDP message sequence number
        delta_timestamp            time since last FicTrac frame
        alt_timestamp              optional FicTrac v2.1.1 timestamp field
        olfactometer_left/right    current odor valve state (AIR/ODOR1-5/OFF…)
        switch_valve_left/right    current switch valve state (CLEAN/ODOR)
        mfc_air_left/right_sp      MFC setpoint commanded
        mfc_odor_left/right_sp     MFC setpoint commanded
        camera_trigger             1 if TRIG_CAMERA is HIGH in the DAQ waveform
        microscope_trigger         1 if TRIG_MICRO is HIGH in the DAQ waveform
        phase                      current protocol phase name
        repeat_idx                 repeat index within phase

        Returns the number of FicTrac frames written.
        """
        if self._fictrac_callback is None:
            return 0
        frames = self._fictrac_callback.frame_array()
        if len(frames) == 0:
            return 0

        t0 = self._t0 if self._t0 else float(frames[0]["wall_time"])
        sr = self.trigger_cfg.sample_rate if self.trigger_cfg else 1000
        # Waveform row indices (matches TRIGGER_LINE_NAMES order in daq_triggers.py)
        WF_CAMERA = 5
        WF_MICRO  = 6
        waveform = self.trigger_waveform   # shape (7, N) or None
        wf_N = waveform.shape[1] if waveform is not None else 0

        def wf_val(row: int, t_s: float) -> int:
            if waveform is None or wf_N == 0:
                return 0
            idx = int(t_s * sr)
            if idx < 0 or idx >= wf_N:
                return 0
            return int(waveform[row, idx])

        # ─── Build state-change event stream ─────────────────────────────────
        # Collect all events that change a tracked state, sorted by wall_time.
        # We use actual wall_time (when the serial command was sent) so the
        # forward-fill reflects real hardware latency, not scheduled times.
        state_events: List[Dict[str, Any]] = [
            e for e in self._event_log
            if e.get("type") in ("OLFACTOMETER", "SWITCH_VALVE", "MFC", "TRIGGER", "START")
        ]
        state_events.sort(key=lambda e: e["wall_time"])

        # ─── Initial state (all defaults) ─────────────────────────────────────
        state: Dict[str, Any] = {
            "olfactometer_left":    "OFF",
            "olfactometer_right":   "OFF",
            "switch_valve_left":    "CLEAN",
            "switch_valve_right":   "CLEAN",
            "mfc_air_left_sp":      0.0,
            "mfc_air_right_sp":     0.0,
            "mfc_odor_left_sp":     0.0,
            "mfc_odor_right_sp":    0.0,
            "phase":                "",
            "repeat_idx":           0,
        }

        # MFC device-key -> state field
        MFC_KEY_MAP = {
            "mfc.air_left_setpoint":    "mfc_air_left_sp",
            "mfc.air_right_setpoint":   "mfc_air_right_sp",
            "mfc.odor_left_setpoint":   "mfc_odor_left_sp",
            "mfc.odor_right_setpoint":  "mfc_odor_right_sp",
        }

        def apply_event(e: Dict[str, Any]) -> None:
            etype = e.get("type", "")
            if etype == "OLFACTOMETER":
                side = e.get("ev_side", "")
                st   = e.get("ev_state", "")
                if side == "left":  state["olfactometer_left"]  = st
                elif side == "right": state["olfactometer_right"] = st
            elif etype == "SWITCH_VALVE":
                side = e.get("ev_side", "")
                st   = e.get("ev_state", "")
                if side == "left":  state["switch_valve_left"]  = st
                elif side == "right": state["switch_valve_right"] = st
            elif etype == "MFC":
                dev = e.get("ev_device", "")
                val = e.get("ev_value", None)
                if val is not None and dev in MFC_KEY_MAP:
                    state[MFC_KEY_MAP[dev]] = float(val)
            # Update protocol state from any labeled event
            if "phase" in e and e["phase"]:
                state["phase"] = e["phase"]
            if "repeat" in e and e["repeat"] is not None:
                state["repeat_idx"] = int(e["repeat"])

        # ─── Write CSV ────────────────────────────────────────────────────────
        ev_ptr = 0  # index into state_events
        n_events = len(state_events)

        out_path = self._run_dir / "experiment_data.csv"
        with open(out_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "experiment_time_s", "wall_time", "frame_cnt",
                # FicTrac tracking
                "del_rot_cam_x", "del_rot_cam_y", "del_rot_cam_z", "del_rot_error",
                "del_rot_lab_x", "del_rot_lab_y", "del_rot_lab_z",
                "abs_ori_cam_x", "abs_ori_cam_y", "abs_ori_cam_z",
                "abs_ori_lab_x", "abs_ori_lab_y", "abs_ori_lab_z",
                "posx", "posy", "heading", "speed", "direction", "intx", "inty",
                "fictrac_timestamp", "seq_num", "delta_timestamp", "alt_timestamp",
                # Odor valve states
                "olfactometer_left", "olfactometer_right",
                "switch_valve_left", "switch_valve_right",
                # MFC setpoints (commanded)
                "mfc_air_left_sp", "mfc_air_right_sp",
                "mfc_odor_left_sp", "mfc_odor_right_sp",
                # NIDAQ hardware signals (from pre-computed waveform)
                "camera_trigger", "microscope_trigger",
                # Protocol state
                "phase", "repeat_idx",
            ])

            for fr in frames:
                fr_wall_time = float(fr["wall_time"])
                exp_t = fr_wall_time - t0

                # Advance event pointer: apply all events up to this frame
                while ev_ptr < n_events and state_events[ev_ptr]["wall_time"] <= fr_wall_time:
                    apply_event(state_events[ev_ptr])
                    ev_ptr += 1

                writer.writerow([
                    f"{exp_t:.6f}",
                    f"{fr_wall_time:.6f}",
                    int(fr["frame_cnt"]),
                    f"{float(fr['del_rot_cam_x']):.8f}",
                    f"{float(fr['del_rot_cam_y']):.8f}",
                    f"{float(fr['del_rot_cam_z']):.8f}",
                    f"{float(fr['del_rot_error']):.8f}",
                    f"{float(fr['del_rot_lab_x']):.8f}",
                    f"{float(fr['del_rot_lab_y']):.8f}",
                    f"{float(fr['del_rot_lab_z']):.8f}",
                    f"{float(fr['abs_ori_cam_x']):.8f}",
                    f"{float(fr['abs_ori_cam_y']):.8f}",
                    f"{float(fr['abs_ori_cam_z']):.8f}",
                    f"{float(fr['abs_ori_lab_x']):.8f}",
                    f"{float(fr['abs_ori_lab_y']):.8f}",
                    f"{float(fr['abs_ori_lab_z']):.8f}",
                    f"{float(fr['posx']):.8f}",
                    f"{float(fr['posy']):.8f}",
                    f"{float(fr['heading']):.8f}",
                    f"{float(fr['speed']):.8f}",
                    f"{float(fr['direction']):.8f}",
                    f"{float(fr['intx']):.8f}",
                    f"{float(fr['inty']):.8f}",
                    f"{float(fr['timestamp']):.8f}",
                    int(fr["seq_num"]),
                    f"{float(fr['delta_timestamp']):.8f}",
                    f"{float(fr['alt_timestamp']):.8f}",
                    state["olfactometer_left"],
                    state["olfactometer_right"],
                    state["switch_valve_left"],
                    state["switch_valve_right"],
                    f"{state['mfc_air_left_sp']:.4f}",
                    f"{state['mfc_air_right_sp']:.4f}",
                    f"{state['mfc_odor_left_sp']:.4f}",
                    f"{state['mfc_odor_right_sp']:.4f}",
                    wf_val(WF_CAMERA, exp_t),
                    wf_val(WF_MICRO,  exp_t),
                    state["phase"],
                    state["repeat_idx"],
                ])

        return int(len(frames))

    def _save_metadata(self) -> None:
        """Save metadata even for dry runs."""
        if self._run_dir is None:
            return

        self._run_dir.mkdir(parents=True, exist_ok=True)

        # Copy protocol and hardware config
        shutil.copy2(self.protocol_path, self._run_dir / "protocol.yaml")
        shutil.copy2(self.hardware_path, self._run_dir / "hardware.yaml")

        meta = {
            "timestamp": datetime.now().isoformat(),
            "protocol": str(self.protocol_path),
            "hardware": str(self.hardware_path),
            "total_duration_ms": self.total_duration_ms,
            "seed": self.seed,
            "dry_run": self.dry_run,
            "trigger_config": asdict(self.trigger_cfg) if self.trigger_cfg else {},
            "experiment_config": asdict(self.cfg),
            "fictrac_runtime": self._fictrac_runtime_info,
            "other_camera_recording": self._other_camera_recording,
            "n_timeline_events": len(self.timeline),
            "n_microscope_triggers": len(self.microscope_times),
            "n_camera_windows": len(self.camera_windows),
        }
        with open(self._run_dir / "meta.json", "w") as f:
            json.dump(meta, f, indent=2, default=str)


# ═══════════════════════════════════════════════════════════════════════════
# CLI entry point
# ═══════════════════════════════════════════════════════════════════════════

def load_experiment_config(
    path: str | Path | None,
    hardware_path: str | Path | None = None,
) -> ExperimentConfig:
    """Load experiment_config.yaml into an ExperimentConfig."""
    raw = _load_yaml_file(path)
    hardware = _load_yaml_file(hardware_path)
    hardware_teensy = _yaml_section(hardware, "teensy")
    hardware_fictrac = _yaml_section(hardware, "fictrac")
    hardware_blackfly = _yaml_section(hardware, "blackfly_defaults")
    hardware_camera_recording = _yaml_section(hardware, "camera_recording")
    hardware_mfc = _yaml_section(hardware, "mfc")
    hardware_daq = _yaml_section(hardware, "daq")
    hardware_data_output = _yaml_section(hardware, "data_output")

    cfg = ExperimentConfig()
    cfg.teensy_port = str(hardware_teensy.get("port", cfg.teensy_port))
    if "teensy_port" in raw:
        _warn_deprecated_experiment_key("teensy_port", hardware_path, "teensy")
        cfg.teensy_port = str(raw["teensy_port"])

    cfg.teensy_baud = int(hardware_teensy.get("baud", cfg.teensy_baud))
    if "teensy_baud" in raw:
        _warn_deprecated_experiment_key("teensy_baud", hardware_path, "teensy")
        cfg.teensy_baud = int(raw["teensy_baud"])

    cfg.fictrac_config = str(hardware_fictrac.get("config", cfg.fictrac_config))
    if "fictrac_config" in raw:
        _warn_deprecated_experiment_key("fictrac_config", hardware_path, "fictrac")
        cfg.fictrac_config = str(raw["fictrac_config"])

    cfg.fictrac_bin = str(hardware_fictrac.get("bin", cfg.fictrac_bin))
    if "fictrac_bin" in raw:
        _warn_deprecated_experiment_key("fictrac_bin", hardware_path, "fictrac")
        cfg.fictrac_bin = str(raw["fictrac_bin"])

    cfg.fictrac_console_out = str(hardware_fictrac.get("console_out", cfg.fictrac_console_out))
    if "fictrac_console_out" in raw:
        _warn_deprecated_experiment_key("fictrac_console_out", hardware_path, "fictrac")
        cfg.fictrac_console_out = str(raw["fictrac_console_out"])

    cfg.fictrac_first_frame_timeout_ms = int(
        hardware_fictrac.get("first_frame_timeout_ms", cfg.fictrac_first_frame_timeout_ms)
    )
    if "fictrac_first_frame_timeout_ms" in raw:
        _warn_deprecated_experiment_key("fictrac_first_frame_timeout_ms", hardware_path, "fictrac")
        cfg.fictrac_first_frame_timeout_ms = int(raw["fictrac_first_frame_timeout_ms"])

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

    second_camera_index = hardware_camera_recording.get("second_camera_index", cfg.second_camera_index)
    cfg.second_camera_index = None if second_camera_index is None else int(second_camera_index)
    if "second_camera_index" in raw:
        _warn_deprecated_experiment_key("second_camera_index", hardware_path, "camera_recording")
        cfg.second_camera_index = int(raw["second_camera_index"])

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

    second_camera_exposure = hardware_camera_recording.get(
        "second_camera_exposure_us",
        hardware_blackfly.get("exposure_us", cfg.second_camera_exposure_us),
    )
    cfg.second_camera_exposure_us = None if second_camera_exposure is None else float(second_camera_exposure)
    if "second_camera_exposure_us" in raw:
        _warn_deprecated_experiment_key("second_camera_exposure_us", hardware_path, "camera_recording")
        cfg.second_camera_exposure_us = float(raw["second_camera_exposure_us"])
    elif "other_camera_exposure_us" in raw:
        _warn_deprecated_experiment_key("other_camera_exposure_us", hardware_path, "camera_recording")
        cfg.second_camera_exposure_us = float(raw["other_camera_exposure_us"])

    second_camera_roi_width = hardware_camera_recording.get(
        "second_camera_roi_width",
        hardware_blackfly.get("roi_width", cfg.second_camera_roi_width),
    )
    cfg.second_camera_roi_width = None if second_camera_roi_width is None else int(second_camera_roi_width)
    if "second_camera_roi_width" in raw:
        _warn_deprecated_experiment_key("second_camera_roi_width", hardware_path, "camera_recording")
        cfg.second_camera_roi_width = int(raw["second_camera_roi_width"])
    elif "other_camera_roi_width" in raw:
        _warn_deprecated_experiment_key("other_camera_roi_width", hardware_path, "camera_recording")
        cfg.second_camera_roi_width = int(raw["other_camera_roi_width"])

    second_camera_roi_height = hardware_camera_recording.get(
        "second_camera_roi_height",
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
            hardware_blackfly.get("binning", cfg.second_camera_binning),
        )
    )
    if "second_camera_binning" in raw:
        _warn_deprecated_experiment_key("second_camera_binning", hardware_path, "camera_recording")
        cfg.second_camera_binning = int(raw["second_camera_binning"])
    elif "other_camera_binning" in raw:
        _warn_deprecated_experiment_key("other_camera_binning", hardware_path, "camera_recording")
        cfg.second_camera_binning = int(raw["other_camera_binning"])

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

    cfg.mfc_mode = str(hardware_mfc.get("mode", cfg.mfc_mode))
    if "mfc_mode" in raw:
        _warn_deprecated_experiment_key("mfc_mode", hardware_path, "mfc")
        cfg.mfc_mode = str(raw["mfc_mode"])

    cfg.mfc_device_map = dict(hardware_mfc.get("device_map", cfg.mfc_device_map))
    if "mfc_device_map" in raw:
        _warn_deprecated_experiment_key("mfc_device_map", hardware_path, "mfc")
        cfg.mfc_device_map = raw["mfc_device_map"]

    cfg.alicat_ports = list(hardware_mfc.get("alicat_ports", cfg.alicat_ports))
    if "alicat_ports" in raw:
        _warn_deprecated_experiment_key("alicat_ports", hardware_path, "mfc")
        cfg.alicat_ports = raw["alicat_ports"]

    cfg.alicat_baud = list(hardware_mfc.get("alicat_baud", cfg.alicat_baud))
    if "alicat_baud" in raw:
        _warn_deprecated_experiment_key("alicat_baud", hardware_path, "mfc")
        cfg.alicat_baud = raw["alicat_baud"]

    cfg.alicat_expected_ids = list(hardware_mfc.get("alicat_expected_ids", cfg.alicat_expected_ids))
    if "alicat_expected_ids" in raw:
        _warn_deprecated_experiment_key("alicat_expected_ids", hardware_path, "mfc")
        cfg.alicat_expected_ids = raw["alicat_expected_ids"]

    cfg.latch_interval_ms = float(hardware_daq.get("latch_interval_ms", cfg.latch_interval_ms))
    if "latch_interval_ms" in raw:
        _warn_deprecated_experiment_key("latch_interval_ms", hardware_path, "daq")
        cfg.latch_interval_ms = float(raw["latch_interval_ms"])

    cfg.data_dir = str(hardware_data_output.get("data_dir", cfg.data_dir))
    if "data_dir" in raw:
        _warn_deprecated_experiment_key("data_dir", hardware_path, "data_output")
        cfg.data_dir = str(raw["data_dir"])

    cfg.open_explorer = bool(hardware_data_output.get("open_explorer", cfg.open_explorer))
    if "open_explorer" in raw:
        _warn_deprecated_experiment_key("open_explorer", hardware_path, "data_output")
        cfg.open_explorer = bool(raw["open_explorer"])

    cfg.explorer_port = int(hardware_data_output.get("explorer_port", cfg.explorer_port))
    if "explorer_port" in raw:
        _warn_deprecated_experiment_key("explorer_port", hardware_path, "data_output")
        cfg.explorer_port = int(raw["explorer_port"])

    cfg.mfc_live_interval_s = float(hardware_mfc.get("live_interval_s", cfg.mfc_live_interval_s))
    if "mfc_live_interval_s" in raw:
        _warn_deprecated_experiment_key("mfc_live_interval_s", hardware_path, "mfc")
        cfg.mfc_live_interval_s = float(raw["mfc_live_interval_s"])
    return cfg


def main():
    parser = argparse.ArgumentParser(
        description="Open-loop experiment runner (computer-timebase + serial)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Dry run (preview timeline, no hardware)
  python -m multibios.experiment --protocol config/example_protocol.yaml --dry-run

  # Full run
  python -m multibios.experiment \\
      --protocol config/example_protocol.yaml \\
      --hardware config/hardware.yaml \\
      --experiment config/experiment_config.yaml
        """,
    )
    parser.add_argument("--protocol", default="config/example_protocol.yaml",
                        help="Protocol YAML file")
    parser.add_argument("--hardware", default="config/hardware.yaml",
                        help="Hardware mapping YAML")
    parser.add_argument("--experiment", default="config/experiment_config.yaml",
                        help="Experiment configuration YAML")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed (overrides protocol YAML)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Preview timeline without hardware")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Print detailed execution log")
    parser.add_argument("--teensy-port", default=None,
                        help="Override Teensy COM port (e.g. COM4)")

    args = parser.parse_args()

    # Load configs
    exp_cfg_path = Path(args.experiment)
    if not exp_cfg_path.exists() and not args.dry_run:
        print(f"WARNING: Experiment config '{args.experiment}' not found, using defaults")
    exp_cfg = load_experiment_config(
        exp_cfg_path if exp_cfg_path.exists() else None,
        hardware_path=args.hardware,
    )

    # Apply CLI overrides
    if args.teensy_port is not None:
        exp_cfg.teensy_port = args.teensy_port

    # Create and run experiment
    runner = ExperimentRunner(
        protocol_path=args.protocol,
        hardware_path=args.hardware,
        experiment_cfg=exp_cfg,
        seed=args.seed,
        dry_run=args.dry_run,
        verbose=args.verbose,
    )

    try:
        runner.prepare()
        runner.preview()
        runner.run()
    except Exception as e:
        # Errors inside run() are already printed; errors in prepare() may not be.
        import traceback
        print(f"\nFATAL: {e}", file=sys.stderr)
        if args.verbose:
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
