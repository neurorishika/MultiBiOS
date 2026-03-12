#!/usr/bin/env python3
"""
experiment.py — Open-loop experiment runner (computer-timebase + serial).

Parses the same YAML protocol format used by the NIDAQ-compiled approach,
but executes it via:

  * **Teensy v2 serial** for odor valve control
  * **AlicatManager serial** for MFC setpoints
  * **NIDAQ finite DO task** for camera/microscope triggers & latch pulses
  * **pybmt / FicTrac** for ball tracking (recorded throughout)

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
The ``ExperimentCallback`` (pybmt callback) exposes the latest FicTrac state
via a thread-safe property.  A future closed-loop engine can read this in
the timeline loop and branch protocol logic on fly behavior.

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
import os
import sys
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple

import numpy as np
import yaml
# pybmt
from pybmt.callback.base import PyBMTCallback
from pybmt.fictrac.driver import FicTracDriver

# MultiBiOS imports
from multibios.daq_triggers import (DAQTriggerManager, TriggerConfig,
                                    build_trigger_waveform)
from multibios.protocol.schema import (BIG_STATE_CODE, SMALL_STATE_CODE,
                                       CompileError)
from multibios.teensy_controller import TeensyController

# ═══════════════════════════════════════════════════════════════════════════
# Data structures
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class TimelineEvent:
    """A single scheduled action in the experiment."""
    time_ms: float
    action: str              # "olfactometer" | "switch_valve" | "mfc" | "log_only"
    device: str              # original device key from YAML
    side: str = ""           # "left" | "right" | ""
    state: str = ""          # e.g. "AIR", "ODOR3", "CLEAN"
    value: float = 0.0       # for MFC setpoints
    phase: str = ""          # which protocol phase this belongs to
    repeat_idx: int = 0      # repeat index within that phase


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
    fictrac_timeout_s: float = 5.0

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


@dataclass
class FicTracFrame:
    """One logged FicTrac frame."""
    wall_time: float          # time.perf_counter() value
    frame_cnt: int
    posx: float
    posy: float
    heading: float
    speed: float
    direction: float
    intx: float
    inty: float
    timestamp: float          # FicTrac internal timestamp
    delta_timestamp: float


# ═══════════════════════════════════════════════════════════════════════════
# FicTrac callback — records every frame, exposes latest state
# ═══════════════════════════════════════════════════════════════════════════

class ExperimentCallback(PyBMTCallback):
    """pybmt callback that logs every FicTrac frame and exposes the latest.

    The experiment runner reads ``latest`` from the main thread; the pybmt
    driver calls ``process_callback`` from its own thread.  Thread safety is
    ensured by the GIL for simple attribute reads and the explicit lock for
    the log list.
    """

    def __init__(self) -> None:
        self.latest: Optional[FicTracFrame] = None
        self._frames: List[FicTracFrame] = []
        self._lock = threading.Lock()
        self._stop = threading.Event()

    def setup_callback(self):
        pass

    def process_callback(self, track_state) -> bool:
        now = time.perf_counter()
        frame = FicTracFrame(
            wall_time=now,
            frame_cnt=track_state.frame_cnt,
            posx=track_state.posx,
            posy=track_state.posy,
            heading=track_state.heading,
            speed=track_state.speed,
            direction=track_state.direction,
            intx=track_state.intx,
            inty=track_state.inty,
            timestamp=track_state.timestamp,
            delta_timestamp=track_state.delta_timestamp,
        )
        self.latest = frame
        with self._lock:
            self._frames.append(frame)
        # Return True to keep running, False when stop requested
        return not self._stop.is_set()

    def shutdown_callback(self):
        pass

    def request_stop(self) -> None:
        self._stop.set()

    @property
    def frames(self) -> List[FicTracFrame]:
        with self._lock:
            return list(self._frames)


# ═══════════════════════════════════════════════════════════════════════════
# Protocol -> timeline compiler
# ═══════════════════════════════════════════════════════════════════════════

def _norm_dev(s: str) -> str:
    return s.strip().lower()


def compile_timeline(
    protocol_yaml: Dict[str, Any],
    seed: Optional[int] = None,
) -> Tuple[List[TimelineEvent], List[float], List[Tuple[float, float]], float]:
    """Parse a protocol YAML and produce a flat timeline of events.

    Returns
    -------
    timeline : list of TimelineEvent
        Sorted by ``time_ms``.
    microscope_times_ms : list of float
        Absolute times for microscope trigger pulses (for the DAQ waveform).
    camera_windows : list of (start_ms, end_ms)
        Windows where camera triggers should be active.
    total_duration_ms : float
        Total experiment duration.
    """
    p = protocol_yaml.get("protocol", {})
    timing = p.get("timing", {})
    seq = protocol_yaml.get("sequence", [])

    # RNG
    if seed is None:
        seed_val = timing.get("seed", None)
        if seed_val is not None:
            seed = int(seed_val)
        else:
            seed = int(np.random.SeedSequence().entropy)
    rng = np.random.default_rng(seed)

    timeline: List[TimelineEvent] = []
    microscope_times: List[float] = []
    camera_windows: List[Tuple[float, float]] = []
    camera_on_at: Optional[float] = None  # tracks camera enable start

    # Expand phases
    expanded = []
    total_ms = 0.0
    for entry in seq:
        name = entry.get("phase", "PHASE")
        dur = int(entry.get("duration", 0))
        if "times" in entry:
            times = int(entry["times"])
        elif "repeat" in entry:
            times = int(entry["repeat"]) + 1
        else:
            times = 1
        if times <= 0:
            raise CompileError(f"Phase '{name}': times must be positive")
        total_ms += dur * times
        expanded.append((name, dur, entry, times))

    # Walk phases
    t_cursor = 0.0
    for name, duration, entry, times in expanded:
        randomize = bool(entry.get("randomize", False))
        actions = entry.get("actions", [])

        # Collect olfactometer state lists
        left_spec = None
        right_spec = None
        for a in actions:
            dev = _norm_dev(a.get("device", ""))
            if dev == "olfactometer.left":
                left_spec = a.get("state", "OFF")
            elif dev == "olfactometer.right":
                right_spec = a.get("state", "OFF")

        left_list = _parse_state_list(left_spec, times)
        right_list = _parse_state_list(right_spec, times)

        # Permutation
        perm = np.arange(times)
        if randomize:
            perm = rng.permutation(times)

        if len(left_list) == times:
            left_list = [left_list[i] for i in perm]
        else:
            left_list = left_list * times

        if len(right_list) == times:
            right_list = [right_list[i] for i in perm]
        else:
            right_list = right_list * times

        # Resolve "|" random choices
        resolved_left = [_resolve_choice(tok, rng) for tok in left_list]
        resolved_right = [_resolve_choice(tok, rng) for tok in right_list]

        # Camera enables (process once, not per repeat)
        for a in actions:
            dev = _norm_dev(a.get("device", ""))
            timing_ms = float(a.get("timing", 0))
            if dev == "triggers.camera_continuous":
                enabled = bool(a.get("state", False))
                abs_t = t_cursor + timing_ms
                if enabled:
                    camera_on_at = abs_t
                else:
                    if camera_on_at is not None:
                        camera_windows.append((camera_on_at, abs_t))
                        camera_on_at = None

        # Per-repeat events
        for rep_idx in range(times):
            t0 = t_cursor + rep_idx * duration
            for a in actions:
                dev = _norm_dev(a.get("device", ""))
                timing_ms = float(a.get("timing", 0))
                t_abs = t0 + timing_ms

                if dev.startswith("mfc."):
                    val = float(a.get("value", a.get("state", 0.0)))
                    timeline.append(TimelineEvent(
                        time_ms=t_abs, action="mfc", device=dev,
                        value=val, phase=name, repeat_idx=rep_idx,
                    ))

                elif dev == "olfactometer.left":
                    timeline.append(TimelineEvent(
                        time_ms=t_abs, action="olfactometer", device=dev,
                        side="left", state=resolved_left[rep_idx],
                        phase=name, repeat_idx=rep_idx,
                    ))

                elif dev == "olfactometer.right":
                    timeline.append(TimelineEvent(
                        time_ms=t_abs, action="olfactometer", device=dev,
                        side="right", state=resolved_right[rep_idx],
                        phase=name, repeat_idx=rep_idx,
                    ))

                elif dev == "switch_valve.left":
                    st = str(a.get("state", a.get("value", "CLEAN"))).strip().upper()
                    timeline.append(TimelineEvent(
                        time_ms=t_abs, action="switch_valve", device=dev,
                        side="left", state=st,
                        phase=name, repeat_idx=rep_idx,
                    ))

                elif dev == "switch_valve.right":
                    st = str(a.get("state", a.get("value", "CLEAN"))).strip().upper()
                    timeline.append(TimelineEvent(
                        time_ms=t_abs, action="switch_valve", device=dev,
                        side="right", state=st,
                        phase=name, repeat_idx=rep_idx,
                    ))

                elif dev == "triggers.microscope":
                    if bool(a.get("state", True)):
                        microscope_times.append(t_abs)
                        timeline.append(TimelineEvent(
                            time_ms=t_abs, action="log_only", device=dev,
                            state="PULSE", phase=name, repeat_idx=rep_idx,
                        ))

                elif dev in ("triggers.camera", "triggers.camera_continuous"):
                    # Camera is handled by DAQ waveform; log the enable/disable
                    timeline.append(TimelineEvent(
                        time_ms=t_abs if rep_idx == 0 else -1,  # log once
                        action="log_only", device=dev,
                        state=str(a.get("state", "")),
                        phase=name, repeat_idx=rep_idx,
                    ))

        t_cursor += duration * times

    # Close open camera window
    if camera_on_at is not None:
        camera_windows.append((camera_on_at, total_ms))

    # Sort timeline
    timeline = [e for e in timeline if e.time_ms >= 0]
    timeline.sort(key=lambda e: e.time_ms)

    return timeline, microscope_times, camera_windows, total_ms


def _parse_state_list(spec, times: int) -> List[str]:
    if spec is None:
        return ["OFF"]
    if isinstance(spec, list):
        toks = [str(x).strip().upper() for x in spec]
    else:
        toks = [p.strip().upper() for p in str(spec).split(",") if p.strip()]
    if not toks:
        toks = ["OFF"]
    if len(toks) not in (1, times):
        # If mismatch, just use single-element broadcast
        toks = [toks[0]]
    return toks


def _resolve_choice(tok: str, rng: np.random.Generator) -> str:
    if "|" not in tok:
        return tok
    alts = [a.strip().upper() for a in tok.split("|") if a.strip()]
    return str(rng.choice(alts))


# ═══════════════════════════════════════════════════════════════════════════
# MFC helper (Alicat serial)
# ═══════════════════════════════════════════════════════════════════════════

async def _set_mfc_setpoints(
    mfc_commands: Dict[str, float],
    mfc_device_map: Dict[str, str],
    alicat_ports: List[str],
    alicat_baud: List[int],
    alicat_expected_ids: List[str] | None = None,
) -> None:
    """Apply MFC setpoints via AlicatManager.

    ``mfc_commands`` maps protocol device names (e.g. ``mfc.air_left_setpoint``)
    to target flow rates. ``mfc_device_map`` may map those to either full
    Alicat manager device names (for example ``A@COM8``) or bare unit IDs
    (for example ``A``).
    """
    from multibios.alicat_manager import AlicatManager

    mgr = AlicatManager()
    # AlicatManager.__init__ already loads the cache; only scan if cache is empty.
    if not mgr.device_map:
        scan_kw: Dict[str, Any] = {}
        if alicat_ports:
            scan_kw["ports"] = alicat_ports
        if alicat_baud:
            scan_kw["baudrates"] = alicat_baud
        if alicat_expected_ids:
            scan_kw["expected_ids"] = alicat_expected_ids
        await mgr.scan(**scan_kw)

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
        self._event_log: List[Dict[str, Any]] = []
        self._run_dir: Optional[Path] = None
        self._t0: float = 0.0  # experiment start (perf_counter)

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
        self._teensy = TeensyController(
            port=self.cfg.teensy_port,
            baudrate=self.cfg.teensy_baud,
        )
        self._teensy.open()
        resp = self._teensy.reset()
        print(f"  Teensy RESET: {resp}")

        # 2. Set initial MFC setpoints (async)
        if self.cfg.mfc_mode == "alicat_serial":
            self._apply_initial_mfcs()

        # 3. Start FicTrac
        if self.cfg.fictrac_config:
            print("Starting FicTrac...")
            self._fictrac_callback = ExperimentCallback()
            self._fictrac_driver = FicTracDriver(
                config_file=self.cfg.fictrac_config,
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
            self._fictrac_thread.start()
            # Wait for first frame
            print("  Waiting for FicTrac first frame...")
            deadline = time.monotonic() + 90.0
            while self._fictrac_callback.latest is None and time.monotonic() < deadline:
                time.sleep(0.5)
            if self._fictrac_callback.latest is None:
                raise RuntimeError("FicTrac did not produce any frames within 90 s")
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

        # 2. Reset Teensy (all valves off)
        if self._teensy is not None and self._teensy.is_open:
            print("  Resetting Teensy (all valves off)...")
            try:
                self._teensy.reset()
            except Exception as e:
                print(f"    WARNING: Teensy reset failed: {e}")
            self._teensy.close()
            self._teensy = None

        # 3. Stop FicTrac
        if self._fictrac_callback is not None:
            print("  Stopping FicTrac...")
            self._fictrac_callback.request_stop()
        if self._fictrac_thread is not None:
            self._fictrac_thread.join(timeout=10.0)
            if self._fictrac_thread.is_alive():
                print("    WARNING: FicTrac thread did not exit cleanly")

        # 4. Zero MFCs
        if self.cfg.mfc_mode == "alicat_serial" and self.cfg.mfc_device_map:
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
        with open(self._run_dir / "timeline.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "time_ms", "action", "device", "side", "state", "value",
                "phase", "repeat_idx",
            ])
            for evt in self.timeline:
                writer.writerow([
                    f"{evt.time_ms:.1f}", evt.action, evt.device,
                    evt.side, evt.state, f"{evt.value:.4f}",
                    evt.phase, evt.repeat_idx,
                ])

        # 5. Merged experiment CSV — primary analysis file
        n_frames = self._save_merged_csv()

        print(f"  FicTrac frames: {n_frames}")
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
            explorer_py = Path(__file__).parent.parent / "explorer.py"
            if not explorer_py.exists():
                print(f"  Explorer: {explorer_py} not found — skipping auto-open")
                return
            subprocess.Popen(
                [sys.executable, str(explorer_py), "--no-browser", "--port", str(port)],
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
        posx / posy                integrated ball position (ball radii)
        heading                    heading angle (radians)
        speed                      ball speed (ball radii / s)
        direction                  movement direction (radians)
        intx / inty                integrated X/Y (alternate FicTrac units)
        fictrac_timestamp          FicTrac internal frame timestamp
        delta_timestamp            time since last FicTrac frame
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
        frames = self._fictrac_callback.frames
        if not frames:
            return 0

        t0 = self._t0 if self._t0 else frames[0].wall_time
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
                "posx", "posy", "heading", "speed", "direction", "intx", "inty",
                "fictrac_timestamp", "delta_timestamp",
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
                exp_t = fr.wall_time - t0

                # Advance event pointer: apply all events up to this frame
                while ev_ptr < n_events and state_events[ev_ptr]["wall_time"] <= fr.wall_time:
                    apply_event(state_events[ev_ptr])
                    ev_ptr += 1

                writer.writerow([
                    f"{exp_t:.6f}",
                    f"{fr.wall_time:.6f}",
                    fr.frame_cnt,
                    f"{fr.posx:.8f}",
                    f"{fr.posy:.8f}",
                    f"{fr.heading:.8f}",
                    f"{fr.speed:.8f}",
                    f"{fr.direction:.8f}",
                    f"{fr.intx:.8f}",
                    f"{fr.inty:.8f}",
                    f"{fr.timestamp:.8f}",
                    f"{fr.delta_timestamp:.8f}",
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

        return len(frames)

    def _save_metadata(self) -> None:
        """Save metadata even for dry runs."""
        if self._run_dir is None:
            return

        self._run_dir.mkdir(parents=True, exist_ok=True)

        # Copy protocol and hardware config
        import shutil
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
            "n_timeline_events": len(self.timeline),
            "n_microscope_triggers": len(self.microscope_times),
            "n_camera_windows": len(self.camera_windows),
        }
        with open(self._run_dir / "meta.json", "w") as f:
            json.dump(meta, f, indent=2, default=str)


# ═══════════════════════════════════════════════════════════════════════════
# CLI entry point
# ═══════════════════════════════════════════════════════════════════════════

def load_experiment_config(path: str | Path) -> ExperimentConfig:
    """Load experiment_config.yaml into an ExperimentConfig."""
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    cfg = ExperimentConfig()
    cfg.teensy_port = raw.get("teensy_port", cfg.teensy_port)
    cfg.teensy_baud = raw.get("teensy_baud", cfg.teensy_baud)
    cfg.fictrac_config = raw.get("fictrac_config", cfg.fictrac_config)
    cfg.fictrac_bin = raw.get("fictrac_bin", cfg.fictrac_bin)
    cfg.fictrac_console_out = raw.get("fictrac_console_out", cfg.fictrac_console_out)
    cfg.fictrac_timeout_s = float(raw.get("fictrac_timeout_s", cfg.fictrac_timeout_s))
    cfg.mfc_mode = raw.get("mfc_mode", cfg.mfc_mode)
    cfg.mfc_device_map = raw.get("mfc_device_map", cfg.mfc_device_map)
    cfg.alicat_ports = raw.get("alicat_ports", cfg.alicat_ports)
    cfg.alicat_baud = raw.get("alicat_baud", cfg.alicat_baud)
    cfg.alicat_expected_ids = raw.get("alicat_expected_ids", cfg.alicat_expected_ids)
    cfg.latch_interval_ms = raw.get("latch_interval_ms", cfg.latch_interval_ms)
    cfg.data_dir = raw.get("data_dir", cfg.data_dir)
    cfg.open_explorer = bool(raw.get("open_explorer", cfg.open_explorer))
    cfg.explorer_port = int(raw.get("explorer_port", cfg.explorer_port))
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
    exp_cfg = ExperimentConfig()  # defaults
    exp_cfg_path = Path(args.experiment)
    if exp_cfg_path.exists():
        exp_cfg = load_experiment_config(exp_cfg_path)
    elif not args.dry_run:
        print(f"WARNING: Experiment config '{args.experiment}' not found, using defaults")

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
