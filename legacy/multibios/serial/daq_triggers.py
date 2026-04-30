#!/usr/bin/env python3
"""
daq_triggers.py — Pre-computed NIDAQ trigger & latch waveform generator.

Generates a *finite* hardware-clocked DO waveform covering the full experiment
duration.  The waveform contains:

  1. **Camera triggers** — periodic pulses on TRIG_CAMERA at the configured FPS.
  2. **Microscope triggers** — pulses on TRIG_MICRO at pre-specified times.
  3. **Latch pulses** — periodic GLOBAL_LOAD_REQ + all RCK_* lines so the
     Teensy-staged shift-register data is committed to outputs.

S-bit lines (S0/S1/S2) and AO channels are *not* used — the Teensy v2
serial controller handles valve state, and AlicatManager handles MFCs.

This module is designed to be started/stopped by the experiment runner.  It
exposes a simple ``DAQTriggerManager`` class that builds the waveform, starts
the task, waits for completion, and cleans up.

Hardware requirement: NI-DAQmx driver + ``nidaqmx`` Python package.
"""
from __future__ import annotations

import numpy as np
import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class TriggerConfig:
    """All timing parameters for the trigger waveform."""

    # DAQ sample rate (Hz) — shared clock for all DO lines
    sample_rate: int = 2000

    # Camera trigger: period and pulse width (ms)
    camera_interval_ms: float = 100.0      # 10 FPS default
    camera_pulse_ms: float = 5.0

    # Microscope trigger pulse width (ms)
    trig_pulse_ms: float = 5.0

    # Latch cadence — how often GLOBAL_LOAD_REQ + RCK fire (ms).
    # Lower = faster commitment of serial-staged data.  50 ms is a
    # good default: <1-frame latency at 30 FPS, ~20 Hz update rate.
    latch_interval_ms: float = 50.0

    # Latch timing (same meaning as in the protocol compiler)
    preload_lead_ms: float = 2.0   # delay from GLOBAL_LOAD_REQ to RCK
    load_req_ms: float = 1.0       # width of GLOBAL_LOAD_REQ pulse
    rck_pulse_ms: float = 1.0      # width of each RCK pulse


# ---------------------------------------------------------------------------
# Which DO lines we use (subset of hardware.yaml)
# ---------------------------------------------------------------------------

# The 7 lines this module drives.  Order matters — it defines the row index
# in the boolean waveform array.
TRIGGER_LINE_NAMES: List[str] = [
    "GLOBAL_LOAD_REQ",
    "RCK_OLFACTOMETER_LEFT",
    "RCK_SWITCHVALVE_LEFT",
    "RCK_OLFACTOMETER_RIGHT",
    "RCK_SWITCHVALVE_RIGHT",
    "TRIG_CAMERA",
    "TRIG_MICRO",
]


def _load_hw(path: str | Path) -> Dict[str, str]:
    """Return {line_name: physical_channel} for the trigger lines."""
    with open(path) as f:
        hw = yaml.safe_load(f)
    do_map: Dict[str, str] = hw.get("digital_outputs", {})
    out: Dict[str, str] = {}
    for name in TRIGGER_LINE_NAMES:
        if name not in do_map:
            raise KeyError(f"hardware.yaml missing required DO line '{name}'")
        out[name] = do_map[name]
    return out


# ---------------------------------------------------------------------------
# Waveform builder
# ---------------------------------------------------------------------------

def build_trigger_waveform(
    total_duration_ms: float,
    cfg: TriggerConfig,
    *,
    microscope_times_ms: Sequence[float] = (),
    camera_enable_windows: Sequence[tuple[float, float]] | None = None,
) -> np.ndarray:
    """Build the complete boolean waveform array.

    Parameters
    ----------
    total_duration_ms : float
        Total experiment duration in ms.
    cfg : TriggerConfig
        Timing parameters.
    microscope_times_ms : sequence of float
        Absolute times (ms from experiment start) to fire TRIG_MICRO.
    camera_enable_windows : list of (start_ms, end_ms) or None
        Windows during which camera triggers are active.  ``None`` means
        "always on from time 0".

    Returns
    -------
    waveform : np.ndarray, shape (7, N), dtype bool
        Row order matches ``TRIGGER_LINE_NAMES``.
    """
    dt = 1000.0 / cfg.sample_rate
    N = max(1, int(round(total_duration_ms / dt)))

    waveform = np.zeros((len(TRIGGER_LINE_NAMES), N), dtype=np.bool_)

    # Index helpers
    IDX_LOAD   = 0  # GLOBAL_LOAD_REQ
    IDX_RCK_OL = 1  # RCK_OLFACTOMETER_LEFT
    IDX_RCK_SL = 2  # RCK_SWITCHVALVE_LEFT
    IDX_RCK_OR = 3  # RCK_OLFACTOMETER_RIGHT
    IDX_RCK_SR = 4  # RCK_SWITCHVALVE_RIGHT
    IDX_CAM    = 5  # TRIG_CAMERA
    IDX_MICRO  = 6  # TRIG_MICRO

    def ms_to_idx(ms: float) -> int:
        return max(0, min(N - 1, int(round(ms / dt))))

    def width_samples(ms: float) -> int:
        return max(1, int(round(ms / dt)))

    # --- 1. Latch pulses (GLOBAL_LOAD_REQ + all RCK) -----------------------
    load_w = width_samples(cfg.load_req_ms)
    rck_w  = width_samples(cfg.rck_pulse_ms)
    rck_delay = ms_to_idx(cfg.preload_lead_ms)  # samples after LOAD start → RCK start

    latch_period = max(1, int(round(cfg.latch_interval_ms / dt)))
    t = 0
    while t < N:
        # GLOBAL_LOAD_REQ pulse
        end_load = min(N, t + load_w)
        waveform[IDX_LOAD, t:end_load] = True

        # RCK pulses (all four) after preload lead
        rck_start = t + rck_delay
        if rck_start < N:
            end_rck = min(N, rck_start + rck_w)
            for idx in (IDX_RCK_OL, IDX_RCK_SL, IDX_RCK_OR, IDX_RCK_SR):
                waveform[idx, rck_start:end_rck] = True

        t += latch_period

    # --- 2. Camera triggers -------------------------------------------------
    cam_period = max(1, int(round(cfg.camera_interval_ms / dt)))
    cam_w = width_samples(cfg.camera_pulse_ms)

    # Build enable mask
    cam_enabled = np.zeros(N, dtype=np.bool_)
    if camera_enable_windows is None:
        cam_enabled[:] = True
    else:
        for start_ms, end_ms in camera_enable_windows:
            s = ms_to_idx(start_ms)
            e = ms_to_idx(end_ms)
            cam_enabled[s:e] = True

    # Place pulses
    enabled = False
    next_tick = 0
    for i in range(N):
        if cam_enabled[i] and not enabled:
            enabled = True
            next_tick = i
        elif not cam_enabled[i] and enabled:
            enabled = False
        if enabled and i == next_tick:
            end_cam = min(N, i + cam_w)
            waveform[IDX_CAM, i:end_cam] = True
            next_tick += cam_period

    # --- 3. Microscope triggers ---------------------------------------------
    micro_w = width_samples(cfg.trig_pulse_ms)
    for t_ms in microscope_times_ms:
        idx = ms_to_idx(t_ms)
        end_micro = min(N, idx + micro_w)
        waveform[IDX_MICRO, idx:end_micro] = True

    return waveform


# ---------------------------------------------------------------------------
# DAQ Task Manager
# ---------------------------------------------------------------------------

class DAQTriggerManager:
    """Manages the NIDAQ DO finite task for trigger & latch generation.

    Usage::

        mgr = DAQTriggerManager("config/hardware.yaml", cfg, waveform)
        mgr.start()          # arms and starts the finite task
        mgr.wait(timeout_s)  # blocks until complete
        mgr.stop()           # cleans up
    """

    def __init__(
        self,
        hw_path: str | Path,
        cfg: TriggerConfig,
        waveform: np.ndarray,
    ) -> None:
        self.hw_map = _load_hw(hw_path)
        self.cfg = cfg
        self.waveform = waveform
        self._task = None

    @property
    def duration_s(self) -> float:
        return self.waveform.shape[1] / self.cfg.sample_rate

    def start(self) -> None:
        """Create, configure, write, and start the NIDAQ DO task."""
        import nidaqmx
        from nidaqmx.constants import AcquisitionType, LineGrouping

        task = nidaqmx.Task("MultiBiOS_Triggers")
        try:
            # Add channels in TRIGGER_LINE_NAMES order
            for name in TRIGGER_LINE_NAMES:
                phys = self.hw_map[name]
                task.do_channels.add_do_chan(
                    phys,
                    name_to_assign_to_lines=name,
                    line_grouping=LineGrouping.CHAN_PER_LINE,
                )

            N = self.waveform.shape[1]
            task.timing.cfg_samp_clk_timing(
                rate=self.cfg.sample_rate,
                sample_mode=AcquisitionType.FINITE,
                samps_per_chan=N,
            )

            task.write(self.waveform, auto_start=False)
            task.start()
            self._task = task
        except Exception:
            task.close()
            raise

    def wait(self, timeout_s: float | None = None) -> None:
        """Block until the finite task completes."""
        if self._task is None:
            return
        to = timeout_s if timeout_s is not None else self.duration_s + 30.0
        self._task.wait_until_done(timeout=to)

    def stop(self) -> None:
        """Stop and close the task (idempotent)."""
        if self._task is not None:
            try:
                self._task.stop()
            except Exception:
                pass
            try:
                self._task.close()
            except Exception:
                pass
            self._task = None

    def is_done(self) -> bool:
        """Non-blocking check whether the finite task has completed."""
        if self._task is None:
            return True
        return self._task.is_task_done()
