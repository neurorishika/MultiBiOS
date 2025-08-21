#!/usr/bin/env python3
"""
NI USB-6363 hardware-clocked protocol parser
"""
from __future__ import annotations
import argparse
import sys
import yaml
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any, Optional

# ----------------------------- Protocol compile -----------------------------

BIG_STATE_CODE = {
    "OFF": 0,
    "AIR": 1,
    "ODOR1": 2,
    "ODOR2": 3,
    "ODOR3": 4,
    "ODOR4": 5,
    "ODOR5": 6,
    "FLUSH": 7,
}

SMALL_STATE_CODE = {
    "CLEAN": 0,
    "ODOR": 1,
}


@dataclass
class TimingConfig:
    base_unit: str  # "ms"
    sample_rate: int  # Hz
    camera_interval_ms: int
    camera_pulse_ms: int
    preload_lead_ms: int  # default lead between LOAD_REQ and RCK
    load_req_ms: int  # width of LOAD_REQ pulse
    rck_pulse_ms: int  # width of RCK pulse
    trig_pulse_ms: int  # width of microscope pulses
    setup_hold_samples: int  # samples to hold S-bits before/after LOAD_REQ


class CompileError(Exception):
    pass


class ProtocolCompiler:
    """
    Compiles YAML protocol into sample-locked digital and analog waveforms.
    """

    def __init__(self, hw: HardwareMap, tcfg: TimingConfig):
        self.hw = hw
        self.tcfg = tcfg
        self.line_order = list(hw.do_lines.keys())  # fixed channel order
        self.line_to_idx = {name: i for i, name in enumerate(self.line_order)}
        self.ao_order = list(hw.ao_channels.keys())
        self.ao_to_idx = {name: i for i, name in enumerate(self.ao_order)}

        # Placeholders (filled after compile)
        self.N = 0
        self.dt_ms = 1000.0 / tcfg.sample_rate
        self.do = None  # shape (num_lines, N) boolean
        self.ao = None  # shape (num_ao, N) float
        self.rck_log: List[Tuple[str, int, float]] = (
            []
        )  # (RCK_name, sample_idx, time_ms)

        # camera continuous bookkeeping
        self.camera_enabled = np.array([], dtype=bool)

    # -------- helpers --------

    def _time_to_idx(self, t_ms: float) -> int:
        idx = int(round(t_ms / self.dt_ms))
        if idx < 0:
            raise CompileError(f"Negative time index at {t_ms} ms")
        return idx

    def _ensure_length(self, N: int):
        self.N = N
        self.do = np.zeros((len(self.line_order), N), dtype=np.bool_)
        self.ao = np.zeros((len(self.ao_order), N), dtype=np.float32)
        self.camera_enabled = np.zeros(N, dtype=np.bool_)

    def _set_line(self, name: str, start: int, end: int, value: bool):
        """Set [start, end) samples inclusive-exclusive."""
        li = self.line_to_idx[name]
        start = max(0, start)
        end = min(self.N, end)
        if start < end:
            self.do[li, start:end] = value

    def _pulse(self, name: str, start_idx: int, width_samples: int):
        self._set_line(name, start_idx, start_idx + width_samples, True)

    def _set_ao_hold(self, chan: str, start_idx: int, value_volts: float):
        """Sets AO from start_idx to end of protocol (until next change)."""
        ai = self.ao_to_idx[chan]
        self.ao[ai, start_idx:] = value_volts

    # -------- device scheduling --------

    def schedule_big(self, side: str, state_name: str, commit_t_ms: float):
        """
        side: 'left' -> A; 'right' -> B
        """
        if state_name not in BIG_STATE_CODE:
            raise CompileError(f"Unknown BIG state '{state_name}'")

        code = BIG_STATE_CODE[state_name]
        S0, S1, S2 = (code & 1), ((code >> 1) & 1), ((code >> 2) & 1)

        if side == "left":
            S0n, S1n, S2n = (
                "OLFACTOMETER_LEFT_S0",
                "OLFACTOMETER_LEFT_S1",
                "OLFACTOMETER_LEFT_S2",
            )
            LOADn, RCKn = "OLFACTOMETER_LEFT_LOAD_REQ", "RCK_OLFACTOMETER_LEFT"
            rck_label = "RCK_OLFACTOMETER_LEFT"
        elif side == "right":
            S0n, S1n, S2n = (
                "OLFACTOMETER_RIGHT_S0",
                "OLFACTOMETER_RIGHT_S1",
                "OLFACTOMETER_RIGHT_S2",
            )
            LOADn, RCKn = "OLFACTOMETER_RIGHT_LOAD_REQ", "RCK_OLFACTOMETER_RIGHT"
            rck_label = "RCK_OLFACTOMETER_RIGHT"
        else:
            raise CompileError("Big side must be 'left' or 'right'.")

        lead = self.tcfg.preload_lead_ms
        loadw = self.tcfg.load_req_ms
        rckw = self.tcfg.rck_pulse_ms
        sh = self.tcfg.setup_hold_samples

        t_load = self._time_to_idx(commit_t_ms - lead)
        t_rck = self._time_to_idx(commit_t_ms)
        w_load = max(1, self._time_to_idx(loadw) - self._time_to_idx(0))
        w_rck = max(1, self._time_to_idx(rckw) - self._time_to_idx(0))

        # Hold S bits around LOAD_REQ
        self._set_line(S0n, t_load - sh, t_load + w_load + sh, bool(S0))
        self._set_line(S1n, t_load - sh, t_load + w_load + sh, bool(S1))
        self._set_line(S2n, t_load - sh, t_load + w_load + sh, bool(S2))
        # LOAD_REQ pulse
        self._pulse(LOADn, t_load, w_load)
        # RCK commit pulse at t
        self._pulse(RCKn, t_rck, w_rck)
        self.rck_log.append((rck_label, t_rck, t_rck * self.dt_ms))

    def schedule_small(self, side: str, state_name: str, commit_t_ms: float):
        """
        side: 'left' -> C; 'right' -> D
        """
        if state_name not in SMALL_STATE_CODE:
            raise CompileError(f"Unknown SMALL state '{state_name}'")

        bit = SMALL_STATE_CODE[state_name]

        if side == "left":
            Sn, LOADn, RCKn = (
                "SWITCHVALVE_LEFT_S",
                "SWITCHVALVE_LEFT_LOAD_REQ",
                "RCK_SWITCHVALVE_LEFT",
            )
            rck_label = "RCK_SWITCHVALVE_LEFT"
        elif side == "right":
            Sn, LOADn, RCKn = (
                "SWITCHVALVE_RIGHT_S",
                "SWITCHVALVE_RIGHT_LOAD_REQ",
                "RCK_SWITCHVALVE_RIGHT",
            )
            rck_label = "RCK_SWITCHVALVE_RIGHT"
        else:
            raise CompileError("Small side must be 'left' or 'right'.")

        lead = self.tcfg.preload_lead_ms
        loadw = self.tcfg.load_req_ms
        rckw = self.tcfg.rck_pulse_ms
        sh = self.tcfg.setup_hold_samples

        t_load = self._time_to_idx(commit_t_ms - lead)
        t_rck = self._time_to_idx(commit_t_ms)
        w_load = max(1, self._time_to_idx(loadw) - self._time_to_idx(0))
        w_rck = max(1, self._time_to_idx(rckw) - self._time_to_idx(0))

        # Hold S bit around LOAD_REQ
        self._set_line(Sn, t_load - sh, t_load + w_load + sh, bool(bit))
        # LOAD_REQ pulse
        self._pulse(LOADn, t_load, w_load)
        # RCK commit
        self._pulse(RCKn, t_rck, w_rck)
        self.rck_log.append((rck_label, t_rck, t_rck * self.dt_ms))

    def schedule_camera_continuous(self, enabled: bool, start_ms: float):
        """Toggle camera continuous mode."""
        start = self._time_to_idx(start_ms)
        self.camera_enabled[start:] = enabled

    def finalize_camera_wave(self, interval_ms: int, pulse_ms: int):
        """Bake the camera pulses into TRIG_CAMERA line wherever enabled."""
        if interval_ms <= 0:
            return
        w = max(1, self._time_to_idx(pulse_ms) - self._time_to_idx(0))
        li = self.line_to_idx["TRIG_CAMERA"]
        period = self._time_to_idx(interval_ms)
        if period <= 0:
            raise CompileError("camera_interval must be >= 1 ms at 1 kHz base rate")
        # Walk across timeline; when enabled, emit pulses every 'period'
        enabled = False
        next_tick = 0
        for i in range(self.N):
            if self.camera_enabled[i] and not enabled:
                enabled = True
                next_tick = i  # start pulses now
            elif not self.camera_enabled[i] and enabled:
                enabled = False
            if enabled and i == next_tick:
                self.do[li, i : i + w] = True
                next_tick += period

    def schedule_microscope_pulse(self, t_ms: float, pulse_ms: Optional[int] = None):
        width = self.tcfg.trig_pulse_ms if pulse_ms is None else pulse_ms
        start = self._time_to_idx(t_ms)
        w = max(1, self._time_to_idx(width) - self._time_to_idx(0))
        self._pulse("TRIG_MICRO", start, w)

    # -------- YAML compile --------

    @staticmethod
    def _norm_dev(s: str) -> str:
        return s.strip().lower()

    def compile_from_yaml(self, y: Dict[str, Any]):
        # Global timing
        p = y.get("protocol", {})
        seq = y.get("sequence", [])

        base_unit = p.get("timing", {}).get("base_unit", "ms")
        if base_unit != "ms":
            raise CompileError("Only 'ms' base_unit is supported.")

        sr = int(p.get("timing", {}).get("sample_rate", 1000))
        self.tcfg.sample_rate = sr
        self.dt_ms = 1000.0 / sr

        camera_interval = int(p.get("timing", {}).get("camera_interval", 0))
        camera_pulse = int(p.get("timing", {}).get("camera_pulse_duration", 5))

        # Expand phases -> total duration
        # Each item is either a single phase (repeat=0/1) or a repeated block (times)
        phase_blocks: List[Tuple[str, int, List[Dict[str, Any]]]] = []
        total_ms = 0
        for entry in seq:
            phase_name = entry.get("phase", "PHASE")
            duration = int(entry.get("duration", 0))
            times = int(entry.get("times", entry.get("repeat", 1) or 1))
            # zero means "no explicit repeats"
            times = times if times > 0 else 1
            # Build the actions per repeat later (for randomization)
            phase_blocks.append((phase_name, times, [entry]))
            total_ms += duration * times

        # Allocate waveforms
        N = int(round(total_ms / self.dt_ms))
        if N <= 0:
            raise CompileError("Total duration resolves to zero samples.")
        self._ensure_length(N)

        # Pass 1: walk phases, schedule actions
        t_cursor = 0.0
        for phase_name, times, entries in phase_blocks:
            entry = entries[0]
            duration = int(entry.get("duration", 0))
            randomize = bool(entry.get("randomize", False))
            actions = entry.get("actions", [])

            # Preprocess any per-phase state lists for big olfactometers
            # e.g., "ODOR1,ODOR2,ODOR3" (string) or ["ODOR1","ODOR2",...]
            left_states = None
            right_states = None
            for a in actions:
                dev = self._norm_dev(a.get("device", ""))
                if dev == "olfactometer.left":
                    st = a.get("state", "")
                    left_states = self._parse_state_list(st)
                elif dev == "olfactometer.right":
                    st = a.get("state", "")
                    right_states = self._parse_state_list(st)

            # If "COPY" is specified for right, we’ll copy the resolved left state for each repeat
            # Build per-repeat state choices
            per_repeat_left = self._resolve_repeat_states(left_states, times, randomize)
            if right_states is None or (
                len(right_states) == 1 and right_states[0].upper() == "COPY"
            ):
                per_repeat_right = ["COPY"] * times
            else:
                per_repeat_right = self._resolve_repeat_states(
                    right_states, times, randomize
                )

            # Execute repeats
            for rep_idx in range(times):
                phase_t0 = t_cursor + rep_idx * duration
                # First collect special toggles so camera can be enabled across the block
                for a in actions:
                    dev = self._norm_dev(a.get("device", ""))
                    timing = float(a.get("timing", 0))  # default 0 within block
                    t_abs = phase_t0 + timing
                    if dev == "triggers.camera_continuous":
                        state = bool(a.get("state", False))
                        self.schedule_camera_continuous(state, t_abs)

                # Now schedule all actions
                for a in actions:
                    dev = self._norm_dev(a.get("device", ""))
                    timing = float(a.get("timing", 0))
                    t_abs = phase_t0 + timing

                    if dev.startswith("mfc."):
                        # volts directly, held from t_abs onward
                        val = float(a.get("value", a.get("state", 0.0)))
                        chan = self._resolve_ao_channel(dev)
                        self._set_ao_hold(chan, self._time_to_idx(t_abs), val)

                    elif dev == "olfactometer.left":
                        state = a.get("state", "OFF")
                        # Per-repeat override if list was provided
                        if len(per_repeat_left) > 0:
                            state = per_repeat_left[rep_idx]
                        # 'state' might still be a comma list; we allow only a single here
                        st_single = self._coerce_single_state(state)
                        self.schedule_big("left", st_single, t_abs)

                    elif dev == "olfactometer.right":
                        state = a.get("state", "OFF")
                        st_single = state
                        if per_repeat_right[rep_idx].upper() == "COPY":
                            st_single = per_repeat_left[rep_idx]
                        else:
                            st_single = self._coerce_single_state(
                                per_repeat_right[rep_idx]
                            )
                        self.schedule_big("right", st_single, t_abs)

                    elif dev == "switch_valve.left":
                        st = a.get("state", a.get("value", "CLEAN"))
                        st_single = self._coerce_small_state(st)
                        self.schedule_small("left", st_single, t_abs)

                    elif dev == "switch_valve.right":
                        st = a.get("state", a.get("value", "CLEAN"))
                        st_single = self._coerce_small_state(st)
                        self.schedule_small("right", st_single, t_abs)

                    elif dev == "triggers.microscope":
                        # 'state: true' => emit one pulse at this time (width from config unless overridden)
                        if bool(a.get("state", True)):
                            width = int(a.get("pulse_ms", self.tcfg.trig_pulse_ms))
                            self.schedule_microscope_pulse(t_abs, width)

                    elif dev in ("triggers.camera", "triggers.camera_continuous"):
                        # already handled as mode toggle
                        pass

                    else:
                        raise CompileError(f"Unsupported device '{dev}' in actions.")

                # end repeat

            # advance cursor after all repeats
            t_cursor += duration * times

        # Bake camera pulses where enabled
        self.finalize_camera_wave(camera_interval, camera_pulse)

        # Fill any AO channels never mentioned with their initial default (0 V)
        # Already zero by default.

    # ------ helpers for YAML fields ------

    def _resolve_ao_channel(self, dev_key: str) -> str:
        # dev_key already lowercased
        if dev_key not in self.ao_to_idx:
            raise CompileError(f"No AO channel mapped for '{dev_key}'")
        return dev_key

    @staticmethod
    def _parse_state_list(state_field: Any) -> Optional[List[str]]:
        if state_field is None:
            return None
        if isinstance(state_field, list):
            return [str(s).strip().upper() for s in state_field]
        s = str(state_field).strip()
        if not s:
            return None
        # allow comma-separated form
        parts = [p.strip().upper() for p in s.split(",")]
        return parts

    @staticmethod
    def _resolve_repeat_states(
        states: Optional[List[str]], times: int, randomize: bool
    ) -> List[str]:
        if not states:
            # default to OFF repeated
            return ["OFF"] * times
        # If a single token (e.g., "AIR"), repeat N times
        if len(states) == 1:
            tok = states[0]
            if tok.upper() == "COPY":
                return ["COPY"] * times
            return [tok] * times
        # If more than one token, either loop or randomize across repeats
        seq = states.copy()
        if randomize:
            rng = np.random.default_rng()
            seq = list(rng.permutation(seq))
        # If times > len(seq), cycle the list
        out = []
        for i in range(times):
            out.append(seq[i % len(seq)])
        return out

    @staticmethod
    def _coerce_single_state(s: Any) -> str:
        # If the YAML had "ODOR1,ODOR2" accidentally here, pick first
        if isinstance(s, list):
            return str(s[0]).strip().upper()
        return str(s).strip().upper()

    @staticmethod
    def _coerce_small_state(s: Any) -> str:
        st = str(s).strip().upper()
        if st not in SMALL_STATE_CODE:
            raise CompileError(
                f"Small olfactometer state must be CLEAN or ODOR, got '{s}'"
            )
        return st
