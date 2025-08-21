#!/usr/bin/env python3
"""
NI USB-6353 hardware-clocked protocol parser + guardrails + seeded randomization
and sticky state-bit rails.
"""
from __future__ import annotations

import yaml
import numpy as np
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
    preload_lead_ms: int  # lead between LOAD_REQ and RCK
    load_req_ms: int  # LOAD_REQ width
    rck_pulse_ms: int  # RCK width
    trig_pulse_ms: int  # microscope pulse width
    setup_hold_samples: int  # samples to hold S-bits before/after LOAD_REQ


class CompileError(Exception):
    pass


class ProtocolCompiler:
    """
    Compiles YAML protocol into sample-locked digital and analog waveforms.
    Adds:
      - Guardrail to prevent overlapping preload->commit windows.
      - Seeded randomization (protocol.timing.seed or CLI override).
      - Sticky S-bit rails representing current state between events.
    """

    def __init__(self, hw: "HardwareMap", tcfg: TimingConfig):
        self.hw = hw
        self.tcfg = tcfg
        self.line_order = list(hw.do_lines.keys())  # fixed channel order
        self.line_to_idx = {name: i for i, name in enumerate(self.line_order)}
        self.ao_order = list(hw.ao_channels.keys())
        self.ao_to_idx = {name: i for i, name in enumerate(self.ao_order)}

        # Placeholders (filled after compile)
        self.N = 0
        self.dt_ms = 1000.0 / tcfg.sample_rate
        self.do = None  # shape (num_lines, N) bool
        self.ao = None  # shape (num_ao, N) float32

        # Logs
        self.rck_log: List[Tuple[str, int, float]] = []  # (signal, sample_idx, time_ms)

        # camera continuous bookkeeping
        self.camera_enabled = np.array([], dtype=bool)

        # Randomization
        self.rng = np.random.default_rng()
        self.rng_seed: Optional[int] = None  # recorded for meta

        # Guardrail bookkeeping: list of (start_idx, end_idx, label)
        self._busy_windows: List[Tuple[int, int, str]] = []

        # Sticky-state transitions per assembly (switch index -> code)
        # We'll fill S-lines in a finalize pass based on these transitions.
        self._transitions_big_left: List[Tuple[int, int]] = (
            []
        )  # (switch_idx, code 0..7)
        self._transitions_big_right: List[Tuple[int, int]] = []
        self._transitions_small_left: List[Tuple[int, int]] = (
            []
        )  # (switch_idx, bit 0/1)
        self._transitions_small_right: List[Tuple[int, int]] = []

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
        side: 'left' -> OLFACTOMETER_LEFT; 'right' -> OLFACTOMETER_RIGHT
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
            trans_list = self._transitions_big_left
        elif side == "right":
            S0n, S1n, S2n = (
                "OLFACTOMETER_RIGHT_S0",
                "OLFACTOMETER_RIGHT_S1",
                "OLFACTOMETER_RIGHT_S2",
            )
            LOADn, RCKn = "OLFACTOMETER_RIGHT_LOAD_REQ", "RCK_OLFACTOMETER_RIGHT"
            rck_label = "RCK_OLFACTOMETER_RIGHT"
            trans_list = self._transitions_big_right
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

        # Sticky S-bits: schedule a transition to 'code' a little BEFORE LOAD_REQ
        switch_idx = max(0, t_load - sh)
        trans_list.append((switch_idx, code))

        # Pulses
        self._pulse(LOADn, t_load, w_load)
        self._pulse(RCKn, t_rck, w_rck)
        self.rck_log.append((rck_label, t_rck, t_rck * self.dt_ms))

        # Guardrail busy window
        self._busy_windows.append(
            (t_load, t_rck + w_rck, f"{rck_label}@{commit_t_ms:.3f}ms")
        )

    def schedule_small(self, side: str, state_name: str, commit_t_ms: float):
        """
        side: 'left' -> SWITCHVALVE_LEFT; 'right' -> SWITCHVALVE_RIGHT
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
            trans_list = self._transitions_small_left
        elif side == "right":
            Sn, LOADn, RCKn = (
                "SWITCHVALVE_RIGHT_S",
                "SWITCHVALVE_RIGHT_LOAD_REQ",
                "RCK_SWITCHVALVE_RIGHT",
            )
            rck_label = "RCK_SWITCHVALVE_RIGHT"
            trans_list = self._transitions_small_right
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

        # Sticky S-bit (single bit)
        switch_idx = max(0, t_load - sh)
        trans_list.append((switch_idx, bit))

        # Pulses
        self._pulse(LOADn, t_load, w_load)
        self._pulse(RCKn, t_rck, w_rck)
        self.rck_log.append((rck_label, t_rck, t_rck * self.dt_ms))

        # Guardrail busy window
        self._busy_windows.append(
            (t_load, t_rck + w_rck, f"{rck_label}@{commit_t_ms:.3f}ms")
        )

    def schedule_camera_continuous(self, enabled: bool, start_ms: float):
        start = self._time_to_idx(start_ms)
        self.camera_enabled[start:] = enabled

    def finalize_camera_wave(self, interval_ms: int, pulse_ms: int):
        if interval_ms <= 0:
            return
        w = max(1, self._time_to_idx(pulse_ms) - self._time_to_idx(0))
        li = self.line_to_idx["TRIG_CAMERA"]
        period = self._time_to_idx(interval_ms)
        if period <= 0:
            raise CompileError("camera_interval must be >= 1 ms at 1 kHz base rate")
        enabled = False
        next_tick = 0
        for i in range(self.N):
            if self.camera_enabled[i] and not enabled:
                enabled = True
                next_tick = i
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
        # Global timing and randomization seed
        p = y.get("protocol", {})
        timing = p.get("timing", {})
        seq = y.get("sequence", [])

        base_unit = timing.get("base_unit", "ms")
        if base_unit != "ms":
            raise CompileError("Only 'ms' base_unit is supported.")

        sr = int(timing.get("sample_rate", 1000))
        self.tcfg.sample_rate = sr
        self.dt_ms = 1000.0 / sr

        # Seed (optional)
        seed = timing.get("seed", None)
        if seed is not None:
            try:
                self.rng_seed = int(seed)
            except Exception:
                raise CompileError(f"Invalid seed value: {seed}")
            self.rng = np.random.default_rng(self.rng_seed)
        else:
            # still record a random seed for provenance
            self.rng_seed = int(np.random.SeedSequence().entropy)

        camera_interval = int(timing.get("camera_interval", 0))
        camera_pulse = int(timing.get("camera_pulse_duration", 5))

        # Expand phases -> total duration
        total_ms = 0
        expanded: List[Tuple[str, int, Dict[str, Any]]] = (
            []
        )  # (phase_name, duration, entry)
        for entry in seq:
            phase_name = entry.get("phase", "PHASE")
            duration = int(entry.get("duration", 0))
            times = int(entry.get("times", entry.get("repeat", 1) or 1))
            times = times if times > 0 else 1
            total_ms += duration * times
            for _ in range(times):
                expanded.append((phase_name, duration, entry))

        # Allocate waveforms
        N = int(round(total_ms / self.dt_ms))
        if N <= 0:
            raise CompileError("Total duration resolves to zero samples.")
        self._ensure_length(N)

        # Pass 1: walk phases, schedule actions
        t_cursor = 0.0
        for phase_name, duration, entry in expanded:
            randomize = bool(entry.get("randomize", False))
            actions = entry.get("actions", [])

            # Per-repeat state lists for big olfactometers
            left_states = None
            right_states = None
            for a in actions:
                dev = self._norm_dev(a.get("device", ""))
                if dev == "olfactometer.left":
                    left_states = self._parse_state_list(a.get("state", ""))
                elif dev == "olfactometer.right":
                    right_states = self._parse_state_list(a.get("state", ""))

            # Resolve per-repeat choices (seeded RNG)
            per_repeat_left = self._resolve_repeat_states(left_states, 1, randomize)
            if right_states is None or (
                len(right_states) == 1 and right_states[0].upper() == "COPY"
            ):
                per_repeat_right = ["COPY"]
            else:
                per_repeat_right = self._resolve_repeat_states(
                    right_states, 1, randomize
                )

            # Execute once (duration already replicated in expanded list)
            # First: collect toggles affecting long windows (camera)
            for a in actions:
                dev = self._norm_dev(a.get("device", ""))
                timing_ms = float(a.get("timing", 0))
                t_abs = t_cursor + timing_ms
                if dev == "triggers.camera_continuous":
                    self.schedule_camera_continuous(bool(a.get("state", False)), t_abs)

            # Now schedule all actions for this block
            for a in actions:
                dev = self._norm_dev(a.get("device", ""))
                timing_ms = float(a.get("timing", 0))
                t_abs = t_cursor + timing_ms

                if dev.startswith("mfc."):
                    val = float(a.get("value", a.get("state", 0.0)))
                    chan = self._resolve_ao_channel(dev)
                    self._set_ao_hold(chan, self._time_to_idx(t_abs), val)

                elif dev == "olfactometer.left":
                    state = a.get("state", "OFF")
                    state = per_repeat_left[0] if len(per_repeat_left) else state
                    st_single = self._coerce_single_state(state)
                    self.schedule_big("left", st_single, t_abs)

                elif dev == "olfactometer.right":
                    state = a.get("state", "OFF")
                    if per_repeat_right[0].upper() == "COPY":
                        st_single = per_repeat_left[0]
                    else:
                        st_single = self._coerce_single_state(per_repeat_right[0])
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
                    if bool(a.get("state", True)):
                        width = int(a.get("pulse_ms", self.tcfg.trig_pulse_ms))
                        self.schedule_microscope_pulse(t_abs, width)

                elif dev in ("triggers.camera", "triggers.camera_continuous"):
                    pass
                else:
                    raise CompileError(f"Unsupported device '{dev}' in actions.")

            # advance cursor
            t_cursor += duration

        # Guardrail: check overlap of busy windows
        self._check_guardrails()

        # Bake camera pulses
        self.finalize_camera_wave(camera_interval, camera_pulse)

        # Finalize sticky S-bit rails from transitions
        self._finalize_state_lines()

    # ------ helpers for YAML fields ------

    def _resolve_ao_channel(self, dev_key: str) -> str:
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
        return [p.strip().upper() for p in s.split(",")]

    def _resolve_repeat_states(
        self, states: Optional[List[str]], times: int, randomize: bool
    ) -> List[str]:
        if not states:
            return ["OFF"] * times
        if len(states) == 1:
            tok = states[0]
            if tok.upper() == "COPY":
                return ["COPY"] * times
            return [tok] * times
        seq = states.copy()
        if randomize:
            seq = list(self.rng.permutation(seq))
        out = []
        for i in range(times):
            out.append(seq[i % len(seq)])
        return out

    @staticmethod
    def _coerce_single_state(s: Any) -> str:
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

    # ------ guardrails & finalize sticky state lines ------

    def _check_guardrails(self):
        if not self._busy_windows:
            return
        # sort by start
        ws = sorted(self._busy_windows, key=lambda x: x[0])
        prev_s, prev_e, prev_lbl = ws[0]
        for s, e, lbl in ws[1:]:
            if s < prev_e:
                # Overlap
                ms_prev_s = prev_s * self.dt_ms
                ms_prev_e = prev_e * self.dt_ms
                ms_s = s * self.dt_ms
                ms_e = e * self.dt_ms
                raise CompileError(
                    "Overlapping pre-load->commit windows detected:\n"
                    f"  {prev_lbl}  [{ms_prev_s:.3f},{ms_prev_e:.3f}] ms\n"
                    f"  {lbl}       [{ms_s:.3f},{ms_e:.3f}] ms\n"
                    "Adjust timings, or increase preload_lead_ms / pulse widths to avoid overlap."
                )
            prev_s, prev_e, prev_lbl = s, e, lbl

    def _finalize_state_lines(self):
        """
        Convert the transition lists into persistent S-bit rails.
        Default initial codes: BIG=OFF (0), SMALL=CLEAN (0).
        Transitions are at (t_load - setup_hold_samples), so S-bits switch
        just before LOAD_REQ and then stay until the next change.
        """

        def fill_big(trans: List[Tuple[int, int]], s0: str, s1: str, s2: str):
            if not trans or trans[0][0] != 0:
                trans = [(0, 0)] + trans  # start at OFF
            trans = sorted(trans, key=lambda x: x[0])
            # collapse duplicates
            collapsed: List[Tuple[int, int]] = []
            for t_idx, code in trans:
                if collapsed and collapsed[-1][1] == code:
                    continue
                collapsed.append((t_idx, code))
            # write segments
            for i in range(len(collapsed)):
                start = collapsed[i][0]
                end = collapsed[i + 1][0] if i + 1 < len(collapsed) else self.N
                code = collapsed[i][1]
                b0, b1, b2 = (code & 1) != 0, (code & 2) != 0, (code & 4) != 0
                self._set_line(s0, start, end, b0)
                self._set_line(s1, start, end, b1)
                self._set_line(s2, start, end, b2)

        def fill_small(trans: List[Tuple[int, int]], s: str):
            if not trans or trans[0][0] != 0:
                trans = [(0, 0)] + trans  # start at CLEAN (0)
            trans = sorted(trans, key=lambda x: x[0])
            collapsed: List[Tuple[int, int]] = []
            for t_idx, bit in trans:
                if collapsed and collapsed[-1][1] == bit:
                    continue
                collapsed.append((t_idx, bit))
            for i in range(len(collapsed)):
                start = collapsed[i][0]
                end = collapsed[i + 1][0] if i + 1 < len(collapsed) else self.N
                bit = collapsed[i][1] != 0
                self._set_line(s, start, end, bit)

        # Fill all assemblies
        fill_big(
            self._transitions_big_left,
            "OLFACTOMETER_LEFT_S0",
            "OLFACTOMETER_LEFT_S1",
            "OLFACTOMETER_LEFT_S2",
        )
        fill_big(
            self._transitions_big_right,
            "OLFACTOMETER_RIGHT_S0",
            "OLFACTOMETER_RIGHT_S1",
            "OLFACTOMETER_RIGHT_S2",
        )
        fill_small(self._transitions_small_left, "SWITCHVALVE_LEFT_S")
        fill_small(self._transitions_small_right, "SWITCHVALVE_RIGHT_S")
