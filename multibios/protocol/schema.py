#!/usr/bin/env python3
"""
NI USB-6353 hardware-clocked protocol parser.

This module provides the core protocol compilation system for MultiBiOS,
converting YAML protocol specifications into hardware-timed digital/analog
output arrays with comprehensive validation and logging.

Features:
- Guardrails against overlapping preload→commit windows
- Seeded randomization with reproducibility guarantees
- Sticky state-bit rails (S-lines reflect current logical state)
- STRICT odor allocation rules (no COPY; '|' per-repeat random choice)
- Verbose compile report for audit & debugging
"""
from __future__ import annotations

import yaml
import numpy as np
import numpy.typing as npt
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any, Optional, Union, Protocol, Literal


# ----------------------------- Type Definitions -----------------------------

# State name literals for type safety
BigStateName = Literal[
    "OFF", "AIR", "ODOR1", "ODOR2", "ODOR3", "ODOR4", "ODOR5", "FLUSH"
]
SmallStateName = Literal["CLEAN", "ODOR"]
SideName = Literal["left", "right"]
DeviceName = str  # Could be more specific but varies by hardware config

# Protocol data structures
ProtocolDict = Dict[str, Any]
TimingDict = Dict[str, Union[int, float, str]]
ActionDict = Dict[str, Union[str, int, float, bool]]
PhaseDict = Dict[str, Union[str, int, bool, List[ActionDict]]]


# Hardware mapping protocol
class HardwareMap(Protocol):
    """Protocol defining the required hardware mapping interface."""

    do_lines: Dict[str, Any]
    ao_channels: Dict[str, Any]


# ----------------------------- State Coding -----------------------------

BIG_STATE_CODE: Dict[BigStateName, int] = {
    "OFF": 0,
    "AIR": 1,
    "ODOR1": 2,
    "ODOR2": 3,
    "ODOR3": 4,
    "ODOR4": 5,
    "ODOR5": 6,
    "FLUSH": 7,
}

SMALL_STATE_CODE: Dict[SmallStateName, int] = {
    "CLEAN": 0,
    "ODOR": 1,
}


@dataclass
class TimingConfig:
    """Hardware timing configuration parameters.

    All timing values are in milliseconds unless otherwise specified.
    These parameters control the precise timing of valve operations
    and must match the firmware expectations.
    """

    base_unit: str = "ms"  # Base timing unit (only "ms" currently supported)
    sample_rate: int = 1000  # DAQ sampling rate in Hz
    camera_interval_ms: int = 100  # Interval between camera triggers
    camera_pulse_ms: int = 5  # Camera trigger pulse width
    preload_lead_ms: int = 10  # Lead time between LOAD_REQ and RCK
    load_req_ms: int = 5  # LOAD_REQ pulse width
    rck_pulse_ms: int = 1  # RCK pulse width
    trig_pulse_ms: int = 5  # Microscope trigger pulse width
    setup_hold_samples: int = 5  # Extra samples to hold S-bits around LOAD_REQ


class CompileError(Exception):
    """Raised when protocol compilation fails due to invalid configuration or timing conflicts."""

    pass


class ProtocolCompiler:
    """
    Compiles YAML protocol specifications into hardware-timed digital/analog arrays.

    This is the core compilation engine that transforms human-readable YAML protocols
    into precise timing arrays suitable for hardware execution. Provides comprehensive
    validation, timing conflict detection, and detailed logging for reproducibility.

    Attributes:
        hw: Hardware mapping configuration
        tcfg: Timing configuration parameters
        N: Total number of samples in compiled protocol
        dt_ms: Time step per sample in milliseconds
        do: Digital output array (num_lines, N)
        ao: Analog output array (num_ao, N)
        rng: Seeded random number generator for reproducible randomization
        report: Comprehensive compilation report for audit trail
    """

    # ---------- Initialization ----------

    def __init__(self, hw: HardwareMap, tcfg: TimingConfig) -> None:
        """Initialize compiler with hardware mapping and timing configuration.

        Args:
            hw: Hardware mapping defining available digital/analog channels
            tcfg: Timing configuration with pulse widths and sample rates
        """
        self.hw: HardwareMap = hw
        self.tcfg: TimingConfig = tcfg

        # Build channel mappings for fast lookup
        self.line_order: List[str] = list(hw.do_lines.keys())
        self.line_to_idx: Dict[str, int] = {
            name: i for i, name in enumerate(self.line_order)
        }
        self.ao_order: List[str] = list(hw.ao_channels.keys())
        self.ao_to_idx: Dict[str, int] = {
            name: i for i, name in enumerate(self.ao_order)
        }

        # Timing arrays (allocated during compilation)
        self.N: int = 0
        self.dt_ms: float = 1000.0 / tcfg.sample_rate
        self.do: Optional[npt.NDArray[np.bool_]] = None  # (num_lines, N)
        self.ao: Optional[npt.NDArray[np.float32]] = None  # (num_ao, N)

        # Execution logs for validation and debugging
        self.rck_log: List[Tuple[str, int, float]] = []  # (signal, sample_idx, time_ms)
        self._busy_windows: List[Tuple[int, int, str]] = (
            []
        )  # (start_idx, end_idx, label)

        # State transition tracking for sticky S-bit rails
        self._transitions_big_left: List[Tuple[int, int]] = (
            []
        )  # (sample_idx, state_code)
        self._transitions_big_right: List[Tuple[int, int]] = []
        self._transitions_small_left: List[Tuple[int, int]] = (
            []
        )  # (sample_idx, bit_value)
        self._transitions_small_right: List[Tuple[int, int]] = []

        # Camera control state tracking
        self.camera_enabled: npt.NDArray[np.bool_] = np.array([], dtype=bool)

        # Reproducible random number generation
        self.rng: np.random.Generator = np.random.default_rng()
        self.rng_seed: Optional[int] = None

        # Comprehensive compilation report for audit and debugging
        self.report: Dict[str, Any] = {
            "timing": {},
            "phases": [],  # One entry per phase instance with resolved states
            "busy_windows": [],  # Timing windows for guardrail validation
            "rck_edges": [],  # All RCK commits with timestamps
            "state_transitions": {  # Sticky S-bit state changes
                "big_left": [],
                "big_right": [],
                "small_left": [],
                "small_right": [],
            },
        }

    # ---------- Utility Methods ----------

    def _time_to_idx(self, t_ms: float) -> int:
        """Convert time in milliseconds to sample index.

        Args:
            t_ms: Time in milliseconds

        Returns:
            Sample index (0-based)

        Raises:
            CompileError: If time is negative
        """
        idx = int(round(t_ms / self.dt_ms))
        if idx < 0:
            raise CompileError(f"Negative time index at {t_ms} ms")
        return idx

    def _ensure_length(self, N: int) -> None:
        """Allocate timing arrays for the specified number of samples.

        Args:
            N: Total number of samples required
        """
        self.N = N
        self.do = np.zeros((len(self.line_order), N), dtype=np.bool_)
        self.ao = np.zeros((len(self.ao_order), N), dtype=np.float32)
        self.camera_enabled = np.zeros(N, dtype=np.bool_)

    def _set_line(self, name: str, start: int, end: int, value: bool) -> None:
        """Set digital output line to specified value over sample range.

        Args:
            name: Digital line name
            start: Start sample index (inclusive)
            end: End sample index (exclusive)
            value: Boolean value to set
        """
        li = self.line_to_idx[name]
        start = max(0, start)
        end = min(self.N, end)
        if start < end and self.do is not None:
            self.do[li, start:end] = value

    def _pulse(self, name: str, start_idx: int, width_samples: int) -> None:
        """Generate a pulse on the specified digital line.

        Args:
            name: Digital line name
            start_idx: Starting sample index
            width_samples: Pulse width in samples
        """
        self._set_line(name, start_idx, start_idx + width_samples, True)

    def _set_ao_hold(self, chan: str, start_idx: int, value_volts: float) -> None:
        """Set analog output channel to specified voltage from start index onwards.

        Args:
            chan: Analog output channel name
            start_idx: Starting sample index
            value_volts: Voltage value to hold
        """
        ai = self.ao_to_idx[chan]
        if self.ao is not None:
            self.ao[ai, start_idx:] = value_volts

    @staticmethod
    def _norm_dev(s: str) -> str:
        """Normalize device name for consistent lookup.

        Args:
            s: Raw device name string

        Returns:
            Normalized device name (lowercase, stripped)
        """
        return s.strip().lower()

    # ---------- Device Scheduling Methods ----------

    def schedule_big(self, side: SideName, state_name: str, commit_t_ms: float) -> None:
        """Schedule a big olfactometer (8-valve) state change with preload→commit timing.

        Args:
            side: Which olfactometer side ("left" or "right")
            state_name: Target state name (OFF, AIR, ODOR1-5, FLUSH)
            commit_t_ms: Absolute time when state change should commit

        Raises:
            CompileError: If state name is invalid or side is unrecognized
        """
        if state_name not in BIG_STATE_CODE:
            raise CompileError(f"Unknown BIG state '{state_name}'")
        code = BIG_STATE_CODE[state_name]

        if side == "left":
            S0n, S1n, S2n = (
                "OLFACTOMETER_LEFT_S0",
                "OLFACTOMETER_LEFT_S1",
                "OLFACTOMETER_LEFT_S2",
            )
            LOADn, RCKn = "OLFACTOMETER_LEFT_LOAD_REQ", "RCK_OLFACTOMETER_LEFT"
            rck_label = "RCK_OLFACTOMETER_LEFT"
            trans_list = self._transitions_big_left
            trans_report = self.report["state_transitions"]["big_left"]
        elif side == "right":
            S0n, S1n, S2n = (
                "OLFACTOMETER_RIGHT_S0",
                "OLFACTOMETER_RIGHT_S1",
                "OLFACTOMETER_RIGHT_S2",
            )
            LOADn, RCKn = "OLFACTOMETER_RIGHT_LOAD_REQ", "RCK_OLFACTOMETER_RIGHT"
            rck_label = "RCK_OLFACTOMETER_RIGHT"
            trans_list = self._transitions_big_right
            trans_report = self.report["state_transitions"]["big_right"]
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

        # Switch S-lines slightly before LOAD_REQ (sticky rails)
        switch_idx = max(0, t_load - sh)
        trans_list.append((switch_idx, code))
        trans_report.append({"t_ms": switch_idx * self.dt_ms, "code": int(code)})

        # Pulses
        self._pulse(LOADn, t_load, w_load)
        self._pulse(RCKn, t_rck, w_rck)

        # Logs
        self.rck_log.append((rck_label, t_rck, t_rck * self.dt_ms))
        self._busy_windows.append(
            (t_load, t_rck + w_rck, f"{rck_label}@{commit_t_ms:.3f}ms")
        )

    def schedule_small(
        self, side: SideName, state_name: str, commit_t_ms: float
    ) -> None:
        """Schedule a small olfactometer (2-valve) state change with preload→commit timing.

        Args:
            side: Which switch valve side ("left" or "right")
            state_name: Target state name (CLEAN or ODOR)
            commit_t_ms: Absolute time when state change should commit

        Raises:
            CompileError: If state name is invalid or side is unrecognized
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
            trans_report = self.report["state_transitions"]["small_left"]
        elif side == "right":
            Sn, LOADn, RCKn = (
                "SWITCHVALVE_RIGHT_S",
                "SWITCHVALVE_RIGHT_LOAD_REQ",
                "RCK_SWITCHVALVE_RIGHT",
            )
            rck_label = "RCK_SWITCHVALVE_RIGHT"
            trans_list = self._transitions_small_right
            trans_report = self.report["state_transitions"]["small_right"]
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

        switch_idx = max(0, t_load - sh)
        trans_list.append((switch_idx, bit))
        trans_report.append({"t_ms": switch_idx * self.dt_ms, "bit": int(bit)})

        self._pulse(LOADn, t_load, w_load)
        self._pulse(RCKn, t_rck, w_rck)

        self.rck_log.append((rck_label, t_rck, t_rck * self.dt_ms))
        self._busy_windows.append(
            (t_load, t_rck + w_rck, f"{rck_label}@{commit_t_ms:.3f}ms")
        )

    def schedule_camera_continuous(self, enabled: bool, start_ms: float) -> None:
        """Enable or disable continuous camera triggering from the specified time.

        Args:
            enabled: Whether to enable (True) or disable (False) camera triggers
            start_ms: Absolute time when the change takes effect
        """
        start = self._time_to_idx(start_ms)
        self.camera_enabled[start:] = enabled

    def finalize_camera_wave(self, interval_ms: int, pulse_ms: int) -> None:
        """Generate periodic camera trigger pulses based on enabled intervals.

        Args:
            interval_ms: Time between trigger pulses in milliseconds
            pulse_ms: Pulse width in milliseconds

        Raises:
            CompileError: If interval is too small for current sample rate
        """
        if interval_ms <= 0:
            return
        w = max(1, self._time_to_idx(pulse_ms) - self._time_to_idx(0))
        li = self.line_to_idx["TRIG_CAMERA"]
        period = self._time_to_idx(interval_ms)
        if period <= 0:
            raise CompileError("camera_interval must be >= 1 ms at current sample rate")
        enabled = False
        next_tick = 0
        if self.do is not None:
            for i in range(self.N):
                if self.camera_enabled[i] and not enabled:
                    enabled = True
                    next_tick = i
                elif not self.camera_enabled[i] and enabled:
                    enabled = False
                if enabled and i == next_tick:
                    self.do[li, i : i + w] = True
                    next_tick += period

    def schedule_microscope_pulse(
        self, t_ms: float, pulse_ms: Optional[int] = None
    ) -> None:
        """Schedule a microscope trigger pulse at the specified time.

        Args:
            t_ms: Absolute time for the trigger pulse
            pulse_ms: Optional pulse width override (uses config default if None)
        """
        width = self.tcfg.trig_pulse_ms if pulse_ms is None else pulse_ms
        start = self._time_to_idx(t_ms)
        w = max(1, self._time_to_idx(width) - self._time_to_idx(0))
        self._pulse("TRIG_MICRO", start, w)

    # ---------- YAML Protocol Compilation ----------

    def compile_from_yaml(self, y: ProtocolDict) -> None:
        """Compile a YAML protocol specification into hardware timing arrays.

        This is the main entry point for protocol compilation. Processes the complete
        YAML specification including timing configuration, phase sequencing, device
        actions, and randomization. Performs comprehensive validation and generates
        detailed compilation reports.

        Args:
            y: Complete YAML protocol dictionary containing 'protocol' and 'sequence' sections

        Raises:
            CompileError: For invalid protocols, timing conflicts, or unsupported features
        """
        # Extract and validate timing configuration
        p = y.get("protocol", {})
        timing: TimingDict = p.get("timing", {})
        seq: List[PhaseDict] = y.get("sequence", [])

        base_unit = timing.get("base_unit", "ms")
        if base_unit != "ms":
            raise CompileError("Only 'ms' base_unit is supported.")

        sr = int(timing.get("sample_rate", 1000))
        self.tcfg.sample_rate = sr
        self.dt_ms = 1000.0 / sr

        seed = timing.get("seed", None)
        if seed is not None:
            try:
                self.rng_seed = int(seed)
            except Exception:
                raise CompileError(f"Invalid seed value: {seed}")
            self.rng = np.random.default_rng(self.rng_seed)
        else:
            self.rng_seed = int(np.random.SeedSequence().entropy)
            self.rng = np.random.default_rng(self.rng_seed)

        camera_interval = int(timing.get("camera_interval", 0))
        camera_pulse = int(timing.get("camera_pulse_duration", 5))

        self.report["timing"] = {
            "sample_rate_hz": sr,
            "dt_ms": self.dt_ms,
            "seed": self.rng_seed,
            "preload_lead_ms": self.tcfg.preload_lead_ms,
            "load_req_ms": self.tcfg.load_req_ms,
            "rck_pulse_ms": self.tcfg.rck_pulse_ms,
            "trig_pulse_ms": self.tcfg.trig_pulse_ms,
            "setup_hold_samples": self.tcfg.setup_hold_samples,
        }

        # Expand phases with correct repeat semantics:
        # If 'times' is present → total repeats = times
        # Else if 'repeat' is present → total repeats = repeat + 1
        # Else → 1
        expanded: List[Tuple[str, int, Dict[str, Any], int]] = (
            []
        )  # (name,duration,entry,times)
        total_ms = 0
        for entry in seq:
            phase_name = entry.get("phase", "PHASE")
            duration = int(entry.get("duration", 0))
            if "times" in entry:
                times = int(entry["times"])
            elif "repeat" in entry:
                times = int(entry["repeat"]) + 1
            else:
                times = 1
            if times <= 0:
                raise CompileError(
                    f"Phase '{phase_name}': times/repeat must be positive."
                )
            total_ms += duration * times
            expanded.append((phase_name, duration, entry, times))

        # Allocate arrays
        N = int(round(total_ms / self.dt_ms))
        if N <= 0:
            raise CompileError("Total duration resolves to zero samples.")
        self._ensure_length(N)

        # Walk phases (one block per phase, with 'times' repeats inside)
        t_cursor = 0.0
        for phase_name, duration, entry, times in expanded:
            randomize = bool(entry.get("randomize", False))
            actions = entry.get("actions", [])

            # Extract olfactometer state specs (strings or lists)
            left_spec, right_spec = None, None
            for a in actions:
                dev = self._norm_dev(a.get("device", ""))
                if dev == "olfactometer.left":
                    left_spec = a.get("state", "OFF")
                elif dev == "olfactometer.right":
                    right_spec = a.get("state", "OFF")

            # Parse state lists; enforce rules; compute shared permutation
            left_list = self._parse_state_list_strict(
                left_spec, times, side="left", phase=phase_name
            )
            right_list = self._parse_state_list_strict(
                right_spec, times, side="right", phase=phase_name
            )

            # Shared permutation if randomize==True and lists are length==times
            perm = np.arange(times)
            if randomize:
                perm = self.rng.permutation(times)

            # Apply permutation only to lists that provided per-repeat vectors
            if len(left_list) == times:
                left_list = [left_list[i] for i in perm]
            else:
                left_list = [left_list[0]] * times

            if len(right_list) == times:
                right_list = [right_list[i] for i in perm]
            else:
                right_list = [right_list[0]] * times

            # Resolve any token with '|' into a concrete state each repeat
            resolved_left = [self._resolve_choice_token(tok) for tok in left_list]
            resolved_right = [self._resolve_choice_token(tok) for tok in right_list]

            # Validate final concrete states
            for tok in resolved_left:
                if tok not in BIG_STATE_CODE:
                    raise CompileError(
                        f"Phase '{phase_name}': left state '{tok}' is invalid."
                    )
            for tok in resolved_right:
                if tok not in BIG_STATE_CODE:
                    raise CompileError(
                        f"Phase '{phase_name}': right state '{tok}' is invalid."
                    )

            # Phase-level report
            self.report["phases"].append(
                {
                    "name": phase_name,
                    "duration_ms": duration,
                    "times": times,
                    "randomize": randomize,
                    "perm": perm.tolist() if randomize else list(range(times)),
                    "olfactometer_left": {
                        "spec": left_spec,
                        "final_sequence": resolved_left,
                    },
                    "olfactometer_right": {
                        "spec": right_spec,
                        "final_sequence": resolved_right,
                    },
                }
            )

            # First pass: camera toggles (long-lived)
            for a in actions:
                dev = self._norm_dev(a.get("device", ""))
                timing_ms = float(a.get("timing", 0))
                if dev == "triggers.camera_continuous":
                    self.schedule_camera_continuous(
                        bool(a.get("state", False)), t_cursor + timing_ms
                    )

            # Schedule repeats
            for rep_idx in range(times):
                phase_t0 = t_cursor + rep_idx * duration

                for a in actions:
                    dev = self._norm_dev(a.get("device", ""))
                    timing_ms = float(a.get("timing", 0))
                    t_abs = phase_t0 + timing_ms

                    if dev.startswith("mfc."):
                        val = float(a.get("value", a.get("state", 0.0)))
                        chan = self._resolve_ao_channel(dev)
                        self._set_ao_hold(chan, self._time_to_idx(t_abs), val)

                    elif dev == "olfactometer.left":
                        self.schedule_big("left", resolved_left[rep_idx], t_abs)

                    elif dev == "olfactometer.right":
                        self.schedule_big("right", resolved_right[rep_idx], t_abs)

                    elif dev == "switch_valve.left":
                        st = (
                            str(a.get("state", a.get("value", "CLEAN"))).strip().upper()
                        )
                        self._validate_small_state(st)
                        self.schedule_small("left", st, t_abs)

                    elif dev == "switch_valve.right":
                        st = (
                            str(a.get("state", a.get("value", "CLEAN"))).strip().upper()
                        )
                        self._validate_small_state(st)
                        self.schedule_small("right", st, t_abs)

                    elif dev == "triggers.microscope":
                        if bool(a.get("state", True)):
                            width = int(a.get("pulse_ms", self.tcfg.trig_pulse_ms))
                            self.schedule_microscope_pulse(t_abs, width)

                    elif dev in ("triggers.camera", "triggers.camera_continuous"):
                        pass
                    else:
                        raise CompileError(f"Unsupported device '{dev}' in actions.")

            # advance time cursor after this phase block
            t_cursor += duration * times

        # Guardrail: no overlapping busy windows
        self._check_guardrails()

        # Camera pulses
        self.finalize_camera_wave(camera_interval, camera_pulse)

        # Sticky S-bit rails populated from transitions
        self._finalize_state_lines()

        # Finalize report logs
        self.report["rck_edges"] = [
            {"signal": sig, "sample_idx": int(si), "time_ms": float(tms)}
            for (sig, si, tms) in self.rck_log
        ]
        self.report["busy_windows"] = [
            {"start_ms": s * self.dt_ms, "end_ms": e * self.dt_ms, "label": lbl}
            for (s, e, lbl) in sorted(self._busy_windows, key=lambda x: x[0])
        ]

    # ---------- Strict Parsing Helpers ----------

    def _resolve_ao_channel(self, dev_key: str) -> str:
        """Resolve device key to analog output channel name.

        Args:
            dev_key: Device key from protocol (e.g., 'mfc.air_left_setpoint')

        Returns:
            Analog output channel name

        Raises:
            CompileError: If no AO channel is mapped for this device
        """
        if dev_key not in self.ao_to_idx:
            raise CompileError(f"No AO channel mapped for '{dev_key}'")
        return dev_key

    @staticmethod
    def _split_commas(s: Any) -> List[str]:
        """Convert state specification into normalized token list.

        Accepts string (comma-separated) or list format. Normalizes whitespace
        and converts to uppercase for consistent state matching.

        Args:
            s: State specification (string, list, or None)

        Returns:
            List of normalized state tokens
        """
        if s is None:
            return []
        if isinstance(s, list):
            return [str(x).strip().upper() for x in s]
        return [p.strip().upper() for p in str(s).split(",") if p.strip()]

    def _parse_state_list_strict(
        self, spec: Any, times: int, *, side: str, phase: str
    ) -> List[str]:
        """Parse and validate olfactometer state specification with strict rules.

        Enforces STRICT allocation rules:
        - Accept exactly 1 token (single state) OR exactly 'times' tokens
        - No 'COPY' allowed (deprecated feature)
        - Tokens may contain '|' choices for per-repeat randomization

        Args:
            spec: State specification (string, list, or None)
            times: Number of phase repetitions expected
            side: Olfactometer side for error messages
            phase: Phase name for error messages

        Returns:
            List of state tokens (length 1 or 'times')

        Raises:
            CompileError: For invalid state specifications or deprecated features
        """
        toks = self._split_commas(spec)
        if any(tok == "COPY" for tok in toks):
            raise CompileError(
                f"Phase '{phase}': 'COPY' is deprecated; specify explicit states for both sides."
            )
        if len(toks) == 0:
            # default single OFF, repeated
            toks = ["OFF"]
        if len(toks) not in (1, times):
            raise CompileError(
                f"Phase '{phase}' ({side}): provide either 1 state or {times} states; got {len(toks)}"
            )
        # Validate base tokens (before resolving '|' choices) — allow '|' alternatives
        for t in toks:
            if "|" in t:
                for alt in t.split("|"):
                    alt = alt.strip().upper()
                    if alt not in BIG_STATE_CODE:
                        raise CompileError(
                            f"Phase '{phase}' ({side}): invalid alternative '{alt}' in '{t}'"
                        )
            else:
                if t not in BIG_STATE_CODE:
                    raise CompileError(f"Phase '{phase}' ({side}): invalid state '{t}'")
        return toks

    def _resolve_choice_token(self, tok: str) -> str:
        """Resolve a state token that may contain '|' alternatives.

        If token contains '|' (e.g., 'ODOR2|ODOR4'), randomly selects one
        alternative using the seeded RNG. Otherwise returns token unchanged.

        Args:
            tok: State token potentially containing '|' choices

        Returns:
            Concrete state name with alternatives resolved
        """
        if "|" not in tok:
            return tok
        alts = [a.strip().upper() for a in tok.split("|") if a.strip()]
        return str(self.rng.choice(alts))

    @staticmethod
    def _validate_small_state(s: str) -> None:
        """Validate that a state name is valid for small olfactometers.

        Args:
            s: State name to validate

        Raises:
            CompileError: If state is not CLEAN or ODOR
        """
        if s not in SMALL_STATE_CODE:
            raise CompileError(
                f"Small olfactometer state must be CLEAN or ODOR, got '{s}'"
            )

    # ---------- Validation and Finalization ----------

    def _check_guardrails(self) -> None:
        """Validate that no preload→commit windows overlap (timing guardrail).

        Overlapping windows would cause hardware conflicts where multiple valve
        assemblies attempt to stage simultaneously. This is a critical safety check.

        Raises:
            CompileError: If overlapping timing windows are detected
        """
        if not self._busy_windows:
            return
        ws = sorted(self._busy_windows, key=lambda x: x[0])
        prev_s, prev_e, prev_lbl = ws[0]
        for s, e, lbl in ws[1:]:
            if s < prev_e:
                raise CompileError(
                    "Overlapping pre-load→commit windows detected:\n"
                    f"  {prev_lbl}  [{prev_s*self.dt_ms:.3f},{prev_e*self.dt_ms:.3f}] ms\n"
                    f"  {lbl}       [{s*self.dt_ms:.3f},{e*self.dt_ms:.3f}] ms\n"
                    "Adjust timings or spacing to avoid overlap."
                )
            prev_s, prev_e, prev_lbl = s, e, lbl

    def _finalize_state_lines(self) -> None:
        """Convert state transitions into persistent S-bit digital output rails.

        Implements "sticky" S-bit behavior where digital lines reflect the current
        logical state between events. Sets defaults at t=0: BIG=OFF(0), SMALL=CLEAN(0).
        """

        def fill_big(trans: List[Tuple[int, int]], s0: str, s1: str, s2: str) -> None:
            """Fill 3-bit big olfactometer state lines from transition list."""
            if not trans or trans[0][0] != 0:
                trans = [(0, 0)] + trans
            trans.sort(key=lambda x: x[0])
            collapsed: List[Tuple[int, int]] = []
            for t_idx, code in trans:
                if collapsed and collapsed[-1][1] == code:
                    continue
                collapsed.append((t_idx, code))
            for i in range(len(collapsed)):
                start = collapsed[i][0]
                end = collapsed[i + 1][0] if i + 1 < len(collapsed) else self.N
                code = collapsed[i][1]
                b0, b1, b2 = (code & 1) != 0, (code & 2) != 0, (code & 4) != 0
                self._set_line(s0, start, end, b0)
                self._set_line(s1, start, end, b1)
                self._set_line(s2, start, end, b2)

        def fill_small(trans: List[Tuple[int, int]], s: str) -> None:
            """Fill 1-bit small olfactometer state line from transition list."""
            if not trans or trans[0][0] != 0:
                trans = [(0, 0)] + trans
            trans.sort(key=lambda x: x[0])
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
