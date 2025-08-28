#!/usr/bin/env python3
"""
NI USB-6353 hardware-clocked protocol parser.

Core compiler for MultiBiOS: compiles YAML → timed DO/AO arrays.
Supports two load modes:

- per_assembly (default): legacy behavior with one LOAD_REQ per assembly.
- global: single GLOBAL_LOAD_REQ stages a 48-bit frame; independent RCK_* commit.

Still includes:
- Sticky S-bit rails
- Seeded randomization (+ strict odor allocation, no COPY, '|' choice per repeat)
- Guardrails (no overlapping preload→commit windows)
- Verbose compile report (now includes load_mode + global windows if used)
"""
from __future__ import annotations

import numpy as np
import numpy.typing as npt
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any, Optional, Union, Protocol, Literal

# ----------------------------- Types & constants -----------------------------

BigStateName = Literal[
    "OFF", "AIR", "ODOR1", "ODOR2", "ODOR3", "ODOR4", "ODOR5", "FLUSH"
]
SmallStateName = Literal["CLEAN", "ODOR"]
SideName = Literal["left", "right"]
LoadMode = Literal["per_assembly", "global"]

ProtocolDict = Dict[str, Any]
TimingDict = Dict[str, Union[int, float, str]]
ActionDict = Dict[str, Union[str, int, float, bool]]
PhaseDict = Dict[str, Union[str, int, bool, List[ActionDict]]]


class HardwareMap(Protocol):
    do_lines: Dict[str, Any]
    ao_channels: Dict[str, Any]


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
SMALL_STATE_CODE: Dict[SmallStateName, int] = {"CLEAN": 0, "ODOR": 1}


@dataclass
class TimingConfig:
    base_unit: str = "ms"
    sample_rate: int = 1000
    camera_interval_ms: int = 100
    camera_pulse_ms: int = 5
    preload_lead_ms: int = 10
    load_req_ms: int = 5
    rck_pulse_ms: int = 1
    trig_pulse_ms: int = 5
    setup_hold_samples: int = 5


class CompileError(Exception):
    pass


# ----------------------------- Compiler -------------------------------------


class ProtocolCompiler:
    """Compile YAML protocol → hardware-timed arrays with guardrails and reporting."""

    def __init__(self, hw: HardwareMap, tcfg: TimingConfig) -> None:
        self.hw = hw
        self.tcfg = tcfg

        # Channels
        self.line_order: List[str] = list(hw.do_lines.keys())
        self.line_to_idx: Dict[str, int] = {n: i for i, n in enumerate(self.line_order)}
        self.ao_order: List[str] = list(hw.ao_channels.keys())
        self.ao_to_idx: Dict[str, int] = {n: i for i, n in enumerate(self.ao_order)}

        # Timing buffers
        self.N: int = 0
        self.dt_ms: float = 1000.0 / tcfg.sample_rate
        self.do: Optional[npt.NDArray[np.bool_]] = None  # (num_lines, N)
        self.ao: Optional[npt.NDArray[np.float32]] = None  # (num_ao, N)

        # Logs
        self.rck_log: List[Tuple[str, int, float]] = []
        self._busy_windows: List[Tuple[int, int, str]] = []  # for per_assembly mode

        # Sticky S transitions
        self._trans_big_L: List[Tuple[int, int]] = []
        self._trans_big_R: List[Tuple[int, int]] = []
        self._trans_small_L: List[Tuple[int, int]] = []
        self._trans_small_R: List[Tuple[int, int]] = []

        # Camera enable rail
        self.camera_enabled: npt.NDArray[np.bool_] = np.array([], dtype=bool)

        # RNG
        self.rng: np.random.Generator = np.random.default_rng()
        self.rng_seed: Optional[int] = None

        # Mode
        self.load_mode: LoadMode = "global"

        # Global-load bookkeeping (only used in load_mode == "global")
        # key = t_load_idx, value = last_end_idx (max of all RCK+w in that group)
        self._global_windows: Dict[int, int] = {}
        self._global_loads_list: List[int] = (
            []
        )  # to preserve insertion order for report

        # Report
        self.report: Dict[str, Any] = {
            "timing": {},
            "mode": "global",
            "phases": [],
            "rck_edges": [],
            "busy_windows": [],
            "global_windows": [],  # only populated in global mode
            "state_transitions": {
                "big_left": [],
                "big_right": [],
                "small_left": [],
                "small_right": [],
            },
        }

    # ---------- utils ----------

    def _time_to_idx(self, t_ms: float) -> int:
        idx = int(round(t_ms / self.dt_ms))
        if idx < 0:
            raise CompileError(f"Negative time index at {t_ms} ms")
        return idx

    def _ensure_length(self, N: int) -> None:
        self.N = N
        self.do = np.zeros((len(self.line_order), N), dtype=np.bool_)
        self.ao = np.zeros((len(self.ao_order), N), dtype=np.float32)
        self.camera_enabled = np.zeros(N, dtype=np.bool_)

    def _set_line(self, name: str, start: int, end: int, value: bool) -> None:
        li = self.line_to_idx[name]
        start, end = max(0, start), min(self.N, end)
        if start < end and self.do is not None:
            self.do[li, start:end] = value

    def _pulse(self, name: str, start_idx: int, width_samples: int) -> None:
        self._set_line(name, start_idx, start_idx + width_samples, True)

    def _set_ao_hold(self, chan: str, start_idx: int, value_volts: float) -> None:
        ai = self.ao_to_idx[chan]
        if self.ao is not None:
            self.ao[ai, start_idx:] = value_volts

    @staticmethod
    def _norm_dev(s: str) -> str:
        return s.strip().lower()

    # ---------- device schedulers ----------

    def schedule_big(self, side: SideName, state_name: str, commit_t_ms: float) -> None:
        if state_name not in BIG_STATE_CODE:
            raise CompileError(f"Unknown BIG state '{state_name}'")
        code = BIG_STATE_CODE[state_name]

        if side == "left":
            S0, S1, S2 = (
                "OLFACTOMETER_LEFT_S0",
                "OLFACTOMETER_LEFT_S1",
                "OLFACTOMETER_LEFT_S2",
            )
            LOADn, RCKn = "OLFACTOMETER_LEFT_LOAD_REQ", "RCK_OLFACTOMETER_LEFT"
            rck_label = "RCK_OLFACTOMETER_LEFT"
            trans_list = self._trans_big_L
            trans_report = self.report["state_transitions"]["big_left"]
        elif side == "right":
            S0, S1, S2 = (
                "OLFACTOMETER_RIGHT_S0",
                "OLFACTOMETER_RIGHT_S1",
                "OLFACTOMETER_RIGHT_S2",
            )
            LOADn, RCKn = "OLFACTOMETER_RIGHT_LOAD_REQ", "RCK_OLFACTOMETER_RIGHT"
            rck_label = "RCK_OLFACTOMETER_RIGHT"
            trans_list = self._trans_big_R
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

        # sticky S before load
        switch_idx = max(0, t_load - sh)
        trans_list.append((switch_idx, code))
        trans_report.append({"t_ms": switch_idx * self.dt_ms, "code": int(code)})

        # pulses depend on mode
        if self.load_mode == "per_assembly":
            self._pulse(LOADn, t_load, w_load)
            self._pulse(RCKn, t_rck, w_rck)
            self.rck_log.append((rck_label, t_rck, t_rck * self.dt_ms))
            self._busy_windows.append(
                (t_load, t_rck + w_rck, f"{rck_label}@{commit_t_ms:.3f}ms")
            )
        else:
            # global mode: record per-assembly RCK; stage GLOBAL later (dedup)
            self._pulse(RCKn, t_rck, w_rck)
            self.rck_log.append((rck_label, t_rck, t_rck * self.dt_ms))
            end_idx = t_rck + w_rck
            prev = self._global_windows.get(t_load, t_load)  # default start
            if t_load not in self._global_windows:
                self._global_loads_list.append(t_load)
            self._global_windows[t_load] = max(prev, end_idx)

    def schedule_small(
        self, side: SideName, state_name: str, commit_t_ms: float
    ) -> None:
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
            trans_list = self._trans_small_L
            trans_report = self.report["state_transitions"]["small_left"]
        elif side == "right":
            Sn, LOADn, RCKn = (
                "SWITCHVALVE_RIGHT_S",
                "SWITCHVALVE_RIGHT_LOAD_REQ",
                "RCK_SWITCHVALVE_RIGHT",
            )
            rck_label = "RCK_SWITCHVALVE_RIGHT"
            trans_list = self._trans_small_R
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

        if self.load_mode == "per_assembly":
            self._pulse(LOADn, t_load, w_load)
            self._pulse(RCKn, t_rck, w_rck)
            self.rck_log.append((rck_label, t_rck, t_rck * self.dt_ms))
            self._busy_windows.append(
                (t_load, t_rck + w_rck, f"{rck_label}@{commit_t_ms:.3f}ms")
            )
        else:
            self._pulse(RCKn, t_rck, w_rck)
            self.rck_log.append((rck_label, t_rck, t_rck * self.dt_ms))
            end_idx = t_rck + w_rck
            if t_load not in self._global_windows:
                self._global_loads_list.append(t_load)
            self._global_windows[t_load] = max(
                self._global_windows.get(t_load, t_load), end_idx
            )

    # camera / microscope

    def schedule_camera_continuous(self, enabled: bool, start_ms: float) -> None:
        start = self._time_to_idx(start_ms)
        self.camera_enabled[start:] = enabled

    def finalize_camera_wave(self, interval_ms: int, pulse_ms: int) -> None:
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
                    enabled, next_tick = True, i
                elif not self.camera_enabled[i] and enabled:
                    enabled = False
                if enabled and i == next_tick:
                    self.do[li, i : i + w] = True
                    next_tick += period

    def schedule_microscope_pulse(
        self, t_ms: float, pulse_ms: Optional[int] = None
    ) -> None:
        width = self.tcfg.trig_pulse_ms if pulse_ms is None else pulse_ms
        start = self._time_to_idx(t_ms)
        w = max(1, self._time_to_idx(width) - self._time_to_idx(0))
        self._pulse("TRIG_MICRO", start, w)

    # ---------- YAML compile ----------

    def compile_from_yaml(self, y: ProtocolDict) -> None:
        p = y.get("protocol", {})
        timing: TimingDict = p.get("timing", {})
        seq: List[PhaseDict] = y.get("sequence", [])

        # timing
        base_unit = timing.get("base_unit", "ms")
        if base_unit != "ms":
            raise CompileError("Only 'ms' base_unit is supported.")
        sr = int(timing.get("sample_rate", 1000))
        self.tcfg.sample_rate = sr
        self.dt_ms = 1000.0 / sr

        # mode
        self.load_mode = str(timing.get("load_mode", "per_assembly")).strip().lower()  # type: ignore
        if self.load_mode not in ("per_assembly", "global"):
            raise CompileError("timing.load_mode must be 'per_assembly' or 'global'")
        self.report["mode"] = self.load_mode

        # RNG
        seed = timing.get("seed", None)
        if seed is not None:
            try:
                self.rng_seed = int(seed)
            except Exception:
                raise CompileError(f"Invalid seed value: {seed}")
        else:
            self.rng_seed = int(np.random.SeedSequence().entropy)
        self.rng = np.random.default_rng(self.rng_seed)

        # camera
        camera_interval = int(timing.get("camera_interval", 0))
        camera_pulse = int(timing.get("camera_pulse_duration", 5))

        # sanity for global mode
        if self.load_mode == "global":
            if "GLOBAL_LOAD_REQ" not in self.line_to_idx:
                raise CompileError(
                    "GLOBAL mode requires DO line 'GLOBAL_LOAD_REQ' mapped in hardware.yaml"
                )

        # report timing
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

        # expand phases (times vs repeat+1)
        expanded: List[Tuple[str, int, Dict[str, Any], int]] = []
        total_ms = 0
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
                raise CompileError(f"Phase '{name}': times/repeat must be positive")
            total_ms += dur * times
            expanded.append((name, dur, entry, times))

        # allocate
        N = int(round(total_ms / self.dt_ms))
        if N <= 0:
            raise CompileError("Total duration resolves to zero samples.")
        self._ensure_length(N)

        # walk phases
        t_cursor = 0.0
        for name, duration, entry, times in expanded:
            randomize = bool(entry.get("randomize", False))
            actions = entry.get("actions", [])

            # collect olfactometer specs
            left_spec, right_spec = None, None
            for a in actions:
                dev = self._norm_dev(a.get("device", ""))
                if dev == "olfactometer.left":
                    left_spec = a.get("state", "OFF")
                elif dev == "olfactometer.right":
                    right_spec = a.get("state", "OFF")

            # strict parse / allocation
            left_list = self._parse_state_list_strict(
                left_spec, times, side="left", phase=name
            )
            right_list = self._parse_state_list_strict(
                right_spec, times, side="right", phase=name
            )

            perm = np.arange(times)
            if randomize:
                perm = self.rng.permutation(times)
            if len(left_list) == times:
                left_list = [left_list[i] for i in perm]
            else:
                left_list = [left_list[0]] * times
            if len(right_list) == times:
                right_list = [right_list[i] for i in perm]
            else:
                right_list = [right_list[0]] * times

            # resolve '|' tokens
            resolved_left = [self._resolve_choice_token(tok) for tok in left_list]
            resolved_right = [self._resolve_choice_token(tok) for tok in right_list]

            # validate
            for tok in resolved_left:
                if tok not in BIG_STATE_CODE:
                    raise CompileError(
                        f"Phase '{name}': left state '{tok}' is invalid."
                    )
            for tok in resolved_right:
                if tok not in BIG_STATE_CODE:
                    raise CompileError(
                        f"Phase '{name}': right state '{tok}' is invalid."
                    )

            # report per-phase
            self.report["phases"].append(
                {
                    "name": name,
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

            # long toggles first (camera)
            for a in actions:
                dev = self._norm_dev(a.get("device", ""))
                timing_ms = float(a.get("timing", 0))
                if dev == "triggers.camera_continuous":
                    self.schedule_camera_continuous(
                        bool(a.get("state", False)), t_cursor + timing_ms
                    )

            # repeats
            for rep_idx in range(times):
                t0 = t_cursor + rep_idx * duration
                for a in actions:
                    dev = self._norm_dev(a.get("device", ""))
                    timing_ms = float(a.get("timing", 0))
                    t_abs = t0 + timing_ms

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

            t_cursor += duration * times

        # finalize: guardrails + camera + sticky rails + report
        if self.load_mode == "per_assembly":
            self._check_guardrails(self._busy_windows, label="preload→commit")
        else:
            self._finalize_global_loads()  # emits GLOBAL_LOAD_REQ pulses + guardrails

        self.finalize_camera_wave(camera_interval, camera_pulse)
        self._finalize_state_lines()

        self.report["rck_edges"] = [
            {"signal": sig, "sample_idx": int(si), "time_ms": float(tms)}
            for (sig, si, tms) in self.rck_log
        ]
        # busy windows
        if self.load_mode == "per_assembly":
            self.report["busy_windows"] = [
                {"start_ms": s * self.dt_ms, "end_ms": e * self.dt_ms, "label": lbl}
                for (s, e, lbl) in sorted(self._busy_windows, key=lambda x: x[0])
            ]

    # ---------- strict helpers ----------

    def _resolve_ao_channel(self, dev_key: str) -> str:
        if dev_key not in self.ao_to_idx:
            raise CompileError(f"No AO channel mapped for '{dev_key}'")
        return dev_key

    @staticmethod
    def _split_commas(s: Any) -> List[str]:
        if s is None:
            return []
        if isinstance(s, list):
            return [str(x).strip().upper() for x in s]
        return [p.strip().upper() for p in str(s).split(",") if p.strip()]

    def _parse_state_list_strict(
        self, spec: Any, times: int, *, side: str, phase: str
    ) -> List[str]:
        toks = self._split_commas(spec)
        if any(tok == "COPY" for tok in toks):
            raise CompileError(
                f"Phase '{phase}': 'COPY' is deprecated; specify explicit states for both sides."
            )
        if len(toks) == 0:
            toks = ["OFF"]
        if len(toks) not in (1, times):
            raise CompileError(
                f"Phase '{phase}' ({side}): provide either 1 state or {times} states; got {len(toks)}"
            )
        # validate tokens or their '|' alternatives
        for t in toks:
            if "|" in t:
                for alt in t.split("|"):
                    altU = alt.strip().upper()
                    if altU not in BIG_STATE_CODE:
                        raise CompileError(
                            f"Phase '{phase}' ({side}): invalid alternative '{alt}' in '{t}'"
                        )
            else:
                if t not in BIG_STATE_CODE:
                    raise CompileError(f"Phase '{phase}' ({side}): invalid state '{t}'")
        return toks

    def _resolve_choice_token(self, tok: str) -> str:
        if "|" not in tok:
            return tok
        alts = [a.strip().upper() for a in tok.split("|") if a.strip()]
        return str(self.rng.choice(alts))

    @staticmethod
    def _validate_small_state(s: str) -> None:
        if s not in SMALL_STATE_CODE:
            raise CompileError(
                f"Small olfactometer state must be CLEAN or ODOR, got '{s}'"
            )

    # ---------- validation/finalization ----------

    def _check_guardrails(
        self, windows: List[Tuple[int, int, str]], *, label: str
    ) -> None:
        if not windows:
            return
        ws = sorted(windows, key=lambda x: x[0])
        prev_s, prev_e, prev_lbl = ws[0]
        for s, e, lbl in ws[1:]:
            if s < prev_e:
                raise CompileError(
                    f"Overlapping {label} windows detected:\n"
                    f"  {prev_lbl}  [{prev_s*self.dt_ms:.3f},{prev_e*self.dt_ms:.3f}] ms\n"
                    f"  {lbl}       [{s*self.dt_ms:.3f},{e*self.dt_ms:.3f}] ms\n"
                    "Adjust timings or spacing to avoid overlap."
                )
            prev_s, prev_e, prev_lbl = s, e, lbl

    def _finalize_global_loads(self) -> None:
        """Emit GLOBAL_LOAD_REQ pulses and guardrail windows per global staging group."""
        loadw = max(1, self._time_to_idx(self.tcfg.load_req_ms) - self._time_to_idx(0))
        li = self.line_to_idx["GLOBAL_LOAD_REQ"]

        # Build ordered groups
        groups = [
            (t_load, self._global_windows[t_load])
            for t_load in sorted(self._global_windows.keys())
        ]

        # Guardrail: no overlapping global windows
        windows_labeled: List[Tuple[int, int, str]] = []
        for idx, (s, e) in enumerate(groups):
            windows_labeled.append((s, e, f"GLOBAL_WINDOW#{idx+1}"))
        self._check_guardrails(windows_labeled, label="GLOBAL preload→commit")

        # Emit pulses and record report
        for s, e in groups:
            self._set_line("GLOBAL_LOAD_REQ", s, s + loadw, True)  # pulse
            self.report["global_windows"].append(
                {"t_load_ms": s * self.dt_ms, "end_ms": e * self.dt_ms}
            )

        # Also publish to generic busy_windows for convenience
        self.report["busy_windows"] = [
            {"start_ms": s * self.dt_ms, "end_ms": e * self.dt_ms, "label": "GLOBAL"}
            for (s, e) in groups
        ]

    def _finalize_state_lines(self) -> None:
        def fill_big(trans: List[Tuple[int, int]], s0: str, s1: str, s2: str) -> None:
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
                bitv = collapsed[i][1] != 0
                self._set_line(s, start, end, bitv)

        fill_big(
            self._trans_big_L,
            "OLFACTOMETER_LEFT_S0",
            "OLFACTOMETER_LEFT_S1",
            "OLFACTOMETER_LEFT_S2",
        )
        fill_big(
            self._trans_big_R,
            "OLFACTOMETER_RIGHT_S0",
            "OLFACTOMETER_RIGHT_S1",
            "OLFACTOMETER_RIGHT_S2",
        )
        fill_small(self._trans_small_L, "SWITCHVALVE_LEFT_S")
        fill_small(self._trans_small_R, "SWITCHVALVE_RIGHT_S")
