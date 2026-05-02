#!/usr/bin/env python3
"""Analog MFC test interface — NI-DAQ AO setpoint + AI readback.

Replaces the legacy serial-based flow_monitor.  Uses the four AO channels
(Dev1/ao0-3) as 0–5 V setpoints and the matched AI channels (Dev1/ai0-3)
to read actual flow feedback, with the same names defined in hardware.yaml.

Voltage–flow mapping (Alicat default 0–5 V full-scale):
    V = (setpoint / full_scale_sccm) * 5.0
    flow_sccm = (V_feedback / 5.0) * full_scale_sccm

Usage:
    # Live monitor (read current flow, set setpoints interactively):
    python tools/manual_checks/mfc_analog_test.py monitor

    # Set specific channels then monitor:
    python tools/manual_checks/mfc_analog_test.py monitor --set air_left=2.5 odor_right=1.0

    # Step sweep — verify linearity, prints pass/fail:
    python tools/manual_checks/mfc_analog_test.py sweep

    # Set then immediately zero on exit (same pattern as flow_monitor):
    python tools/manual_checks/mfc_analog_test.py monitor --set air_left=3.0 --zero-on-exit

    # Dry-run: print config and exit without touching hardware:
    python tools/manual_checks/mfc_analog_test.py monitor --dry-run

All modes accept --hardware <path> (default: config/hardware.yaml relative to
the MultiBiOS repo root).
"""
from __future__ import annotations

import argparse
import contextlib
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import yaml

try:
    import nidaqmx
    from nidaqmx.constants import AcquisitionType, TerminalConfiguration
    _NIDAQMX_AVAILABLE = True
except ImportError:
    _NIDAQMX_AVAILABLE = False


# ─────────────────────────────────────────────────────────────────────────────
# Hardware config loader
# ─────────────────────────────────────────────────────────────────────────────

# Logical → hardware.yaml key mapping for MFC channels.
_AO_KEYS = {
    "air_left":   "mfc.air_left_setpoint",
    "air_right":  "mfc.air_right_setpoint",
    "odor_left":  "mfc.odor_left_setpoint",
    "odor_right": "mfc.odor_right_setpoint",
}
_AI_KEYS = {
    "air_left":   "mfc.air_left_flowrate",
    "air_right":  "mfc.air_right_flowrate",
    "odor_left":  "mfc.odor_left_flowrate",
    "odor_right": "mfc.odor_right_flowrate",
}

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_HW = _REPO_ROOT / "config" / "hardware.yaml"


@dataclass
class MFCChannels:
    """Resolved physical NI-DAQ channel strings for all four MFCs."""
    ao: Dict[str, str]   # logical name → "Dev1/aoN"
    ai: Dict[str, str]   # logical name → "Dev1/aiN"
    ao_min_v: float = 0.0
    ao_max_v: float = 5.0
    ai_min_v: float = 0.0
    ai_max_v: float = 5.0


def load_mfc_channels(hardware_yaml: Path) -> MFCChannels:
    """Parse hardware.yaml and extract MFC AO/AI physical channels."""
    raw = yaml.safe_load(hardware_yaml.read_text()) or {}
    ao_hw = raw.get("analog_outputs", {})
    ai_hw = raw.get("analog_inputs", {})

    ao: Dict[str, str] = {}
    ai: Dict[str, str] = {}
    missing_ao, missing_ai = [], []

    for logical, yaml_key in _AO_KEYS.items():
        if yaml_key in ao_hw:
            ao[logical] = ao_hw[yaml_key]
        else:
            missing_ao.append(yaml_key)

    for logical, yaml_key in _AI_KEYS.items():
        if yaml_key in ai_hw:
            ai[logical] = ai_hw[yaml_key]
        else:
            missing_ai.append(yaml_key)

    if missing_ao or missing_ai:
        raise KeyError(
            f"Missing channels in {hardware_yaml}:\n"
            + (f"  AO: {missing_ao}\n" if missing_ao else "")
            + (f"  AI: {missing_ai}\n" if missing_ai else "")
        )

    return MFCChannels(ao=ao, ai=ai)


# ─────────────────────────────────────────────────────────────────────────────
# Low-level DAQ helpers
# ─────────────────────────────────────────────────────────────────────────────

def _ao_channel_str(channels: MFCChannels, names: List[str]) -> str:
    return ",".join(channels.ao[n] for n in names)


def _ai_channel_str(channels: MFCChannels, names: List[str]) -> str:
    return ",".join(channels.ai[n] for n in names)


def write_setpoints(
    channels: MFCChannels,
    setpoints_v: Dict[str, float],
    *,
    verify: bool = True,
) -> None:
    """Write AO setpoints (volts).  Clamps to [ao_min_v, ao_max_v].

    Args:
        channels:     MFCChannels from load_mfc_channels().
        setpoints_v:  {logical_name: voltage} — only channels in this dict are touched.
        verify:       If True, immediately read back AI and warn on mismatch > 0.1 V.
    """
    names = list(setpoints_v.keys())
    values = np.array(
        [float(np.clip(setpoints_v[n], channels.ao_min_v, channels.ao_max_v)) for n in names],
        dtype=np.float64,
    )

    with nidaqmx.Task() as ao_task:
        ao_task.ao_channels.add_ao_voltage_chan(
            _ao_channel_str(channels, names),
            min_val=channels.ao_min_v,
            max_val=channels.ao_max_v,
        )
        ao_task.write(values, auto_start=True)

    if verify:
        time.sleep(0.05)
        readback = read_feedback(channels, names, n_samples=10)
        for n in names:
            expected = setpoints_v[n]
            actual = readback[n]
            if abs(actual - expected) > 0.15:
                print(
                    f"  ⚠  {n}: set {expected:.3f} V, feedback {actual:.3f} V "
                    f"(Δ={abs(actual-expected):.3f} V)"
                )


def read_feedback(
    channels: MFCChannels,
    names: Optional[List[str]] = None,
    *,
    n_samples: int = 50,
    rate: float = 1000.0,
) -> Dict[str, float]:
    """Read AI feedback channels and return mean voltage per channel.

    Args:
        channels:  MFCChannels from load_mfc_channels().
        names:     Subset of logical names to read. Defaults to all four.
        n_samples: Samples per channel to average.
        rate:      Sampling rate in Hz.

    Returns:
        {logical_name: mean_voltage}
    """
    if names is None:
        names = list(channels.ai.keys())

    with nidaqmx.Task() as ai_task:
        ai_task.ai_channels.add_ai_voltage_chan(
            _ai_channel_str(channels, names),
            terminal_config=TerminalConfiguration.RSE,
            min_val=channels.ai_min_v,
            max_val=channels.ai_max_v,
        )
        ai_task.timing.cfg_samp_clk_timing(
            rate=rate,
            sample_mode=AcquisitionType.FINITE,
            samps_per_chan=n_samples,
        )
        raw = ai_task.read(number_of_samples_per_channel=n_samples)

    # nidaqmx returns list-of-lists for multi-channel, list for single
    if len(names) == 1:
        raw = [raw]

    return {n: float(np.mean(raw[i])) for i, n in enumerate(names)}


def zero_all(channels: MFCChannels) -> None:
    """Drive all AO channels to 0 V (safe shutdown)."""
    write_setpoints(channels, {n: 0.0 for n in channels.ao}, verify=False)


# ─────────────────────────────────────────────────────────────────────────────
# Test framework
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SweepResult:
    channel: str
    levels_v: List[float]
    measured_v: List[float]
    errors_v: List[float] = field(default_factory=list)

    def passed(self, tolerance_v: float = 0.1) -> bool:
        return all(abs(e) <= tolerance_v for e in self.errors_v)

    def summary(self, tolerance_v: float = 0.1) -> str:
        worst = max(abs(e) for e in self.errors_v) if self.errors_v else 0.0
        status = "PASS" if self.passed(tolerance_v) else "FAIL"
        return f"[{status}] {self.channel:<15}  worst_error={worst:.4f} V  (tol={tolerance_v:.3f} V)"


def run_sweep(
    channels: MFCChannels,
    *,
    levels_v: Optional[List[float]] = None,
    dwell_s: float = 0.5,
    tolerance_v: float = 0.1,
    verbose: bool = True,
) -> List[SweepResult]:
    """Step each AO channel through a set of voltage levels and verify AI readback.

    One channel is swept at a time; all others are held at 0 V.

    Args:
        channels:     MFCChannels from load_mfc_channels().
        levels_v:     Voltage steps to apply. Defaults to [0,1,2,3,4,5,0].
        dwell_s:      Seconds to hold each level before reading.
        tolerance_v:  Max allowed |AO - AI| error for PASS.
        verbose:      Print progress.

    Returns:
        List of SweepResult, one per channel.
    """
    if levels_v is None:
        levels_v = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 0.0]

    results: List[SweepResult] = []
    names = list(channels.ao.keys())

    print(f"\nSweep: {levels_v} V  dwell={dwell_s:.2f}s  tol={tolerance_v:.3f} V\n")

    for name in names:
        measured: List[float] = []
        errors: List[float] = []

        if verbose:
            print(f"  Channel: {name}")

        for v in levels_v:
            # Set target channel; hold all others at 0
            sp = {n: 0.0 for n in names}
            sp[name] = v
            write_setpoints(channels, sp, verify=False)
            time.sleep(dwell_s)

            fb = read_feedback(channels, [name], n_samples=50)
            meas = fb[name]
            err = meas - v
            measured.append(meas)
            errors.append(err)

            if verbose:
                ok = "✓" if abs(err) <= tolerance_v else "✗"
                print(f"    {ok}  set={v:.2f} V  read={meas:.4f} V  err={err:+.4f} V")

        results.append(SweepResult(
            channel=name,
            levels_v=levels_v,
            measured_v=measured,
            errors_v=errors,
        ))

    # Zero everything at end of sweep
    write_setpoints(channels, {n: 0.0 for n in names}, verify=False)

    print("\n── Sweep Summary ──────────────────────────────")
    all_pass = True
    for r in results:
        print(" ", r.summary(tolerance_v))
        if not r.passed(tolerance_v):
            all_pass = False
    print(f"\n{'ALL PASS' if all_pass else 'FAILURES DETECTED'}")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Live monitor  (mirrors flow_monitor.py style)
# ─────────────────────────────────────────────────────────────────────────────

def _render_monitor(
    setpoints_v: Dict[str, float],
    feedback_v: Dict[str, float],
    stats: dict,
    full_scale_sccm: Dict[str, float],
) -> str:
    lines: List[str] = []
    names = list(setpoints_v.keys())
    width = max(len(n) for n in names)

    lines.append(f"{'Channel':<{width}}  {'Setpt (V)':>10}  {'Feedback (V)':>13}  {'Flow (% FS)':>12}  {'Δ (V)':>8}")
    lines.append("─" * (width + 60))

    for n in names:
        sp = setpoints_v.get(n, 0.0)
        fb = feedback_v.get(n, float("nan"))
        delta = fb - sp
        fs = full_scale_sccm.get(n, 1.0)
        pct = (fb / 5.0 * 100.0) if not (fb != fb) else float("nan")  # nan check
        ok = "✓" if abs(delta) <= 0.15 else "⚠"
        lines.append(
            f"{n:<{width}}  {sp:>10.3f}  {fb:>13.4f}  {pct:>11.1f}%  {delta:>+8.4f}  {ok}"
        )

    lines.append("")
    lines.append(
        f"rate={stats['rate_hz']:.1f} Hz  "
        f"read={stats['read_ms']:.1f} ms  "
        f"period(avg/min/max)={stats['avg_p']:.1f}/{stats['min_p']:.1f}/{stats['max_p']:.1f} ms  "
        f"jitter(avg/max)={stats['avg_j']:.2f}/{stats['max_j']:.2f} ms"
    )
    lines.append("Ctrl+C to quit")
    return "\n".join(lines)


def run_monitor(
    channels: MFCChannels,
    *,
    setpoints_v: Optional[Dict[str, float]] = None,
    interval_s: float = 0.5,
    zero_on_exit: bool = True,
    full_scale_sccm: Optional[Dict[str, float]] = None,
) -> None:
    """Live monitor: apply setpoints, continuously read AI feedback, print stats.

    Args:
        channels:       MFCChannels from load_mfc_channels().
        setpoints_v:    Initial setpoints in volts. Defaults to all 0 V.
        interval_s:     Polling interval. Default 0.5 s.
        zero_on_exit:   Drive all AO to 0 V on Ctrl+C. Default True.
        full_scale_sccm: For display only — maps logical name → sccm full scale.
    """
    names = list(channels.ao.keys())
    sp = {n: 0.0 for n in names}
    if setpoints_v:
        for k, v in setpoints_v.items():
            if k not in sp:
                raise KeyError(f"Unknown MFC channel '{k}'. Valid: {names}")
            sp[k] = float(v)

    if full_scale_sccm is None:
        full_scale_sccm = {n: 1.0 for n in names}  # display as raw volts

    # Apply initial setpoints
    print("Applying setpoints …")
    write_setpoints(channels, sp)
    print("Starting monitor. Ctrl+C to stop.\n")

    periods: List[float] = []
    jitters: List[float] = []
    last_start: Optional[float] = None

    sys.stdout.write("\033[?25l")  # hide cursor
    sys.stdout.flush()

    try:
        next_tick = time.perf_counter()
        while True:
            scheduled = next_tick
            now = time.perf_counter()
            if now < scheduled:
                time.sleep(scheduled - now)

            start = time.perf_counter()
            if last_start is not None:
                periods.append(start - last_start)
            last_start = start
            jitters.append(start - scheduled)

            feedback = read_feedback(channels, names, n_samples=20, rate=500.0)
            read_done = time.perf_counter()
            read_ms = (read_done - start) * 1000.0

            if len(periods) > 200:
                periods.pop(0)
            if len(jitters) > 200:
                jitters.pop(0)

            stats = {
                "rate_hz": 1.0 / interval_s,
                "read_ms": read_ms,
                "avg_p": statistics.fmean(periods) * 1000 if periods else 0.0,
                "min_p": min(periods) * 1000 if periods else 0.0,
                "max_p": max(periods) * 1000 if periods else 0.0,
                "avg_j": statistics.fmean(abs(x) for x in jitters) * 1000 if jitters else 0.0,
                "max_j": max(abs(x) for x in jitters) * 1000 if jitters else 0.0,
            }

            text = _render_monitor(sp, feedback, stats, full_scale_sccm)
            sys.stdout.write("\033[H\033[J" + text + "\n")
            sys.stdout.flush()

            next_tick += interval_s

    except KeyboardInterrupt:
        pass
    finally:
        sys.stdout.write("\033[?25h\n")  # restore cursor
        sys.stdout.flush()
        if zero_on_exit:
            print("Zeroing all MFC setpoints …")
            zero_all(channels)
            print("Done.")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse_setpoints(args: List[str]) -> Dict[str, float]:
    """Parse 'name=voltage' strings from CLI args."""
    out: Dict[str, float] = {}
    for s in args:
        if "=" not in s:
            raise argparse.ArgumentTypeError(f"Expected name=voltage, got '{s}'")
        k, _, v = s.partition("=")
        out[k.strip()] = float(v.strip())
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analog MFC test interface — NI-DAQ AO setpoint + AI readback.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--hardware", type=Path, default=_DEFAULT_HW,
        help=f"Path to hardware.yaml (default: {_DEFAULT_HW})",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print config and exit without touching hardware",
    )

    sub = parser.add_subparsers(dest="command", required=True)

    # ── monitor ──────────────────────────────────────────────────────────────
    mon = sub.add_parser("monitor", help="Live setpoint + feedback display")
    mon.add_argument(
        "--set", nargs="*", default=[],
        metavar="NAME=V",
        help="Initial setpoints, e.g. --set air_left=2.5 odor_right=1.0",
    )
    mon.add_argument(
        "--interval", type=float, default=0.5,
        help="Polling interval in seconds (default: 0.5)",
    )
    mon.add_argument(
        "--no-zero-on-exit", dest="zero_on_exit", action="store_false",
        help="Do NOT zero setpoints on Ctrl+C",
    )

    # ── sweep ─────────────────────────────────────────────────────────────────
    sw = sub.add_parser("sweep", help="Step through voltage levels and verify readback")
    sw.add_argument(
        "--levels", nargs="*", type=float,
        default=[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 0.0],
        help="Voltage levels to step through (default: 0 1 2 3 4 5 0)",
    )
    sw.add_argument(
        "--dwell", type=float, default=0.5,
        help="Seconds to hold each level (default: 0.5)",
    )
    sw.add_argument(
        "--tolerance", type=float, default=0.1,
        help="Max allowed |AO - AI| error in volts for PASS (default: 0.1)",
    )
    sw.add_argument(
        "--channels", nargs="*",
        choices=list(_AO_KEYS.keys()),
        default=None,
        help="Subset of channels to sweep (default: all four)",
    )

    ns = parser.parse_args()

    if not _NIDAQMX_AVAILABLE:
        print("ERROR: nidaqmx is not installed. Run: pip install nidaqmx")
        sys.exit(1)

    channels = load_mfc_channels(ns.hardware)

    print(f"Hardware: {ns.hardware}")
    print("MFC AO channels:")
    for n, p in channels.ao.items():
        print(f"  {n:<15} AO -> {p}  AI -> {channels.ai[n]}")

    if ns.dry_run:
        print("\n[dry-run] Hardware not touched.")
        return

    if ns.command == "monitor":
        setpoints_v = _parse_setpoints(ns.set)
        run_monitor(
            channels,
            setpoints_v=setpoints_v or None,
            interval_s=ns.interval,
            zero_on_exit=ns.zero_on_exit,
        )

    elif ns.command == "sweep":
        # Restrict to requested channels by temporarily filtering
        if ns.channels:
            channels = MFCChannels(
                ao={k: v for k, v in channels.ao.items() if k in ns.channels},
                ai={k: v for k, v in channels.ai.items() if k in ns.channels},
            )
        results = run_sweep(
            channels,
            levels_v=ns.levels,
            dwell_s=ns.dwell,
            tolerance_v=ns.tolerance,
        )
        sys.exit(0 if all(r.passed(ns.tolerance) for r in results) else 1)


if __name__ == "__main__":
    main()
