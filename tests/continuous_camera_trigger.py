#!/usr/bin/env python3
"""Continuously output camera trigger pulses on TRIG_CAMERA for Blackfly trigger testing.

Usage:
    python tests/continuous_camera_trigger.py --fps 30

This script continuously regenerates a single-period pulse train on the
TRIG_CAMERA digital output line defined in config/hardware.yaml.
Use it after configuring both Blackfly S cameras for external trigger mode.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import nidaqmx
import numpy as np
import yaml
from nidaqmx.constants import AcquisitionType, Edge, LineGrouping


def load_hardware_config(path: Path) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return yaml.safe_load(handle)
    except Exception as exc:
        print(f"Error loading hardware config from {path}: {exc}")
        sys.exit(1)


def resolve_hardware_path(raw_path: str) -> Path:
    hw_path = Path(raw_path)
    if hw_path.exists():
        return hw_path

    script_dir = Path(__file__).parent.absolute()
    candidate = script_dir.parent / raw_path
    if candidate.exists():
        return candidate

    print(f"Hardware config not found at {hw_path}")
    sys.exit(1)


def build_trigger_period(sample_rate: int, fps: float, pulse_ms: float) -> np.ndarray:
    period_s = 1.0 / fps
    total_samples = max(2, int(round(period_s * sample_rate)))
    pulse_samples = max(1, int(round(pulse_ms * sample_rate / 1000.0)))
    pulse_samples = min(pulse_samples, total_samples)

    waveform = np.zeros(total_samples, dtype=np.bool_)
    waveform[:pulse_samples] = True
    return waveform


def main() -> None:
    parser = argparse.ArgumentParser(description="Continuously output camera trigger pulses for Blackfly trigger testing.")
    parser.add_argument("--hardware", default="config/hardware.yaml", help="Path to hardware.yaml (default: config/hardware.yaml)")
    parser.add_argument("--fps", type=float, default=30.0, help="Trigger frequency in frames per second (default: 30)")
    parser.add_argument("--pulse-ms", type=float, default=5.0, help="Trigger pulse width in ms (default: 5)")
    parser.add_argument("--rate", type=int, default=10000, help="DAQ sample rate in Hz (default: 10000)")
    args = parser.parse_args()

    hw_path = resolve_hardware_path(args.hardware)
    print(f"Loading hardware from {hw_path}")
    hw = load_hardware_config(hw_path)

    trig_line = (hw.get("digital_outputs") or {}).get("TRIG_CAMERA")
    if not trig_line:
        print("TRIG_CAMERA not found in hardware config.")
        sys.exit(1)

    waveform = build_trigger_period(args.rate, args.fps, args.pulse_ms)
    period_ms = len(waveform) * 1000.0 / args.rate

    print("Continuous camera trigger test")
    print(f"  Line:        {trig_line}")
    print(f"  FPS:         {args.fps:.3f}")
    print(f"  Pulse width: {args.pulse_ms:.3f} ms")
    print(f"  Sample rate: {args.rate} Hz")
    print(f"  Period:      {period_ms:.3f} ms ({len(waveform)} samples)")
    print("")
    print("Before starting:")
    print("  1. Run python -m multibios.blackfly.setup_daq_mode to put both cameras in external-trigger mode.")
    print("  2. Open SpinView or your Blackfly acquisition app and arm the cameras for acquisition.")
    print("  3. Press Ctrl+C here to stop the trigger train.")

    try:
        with nidaqmx.Task("CONTINUOUS_CAMERA_TRIGGER") as task:
            task.do_channels.add_do_chan(
                trig_line,
                name_to_assign_to_lines="TRIG_CAMERA",
                line_grouping=LineGrouping.CHAN_PER_LINE,
            )
            task.timing.cfg_samp_clk_timing(
                rate=args.rate,
                active_edge=Edge.RISING,
                sample_mode=AcquisitionType.CONTINUOUS,
                samps_per_chan=len(waveform),
            )
            task.write(waveform, auto_start=False)
            print("Starting continuous trigger generation ...")
            task.start()
            while True:
                time.sleep(0.5)
    except KeyboardInterrupt:
        print("\nStopping continuous trigger generation ...")
    except Exception as exc:
        print(f"\nError: {exc}")
        raise
    finally:
        print("Done.")


if __name__ == "__main__":
    main()