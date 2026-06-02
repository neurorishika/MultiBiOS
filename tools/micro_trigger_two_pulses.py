#!/usr/bin/env python3
"""Send two microscope trigger pulses: one immediately, one 10s later.

Default output line is TRIG_MICRO from config/hardware.yaml.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import nidaqmx
import yaml


def resolve_hardware_path(raw_path: str) -> Path:
    hw_path = Path(raw_path)
    if hw_path.exists():
        return hw_path
    script_dir = Path(__file__).parent.absolute()
    candidate = script_dir.parent / raw_path
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"Hardware config not found at {hw_path}")


def load_trigger_line(hardware_path: Path) -> str:
    config = yaml.safe_load(hardware_path.read_text(encoding="utf-8")) or {}
    digital_outputs = config.get("digital_outputs") or {}
    trigger_line = digital_outputs.get("TRIG_MICRO")
    if not trigger_line:
        raise KeyError("TRIG_MICRO not found in hardware config")
    return str(trigger_line)


def pulse(task: nidaqmx.Task, pulse_width_s: float, label: str) -> None:
    task.write(True)
    time.sleep(pulse_width_s)
    task.write(False)
    print(label)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Send one microscope trigger pulse now and another after 10 seconds."
    )
    parser.add_argument(
        "--hardware",
        default="config/hardware.yaml",
        help="Path to hardware.yaml (default: config/hardware.yaml)",
    )
    parser.add_argument(
        "--delay-s",
        type=float,
        default=10.0,
        help="Delay between pulse starts in seconds (default: 10.0)",
    )
    parser.add_argument(
        "--pulse-width-ms",
        type=float,
        default=10.0,
        help="Pulse width in milliseconds (default: 10.0)",
    )
    args = parser.parse_args()

    hardware_path = resolve_hardware_path(args.hardware)
    line = load_trigger_line(hardware_path)
    pulse_width_s = max(0.001, args.pulse_width_ms / 1000.0)

    print(f"Two-pulse test on {line}")

    with nidaqmx.Task("MICRO_TRIGGER_TWO_PULSES") as task:
        task.do_channels.add_do_chan(line)
        task.write(False)

        t0 = time.perf_counter()
        pulse(task, pulse_width_s, "Pulse 1 sent at t=0s")

        remaining = args.delay_s - (time.perf_counter() - t0)
        if remaining > 0:
            time.sleep(remaining)

        pulse(task, pulse_width_s, f"Pulse 2 sent at t={args.delay_s:g}s")

    print("Done")


if __name__ == "__main__":
    main()