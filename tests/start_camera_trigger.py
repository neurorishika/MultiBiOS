#!/usr/bin/env python3
"""Send a single trigger pulse to start cameras in AcquisitionStart mode.

Usage:
    python tests/start_camera_trigger.py                # single start pulse
    python tests/start_camera_trigger.py --stop         # send stop pulse

In AcquisitionStart mode the cameras free-run at their configured fps
after receiving one rising edge.  This script sends that single pulse.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import nidaqmx
import yaml
from nidaqmx.constants import LineGrouping


def load_hardware_config(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def resolve_hardware_path(raw_path: str) -> Path:
    hw_path = Path(raw_path)
    if hw_path.exists():
        return hw_path
    candidate = Path(__file__).parent.absolute().parent / raw_path
    if candidate.exists():
        return candidate
    sys.exit(f"Hardware config not found at {hw_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Send a single trigger pulse to start/stop camera acquisition.")
    parser.add_argument("--hardware", default="config/hardware.yaml",
                        help="Path to hardware.yaml")
    parser.add_argument("--pulse-ms", type=float, default=5.0,
                        help="Pulse width in ms (default: 5)")
    args = parser.parse_args()

    hw_path = resolve_hardware_path(args.hardware)
    hw = load_hardware_config(hw_path)
    trig_line = (hw.get("digital_outputs") or {}).get("TRIG_CAMERA")
    if not trig_line:
        sys.exit("TRIG_CAMERA not found in hardware config.")

    print(f"Sending single trigger pulse on {trig_line} ({args.pulse_ms} ms) ...")

    with nidaqmx.Task("START_CAMERA_TRIGGER") as task:
        task.do_channels.add_do_chan(
            trig_line,
            line_grouping=LineGrouping.CHAN_PER_LINE,
        )
        # Rising edge → hold → falling edge
        task.write(True)
        time.sleep(args.pulse_ms / 1000.0)
        task.write(False)

    print("Done — cameras should now be free-running.")
    print("To stop: close SpinView or your Blackfly acquisition app, or re-configure with python -m multibios.blackfly.setup_daq_mode")


if __name__ == "__main__":
    main()
