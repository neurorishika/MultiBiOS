#!/usr/bin/env python3
"""Verify Blackfly return-line visibility on NI-DAQ digital inputs.

This test toggles a selected camera GPIO line using the camera's internal
UserOutput signal and reads the configured NI-DAQ digital input lines at the
same time. It is intended to answer a narrow bench question:

  does the camera return wire actually produce a DAQ-visible signal?

For the BFS-U3-13Y3M cameras on this rig, the vendor documentation and live
camera readback indicate:

- Line1 = white wire = opto-coupled output
- Line2 = red wire = open-drain GPIO / optional 3.3 V rail

If you are testing the white wire, use --line line1 and make sure the blue
Opto GND wire is wired into the measurement circuit.
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

    candidate = Path(__file__).parent.absolute().parent / raw_path
    if candidate.exists():
        return candidate

    raise SystemExit(f"Hardware config not found at {hw_path}")


def read_daq_lines(di_task) -> tuple[bool, bool]:
    values = di_task.read()
    return bool(values[0]), bool(values[1])


def configure_camera_user_output(nm, line_name: str) -> tuple[str, str]:
    import PySpin

    line_to_source = {
        "line1": ("Line1", "UserOutput1", "UserOutput1"),
        "line2": ("Line2", "UserOutput2", "UserOutput2"),
    }
    camera_line_name, line_source_name, selector_name = line_to_source[line_name]

    line_selector = PySpin.CEnumerationPtr(nm.GetNode("LineSelector"))
    line_mode = PySpin.CEnumerationPtr(nm.GetNode("LineMode"))
    line_source = PySpin.CEnumerationPtr(nm.GetNode("LineSource"))
    user_output_selector = PySpin.CEnumerationPtr(nm.GetNode("UserOutputSelector"))
    user_output_value = PySpin.CBooleanPtr(nm.GetNode("UserOutputValue"))
    line_format = PySpin.CEnumerationPtr(nm.GetNode("LineFormat"))

    line_entry = line_selector.GetEntryByName(camera_line_name)
    if not PySpin.IsReadable(line_entry):
        raise RuntimeError(f"{camera_line_name} is not available on this camera.")
    line_selector.SetIntValue(line_entry.GetValue())

    output_entry = line_mode.GetEntryByName("Output")
    if not PySpin.IsReadable(output_entry) or not PySpin.IsWritable(line_mode):
        raise RuntimeError(f"{camera_line_name} cannot be set to Output mode.")
    line_mode.SetIntValue(output_entry.GetValue())

    source_entry = line_source.GetEntryByName(line_source_name)
    if not PySpin.IsReadable(source_entry) or not PySpin.IsWritable(line_source):
        raise RuntimeError(f"{line_source_name} is not available as a LineSource.")
    line_source.SetIntValue(source_entry.GetValue())

    selector_entry = user_output_selector.GetEntryByName(selector_name)
    if not PySpin.IsReadable(selector_entry) or not PySpin.IsWritable(user_output_selector):
        raise RuntimeError(f"{selector_name} is not available in UserOutputSelector.")
    user_output_selector.SetIntValue(selector_entry.GetValue())
    if not PySpin.IsWritable(user_output_value):
        raise RuntimeError("UserOutputValue is not writable.")

    format_name = "n/a"
    if PySpin.IsReadable(line_format):
        current = line_format.GetCurrentEntry()
        if PySpin.IsReadable(current):
            format_name = current.GetSymbolic()

    return camera_line_name, format_name


def set_user_output(nm, value: bool) -> None:
    import PySpin

    node = PySpin.CBooleanPtr(nm.GetNode("UserOutputValue"))
    if not PySpin.IsWritable(node):
        raise RuntimeError("UserOutputValue is not writable.")
    node.SetValue(value)


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify Blackfly return-line visibility on NI-DAQ DI lines.")
    parser.add_argument("--hardware", default="config/hardware.yaml", help="Path to hardware.yaml")
    parser.add_argument("--line", choices=["line1", "line2"], default="line1", help="Camera GPIO line to drive. Use line1 for the white wire on BFS-U3-13Y3M.")
    parser.add_argument("--hold-ms", type=float, default=200.0, help="How long to hold each LOW/HIGH state before reading DAQ inputs.")
    args = parser.parse_args()

    try:
        import PySpin
    except ImportError as exc:
        raise SystemExit("PySpin not found. Use the multibios-blackfly environment.") from exc

    hw = load_hardware_config(resolve_hardware_path(args.hardware))
    digital_inputs = hw.get("digital_inputs") or {}
    front_line = digital_inputs.get("CAMERA_FRONT_O1")
    side_line = digital_inputs.get("CAMERA_SIDE_O1")
    if not front_line or not side_line:
        raise SystemExit("CAMERA_FRONT_O1 and CAMERA_SIDE_O1 must be defined in hardware.yaml.")

    print("Camera return-line verification")
    print(f"  Selected camera line: {args.line}")
    print(f"  DAQ DI front:         {front_line}")
    print(f"  DAQ DI side:          {side_line}")
    print("")
    if args.line == "line1":
        print("Expected wiring for white-wire test:")
        print("  white wire = Line1 opto output")
        print("  blue wire  = Opto GND return for that isolated output")
        print("  If blue is not wired into the measurement circuit, DAQ reads will stay flat.")
    else:
        print("Expected wiring for red-wire test:")
        print("  red wire = Line2 open-drain GPIO")
        print("  Open-drain outputs require an external pull-up to produce a logic high.")
    print("")

    di_task = nidaqmx.Task("VERIFY_CAMERA_RETURN_LINE")
    di_task.di_channels.add_di_chan(
        f"{front_line},{side_line}",
        line_grouping=LineGrouping.CHAN_PER_LINE,
    )

    system = PySpin.System.GetInstance()
    cam_list = system.GetCameras()
    cams = []
    nodemaps = []
    cam = None
    nm = None

    try:
        num = cam_list.GetSize()
        if num == 0:
            raise SystemExit("No cameras found.")

        baseline = read_daq_lines(di_task)
        print(f"Baseline DAQ state: FRONT={baseline[0]}, SIDE={baseline[1]}")

        for index in range(min(num, 2)):
            cam = cam_list.GetByIndex(index)
            cam.Init()
            cams.append(cam)
            nm = cam.GetNodeMap()
            nodemaps.append(nm)

            camera_line_name, format_name = configure_camera_user_output(nm, args.line)
            print("")
            print(f"Camera {index} driving {camera_line_name} [{format_name}]")

            set_user_output(nm, False)
            time.sleep(args.hold_ms / 1000.0)
            front_low, side_low = read_daq_lines(di_task)

            set_user_output(nm, True)
            time.sleep(args.hold_ms / 1000.0)
            front_high, side_high = read_daq_lines(di_task)

            set_user_output(nm, False)
            time.sleep(args.hold_ms / 1000.0)
            front_low_2, side_low_2 = read_daq_lines(di_task)

            print(f"  LOW  -> FRONT={front_low}, SIDE={side_low}")
            print(f"  HIGH -> FRONT={front_high}, SIDE={side_high}")
            print(f"  LOW  -> FRONT={front_low_2}, SIDE={side_low_2}")

            front_changed = (front_high != front_low) or (front_high != front_low_2)
            side_changed = (side_high != side_low) or (side_high != side_low_2)
            if front_changed:
                print("  >>> FRONT return line changed at the DAQ <<<")
            if side_changed:
                print("  >>> SIDE return line changed at the DAQ <<<")
            if not front_changed and not side_changed:
                print("  No DAQ-visible change detected.")
                if args.line == "line1":
                    print("  Likely causes: missing blue Opto GND reference, isolated-output bias path missing, or wire not landed on the expected pin.")
                else:
                    print("  Likely causes: missing pull-up on the open-drain line, or wire not landed on the expected pin.")
    finally:
        di_task.close()
        cam = None
        nm = None
        for idx in range(len(cams)):
            try:
                cams[idx].DeInit()
            except Exception:
                pass
            cams[idx] = None
        nodemaps.clear()
        cams.clear()
        cam_list.Clear()
        system.ReleaseInstance()


if __name__ == "__main__":
    main()