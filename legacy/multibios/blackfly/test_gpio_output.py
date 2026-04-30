#!/usr/bin/env python3
"""Test Blackfly GPIO output by toggling UserOutput on Line1 and Line2.

This script sets Line2 (and optionally Line1) to UserOutput mode and
toggles it on/off so you can see the signal on a scope or DAQ DI.

This helps determine:
  1. Whether Line2 is physically connected to the DAQ return wires
  2. Whether the output driver works (or needs a pull-up)
  3. Which physical pin corresponds to which logical line

Usage:
    conda activate multibios-blackfly
    python -m multibios.blackfly.legacy.test_gpio_output
    python -m multibios.blackfly.legacy.test_gpio_output --line Line1
    python -m multibios.blackfly.legacy.test_gpio_output --both
"""

from __future__ import annotations

import argparse
import gc
import sys
import time

try:
    import PySpin
except ImportError:
    sys.exit("PySpin not found")


def toggle_test(cam, cam_idx: int, lines: list, period: float = 0.5) -> None:
    nm = cam.GetNodeMap()
    sel = PySpin.CEnumerationPtr(nm.GetNode("LineSelector"))

    # Configure each line for UserOutput
    for line_name in lines:
        entry = sel.GetEntryByName(line_name)
        if not PySpin.IsReadable(entry):
            print(f"  Camera {cam_idx}: {line_name} not available, skipping.")
            continue
        sel.SetIntValue(entry.GetValue())

        mode = PySpin.CEnumerationPtr(nm.GetNode("LineMode"))
        out_entry = mode.GetEntryByName("Output")
        if PySpin.IsReadable(out_entry) and PySpin.IsWritable(mode):
            mode.SetIntValue(out_entry.GetValue())
        else:
            print(f"  Camera {cam_idx}: {line_name} cannot be set to Output.")
            continue

        src = PySpin.CEnumerationPtr(nm.GetNode("LineSource"))
        # UserOutput1 for Line1, UserOutput2 for Line2
        uo_name = f"UserOutput{line_name[-1]}"
        uo_entry = src.GetEntryByName(uo_name)
        if PySpin.IsReadable(uo_entry) and PySpin.IsWritable(src):
            src.SetIntValue(uo_entry.GetValue())
            print(f"  Camera {cam_idx}: {line_name} → {uo_name}")
        else:
            print(f"  Camera {cam_idx}: {line_name} cannot use {uo_name}.")

    return lines


def set_user_output(nm, output_name: str, value: bool) -> bool:
    node = PySpin.CBooleanPtr(nm.GetNode(output_name))
    if PySpin.IsWritable(node):
        node.SetValue(value)
        return True
    return False


def main():
    parser = argparse.ArgumentParser(description="Toggle Flea3 GPIO for wire testing")
    parser.add_argument("--line", default="Line2", help="Which line to toggle (default: Line2)")
    parser.add_argument("--both", action="store_true", help="Toggle both Line1 and Line2")
    parser.add_argument("--period", type=float, default=0.5, help="Toggle period in seconds (default: 0.5)")
    parser.add_argument("--camera", type=int, default=None, help="Camera index (default: all)")
    args = parser.parse_args()

    lines = ["Line1", "Line2"] if args.both else [args.line]

    system = PySpin.System.GetInstance()
    cam_list = system.GetCameras()
    num = cam_list.GetSize()
    if num == 0:
        cam_list.Clear()
        system.ReleaseInstance()
        sys.exit("No cameras found.")

    indices = [args.camera] if args.camera is not None else list(range(num))
    cams = []
    for i in indices:
        cam = cam_list.GetByIndex(i)
        cam.Init()
        cams.append((i, cam))
        toggle_test(cam, i, lines)

    # Build list of UserOutput names to toggle
    uo_names = []
    for line_name in lines:
        uo_names.append(f"UserOutput{line_name[-1]}")

    print(f"\nToggling {lines} at {1/args.period:.1f} Hz on camera(s) {indices}")
    print("Watch scope/DAQ for signal. Press Ctrl+C to stop.\n")

    try:
        state = False
        while True:
            state = not state
            for i, cam in cams:
                nm = cam.GetNodeMap()
                for uo in uo_names:
                    set_user_output(nm, uo, state)
            status = "HIGH" if state else "LOW"
            print(f"  {status}", end="\r")
            time.sleep(args.period)
    except KeyboardInterrupt:
        print("\nStopping...")
        # Set outputs low
        for i, cam in cams:
            nm = cam.GetNodeMap()
            for uo in uo_names:
                set_user_output(nm, uo, False)

    # Cleanup
    for idx in range(len(cams)):
        i, cam = cams[idx]
        try:
            cam.DeInit()
        except Exception:
            pass
        cams[idx] = (i, None)
    gc.collect()
    cams.clear()
    cam_list.Clear()
    system.ReleaseInstance()
    print("Done.")


if __name__ == "__main__":
    main()
