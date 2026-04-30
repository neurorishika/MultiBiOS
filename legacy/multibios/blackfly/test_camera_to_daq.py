#!/usr/bin/env python3
"""Combined Camera GPIO + DAQ readback test.

Toggles the camera Line1 and Line2 via UserOutputSelector/UserOutputValue
while simultaneously reading the DAQ digital inputs (CAMERA_FRONT_O1 and
CAMERA_SIDE_O1) to verify the physical wiring from camera GPIO to DAQ DI.

This determines:
  1. Whether camera Line2 output reaches the DAQ
  2. Which camera line is wired to which DAQ input
  3. Whether pull-up resistors are needed (open-drain output)

Usage:
    conda activate multibios-blackfly
    python -m multibios.blackfly.legacy.test_camera_to_daq
"""

from __future__ import annotations

import gc
import sys
import time

try:
    import PySpin
except ImportError:
    sys.exit("PySpin not found. Use multibios-blackfly environment.")

try:
    import nidaqmx
    from nidaqmx.constants import LineGrouping
except ImportError:
    sys.exit("nidaqmx not found. Install with: pip install nidaqmx")


# DAQ lines from hardware.yaml
DAQ_FRONT = "Dev1/port0/line29"  # CAMERA_FRONT_O1 (Pin 125)
DAQ_SIDE  = "Dev1/port0/line27"  # CAMERA_SIDE_O1 (Pin 123)


def read_daq(task) -> tuple:
    vals = task.read()
    return vals[0], vals[1]


def test_camera_gpio(cam, cam_idx: int, nm, di_task) -> None:
    """Test each output line on one camera by toggling UserOutput."""
    sel_node = PySpin.CEnumerationPtr(nm.GetNode("LineSelector"))
    uos_node = PySpin.CEnumerationPtr(nm.GetNode("UserOutputSelector"))
    uov_node = PySpin.CBooleanPtr(nm.GetNode("UserOutputValue"))

    if not PySpin.IsWritable(uos_node) or not PySpin.IsWritable(uov_node):
        print(f"  Camera {cam_idx}: UserOutputSelector/Value not writable!")
        return

    # On Blackfly S, LineSource uses "UserOutput1" etc., and
    # UserOutputSelector entries are named "UserOutput0".."UserOutput3".
    line_to_uo = {
        "Line1": ("UserOutput1", "UserOutput1"),
        "Line2": ("UserOutput2", "UserOutput2"),
    }

    for line_name, (src_name, uo_sel_name) in line_to_uo.items():
        # Select this line
        line_entry = sel_node.GetEntryByName(line_name)
        if not PySpin.IsReadable(line_entry):
            print(f"  Camera {cam_idx} {line_name}: not available, skip.")
            continue
        sel_node.SetIntValue(line_entry.GetValue())

        # Set line mode to Output
        mode_node = PySpin.CEnumerationPtr(nm.GetNode("LineMode"))
        out_entry = mode_node.GetEntryByName("Output")
        if not (PySpin.IsReadable(out_entry) and PySpin.IsWritable(mode_node)):
            print(f"  Camera {cam_idx} {line_name}: cannot set Output mode, skip.")
            continue
        mode_node.SetIntValue(out_entry.GetValue())

        # Set LineSource to UserOutput<N>
        src_node = PySpin.CEnumerationPtr(nm.GetNode("LineSource"))
        src_entry = src_node.GetEntryByName(src_name)
        if not (PySpin.IsReadable(src_entry) and PySpin.IsWritable(src_node)):
            print(f"  Camera {cam_idx} {line_name}: {src_name} not available as LineSource, skip.")
            continue
        src_node.SetIntValue(src_entry.GetValue())

        # Select the UserOutput in the selector
        uo_entry = uos_node.GetEntryByName(uo_sel_name)
        if not PySpin.IsReadable(uo_entry):
            print(f"  Camera {cam_idx}: {uo_sel_name} not in UserOutputSelector, skip.")
            continue
        uos_node.SetIntValue(uo_entry.GetValue())

        print(f"\n  --- Camera {cam_idx} {line_name} (src={src_name}, sel={uo_sel_name}) ---")

        # Toggle LOW
        uov_node.SetValue(False)
        time.sleep(0.15)
        front_lo, side_lo = read_daq(di_task)

        # Toggle HIGH
        uov_node.SetValue(True)
        time.sleep(0.15)
        front_hi, side_hi = read_daq(di_task)

        # Toggle LOW again
        uov_node.SetValue(False)
        time.sleep(0.15)
        front_lo2, side_lo2 = read_daq(di_task)

        print(f"    LOW  -> FRONT={front_lo}, SIDE={side_lo}")
        print(f"    HIGH -> FRONT={front_hi}, SIDE={side_hi}")
        print(f"    LOW  -> FRONT={front_lo2}, SIDE={side_lo2}")

        front_changed = (front_hi != front_lo) or (front_hi != front_lo2)
        side_changed = (side_hi != side_lo) or (side_hi != side_lo2)
        if front_changed:
            print(f"    >>> Camera {cam_idx} {line_name} -> CAMERA_FRONT_O1 (line29) <<<")
        if side_changed:
            print(f"    >>> Camera {cam_idx} {line_name} -> CAMERA_SIDE_O1 (line27) <<<")
        if not front_changed and not side_changed:
            print(f"    NO change detected on DAQ inputs.")
            print(f"    (may need pull-up, or this line not wired to DAQ)")

        # Also read LineStatusAll for reference
        lsa = PySpin.CIntegerPtr(nm.GetNode("LineStatusAll"))
        if PySpin.IsReadable(lsa):
            val = lsa.GetValue()
            print(f"    LineStatusAll = {val} (0b{val:04b})")


def main():
    system = PySpin.System.GetInstance()
    cam_list = system.GetCameras()
    num = cam_list.GetSize()
    if num == 0:
        cam_list.Clear()
        system.ReleaseInstance()
        sys.exit("No cameras found.")

    print(f"Found {num} camera(s).\n")

    # Setup DAQ DI task
    di_task = nidaqmx.Task("CAM_GPIO_READBACK")
    di_task.di_channels.add_di_chan(
        f"{DAQ_FRONT},{DAQ_SIDE}",
        line_grouping=LineGrouping.CHAN_PER_LINE,
    )

    baseline = read_daq(di_task)
    print(f"Baseline DAQ: FRONT={baseline[0]}, SIDE={baseline[1]}")

    cams = []
    nodemaps = []
    cam = None
    nm = None
    for i in range(min(num, 2)):
        cam = cam_list.GetByIndex(i)
        cam.Init()
        cams.append(cam)
        nm = cam.GetNodeMap()
        nodemaps.append(nm)
        test_camera_gpio(cam, i, nm, di_task)

    di_task.close()

    # Cleanup - release nodemaps first, then cameras
    for idx in range(len(nodemaps)):
        nodemaps[idx] = None
    nodemaps.clear()
    cam = None
    nm = None
    for idx in range(len(cams)):
        try:
            cams[idx].DeInit()
        except Exception:
            pass
        cams[idx] = None
    gc.collect()
    cams.clear()
    cam_list.Clear()
    system.ReleaseInstance()
    print("\nDone.")


if __name__ == "__main__":
    main()
