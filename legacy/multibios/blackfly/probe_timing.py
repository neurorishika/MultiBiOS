#!/usr/bin/env python3
"""Probe Flea3 camera timing parameters to diagnose trigger-rate limits.

Reads back all relevant timing/trigger/ROI settings from each camera and
computes the theoretical max trigger rate.  Optionally sweeps exposure times
to find the exact crossover point.

Usage:
    conda activate multibios-blackfly
    python -m multibios.blackfly.legacy.probe_timing
    python -m multibios.blackfly.legacy.probe_timing --sweep
    python -m multibios.blackfly.legacy.probe_timing --test-overlap
"""

from __future__ import annotations

import argparse
import sys

try:
    import PySpin
except ImportError:
    sys.exit("PySpin not found. Activate multibios-blackfly environment.")


def _read_enum(nm, name: str) -> str:
    node = PySpin.CEnumerationPtr(nm.GetNode(name))
    if not PySpin.IsReadable(node):
        return "<not readable>"
    entry = node.GetCurrentEntry()
    return entry.GetSymbolic() if PySpin.IsReadable(entry) else "<no entry>"


def _read_float(nm, name: str) -> str:
    node = PySpin.CFloatPtr(nm.GetNode(name))
    if not PySpin.IsReadable(node):
        return "<not readable>"
    return f"{node.GetValue():.2f}  (min={node.GetMin():.2f}, max={node.GetMax():.2f})"


def _read_int(nm, name: str) -> str:
    node = PySpin.CIntegerPtr(nm.GetNode(name))
    if not PySpin.IsReadable(node):
        return "<not readable>"
    return f"{node.GetValue()}  (min={node.GetMin()}, max={node.GetMax()}, inc={node.GetInc()})"


def _read_bool(nm, name: str) -> str:
    node = PySpin.CBooleanPtr(nm.GetNode(name))
    if not PySpin.IsReadable(node):
        return "<not readable>"
    return str(node.GetValue())


def _enum_entries(nm, name: str) -> list:
    """Return list of symbolic entry names available for an enum node."""
    node = PySpin.CEnumerationPtr(nm.GetNode(name))
    if not PySpin.IsReadable(node):
        return []
    entries = node.GetEntries()
    result = []
    for e in entries:
        entry = PySpin.CEnumEntryPtr(e)
        if PySpin.IsReadable(entry):
            result.append(entry.GetSymbolic())
    return result


def report_camera(cam, idx: int) -> None:
    nm = cam.GetNodeMap()
    tl = cam.GetTLDeviceNodeMap()

    model_n = PySpin.CStringPtr(tl.GetNode("DeviceModelName"))
    sn_n = PySpin.CStringPtr(tl.GetNode("DeviceSerialNumber"))
    model = model_n.GetValue() if PySpin.IsReadable(model_n) else "?"
    sn = sn_n.GetValue() if PySpin.IsReadable(sn_n) else "?"

    print(f"\n{'='*60}")
    print(f"  Camera {idx}: {model}  [S/N {sn}]")
    print(f"{'='*60}")

    print("\n--- Image Format ---")
    for name in ["Width", "Height", "OffsetX", "OffsetY"]:
        print(f"  {name:30s} = {_read_int(nm, name)}")
    for name in ["PixelFormat", "PixelSize"]:
        print(f"  {name:30s} = {_read_enum(nm, name)}")
    for name in ["BinningHorizontal", "BinningVertical",
                  "DecimationHorizontal", "DecimationVertical"]:
        print(f"  {name:30s} = {_read_int(nm, name)}")

    print("\n--- Acquisition / Frame Rate ---")
    print(f"  {'AcquisitionMode':30s} = {_read_enum(nm, 'AcquisitionMode')}")
    for name in ["AcquisitionFrameRateEnable", "AcquisitionFrameRateEnabled"]:
        print(f"  {name:30s} = {_read_bool(nm, name)}")
    for name in ["AcquisitionFrameRate", "AcquisitionFrameRateAbs",
                  "AcquisitionResultingFrameRate"]:
        print(f"  {name:30s} = {_read_float(nm, name)}")

    print("\n--- Exposure ---")
    print(f"  {'ExposureAuto':30s} = {_read_enum(nm, 'ExposureAuto')}")
    print(f"  {'ExposureMode':30s} = {_read_enum(nm, 'ExposureMode')}")
    for name in ["ExposureTime", "ExposureTimeAbs"]:
        print(f"  {name:30s} = {_read_float(nm, name)}")

    print("\n--- Trigger ---")
    print(f"  {'TriggerMode':30s} = {_read_enum(nm, 'TriggerMode')}")
    print(f"  {'TriggerSelector':30s} = {_read_enum(nm, 'TriggerSelector')}")
    print(f"  {'TriggerSource':30s} = {_read_enum(nm, 'TriggerSource')}")
    print(f"  {'TriggerActivation':30s} = {_read_enum(nm, 'TriggerActivation')}")
    for name in ["TriggerDelay", "TriggerDelayAbs"]:
        print(f"  {name:30s} = {_read_float(nm, name)}")

    # TriggerOverlap — this is the critical one
    overlap_val = _read_enum(nm, "TriggerOverlap")
    overlap_entries = _enum_entries(nm, "TriggerOverlap")
    print(f"  {'TriggerOverlap':30s} = {overlap_val}")
    print(f"  {'  available entries':30s} = {overlap_entries}")

    print("\n--- Link Throughput ---")
    print(f"  {'DeviceLinkThroughputLimitMode':30s} = {_read_enum(nm, 'DeviceLinkThroughputLimitMode')}")
    for name in ["DeviceLinkThroughputLimit", "DeviceLinkSpeed",
                  "StreamBytesPerSecond"]:
        val = _read_int(nm, name)
        if val == "<not readable>":
            val = _read_float(nm, name)
        print(f"  {name:30s} = {val}")

    print("\n--- GPIO / Output ---")
    for line in ["Line0", "Line1", "Line2", "Line3"]:
        # Select the line first
        sel_node = PySpin.CEnumerationPtr(nm.GetNode("LineSelector"))
        if PySpin.IsWritable(sel_node):
            entry = sel_node.GetEntryByName(line)
            if PySpin.IsReadable(entry):
                sel_node.SetIntValue(entry.GetValue())
                mode = _read_enum(nm, "LineMode")
                source = _read_enum(nm, "LineSource")
                print(f"  {line:30s} = mode={mode}, source={source}")

    # Timing estimate
    print("\n--- Timing Estimate ---")
    w_node = PySpin.CIntegerPtr(nm.GetNode("Width"))
    h_node = PySpin.CIntegerPtr(nm.GetNode("Height"))
    exp_node = PySpin.CFloatPtr(nm.GetNode("ExposureTime"))
    if not PySpin.IsReadable(exp_node):
        exp_node = PySpin.CFloatPtr(nm.GetNode("ExposureTimeAbs"))

    if PySpin.IsReadable(h_node) and PySpin.IsReadable(exp_node):
        height = h_node.GetValue()
        exp_us = exp_node.GetValue()
        exp_ms = exp_us / 1000.0
        # Estimate readout from the 28 Hz observation:
        # At 28 Hz with 5ms exposure → 35.7ms total → readout ≈ 30.7ms at 1552 rows
        # Readout per row ≈ 30.7ms / 1552 ≈ 19.8 µs/row
        READOUT_PER_ROW_US = 19.8  # estimated from 28Hz@5ms@1552rows
        readout_ms = height * READOUT_PER_ROW_US / 1000.0
        no_overlap_ms = exp_ms + readout_ms
        overlap_ms = max(exp_ms, readout_ms)

        print(f"  Current height:       {height} rows")
        print(f"  Current exposure:     {exp_ms:.1f} ms")
        print(f"  Estimated readout:    {readout_ms:.1f} ms  ({READOUT_PER_ROW_US:.1f} µs/row)")
        print(f"  Frame time (no overlap):  {no_overlap_ms:.1f} ms → max {1000/no_overlap_ms:.1f} Hz")
        print(f"  Frame time (overlap):     {overlap_ms:.1f} ms → max {1000/overlap_ms:.1f} Hz")
        print(f"  TriggerOverlap = {overlap_val}")

        if overlap_val == "Off" or overlap_val == "<not readable>":
            print(f"\n  >>> WITHOUT overlap, max trigger rate ≈ {1000/no_overlap_ms:.0f} Hz <<<")
            print(f"  To reach 60 Hz (no overlap):")
            target_ms = 1000.0 / 60.0
            # max_height at 1ms exposure
            for test_exp in [1.0, 2.0, 5.0]:
                avail_readout = target_ms - test_exp
                if avail_readout > 0:
                    max_rows = int(avail_readout * 1000 / READOUT_PER_ROW_US)
                    print(f"    exposure={test_exp}ms → need height ≤ {max_rows} rows")
                else:
                    print(f"    exposure={test_exp}ms → impossible even with 0 rows!")
        else:
            print(f"\n  >>> WITH overlap, max trigger rate ≈ {1000/overlap_ms:.0f} Hz <<<")


def test_overlap_entries(cam, idx: int) -> None:
    """Try setting TriggerOverlap to every available entry and report success."""
    nm = cam.GetNodeMap()
    entries = _enum_entries(nm, "TriggerOverlap")
    print(f"\n--- Camera {idx}: Testing TriggerOverlap entries ---")
    print(f"  Available: {entries}")

    for entry_name in entries:
        node = PySpin.CEnumerationPtr(nm.GetNode("TriggerOverlap"))
        if not PySpin.IsWritable(node):
            print(f"  {entry_name:20s} → node NOT writable")
            continue
        entry = node.GetEntryByName(entry_name)
        if not PySpin.IsReadable(entry):
            print(f"  {entry_name:20s} → entry NOT readable")
            continue
        try:
            node.SetIntValue(entry.GetValue())
            # Read back
            readback = node.GetCurrentEntry().GetSymbolic()
            print(f"  {entry_name:20s} → SET OK, readback = {readback}")
        except PySpin.SpinnakerException as exc:
            print(f"  {entry_name:20s} → FAILED: {exc}")


def main():
    parser = argparse.ArgumentParser(description="Probe Flea3 timing parameters")
    parser.add_argument("--test-overlap", action="store_true",
                        help="Test all TriggerOverlap entries")
    parser.add_argument("--camera", type=int, default=None,
                        help="Probe only camera index N (default: all)")
    args = parser.parse_args()

    system = PySpin.System.GetInstance()
    cam_list = system.GetCameras()
    num = cam_list.GetSize()

    if num == 0:
        cam_list.Clear()
        system.ReleaseInstance()
        sys.exit("No cameras found.")

    print(f"Found {num} camera(s).")
    indices = [args.camera] if args.camera is not None else range(num)

    cams = []
    for i in indices:
        cam = cam_list.GetByIndex(i)
        cam.Init()
        cams.append((i, cam))

    try:
        for idx in range(len(cams)):
            i, cam = cams[idx]
            report_camera(cam, i)
            if args.test_overlap:
                test_overlap_entries(cam, i)
    finally:
        import gc
        for idx in range(len(cams)):
            i, cam = cams[idx]
            try:
                cam.DeInit()
            except Exception:
                pass
            cams[idx] = (i, None)
        del cam
        gc.collect()
        cams.clear()
        cam_list.Clear()
        system.ReleaseInstance()

    print("\nDone.")


if __name__ == "__main__":
    main()
