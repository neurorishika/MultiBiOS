#!/usr/bin/env python3
"""Probe all GPIO LineSource options on each Flea3 camera."""
import gc
import sys

try:
    import PySpin
except ImportError:
    sys.exit("PySpin not found")

system = PySpin.System.GetInstance()
cam_list = system.GetCameras()
num = cam_list.GetSize()
print(f"Found {num} camera(s).\n")

cams = []
for ci in range(num):
    cam = cam_list.GetByIndex(ci)
    cam.Init()
    cams.append(cam)

    tl = cam.GetTLDeviceNodeMap()
    mn = PySpin.CStringPtr(tl.GetNode("DeviceModelName"))
    sn = PySpin.CStringPtr(tl.GetNode("DeviceSerialNumber"))
    print(f"=== Camera {ci}: {mn.GetValue()} [S/N {sn.GetValue()}] ===")

    nm = cam.GetNodeMap()
    sel = PySpin.CEnumerationPtr(nm.GetNode("LineSelector"))

    for line_name in ["Line0", "Line1", "Line2", "Line3"]:
        entry = sel.GetEntryByName(line_name)
        if not PySpin.IsReadable(entry):
            print(f"\n  {line_name}: not available")
            continue
        sel.SetIntValue(entry.GetValue())

        mode_node = PySpin.CEnumerationPtr(nm.GetNode("LineMode"))
        mode_val = mode_node.GetCurrentEntry().GetSymbolic() if PySpin.IsReadable(mode_node) else "?"
        mode_entries = []
        if PySpin.IsReadable(mode_node):
            for e in mode_node.GetEntries():
                me = PySpin.CEnumEntryPtr(e)
                if PySpin.IsReadable(me):
                    mode_entries.append(me.GetSymbolic())

        src_node = PySpin.CEnumerationPtr(nm.GetNode("LineSource"))
        src_val = "?"
        src_entries = []
        if PySpin.IsReadable(src_node):
            src_val = src_node.GetCurrentEntry().GetSymbolic()
            for e in src_node.GetEntries():
                se = PySpin.CEnumEntryPtr(e)
                if PySpin.IsReadable(se):
                    src_entries.append(se.GetSymbolic())

        print(f"\n  {line_name}:")
        print(f"    Mode:    {mode_val}  (available: {mode_entries})")
        print(f"    Source:  {src_val}  (available: {src_entries})")
        print(f"    Writable: mode={PySpin.IsWritable(mode_node)}, source={PySpin.IsWritable(src_node)}")

    # Try setting Line2 source to ExposureActive and read back
    sel.SetIntValue(sel.GetEntryByName("Line2").GetValue())
    src_node = PySpin.CEnumerationPtr(nm.GetNode("LineSource"))
    if PySpin.IsWritable(src_node):
        ea = src_node.GetEntryByName("ExposureActive")
        if PySpin.IsReadable(ea):
            src_node.SetIntValue(ea.GetValue())
            rb = src_node.GetCurrentEntry().GetSymbolic()
            print(f"\n  Line2 → set ExposureActive → readback: {rb}")
        else:
            print("\n  ExposureActive entry not readable for Line2!")
            # Try all writable entries
            print("  Trying all readable entries on Line2:")
            for e in src_node.GetEntries():
                se = PySpin.CEnumEntryPtr(e)
                if PySpin.IsReadable(se):
                    try:
                        src_node.SetIntValue(se.GetValue())
                        rb = src_node.GetCurrentEntry().GetSymbolic()
                        print(f"    {se.GetSymbolic():30s} → readback: {rb}")
                    except Exception as ex:
                        print(f"    {se.GetSymbolic():30s} → FAILED: {ex}")
    else:
        print("\n  Line2 LineSource NOT writable!")
    print()

# Cleanup
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
print("Done.")
