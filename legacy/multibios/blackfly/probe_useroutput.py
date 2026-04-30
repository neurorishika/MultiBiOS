#!/usr/bin/env python3
"""Probe UserOutput and all GPIO-related nodes."""
import gc
import sys

try:
    import PySpin
except ImportError:
    sys.exit("PySpin not found")

system = PySpin.System.GetInstance()
cam_list = system.GetCameras()
cam = cam_list.GetByIndex(0)
cam.Init()
nm = cam.GetNodeMap()

print("=== UserOutput / GPIO nodes ===\n")

for name in ['UserOutput0', 'UserOutput1', 'UserOutput2', 'UserOutput3',
             'UserOutputValue', 'UserOutputSelector', 'LineInverter',
             'LineStatus', 'LineStatusAll',
             'UserOutputValue0', 'UserOutputValue1', 'UserOutputValue2',
             'UserSetSelector', 'UserSetDefault']:
    node = nm.GetNode(name)
    if node is None:
        print(f"  {name:30s} = <does not exist>")
        continue
    # Bool
    bn = PySpin.CBooleanPtr(node)
    if PySpin.IsReadable(bn):
        print(f"  {name:30s} = bool({bn.GetValue()})  writable={PySpin.IsWritable(bn)}")
        continue
    # Int
    tn = PySpin.CIntegerPtr(node)
    if PySpin.IsReadable(tn):
        print(f"  {name:30s} = int({tn.GetValue()})  writable={PySpin.IsWritable(tn)}")
        continue
    # Enum
    en = PySpin.CEnumerationPtr(node)
    if PySpin.IsReadable(en):
        cur = en.GetCurrentEntry()
        cur_name = cur.GetSymbolic() if PySpin.IsReadable(cur) else "?"
        entries = []
        for e in en.GetEntries():
            ee = PySpin.CEnumEntryPtr(e)
            if PySpin.IsReadable(ee):
                entries.append(ee.GetSymbolic())
        print(f"  {name:30s} = enum({cur_name})  entries={entries}  writable={PySpin.IsWritable(en)}")
        continue
    print(f"  {name:30s} = <exists but not readable>")

print("\n=== Line Status per-line ===\n")

sel = PySpin.CEnumerationPtr(nm.GetNode("LineSelector"))
for ln in ["Line0", "Line1", "Line2", "Line3"]:
    e = sel.GetEntryByName(ln)
    if not PySpin.IsReadable(e):
        continue
    sel.SetIntValue(e.GetValue())
    ls = PySpin.CBooleanPtr(nm.GetNode("LineStatus"))
    status = ls.GetValue() if PySpin.IsReadable(ls) else "?"
    inv = PySpin.CBooleanPtr(nm.GetNode("LineInverter"))
    inv_val = inv.GetValue() if PySpin.IsReadable(inv) else "?"
    print(f"  {ln}: status={status}, inverter={inv_val}")

# Try setting Line2 to UserOutput and toggling via UserOutputSelector
print("\n=== Attempting UserOutputSelector approach ===\n")
sel.SetIntValue(sel.GetEntryByName("Line2").GetValue())
src = PySpin.CEnumerationPtr(nm.GetNode("LineSource"))
# Set to ExposureActive first — which we know works
ea = src.GetEntryByName("ExposureActive")
if PySpin.IsReadable(ea) and PySpin.IsWritable(src):
    src.SetIntValue(ea.GetValue())
    print(f"  Line2 source set to ExposureActive")

# Now check UserOutputSelector
uos = PySpin.CEnumerationPtr(nm.GetNode("UserOutputSelector"))
if PySpin.IsReadable(uos):
    cur = uos.GetCurrentEntry()
    entries = []
    for e in uos.GetEntries():
        ee = PySpin.CEnumEntryPtr(e)
        if PySpin.IsReadable(ee):
            entries.append(ee.GetSymbolic())
    print(f"  UserOutputSelector = {cur.GetSymbolic()}  entries={entries}  writable={PySpin.IsWritable(uos)}")

    # Try selecting UserOutput2 then setting UserOutputValue
    uo2 = uos.GetEntryByName("UserOutput2")
    if PySpin.IsReadable(uo2) and PySpin.IsWritable(uos):
        uos.SetIntValue(uo2.GetValue())
        uov = PySpin.CBooleanPtr(nm.GetNode("UserOutputValue"))
        if PySpin.IsWritable(uov):
            print(f"  UserOutputValue is WRITABLE! Current={uov.GetValue()}")
            uov.SetValue(True)
            ls = PySpin.CBooleanPtr(nm.GetNode("LineStatus"))
            print(f"  Set True → LineStatus={ls.GetValue()}")
            uov.SetValue(False)
            print(f"  Set False → LineStatus={ls.GetValue()}")
        else:
            print(f"  UserOutputValue not writable.")
else:
    print(f"  UserOutputSelector not available.")

cam.DeInit()
del cam, nm
gc.collect()
cam_list.Clear()
system.ReleaseInstance()
print("\nDone.")
