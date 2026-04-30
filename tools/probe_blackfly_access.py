from __future__ import annotations

import sys
import time


def _enum_set(nodemap, node_name: str, entry_name: str) -> bool:
    node = PySpin.CEnumerationPtr(nodemap.GetNode(node_name))
    if not PySpin.IsReadable(node) or not PySpin.IsWritable(node):
        return False
    entry = node.GetEntryByName(entry_name)
    if not PySpin.IsReadable(entry):
        return False
    node.SetIntValue(entry.GetValue())
    return True


def _enum_info(nodemap, node_name: str) -> tuple[bool, bool, object | None, list[str]]:
    node = PySpin.CEnumerationPtr(nodemap.GetNode(node_name))
    if node is None:
        return False, False, None, []
    readable = PySpin.IsReadable(node)
    writable = PySpin.IsWritable(node)
    value = None
    entries: list[str] = []
    if readable:
        try:
            current = node.GetCurrentEntry()
            if PySpin.IsReadable(current):
                value = current.GetSymbolic()
        except Exception:
            value = None
        try:
            for entry_handle in node.GetEntries():
                entry = PySpin.CEnumEntryPtr(entry_handle)
                if PySpin.IsReadable(entry):
                    entries.append(entry.GetSymbolic())
        except Exception:
            entries = []
    return readable, writable, value, entries


def _command_execute(nodemap, node_name: str) -> bool:
    node = PySpin.CCommandPtr(nodemap.GetNode(node_name))
    if not PySpin.IsReadable(node) or not PySpin.IsWritable(node):
        return False
    node.Execute()
    return True

try:
    import PySpin  # type: ignore[import-not-found]
except ImportError:
    sys.exit("PySpin not available")


def _access_mode_name(mode: int) -> str:
    mapping = {
        int(PySpin.RO): "RO",
        int(PySpin.RW): "RW",
        int(PySpin.WO): "WO",
        int(PySpin.NA): "NA",
        int(PySpin.NI): "NI",
    }
    return mapping.get(int(mode), str(mode))


def _node_rw(nodemap, name: str) -> tuple[bool, bool, object | None]:
    for ptr_type in (PySpin.CIntegerPtr, PySpin.CEnumerationPtr, PySpin.CBooleanPtr, PySpin.CFloatPtr):
        try:
            node = ptr_type(nodemap.GetNode(name))
        except Exception:
            continue
        if node is None:
            continue
        readable = PySpin.IsReadable(node)
        writable = PySpin.IsWritable(node)
        value = None
        if readable:
            try:
                if hasattr(node, "GetValue"):
                    value = node.GetValue()
                elif hasattr(node, "GetCurrentEntry"):
                    entry = node.GetCurrentEntry()
                    value = entry.GetSymbolic() if PySpin.IsReadable(entry) else None
            except Exception:
                value = None
        return readable, writable, value
    return False, False, None


def _int_node_set(nodemap, node_name: str, value: int) -> bool:
    node = PySpin.CIntegerPtr(nodemap.GetNode(node_name))
    if not PySpin.IsReadable(node) or not PySpin.IsWritable(node):
        return False
    inc = int(node.GetInc()) or 1
    value = max(int(node.GetMin()), min(int(node.GetMax()), value))
    value = (value // inc) * inc
    node.SetValue(value)
    return True


def _report_camera(cam, idx: int) -> bool:
    tl = cam.GetTLDeviceNodeMap()
    nm = cam.GetNodeMap()
    model = PySpin.CStringPtr(tl.GetNode("DeviceModelName"))
    serial = PySpin.CStringPtr(tl.GetNode("DeviceSerialNumber"))
    if PySpin.IsReadable(model):
        print(f"cam{idx}_model={model.GetValue()}")
    if PySpin.IsReadable(serial):
        print(f"cam{idx}_serial={serial.GetValue()}")

    for node_name in (
        "Width",
        "Height",
        "OffsetX",
        "OffsetY",
        "BinningHorizontal",
        "BinningVertical",
        "DecimationHorizontal",
        "DecimationVertical",
        "CenterX",
        "CenterY",
        "PixelFormat",
        "TestPattern",
        "SequencerMode",
        "TriggerMode",
        "AcquisitionMode",
        "TLParamsLocked",
        "UserSetSelector",
    ):
        readable, writable, value = _node_rw(nm, node_name)
        print(
            f"cam{idx}_{node_name}=readable:{readable},writable:{writable},value:{value}"
        )

    us_readable, us_writable, us_value, us_entries = _enum_info(nm, "UserSetSelector")
    print(
        f"cam{idx}_UserSetSelector_enum=readable:{us_readable},writable:{us_writable},"
        f"value:{us_value},entries:{us_entries}"
    )

    print(f"cam{idx}_set_trigger_off={_enum_set(nm, 'TriggerMode', 'Off')}")
    print(f"cam{idx}_set_acquisition_continuous={_enum_set(nm, 'AcquisitionMode', 'Continuous')}")
    print(f"cam{idx}_set_sequencer_off={_enum_set(nm, 'SequencerMode', 'Off')}")
    print(f"cam{idx}_set_userset_default={_enum_set(nm, 'UserSetSelector', 'Default')}")
    print(f"cam{idx}_userset_load={_command_execute(nm, 'UserSetLoad')}")
    if idx == 1 and us_entries:
        for entry_name in us_entries:
            set_ok = _enum_set(nm, 'UserSetSelector', entry_name)
            load_ok = _command_execute(nm, 'UserSetLoad') if set_ok else False
            width_now = PySpin.IsWritable(PySpin.CIntegerPtr(nm.GetNode('Width')))
            height_now = PySpin.IsWritable(PySpin.CIntegerPtr(nm.GetNode('Height')))
            print(
                f"cam{idx}_userset_attempt={entry_name},selector_ok:{set_ok},load_ok:{load_ok},"
                f"width_w:{width_now},height_w:{height_now}"
            )
    print(f"cam{idx}_acquisition_stop={_command_execute(nm, 'AcquisitionStop')}")
    print(f"cam{idx}_acquisition_abort={_command_execute(nm, 'AcquisitionAbort')}")
    for int_name in (
        "BinningHorizontal",
        "BinningVertical",
        "DecimationHorizontal",
        "DecimationVertical",
    ):
        print(f"cam{idx}_{int_name}_set_1={_int_node_set(nm, int_name, 1)}")
    for bool_name in ("CenterX", "CenterY"):
        node = PySpin.CBooleanPtr(nm.GetNode(bool_name))
        if PySpin.IsReadable(node) and PySpin.IsWritable(node):
            try:
                node.SetValue(False)
                print(f"cam{idx}_{bool_name}_set_false=True")
            except Exception as exc:
                print(f"cam{idx}_{bool_name}_set_false_error={exc}")
    width = PySpin.CIntegerPtr(nm.GetNode('Width'))
    height = PySpin.CIntegerPtr(nm.GetNode('Height'))
    width_w = PySpin.IsWritable(width)
    height_w = PySpin.IsWritable(height)
    print(f"cam{idx}_after_normalize_width_w={width_w}")
    print(f"cam{idx}_after_normalize_height_w={height_w}")
    return width_w and height_w


def main() -> int:
    system = PySpin.System.GetInstance()
    cam_list = system.GetCameras()
    cams = []
    try:
        print(f"num_cameras={cam_list.GetSize()}")
        needs_post_reset_check = False
        for idx in range(cam_list.GetSize()):
            cam = cam_list.GetByIndex(idx)
            cams.append(cam)
            print(f"cam{idx}_transport_access={_access_mode_name(cam.GetAccessMode())}")
            try:
                cam.Init()
            except Exception as exc:
                print(f"cam{idx}_init_error={exc}")
                continue

            writable = _report_camera(cam, idx)
            if not writable and idx == 1:
                reset_ok = _command_execute(cam.GetNodeMap(), "DeviceReset")
                print(f"cam{idx}_device_reset={reset_ok}")
                if reset_ok:
                    needs_post_reset_check = True
                    time.sleep(2.0)

            try:
                cam.DeInit()
            except Exception:
                pass

        if needs_post_reset_check:
            cams.clear()
            cam_list.Clear()
            system.ReleaseInstance()

            system = PySpin.System.GetInstance()
            cam_list = system.GetCameras()
            print(f"post_reset_num_cameras={cam_list.GetSize()}")
            if cam_list.GetSize() > 1:
                cam = cam_list.GetByIndex(1)
                cams.append(cam)
                print(f"cam1_post_reset_transport_access={_access_mode_name(cam.GetAccessMode())}")
                cam.Init()
                _report_camera(cam, 1)
                cam.DeInit()

        return 0
    finally:
        for cam in cams:
            try:
                cam.DeInit()
            except Exception:
                pass
        cams.clear()
        cam_list.Clear()
        try:
            system.ReleaseInstance()
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())