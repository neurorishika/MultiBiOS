from __future__ import annotations

import argparse
import json
import os
import sys


def _read_geometry(pyspin, cam) -> dict[str, int | None]:
    nm = cam.GetNodeMap()

    def _get_int(name: str) -> int | None:
        node = pyspin.CIntegerPtr(nm.GetNode(name))
        return int(node.GetValue()) if pyspin.IsReadable(node) else None

    return {
        "width": _get_int("Width"),
        "height": _get_int("Height"),
        "offset_x": _get_int("OffsetX"),
        "offset_y": _get_int("OffsetY"),
    }


def _read_camera_state(pyspin, cam) -> dict[str, object]:
    nm = cam.GetNodeMap()
    tl = cam.GetTLDeviceNodeMap()

    def _get_int(name: str) -> int | None:
        node = pyspin.CIntegerPtr(nm.GetNode(name))
        return int(node.GetValue()) if pyspin.IsReadable(node) else None

    def _get_int_max(name: str) -> int | None:
        node = pyspin.CIntegerPtr(nm.GetNode(name))
        return int(node.GetMax()) if pyspin.IsReadable(node) else None

    def _get_bool(name: str) -> bool | None:
        node = pyspin.CBooleanPtr(nm.GetNode(name))
        return bool(node.GetValue()) if pyspin.IsReadable(node) else None

    def _get_enum(name: str) -> str | None:
        node = pyspin.CEnumerationPtr(nm.GetNode(name))
        if not pyspin.IsReadable(node):
            return None
        entry = node.GetCurrentEntry()
        return entry.GetSymbolic() if pyspin.IsReadable(entry) else None

    def _get_string(name: str) -> str | None:
        node = pyspin.CStringPtr(tl.GetNode(name))
        return node.GetValue() if pyspin.IsReadable(node) else None

    def _is_writable(name: str) -> bool:
        node = pyspin.CValuePtr(nm.GetNode(name))
        return bool(pyspin.IsWritable(node))

    return {
        "serial": _get_string("DeviceSerialNumber"),
        "model": _get_string("DeviceModelName"),
        "width": _get_int("Width"),
        "height": _get_int("Height"),
        "offset_x": _get_int("OffsetX"),
        "offset_y": _get_int("OffsetY"),
        "sensor_width": _get_int("SensorWidth"),
        "sensor_height": _get_int("SensorHeight"),
        "width_max": _get_int_max("Width"),
        "height_max": _get_int_max("Height"),
        "offset_x_max": _get_int_max("OffsetX"),
        "offset_y_max": _get_int_max("OffsetY"),
        "width_writable": _is_writable("Width"),
        "height_writable": _is_writable("Height"),
        "offset_x_writable": _is_writable("OffsetX"),
        "offset_y_writable": _is_writable("OffsetY"),
        "trigger_mode": _get_enum("TriggerMode"),
        "acquisition_mode": _get_enum("AcquisitionMode"),
        "trigger_source": _get_enum("TriggerSource"),
        "trigger_overlap": _get_enum("TriggerOverlap"),
        "line_source": _get_enum("LineSource"),
        "tlparams_locked": _get_int("TLParamsLocked"),
        "is_streaming": _get_bool("AcquisitionStatus"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Preconfigure one Blackfly camera in a helper process.")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--list-cameras", action="store_true")
    parser.add_argument("--inspect", action="store_true")
    parser.add_argument("--reset-editable", action="store_true")
    parser.add_argument("--exposure-us", type=float, default=None)
    parser.add_argument("--roi-width", type=int, default=None)
    parser.add_argument("--roi-height", type=int, default=None)
    parser.add_argument("--binning", type=int, default=1)
    parser.add_argument("--gain-db", type=float, default=None)
    parser.add_argument("--gamma", type=float, default=None)
    args = parser.parse_args()

    import PySpin  # type: ignore

    from multibios.blackfly.live_view import (configure_camera_daq_mode,
                                              reset_camera_to_editable_mode)

    system = PySpin.System.GetInstance()
    cam_list = system.GetCameras()
    camera_count = cam_list.GetSize()
    if args.list_cameras:
        cameras: list[dict[str, object]] = []
        for index in range(camera_count):
            cam = cam_list.GetByIndex(index)
            tl = cam.GetTLDeviceNodeMap()
            serial_node = PySpin.CStringPtr(tl.GetNode("DeviceSerialNumber"))
            model_node = PySpin.CStringPtr(tl.GetNode("DeviceModelName"))
            cameras.append(
                {
                    "camera_index": index,
                    "serial": serial_node.GetValue() if PySpin.IsReadable(serial_node) else None,
                    "model": model_node.GetValue() if PySpin.IsReadable(model_node) else None,
                }
            )
        print(json.dumps({"ok": True, "cameras": cameras}))
        sys.stdout.flush()
        os._exit(0)

    if args.camera_index < 0 or args.camera_index >= camera_count:
        print(
            json.dumps(
                {
                    "ok": False,
                    "error": f"Requested camera index {args.camera_index}, but only {camera_count} camera(s) were found.",
                }
            ),
            file=sys.stderr,
        )
        sys.stderr.flush()
        os._exit(2)

    if args.reset_editable:
        try:
            reset_camera_to_editable_mode(args.camera_index)
        except Exception as exc:
            print(json.dumps({"ok": False, "camera_index": args.camera_index, "error": str(exc)}))
            sys.stdout.flush()
            os._exit(1)
        print(json.dumps({"ok": True, "camera_index": args.camera_index, "mode": "reset_editable"}))
        sys.stdout.flush()
        os._exit(0)

    cam = cam_list.GetByIndex(args.camera_index)
    cam.Init()
    if args.inspect:
        print(json.dumps({"ok": True, "camera_index": args.camera_index, **_read_camera_state(PySpin, cam)}))
        sys.stdout.flush()
        os._exit(0)

    configure_camera_daq_mode(
        cam,
        exposure_us=args.exposure_us,
        roi_width=args.roi_width,
        roi_height=args.roi_height,
        binning=args.binning,
        gain_db=args.gain_db,
        gamma=args.gamma,
    )
    print(json.dumps({"ok": True, "camera_index": args.camera_index, **_read_geometry(PySpin, cam)}))
    sys.stdout.flush()

    # Bypass PySpin teardown in this short-lived helper process. On this rig,
    # normal destruction can abort the interpreter after successful config.
    os._exit(0)


if __name__ == "__main__":
    main()