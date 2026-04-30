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


def main() -> None:
    parser = argparse.ArgumentParser(description="Preconfigure one Blackfly camera in a helper process.")
    parser.add_argument("--camera-index", type=int, required=True)
    parser.add_argument("--exposure-us", type=float, default=None)
    parser.add_argument("--roi-width", type=int, default=None)
    parser.add_argument("--roi-height", type=int, default=None)
    parser.add_argument("--binning", type=int, default=1)
    parser.add_argument("--gain-db", type=float, default=None)
    parser.add_argument("--gamma", type=float, default=None)
    args = parser.parse_args()

    import PySpin  # type: ignore

    from multibios.blackfly.live_view import configure_camera_daq_mode

    system = PySpin.System.GetInstance()
    cam_list = system.GetCameras()
    camera_count = cam_list.GetSize()
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

    cam = cam_list.GetByIndex(args.camera_index)
    cam.Init()
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