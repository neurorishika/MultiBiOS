#!/usr/bin/env python3
"""Sweep DAQ-mode ROI requests against a Blackfly camera in isolated subprocesses.

This probe is intentionally process-isolated because PySpin teardown on this rig
can abort the interpreter after a successful configuration attempt.

Typical usage from the MultiBiOS root:

    python tools/manual_checks/camera_roi_sweep.py --camera-index 0
    python tools/manual_checks/camera_roi_sweep.py --camera-index 1 --sizes 400x400 512x512 640x640
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


DEFAULT_SIZES = [
    "400x400",
    "512x512",
    "640x640",
    "800x800",
    "1024x768",
    "1280x1024",
]


def _parse_size(text: str) -> tuple[int, int]:
    raw = text.strip().lower().replace(" ", "")
    if "x" not in raw:
        raise argparse.ArgumentTypeError(f"Invalid ROI size '{text}'. Use WIDTHxHEIGHT.")
    width_str, height_str = raw.split("x", 1)
    try:
        width = int(width_str)
        height = int(height_str)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid ROI size '{text}'. Use WIDTHxHEIGHT.") from exc
    if width <= 0 or height <= 0:
        raise argparse.ArgumentTypeError("ROI width and height must be positive.")
    return width, height


def _child_probe(args: argparse.Namespace) -> None:
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
                    "camera_index": args.camera_index,
                    "error": f"Requested camera index {args.camera_index}, but only {camera_count} camera(s) were found.",
                }
            )
        )
        sys.stdout.flush()
        os._exit(2)

    cam = cam_list.GetByIndex(args.camera_index)
    cam.Init()
    nm = cam.GetNodeMap()

    def _int_value(name: str) -> int | None:
        node = PySpin.CIntegerPtr(nm.GetNode(name))
        return int(node.GetValue()) if PySpin.IsReadable(node) else None

    def _int_max(name: str) -> int | None:
        node = PySpin.CIntegerPtr(nm.GetNode(name))
        return int(node.GetMax()) if PySpin.IsReadable(node) else None

    def _writable(name: str) -> bool:
        node = PySpin.CIntegerPtr(nm.GetNode(name))
        return bool(PySpin.IsWritable(node))

    def _string_value(name: str) -> str | None:
        node = PySpin.CStringPtr(cam.GetTLDeviceNodeMap().GetNode(name))
        return node.GetValue() if PySpin.IsReadable(node) else None

    def _state() -> dict[str, Any]:
        return {
            "width": _int_value("Width"),
            "height": _int_value("Height"),
            "offset_x": _int_value("OffsetX"),
            "offset_y": _int_value("OffsetY"),
            "width_max": _int_max("Width"),
            "height_max": _int_max("Height"),
            "offset_x_max": _int_max("OffsetX"),
            "offset_y_max": _int_max("OffsetY"),
            "sensor_width": _int_value("SensorWidth"),
            "sensor_height": _int_value("SensorHeight"),
            "width_writable": _writable("Width"),
            "height_writable": _writable("Height"),
        }

    payload: dict[str, Any] = {
        "ok": True,
        "camera_index": args.camera_index,
        "requested_width": args.roi_width,
        "requested_height": args.roi_height,
        "exposure_us": args.exposure_us,
        "binning": args.binning,
        "serial": _string_value("DeviceSerialNumber"),
        "model": _string_value("DeviceModelName"),
        "before": _state(),
    }
    try:
        configure_camera_daq_mode(
            cam,
            exposure_us=args.exposure_us,
            roi_width=args.roi_width,
            roi_height=args.roi_height,
            binning=args.binning,
            gain_db=args.gain_db,
            gamma=args.gamma,
        )
        payload["after"] = _state()
    except Exception as exc:
        payload["ok"] = False
        payload["error"] = str(exc)
        payload["after"] = _state()

    print(json.dumps(payload))
    sys.stdout.flush()
    os._exit(0)


def _run_probe(
    *,
    camera_index: int,
    roi_width: int,
    roi_height: int,
    exposure_us: float,
    binning: int,
    gain_db: float | None,
    gamma: float | None,
    timeout_s: float,
) -> dict[str, Any]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--child",
        "--camera-index",
        str(camera_index),
        "--roi-width",
        str(roi_width),
        "--roi-height",
        str(roi_height),
        "--exposure-us",
        str(exposure_us),
        "--binning",
        str(binning),
    ]
    if gain_db is not None:
        command.extend(["--gain-db", str(gain_db)])
    if gamma is not None:
        command.extend(["--gamma", str(gamma)])

    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=timeout_s,
        cwd=str(Path(__file__).resolve().parent.parent),
    )
    lines = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    if not lines:
        return {
            "ok": False,
            "camera_index": camera_index,
            "requested_width": roi_width,
            "requested_height": roi_height,
            "error": completed.stderr.strip() or f"child returned {completed.returncode} without JSON output",
        }
    try:
        payload = json.loads(lines[-1])
    except json.JSONDecodeError:
        return {
            "ok": False,
            "camera_index": camera_index,
            "requested_width": roi_width,
            "requested_height": roi_height,
            "error": f"could not parse child JSON output: {lines[-1]}",
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        }
    if completed.returncode not in (0, 2) and "error" not in payload:
        payload["error"] = completed.stderr.strip() or f"child returned {completed.returncode}"
        payload["ok"] = False
    return payload


def _print_summary(results: list[dict[str, Any]]) -> None:
    print()
    print("ROI sweep summary")
    print("requested   ok    before_writable  after_size        after_offset      notes")
    for result in results:
        req = f"{result.get('requested_width')}x{result.get('requested_height')}"
        before = result.get("before") or {}
        after = result.get("after") or {}
        before_writable = f"W={int(bool(before.get('width_writable')))} H={int(bool(before.get('height_writable')))}"
        after_size = f"{after.get('width')}x{after.get('height')}"
        after_offset = f"{after.get('offset_x')},{after.get('offset_y')}"
        note = result.get("error", "")
        print(f"{req:<11} {str(result.get('ok')):<5} {before_writable:<16} {after_size:<16} {after_offset:<17} {note}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Sweep ROI requests against one camera in isolated subprocesses.")
    parser.add_argument("--camera-index", type=int, required=True)
    parser.add_argument("--sizes", nargs="+", default=DEFAULT_SIZES, help="ROI sizes as WIDTHxHEIGHT entries.")
    parser.add_argument("--exposure-us", type=float, default=4500.0)
    parser.add_argument("--binning", type=int, default=1)
    parser.add_argument("--gain-db", type=float, default=None)
    parser.add_argument("--gamma", type=float, default=None)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument("--child", action="store_true")
    parser.add_argument("--roi-width", type=int, default=None)
    parser.add_argument("--roi-height", type=int, default=None)
    args = parser.parse_args()

    if args.child:
        if args.roi_width is None or args.roi_height is None:
            raise SystemExit("--child requires --roi-width and --roi-height")
        _child_probe(args)
        return 0

    sizes = [_parse_size(item) for item in args.sizes]
    results: list[dict[str, Any]] = []
    for width, height in sizes:
        print(f"Probing camera {args.camera_index} with ROI {width}x{height} ...")
        results.append(
            _run_probe(
                camera_index=args.camera_index,
                roi_width=width,
                roi_height=height,
                exposure_us=args.exposure_us,
                binning=args.binning,
                gain_db=args.gain_db,
                gamma=args.gamma,
                timeout_s=args.timeout,
            )
        )

    _print_summary(results)
    if args.json_out is not None:
        args.json_out.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print()
        print(f"Saved JSON results to {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())