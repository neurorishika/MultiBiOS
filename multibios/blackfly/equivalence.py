#!/usr/bin/env python3
"""Systematically compare and harmonize the two Blackfly cameras on this system.

What this script does:
1. Connects to the first two Spinnaker cameras.
2. Captures a baseline snapshot of transport, image-format, and timing nodes.
3. Optionally loads each camera's Default user set for a clean session state.
4. Applies the same canonical session configuration to both cameras.
5. Measures each camera independently to find its true source frame rate.
6. Sets both cameras to a common rate equal to the slower camera.
7. Measures both cameras simultaneously at that common rate.
8. Writes a JSON report with mismatches and timing statistics.

This script does not save anything permanently to camera flash. It only changes
the active session configuration.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import statistics as stats
import sys
import threading
import time
from pathlib import Path

try:
    import PySpin
except ImportError:
    sys.exit("PySpin not found. Activate multibios-blackfly environment.")

from .live_view import (_bool_set, _enum_set, _int_node_max, _int_node_set,
                        _maximize_link_throughput, connect_cameras,
                        release_cameras)

REPORT_DIR = Path("captured_frames") / "equivalence_tests"
DEFAULT_ROI_WIDTH = 400
DEFAULT_ROI_HEIGHT = 400
DEFAULT_TIMEOUT_MS = 1000
DEFAULT_DURATION_S = 5.0
COMPARE_NODES = [
    "DeviceVendorName",
    "DeviceModelName",
    "DeviceSerialNumber",
    "DeviceVersion",
    "DeviceLinkSpeed",
    "DeviceLinkThroughputLimitMode",
    "DeviceLinkThroughputLimit",
    "LinkU3VCurrentSpeed",
    "PixelFormat",
    "Width",
    "Height",
    "OffsetX",
    "OffsetY",
    "BinningHorizontal",
    "BinningVertical",
    "DecimationHorizontal",
    "DecimationVertical",
    "ReverseX",
    "ReverseY",
    "TestPattern",
    "IspEnable",
    "GammaEnable",
    "LUTEnable",
    "SharpeningEnable",
    "ExposureAuto",
    "ExposureMode",
    "ExposureTime",
    "ExposureTimeAbs",
    "GainAuto",
    "Gain",
    "GainDB",
    "AcquisitionMode",
    "AcquisitionFrameRateEnable",
    "AcquisitionFrameRateEnabled",
    "AcquisitionFrameRate",
    "AcquisitionFrameRateAbs",
    "AcquisitionResultingFrameRate",
    "TriggerMode",
    "TriggerSource",
    "TriggerSelector",
    "StreamBufferHandlingMode",
    "StreamBufferCountMode",
    "StreamBufferCountManual",
    "StreamAnnounceBufferMinimum",
    "UserSetSelector",
    "UserSetDefault",
]


def _read_enum(nodemap, name: str):
    node = PySpin.CEnumerationPtr(nodemap.GetNode(name))
    if not PySpin.IsReadable(node):
        return None
    entry = node.GetCurrentEntry()
    return entry.GetSymbolic() if PySpin.IsReadable(entry) else None


def _read_bool(nodemap, name: str):
    node = PySpin.CBooleanPtr(nodemap.GetNode(name))
    if not PySpin.IsReadable(node):
        return None
    return bool(node.GetValue())


def _read_int(nodemap, name: str):
    node = PySpin.CIntegerPtr(nodemap.GetNode(name))
    if not PySpin.IsReadable(node):
        return None
    return int(node.GetValue())


def _read_float(nodemap, name: str):
    node = PySpin.CFloatPtr(nodemap.GetNode(name))
    if not PySpin.IsReadable(node):
        return None
    return float(node.GetValue())


def _read_string(nodemap, name: str):
    node = PySpin.CStringPtr(nodemap.GetNode(name))
    if not PySpin.IsReadable(node):
        return None
    return node.GetValue()


def _read_node(nodemap, name: str):
    for kind, reader in (
        ("enum", _read_enum),
        ("bool", _read_bool),
        ("int", _read_int),
        ("float", _read_float),
        ("string", _read_string),
    ):
        try:
            value = reader(nodemap, name)
        except Exception:
            value = None
        if value is not None:
            return {"type": kind, "value": value}
    return {"type": "unavailable", "value": None}


def _command_execute(nodemap, name: str) -> bool:
    node = PySpin.CCommandPtr(nodemap.GetNode(name))
    if not PySpin.IsWritable(node):
        return False
    node.Execute()
    return True


def _set_centered_roi(cam, width: int, height: int) -> tuple[int, int, int, int]:
    nm = cam.GetNodeMap()
    _int_node_set(nm, "OffsetX", 0)
    _int_node_set(nm, "OffsetY", 0)

    width_max = _int_node_max(nm, "Width")
    height_max = _int_node_max(nm, "Height")
    if not width_max or not height_max:
        raise RuntimeError("Could not read maximum Width/Height from camera.")

    width_node = PySpin.CIntegerPtr(nm.GetNode("Width"))
    height_node = PySpin.CIntegerPtr(nm.GetNode("Height"))
    offset_x_node = PySpin.CIntegerPtr(nm.GetNode("OffsetX"))
    offset_y_node = PySpin.CIntegerPtr(nm.GetNode("OffsetY"))

    width_inc = int(width_node.GetInc()) or 1
    height_inc = int(height_node.GetInc()) or 1
    offset_x_inc = int(offset_x_node.GetInc()) if PySpin.IsReadable(offset_x_node) else 1
    offset_y_inc = int(offset_y_node.GetInc()) if PySpin.IsReadable(offset_y_node) else 1
    offset_x_inc = offset_x_inc or 1
    offset_y_inc = offset_y_inc or 1

    width = max(int(width_node.GetMin()), min(width_max, int(width)))
    height = max(int(height_node.GetMin()), min(height_max, int(height)))
    width = max(width_inc, (width // width_inc) * width_inc)
    height = max(height_inc, (height // height_inc) * height_inc)

    _int_node_set(nm, "Width", width)
    _int_node_set(nm, "Height", height)
    _int_node_set(nm, "OffsetX", ((width_max - width) // 2 // offset_x_inc) * offset_x_inc)
    _int_node_set(nm, "OffsetY", ((height_max - height) // 2 // offset_y_inc) * offset_y_inc)

    return (
        _read_int(nm, "Width") or width,
        _read_int(nm, "Height") or height,
        _read_int(nm, "OffsetX") or 0,
        _read_int(nm, "OffsetY") or 0,
    )


def _set_minimum_exposure(nodemap) -> float | None:
    _enum_set(nodemap, "ExposureAuto", "Off")
    _enum_set(nodemap, "ExposureMode", "Timed")
    for name in ("ExposureTime", "ExposureTimeAbs"):
        node = PySpin.CFloatPtr(nodemap.GetNode(name))
        if not PySpin.IsReadable(node) or not PySpin.IsWritable(node):
            continue
        node.SetValue(float(node.GetMin()))
        return float(node.GetValue())
    return None


def _get_frame_rate_node(nodemap):
    for name in ("AcquisitionFrameRate", "AcquisitionFrameRateAbs"):
        node = PySpin.CFloatPtr(nodemap.GetNode(name))
        if PySpin.IsReadable(node):
            return name, node
    return None, None


def _set_target_frame_rate(nodemap, target_fps: float | None) -> tuple[float | None, str]:
    _enum_set(nodemap, "AcquisitionMode", "Continuous")
    enabled = (
        _bool_set(nodemap, "AcquisitionFrameRateEnable", True, silent=True)
        or _bool_set(nodemap, "AcquisitionFrameRateEnabled", True, silent=True)
    )
    node_name, node = _get_frame_rate_node(nodemap)
    if node is None:
        return None, "no-frame-rate-node"
    if not PySpin.IsWritable(node):
        return float(node.GetValue()), "frame-rate-read-only"

    if target_fps is None:
        target = float(node.GetMax())
        mode = f"{node_name}=max"
    else:
        target = max(float(node.GetMin()), min(float(node.GetMax()), target_fps))
        mode = f"{node_name}=target"
    node.SetValue(target)
    if not enabled:
        mode += ":no-enable-node"
    return float(node.GetValue()), mode


def _snapshot_camera(cam) -> dict:
    nm = cam.GetNodeMap()
    tl = cam.GetTLDeviceNodeMap()
    snapshot = {}
    for name in COMPARE_NODES:
        snapshot[name] = _read_node(nm, name)
        if snapshot[name]["type"] == "unavailable":
            snapshot[name] = _read_node(tl, name)
    return snapshot


def _values_equal(left: dict, right: dict) -> bool:
    if left["type"] != right["type"]:
        return False
    if left["type"] == "float":
        return math.isclose(float(left["value"]), float(right["value"]), rel_tol=1e-9, abs_tol=1e-6)
    return left["value"] == right["value"]


def _diff_snapshots(left: dict, right: dict) -> list[dict]:
    diffs = []
    for name in sorted(set(left) | set(right)):
        lval = left.get(name, {"type": "missing", "value": None})
        rval = right.get(name, {"type": "missing", "value": None})
        if not _values_equal(lval, rval):
            diffs.append({"node": name, "camera0": lval, "camera1": rval})
    return diffs


def _load_default_userset(cam) -> bool:
    nm = cam.GetNodeMap()
    if not _enum_set(nm, "UserSetSelector", "Default"):
        return False
    return _command_execute(nm, "UserSetLoad")


def _configure_canonical(cam, roi_width: int, roi_height: int,
                         frame_rate_target: float | None = None) -> dict:
    nm = cam.GetNodeMap()
    sn = cam.GetTLStreamNodeMap()

    _load_default_userset(cam)
    _enum_set(nm, "TriggerMode", "Off")
    _enum_set(nm, "PixelFormat", "Mono8")
    _int_node_set(nm, "BinningHorizontal", 1)
    _int_node_set(nm, "BinningVertical", 1)
    _int_node_set(nm, "DecimationHorizontal", 1)
    _int_node_set(nm, "DecimationVertical", 1)
    _bool_set(nm, "ReverseX", False, silent=True)
    _bool_set(nm, "ReverseY", False, silent=True)
    _enum_set(nm, "TestPattern", "Off")
    _bool_set(nm, "IspEnable", False, silent=True)
    _bool_set(nm, "GammaEnable", False, silent=True)
    _bool_set(nm, "LUTEnable", False, silent=True)
    _bool_set(nm, "SharpeningEnable", False, silent=True)
    _enum_set(nm, "GainAuto", "Off")
    _maximize_link_throughput(nm)
    roi = _set_centered_roi(cam, roi_width, roi_height)
    exposure_us = _set_minimum_exposure(nm)
    set_fps, fps_mode = _set_target_frame_rate(nm, frame_rate_target)
    _enum_set(sn, "StreamBufferHandlingMode", "NewestOnly")

    return {
        "roi": {"width": roi[0], "height": roi[1], "offset_x": roi[2], "offset_y": roi[3]},
        "exposure_us": exposure_us,
        "frame_rate_set": set_fps,
        "frame_rate_mode": fps_mode,
        "frame_rate_resulting": _read_float(nm, "AcquisitionResultingFrameRate"),
        "device_link_speed": _read_int(nm, "DeviceLinkSpeed"),
        "device_link_throughput_limit": _read_int(nm, "DeviceLinkThroughputLimit"),
    }


def _grab_stats_from_samples(frame_ids: list[int], timestamps: list[int]) -> dict:
    if len(frame_ids) < 2 or len(timestamps) < 2:
        return {
            "frames_saved": len(frame_ids),
            "duration_s": 0.0,
            "saved_fps": None,
            "source_fps": None,
            "skipped_frames": None,
            "normalized_dt_ms_mean": None,
            "normalized_dt_ms_std": None,
            "normalized_dt_ms_min": None,
            "normalized_dt_ms_max": None,
        }

    id_diffs = [b - a for a, b in zip(frame_ids, frame_ids[1:])]
    dt_ns = [b - a for a, b in zip(timestamps, timestamps[1:])]
    duration_s = (timestamps[-1] - timestamps[0]) / 1e9
    total_source_frames = frame_ids[-1] - frame_ids[0] + 1
    skipped_frames = sum(max(0, diff - 1) for diff in id_diffs)
    normalized_dt_ms = [(dt / diff) / 1e6 for dt, diff in zip(dt_ns, id_diffs) if diff > 0]

    return {
        "frames_saved": len(frame_ids),
        "frame_id_first": frame_ids[0],
        "frame_id_last": frame_ids[-1],
        "duration_s": duration_s,
        "saved_fps": ((len(frame_ids) - 1) / duration_s) if duration_s > 0 else None,
        "source_fps": ((total_source_frames - 1) / duration_s) if duration_s > 0 else None,
        "skipped_frames": skipped_frames,
        "normalized_dt_ms_mean": stats.mean(normalized_dt_ms),
        "normalized_dt_ms_std": stats.pstdev(normalized_dt_ms),
        "normalized_dt_ms_min": min(normalized_dt_ms),
        "normalized_dt_ms_max": max(normalized_dt_ms),
        "id_step_histogram": {str(step): id_diffs.count(step) for step in sorted(set(id_diffs))},
    }


def _acquire_samples(cam, duration_s: float, timeout_ms: int) -> dict:
    frame_ids: list[int] = []
    timestamps: list[int] = []
    errors: list[str] = []

    cam.BeginAcquisition()
    try:
        t_end = time.perf_counter() + duration_s
        while time.perf_counter() < t_end:
            try:
                img = cam.GetNextImage(timeout_ms)
            except PySpin.SpinnakerException as exc:
                errors.append(str(exc))
                break

            try:
                if img.IsIncomplete():
                    errors.append(f"Incomplete image status={img.GetImageStatus()}")
                    continue
                frame_ids.append(int(img.GetFrameID()))
                timestamps.append(int(img.GetTimeStamp()))
            finally:
                img.Release()
    finally:
        try:
            cam.EndAcquisition()
        except Exception:
            pass

    result = _grab_stats_from_samples(frame_ids, timestamps)
    result["errors"] = errors
    return result


def _acquire_samples_thread(cam, duration_s: float, timeout_ms: int, barrier: threading.Barrier,
                            out: dict, key: str) -> None:
    frame_ids: list[int] = []
    timestamps: list[int] = []
    errors: list[str] = []

    try:
        cam.BeginAcquisition()
        barrier.wait(timeout=5.0)
        t_end = time.perf_counter() + duration_s
        while time.perf_counter() < t_end:
            try:
                img = cam.GetNextImage(timeout_ms)
            except PySpin.SpinnakerException as exc:
                errors.append(str(exc))
                break

            try:
                if img.IsIncomplete():
                    errors.append(f"Incomplete image status={img.GetImageStatus()}")
                    continue
                frame_ids.append(int(img.GetFrameID()))
                timestamps.append(int(img.GetTimeStamp()))
            finally:
                img.Release()
    finally:
        try:
            cam.EndAcquisition()
        except Exception:
            pass

    out[key] = _grab_stats_from_samples(frame_ids, timestamps)
    out[key]["errors"] = errors


def _measure_simultaneous(cams: list, duration_s: float, timeout_ms: int) -> dict:
    barrier = threading.Barrier(2)
    results: dict = {}
    threads = [
        threading.Thread(
            target=_acquire_samples_thread,
            args=(cams[idx], duration_s, timeout_ms, barrier, results, f"camera{idx}"),
            daemon=True,
        )
        for idx in range(2)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    return results


def _estimate_common_fps(individual_results: list[dict]) -> float:
    measured = [r["source_fps"] for r in individual_results if r.get("source_fps")]
    if not measured:
        raise RuntimeError("Could not estimate a common frame rate from individual measurements.")
    # Small guard band so both cameras can actually sustain the chosen value together.
    return max(1.0, min(measured) * 0.99)


def main() -> None:
    parser = argparse.ArgumentParser(description="Systematically test and harmonize two Blackfly cameras")
    parser.add_argument("--width", type=int, default=DEFAULT_ROI_WIDTH,
                        help=f"Centered ROI width in pixels (default: {DEFAULT_ROI_WIDTH})")
    parser.add_argument("--height", type=int, default=DEFAULT_ROI_HEIGHT,
                        help=f"Centered ROI height in pixels (default: {DEFAULT_ROI_HEIGHT})")
    parser.add_argument("--duration", type=float, default=DEFAULT_DURATION_S,
                        help=f"Measurement duration per phase in seconds (default: {DEFAULT_DURATION_S})")
    parser.add_argument("--timeout-ms", type=int, default=DEFAULT_TIMEOUT_MS,
                        help=f"GetNextImage timeout in ms (default: {DEFAULT_TIMEOUT_MS})")
    parser.add_argument("--keep-current-userset", action="store_true",
                        help="Skip loading the Default user set before canonical session config")
    parser.add_argument("--report-dir", type=Path, default=REPORT_DIR,
                        help=f"Directory for JSON reports (default: {REPORT_DIR})")
    args = parser.parse_args()

    if args.width <= 0 or args.height <= 0:
        raise SystemExit("ROI width and height must be positive integers.")
    if args.duration <= 0 or args.timeout_ms <= 0:
        raise SystemExit("Duration and timeout must be positive.")

    args.report_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.report_dir / f"equivalence_{time.strftime('%Y%m%d_%H%M%S')}.json"

    system, cam_list, cams = connect_cameras()
    try:
        print("\nCollecting baseline snapshots ...")
        baseline = []
        for idx, cam in enumerate(cams):
            model = _read_string(cam.GetTLDeviceNodeMap(), "DeviceModelName") or "?"
            serial = _read_string(cam.GetTLDeviceNodeMap(), "DeviceSerialNumber") or "?"
            baseline.append({
                "camera_index": idx,
                "model": model,
                "serial": serial,
                "snapshot": _snapshot_camera(cam),
            })
            print(f"  Camera {idx}: {model} [S/N {serial}]")

        if not args.keep_current_userset:
            print("\nLoading Default user set on both cameras ...")
            for idx, cam in enumerate(cams):
                ok = _load_default_userset(cam)
                print(f"  Camera {idx}: {'loaded Default userset' if ok else 'Default userset unavailable'}")

        print("\nApplying canonical identical session configuration ...")
        canonical = []
        for idx, cam in enumerate(cams):
            cfg = _configure_canonical(cam, args.width, args.height, frame_rate_target=None)
            canonical.append(cfg)
            print(
                f"  Camera {idx}: roi={cfg['roi']['width']}x{cfg['roi']['height']} "
                f"offset=({cfg['roi']['offset_x']},{cfg['roi']['offset_y']}) "
                f"exp={cfg['exposure_us']}us result_fps={cfg['frame_rate_resulting']}"
            )

        canonical_snapshots = [_snapshot_camera(cam) for cam in cams]

        print("\nMeasuring each camera independently ...")
        individual_results = []
        for idx, cam in enumerate(cams):
            result = _acquire_samples(cam, args.duration, args.timeout_ms)
            individual_results.append(result)
            print(
                f"  Camera {idx}: source_fps={result['source_fps']:.3f} "
                f"saved_fps={result['saved_fps']:.3f} skipped={result['skipped_frames']}"
            )

        common_fps = _estimate_common_fps(individual_results)
        print(f"\nSetting both cameras to a common session rate of {common_fps:.3f} fps ...")
        common_cfg = []
        for idx, cam in enumerate(cams):
            cfg = _configure_canonical(cam, args.width, args.height, frame_rate_target=common_fps)
            common_cfg.append(cfg)
            print(
                f"  Camera {idx}: set={cfg['frame_rate_set']:.3f} "
                f"resulting={cfg['frame_rate_resulting']:.3f}"
            )

        print("\nMeasuring both cameras simultaneously at the common rate ...")
        simultaneous = _measure_simultaneous(cams, args.duration, args.timeout_ms)
        for idx in range(2):
            result = simultaneous.get(f"camera{idx}", {})
            print(
                f"  Camera {idx}: source_fps={result.get('source_fps')} "
                f"saved_fps={result.get('saved_fps')} skipped={result.get('skipped_frames')}"
            )

        report = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "report_path": str(report_path),
            "baseline": baseline,
            "baseline_differences": _diff_snapshots(baseline[0]["snapshot"], baseline[1]["snapshot"]),
            "canonical_config": canonical,
            "canonical_snapshots": canonical_snapshots,
            "canonical_differences": _diff_snapshots(canonical_snapshots[0], canonical_snapshots[1]),
            "individual_measurements": individual_results,
            "common_rate_target_fps": common_fps,
            "common_rate_config": common_cfg,
            "simultaneous_measurements": simultaneous,
        }
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

        print(f"\nWrote report to {report_path}")
    finally:
        # PySpin camera objects are reference-counted; the last loop variable can
        # keep a camera alive until function exit unless it is cleared explicitly.
        cam = None
        baseline = None
        canonical = None
        canonical_snapshots = None
        individual_results = None
        common_cfg = None
        simultaneous = None
        report = None
        gc.collect()
        release_cameras(system, cam_list, cams, restore_daq=False)
        gc.collect()


if __name__ == "__main__":
    main()