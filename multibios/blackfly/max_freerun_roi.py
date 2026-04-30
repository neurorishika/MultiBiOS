#!/usr/bin/env python3
"""Headless dual Blackfly capture on a centered ROI at maximum free-run rate.

This script is optimized for sustained recording throughput rather than live
viewing. It configures both cameras for Mono8 free-run on a centered ROI,
forces a fixed exposure, and appends raw frames to disk using a background
writer thread.

Output layout per run:
    captured_frames/max_freerun_roi_YYYYMMDD_HHMMSS/
        manifest.json
        frame_index.csv
        cam0_frames.bin
        cam1_frames.bin

Each .bin file contains back-to-back raw Mono8 frames in row-major order.
Use manifest.json to recover width, height, dtype, and frame counts.

Run from the PySpin environment:
    conda activate multibios-blackfly
    python -m multibios.blackfly.max_freerun_roi
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import queue
import statistics as stats
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np

try:
    import PySpin
except ImportError:
    sys.exit(
        "\nPySpin not found.\n"
        "Activate the correct environment first:\n"
        "  conda activate multibios-blackfly\n"
        "  pip install <path>\\assets\\spinnaker_python-4.3.0.189-cp310-cp310-win_amd64\\spinnaker_python-4.3.0.189-cp310-cp310-win_amd64.whl\n"
    )

from .live_view import (_bool_set, _enum_set, _int_node_max, _int_node_set,
                        _maximize_link_throughput, connect_cameras,
                        release_cameras)

DEFAULT_ROI_WIDTH = 400
DEFAULT_ROI_HEIGHT = 400
DEFAULT_TIMEOUT_MS = 1000
DEFAULT_QUEUE_SIZE = 512
DEFAULT_STREAM_BUFFER_COUNT = 256
SAVE_ROOT = Path("captured_frames")
LOSSLESS_VIDEO_CANDIDATES = [
    ("FFV1", ".mkv"),
    ("FFV1", ".avi"),
    ("HFYU", ".avi"),
]


def _read_float(nodemap, node_name: str) -> float | None:
    node = PySpin.CFloatPtr(nodemap.GetNode(node_name))
    if not PySpin.IsReadable(node):
        return None
    return float(node.GetValue())


def _read_int(nodemap, node_name: str) -> int | None:
    node = PySpin.CIntegerPtr(nodemap.GetNode(node_name))
    if not PySpin.IsReadable(node):
        return None
    return int(node.GetValue())


def _percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    idx = (len(ordered) - 1) * p
    lo = int(math.floor(idx))
    hi = int(math.ceil(idx))
    if lo == hi:
        return ordered[lo]
    frac = idx - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def _camera_identity(cam) -> tuple[str, str]:
    tl = cam.GetTLDeviceNodeMap()
    model_n = PySpin.CStringPtr(tl.GetNode("DeviceModelName"))
    sn_n = PySpin.CStringPtr(tl.GetNode("DeviceSerialNumber"))
    model = model_n.GetValue() if PySpin.IsReadable(model_n) else "?"
    serial = sn_n.GetValue() if PySpin.IsReadable(sn_n) else "?"
    return model, serial


def _set_centered_roi(cam, width: int, height: int) -> tuple[int, int, int, int]:
    """Apply a centered ROI and return (width, height, offset_x, offset_y)."""
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

    if not PySpin.IsWritable(width_node) or not PySpin.IsWritable(height_node):
        raise RuntimeError("Width/Height nodes are not writable.")

    width_inc = int(width_node.GetInc()) or 1
    height_inc = int(height_node.GetInc()) or 1

    width = max(int(width_node.GetMin()), min(width_max, int(width)))
    height = max(int(height_node.GetMin()), min(height_max, int(height)))
    width = max(width_inc, (width // width_inc) * width_inc)
    height = max(height_inc, (height // height_inc) * height_inc)

    _int_node_set(nm, "Width", width)
    _int_node_set(nm, "Height", height)

    offset_x_inc = int(offset_x_node.GetInc()) if PySpin.IsReadable(offset_x_node) else 1
    offset_y_inc = int(offset_y_node.GetInc()) if PySpin.IsReadable(offset_y_node) else 1
    offset_x_inc = offset_x_inc or 1
    offset_y_inc = offset_y_inc or 1

    offset_x = ((width_max - width) // 2 // offset_x_inc) * offset_x_inc
    offset_y = ((height_max - height) // 2 // offset_y_inc) * offset_y_inc

    _int_node_set(nm, "OffsetX", offset_x)
    _int_node_set(nm, "OffsetY", offset_y)

    actual_width = int(width_node.GetValue())
    actual_height = int(height_node.GetValue())
    actual_offset_x = int(offset_x_node.GetValue()) if PySpin.IsReadable(offset_x_node) else offset_x
    actual_offset_y = int(offset_y_node.GetValue()) if PySpin.IsReadable(offset_y_node) else offset_y
    return actual_width, actual_height, actual_offset_x, actual_offset_y


def _set_exposure(nodemap, exposure_us: float | None) -> dict:
    _enum_set(nodemap, "ExposureAuto", "Off")
    _enum_set(nodemap, "ExposureMode", "Timed")

    for node_name in ("ExposureTime", "ExposureTimeAbs"):
        node = PySpin.CFloatPtr(nodemap.GetNode(node_name))
        if not PySpin.IsReadable(node) or not PySpin.IsWritable(node):
            continue
        min_exposure = float(node.GetMin())
        max_exposure = float(node.GetMax())
        target_exposure = min_exposure if exposure_us is None else float(exposure_us)
        target_exposure = max(min_exposure, min(max_exposure, target_exposure))
        node.SetValue(target_exposure)
        return {
            "requested_exposure_us": None if exposure_us is None else float(exposure_us),
            "exposure_us": float(node.GetValue()),
            "exposure_min_us": min_exposure,
            "exposure_max_us": max_exposure,
            "exposure_node": node_name,
        }
    return {
        "requested_exposure_us": None if exposure_us is None else float(exposure_us),
        "exposure_us": None,
        "exposure_min_us": None,
        "exposure_max_us": None,
        "exposure_node": None,
    }


def _set_minimum_gain(nodemap) -> float | None:
    _enum_set(nodemap, "GainAuto", "Off")

    for node_name in ("Gain", "GainDB"):
        node = PySpin.CFloatPtr(nodemap.GetNode(node_name))
        if not PySpin.IsReadable(node) or not PySpin.IsWritable(node):
            continue
        min_gain = float(node.GetMin())
        node.SetValue(min_gain)
        return float(node.GetValue())
    return None


def _set_max_free_run_rate(nodemap) -> tuple[float | None, str]:
    _enum_set(nodemap, "AcquisitionMode", "Continuous")

    enabled = (
        _bool_set(nodemap, "AcquisitionFrameRateEnable", True, silent=True)
        or _bool_set(nodemap, "AcquisitionFrameRateEnabled", True, silent=True)
    )

    for node_name in ("AcquisitionFrameRate", "AcquisitionFrameRateAbs"):
        node = PySpin.CFloatPtr(nodemap.GetNode(node_name))
        if not PySpin.IsReadable(node):
            continue
        max_rate = float(node.GetMax())
        if PySpin.IsWritable(node):
            node.SetValue(max_rate)
            return float(node.GetValue()), f"{node_name}=max"

    if enabled:
        return None, "frame-rate-enable-only"
    return None, "uncontrolled-free-run"


def _configure_stream_buffer(cam, count: int) -> int | None:
    sn = cam.GetTLStreamNodeMap()
    _enum_set(sn, "StreamBufferHandlingMode", "OldestFirst")
    _enum_set(sn, "StreamBufferCountMode", "Manual")
    _int_node_set(sn, "StreamBufferCountManual", count)
    return _read_int(sn, "StreamBufferCountManual")


def _estimate_sync_freerun_limit(configs: list[dict]) -> dict:
    resulting_pairs = [
        (idx, float(cfg["resulting_fps"]))
        for idx, cfg in enumerate(configs)
        if cfg.get("resulting_fps") is not None and float(cfg["resulting_fps"]) > 0
    ]
    exposure_pairs = [
        (idx, 1_000_000.0 / float(cfg["exposure_us"]))
        for idx, cfg in enumerate(configs)
        if cfg.get("exposure_us") is not None and float(cfg["exposure_us"]) > 0
    ]

    sync_limit_fps = None
    sync_limit_camera = None
    if resulting_pairs:
        sync_limit_camera, sync_limit_fps = min(resulting_pairs, key=lambda item: item[1])

    exposure_limit_fps = None
    exposure_limit_camera = None
    if exposure_pairs:
        exposure_limit_camera, exposure_limit_fps = min(exposure_pairs, key=lambda item: item[1])

    return {
        "synchronized_freerun_fps": sync_limit_fps,
        "synchronized_freerun_bottleneck_camera": sync_limit_camera,
        "exposure_only_fps_limit": exposure_limit_fps,
        "exposure_only_bottleneck_camera": exposure_limit_camera,
    }


def _configure_camera_for_max_freerun(
    cam,
    roi_width: int,
    roi_height: int,
    stream_buffer_count: int,
    exposure_us: float | None,
) -> dict:
    nm = cam.GetNodeMap()

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
    _maximize_link_throughput(nm)

    roi_w, roi_h, off_x, off_y = _set_centered_roi(cam, roi_width, roi_height)
    exposure_info = _set_exposure(nm, exposure_us)
    gain_value = _set_minimum_gain(nm)
    target_fps, fps_mode = _set_max_free_run_rate(nm)
    resulting_fps = _read_float(nm, "AcquisitionResultingFrameRate")
    actual_buffer_count = _configure_stream_buffer(cam, stream_buffer_count)

    return {
        "roi_w": roi_w,
        "roi_h": roi_h,
        "off_x": off_x,
        "off_y": off_y,
        **exposure_info,
        "gain_value": gain_value,
        "target_fps": target_fps,
        "resulting_fps": resulting_fps,
        "fps_mode": fps_mode,
        "stream_buffer_count": actual_buffer_count,
    }


def _grab_one_mono(cam, timeout_ms: int) -> dict | None:
    try:
        img = cam.GetNextImage(timeout_ms)
    except PySpin.SpinnakerException as exc:
        print(f"  [warn] Grab failed: {exc}")
        return None

    try:
        if img.IsIncomplete():
            print(f"  [warn] Incomplete frame (status {img.GetImageStatus()})")
            return None
        return {
            "frame": img.GetNDArray().copy(),
            "frame_id": int(img.GetFrameID()),
            "camera_timestamp": int(img.GetTimeStamp()),
            "host_timestamp_ns": time.time_ns(),
        }
    finally:
        img.Release()


def _grab_worker(cam, out_q: queue.Queue, timeout_ms: int) -> None:
    out_q.put(_grab_one_mono(cam, timeout_ms))


def grab_pair_mono(cams: list, timeout_ms: int) -> tuple[dict | None, dict | None]:
    qs = [queue.Queue(maxsize=1), queue.Queue(maxsize=1)]
    threads = [
        threading.Thread(target=_grab_worker, args=(cams[i], qs[i], timeout_ms), daemon=True)
        for i in range(2)
    ]

    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=(timeout_ms / 1000.0) + 0.5)

    packets = []
    for q in qs:
        try:
            packets.append(q.get_nowait())
        except queue.Empty:
            packets.append(None)
    return packets[0], packets[1]


def _writer_worker(
    run_dir: Path,
    frame_q: queue.Queue,
    manifest: dict,
    writer_error: list,
) -> None:
    cam_paths = [run_dir / "cam0_frames.bin", run_dir / "cam1_frames.bin"]
    csv_path = run_dir / "frame_index.csv"
    counts = [0, 0]
    pair_count = 0

    try:
        with open(cam_paths[0], "wb") as cam0_fh, open(cam_paths[1], "wb") as cam1_fh, open(
            csv_path, "w", newline="", encoding="utf-8"
        ) as csv_fh:
            writer = csv.writer(csv_fh)
            writer.writerow(
                [
                    "pair_index",
                    "camera_index",
                    "frame_index",
                    "frame_id",
                    "camera_timestamp",
                    "host_timestamp_ns",
                ]
            )

            while True:
                item = frame_q.get()
                if item is None:
                    frame_q.task_done()
                    break

                pair_index = item["pair_index"]
                packets = item["packets"]

                cam0_fh.write(packets[0]["frame"].tobytes(order="C"))
                cam1_fh.write(packets[1]["frame"].tobytes(order="C"))

                for camera_index, packet in enumerate(packets):
                    writer.writerow(
                        [
                            pair_index,
                            camera_index,
                            counts[camera_index],
                            packet["frame_id"],
                            packet["camera_timestamp"],
                            packet["host_timestamp_ns"],
                        ]
                    )
                    counts[camera_index] += 1

                pair_count += 1
                frame_q.task_done()

        manifest["saved_pairs"] = pair_count
        manifest["camera0_saved_frames"] = counts[0]
        manifest["camera1_saved_frames"] = counts[1]
        manifest["completed_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    except Exception as exc:
        writer_error.append(exc)


def _enqueue_pair(frame_q: queue.Queue, item: dict, writer_error: list) -> None:
    while True:
        try:
            frame_q.put(item, timeout=0.5)
            return
        except queue.Full:
            if writer_error:
                raise RuntimeError(f"Writer thread failed: {writer_error[0]}")


def _analyze_rows(rows: list[dict]) -> dict:
    if not rows:
        return {
            "frames_saved": 0,
            "frame_id_first": None,
            "frame_id_last": None,
            "duration_s": 0.0,
            "saved_fps": None,
            "source_fps": None,
            "skipped_frames": None,
            "drop_pct_vs_source": None,
            "normalized_dt_ms_mean": None,
            "normalized_dt_ms_std": None,
            "normalized_dt_ms_min": None,
            "normalized_dt_ms_p50": None,
            "normalized_dt_ms_p95": None,
            "normalized_dt_ms_p99": None,
            "normalized_dt_ms_max": None,
            "id_step_histogram": {},
            "pair_index_gap_histogram": {},
        }

    frame_ids = [row["frame_id"] for row in rows]
    timestamps = [row["camera_timestamp"] for row in rows]
    pair_indices = [row["pair_index"] for row in rows]

    if len(rows) < 2:
        return {
            "frames_saved": len(rows),
            "frame_id_first": frame_ids[0],
            "frame_id_last": frame_ids[-1],
            "duration_s": 0.0,
            "saved_fps": None,
            "source_fps": None,
            "skipped_frames": 0,
            "drop_pct_vs_source": 0.0,
            "normalized_dt_ms_mean": None,
            "normalized_dt_ms_std": None,
            "normalized_dt_ms_min": None,
            "normalized_dt_ms_p50": None,
            "normalized_dt_ms_p95": None,
            "normalized_dt_ms_p99": None,
            "normalized_dt_ms_max": None,
            "id_step_histogram": {},
            "pair_index_gap_histogram": {},
        }

    id_diffs = [b - a for a, b in zip(frame_ids, frame_ids[1:])]
    dt_ns = [b - a for a, b in zip(timestamps, timestamps[1:])]
    pair_diffs = [b - a for a, b in zip(pair_indices, pair_indices[1:])]
    duration_s = (timestamps[-1] - timestamps[0]) / 1e9
    total_source_frames = frame_ids[-1] - frame_ids[0] + 1
    skipped_frames = sum(max(0, diff - 1) for diff in id_diffs)
    normalized_dt_ms = [(dt / diff) / 1e6 for dt, diff in zip(dt_ns, id_diffs) if diff > 0]

    return {
        "frames_saved": len(rows),
        "frame_id_first": frame_ids[0],
        "frame_id_last": frame_ids[-1],
        "duration_s": duration_s,
        "saved_fps": ((len(rows) - 1) / duration_s) if duration_s > 0 else None,
        "source_fps": ((total_source_frames - 1) / duration_s) if duration_s > 0 else None,
        "skipped_frames": skipped_frames,
        "drop_pct_vs_source": (100.0 * skipped_frames / total_source_frames) if total_source_frames > 0 else 0.0,
        "normalized_dt_ms_mean": stats.mean(normalized_dt_ms),
        "normalized_dt_ms_std": stats.pstdev(normalized_dt_ms),
        "normalized_dt_ms_min": min(normalized_dt_ms),
        "normalized_dt_ms_p50": _percentile(normalized_dt_ms, 0.50),
        "normalized_dt_ms_p95": _percentile(normalized_dt_ms, 0.95),
        "normalized_dt_ms_p99": _percentile(normalized_dt_ms, 0.99),
        "normalized_dt_ms_max": max(normalized_dt_ms),
        "id_step_histogram": {str(step): id_diffs.count(step) for step in sorted(set(id_diffs))},
        "pair_index_gap_histogram": {str(step): pair_diffs.count(step) for step in sorted(set(pair_diffs))},
    }


def _run_latency_analysis(run_dir: Path, manifest: dict) -> dict:
    csv_path = run_dir / "frame_index.csv"
    rows_by_camera = {0: [], 1: []}

    with open(csv_path, newline="", encoding="utf-8") as csv_fh:
        reader = csv.DictReader(csv_fh)
        for row in reader:
            cam = int(row["camera_index"])
            rows_by_camera[cam].append(
                {
                    "pair_index": int(row["pair_index"]),
                    "frame_index": int(row["frame_index"]),
                    "frame_id": int(row["frame_id"]),
                    "camera_timestamp": int(row["camera_timestamp"]),
                    "host_timestamp_ns": int(row["host_timestamp_ns"]),
                }
            )

    analysis = {
        "run_dir": str(run_dir),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "camera0": _analyze_rows(rows_by_camera[0]),
        "camera1": _analyze_rows(rows_by_camera[1]),
    }
    analysis["no_dropped_frames"] = (
        analysis["camera0"]["skipped_frames"] == 0 and analysis["camera1"]["skipped_frames"] == 0
    )
    analysis["pair_count_match"] = (
        analysis["camera0"]["frames_saved"] == analysis["camera1"]["frames_saved"]
    )

    analysis_path = run_dir / "latency_analysis.json"
    analysis_path.write_text(json.dumps(analysis, indent=2), encoding="utf-8")
    manifest["latency_analysis"] = analysis
    manifest["latency_analysis_path"] = str(analysis_path)

    print("Latency analysis:")
    for cam_idx in (0, 1):
        cam_stats = analysis[f"camera{cam_idx}"]
        fps_txt = "n/a" if cam_stats["source_fps"] is None else f"{cam_stats['source_fps']:.3f}"
        std_txt = "n/a" if cam_stats["normalized_dt_ms_std"] is None else f"{cam_stats['normalized_dt_ms_std']:.6f} ms"
        print(
            f"  Camera {cam_idx}: source_fps={fps_txt} "
            f"skipped={cam_stats['skipped_frames']} jitter_std={std_txt}"
        )
    if analysis["no_dropped_frames"]:
        print("  No dropped frames detected in saved streams.")
    else:
        print("  [warn] Dropped frames detected in saved streams.")
    return analysis


def _codec_candidates():
    for fourcc_name, suffix in LOSSLESS_VIDEO_CANDIDATES:
        yield cv2.VideoWriter_fourcc(*fourcc_name), fourcc_name, suffix


def _convert_bin_to_lossless_video(run_dir: Path, manifest: dict, camera_index: int) -> dict:
    width = int(manifest["roi_width"])
    height = int(manifest["roi_height"])
    cam_stats = manifest.get("latency_analysis", {}).get(f"camera{camera_index}", {})
    fps = cam_stats.get("source_fps") or cam_stats.get("saved_fps") or manifest["cameras"][camera_index].get("resulting_fps")
    if not fps or fps <= 0:
        raise RuntimeError(f"Could not determine fps for camera {camera_index} video conversion.")

    frame_count = int(manifest.get(f"camera{camera_index}_saved_frames", 0))
    bin_path = run_dir / f"cam{camera_index}_frames.bin"
    if frame_count <= 0 or not bin_path.exists():
        raise RuntimeError(f"Missing raw frame stream for camera {camera_index}: {bin_path}")

    frames = np.memmap(bin_path, dtype=np.uint8, mode="r", shape=(frame_count, height, width))
    chosen = None
    writer = None
    video_path = None
    trial_frame = cv2.cvtColor(np.asarray(frames[0]), cv2.COLOR_GRAY2BGR)

    for fourcc, fourcc_name, suffix in _codec_candidates():
        candidate_path = run_dir / f"cam{camera_index}_lossless{suffix}"
        test_writer = cv2.VideoWriter(str(candidate_path), fourcc, float(fps), (width, height), True)
        if not test_writer.isOpened():
            test_writer.release()
            continue
        try:
            test_writer.write(trial_frame)
            chosen = fourcc_name
            writer = test_writer
            video_path = candidate_path
            break
        except Exception:
            test_writer.release()
            continue

    if writer is None or video_path is None or chosen is None:
        raise RuntimeError(
            f"Could not open a lossless VideoWriter for camera {camera_index}. "
            "Tried FFV1 and HFYU via OpenCV."
        )

    try:
        for idx in range(1, frame_count):
            writer.write(cv2.cvtColor(np.asarray(frames[idx]), cv2.COLOR_GRAY2BGR))
    finally:
        writer.release()

    return {
        "camera_index": camera_index,
        "path": str(video_path),
        "codec": chosen,
        "fps": float(fps),
        "frame_count": frame_count,
    }


def _convert_all_bins_to_videos(run_dir: Path, manifest: dict) -> list[dict]:
    videos = []
    for camera_index in (0, 1):
        print(f"Converting camera {camera_index} raw stream to lossless video ...")
        try:
            info = _convert_bin_to_lossless_video(run_dir, manifest, camera_index)
            print(
                f"  Camera {camera_index}: wrote {info['path']} "
                f"[{info['codec']}, {info['fps']:.3f} fps]"
            )
            videos.append(info)
        except Exception as exc:
            print(f"  [warn] Camera {camera_index} video conversion failed: {exc}")
            videos.append({
                "camera_index": camera_index,
                "error": str(exc),
            })
    return videos


def run(
    roi_width: int,
    roi_height: int,
    timeout_ms: int,
    queue_size: int,
    stream_buffer_count: int,
    exposure_us: float | None,
    duration_s: float | None,
    max_pairs: int | None,
) -> None:
    run_dir = SAVE_ROOT / f"max_freerun_roi_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)

    system, cam_list, cams = connect_cameras()
    writer_error: list = []
    frame_q: queue.Queue = queue.Queue(maxsize=queue_size)
    writer_manifest: dict = {
        "format": "raw-mono8-stream",
        "dtype": "uint8",
        "roi_width": roi_width,
        "roi_height": roi_height,
        "timeout_ms": timeout_ms,
        "queue_size": queue_size,
        "stream_buffer_count_requested": stream_buffer_count,
        "run_dir": str(run_dir),
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    writer_thread = threading.Thread(
        target=_writer_worker,
        args=(run_dir, frame_q, writer_manifest, writer_error),
        daemon=True,
    )
    capture_started = False

    try:
        print("\nConfiguring both cameras for maximum free-run ROI recording ...")
        configs = []
        for idx, cam in enumerate(cams):
            model, serial = _camera_identity(cam)
            print(f"Camera {idx}: {model} [S/N {serial}]")
            cfg = _configure_camera_for_max_freerun(
                cam,
                roi_width=roi_width,
                roi_height=roi_height,
                stream_buffer_count=stream_buffer_count,
                exposure_us=exposure_us,
            )
            cfg["model"] = model
            cfg["serial"] = serial
            configs.append(cfg)

            exp_txt = "n/a" if cfg["exposure_us"] is None else f"{cfg['exposure_us']:.0f} us"
            exp_req_txt = (
                "camera minimum"
                if cfg["requested_exposure_us"] is None
                else f"{cfg['requested_exposure_us']:.0f} us"
            )
            gain_txt = "n/a" if cfg["gain_value"] is None else f"{cfg['gain_value']:.3f}"
            tgt_txt = "camera max" if cfg["target_fps"] is None else f"{cfg['target_fps']:.2f} fps"
            res_txt = "n/a" if cfg["resulting_fps"] is None else f"{cfg['resulting_fps']:.2f} fps"
            buf_txt = "n/a" if cfg["stream_buffer_count"] is None else str(cfg["stream_buffer_count"])
            print(
                f"  ROI {cfg['roi_w']}x{cfg['roi_h']} at offset ({cfg['off_x']}, {cfg['off_y']})\n"
                f"  Exposure request {exp_req_txt} -> applied {exp_txt}\n"
                f"  Gain {gain_txt}\n"
                f"  Free-run target {tgt_txt} [{cfg['fps_mode']}]\n"
                f"  Resulting frame rate {res_txt}\n"
                f"  Stream buffer count {buf_txt}"
            )

        sync_limit = _estimate_sync_freerun_limit(configs)
        writer_manifest["cameras"] = configs
        writer_manifest["synchronized_freerun_limit"] = sync_limit

        sync_txt = "n/a"
        if sync_limit["synchronized_freerun_fps"] is not None:
            sync_txt = (
                f"{sync_limit['synchronized_freerun_fps']:.2f} fps "
                f"(camera {sync_limit['synchronized_freerun_bottleneck_camera']})"
            )

        exposure_limit_txt = "n/a"
        if sync_limit["exposure_only_fps_limit"] is not None:
            exposure_limit_txt = (
                f"{sync_limit['exposure_only_fps_limit']:.2f} fps "
                f"(camera {sync_limit['exposure_only_bottleneck_camera']})"
            )

        print("Synchronized free-run ceiling under current exposure:")
        print(f"  Camera-reported synchronized max {sync_txt}")
        print(f"  Exposure-only upper bound {exposure_limit_txt}")

        writer_thread.start()
        capture_started = True

        for cam in cams:
            cam.BeginAcquisition()

        print(f"\nRecording to {run_dir}")
        print("Press Ctrl+C to stop.\n")

        pair_index = 0
        dropped_pairs = 0
        t_start = time.perf_counter()
        t_last = t_start
        pairs_in_window = 0

        while True:
            if writer_error:
                raise RuntimeError(f"Writer thread failed: {writer_error[0]}")

            elapsed_total = time.perf_counter() - t_start
            if duration_s is not None and elapsed_total >= duration_s:
                print("Requested duration reached.")
                break
            if max_pairs is not None and pair_index >= max_pairs:
                print("Requested pair count reached.")
                break

            packet0, packet1 = grab_pair_mono(cams, timeout_ms)
            if packet0 is None or packet1 is None:
                dropped_pairs += 1
                continue

            _enqueue_pair(
                frame_q,
                {"pair_index": pair_index, "packets": (packet0, packet1)},
                writer_error,
            )
            pair_index += 1
            pairs_in_window += 1

            elapsed_window = time.perf_counter() - t_last
            if elapsed_window >= 1.0:
                print(
                    f"  acquired {pairs_in_window / elapsed_window:.1f} pair/s  "
                    f"queued {frame_q.qsize()}  dropped {dropped_pairs}"
                )
                pairs_in_window = 0
                t_last = time.perf_counter()

    except KeyboardInterrupt:
        print("\nStopping on Ctrl+C ...")
    finally:
        print("Flushing writer queue ...")
        if writer_thread.is_alive():
            while True:
                try:
                    frame_q.put(None, timeout=0.5)
                    break
                except queue.Full:
                    if writer_error or not writer_thread.is_alive():
                        break
            writer_thread.join(timeout=30.0)

        if writer_error:
            print(f"  [warn] Writer thread failed: {writer_error[0]}")

        if not writer_error and capture_started and (run_dir / "frame_index.csv").exists():
            latency_analysis = _run_latency_analysis(run_dir, writer_manifest)
            writer_manifest["no_dropped_frames"] = latency_analysis["no_dropped_frames"]
            writer_manifest["lossless_videos"] = _convert_all_bins_to_videos(run_dir, writer_manifest)
            (run_dir / "manifest.json").write_text(json.dumps(writer_manifest, indent=2), encoding="utf-8")
        elif not writer_error:
            print("  [warn] Skipping latency analysis because capture never started successfully.")

        print("Stopping cameras ...")
        # PySpin camera objects are reference-counted. Loop variables and other
        # lingering locals can keep a camera alive until function exit unless
        # they are cleared explicitly before ReleaseInstance().
        cam = None
        cfg = None
        configs = None
        packet0 = None
        packet1 = None
        latency_analysis = None
        gc.collect()
        release_cameras(system, cam_list, cams, restore_daq=False)
        gc.collect()
        print("Done.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Record from both Flea3 cameras on a centered ROI at maximum free-run rate."
    )
    parser.add_argument(
        "--width",
        type=int,
        default=DEFAULT_ROI_WIDTH,
        help=f"Centered ROI width in pixels (default: {DEFAULT_ROI_WIDTH}).",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=DEFAULT_ROI_HEIGHT,
        help=f"Centered ROI height in pixels (default: {DEFAULT_ROI_HEIGHT}).",
    )
    parser.add_argument(
        "--timeout-ms",
        type=int,
        default=DEFAULT_TIMEOUT_MS,
        help=f"Per-frame camera timeout in ms (default: {DEFAULT_TIMEOUT_MS}).",
    )
    parser.add_argument(
        "--queue-size",
        type=int,
        default=DEFAULT_QUEUE_SIZE,
        help=f"Capture-to-writer queue size (default: {DEFAULT_QUEUE_SIZE}).",
    )
    parser.add_argument(
        "--stream-buffer-count",
        type=int,
        default=DEFAULT_STREAM_BUFFER_COUNT,
        help=(
            "Requested camera stream buffer count to absorb disk bursts "
            f"(default: {DEFAULT_STREAM_BUFFER_COUNT})."
        ),
    )
    parser.add_argument(
        "--exposure-us",
        type=float,
        default=None,
        help="Fixed exposure in microseconds. Default: camera minimum.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Optional recording duration in seconds. Default: run until Ctrl+C.",
    )
    parser.add_argument(
        "--max-pairs",
        type=int,
        default=None,
        help="Optional maximum number of stereo pairs to record.",
    )
    args = parser.parse_args()

    if args.width <= 0 or args.height <= 0:
        raise SystemExit("ROI width and height must be positive integers.")
    if args.timeout_ms <= 0:
        raise SystemExit("--timeout-ms must be > 0.")
    if args.queue_size <= 0:
        raise SystemExit("--queue-size must be > 0.")
    if args.stream_buffer_count <= 0:
        raise SystemExit("--stream-buffer-count must be > 0.")
    if args.exposure_us is not None and args.exposure_us <= 0:
        raise SystemExit("--exposure-us must be > 0 when provided.")
    if args.duration is not None and args.duration <= 0:
        raise SystemExit("--duration must be > 0 when provided.")
    if args.max_pairs is not None and args.max_pairs <= 0:
        raise SystemExit("--max-pairs must be > 0 when provided.")

    run(
        roi_width=args.width,
        roi_height=args.height,
        timeout_ms=args.timeout_ms,
        queue_size=args.queue_size,
        stream_buffer_count=args.stream_buffer_count,
        exposure_us=args.exposure_us,
        duration_s=args.duration,
        max_pairs=args.max_pairs,
    )


if __name__ == "__main__":
    main()
