from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from multibios.blackfly.triggered_camera_record import LOSSLESS_VIDEO_CANDIDATES


def _codec_candidates():
    for fourcc_name, suffix in LOSSLESS_VIDEO_CANDIDATES:
        yield cv2.VideoWriter_fourcc(*fourcc_name), fourcc_name, suffix


def _discover_manifest(run_dir: Path) -> Path | None:
    manifests = sorted(
        path
        for path in run_dir.glob("fictrac-raw-*.json")
        if not path.name.endswith("recording.json")
    )
    return manifests[-1] if manifests else None


def _read_rows(csv_path: Path) -> list[dict[str, int]]:
    rows: list[dict[str, int]] = []
    with open(csv_path, newline="", encoding="utf-8") as csv_fh:
        reader = csv.DictReader(csv_fh)
        for row in reader:
            frame_index = row.get("frame_index")
            log_frame = row.get("log_frame")
            chunk_index = row.get("chunk_index")
            chunk_frame_index = row.get("chunk_frame_index")
            if None in (frame_index, log_frame, chunk_index, chunk_frame_index):
                continue
            rows.append(
                {
                    "frame_index": int(frame_index),
                    "log_frame": int(log_frame),
                    "chunk_index": int(chunk_index),
                    "chunk_frame_index": int(chunk_frame_index),
                }
            )
    return rows


def _resolve_path(path_str: str, run_dir: Path) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return (run_dir / path).resolve()


def _analyze_rows(rows: list[dict[str, int]], expected_frame_count: int | None) -> dict[str, Any]:
    frames_saved = len(rows)
    first_log_frame = rows[0]["log_frame"] if rows else None
    last_log_frame = rows[-1]["log_frame"] if rows else None
    skipped_log_frames = 0
    for prev, curr in zip(rows, rows[1:]):
        skipped_log_frames += max(curr["log_frame"] - prev["log_frame"] - 1, 0)

    missing_frames = None
    no_dropped_frames = None
    if expected_frame_count is not None:
        missing_frames = max(int(expected_frame_count) - frames_saved, 0)
        no_dropped_frames = frames_saved == int(expected_frame_count)

    return {
        "frames_saved": frames_saved,
        "first_log_frame": first_log_frame,
        "last_log_frame": last_log_frame,
        "skipped_log_frames": skipped_log_frames,
        "missing_frames_vs_expected": missing_frames,
        "no_dropped_frames": no_dropped_frames,
    }


def _convert_chunks_to_lossless_video(
    manifest: dict[str, Any],
    *,
    run_dir: Path,
    nominal_fps: float | None,
) -> dict[str, Any]:
    width = int(manifest["frame_width"])
    height = int(manifest["frame_height"])
    channels = int(manifest.get("channels", 3))
    frame_bytes = width * height * channels
    fps = float(manifest.get("fps") or 0.0)
    if fps <= 0:
        if nominal_fps is None or float(nominal_fps) <= 0:
            raise RuntimeError("Could not determine fps for FicTrac lossless conversion.")
        fps = float(nominal_fps)

    stem = Path(str(manifest.get("manifest_path", run_dir / "fictrac-raw"))).stem
    base_stem = stem.replace(".json", "") + "-lossless"
    writer = None
    output_path: Path | None = None
    codec_name: str | None = None
    for fourcc, candidate_name, suffix in _codec_candidates():
        candidate_path = run_dir / f"{base_stem}{suffix}"
        candidate = cv2.VideoWriter(
            str(candidate_path),
            fourcc,
            fps,
            (width, height),
            isColor=(channels != 1),
        )
        if candidate.isOpened():
            writer = candidate
            output_path = candidate_path
            codec_name = candidate_name
            break
        candidate.release()
    if writer is None or output_path is None or codec_name is None:
        raise RuntimeError("Could not open a lossless VideoWriter for FicTrac conversion.")

    frames_written = 0
    try:
        for chunk_path_str in manifest.get("chunk_paths", []):
            chunk_path = _resolve_path(str(chunk_path_str), run_dir)
            chunk_size = chunk_path.stat().st_size
            if chunk_size == 0:
                continue
            chunk_frames = chunk_size // frame_bytes
            if chunk_frames == 0:
                continue
            chunk = np.memmap(
                chunk_path,
                dtype=np.uint8,
                mode="r",
                shape=(chunk_frames, height, width, channels),
            )
            for frame in chunk:
                frame_array = np.asarray(frame)
                if channels == 1:
                    writer.write(cv2.cvtColor(frame_array.reshape(height, width), cv2.COLOR_GRAY2BGR))
                else:
                    writer.write(frame_array)
                frames_written += 1
            del chunk
    finally:
        writer.release()

    return {
        "path": str(output_path),
        "codec": codec_name,
        "fps": fps,
        "frames_written": frames_written,
    }


def postprocess_fictrac_raw_recording(
    *,
    run_dir: Path,
    runtime_info: dict[str, Any],
    frame_count: int | None,
    expected_frame_count: int | None,
    legacy_raw_videos: list[str],
    legacy_saved_raw_frames: int | None,
) -> dict[str, Any]:
    callback_frames = None if frame_count is None else int(frame_count)
    manifest_path = _discover_manifest(run_dir)
    if manifest_path is None:
        saved_raw_frames = legacy_saved_raw_frames
        actual_frames = saved_raw_frames if saved_raw_frames is not None else callback_frames
        missing_frames = None
        no_dropped_frames = None
        if expected_frame_count is not None and actual_frames is not None:
            missing_frames = max(int(expected_frame_count) - actual_frames, 0)
            no_dropped_frames = actual_frames == int(expected_frame_count)
        return {
            "camera_index": runtime_info.get("fictrac_camera_index"),
            "save_raw": bool(runtime_info.get("save_raw", False)),
            "video_codec": runtime_info.get("video_codec"),
            "camera_fps": runtime_info.get("camera_fps"),
            "output_base": runtime_info.get("output_base"),
            "raw_videos": legacy_raw_videos,
            "raw_stream_manifest": None,
            "raw_stream_chunks": [],
            "callback_frames": callback_frames,
            "saved_raw_frames": saved_raw_frames,
            "actual_frames": actual_frames,
            "expected_frames": expected_frame_count,
            "missing_frames_vs_expected": missing_frames,
            "no_dropped_frames": no_dropped_frames,
        }

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    index_path = _resolve_path(str(manifest["frame_index_path"]), run_dir)
    rows = _read_rows(index_path)
    analysis = _analyze_rows(rows, expected_frame_count)
    nominal_fps = runtime_info.get("camera_fps")
    lossless_video = None
    if analysis["frames_saved"] > 0:
        lossless_video = _convert_chunks_to_lossless_video(manifest, run_dir=run_dir, nominal_fps=nominal_fps)

    saved_raw_frames = analysis["frames_saved"]
    if lossless_video is not None:
        saved_raw_frames = max(saved_raw_frames, int(lossless_video["frames_written"]))

    missing_frames = None
    no_dropped_frames = analysis["no_dropped_frames"]
    if expected_frame_count is not None:
        missing_frames = max(int(expected_frame_count) - saved_raw_frames, 0)
        no_dropped_frames = saved_raw_frames == int(expected_frame_count)

    return {
        "camera_index": runtime_info.get("fictrac_camera_index"),
        "save_raw": bool(runtime_info.get("save_raw", False)),
        "video_codec": runtime_info.get("video_codec"),
        "camera_fps": runtime_info.get("camera_fps"),
        "output_base": runtime_info.get("output_base"),
        "raw_videos": legacy_raw_videos + ([lossless_video["path"]] if lossless_video else []),
        "raw_stream_manifest": str(manifest_path),
        "raw_stream_format": manifest.get("format"),
        "raw_stream_chunks": [str(_resolve_path(str(path), run_dir)) for path in manifest.get("chunk_paths", [])],
        "callback_frames": callback_frames,
        "saved_raw_frames": saved_raw_frames,
        "actual_frames": saved_raw_frames,
        "expected_frames": expected_frame_count,
        "missing_frames_vs_expected": missing_frames,
        "no_dropped_frames": no_dropped_frames,
        "first_log_frame": analysis["first_log_frame"],
        "last_log_frame": analysis["last_log_frame"],
        "skipped_log_frames": analysis["skipped_log_frames"],
        "lossless_video": lossless_video,
    }