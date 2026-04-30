from __future__ import annotations

import csv
import gc
import json
import math
import queue
import statistics as stats
import threading
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np


LOSSLESS_VIDEO_CANDIDATES = [
    ("FFV1", ".avi"),
    ("HFYU", ".avi"),
    ("FFV1", ".mkv"),
]


_LEAKED_PYSPIN_REFS: list[object] = []


class TriggeredCameraRecorder:
    """Record one hardware-triggered Blackfly camera into a raw frame stream."""

    def __init__(
        self,
        *,
        camera_index: int,
        run_dir: str | Path,
        timeout_ms: int = 250,
        queue_size: int = 512,
        stream_buffer_count: int = 256,
        exposure_us: float | None = None,
        roi_width: int | None = None,
        roi_height: int | None = None,
        binning: int = 1,
        gain_db: float | None = None,
        gamma: float | None = None,
    ) -> None:
        self.camera_index = int(camera_index)
        self.run_dir = Path(run_dir)
        self.timeout_ms = max(1, int(timeout_ms))
        self.queue_size = max(1, int(queue_size))
        self.stream_buffer_count = max(1, int(stream_buffer_count))
        self.exposure_us = exposure_us
        self.roi_width = roi_width
        self.roi_height = roi_height
        self.binning = max(1, int(binning))
        self.gain_db = gain_db
        self.gamma = gamma

        stem = f"blackfly_cam{self.camera_index}"
        self._bin_path = self.run_dir / f"{stem}_frames.bin"
        self._csv_path = self.run_dir / f"{stem}_frame_index.csv"
        self._manifest_path = self.run_dir / f"{stem}_manifest.json"

        self._frame_q: queue.Queue[dict[str, Any] | None] = queue.Queue(maxsize=self.queue_size)
        self._stop_event = threading.Event()
        self._capture_thread: threading.Thread | None = None
        self._writer_thread: threading.Thread | None = None

        self._system = None
        self._cam_list = None
        self._cam = None
        self._PySpin = None

        self._capture_error: Exception | None = None
        self._writer_error: Exception | None = None
        self._manifest: dict[str, Any] = {
            "camera_index": self.camera_index,
            "format": "raw-mono8-stream",
            "dtype": "uint8",
            "timeout_ms": self.timeout_ms,
            "queue_size": self.queue_size,
            "stream_buffer_count_requested": self.stream_buffer_count,
            "binning": self.binning,
            "requested_exposure_us": self.exposure_us,
            "requested_roi_width": self.roi_width,
            "requested_roi_height": self.roi_height,
            "requested_gain_db": self.gain_db,
            "requested_gamma": self.gamma,
            "frame_bin_path": str(self._bin_path),
            "frame_index_path": str(self._csv_path),
            "manifest_path": str(self._manifest_path),
            "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

    @property
    def manifest_path(self) -> Path:
        return self._manifest_path

    @property
    def manifest(self) -> dict[str, Any]:
        return dict(self._manifest)

    def start(self) -> dict[str, Any]:
        if self._capture_thread is not None:
            raise RuntimeError("TriggeredCameraRecorder.start() called more than once.")

        self.run_dir.mkdir(parents=True, exist_ok=True)

        try:
            import PySpin  # type: ignore

            from .live_view import configure_camera_daq_mode

            self._PySpin = PySpin
            self._system = PySpin.System.GetInstance()
            self._cam_list = self._system.GetCameras()
            camera_count = self._cam_list.GetSize()
            if camera_count <= self.camera_index:
                raise RuntimeError(
                    f"Requested camera index {self.camera_index}, but only {camera_count} camera(s) were found."
                )

            self._cam = self._cam_list.GetByIndex(self.camera_index)
            self._cam.Init()

            self._manifest.update(self._read_camera_identity())
            self._configure_stream_buffer(self.stream_buffer_count)
            configure_camera_daq_mode(
                self._cam,
                exposure_us=self.exposure_us,
                roi_width=self.roi_width,
                roi_height=self.roi_height,
                binning=self.binning,
                gain_db=self.gain_db,
                gamma=self.gamma,
            )
            self._manifest.update(self._read_camera_geometry())

            self._writer_thread = threading.Thread(
                target=self._writer_loop,
                name=f"BlackflyWriter-{self.camera_index}",
                daemon=True,
            )
            self._capture_thread = threading.Thread(
                target=self._capture_loop,
                name=f"BlackflyCapture-{self.camera_index}",
                daemon=True,
            )

            self._writer_thread.start()
            self._cam.BeginAcquisition()
            self._capture_thread.start()
            return self.manifest
        except Exception:
            self._cleanup_camera()
            raise

    def stop(self) -> dict[str, Any]:
        self._stop_event.set()

        if self._capture_thread is not None:
            self._capture_thread.join(timeout=10.0)
        self._cleanup_camera()

        if self._writer_thread is not None:
            self._enqueue_writer_sentinel()
            self._writer_thread.join(timeout=30.0)

        self._manifest["completed_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        self._manifest_path.write_text(json.dumps(self._manifest, indent=2), encoding="utf-8")
        self.raise_if_failed()
        return self.manifest

    def raise_if_failed(self) -> None:
        if self._capture_error is not None:
            raise RuntimeError(f"Camera {self.camera_index} capture failed: {self._capture_error}")
        if self._writer_error is not None:
            raise RuntimeError(f"Camera {self.camera_index} writer failed: {self._writer_error}")

    def _enqueue_writer_sentinel(self) -> None:
        while True:
            try:
                self._frame_q.put(None, timeout=0.5)
                return
            except queue.Full:
                if self._writer_error is not None:
                    return

    def _capture_loop(self) -> None:
        assert self._cam is not None
        PySpin = self._PySpin
        saw_frame_after_stop = False

        while True:
            try:
                image = self._cam.GetNextImage(self.timeout_ms)
            except PySpin.SpinnakerException as exc:
                if self._stop_event.is_set():
                    break
                message = str(exc).lower()
                if "timeout" in message:
                    continue
                self._capture_error = exc
                return

            try:
                if image.IsIncomplete():
                    self._manifest["incomplete_frames"] = int(self._manifest.get("incomplete_frames", 0)) + 1
                    continue

                packet = {
                    "frame_index": int(self._manifest.get("saved_frames", 0)),
                    "frame_id": int(image.GetFrameID()),
                    "camera_timestamp": int(image.GetTimeStamp()),
                    "host_timestamp_ns": time.time_ns(),
                    "frame": image.GetNDArray().copy(),
                }
                self._enqueue_packet(packet)
                saw_frame_after_stop = saw_frame_after_stop or self._stop_event.is_set()
            finally:
                image.Release()

            if self._stop_event.is_set() and saw_frame_after_stop:
                # After stop() we keep draining until the first timeout.
                saw_frame_after_stop = True

    def _enqueue_packet(self, packet: dict[str, Any]) -> None:
        while True:
            try:
                self._frame_q.put(packet, timeout=0.5)
                return
            except queue.Full:
                if self._stop_event.is_set() and self._writer_error is not None:
                    return
                if self._writer_error is not None:
                    return

    def _writer_loop(self) -> None:
        saved_frames = 0
        first_frame_id: int | None = None
        last_frame_id: int | None = None

        try:
            with open(self._bin_path, "wb") as bin_fh, open(
                self._csv_path, "w", newline="", encoding="utf-8"
            ) as csv_fh:
                writer = csv.writer(csv_fh)
                writer.writerow(
                    [
                        "frame_index",
                        "frame_id",
                        "camera_timestamp",
                        "host_timestamp_ns",
                    ]
                )

                while True:
                    item = self._frame_q.get()
                    if item is None:
                        self._frame_q.task_done()
                        break

                    frame = item["frame"]
                    if first_frame_id is None:
                        first_frame_id = int(item["frame_id"])
                        self._manifest["frame_width"] = int(frame.shape[1])
                        self._manifest["frame_height"] = int(frame.shape[0])
                    last_frame_id = int(item["frame_id"])

                    bin_fh.write(frame.tobytes(order="C"))
                    writer.writerow(
                        [
                            saved_frames,
                            item["frame_id"],
                            item["camera_timestamp"],
                            item["host_timestamp_ns"],
                        ]
                    )
                    saved_frames += 1
                    self._manifest["saved_frames"] = saved_frames
                    self._frame_q.task_done()

            self._manifest["frame_id_first"] = first_frame_id
            self._manifest["frame_id_last"] = last_frame_id
        except Exception as exc:
            self._writer_error = exc

    def _read_camera_identity(self) -> dict[str, Any]:
        assert self._cam is not None
        PySpin = self._PySpin
        tl = self._cam.GetTLDeviceNodeMap()
        model_n = PySpin.CStringPtr(tl.GetNode("DeviceModelName"))
        serial_n = PySpin.CStringPtr(tl.GetNode("DeviceSerialNumber"))
        model = model_n.GetValue() if PySpin.IsReadable(model_n) else "?"
        serial = serial_n.GetValue() if PySpin.IsReadable(serial_n) else "?"
        return {"model": model, "serial": serial}

    def _read_camera_geometry(self) -> dict[str, Any]:
        assert self._cam is not None
        PySpin = self._PySpin
        nm = self._cam.GetNodeMap()
        width_n = PySpin.CIntegerPtr(nm.GetNode("Width"))
        height_n = PySpin.CIntegerPtr(nm.GetNode("Height"))
        offset_x_n = PySpin.CIntegerPtr(nm.GetNode("OffsetX"))
        offset_y_n = PySpin.CIntegerPtr(nm.GetNode("OffsetY"))
        gain_n = PySpin.CFloatPtr(nm.GetNode("Gain"))
        gamma_n = PySpin.CFloatPtr(nm.GetNode("Gamma"))
        return {
            "configured_width": int(width_n.GetValue()) if PySpin.IsReadable(width_n) else None,
            "configured_height": int(height_n.GetValue()) if PySpin.IsReadable(height_n) else None,
            "configured_offset_x": int(offset_x_n.GetValue()) if PySpin.IsReadable(offset_x_n) else None,
            "configured_offset_y": int(offset_y_n.GetValue()) if PySpin.IsReadable(offset_y_n) else None,
            "configured_gain_db": float(gain_n.GetValue()) if PySpin.IsReadable(gain_n) else None,
            "configured_gamma": float(gamma_n.GetValue()) if PySpin.IsReadable(gamma_n) else None,
        }

    def _configure_stream_buffer(self, count: int) -> None:
        assert self._cam is not None
        PySpin = self._PySpin
        sn = self._cam.GetTLStreamNodeMap()
        handling = PySpin.CEnumerationPtr(sn.GetNode("StreamBufferHandlingMode"))
        if PySpin.IsReadable(handling) and PySpin.IsWritable(handling):
            entry = handling.GetEntryByName("OldestFirst")
            if PySpin.IsReadable(entry):
                handling.SetIntValue(entry.GetValue())

        count_mode = PySpin.CEnumerationPtr(sn.GetNode("StreamBufferCountMode"))
        if PySpin.IsReadable(count_mode) and PySpin.IsWritable(count_mode):
            entry = count_mode.GetEntryByName("Manual")
            if PySpin.IsReadable(entry):
                count_mode.SetIntValue(entry.GetValue())

        manual = PySpin.CIntegerPtr(sn.GetNode("StreamBufferCountManual"))
        if PySpin.IsReadable(manual) and PySpin.IsWritable(manual):
            clamped = max(int(manual.GetMin()), min(int(manual.GetMax()), int(count)))
            manual.SetValue(clamped)
            self._manifest["stream_buffer_count"] = int(manual.GetValue())

    def _cleanup_camera(self) -> None:
        cam = self._cam
        if cam is not None:
            try:
                cam.EndAcquisition()
            except Exception:
                pass
            try:
                cam.DeInit()
            except Exception:
                pass
        self._cam = None
        if cam is not None:
            del cam
        gc.collect()

        if self._cam_list is not None:
            try:
                self._cam_list.Clear()
            except Exception:
                pass
            _LEAKED_PYSPIN_REFS.append(self._cam_list)

        # Dropping the last references to these PySpin objects can abort the
        # interpreter on this rig, so keep them alive until process teardown.
        if self._system is not None:
            _LEAKED_PYSPIN_REFS.append(self._system)
        if self._PySpin is not None:
            _LEAKED_PYSPIN_REFS.append(self._PySpin)


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


def _codec_candidates():
    for fourcc_name, suffix in LOSSLESS_VIDEO_CANDIDATES:
        yield cv2.VideoWriter_fourcc(*fourcc_name), fourcc_name, suffix


def _read_rows(csv_path: Path) -> list[dict[str, int]]:
    rows: list[dict[str, int]] = []
    with open(csv_path, newline="", encoding="utf-8") as csv_fh:
        reader = csv.DictReader(csv_fh)
        for row in reader:
            rows.append(
                {
                    "frame_index": int(row["frame_index"]),
                    "frame_id": int(row["frame_id"]),
                    "camera_timestamp": int(row["camera_timestamp"]),
                    "host_timestamp_ns": int(row["host_timestamp_ns"]),
                }
            )
    return rows


def _analyze_rows(rows: list[dict[str, int]], expected_frame_count: int | None) -> dict[str, Any]:
    frames_saved = len(rows)
    if frames_saved == 0:
        no_dropped_frames = expected_frame_count in (None, 0)
        missing_frames = None if expected_frame_count is None else max(int(expected_frame_count), 0)
        return {
            "frames_saved": 0,
            "frame_id_first": None,
            "frame_id_last": None,
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
            "expected_frame_count": expected_frame_count,
            "missing_frames_vs_expected": missing_frames,
            "frame_count_matches_expected": no_dropped_frames,
            "no_dropped_frames": no_dropped_frames,
        }

    frame_ids = [row["frame_id"] for row in rows]
    timestamps = [row["camera_timestamp"] for row in rows]
    id_diffs = [b - a for a, b in zip(frame_ids, frame_ids[1:])]
    dt_ns = [b - a for a, b in zip(timestamps, timestamps[1:])]
    skipped_frames = sum(max(0, diff - 1) for diff in id_diffs)
    duration_s = (timestamps[-1] - timestamps[0]) / 1e9 if frames_saved > 1 else 0.0
    total_source_frames = frame_ids[-1] - frame_ids[0] + 1 if frames_saved > 0 else 0
    normalized_dt_ms = [(dt / diff) / 1e6 for dt, diff in zip(dt_ns, id_diffs) if diff > 0]
    missing_frames = None if expected_frame_count is None else max(int(expected_frame_count) - frames_saved, 0)
    frame_count_matches_expected = expected_frame_count is None or frames_saved == int(expected_frame_count)
    no_dropped_frames = skipped_frames == 0 and frame_count_matches_expected

    return {
        "frames_saved": frames_saved,
        "frame_id_first": frame_ids[0],
        "frame_id_last": frame_ids[-1],
        "duration_s": duration_s,
        "saved_fps": ((frames_saved - 1) / duration_s) if duration_s > 0 and frames_saved > 1 else None,
        "source_fps": ((total_source_frames - 1) / duration_s) if duration_s > 0 and total_source_frames > 1 else None,
        "skipped_frames": skipped_frames,
        "drop_pct_vs_source": (100.0 * skipped_frames / total_source_frames) if total_source_frames > 0 else 0.0,
        "normalized_dt_ms_mean": stats.mean(normalized_dt_ms) if normalized_dt_ms else None,
        "normalized_dt_ms_std": stats.pstdev(normalized_dt_ms) if len(normalized_dt_ms) > 1 else 0.0 if normalized_dt_ms else None,
        "normalized_dt_ms_min": min(normalized_dt_ms) if normalized_dt_ms else None,
        "normalized_dt_ms_p50": _percentile(normalized_dt_ms, 0.50),
        "normalized_dt_ms_p95": _percentile(normalized_dt_ms, 0.95),
        "normalized_dt_ms_p99": _percentile(normalized_dt_ms, 0.99),
        "normalized_dt_ms_max": max(normalized_dt_ms) if normalized_dt_ms else None,
        "expected_frame_count": expected_frame_count,
        "missing_frames_vs_expected": missing_frames,
        "frame_count_matches_expected": frame_count_matches_expected,
        "no_dropped_frames": no_dropped_frames,
    }


def _convert_bin_to_lossless_video(
    manifest: dict[str, Any],
    analysis: dict[str, Any],
    nominal_fps: float | None,
) -> dict[str, Any]:
    run_dir = Path(manifest["manifest_path"]).parent
    frame_count = int(manifest.get("saved_frames", 0))
    source_width = int(manifest["frame_width"])
    source_height = int(manifest["frame_height"])
    requested_width = manifest.get("requested_roi_width")
    requested_height = manifest.get("requested_roi_height")
    crop_width = (
        int(requested_width)
        if requested_width is not None and 0 < int(requested_width) <= source_width
        else source_width
    )
    crop_height = (
        int(requested_height)
        if requested_height is not None and 0 < int(requested_height) <= source_height
        else source_height
    )
    crop_x = max(0, (source_width - crop_width) // 2)
    crop_y = max(0, (source_height - crop_height) // 2)
    bin_path = Path(manifest["frame_bin_path"])
    if frame_count <= 0 or not bin_path.exists():
        raise RuntimeError(f"Missing raw frame stream for conversion: {bin_path}")

    fps = analysis.get("source_fps") or analysis.get("saved_fps") or nominal_fps
    if not fps or float(fps) <= 0:
        raise RuntimeError("Could not determine fps for second-camera lossless conversion.")

    frames = np.memmap(bin_path, dtype=np.uint8, mode="r", shape=(frame_count, source_height, source_width))

    def _frame_bgr(index: int) -> np.ndarray:
        frame = np.asarray(frames[index])
        if crop_width != source_width or crop_height != source_height:
            frame = frame[crop_y:crop_y + crop_height, crop_x:crop_x + crop_width]
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

    trial_frame = _frame_bgr(0)

    writer = None
    chosen = None
    video_path = None
    stem = f"blackfly_cam{manifest['camera_index']}_lossless"
    for fourcc, fourcc_name, suffix in _codec_candidates():
        candidate_path = run_dir / f"{stem}{suffix}"
        test_writer = cv2.VideoWriter(str(candidate_path), fourcc, float(fps), (crop_width, crop_height), True)
        if not test_writer.isOpened():
            test_writer.release()
            continue
        try:
            test_writer.write(trial_frame)
            writer = test_writer
            chosen = fourcc_name
            video_path = candidate_path
            break
        except Exception:
            test_writer.release()

    if writer is None or chosen is None or video_path is None:
        raise RuntimeError("Could not open a lossless VideoWriter for second-camera conversion.")

    try:
        for idx in range(1, frame_count):
            writer.write(_frame_bgr(idx))
    finally:
        writer.release()

    return {
        "path": str(video_path),
        "codec": chosen,
        "fps": float(fps),
        "frame_count": frame_count,
        "width": crop_width,
        "height": crop_height,
        "cropped_from_width": source_width,
        "cropped_from_height": source_height,
    }


def postprocess_triggered_camera_recording(
    manifest: dict[str, Any],
    *,
    expected_frame_count: int | None = None,
    nominal_fps: float | None = None,
    convert_to_lossless_mkv: bool = True,
) -> dict[str, Any]:
    recording = dict(manifest)
    csv_path = Path(recording["frame_index_path"])
    analysis = _analyze_rows(_read_rows(csv_path), expected_frame_count)
    analysis_path = Path(recording["manifest_path"]).with_name(
        f"blackfly_cam{recording['camera_index']}_analysis.json"
    )
    analysis_path.write_text(json.dumps(analysis, indent=2), encoding="utf-8")

    recording["expected_frame_count"] = expected_frame_count
    recording["nominal_trigger_fps"] = nominal_fps
    recording["analysis_path"] = str(analysis_path)
    recording["analysis"] = analysis
    recording["no_dropped_frames"] = analysis["no_dropped_frames"]

    if convert_to_lossless_mkv and int(recording.get("saved_frames", 0)) > 0:
        recording["lossless_video"] = _convert_bin_to_lossless_video(recording, analysis, nominal_fps)

    Path(recording["manifest_path"]).write_text(json.dumps(recording, indent=2), encoding="utf-8")
    return recording