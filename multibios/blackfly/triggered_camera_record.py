from __future__ import annotations

import csv
import json
import queue
import threading
import time
from pathlib import Path
from typing import Any


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
        return {
            "configured_width": int(width_n.GetValue()) if PySpin.IsReadable(width_n) else None,
            "configured_height": int(height_n.GetValue()) if PySpin.IsReadable(height_n) else None,
            "configured_offset_x": int(offset_x_n.GetValue()) if PySpin.IsReadable(offset_x_n) else None,
            "configured_offset_y": int(offset_y_n.GetValue()) if PySpin.IsReadable(offset_y_n) else None,
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

        if self._cam_list is not None:
            try:
                self._cam_list.Clear()
            except Exception:
                pass
        self._cam_list = None

        if self._system is not None:
            try:
                self._system.ReleaseInstance()
            except Exception:
                pass
        self._system = None