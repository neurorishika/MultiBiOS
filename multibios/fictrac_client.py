from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Protocol, Tuple

import numpy as np

from multibios.fictrac_runtime import build_fictrac_subprocess_env


@dataclass(slots=True)
class FicTracState:
    """Parsed FicTrac UDP state with the full upstream variable set."""

    frame_cnt: int
    del_rot_cam_x: float = 0.0
    del_rot_cam_y: float = 0.0
    del_rot_cam_z: float = 0.0
    del_rot_error: float = 0.0
    del_rot_lab_x: float = 0.0
    del_rot_lab_y: float = 0.0
    del_rot_lab_z: float = 0.0
    abs_ori_cam_x: float = 0.0
    abs_ori_cam_y: float = 0.0
    abs_ori_cam_z: float = 0.0
    abs_ori_lab_x: float = 0.0
    abs_ori_lab_y: float = 0.0
    abs_ori_lab_z: float = 0.0
    posx: float = 0.0
    posy: float = 0.0
    heading: float = 0.0
    direction: float = 0.0
    speed: float = 0.0
    intx: float = 0.0
    inty: float = 0.0
    timestamp: float = 0.0
    seq_num: int = 0
    delta_timestamp: float = 0.0
    alt_timestamp: float = 0.0

    @classmethod
    def from_udp_message(cls, payload: str) -> "FicTracState":
        values = [part.strip() for part in payload.split(",")]
        if values and values[0].upper() == "FT":
            values = values[1:]

        if len(values) not in (24, 25):
            raise ValueError(
                f"Message from FicTrac did not have the expected number of fields: {len(values)}"
            )

        return cls(
            frame_cnt=int(values[0]),
            del_rot_cam_x=float(values[1]),
            del_rot_cam_y=float(values[2]),
            del_rot_cam_z=float(values[3]),
            del_rot_error=float(values[4]),
            del_rot_lab_x=float(values[5]),
            del_rot_lab_y=float(values[6]),
            del_rot_lab_z=float(values[7]),
            abs_ori_cam_x=float(values[8]),
            abs_ori_cam_y=float(values[9]),
            abs_ori_cam_z=float(values[10]),
            abs_ori_lab_x=float(values[11]),
            abs_ori_lab_y=float(values[12]),
            abs_ori_lab_z=float(values[13]),
            posx=float(values[14]),
            posy=float(values[15]),
            heading=float(values[16]),
            direction=float(values[17]),
            speed=float(values[18]),
            intx=float(values[19]),
            inty=float(values[20]),
            timestamp=float(values[21]),
            seq_num=int(values[22]),
            delta_timestamp=float(values[23]),
            alt_timestamp=float(values[24]) if len(values) > 24 else 0.0,
        )


@dataclass(slots=True)
class FicTracFrame:
    wall_time: float
    frame_cnt: int
    del_rot_cam_x: float = 0.0
    del_rot_cam_y: float = 0.0
    del_rot_cam_z: float = 0.0
    del_rot_error: float = 0.0
    del_rot_lab_x: float = 0.0
    del_rot_lab_y: float = 0.0
    del_rot_lab_z: float = 0.0
    abs_ori_cam_x: float = 0.0
    abs_ori_cam_y: float = 0.0
    abs_ori_cam_z: float = 0.0
    abs_ori_lab_x: float = 0.0
    abs_ori_lab_y: float = 0.0
    abs_ori_lab_z: float = 0.0
    posx: float = 0.0
    posy: float = 0.0
    heading: float = 0.0
    speed: float = 0.0
    direction: float = 0.0
    intx: float = 0.0
    inty: float = 0.0
    timestamp: float = 0.0
    seq_num: int = 0
    delta_timestamp: float = 0.0
    alt_timestamp: float = 0.0

    @classmethod
    def from_state(cls, state: FicTracState, wall_time: float) -> "FicTracFrame":
        return cls(
            wall_time=wall_time,
            frame_cnt=state.frame_cnt,
            del_rot_cam_x=state.del_rot_cam_x,
            del_rot_cam_y=state.del_rot_cam_y,
            del_rot_cam_z=state.del_rot_cam_z,
            del_rot_error=state.del_rot_error,
            del_rot_lab_x=state.del_rot_lab_x,
            del_rot_lab_y=state.del_rot_lab_y,
            del_rot_lab_z=state.del_rot_lab_z,
            abs_ori_cam_x=state.abs_ori_cam_x,
            abs_ori_cam_y=state.abs_ori_cam_y,
            abs_ori_cam_z=state.abs_ori_cam_z,
            abs_ori_lab_x=state.abs_ori_lab_x,
            abs_ori_lab_y=state.abs_ori_lab_y,
            abs_ori_lab_z=state.abs_ori_lab_z,
            posx=state.posx,
            posy=state.posy,
            heading=state.heading,
            speed=state.speed,
            direction=state.direction,
            intx=state.intx,
            inty=state.inty,
            timestamp=state.timestamp,
            seq_num=state.seq_num,
            delta_timestamp=state.delta_timestamp,
            alt_timestamp=state.alt_timestamp,
        )


FICTRAC_FRAME_DTYPE = np.dtype(
    [
        ("wall_time", np.float64),
        ("frame_cnt", np.int64),
        ("del_rot_cam_x", np.float64),
        ("del_rot_cam_y", np.float64),
        ("del_rot_cam_z", np.float64),
        ("del_rot_error", np.float64),
        ("del_rot_lab_x", np.float64),
        ("del_rot_lab_y", np.float64),
        ("del_rot_lab_z", np.float64),
        ("abs_ori_cam_x", np.float64),
        ("abs_ori_cam_y", np.float64),
        ("abs_ori_cam_z", np.float64),
        ("abs_ori_lab_x", np.float64),
        ("abs_ori_lab_y", np.float64),
        ("abs_ori_lab_z", np.float64),
        ("posx", np.float64),
        ("posy", np.float64),
        ("heading", np.float64),
        ("speed", np.float64),
        ("direction", np.float64),
        ("intx", np.float64),
        ("inty", np.float64),
        ("timestamp", np.float64),
        ("seq_num", np.int64),
        ("delta_timestamp", np.float64),
        ("alt_timestamp", np.float64),
    ]
)


def frame_to_record(frame: FicTracFrame) -> tuple[object, ...]:
    return (
        frame.wall_time,
        frame.frame_cnt,
        frame.del_rot_cam_x,
        frame.del_rot_cam_y,
        frame.del_rot_cam_z,
        frame.del_rot_error,
        frame.del_rot_lab_x,
        frame.del_rot_lab_y,
        frame.del_rot_lab_z,
        frame.abs_ori_cam_x,
        frame.abs_ori_cam_y,
        frame.abs_ori_cam_z,
        frame.abs_ori_lab_x,
        frame.abs_ori_lab_y,
        frame.abs_ori_lab_z,
        frame.posx,
        frame.posy,
        frame.heading,
        frame.speed,
        frame.direction,
        frame.intx,
        frame.inty,
        frame.timestamp,
        frame.seq_num,
        frame.delta_timestamp,
        frame.alt_timestamp,
    )


def record_to_frame(record: np.void) -> FicTracFrame:
    return FicTracFrame(
        wall_time=float(record["wall_time"]),
        frame_cnt=int(record["frame_cnt"]),
        del_rot_cam_x=float(record["del_rot_cam_x"]),
        del_rot_cam_y=float(record["del_rot_cam_y"]),
        del_rot_cam_z=float(record["del_rot_cam_z"]),
        del_rot_error=float(record["del_rot_error"]),
        del_rot_lab_x=float(record["del_rot_lab_x"]),
        del_rot_lab_y=float(record["del_rot_lab_y"]),
        del_rot_lab_z=float(record["del_rot_lab_z"]),
        abs_ori_cam_x=float(record["abs_ori_cam_x"]),
        abs_ori_cam_y=float(record["abs_ori_cam_y"]),
        abs_ori_cam_z=float(record["abs_ori_cam_z"]),
        abs_ori_lab_x=float(record["abs_ori_lab_x"]),
        abs_ori_lab_y=float(record["abs_ori_lab_y"]),
        abs_ori_lab_z=float(record["abs_ori_lab_z"]),
        posx=float(record["posx"]),
        posy=float(record["posy"]),
        heading=float(record["heading"]),
        speed=float(record["speed"]),
        direction=float(record["direction"]),
        intx=float(record["intx"]),
        inty=float(record["inty"]),
        timestamp=float(record["timestamp"]),
        seq_num=int(record["seq_num"]),
        delta_timestamp=float(record["delta_timestamp"]),
        alt_timestamp=float(record["alt_timestamp"]),
    )


class FicTracCallback(Protocol):
    def setup_callback(self) -> None:
        ...

    def process_callback(self, track_state: FicTracState) -> bool:
        ...

    def shutdown_callback(self) -> None:
        ...


class BaseFicTracCallback:
    def setup_callback(self) -> None:
        pass

    def process_callback(self, track_state: FicTracState) -> bool:
        return True

    def shutdown_callback(self) -> None:
        pass


class FicTracFrameStore:
    """Efficient frame store for both logging and future closed-loop reads.

    Design goals:
    - cheap append from the receiver thread
    - O(1) latest-frame access
    - wait-for-next-frame without forcing backlog processing
    - compact numeric storage for long runs
    - short recent-history ring for closed-loop filters
    """

    def __init__(self, *, chunk_size: int = 8192, recent_capacity: int = 2048) -> None:
        self._chunk_size = max(1, int(chunk_size))
        self._recent_capacity = max(1, int(recent_capacity))
        self._chunks: list[np.ndarray] = []
        self._current = np.empty(self._chunk_size, dtype=FICTRAC_FRAME_DTYPE)
        self._current_size = 0
        self._count = 0

        self._recent = np.empty(self._recent_capacity, dtype=FICTRAC_FRAME_DTYPE)
        self._recent_count = 0
        self._recent_write_idx = 0

        self._latest_frame: Optional[FicTracFrame] = None
        self._latest_seq = -1
        self._cond = threading.Condition()

    @property
    def count(self) -> int:
        return self._count

    @property
    def latest(self) -> Optional[FicTracFrame]:
        with self._cond:
            return self._latest_frame

    @property
    def latest_seq(self) -> int:
        with self._cond:
            return self._latest_seq

    def append(self, frame: FicTracFrame) -> int:
        record = frame_to_record(frame)
        with self._cond:
            if self._current_size >= self._chunk_size:
                self._chunks.append(self._current)
                self._current = np.empty(self._chunk_size, dtype=FICTRAC_FRAME_DTYPE)
                self._current_size = 0

            self._current[self._current_size] = record
            self._current_size += 1
            self._count += 1

            self._recent[self._recent_write_idx] = record
            self._recent_write_idx = (self._recent_write_idx + 1) % self._recent_capacity
            self._recent_count = min(self._recent_count + 1, self._recent_capacity)

            self._latest_frame = frame
            self._latest_seq = self._count - 1
            self._cond.notify_all()
            return self._latest_seq

    def get_latest(self) -> tuple[int, Optional[FicTracFrame]]:
        with self._cond:
            return self._latest_seq, self._latest_frame

    def wait_for_next(self, after_seq: int = -1, timeout: float | None = None) -> tuple[int, Optional[FicTracFrame]]:
        with self._cond:
            if self._latest_seq > after_seq:
                return self._latest_seq, self._latest_frame

            if not self._cond.wait_for(lambda: self._latest_seq > after_seq, timeout=timeout):
                return self._latest_seq, self._latest_frame
            return self._latest_seq, self._latest_frame

    def recent_array(self, max_count: int | None = None) -> np.ndarray:
        with self._cond:
            n = self._recent_count if max_count is None else min(self._recent_count, max(0, int(max_count)))
            if n <= 0:
                return np.empty(0, dtype=FICTRAC_FRAME_DTYPE)

            start = (self._recent_write_idx - n) % self._recent_capacity
            if start + n <= self._recent_capacity:
                return self._recent[start:start + n].copy()

            first = self._recent[start:].copy()
            second = self._recent[: n - len(first)].copy()
            return np.concatenate([first, second])

    def frame_array(self) -> np.ndarray:
        with self._cond:
            arrays: list[np.ndarray] = []
            if self._chunks:
                arrays.extend(self._chunks)
            if self._current_size:
                arrays.append(self._current[:self._current_size].copy())

        if not arrays:
            return np.empty(0, dtype=FICTRAC_FRAME_DTYPE)
        if len(arrays) == 1:
            return arrays[0]
        return np.concatenate(arrays)

    def save_npz(self, path: str | Path) -> int:
        frames = self.frame_array()
        np.savez_compressed(path, frames=frames)
        return int(len(frames))


class FicTracDriver:
    """Minimal in-repo FicTrac launcher and UDP receiver.

    This replaces the small subset of pybmt that MultiBiOS currently uses.
    """

    def __init__(
        self,
        config_file: str | None = None,
        remote_endpoint_url: str | None = None,
        console_ouput_file: str = "output.txt",
        track_change_callback: Optional[FicTracCallback] = None,
        pgr_enable: bool = False,
        plot_on: bool = False,
        fic_trac_bin_path: str | None = None,
    ) -> None:
        self.track_change_callback = track_change_callback or BaseFicTracCallback()
        self.plot_on = plot_on
        self.average_fps_threshold = 0
        self.max_num_connect_retries = 60
        self.max_message_silence_s = 10.0

        self.console_output_file = console_ouput_file
        self.pgr_enable = pgr_enable
        self.config_file = config_file
        self.fictrac_process: Optional[subprocess.Popen[str]] = None
        self._console_handle = None
        self._fictrac_terminated_by_driver = False
        self.frame_cnt = 0
        self.skipped_frames = 0
        self._launch_wall_time: float | None = None
        self._first_packet_wall_time: float | None = None
        self._initial_wait_timeout_s: float | None = 60.0
        self._diagnostics_path: Optional[Path] = None
        self._diagnostics: dict[str, object] = {
            "config_file": config_file,
            "console_output_file": console_ouput_file,
            "remote_endpoint_url": remote_endpoint_url,
            "start_fictrac": remote_endpoint_url is None,
        }

        if remote_endpoint_url is not None:
            parts = str(remote_endpoint_url).split(":")
            self.udp_port = int(parts[-1])
            self.start_fictrac = False
            self.fictrac_bin_fullpath = ""
        else:
            self.start_fictrac = True
            self.udp_port = 5556
            self.fictrac_bin_fullpath = self._resolve_binary(fic_trac_bin_path)

        self._load_runtime_config_diagnostics()

    def _load_runtime_config_diagnostics(self) -> None:
        if not self.config_file:
            return

        config_path = Path(self.config_file).expanduser().resolve()
        self._diagnostics_path = config_path.with_name("fictrac_driver_diagnostics.json")
        config_values: dict[str, str] = {}
        try:
            with open(config_path, encoding="utf-8") as handle:
                for raw_line in handle:
                    line = raw_line.strip()
                    if not line or line.startswith("#") or ":" not in line:
                        continue
                    key, _, value = line.partition(":")
                    config_values[key.strip()] = value.strip()
        except OSError as exc:
            self._diagnostics["config_read_error"] = str(exc)
            self._write_diagnostics()
            return

        self._diagnostics["config_values"] = {
            key: config_values.get(key)
            for key in (
                "src_fn",
                "src_fps",
                "src_first_frame_timeout_ms",
                "sock_host",
                "sock_port",
                "save_raw",
                "vid_codec",
                "output_fn",
            )
            if key in config_values
        }

        timeout_ms_raw = config_values.get("src_first_frame_timeout_ms")
        if timeout_ms_raw is not None:
            try:
                timeout_ms = int(float(timeout_ms_raw))
            except ValueError:
                self._diagnostics["src_first_frame_timeout_ms_parse_error"] = timeout_ms_raw
            else:
                self._initial_wait_timeout_s = None if timeout_ms <= 0 else timeout_ms / 1000.0

        sock_port_raw = config_values.get("sock_port")
        if sock_port_raw is not None:
            try:
                self.udp_port = int(sock_port_raw)
            except ValueError:
                self._diagnostics["sock_port_parse_error"] = sock_port_raw

        self._diagnostics["initial_wait_timeout_s"] = self._initial_wait_timeout_s
        self._write_diagnostics()

    def _write_diagnostics(self) -> None:
        if self._diagnostics_path is None:
            return
        try:
            self._diagnostics_path.write_text(json.dumps(self._diagnostics, indent=2), encoding="utf-8")
        except OSError:
            pass

    def _resolve_binary(self, fic_trac_bin_path: str | None) -> str:
        if fic_trac_bin_path:
            return fic_trac_bin_path

        binary_name = "fictrac-pgr" if self.pgr_enable else "fictrac"
        if os.name == "nt":
            binary_name += ".exe"
        resolved = shutil.which(binary_name)
        if resolved is None:
            raise RuntimeError(f"Could not find {binary_name} on PATH")
        return resolved

    def request_stop(self) -> None:
        self._fictrac_terminated_by_driver = True
        if self.fictrac_process is not None and self.fictrac_process.poll() is None:
            self.fictrac_process.terminate()

    def run(self) -> None:
        self.track_change_callback.setup_callback()
        udp_socket = self._setup_udp_socket()
        self._diagnostics["udp_port"] = self.udp_port
        self._diagnostics["binary_path"] = self.fictrac_bin_fullpath
        self._diagnostics["process_cwd"] = os.path.dirname(self.fictrac_bin_fullpath) or None
        self._diagnostics["pid"] = None
        self._launch_wall_time = time.monotonic()
        self._diagnostics["launch_wall_time"] = self._launch_wall_time
        self._write_diagnostics()

        try:
            if self.start_fictrac:
                if not self.config_file:
                    raise RuntimeError("config_file is required when launching FicTrac locally")

                popen_kwargs: dict[str, object] = {
                    "cwd": os.path.dirname(self.fictrac_bin_fullpath) or None,
                    "env": build_fictrac_subprocess_env(
                        fictrac_bin_path=self.fictrac_bin_fullpath,
                    ),
                    "text": True,
                }

                if os.name == "nt":
                    # FicTrac's Windows camera path runs when attached to a real console,
                    # but exits immediately when stdout/stderr are redirected.
                    self._console_handle = None
                else:
                    console_path = Path(self.console_output_file).expanduser()
                    if not console_path.is_absolute():
                        console_path = Path.cwd() / console_path
                    console_path.parent.mkdir(parents=True, exist_ok=True)
                    self._console_handle = console_path.open("w", encoding="utf-8", newline="")
                    popen_kwargs["stdout"] = self._console_handle
                    popen_kwargs["stderr"] = subprocess.STDOUT

                self.fictrac_process = subprocess.Popen(
                    [self.fictrac_bin_fullpath, os.path.abspath(self.config_file)],
                    **popen_kwargs,
                )
                self._diagnostics["pid"] = self.fictrac_process.pid
                self._diagnostics["spawned"] = True
                self._write_diagnostics()

            self._process_messages(udp_socket)

            if (
                not self._fictrac_terminated_by_driver
                and self.fictrac_process is not None
                and self.fictrac_process.poll() is not None
                and self.fictrac_process.returncode not in (None, 0)
            ):
                raise RuntimeError(
                    "FicTrac failed because of an application error. "
                    f"Return code: {self.fictrac_process.returncode}. "
                    f"Consult the FicTrac console output file ({self.console_output_file})."
                )

            if self.frame_cnt == 0:
                process_state = "not started"
                if self.fictrac_process is not None:
                    returncode = self.fictrac_process.poll()
                    process_state = (
                        "still running" if returncode is None else f"exited with return code {returncode}"
                    )
                raise RuntimeError(
                    "Zero frames processed. FicTrac failed because of an application error. "
                    f"Process state: {process_state}. "
                    f"Consult the FicTrac console output file ({self.console_output_file})."
                )
        finally:
            if self.fictrac_process is not None:
                self._diagnostics["final_returncode"] = self.fictrac_process.poll()
            self._diagnostics["frame_cnt"] = self.frame_cnt
            self._diagnostics["skipped_frames"] = self.skipped_frames
            self._diagnostics["first_packet_wall_time"] = self._first_packet_wall_time
            self._diagnostics["terminated_by_driver"] = self._fictrac_terminated_by_driver
            self._write_diagnostics()
            if self.fictrac_process is not None and not self._fictrac_terminated_by_driver:
                self.request_stop()
            udp_socket.close()
            if self._console_handle is not None:
                self._console_handle.close()
                self._console_handle = None
            self.track_change_callback.shutdown_callback()

    def _setup_udp_socket(self) -> socket.socket:
        udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        udp_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        udp_socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4 * 1024 * 1024)
        udp_socket.bind(("", self.udp_port))
        udp_socket.settimeout(1.0)
        return udp_socket

    def _process_messages(self, udp_socket: socket.socket) -> None:
        time_history: list[float] = []
        last_packet_wall_time = time.monotonic()
        last_state: Optional[FicTracState] = None
        initial_wait_start = time.monotonic()

        while True:
            try:
                raw, _ = udp_socket.recvfrom(4096)
                payload = raw.decode("utf-8").strip()
                last_packet_wall_time = time.monotonic()
                if self._first_packet_wall_time is None:
                    self._first_packet_wall_time = last_packet_wall_time
                    self._diagnostics["first_packet_wall_time"] = self._first_packet_wall_time
                    self._write_diagnostics()
            except socket.timeout:
                if not self.start_fictrac:
                    raise RuntimeError("Socket timed out. Couldn't reach FicTrac.")

                if self.fictrac_process is not None and self.fictrac_process.poll() is not None:
                    self._diagnostics["initial_wait_process_exit_returncode"] = self.fictrac_process.returncode
                    self._write_diagnostics()
                    break

                if self.frame_cnt == 0:
                    if (
                        self._initial_wait_timeout_s is not None
                        and time.monotonic() - initial_wait_start >= self._initial_wait_timeout_s
                    ):
                        self._diagnostics["initial_wait_timeout_hit"] = self._initial_wait_timeout_s
                        self._write_diagnostics()
                        break
                    continue

                silent_for_s = time.monotonic() - last_packet_wall_time
                if silent_for_s >= self.max_message_silence_s:
                    raise RuntimeError(
                        f"FicTrac UDP stream stalled for {silent_for_s:.1f} s while the process was still running."
                    )
                continue

            if payload == "END":
                break

            t0 = time.perf_counter()
            state = FicTracState.from_udp_message(payload)
            if last_state is not None and state.frame_cnt - last_state.frame_cnt != 1:
                skipped = state.frame_cnt - last_state.frame_cnt - 1
                if skipped > 0:
                    self.skipped_frames += skipped
                    print(
                        "Warning: FicTrac skipped {} frame(s) (oldFrame={}, newFrame={}). Total skipped: {}.".format(
                            skipped,
                            last_state.frame_cnt,
                            state.frame_cnt,
                            self.skipped_frames,
                        )
                    )
            last_state = state

            should_continue = self.track_change_callback.process_callback(state)
            self.frame_cnt += 1

            dt = time.perf_counter() - t0
            time_history.append(dt)
            if len(time_history) > 10:
                time_history.pop(0)

            if self.average_fps_threshold and self.frame_cnt > 300:
                avg_fps = 1.0 / (sum(time_history) / len(time_history))
                if avg_fps < self.average_fps_threshold:
                    if self.fictrac_process is not None:
                        self.request_stop()
                    raise RuntimeError(
                        f"Average FPS fell below avg_fps_threshold ({self.average_fps_threshold})."
                    )

            if not should_continue:
                break

        if self.fictrac_process is not None and not self._fictrac_terminated_by_driver:
            self.request_stop()