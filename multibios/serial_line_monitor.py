from __future__ import annotations

import threading
import time
from collections.abc import Callable
from typing import Any, Optional

import serial


LineListener = Callable[[str, float], None]


class SerialLineMonitor:
    """Generic line-oriented serial monitor with optional transcript capture.

    This abstraction is intentionally transport-focused: it owns serial open/close,
    background line reading, listener fan-out, and transcript retention. Protocol-
    specific command semantics belong in higher-level controllers.
    """

    def __init__(
        self,
        port: str,
        baudrate: int = 115_200,
        timeout: float = 1.0,
        *,
        boot_delay_s: float = 0.0,
        reset_input_buffer_on_open: bool = False,
    ) -> None:
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.boot_delay_s = boot_delay_s
        self.reset_input_buffer_on_open = reset_input_buffer_on_open

        self._ser: Optional[serial.Serial] = None
        self._lock = threading.Lock()
        self._reader_stop = threading.Event()
        self._reader_thread: Optional[threading.Thread] = None
        self._listeners: set[LineListener] = set()
        self._transcript_lock = threading.Lock()
        self._transcript: list[dict[str, Any]] = []

    @property
    def is_open(self) -> bool:
        return self._ser is not None and self._ser.is_open

    def open(self) -> None:
        with self._lock:
            if self._ser is not None and self._ser.is_open:
                return
            self._ser = serial.Serial(
                self.port,
                self.baudrate,
                timeout=self.timeout,
                write_timeout=self.timeout,
            )
        if self.boot_delay_s > 0:
            time.sleep(self.boot_delay_s)
        if self.reset_input_buffer_on_open and self._ser is not None:
            self._ser.reset_input_buffer()
        self._reader_stop.clear()
        self._reader_thread = threading.Thread(
            target=self._reader_loop,
            daemon=True,
            name=f"SerialReader[{self.port}]",
        )
        self._reader_thread.start()

    def close(self) -> None:
        reader_thread: Optional[threading.Thread] = None
        serial_port: Optional[serial.Serial] = None
        with self._lock:
            self._reader_stop.set()
            reader_thread = self._reader_thread
            self._reader_thread = None
            serial_port = self._ser
            self._ser = None
        if serial_port is not None:
            try:
                serial_port.close()
            except Exception:
                pass
        if reader_thread is not None:
            reader_thread.join(timeout=self.timeout + 0.5)

    def add_listener(self, listener: LineListener) -> None:
        with self._lock:
            self._listeners.add(listener)

    def remove_listener(self, listener: LineListener) -> None:
        with self._lock:
            self._listeners.discard(listener)

    def write_line(self, line: str) -> None:
        payload = line if line.endswith("\n") else f"{line}\n"
        with self._lock:
            if self._ser is None or not self._ser.is_open:
                raise RuntimeError("Serial port not open")
            self._record_transcript(direction="tx", line=payload.strip())
            self._ser.write(payload.encode("ascii"))
            self._ser.flush()

    def get_transcript(self) -> list[dict[str, Any]]:
        with self._transcript_lock:
            return [dict(entry) for entry in self._transcript]

    def _record_transcript(self, *, direction: str, line: str) -> None:
        with self._transcript_lock:
            self._transcript.append(
                {
                    "wall_time": time.perf_counter(),
                    "direction": direction,
                    "line": line,
                }
            )

    def _reader_loop(self) -> None:
        while not self._reader_stop.is_set():
            serial_port = self._ser
            if serial_port is None or not serial_port.is_open:
                return
            try:
                raw = serial_port.readline()
            except Exception:
                if self._reader_stop.is_set():
                    return
                continue
            if not raw:
                continue
            line = raw.decode("ascii", errors="replace").strip()
            if not line:
                continue

            self._record_transcript(direction="rx", line=line)
            with self._lock:
                listeners = tuple(self._listeners)
            wall_time = time.perf_counter()
            for listener in listeners:
                listener(line, wall_time)