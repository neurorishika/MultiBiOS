from __future__ import annotations

import queue
import threading
import time

import serial

from multibios.serial_line_monitor import SerialLineMonitor
from MultiBiOS.legacy.multibios.serial.teensy_controller import TeensyController


class _FakeSerial:
    def __init__(self, *args, **kwargs) -> None:
        self.timeout = kwargs.get("timeout", 0.1)
        self.write_timeout = kwargs.get("write_timeout", 0.1)
        self.is_open = True
        self._rx_queue: queue.Queue[bytes] = queue.Queue()
        self._writes: list[str] = []
        self._lock = threading.Lock()

    def reset_input_buffer(self) -> None:
        return None

    def write(self, payload: bytes) -> int:
        text = payload.decode("ascii")
        with self._lock:
            self._writes.append(text)

        command = text.strip()
        if command == "@1 RESET":
            self._rx_queue.put(b"BOOT controller=ready\n")
            self._rx_queue.put(b"@1 OK RESET\n")
        elif command == "@2 STATUS":
            self._rx_queue.put(b"@2 MODE OPEN_LOOP\n")
            self._rx_queue.put(b"@2 READY LEFT=1 RIGHT=1\n")
        elif command == "@3 ODR SV1":
            self._rx_queue.put(b"@3 OK ODR SV1\n")
        return len(payload)

    def flush(self) -> None:
        return None

    def readline(self) -> bytes:
        try:
            return self._rx_queue.get(timeout=self.timeout)
        except queue.Empty:
            return b""

    def close(self) -> None:
        self.is_open = False


def test_serial_line_monitor_captures_unsolicited_and_tagged_lines(monkeypatch) -> None:
    monkeypatch.setattr(serial, "Serial", _FakeSerial)

    monitor = SerialLineMonitor(
        "COM_TEST",
        timeout=0.05,
        boot_delay_s=0.0,
        reset_input_buffer_on_open=True,
    )
    monitor.open()
    try:
        monitor.write_line("@1 RESET")
        monitor.write_line("@2 STATUS")
        monitor.write_line("@3 ODR SV1")
        time.sleep(0.05)
    finally:
        monitor.close()

    transcript = monitor.get_transcript()
    lines = [entry["line"] for entry in transcript]

    assert "@1 RESET" in lines
    assert "BOOT controller=ready" in lines
    assert "@1 OK RESET" in lines
    assert "@2 STATUS" in lines
    assert "@2 MODE OPEN_LOOP" in lines
    assert "@2 READY LEFT=1 RIGHT=1" in lines
    assert "@3 ODR SV1" in lines
    assert "@3 OK ODR SV1" in lines


def test_teensy_controller_send_timeout_cleans_pending_queue(monkeypatch) -> None:
    class _NoResponseSerial(_FakeSerial):
        def write(self, payload: bytes) -> int:
            text = payload.decode("ascii")
            with self._lock:
                self._writes.append(text)
            return len(payload)

    monkeypatch.setattr(serial, "Serial", _NoResponseSerial)

    monitor = SerialLineMonitor("COM_TEST", timeout=0.02, boot_delay_s=0.0)
    controller = TeensyController("COM_TEST", timeout=0.02, serial_monitor=monitor)
    controller.open()
    try:
        try:
            controller.send("PING")
        except TimeoutError:
            pass
        else:
            raise AssertionError("Expected TimeoutError")
        assert controller._response_queues == {}
    finally:
        controller.close()


def test_teensy_controller_routes_tagged_status_responses(monkeypatch) -> None:
    monkeypatch.setattr(serial, "Serial", _FakeSerial)

    monitor = SerialLineMonitor("COM_TEST", timeout=0.05, boot_delay_s=0.0)
    controller = TeensyController("COM_TEST", timeout=0.05, serial_monitor=monitor)
    controller.open()
    try:
        assert controller.reset() == "OK RESET"
        assert controller.status() == "MODE OPEN_LOOP\nREADY LEFT=1 RIGHT=1"
        assert controller.set_switch_valve("left", "ODOR") == "OK ODR SV1"
    finally:
        controller.close()