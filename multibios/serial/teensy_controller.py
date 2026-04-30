#!/usr/bin/env python3
"""
teensy_controller.py — Thread-safe serial interface to the Teensy v2 firmware.

Wraps the ``v2_usb_serial_controller`` ASCII protocol with high-level Python
methods for odor valve control.  Designed for the computer-timebase integration
approach where odor state is decided by the host and the NIDAQ only latches
the shift-register outputs.

Key design points
-----------------
* **Thread-safe**: all serial I/O goes through a single ``threading.Lock``.
* **Command tagging**: every command is auto-tagged (``@N cmd``) so responses
  can be reliably matched even if heartbeat or watch output is interleaved.
* **Future closed-loop ready**: the ``send()`` method is fast enough (~1 ms
  round-trip on USB serial) for real-time callbacks.
"""
from __future__ import annotations

import queue
import re
import threading
import time
from typing import Optional

from multibios.serial_line_monitor import SerialLineMonitor


# ── State / command mappings ────────────────────────────────────────────────

# Protocol YAML states → Teensy v2 ASCII commands
# The target suffix (OLF1/OLF2/SV1/SV2) is appended by the high-level methods.
_BIG_STATE_CMD = {
    "OFF":   None,      # special: clear olfactometer register
    "AIR":   "CTRL",
    "ODOR1": "OD1",
    "ODOR2": "OD2",
    "ODOR3": "OD3",
    "ODOR4": "OD4",
    "ODOR5": "OD5",
    "FLUSH": None,      # special: set all bits
}

_SMALL_STATE_CMD = {
    "CLEAN": "CLN",
    "ODOR":  "ODR",
}

_SIDE_TO_OLF = {"left": "OLF1", "right": "OLF2"}
_SIDE_TO_SV  = {"left": "SV1",  "right": "SV2"}
_TAGGED_LINE_RE = re.compile(r"^@(?P<tag>\d+)\s+(?P<body>.*)$")


class TeensyController:
    """High-level serial interface to Teensy v2 USB serial controller.

    Parameters
    ----------
    port : str
        Serial port, e.g. ``"COM3"`` or ``"/dev/ttyACM0"``.
    baudrate : int
        Baud rate.  The Teensy 4.1 USB serial ignores this (always full-speed
        USB) but we set it for compatibility.
    timeout : float
        Read timeout in seconds for each response line.
    """

    def __init__(
        self,
        port: str,
        baudrate: int = 115_200,
        timeout: float = 1.0,
        *,
        serial_monitor: SerialLineMonitor | None = None,
    ) -> None:
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout

        self._lock = threading.Lock()
        self._tag = 0  # auto-incrementing command tag
        self._response_queues: dict[int, queue.Queue[str]] = {}
        self._serial_monitor = serial_monitor or SerialLineMonitor(
            port=port,
            baudrate=baudrate,
            timeout=timeout,
            boot_delay_s=0.5,
            reset_input_buffer_on_open=True,
        )

    # ── Connection lifecycle ────────────────────────────────────────────────

    def open(self) -> None:
        """Open the serial port."""
        self._serial_monitor.open()
        self._serial_monitor.add_listener(self._handle_line)

    def close(self) -> None:
        """Close the serial port."""
        self._serial_monitor.remove_listener(self._handle_line)
        with self._lock:
            self._response_queues.clear()
        self._serial_monitor.close()

    @property
    def is_open(self) -> bool:
        return self._serial_monitor.is_open

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *exc):
        self.close()

    # ── Low-level I/O ───────────────────────────────────────────────────────

    def _handle_line(self, line: str, wall_time: float) -> None:
        del wall_time
        match = _TAGGED_LINE_RE.match(line)
        if match is None:
            return
        tag = int(match.group("tag"))
        body = match.group("body")

        with self._lock:
            response_queue = self._response_queues.get(tag)
        if response_queue is not None:
            response_queue.put(body)

    def _write_tagged_command(
        self,
        command: str,
        *,
        expect_response: bool,
    ) -> tuple[int, queue.Queue[str] | None]:
        with self._lock:
            if not self._serial_monitor.is_open:
                raise RuntimeError("Serial port not open")

            self._tag += 1
            tag = self._tag
            tagged = f"@{tag} {command}"
            response_queue: queue.Queue[str] | None = None
            if expect_response:
                response_queue = queue.Queue()
                self._response_queues[tag] = response_queue
            self._serial_monitor.write_line(tagged)
            return tag, response_queue

    def _clear_response_queue(self, tag: int) -> None:
        with self._lock:
            self._response_queues.pop(tag, None)

    def send(self, command: str, *, timeout: float | None = None) -> str:
        """Send a tagged command and return the first tagged response line.

        Thread-safe.  Blocks until the response (or timeout).

        Returns
        -------
        response : str
            The response line with the tag prefix stripped.
        """
        tag, response_queue = self._write_tagged_command(command, expect_response=True)
        assert response_queue is not None
        try:
            return response_queue.get(timeout=timeout or self.timeout)
        except queue.Empty as exc:
            raise TimeoutError(f"No response for tag @{tag}: {command!r}") from exc
        finally:
            self._clear_response_queue(tag)

    def send_nowait(self, command: str) -> None:
        """Send a command without waiting for a response (fire-and-forget).

        Useful for time-critical paths where you don't need confirmation.
        """
        self._write_tagged_command(command, expect_response=False)

    # ── High-level odor commands ────────────────────────────────────────────

    def set_olfactometer(self, side: str, state: str, *, wait: bool = True) -> str | None:
        """Set an olfactometer to the given protocol state.

        Parameters
        ----------
        side : ``"left"`` or ``"right"``
        state : one of ``"OFF"``, ``"AIR"``, ``"ODOR1"``–``"ODOR5"``, ``"FLUSH"``
        wait : if True, block for acknowledgement

        Returns
        -------
        response : str or None
        """
        side = side.strip().lower()
        state = state.strip().upper()
        target = _SIDE_TO_OLF.get(side)
        if target is None:
            raise ValueError(f"Invalid side '{side}'; use 'left' or 'right'")

        if state == "OFF":
            # Clear the olfactometer by setting CTRL then turning off those bits
            # CTRL sets bits 0-1; we clear them to get a zero register
            cmd = f"CTRL {target}"
            if wait:
                self.send(cmd)
            else:
                self.send_nowait(cmd)
            # Now clear the two bits that CTRL just set
            if side == "left":
                bits = "OLF1_0 OLF1_1"
            else:
                bits = "OLF2_0 OLF2_1"
            off_cmd = f"OFF {bits}"
            if wait:
                resp = self.send(off_cmd)
                # Must SEND to push via SPI
                self.send("SEND")
                return resp
            else:
                self.send_nowait(off_cmd)
                self.send_nowait("SEND")
                return None

        elif state == "FLUSH":
            # All 12 bits on.  Easiest: use ON for all bits of this olfactometer
            prefix = "OLF1" if side == "left" else "OLF2"
            bits = " ".join(f"{prefix}_{i}" for i in range(12))
            # First clear by running CTRL (which clears then sets 0,1)
            self.send_nowait(f"CTRL {target}")
            cmd = f"ON {bits}"
            if wait:
                resp = self.send(cmd)
                self.send("SEND")
                return resp
            else:
                self.send_nowait(cmd)
                self.send_nowait("SEND")
                return None

        else:
            teensy_cmd = _BIG_STATE_CMD.get(state)
            if teensy_cmd is None:
                raise ValueError(f"Unknown olfactometer state '{state}'")
            cmd = f"{teensy_cmd} {target}"
            if wait:
                return self.send(cmd)
            else:
                self.send_nowait(cmd)
                return None

    def set_switch_valve(self, side: str, state: str, *, wait: bool = True) -> str | None:
        """Set a switch valve to CLEAN or ODOR.

        Parameters
        ----------
        side : ``"left"`` or ``"right"``
        state : ``"CLEAN"`` or ``"ODOR"``
        """
        side = side.strip().lower()
        state = state.strip().upper()
        target = _SIDE_TO_SV.get(side)
        if target is None:
            raise ValueError(f"Invalid side '{side}'")
        teensy_cmd = _SMALL_STATE_CMD.get(state)
        if teensy_cmd is None:
            raise ValueError(f"Unknown switch-valve state '{state}'")
        cmd = f"{teensy_cmd} {target}"
        if wait:
            return self.send(cmd)
        else:
            self.send_nowait(cmd)
            return None

    def reset(self) -> str:
        """RESET — clears all staged bits and sends (all valves off)."""
        return self.send("RESET")

    def status(self) -> str:
        """Return the full STATUS dump from the Teensy."""
        tag, response_queue = self._write_tagged_command("STATUS", expect_response=True)
        assert response_queue is not None
        lines: list[str] = []
        deadline = time.monotonic() + 2.0
        try:
            while time.monotonic() < deadline:
                remaining = deadline - time.monotonic()
                wait_timeout = remaining if not lines else min(0.1, remaining)
                try:
                    lines.append(response_queue.get(timeout=wait_timeout))
                except queue.Empty:
                    if lines:
                        break
            return "\n".join(lines)
        finally:
            self._clear_response_queue(tag)

    def print_state(self) -> str:
        """PRINT — return the current staged bitstring."""
        return self.send("PRINT")
