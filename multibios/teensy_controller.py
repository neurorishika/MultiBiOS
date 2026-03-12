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

import re
import threading
import time
from typing import Optional

import serial


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
    ) -> None:
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout

        self._ser: Optional[serial.Serial] = None
        self._lock = threading.Lock()
        self._tag = 0  # auto-incrementing command tag

    # ── Connection lifecycle ────────────────────────────────────────────────

    def open(self) -> None:
        """Open the serial port."""
        with self._lock:
            if self._ser is not None and self._ser.is_open:
                return
            self._ser = serial.Serial(
                self.port,
                self.baudrate,
                timeout=self.timeout,
                write_timeout=self.timeout,
            )
            # Teensy resets on serial open — give it time to boot
            time.sleep(0.5)
            self._ser.reset_input_buffer()

    def close(self) -> None:
        """Close the serial port."""
        with self._lock:
            if self._ser is not None:
                try:
                    self._ser.close()
                except Exception:
                    pass
                self._ser = None

    @property
    def is_open(self) -> bool:
        return self._ser is not None and self._ser.is_open

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *exc):
        self.close()

    # ── Low-level I/O ───────────────────────────────────────────────────────

    def send(self, command: str, *, timeout: float | None = None) -> str:
        """Send a tagged command and return the first tagged response line.

        Thread-safe.  Blocks until the response (or timeout).

        Returns
        -------
        response : str
            The response line with the tag prefix stripped.
        """
        with self._lock:
            if self._ser is None or not self._ser.is_open:
                raise RuntimeError("Serial port not open")

            self._tag += 1
            tag = self._tag
            tagged = f"@{tag} {command}\n"

            self._ser.reset_input_buffer()
            self._ser.write(tagged.encode("ascii"))
            self._ser.flush()

            # Read lines until we find our tagged response
            deadline = time.monotonic() + (timeout or self.timeout)
            prefix = f"@{tag} "
            while time.monotonic() < deadline:
                raw = self._ser.readline()
                if not raw:
                    continue
                line = raw.decode("ascii", errors="replace").strip()
                if line.startswith(prefix):
                    return line[len(prefix):]
            raise TimeoutError(f"No response for tag @{tag}: {command!r}")

    def send_nowait(self, command: str) -> None:
        """Send a command without waiting for a response (fire-and-forget).

        Useful for time-critical paths where you don't need confirmation.
        """
        with self._lock:
            if self._ser is None or not self._ser.is_open:
                raise RuntimeError("Serial port not open")
            self._tag += 1
            tagged = f"@{self._tag} {command}\n"
            self._ser.write(tagged.encode("ascii"))
            self._ser.flush()

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
        # STATUS produces multiple lines; collect them all
        with self._lock:
            if self._ser is None or not self._ser.is_open:
                raise RuntimeError("Serial port not open")
            self._tag += 1
            tag = self._tag
            tagged = f"@{tag} STATUS\n"
            self._ser.reset_input_buffer()
            self._ser.write(tagged.encode("ascii"))
            self._ser.flush()

            lines = []
            deadline = time.monotonic() + 2.0
            prefix = f"@{tag} "
            while time.monotonic() < deadline:
                raw = self._ser.readline()
                if not raw:
                    if lines:
                        break
                    continue
                line = raw.decode("ascii", errors="replace").strip()
                if line.startswith(prefix):
                    lines.append(line[len(prefix):])
            return "\n".join(lines)

    def print_state(self) -> str:
        """PRINT — return the current staged bitstring."""
        return self.send("PRINT")
