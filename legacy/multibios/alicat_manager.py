"""alicat_manager.py — Auto-discovery and management wrapper for Alicat flow devices.

Usage (interactive / script):
    python alicat_manager.py              # scan all COM ports and print map
    python alicat_manager.py --ports COM7 COM8 --baud 115200

Programmatic:
    from alicat_manager import AlicatManager
    import asyncio

    mgr = AlicatManager()                 # loads cache from disk automatically

    # First time (or to refresh):
    asyncio.run(mgr.scan())

    # Show what was found:
    mgr.show_map()
    print(mgr.names())

    # Read all devices:
    states = asyncio.run(mgr.get_all())
    mgr.print_states(states)

    # Read a subset:
    states = asyncio.run(mgr.get_all(["C@COM7", "A@COM7"]))
"""
from __future__ import annotations

import asyncio
import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import serial.tools.list_ports
from alicat import FlowController, FlowMeter
from alicat.driver import MaxRampTimeUnit

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

STANDARD_BAUDRATES: list[int] = [19200, 115200, 57600, 38400, 9600]
UNIT_IDS: list[str] = [chr(i) for i in range(ord('A'), ord('Z') + 1)]
DEFAULT_CACHE: Path = Path(__file__).parent.parent / ".alicat_device_cache.json"

# ──────────────────────────────────────────────────────────────────────────────
# Low-level probe
# ──────────────────────────────────────────────────────────────────────────────

async def _probe_unit(
    port: str,
    baudrate: int,
    unit: str,
    timeout: float = 0.75,
) -> dict | None:
    """Attempt to communicate with a single unit on a port/baud combination.

    Tries FlowController first (superset of FlowMeter), then falls back to
    FlowMeter.  Returns a device-info dict on success, or None on failure.
    """
    for cls, dtype in [(FlowController, "controller"), (FlowMeter, "meter")]:
        dev = cls(address=port, unit=unit, baudrate=baudrate)
        try:
            data = await asyncio.wait_for(dev.get(), timeout=timeout)
            if data:
                return {
                    "port": port,
                    "baudrate": baudrate,
                    "unit": unit,
                    "type": dtype,
                    "last_state": data,
                }
        except BaseException:
            pass
        finally:
            try:
                await dev.close()
            except BaseException:
                pass
    return None


# ──────────────────────────────────────────────────────────────────────────────
# Cache helpers
# ──────────────────────────────────────────────────────────────────────────────

def _serializable(device_map: dict) -> dict:
    """Return a JSON-serializable copy of device_map."""
    out = {}
    for name, info in device_map.items():
        entry = dict(info)
        if entry.get("last_state"):
            entry["last_state"] = {
                k: v if isinstance(v, (int, float, str, bool, type(None))) else str(v)
                for k, v in entry["last_state"].items()
            }
        out[name] = entry
    return out


def save_cache(device_map: dict, cache_file: Path | str = DEFAULT_CACHE) -> None:
    """Write device map to disk."""
    with open(cache_file, "w") as f:
        json.dump(_serializable(device_map), f, indent=2)


def load_cache(cache_file: Path | str = DEFAULT_CACHE) -> dict[str, dict]:
    """Load device map from disk. Returns empty dict if no cache exists."""
    cache_file = Path(cache_file)
    if not cache_file.exists():
        return {}
    with open(cache_file) as f:
        return json.load(f)


# ──────────────────────────────────────────────────────────────────────────────
# Top-level scan
# ──────────────────────────────────────────────────────────────────────────────

async def _probe_all_units(
    port: str,
    baudrate: int,
    units: list[str],
    timeout: float,
) -> dict[str, dict]:
    """Probe all unit IDs sequentially on a single port/baud.

    Serial ports share one underlying connection per address (alicat driver
    caches by port name), so concurrent access causes read collisions.
    Units on the *same* port must be probed one-at-a-time.
    """
    found: dict[str, dict] = {}
    for u in units:
        info = await _probe_unit(port, baudrate, u, timeout=timeout)
        if info is not None:
            found[f"{u}@{port}"] = info
    return found


async def _find_baud_for_port(
    port: str,
    baudrates: list[int],
    units: list[str],
    timeout: float,
    verbose: bool,
) -> tuple[int | None, dict[str, dict]]:
    """Try baud rates in order; lock to the first one that yields any device.

    Returns (locked_baud_or_None, {name: info}).
    All units are probed concurrently within each baud attempt, so we exit as
    soon as any rate works without trying the rest.
    """
    for baud in baudrates:
        if verbose:
            print(f"  {port} @ {baud:>6} baud … ", end="", flush=True)
        hits = await _probe_all_units(port, baud, units, timeout)
        if hits:
            labels = [f"{info['unit']}({info['type'][0]})" for info in hits.values()]
            skipped = baudrates[baudrates.index(baud) + 1:]
            if verbose:
                print(f"found {', '.join(labels)}")
                if skipped:
                    print(f"    ↳ locked @ {baud} baud — skipping {skipped}")
            return baud, hits
        if verbose:
            print("nothing")
    return None, {}


async def scan(
    ports: list[str] | None = None,
    baudrates: list[int] | None = None,
    units: list[str] | None = None,
    timeout: float = 0.75,
    verbose: bool = True,
    cache_file: Path | str = DEFAULT_CACHE,
    expected_ids: list[str] | None = None,
) -> dict[str, dict]:
    """Scan COM ports for Alicat flow devices and cache results.

    Strategy:
    - Multiple ports are scanned concurrently (each port is an independent bus).
    - Unit IDs are probed sequentially within a port (shared serial connection).
    - For each port, baud rates are tried in order and locked on first hit.
    - Remaining baud rates are skipped once any device responds.
    - A warning is printed if devices on different ports use different baud rates.

    Args:
        ports:        COM ports to scan. Defaults to all currently active ports.
        baudrates:    Baud rates to try. Defaults to STANDARD_BAUDRATES.
        units:        Unit IDs to probe (A-Z subset). Defaults to all 26.
        timeout:      Per-unit read timeout in seconds.
        verbose:      Print progress to stdout.
        cache_file:   Where to save the resulting device map.
        expected_ids: Optional list of unit IDs (e.g. ['A','B','C','D']).
                      When provided, scanning stops as soon as every listed ID
                      has been found on at least one port — the remaining port
                      tasks are cancelled.  A warning is also printed if the
                      same unit ID appears on more than one port.

    Returns:
        device_map: {"<unit>@<port>": {port, baudrate, unit, type, last_state}}
    """
    ports     = ports     or [p.device for p in serial.tools.list_ports.comports()]
    baudrates = baudrates or STANDARD_BAUDRATES
    units     = units     or UNIT_IDS

    # Normalise expected_ids to uppercase set, deduplicated
    want: set[str] | None = None
    if expected_ids is not None:
        want = {uid.upper() for uid in expected_ids}
        if verbose:
            print(f"Expected IDs : {sorted(want)}  (scan will stop early once all found)")

    if verbose:
        print(f"Active ports : {ports}")
        print(f"Baud rates   : {baudrates}")
        print(f"Unit IDs     : A-Z ({len(units)} slots)\n")

    device_map: dict[str, dict] = {}
    port_bauds: dict[str, int] = {}   # port → locked baud, for mismatch check

    # Suppress alicat logger noise (timeouts/disconnects are expected during scan)
    alicat_logger = logging.getLogger("alicat")
    _saved_level = alicat_logger.level
    alicat_logger.setLevel(logging.CRITICAL)

    async def _port_job(port: str) -> tuple[str, int | None, dict]:
        """Probe one port and return (port, locked_baud, hits)."""
        locked_baud, hits = await _find_baud_for_port(port, baudrates, units, timeout, verbose)
        return port, locked_baud, hits

    completed_results: list[tuple[str, int | None, dict]] = []

    try:
        # Probe ports concurrently — each port is an independent serial bus.
        # When expected_ids is set we use as_completed so we can cancel the
        # remaining tasks the moment every expected unit has been found.
        port_tasks = [asyncio.ensure_future(_port_job(p)) for p in ports]

        try:
            for coro in asyncio.as_completed(port_tasks):
                port, locked_baud, hits = await coro
                completed_results.append((port, locked_baud, hits))

                if want is not None:
                    # Collect all unit letters found so far
                    found_ids: set[str] = set()
                    for _p, _baud, _hits in completed_results:
                        for info in _hits.values():
                            found_ids.add(info["unit"].upper())
                    if want.issubset(found_ids):
                        if verbose:
                            remaining = sum(1 for t in port_tasks if not t.done())
                            if remaining > 0:
                                print(
                                    f"\n  All expected IDs {sorted(want)} found — "
                                    f"cancelling {remaining} remaining port task(s)."
                                )
                        for t in port_tasks:
                            if not t.done():
                                t.cancel()
                        break
        except asyncio.CancelledError:
            pass

        # Drain cancelled tasks (suppress CancelledError)
        for t in port_tasks:
            if not t.done():
                try:
                    await t
                except (asyncio.CancelledError, Exception):
                    pass
            elif not t.cancelled():
                exc = t.exception()
                if exc is not None:
                    pass  # swallow background probe errors

    finally:
        alicat_logger.setLevel(_saved_level)

    for port, locked_baud, hits in completed_results:
        if locked_baud is not None:
            port_bauds[port] = locked_baud
            device_map.update(hits)

    # ── Duplicate unit-ID check ──────────────────────────────────────────────
    # Build a map: unit_letter → list of ports it was found on
    uid_to_ports: dict[str, list[str]] = {}
    for name, info in device_map.items():
        uid = info["unit"].upper()
        uid_to_ports.setdefault(uid, []).append(info["port"])

    for uid, found_on in uid_to_ports.items():
        if len(found_on) > 1:
            print(
                f"\n  !! WARNING: Unit ID '{uid}' found on multiple ports: "
                f"{', '.join(found_on)}\n"
                f"     Each Alicat unit ID must be unique across your setup.\n"
                f"     Check device address settings or cabling before proceeding.\n"
            )

    # Warn if different ports have different locked baud rates
    unique_bauds = set(port_bauds.values())
    if len(unique_bauds) > 1:
        print("\n  ⚠  WARNING: devices found at mixed baud rates across ports:")
        for p, b in port_bauds.items():
            print(f"       {p} -> {b} baud")
        print("     Ensure each port's devices are all configured at the same rate.\n")

    if want is not None:
        missing = want - set(uid_to_ports.keys())
        if missing:
            print(
                f"\n  ⚠  WARNING: Expected unit ID(s) {sorted(missing)} were NOT found "
                f"during scan. Check connections and power.\n"
            )

    save_cache(device_map, cache_file)

    if verbose:
        print(f"\n{len(device_map)} device(s) found. Cache saved -> {cache_file}")

    return device_map


# ──────────────────────────────────────────────────────────────────────────────
# Manager class
# ──────────────────────────────────────────────────────────────────────────────

class AlicatManager:
    """High-level manager for one or more Alicat flow devices.

    Loads a cached device map automatically on construction.  Call scan() to
    discover devices if the cache is empty or stale.

    Device names are "<unit>@<port>", e.g. "C@COM7".
    """

    def __init__(
        self,
        device_map: dict[str, dict] | None = None,
        cache_file: Path | str = DEFAULT_CACHE,
    ) -> None:
        self.cache_file = Path(cache_file)
        self.device_map: dict[str, dict] = device_map or load_cache(cache_file)

    # ── Discovery ─────────────────────────────────────────────────────────────

    async def scan(self, **kwargs: Any) -> dict[str, dict]:
        """(Re-)scan ports and refresh the device map.

        Pass ``expected_ids=['A','B','C','D']`` to enable early-exit once all
        listed units are found, and to get a warning if any are missing.
        Keyword args forwarded to the module-level scan().
        """
        self.device_map = await scan(cache_file=self.cache_file, **kwargs)
        return self.device_map

    # ── Inspection ────────────────────────────────────────────────────────────

    def names(self) -> list[str]:
        """Return all known device names."""
        return list(self.device_map.keys())

    def info(self, name: str) -> dict:
        """Return cached metadata for a device."""
        if name not in self.device_map:
            raise KeyError(f"Unknown device '{name}'. Known: {self.names()}")
        return self.device_map[name]

    def show_map(self) -> None:
        """Print a formatted summary of all cached devices."""
        if not self.device_map:
            print("No devices cached. Run scan() first.")
            return
        header = f"{'Name':<12} {'Type':<12} {'Port':<10} {'Baud':>8}  Unit"
        print(header)
        print("─" * len(header))
        for name, d in self.device_map.items():
            print(f"{name:<12} {d['type']:<12} {d['port']:<10} {d['baudrate']:>8}  {d['unit']}")

    # ── Internal helpers ──────────────────────────────────────────────────────


    @asynccontextmanager
    async def _open(self, name: str, require_controller: bool = False):
        """Async context manager that yields an open Alicat driver instance.

        Args:
            name: Device name, e.g. "C@COM7".
            require_controller: If True, raises TypeError for FlowMeter-only devices.
        """
        d = self.info(name)
        if require_controller and d["type"] != "controller":
            raise TypeError(f"'{name}' is a FlowMeter — this operation requires a FlowController.")
        cls = FlowController if d["type"] == "controller" else FlowMeter
        async with cls(address=d["port"], unit=d["unit"], baudrate=d["baudrate"]) as dev:
            yield dev

    # ── State queries ─────────────────────────────────────────────────────────

    async def get(self, name: str) -> dict[str, Any]:
        """Read current state of a single device.

        Args:
            name: Device name, e.g. "C@COM7".

        Returns:
            State dict from the Alicat driver (pressure, temperature, flow, …).
        """
        async with self._open(name) as dev:
            state = await dev.get()
        self.device_map[name]["last_state"] = state
        return state

    async def get_all(self, names: list[str] | None = None) -> dict[str, Any]:
        """Read current state of all (or a subset of) devices concurrently.

        Args:
            names: Device names to query. Defaults to all cached devices.

        Returns:
            {device_name: state_dict}
            On per-device errors, returns {device_name: {"error": "<message>"}}.
        """
        names = names or self.names()
        if not names:
            raise RuntimeError("No devices available. Run scan() first.")

        async def _safe(name: str) -> tuple[str, Any]:
            try:
                return name, await self.get(name)
            except Exception as exc:
                return name, {"error": str(exc)}

        results = await asyncio.gather(*[_safe(n) for n in names])
        return dict(results)

    # ── Setpoint control (FlowController only) ──────────────────────────────

    async def set_flow_rate(self, name: str, flowrate: float) -> None:
        """Set the mass-flow setpoint.

        Args:
            name:     Device name, e.g. "C@COM7".
            flowrate: Target flow rate in device purchase units.

        Note: Always opens as FlowController regardless of cached type, since
        hardware controllers are sometimes cached as meters if the initial
        control-point query timed out during scan.
        """
        d = self.info(name)
        async with FlowController(address=d["port"], unit=d["unit"], baudrate=d["baudrate"]) as dev:
            await dev.set_flow_rate(flowrate)

    async def set_pressure(self, name: str, pressure: float) -> None:
        """Set the pressure setpoint.

        Args:
            name:     Device name, e.g. "C@COM7".
            pressure: Target pressure in device purchase units (typically psia).
        """
        async with self._open(name, require_controller=True) as dev:
            await dev.set_pressure(pressure)

    async def set_all(self, setpoints: dict[str, float]) -> dict[str, Any]:
        """Set flow-rate setpoints, running ports concurrently but each port sequentially.

        Devices on the same COM port share one serial connection, so their
        commands are serialised to avoid read collisions.  Different ports
        are handled concurrently.

        Args:
            setpoints: {device_name: flowrate}, e.g. {"C@COM7": 10.0, "A@COM7": 5.5}

        Returns:
            {device_name: None} on success, {device_name: {"error": msg}} on failure.
        """
        # Group by port to ensure sequential access per bus
        from collections import defaultdict
        by_port: dict[str, list[tuple[str, float]]] = defaultdict(list)
        for name, val in setpoints.items():
            port = self.info(name)["port"]
            by_port[port].append((name, val))

        async def _set_port(items: list[tuple[str, float]]) -> list[tuple[str, Any]]:
            out = []
            for name, val in items:
                try:
                    await self.set_flow_rate(name, val)
                    out.append((name, None))
                except Exception as exc:
                    out.append((name, {"error": str(exc)}))
            return out

        port_results = await asyncio.gather(*[_set_port(items) for items in by_port.values()])
        return {name: err for pairs in port_results for name, err in pairs}

    async def hold(self, name: str) -> None:
        """Hold the valve at its current position (firmware 5v07)."""
        async with self._open(name, require_controller=True) as dev:
            await dev.hold()

    async def cancel_hold(self, name: str) -> None:
        """Cancel a previously set valve hold."""
        async with self._open(name, require_controller=True) as dev:
            await dev.cancel_hold()

    async def get_pid(self, name: str) -> dict[str, Any]:
        """Read PID parameters (loop_type, P, D, I)."""
        async with self._open(name, require_controller=True) as dev:
            return await dev.get_pid()

    async def set_pid(
        self,
        name: str,
        p: int | None = None,
        i: int | None = None,
        d: int | None = None,
        loop_type: str | None = None,
    ) -> None:
        """Set PID parameters.  Pass only the values you want to change.

        Args:
            name:      Device name.
            p:         Proportional gain.
            i:         Integral gain (PD2I loop only).
            d:         Derivative gain.
            loop_type: 'PD/PDF' or 'PD2I'.
        """
        async with self._open(name, require_controller=True) as dev:
            await dev.set_pid(p=p, i=i, d=d, loop_type=loop_type)

    async def get_ramp_config(self, name: str) -> dict[str, bool]:
        """Get setpoint ramp enable flags: up, down, zero, power (firmware 10v05)."""
        async with self._open(name, require_controller=True) as dev:
            return await dev.get_ramp_config()

    async def set_ramp_config(self, name: str, config: dict[str, bool]) -> None:
        """Configure setpoint ramp behaviour (firmware 10v05).

        Args:
            name:   Device name.
            config: Dict with any of: 'up', 'down', 'zero', 'power' → bool.
        """
        async with self._open(name, require_controller=True) as dev:
            await dev.set_ramp_config(config)

    async def get_maxramp(self, name: str) -> dict[str, float | str]:
        """Get maximum ramp rate (firmware 7v11)."""
        async with self._open(name, require_controller=True) as dev:
            return await dev.get_maxramp()

    async def set_maxramp(self, name: str, max_ramp: float, unit_time: MaxRampTimeUnit) -> None:
        """Set maximum ramp rate (firmware 7v11).

        Args:
            name:      Device name.
            max_ramp:  Maximum ramp rate value.
            unit_time: Time unit — one of 'ms', 's', 'm', 'h', 'd'.
        """
        async with self._open(name, require_controller=True) as dev:
            await dev.set_maxramp(max_ramp, unit_time)

    async def get_totalizer_batch(self, name: str, batch: int = 1) -> str:
        """Get totalizer batch volume (firmware 10v00)."""
        async with self._open(name, require_controller=True) as dev:
            return await dev.get_totalizer_batch(batch)

    async def set_totalizer_batch(
        self,
        name: str,
        batch_volume: float,
        batch: int = 1,
        units: str = "default",
    ) -> None:
        """Set totalizer batch volume (firmware 10v00)."""
        async with self._open(name, require_controller=True) as dev:
            await dev.set_totalizer_batch(batch_volume, batch, units)

    # ── Gas / meter utilities (FlowMeter and FlowController) ─────────────────

    async def set_gas(self, name: str, gas: str | int) -> None:
        """Set the gas type.

        Args:
            name: Device name.
            gas:  Gas name string (e.g. 'N2', 'Air') or mix number integer.
        """
        async with self._open(name) as dev:
            await dev.set_gas(gas)

    async def lock(self, name: str) -> None:
        """Lock the front-panel buttons."""
        async with self._open(name) as dev:
            await dev.lock()

    async def unlock(self, name: str) -> None:
        """Unlock the front-panel buttons."""
        async with self._open(name) as dev:
            await dev.unlock()

    async def is_locked(self, name: str) -> bool:
        """Return True if front-panel buttons are locked."""
        async with self._open(name) as dev:
            return await dev.is_locked()

    async def tare_pressure(self, name: str) -> None:
        """Tare the pressure sensor."""
        async with self._open(name) as dev:
            await dev.tare_pressure()

    async def tare_volumetric(self, name: str) -> None:
        """Tare the volumetric flow sensor."""
        async with self._open(name) as dev:
            await dev.tare_volumetric()

    async def reset_totalizer(self, name: str) -> None:
        """Reset the totalizer counter."""
        async with self._open(name) as dev:
            await dev.reset_totalizer()

    async def get_firmware(self, name: str) -> str:
        """Return firmware version string."""
        async with self._open(name) as dev:
            return await dev.get_firmware()

    # ── Display ───────────────────────────────────────────────────────────────

    @staticmethod
    def print_states(states: dict[str, Any]) -> None:
        """Pretty-print a states dict returned by get_all()."""
        for name, state in states.items():
            print(f"[{name}]")
            if isinstance(state, dict) and "error" in state:
                print(f"  ✗ {state['error']}")
            elif isinstance(state, dict):
                for k, v in state.items():
                    print(f"  {k}: {v}")
            else:
                print(f"  {state}")


# ──────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────────────────────────────────────

async def _cli(args: list[str]) -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Scan for Alicat flow devices.")
    parser.add_argument("--ports",    nargs="*", help="COM ports to scan (default: all active)")
    parser.add_argument("--baud",     nargs="*", type=int, help="Baud rates to try")
    parser.add_argument("--units",    nargs="*", help="Unit IDs to probe, e.g. A B C")
    parser.add_argument("--timeout",  type=float, default=0.75, help="Per-unit timeout (s)")
    parser.add_argument("--expected", nargs="*", dest="expected_ids",
                        help="Expected unit IDs (e.g. A B C D). Scan stops early once all found.")
    parser.add_argument("--states", action="store_true", help="After scan, print live states")
    ns = parser.parse_args(args)

    mgr = AlicatManager()
    await mgr.scan(
        ports=ns.ports,
        baudrates=ns.baud,
        units=ns.units,
        timeout=ns.timeout,
        expected_ids=ns.expected_ids,
    )
    mgr.show_map()

    if ns.states:
        print("\nReading states…")
        states = await mgr.get_all()
        mgr.print_states(states)


if __name__ == "__main__":
    import sys
    asyncio.run(_cli(sys.argv[1:]))
