"""flow_monitor.py — fixed-rate Alicat monitor with jitter stats."""
from __future__ import annotations

import argparse
import asyncio
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor

from multibios.alicat_manager import AlicatManager
from alicat import FlowMeter

_executor = ThreadPoolExecutor(max_workers=1)


def _render(results: dict, stats: dict) -> str:
    lines: list[str] = []
    names = sorted(results.keys())
    width = max((len(n) for n in names), default=10)

    lines.append(f"{'Device':<{width}}  {'mass_flow':>14}  {'status'}")
    lines.append("─" * (width + 42))

    for name in names:
        state = results[name]
        if isinstance(state, dict) and "error" not in state:
            flow = state.get("mass_flow", "n/a")
            gas = state.get("gas", "")
            setpt = state.get("setpoint", "")
            if isinstance(flow, (int, float)):
                flow_txt = f"{flow:>14.4f}"
            else:
                flow_txt = f"{str(flow):>14}"
            lines.append(f"{name:<{width}}  {flow_txt}  gas={gas}  setpoint={setpt}")
        else:
            err = state.get("error", "unknown error") if isinstance(state, dict) else str(state)
            lines.append(f"{name:<{width}}  {'ERROR':>14}  {err}")

    lines.append("")
    lines.append(
        f"target={stats['target_ms']:.1f} ms  "
        f"read={stats['read_ms']:.1f} ms  "
        f"period(avg/min/max)={stats['avg_period_ms']:.1f}/{stats['min_period_ms']:.1f}/{stats['max_period_ms']:.1f} ms  "
        f"jitter(avg/max)={stats['avg_jitter_ms']:.2f}/{stats['max_jitter_ms']:.2f} ms  "
        f"overruns={stats['overruns']}"
    )
    lines.append("Ctrl+C to quit")
    return "\n".join(lines)


def _print_frame(text: str) -> None:
    sys.stdout.write("\033[H\033[J" + text + "\n")
    sys.stdout.flush()


async def _printer(frame_q: asyncio.Queue[str]) -> None:
    loop = asyncio.get_running_loop()
    while True:
        text = await frame_q.get()
        try:
            await loop.run_in_executor(_executor, _print_frame, text)
        finally:
            frame_q.task_done()


async def monitor(interval: float, do_scan: bool, scan_kwargs: dict, setpoint: float | None) -> None:
    mgr = AlicatManager()

    if do_scan:
        print("Scanning for devices …")
        await mgr.scan(**scan_kwargs)

    if not mgr.names():
        print("No devices found. Run with --scan or ensure the cache is populated.")
        return

    mgr.show_map()
    await asyncio.sleep(0.5)

    # ── Optional pre-run setpoint ──────────────────────────────────────────────
    controllers = [
        n for n in mgr.names()
        if mgr.info(n)["type"] == "controller"
        or "setpoint" in (mgr.info(n).get("last_state") or {})
    ]

    async def _apply_setpoints(target: float, label: str) -> None:
        """Set, verify readback, retry once, then warn on persistent mismatch."""
        print(f"{label} {target} on {controllers} …")
        results = await mgr.set_all({n: target for n in controllers})
        for name, err in results.items():
            if err:
                print(f"  ⚠  {name}: set failed — {err['error']}")

        # Readback verification
        await asyncio.sleep(0.15)
        states = await mgr.get_all(controllers)
        retry = {}
        for name, state in states.items():
            if isinstance(state, dict) and "error" not in state:
                actual = state.get("setpoint", None)
                if actual is not None and abs(float(actual) - target) > 0.01:
                    retry[name] = target
                    print(f"  ↻  {name}: setpoint readback {actual} ≠ {target}, retrying…")

        if retry:
            await mgr.set_all(retry)
            await asyncio.sleep(0.15)
            states2 = await mgr.get_all(list(retry.keys()))
            for name, state in states2.items():
                if isinstance(state, dict) and "error" not in state:
                    actual = state.get("setpoint", None)
                    if actual is not None and abs(float(actual) - target) > 0.01:
                        print(f"  ✗  {name}: setpoint stuck at {actual} after retry — check device!")
                    else:
                        print(f"  ✓  {name}: setpoint confirmed {actual}")

    if setpoint is not None and controllers:
        await _apply_setpoints(setpoint, "Setting")
        await asyncio.sleep(0.1)

    frame_q: asyncio.Queue[str] = asyncio.Queue(maxsize=1)
    printer_task = asyncio.create_task(_printer(frame_q))

    periods: list[float] = []
    jitters: list[float] = []
    overruns = 0
    last_start: float | None = None

    sys.stdout.write("\033[?25l")
    sys.stdout.flush()

    try:
        next_tick = time.perf_counter()

        while True:
            scheduled = next_tick
            now = time.perf_counter()
            if now < scheduled:
                await asyncio.sleep(scheduled - now)

            start = time.perf_counter()

            if last_start is not None:
                periods.append(start - last_start)
            last_start = start

            jitters.append(start - scheduled)

            results = await mgr.get_all()
            read_done = time.perf_counter()
            read_time = read_done - start

            if len(periods) > 500:
                periods.pop(0)
            if len(jitters) > 500:
                jitters.pop(0)

            stats = {
                "target_ms": interval * 1000,
                "read_ms": read_time * 1000,
                "avg_period_ms": (statistics.fmean(periods) * 1000) if periods else 0.0,
                "min_period_ms": (min(periods) * 1000) if periods else 0.0,
                "max_period_ms": (max(periods) * 1000) if periods else 0.0,
                "avg_jitter_ms": (statistics.fmean(abs(x) for x in jitters) * 1000) if jitters else 0.0,
                "max_jitter_ms": (max(abs(x) for x in jitters) * 1000) if jitters else 0.0,
                "overruns": overruns,
            }

            text = _render(results, stats)

            if frame_q.full():
                try:
                    frame_q.get_nowait()
                    frame_q.task_done()
                except asyncio.QueueEmpty:
                    pass
            await frame_q.put(text)

            next_tick += interval
            if read_done > next_tick:
                missed = int((read_done - next_tick) // interval) + 1
                overruns += missed
                next_tick += missed * interval

    except asyncio.CancelledError:
        pass
    finally:
        printer_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await printer_task
        # ── Zero all controllers on exit ───────────────────────────────────────
        if controllers:
            sys.stdout.write("\033[?25h\n")
            sys.stdout.flush()
            # Purge any stale/broken connections left by a cancelled get_all()
            # so the zeroing opens fresh serial connections.
            for _port, (_conn, _) in list(FlowMeter.open_ports.items()):
                with contextlib.suppress(Exception):
                    _conn.writer.close()
            FlowMeter.open_ports.clear()
            await _apply_setpoints(0.0, "Zeroing")
        sys.stdout.write("\033[?25h\n")
        sys.stdout.flush()


def main() -> None:
    parser = argparse.ArgumentParser(description="Fixed-rate live mass-flow monitor for Alicat devices.")
    parser.add_argument("--interval", type=float, default=0.5,
                        help="Target sampling interval in seconds")
    parser.add_argument("--scan", action="store_true",
                        help="Re-scan COM ports before starting")
    parser.add_argument("--ports", nargs="*", help="COM ports to scan")
    parser.add_argument("--baud", nargs="*", type=int, help="Baud rates to try during scan")
    parser.add_argument("--setpoint", type=float, default=None,
                        help="Flow-rate setpoint applied to all controllers before the run "
                             "(zeroed automatically on exit)")
    ns = parser.parse_args()

    scan_kwargs: dict = {}
    if ns.ports:
        scan_kwargs["ports"] = ns.ports
    if ns.baud:
        scan_kwargs["baudrates"] = ns.baud

    try:
        asyncio.run(monitor(ns.interval, ns.scan, scan_kwargs, ns.setpoint))
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    import contextlib
    main()
