#!/usr/bin/env python3
"""Run a short live FicTrac acquisition probe through the internal client.

This is a bounded end-to-end validation of the current side-camera -> FicTrac
-> MultiBiOS callback path. It launches FicTrac with the given config, prints a
few realtime callback frames, and then exits cleanly.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from multibios.fictrac_client import BaseFicTracCallback, FicTracDriver
from multibios.fictrac_config import default_fictrac_config_path
from multibios.fictrac_runtime import prepare_fictrac_runtime


class ProbeCallback(BaseFicTracCallback):
    def __init__(self, frame_limit: int) -> None:
        self.frame_limit = frame_limit
        self.frames: list[dict[str, float | int]] = []

    def setup_callback(self):
        print("callback_setup")

    def shutdown_callback(self):
        print("callback_shutdown")

    def process_callback(self, track_state):
        row = {
            "frame_cnt": int(track_state.frame_cnt),
            "timestamp": float(track_state.timestamp),
            "speed": float(track_state.speed),
            "heading": float(track_state.heading),
        }
        self.frames.append(row)
        print(
            f"frame {row['frame_cnt']} ts={row['timestamp']:.6f} "
            f"speed={row['speed']:.6f} heading={row['heading']:.6f}"
        )
        return len(self.frames) < self.frame_limit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Short live FicTrac acquisition probe")
    parser.add_argument(
        "--config",
        type=Path,
        default=default_fictrac_config_path(),
        help="Path to FicTrac camera config",
    )
    parser.add_argument(
        "--fictrac-bin",
        type=Path,
        default=Path(r"C:\Rishika\MultiBiOS\assets\fictrac-spinnaker\fictrac-spinnaker.exe"),
        help="Path to FicTrac binary",
    )
    parser.add_argument(
        "--console-output",
        type=Path,
        default=Path("fictrac_probe_output.txt"),
        help="Console output file path passed to FicTracDriver",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=5,
        help="Number of callback frames to collect before exiting",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.frames <= 0:
        raise SystemExit("--frames must be > 0")
    if not args.config.exists():
        raise SystemExit(f"Config file not found: {args.config}")
    if not args.fictrac_bin.exists():
        raise SystemExit(f"FicTrac binary not found: {args.fictrac_bin}")

    runtime_dirs = prepare_fictrac_runtime()
    if runtime_dirs:
        print("runtime_dirs")
        for runtime_dir in runtime_dirs:
            print(runtime_dir)

    callback = ProbeCallback(frame_limit=args.frames)
    driver = FicTracDriver(
        config_file=str(args.config),
        console_ouput_file=str(args.console_output),
        track_change_callback=callback,
        plot_on=False,
        fic_trac_bin_path=str(args.fictrac_bin),
    )
    driver.average_fps_threshold = 0

    started = time.perf_counter()
    try:
        driver.run()
    except Exception as exc:
        print(f"probe_error: {exc}")
        return 1
    finally:
        elapsed = time.perf_counter() - started
        print(f"probe_elapsed_s {elapsed:.3f}")
        print(
            "probe_summary "
            + json.dumps(
                {
                    "frames_received": len(callback.frames),
                    "first_frame": callback.frames[0] if callback.frames else None,
                    "last_frame": callback.frames[-1] if callback.frames else None,
                }
            )
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())