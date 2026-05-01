#!/usr/bin/env python3
"""Run a short live FicTrac acquisition probe through the internal client.

This is a bounded end-to-end validation of the current side-camera -> FicTrac
-> MultiBiOS callback path. It launches FicTrac with the given config, prints a
few realtime callback frames, and then exits cleanly.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import yaml

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
    parser.add_argument(
        "--hardware",
        type=Path,
        default=None,
        help="Optional hardware.yaml used to resolve FicTrac camera serial to the current Spinnaker index",
    )
    return parser.parse_args()


def _read_fictrac_camera_index(config_path: Path) -> int | None:
    for raw_line in config_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        key, _, value = line.partition(":")
        if key.strip() != "src_fn":
            continue
        try:
            return int(value.strip())
        except ValueError:
            return None
    return None


def _upsert_config_line(lines: list[str], key: str, value: str) -> None:
    rendered = f"{key:<17}: {value}\n"
    for idx, raw_line in enumerate(lines):
        line = raw_line.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        current_key, _, _ = line.partition(":")
        if current_key.strip() == key:
            lines[idx] = rendered
            return
    lines.append(rendered)


def _list_cameras() -> list[dict[str, Any]]:
    completed = subprocess.run(
        [sys.executable, "-m", "multibios.blackfly.preconfigure_camera", "--list-cameras"],
        capture_output=True,
        text=True,
        timeout=30.0,
    )
    stdout = completed.stdout.strip()
    stderr = completed.stderr.strip()
    if completed.returncode != 0:
        raise RuntimeError(stderr or stdout or f"camera enumeration failed with exit code {completed.returncode}")
    payload = json.loads(stdout.splitlines()[-1]) if stdout else {}
    cameras = payload.get("cameras")
    return cameras if isinstance(cameras, list) else []


def _resolve_probe_config(config_path: Path, hardware_path: Path | None) -> tuple[Path, int | None]:
    if hardware_path is None or not hardware_path.exists():
        return config_path, _read_fictrac_camera_index(config_path)

    hardware = yaml.safe_load(hardware_path.read_text(encoding="utf-8")) or {}
    fictrac_cfg = hardware.get("fictrac") if isinstance(hardware, dict) else {}
    camera_serial = ""
    if isinstance(fictrac_cfg, dict):
        camera_serial = str(fictrac_cfg.get("camera_serial") or "")
    if not camera_serial:
        return config_path, _read_fictrac_camera_index(config_path)

    for camera in _list_cameras():
        if str(camera.get("serial")) != camera_serial:
            continue
        camera_index = int(camera.get("camera_index"))
        lines = config_path.read_text(encoding="utf-8").splitlines(keepends=True)
        _upsert_config_line(lines, "src_fn", str(camera_index))
        runtime_path = config_path.with_name("fictrac_probe_runtime_config.txt")
        runtime_path.write_text("".join(lines), encoding="utf-8")
        print(
            "probe_camera_resolve "
            + json.dumps(
                {
                    "camera_serial": camera_serial,
                    "camera_index": camera_index,
                    "runtime_config": str(runtime_path),
                }
            )
        )
        return runtime_path, camera_index

    raise RuntimeError(f"Configured FicTrac camera serial {camera_serial} was not found among connected cameras")


def _reset_probe_camera(camera_index: int) -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "multibios.blackfly.preconfigure_camera",
            "--camera-index",
            str(camera_index),
            "--reset-editable",
        ],
        capture_output=True,
        text=True,
        timeout=30.0,
    )
    stdout = completed.stdout.strip()
    stderr = completed.stderr.strip()
    if completed.returncode != 0:
        detail = stderr or stdout or f"return code {completed.returncode}"
        print(f"probe_camera_reset_error {detail}")
        return
    if stdout:
        print(f"probe_camera_reset {stdout.splitlines()[-1]}")


def _inspect_probe_camera(label: str, camera_index: int) -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "multibios.blackfly.preconfigure_camera",
            "--camera-index",
            str(camera_index),
            "--inspect",
        ],
        capture_output=True,
        text=True,
        timeout=30.0,
    )
    stdout = completed.stdout.strip()
    stderr = completed.stderr.strip()
    if completed.returncode != 0:
        detail = stderr or stdout or f"return code {completed.returncode}"
        print(f"probe_camera_state_{label}_error {detail}")
        return
    if stdout:
        print(f"probe_camera_state_{label} {stdout.splitlines()[-1]}")


def main() -> int:
    args = parse_args()
    if args.frames <= 0:
        raise SystemExit("--frames must be > 0")
    if not args.config.exists():
        raise SystemExit(f"Config file not found: {args.config}")
    if not args.fictrac_bin.exists():
        raise SystemExit(f"FicTrac binary not found: {args.fictrac_bin}")

    probe_config_path, fictrac_camera_index = _resolve_probe_config(args.config, args.hardware)

    runtime_dirs = prepare_fictrac_runtime()
    if runtime_dirs:
        print("runtime_dirs")
        for runtime_dir in runtime_dirs:
            print(runtime_dir)

    callback = ProbeCallback(frame_limit=args.frames)
    driver = FicTracDriver(
        config_file=str(probe_config_path),
        console_ouput_file=str(args.console_output),
        track_change_callback=callback,
        plot_on=False,
        fic_trac_bin_path=str(args.fictrac_bin),
    )
    driver.average_fps_threshold = 0

    if fictrac_camera_index is not None:
        _inspect_probe_camera("before", fictrac_camera_index)

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
        if fictrac_camera_index is not None:
            _inspect_probe_camera("after_run", fictrac_camera_index)
            _reset_probe_camera(fictrac_camera_index)
            _inspect_probe_camera("after_reset", fictrac_camera_index)

    return 0


if __name__ == "__main__":
    sys.exit(main())