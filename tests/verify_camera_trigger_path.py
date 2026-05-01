#!/usr/bin/env python3
"""Verify NI-DAQ camera trigger generation against camera return lines.

This test generates a finite pulse train on TRIG_CAMERA and samples one or more
digital input lines on the same hardware timebase. Its main purpose is to tell
you whether the cameras are actually accepting each commanded trigger.

Typical usage from the MultiBiOS root:

    python tests/verify_camera_trigger_path.py --fps 60 --duration 3

Optional:
    --trigger-monitor Dev1/port0/line28

If you wire TRIG_CAMERA to a spare port0 input as a loopback monitor, the
script will also measure the actual trigger edges seen by the DAQ.
"""

from __future__ import annotations

import argparse
import csv
import threading
import sys
import time
from pathlib import Path

import nidaqmx
import numpy as np
import yaml
from nidaqmx.constants import AcquisitionType, Edge, LineGrouping

from multibios.blackfly.trigger_path_analysis import classify_missing_events, compute_trigger_timing_budget


def _read_float_node(nodemap, *node_names: str) -> float | None:
    try:
        import PySpin
    except ImportError:
        return None

    for node_name in node_names:
        node = PySpin.CFloatPtr(nodemap.GetNode(node_name))
        if PySpin.IsReadable(node):
            return float(node.GetValue())
    return None


def _read_enum_symbolic(nodemap, node_name: str) -> str | None:
    try:
        import PySpin
    except ImportError:
        return None

    node = PySpin.CEnumerationPtr(nodemap.GetNode(node_name))
    if not PySpin.IsReadable(node):
        return None
    entry = node.GetCurrentEntry()
    if not PySpin.IsReadable(entry):
        return None
    return entry.GetSymbolic()


def _read_int_node(nodemap, node_name: str) -> int | None:
    try:
        import PySpin
    except ImportError:
        return None

    node = PySpin.CIntegerPtr(nodemap.GetNode(node_name))
    if not PySpin.IsReadable(node):
        return None
    return int(node.GetValue())


def _read_string_node(nodemap, node_name: str) -> str | None:
    try:
        import PySpin
    except ImportError:
        return None

    node = PySpin.CStringPtr(nodemap.GetNode(node_name))
    if not PySpin.IsReadable(node):
        return None
    return str(node.GetValue())


def _read_camera_trigger_state(cam, camera_index: int) -> dict[str, float | int | str | None]:
    nm = cam.GetNodeMap()
    tl = cam.GetTLDeviceNodeMap()
    return {
        "camera": f"Camera {camera_index}",
        "serial": _read_string_node(tl, "DeviceSerialNumber"),
        "exposure_us": _read_float_node(nm, "ExposureTime", "ExposureTimeAbs"),
        "trigger_delay_us": _read_float_node(nm, "TriggerDelay", "TriggerDelayAbs"),
        "trigger_overlap": _read_enum_symbolic(nm, "TriggerOverlap"),
        "line_source": _read_enum_symbolic(nm, "LineSource"),
        "roi_width": _read_int_node(nm, "Width"),
        "roi_height": _read_int_node(nm, "Height"),
    }


def _resolve_arm_camera_settings(hw: dict, args: argparse.Namespace) -> dict[str, float | int | None]:
    defaults = hw.get("blackfly_defaults") or {}
    if not isinstance(defaults, dict):
        defaults = {}

    exposure_us = args.exposure_us if args.exposure_us is not None else defaults.get("exposure_us", 5000.0)
    roi_width = args.roi_width if args.roi_width is not None else defaults.get("roi_width")
    roi_height = args.roi_height if args.roi_height is not None else defaults.get("roi_height")
    binning = args.binning if args.binning is not None else defaults.get("binning", 1)
    gain_db = args.gain_db if args.gain_db is not None else defaults.get("gain_db")
    gamma = args.gamma if args.gamma is not None else defaults.get("gamma")

    return {
        "exposure_us": None if exposure_us is None else float(exposure_us),
        "roi_width": None if roi_width in (None, "") else int(roi_width),
        "roi_height": None if roi_height in (None, "") else int(roi_height),
        "binning": 1 if binning in (None, "") else int(binning),
        "gain_db": None if gain_db in (None, "") else float(gain_db),
        "gamma": None if gamma in (None, "") else float(gamma),
    }


def _arm_cameras_for_daq(arm_settings: dict[str, float | int | None]):
    try:
        from multibios.blackfly.live_view import (connect_cameras,
                                                  configure_camera_daq_mode,
                                                  release_cameras)
    except ImportError as exc:
        raise SystemExit(
            "--arm-cameras requires the multibios-blackfly environment with PySpin installed."
        ) from exc

    system, cam_list, cams = connect_cameras()
    camera_states: list[dict[str, float | int | str | None]] = []
    try:
        for idx, cam in enumerate(cams):
            print(f"Preparing camera {idx} for DAQ-triggered acquisition ...")
            configure_camera_daq_mode(
                cam,
                exposure_us=arm_settings["exposure_us"],
                roi_width=arm_settings["roi_width"],
                roi_height=arm_settings["roi_height"],
                binning=int(arm_settings["binning"]),
                gain_db=arm_settings["gain_db"],
                gamma=arm_settings["gamma"],
            )
            camera_states.append(_read_camera_trigger_state(cam, idx))
        for idx, cam in enumerate(cams):
            cam.BeginAcquisition()
            print(f"  Camera {idx} acquisition started.")
        return system, cam_list, cams, release_cameras, camera_states
    except Exception:
        release_cameras(system, cam_list, cams, restore_daq=False)
        raise


def _collect_camera_frames(cams, duration_s: float, timeout_ms: int) -> list[dict[str, float | int | str | None]]:
    results: list[dict[str, float | int | str | None]] = [
        {
            "camera": f"Camera {idx}",
            "frames": 0,
            "first_ts_ns": None,
            "last_ts_ns": None,
            "incomplete": 0,
            "timestamps_ns": [],
        }
        for idx in range(len(cams))
    ]

    stop_event = threading.Event()
    start_barrier = threading.Barrier(len(cams) + 1)
    workers: list[threading.Thread] = []

    def worker(cam_index: int, cam) -> None:
        entry = results[cam_index]
        start_barrier.wait()
        while not stop_event.is_set():
            try:
                img = cam.GetNextImage(timeout_ms)
            except Exception:
                continue

            try:
                if img.IsIncomplete():
                    entry["incomplete"] = int(entry["incomplete"]) + 1
                    continue

                frame_ts_ns = None
                try:
                    frame_ts_ns = int(img.GetTimeStamp())
                except Exception:
                    frame_ts_ns = None

                entry["frames"] = int(entry["frames"]) + 1
                if frame_ts_ns is not None:
                    if entry["first_ts_ns"] is None:
                        entry["first_ts_ns"] = frame_ts_ns
                    entry["last_ts_ns"] = frame_ts_ns
                    entry["timestamps_ns"].append(frame_ts_ns)
            finally:
                try:
                    img.Release()
                except Exception:
                    pass

    for idx, cam in enumerate(cams):
        thread = threading.Thread(target=worker, args=(idx, cam), daemon=True)
        thread.start()
        workers.append(thread)

    start_barrier.wait()
    time.sleep(duration_s)
    stop_event.set()

    for thread in workers:
        thread.join(timeout=max(2.0, duration_s + 1.0))

    return results


def load_hardware_config(path: Path) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return yaml.safe_load(handle)
    except Exception as exc:
        print(f"Error loading hardware config from {path}: {exc}")
        sys.exit(1)


def resolve_hardware_path(raw_path: str) -> Path:
    hw_path = Path(raw_path)
    if hw_path.exists():
        return hw_path

    script_dir = Path(__file__).parent.absolute()
    candidate = script_dir.parent / raw_path
    if candidate.exists():
        return candidate

    print(f"Hardware config not found at {hw_path}")
    sys.exit(1)


def build_trigger_waveform(sample_rate: int, fps: float, pulse_ms: float, duration_s: float) -> np.ndarray:
    total_samples = max(2, int(round(sample_rate * duration_s)))
    samples_per_period = max(2, int(round(sample_rate / fps)))
    pulse_samples = max(1, int(round(sample_rate * pulse_ms / 1000.0)))
    pulse_samples = min(pulse_samples, samples_per_period)

    waveform = np.zeros(total_samples, dtype=np.bool_)
    for start in range(0, total_samples, samples_per_period):
        stop = min(start + pulse_samples, total_samples)
        waveform[start:stop] = True
    return waveform


def count_rising_edges(samples: np.ndarray) -> int:
    digital = np.asarray(samples, dtype=np.bool_)
    if digital.size == 0:
        return 0
    return int(np.count_nonzero(~digital[:-1] & digital[1:])) + int(digital[0])


def compute_edge_times(samples: np.ndarray, sample_rate: float) -> np.ndarray:
    digital = np.asarray(samples, dtype=np.bool_)
    if digital.size == 0:
        return np.array([], dtype=float)
    edge_indices = np.flatnonzero(~digital[:-1] & digital[1:]) + 1
    if digital[0]:
        edge_indices = np.concatenate(([0], edge_indices))
    return edge_indices.astype(float) / sample_rate


def normalise_read_data(data, channel_count: int, sample_count: int) -> np.ndarray:
    arr = np.asarray(data)
    if channel_count == 1:
        return arr.reshape(1, sample_count)
    return arr.reshape(channel_count, sample_count)


def _print_missing_event_summary(label: str, summary: dict[str, float | int | str | None]) -> None:
    classification = str(summary["classification"])
    missing_count = int(summary["missing_count"])
    max_abs_residual_ms = None if summary["max_abs_residual_s"] is None else 1000.0 * float(summary["max_abs_residual_s"])
    span_error_ms = 1000.0 * float(summary["span_error_s"])

    message = (
        f"  {label:16s} class={classification:16s} "
        f"missing={missing_count:2d} span_error={span_error_ms:+8.3f} ms"
    )
    if max_abs_residual_ms is not None:
        message += f" residual={max_abs_residual_ms:7.3f} ms"
    if summary["missing_edge_number"] is not None:
        message += (
            f" edge#{int(summary['missing_edge_number']):4d}"
            f" @ {1000.0 * float(summary['missing_edge_time_s']):8.3f} ms"
        )
    print(message)


def _print_timing_budget(camera_states: list[dict[str, float | int | str | None]], fps: float, pulse_ms: float, requested_exposure_us: float) -> None:
    print("")
    print("Timing budget")
    for state in camera_states:
        exposure_us = float(state["exposure_us"] if state["exposure_us"] is not None else requested_exposure_us)
        trigger_delay_us = float(state["trigger_delay_us"] or 0.0)
        overlap_mode = state["trigger_overlap"] if state["trigger_overlap"] is not None else "unknown"
        budget = compute_trigger_timing_budget(
            fps_hz=fps,
            pulse_width_ms=pulse_ms,
            exposure_us=exposure_us,
            trigger_delay_us=trigger_delay_us,
            overlap_mode=str(overlap_mode),
        )
        serial = f" serial={state['serial']}" if state.get("serial") else ""
        roi = ""
        if state.get("roi_width") is not None and state.get("roi_height") is not None:
            roi = f" roi={int(state['roi_width'])}x{int(state['roi_height'])}"
        print(
            f"  {state['camera']:16s} period={float(budget['period_ms']):6.3f} ms"
            f" pulse={float(budget['pulse_width_ms']):5.3f} ms"
            f" exposure={float(budget['exposure_ms']):5.3f} ms"
            f" delay={float(budget['trigger_delay_ms']):5.3f} ms"
            f" active={float(budget['active_window_ms']):5.3f} ms"
            f" slack={float(budget['slack_to_period_ms']):+6.3f} ms"
            f" overlap={overlap_mode}{roi}{serial}"
        )
        if bool(budget["over_period"]):
            print("    [warn] Exposure plus trigger delay exceeds the frame period.")
        elif bool(budget["tight_timing"]):
            print("    [warn] Less than 0.5 ms headroom remains before the next trigger.")
        if overlap_mode not in ("ReadOut", "PreviousFrame"):
            print("    [warn] TriggerOverlap is not enabled; readout time will also limit trigger acceptance.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify camera trigger and return edge rates on the NI-DAQ.")
    parser.add_argument("--hardware", default="config/hardware.yaml", help="Path to hardware.yaml (default: config/hardware.yaml)")
    parser.add_argument("--fps", type=float, default=60.0, help="Commanded trigger frequency in Hz (default: 60)")
    parser.add_argument("--duration", type=float, default=3.0, help="Verification duration in seconds (default: 3)")
    parser.add_argument("--pulse-ms", type=float, default=1.0, help="Trigger pulse width in ms (default: 1)")
    parser.add_argument("--rate", type=int, default=10000, help="Sample rate in Hz for DO/DI (default: 10000)")
    parser.add_argument("--trigger-monitor", default=None, help="Optional spare DI line wired to TRIG_CAMERA for loopback measurement")
    parser.add_argument("--csv", type=Path, default=None, help="Optional CSV path for edge timestamps")
    parser.add_argument("--arm-cameras", action="store_true", help="Configure both cameras for DAQ trigger mode and begin acquisition before the trigger train")
    parser.add_argument("--exposure-us", type=float, default=None, help="Exposure to apply when --arm-cameras is used (default: blackfly_defaults.exposure_us or 5000)")
    parser.add_argument("--roi-width", type=int, default=None, help="ROI width to apply when --arm-cameras is used (default: blackfly_defaults.roi_width)")
    parser.add_argument("--roi-height", type=int, default=None, help="ROI height to apply when --arm-cameras is used (default: blackfly_defaults.roi_height)")
    parser.add_argument("--binning", type=int, default=None, help="Sensor binning to apply when --arm-cameras is used (default: blackfly_defaults.binning or 1)")
    parser.add_argument("--gain-db", type=float, default=None, help="Gain to apply when --arm-cameras is used (default: blackfly_defaults.gain_db)")
    parser.add_argument("--gamma", type=float, default=None, help="Gamma to apply when --arm-cameras is used (default: blackfly_defaults.gamma)")
    parser.add_argument("--camera-timeout-ms", type=int, default=250, help="GetNextImage timeout while --arm-cameras is active (default: 250)")
    args = parser.parse_args()

    if args.fps <= 0:
        raise SystemExit("--fps must be > 0")
    if args.duration <= 0:
        raise SystemExit("--duration must be > 0")
    if args.rate <= 0:
        raise SystemExit("--rate must be > 0")

    hw_path = resolve_hardware_path(args.hardware)
    print(f"Loading hardware from {hw_path}")
    hw = load_hardware_config(hw_path)

    digital_outputs = hw.get("digital_outputs") or {}
    digital_inputs = hw.get("digital_inputs") or {}

    trig_line = digital_outputs.get("TRIG_CAMERA")
    front_line = digital_inputs.get("CAMERA_FRONT_O1")
    side_line = digital_inputs.get("CAMERA_SIDE_O1")

    if not trig_line:
        raise SystemExit("TRIG_CAMERA not found in hardware config.")
    if not front_line or not side_line:
        raise SystemExit("CAMERA_FRONT_O1 and CAMERA_SIDE_O1 must be present in hardware config.")

    device = str(hw.get("device") or trig_line.split("/")[0])
    do_sample_clock = f"/{device}/do/SampleClock"
    do_start_trigger = f"/{device}/do/StartTrigger"

    di_channels: list[tuple[str, str]] = []
    if args.trigger_monitor:
        di_channels.append(("TRIG_MONITOR", args.trigger_monitor))
    di_channels.extend(
        [
            ("CAMERA_FRONT_O1", front_line),
            ("CAMERA_SIDE_O1", side_line),
        ]
    )

    waveform = build_trigger_waveform(args.rate, args.fps, args.pulse_ms, args.duration)
    sample_count = len(waveform)
    expected_edges = count_rising_edges(waveform)
    actual_duration = sample_count / args.rate

    print("Camera trigger path verification")
    print(f"  Trigger DO:      {trig_line}")
    print(f"  Monitor DIs:     {', '.join(name for name, _ in di_channels)}")
    print(f"  Commanded fps:   {args.fps:.3f}")
    print(f"  Pulse width:     {args.pulse_ms:.3f} ms")
    print(f"  Sample rate:     {args.rate} Hz")
    print(f"  Test duration:   {actual_duration:.3f} s")
    print(f"  Expected edges:  {expected_edges}")

    read_matrix = None
    frame_results = None
    start_time = time.perf_counter()
    camera_session = None
    camera_states: list[dict[str, float | int | str | None]] = []
    commanded_edge_times = compute_edge_times(waveform, args.rate)
    arm_settings = _resolve_arm_camera_settings(hw, args)

    if args.arm_cameras:
        print(
            "  Arm settings:   "
            f"exposure={float(arm_settings['exposure_us']):.0f} us, "
            f"roi={arm_settings['roi_width']}x{arm_settings['roi_height']}, "
            f"binning={int(arm_settings['binning'])}, "
            f"gain={arm_settings['gain_db']}, gamma={arm_settings['gamma']}"
        )

    try:
        if args.arm_cameras:
            camera_session = _arm_cameras_for_daq(arm_settings)
            _, _, cams, _, camera_states = camera_session
            frame_worker = threading.Thread(
                target=lambda: None,
                daemon=True,
            )

        with nidaqmx.Task("VERIFY_CAMERA_TRIGGER_DO") as do_task, nidaqmx.Task("VERIFY_CAMERA_TRIGGER_DI") as di_task:
            do_task.do_channels.add_do_chan(
                trig_line,
                name_to_assign_to_lines="TRIG_CAMERA",
                line_grouping=LineGrouping.CHAN_PER_LINE,
            )
            do_task.timing.cfg_samp_clk_timing(
                rate=args.rate,
                active_edge=Edge.RISING,
                sample_mode=AcquisitionType.FINITE,
                samps_per_chan=sample_count,
            )
            do_task.write(waveform, auto_start=False)

            for channel_name, channel_path in di_channels:
                di_task.di_channels.add_di_chan(
                    channel_path,
                    name_to_assign_to_lines=channel_name,
                    line_grouping=LineGrouping.CHAN_PER_LINE,
                )
            di_task.timing.cfg_samp_clk_timing(
                rate=args.rate,
                source=do_sample_clock,
                active_edge=Edge.RISING,
                sample_mode=AcquisitionType.FINITE,
                samps_per_chan=sample_count,
            )
            di_task.triggers.start_trigger.cfg_dig_edge_start_trig(do_start_trigger)

            print("Arming DI capture and starting DO waveform ...")
            frame_reader = None
            frame_box: dict[str, object] = {}
            if args.arm_cameras:
                _, _, cams, _, _camera_states = camera_session

                def run_frame_capture() -> None:
                    frame_box["results"] = _collect_camera_frames(cams, actual_duration + 0.5, args.camera_timeout_ms)

                frame_reader = threading.Thread(target=run_frame_capture, daemon=True)
                frame_reader.start()

            di_task.start()
            do_task.start()
            read_data = di_task.read(number_of_samples_per_channel=sample_count, timeout=max(10.0, actual_duration + 5.0))
            do_task.wait_until_done(timeout=max(10.0, actual_duration + 5.0))
            read_matrix = normalise_read_data(read_data, len(di_channels), sample_count)
            if frame_reader is not None:
                frame_reader.join(timeout=max(5.0, actual_duration + 3.0))
                frame_results = frame_box.get("results")
    except Exception as exc:
        print(f"Verification failed: {exc}")
        raise
    finally:
        if camera_session is not None:
            system, cam_list, cams, release_cameras, _camera_states = camera_session
            release_cameras(system, cam_list, cams, restore_daq=False)

    elapsed = time.perf_counter() - start_time
    print(f"DAQ tasks completed in {elapsed:.3f} s")

    rows: list[tuple[str, int, float, float]] = []
    csv_rows: list[tuple[str, int, float]] = []
    channel_edge_times: dict[str, np.ndarray] = {}
    for channel_index, (channel_name, _) in enumerate(di_channels):
        samples = read_matrix[channel_index]
        edge_count = count_rising_edges(samples)
        measured_fps = edge_count / actual_duration if actual_duration > 0 else 0.0
        acceptance_pct = (100.0 * edge_count / expected_edges) if expected_edges > 0 else 0.0
        rows.append((channel_name, edge_count, measured_fps, acceptance_pct))
        edge_times = compute_edge_times(samples, args.rate)
        channel_edge_times[channel_name] = edge_times

        for edge_number, edge_time in enumerate(edge_times, start=1):
            csv_rows.append((channel_name, edge_number, edge_time))

    print("")
    print("Results")
    print(f"  Commanded trigger edges: {expected_edges} ({expected_edges / actual_duration:.3f} Hz)")
    for channel_name, edge_count, measured_fps, acceptance_pct in rows:
        print(
            f"  {channel_name:16s} edges={edge_count:4d}  "
            f"rate={measured_fps:7.3f} Hz  acceptance={acceptance_pct:6.2f}%"
        )

    if frame_results:
        print("")
        print("Camera acquisitions")
        for entry in frame_results:
            frame_count = int(entry["frames"])
            measured_fps = frame_count / actual_duration if actual_duration > 0 else 0.0
            acceptance_pct = (100.0 * frame_count / expected_edges) if expected_edges > 0 else 0.0
            print(
                f"  {entry['camera']:16s} frames={frame_count:4d}  "
                f"rate={measured_fps:7.3f} Hz  acceptance={acceptance_pct:6.2f}%  "
                f"incomplete={int(entry['incomplete'])}"
            )

    print("")
    print("Missing-edge analysis")
    for channel_name, edge_times in channel_edge_times.items():
        summary = classify_missing_events(commanded_edge_times, edge_times)
        _print_missing_event_summary(channel_name, summary)

    if frame_results:
        for entry in frame_results:
            timestamp_ns = np.asarray(entry.get("timestamps_ns") or [], dtype=np.int64)
            observed_times_s = timestamp_ns.astype(np.float64) * 1e-9
            summary = classify_missing_events(commanded_edge_times, observed_times_s)
            _print_missing_event_summary(entry["camera"], summary)

    if camera_states:
        _print_timing_budget(camera_states, args.fps, args.pulse_ms, args.exposure_us)

    if args.csv is not None:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with open(args.csv, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["channel", "edge_index", "edge_time_s"])
            writer.writerows(csv_rows)
        print(f"\nWrote edge timestamps to {args.csv}")

    print("")
    print("Interpretation")
    if args.trigger_monitor:
        print("  - Compare TRIG_MONITOR to the commanded trigger edge count.")
        print("  - If TRIG_MONITOR is low, the generator/wiring is the issue.")
    print("  - Compare CAMERA_FRONT_O1 and CAMERA_SIDE_O1 to the commanded trigger count.")
    print("  - If camera return rates are lower than the commanded rate, the cameras are rejecting triggers or FlyCap2 is not armed as expected.")
    print("  - Missing-edge analysis distinguishes boundary misses from internal misses when trigger timing is otherwise stable.")
    if args.arm_cameras:
        print("  - Camera acquisitions report how many frames PySpin actually received while the trigger train was running.")
        print("  - Timing budget reports the actual exposure and trigger-delay headroom against the commanded trigger period.")


if __name__ == "__main__":
    main()