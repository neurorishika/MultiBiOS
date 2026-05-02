#!/usr/bin/env python3
"""Scope-safe DAQ preflight tests for camera trigger and MFC AO lines.

This script is intended to be run before connecting external devices.

Available modes:
- trigger: hardware-timed pulse train on TRIG_CAMERA only
- analog: sequential stepped AO outputs on the four MFC setpoint channels
- both: run trigger test, then analog test
"""

from __future__ import annotations

import argparse
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import nidaqmx
import numpy as np
import yaml
from nidaqmx.constants import AcquisitionType, Edge, LineGrouping
from nidaqmx.stream_writers import AnalogMultiChannelWriter


@dataclass
class HardwareMap:
    device: str
    digital_outputs: Dict[str, str]
    analog_outputs: Dict[str, str]


def load_hardware(path: Path) -> HardwareMap:
    raw = yaml.safe_load(path.read_text()) or {}
    return HardwareMap(
        device=raw["device"],
        digital_outputs=raw.get("digital_outputs", {}),
        analog_outputs=raw.get("analog_outputs", {}),
    )


def setup_logging(verbose: bool) -> logging.Logger:
    logger = logging.getLogger("preconnect_scope_test")
    logger.handlers.clear()
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
    return logger


def build_trigger_waveform(
    duration_s: float,
    sample_rate: int,
    period_ms: float,
    pulse_ms: float,
) -> np.ndarray:
    total_samples = max(1, int(round(duration_s * sample_rate)))
    waveform = np.zeros(total_samples, dtype=np.bool_)

    period_samples = max(1, int(round(period_ms * sample_rate / 1000.0)))
    pulse_samples = max(1, int(round(pulse_ms * sample_rate / 1000.0)))

    for start in range(0, total_samples, period_samples):
        stop = min(total_samples, start + pulse_samples)
        waveform[start:stop] = True

    return waveform


def run_trigger_test(
    hw: HardwareMap,
    *,
    sample_rate: int,
    duration_s: float,
    period_ms: float,
    pulse_ms: float,
    logger: logging.Logger,
) -> None:
    trigger_line = hw.digital_outputs.get("TRIG_CAMERA")
    if not trigger_line:
        raise KeyError("TRIG_CAMERA is not defined in hardware.yaml")

    waveform = build_trigger_waveform(duration_s, sample_rate, period_ms, pulse_ms)
    total_samples = waveform.shape[0]
    timeout = max(5.0, duration_s + 5.0)

    logger.info("=" * 60)
    logger.info("TRIGGER TEST")
    logger.info("=" * 60)
    logger.info(f"Line:        TRIG_CAMERA -> {trigger_line}")
    logger.info(f"Sample rate: {sample_rate} Hz")
    logger.info(f"Duration:    {duration_s:.2f} s")
    logger.info(f"Period:      {period_ms:.2f} ms")
    logger.info(f"Pulse width: {pulse_ms:.2f} ms")
    logger.info("")
    logger.info("Scope connection:")
    logger.info("  Probe tip   -> TRIG_CAMERA terminal")
    logger.info("  Probe ground-> DGND terminal")
    logger.info("")
    logger.info("Expected waveform:")
    logger.info("  Idle low, repeated high pulses with the configured period and width")
    logger.info("")

    with nidaqmx.Task("PRECONNECT_TRIGGER") as task:
        task.do_channels.add_do_chan(trigger_line, line_grouping=LineGrouping.CHAN_PER_LINE)
        task.timing.cfg_samp_clk_timing(
            rate=sample_rate,
            active_edge=Edge.RISING,
            sample_mode=AcquisitionType.FINITE,
            samps_per_chan=total_samples,
        )
        task.write(waveform, auto_start=False)

        logger.info("Running trigger test now...")
        task.start()
        task.wait_until_done(timeout=timeout)
        task.stop()
        logger.info("Trigger test complete.")


def build_analog_waveform(
    ao_names: List[str],
    sample_rate: int,
    dwell_s: float,
    levels: List[float],
) -> np.ndarray:
    dwell_samples = max(1, int(round(dwell_s * sample_rate)))
    segments_per_channel = len(levels)
    total_samples = dwell_samples * segments_per_channel * len(ao_names)
    waveform = np.zeros((len(ao_names), total_samples), dtype=np.float64)

    cursor = 0
    for chan_idx in range(len(ao_names)):
        for level in levels:
            segment = slice(cursor, cursor + dwell_samples)
            waveform[:, segment] = 0.0
            waveform[chan_idx, segment] = level
            cursor += dwell_samples

    return waveform


def run_analog_test(
    hw: HardwareMap,
    *,
    sample_rate: int,
    dwell_s: float,
    logger: logging.Logger,
) -> None:
    ordered_names = [
        "mfc.air_left_setpoint",
        "mfc.air_right_setpoint",
        "mfc.odor_left_setpoint",
        "mfc.odor_right_setpoint",
    ]
    missing = [name for name in ordered_names if name not in hw.analog_outputs]
    if missing:
        raise KeyError(f"Missing AO channels in hardware.yaml: {missing}")

    ao_phys = [hw.analog_outputs[name] for name in ordered_names]
    levels = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 0.0]
    waveform = build_analog_waveform(ordered_names, sample_rate, dwell_s, levels)
    total_samples = waveform.shape[1]
    total_duration = total_samples / sample_rate
    timeout = max(10.0, total_duration + 5.0)

    logger.info("=" * 60)
    logger.info("ANALOG OUTPUT TEST")
    logger.info("=" * 60)
    logger.info(f"Sample rate: {sample_rate} Hz")
    logger.info(f"Dwell time:  {dwell_s:.2f} s per level")
    logger.info(f"Levels:      {levels}")
    logger.info(f"Duration:    {total_duration:.2f} s total")
    logger.info("")
    logger.info("Channels under test:")
    for name, phys in zip(ordered_names, ao_phys):
        logger.info(f"  {name} -> {phys}")
    logger.info("")
    logger.info("Scope connection:")
    logger.info("  Probe one AO channel at a time against its adjacent AO GND terminal")
    logger.info("")
    logger.info("Expected waveform:")
    logger.info("  One AO channel at a time steps through 0,1,2,3,4,5,0 V while the other AO lines remain at 0 V")
    logger.info("")

    with nidaqmx.Task("PRECONNECT_ANALOG") as task:
        task.ao_channels.add_ao_voltage_chan(
            ",".join(ao_phys),
            min_val=0.0,
            max_val=5.0,
        )
        task.timing.cfg_samp_clk_timing(
            rate=sample_rate,
            active_edge=Edge.RISING,
            sample_mode=AcquisitionType.FINITE,
            samps_per_chan=total_samples,
        )
        AnalogMultiChannelWriter(task.out_stream).write_many_sample(waveform)

        logger.info("Running analog output test now...")
        task.start()
        task.wait_until_done(timeout=timeout)
        task.stop()
        logger.info("Analog output test complete.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scope-safe NI-DAQ preconnect tests for camera trigger and MFC AO lines"
    )
    parser.add_argument(
        "mode",
        choices=["trigger", "analog", "both"],
        help="Which preconnect test to run",
    )
    parser.add_argument(
        "--hardware",
        default="config/hardware.yaml",
        help="Hardware configuration YAML file",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=10000,
        help="DAQ sample rate in Hz",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=5.0,
        help="Trigger-test duration in seconds",
    )
    parser.add_argument(
        "--period-ms",
        type=float,
        default=1000.0,
        help="Trigger pulse period in ms",
    )
    parser.add_argument(
        "--pulse-ms",
        type=float,
        default=10.0,
        help="Trigger pulse width in ms",
    )
    parser.add_argument(
        "--dwell",
        type=float,
        default=1.0,
        help="Analog output level dwell time in seconds",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable more verbose logging",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger = setup_logging(args.verbose)

    hw = load_hardware(Path(args.hardware))
    logger.info(f"Loaded hardware map for device {hw.device}")
    logger.info("External cameras and MFCs should remain disconnected during this preflight test.")
    logger.info("")

    if args.mode in ("trigger", "both"):
        run_trigger_test(
            hw,
            sample_rate=args.sample_rate,
            duration_s=args.duration,
            period_ms=args.period_ms,
            pulse_ms=args.pulse_ms,
            logger=logger,
        )
        logger.info("")

    if args.mode in ("analog", "both"):
        run_analog_test(
            hw,
            sample_rate=args.sample_rate,
            dwell_s=args.dwell,
            logger=logger,
        )

    logger.info("")
    logger.info("Preconnect scope test sequence finished.")


if __name__ == "__main__":
    main()