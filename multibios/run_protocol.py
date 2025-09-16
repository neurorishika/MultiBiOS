#!/usr/bin/env python3
"""
Run hardware-clocked NI USB-6353 protocol and log MFC analog feedback + READY DI rails.

- DO (master): drives S bits, LOAD_REQ, RCK, triggers
- AO (slave): drives MFC setpoints
- AI (slave): records MFC feedback (0–10 V) locked to DO sample clock
- DI (slave): records READY rails from Teensy, locked to DO sample clock

Artifacts are written to data/runs/YYYY-MM-DD_HH-MM-SS/
- compiled_do.npz / compiled_ao.npz
- capture_ai.npz (MFC feedback, optional)
- capture_di.npz (READY rails, optional)
- do_map.json / ao_map.json / di_map.json
- rck_edges.csv (planned commits)
- digital_edges.csv (rising/falling edges for all DO lines)
- ready_edges.csv (rising/falling edges for READY DI lines, if present)
- preview.html (interactive Plotly: DO + AO + AI/DI overlays)
"""

from __future__ import annotations

import argparse, json, time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Optional, List

import numpy as np
import yaml

import nidaqmx
from nidaqmx.constants import AcquisitionType, Edge, LineGrouping
from nidaqmx.stream_writers import DigitalMultiChannelWriter, AnalogMultiChannelWriter
from nidaqmx.stream_readers import AnalogMultiChannelReader, DigitalMultiChannelReader
from multibios.viz_helpers import make_protocol_figure, write_edge_csv

# Plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Compiler
from multibios.protocol.schema import ProtocolCompiler, TimingConfig, CompileError

# Visualization helpers
from multibios.viz_helpers import make_protocol_figure, write_edge_csv


# ----------------------------- hardware adapter -----------------------------
@dataclass
class HardwareMap:
    device: str
    digital_outputs: Dict[str, str]
    analog_outputs: Dict[str, str]
    analog_inputs: Dict[str, str]
    digital_inputs: Dict[str, str]  # READY rails (Teensy -> NI-DAQ)

    # adapter fields the compiler expects
    @property
    def do_lines(self) -> Dict[str, str]:
        return self.digital_outputs

    @property
    def ao_channels(self) -> Dict[str, str]:
        return self.analog_outputs


def load_hardware(path: Path) -> HardwareMap:
    y = yaml.safe_load(path.read_text())
    return HardwareMap(
        device=y["device"],
        digital_outputs=y.get("digital_outputs", {}),
        analog_outputs=y.get("analog_outputs", {}),
        analog_inputs=y.get("analog_inputs", {}),
        digital_inputs=y.get("digital_inputs", {}),
    )


# ----------------------------- logging utils --------------------------------
def ensure_run_dir(root: Path) -> Path:
    ts = time.strftime("%Y-%m-%d_%H-%M-%S")
    d = root / ts
    d.mkdir(parents=True, exist_ok=False)
    return d


# ----------------------------- main -----------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Run NI 6353 hardware-clocked protocol with AI/DI logging."
    )
    ap.add_argument(
        "--yaml", default="config/example_protocol.yaml", help="Protocol YAML"
    )
    ap.add_argument(
        "--hardware", default="config/hardware.yaml", help="Hardware map YAML"
    )
    ap.add_argument("--device", help="Override device name (else from hardware.yaml)")
    ap.add_argument("--dry-run", action="store_true", help="Compile only; no hardware")
    ap.add_argument(
        "--interactive",
        action="store_true",
        help="Always save interactive HTML preview",
    )
    ap.add_argument("--out-root", default="data/runs", help="Run folder root")
    # Optional pulse tuning overrides (otherwise read from YAML)
    ap.add_argument("--preload-lead-ms", type=int)
    ap.add_argument("--load-req-ms", type=int)
    ap.add_argument("--rck-ms", type=int)
    ap.add_argument("--trig-ms", type=int)
    ap.add_argument(
        "--seed",
        type=int,
        help="Override protocol.timing.seed for reproducible randomization",
    )
    args = ap.parse_args()

    proto_path = Path(args.yaml)
    hw_path = Path(args.hardware)
    if not proto_path.exists():
        raise SystemExit(f"Protocol file not found: {proto_path}")
    if not hw_path.exists():
        raise SystemExit(f"Hardware file not found: {hw_path}")

    hw = load_hardware(hw_path)
    if args.device:
        hw.device = args.device

    # Timing config
    y = yaml.safe_load(proto_path.read_text())
    if args.seed is not None:
        y.setdefault("protocol", {}).setdefault("timing", {})["seed"] = int(args.seed)
    t = y.get("protocol", {}).get("timing", {})
    tcfg = TimingConfig(
        base_unit=t.get("base_unit", "ms"),
        sample_rate=int(t.get("sample_rate", 1000)),
        camera_interval_ms=int(t.get("camera_interval", 0)),
        camera_pulse_ms=int(t.get("camera_pulse_duration", 5)),
        preload_lead_ms=int(
            args.preload_lead_ms
            if args.preload_lead_ms is not None
            else t.get("preload_lead_ms", 2)
        ),
        load_req_ms=int(
            args.load_req_ms
            if args.load_req_ms is not None
            else t.get("load_req_ms", 1)
        ),
        rck_pulse_ms=int(
            args.rck_ms if args.rck_ms is not None else t.get("rck_pulse_ms", 1)
        ),
        trig_pulse_ms=int(
            args.trig_ms if args.trig_ms is not None else t.get("trig_pulse_ms", 5)
        ),
        setup_hold_samples=int(t.get("setup_hold_samples", 5)),
    )

    # Compile
    comp = ProtocolCompiler(hw, tcfg)
    try:
        comp.compile_from_yaml(y)
    except CompileError as e:
        raise SystemExit(f"[compile error] {e}")

    # Run folder + inputs
    run_dir = ensure_run_dir(Path(args.out_root))
    (run_dir / "compile_report.json").write_text(json.dumps(comp.report, indent=2))
    (run_dir / "protocol.yaml").write_text(proto_path.read_text())
    (run_dir / "hardware.yaml").write_text(hw_path.read_text())
    (run_dir / "meta.json").write_text(
        json.dumps(
            {
                "device": hw.device,
                "sample_rate": comp.tcfg.sample_rate,
                "duration_ms": comp.N * comp.dt_ms,
                "rng_seed": comp.rng_seed,
                "args": vars(args),
            },
            indent=2,
        )
    )
    # Planned RCK edges
    with (run_dir / "rck_edges.csv").open("w") as f:
        f.write("signal,sample_idx,time_ms\n")
        for sig, si, tms in comp.rck_log:
            f.write(f"{sig},{si},{tms:.3f}\n")

    # Save compiled arrays + maps
    do_names = comp.line_order
    ao_names = comp.ao_order
    (run_dir / "do_map.json").write_text(
        json.dumps(
            {"names": do_names, "phys": [hw.digital_outputs[n] for n in do_names]},
            indent=2,
        )
    )
    (run_dir / "ao_map.json").write_text(
        json.dumps(
            {"names": ao_names, "phys": [hw.analog_outputs[n] for n in ao_names]},
            indent=2,
        )
    )
    # DI map (READY inputs) — write even if empty for consistency
    di_names_cfg = list(hw.digital_inputs.keys())
    (run_dir / "di_map.json").write_text(
        json.dumps(
            {
                "names": di_names_cfg,
                "phys": [hw.digital_inputs[n] for n in di_names_cfg],
            },
            indent=2,
        )
    )

    np.savez_compressed(run_dir / "compiled_do.npz", data=comp.do.astype(np.bool_))
    np.savez_compressed(run_dir / "compiled_ao.npz", data=comp.ao.astype(np.float32))

    # Digital edge log (super helpful to diff runs)
    write_edge_csv(
        run_dir / "digital_edges.csv", do_names, comp.do.astype(bool), comp.dt_ms
    )

    # --- Dry run or interactive preview
    t_ms = np.arange(comp.N) * comp.dt_ms
    fig = make_protocol_figure(
        t_ms,
        comp.do.astype(bool),
        do_names,
        comp.ao,
        ao_names,
        title="Preview (no DAQ)",
        rck_log=comp.rck_log,
    )
    fig.write_html(run_dir / "preview.html", include_plotlyjs="cdn")

    if args.dry_run:
        print(f"Dry-run complete. Preview: {run_dir/'preview.html'}")
        return

    # --- DAQ execution: DO master, AO slave, AI slave (MFC feedback), DI slave (READY)
    N = comp.N
    rate = comp.tcfg.sample_rate

    ai_names = list(hw.analog_inputs.keys())
    ai_phys = [hw.analog_inputs[n] for n in ai_names]

    di_names = list(hw.digital_inputs.keys())
    di_phys = [hw.digital_inputs[n] for n in di_names]

    with (
        nidaqmx.Task("DO_MASTER") as do_task,
        nidaqmx.Task("AO_SLAVE") as ao_task,
        nidaqmx.Task("AI_SLAVE") as ai_task,
        nidaqmx.Task("DI_READY") as di_task,
    ):

        # DO master lines
        for ch in [hw.digital_outputs[n] for n in do_names]:
            do_task.do_channels.add_do_chan(
                ch, line_grouping=LineGrouping.CHAN_PER_LINE
            )
        do_task.timing.cfg_samp_clk_timing(
            rate=rate,
            active_edge=Edge.RISING,
            sample_mode=AcquisitionType.FINITE,
            samps_per_chan=N,
        )
        DigitalMultiChannelWriter(do_task.out_stream).write_many_sample(comp.do)

        # AO slave
        if ao_names:
            ao_task.ao_channels.add_ao_voltage_chan(
                ",".join([hw.analog_outputs[n] for n in ao_names]),
                min_val=0.0,
                max_val=5.0,
            )
            ao_task.timing.cfg_samp_clk_timing(
                rate=rate,
                source=f"/{hw.device}/do/SampleClock",
                active_edge=Edge.RISING,
                sample_mode=AcquisitionType.FINITE,
                samps_per_chan=N,
            )
            ao_task.triggers.start_trigger.cfg_dig_edge_start_trig(
                f"/{hw.device}/do/StartTrigger"
            )
            AnalogMultiChannelWriter(ao_task.out_stream).write_many_sample(comp.ao)

        # AI slave (MFC feedback)
        ai_buf = None
        if ai_phys:
            ai_task.ai_channels.add_ai_voltage_chan(
                ",".join(ai_phys), min_val=0.0, max_val=10.0
            )
            ai_task.timing.cfg_samp_clk_timing(
                rate=rate,
                source=f"/{hw.device}/do/SampleClock",
                active_edge=Edge.RISING,
                sample_mode=AcquisitionType.FINITE,
                samps_per_chan=N,
            )
            ai_task.triggers.start_trigger.cfg_dig_edge_start_trig(
                f"/{hw.device}/do/StartTrigger"
            )
            ai_reader = AnalogMultiChannelReader(ai_task.in_stream)
            ai_buf = np.zeros((len(ai_phys), N), dtype=np.float64)

        # DI slave (READY inputs from Teensy)
        di_buf = None
        if di_phys:
            for ch in di_phys:
                di_task.di_channels.add_di_chan(
                    ch, line_grouping=LineGrouping.CHAN_PER_LINE
                )
            di_task.timing.cfg_samp_clk_timing(
                rate=rate,
                source=f"/{hw.device}/do/SampleClock",
                active_edge=Edge.RISING,
                sample_mode=AcquisitionType.FINITE,
                samps_per_chan=N,
            )
            di_task.triggers.start_trigger.cfg_dig_edge_start_trig(
                f"/{hw.device}/do/StartTrigger"
            )
            di_reader = DigitalMultiChannelReader(di_task.in_stream)
            di_buf = np.zeros((len(di_phys), N), dtype=np.bool_)

        # Start slaves then DO
        if ao_names:
            ao_task.start()
        if ai_phys:
            ai_task.start()
        if di_phys:
            di_task.start()
        do_task.start()

        do_task.wait_until_done(timeout=max(10.0, N / rate + 5.0))
        do_task.stop()
        if ao_names:
            ao_task.stop()

        if ai_phys:
            ai_reader.read_many_sample(
                ai_buf,
                number_of_samples_per_channel=N,
                timeout=max(10.0, N / rate + 5.0),
            )
            ai_task.stop()
            np.savez_compressed(
                run_dir / "capture_ai.npz",
                names=np.array(ai_names, dtype=object),
                data=ai_buf.astype(np.float32),
            )

        if di_phys:
            di_reader.read_many_sample(
                di_buf,
                number_of_samples_per_channel=N,
                timeout=max(10.0, N / rate + 5.0),
            )
            di_task.stop()
            np.savez_compressed(
                run_dir / "capture_di.npz",
                names=np.array(di_names, dtype=object),
                data=di_buf.astype(np.bool_),
            )

    # Post-run interactive viz with AI/DI overlays (if recorded)
    di_names_overlay = di_data_overlay = None
    if (run_dir / "capture_di.npz").exists():
        npz_di = np.load(run_dir / "capture_di.npz", allow_pickle=True)
        di_names_overlay = list(npz_di["names"])
        di_data_overlay = npz_di["data"].astype(bool)

    ai_names_overlay = ai_data_overlay = None
    if (run_dir / "capture_ai.npz").exists():
        npz_ai = np.load(run_dir / "capture_ai.npz", allow_pickle=True)
        ai_names_overlay = list(npz_ai["names"])
        ai_data_overlay = npz_ai["data"]

    fig = make_protocol_figure(
        t_ms,
        comp.do.astype(bool),
        do_names,
        comp.ao,
        ao_names,
        ai=ai_data_overlay,
        ai_names=ai_names_overlay,
        di=di_data_overlay,
        di_names=di_names_overlay,
        rck_log=comp.rck_log,
        title="Protocol (DO/AO) + READY (DI) + MFC Feedback (AI)",
    )
    fig.write_html(run_dir / "preview.html", include_plotlyjs="cdn")

    # READY edge log if present
    if (run_dir / "capture_di.npz").exists():
        write_edge_csv(
            run_dir / "ready_edges.csv", di_names_overlay, di_data_overlay, comp.dt_ms
        )

    print(f"Run complete. See interactive preview: {run_dir/'preview.html'}")


if __name__ == "__main__":
    main()
