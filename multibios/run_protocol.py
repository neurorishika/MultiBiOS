#!/usr/bin/env python3
"""
Run hardware-clocked NI USB-6363 protocol and log MFC analog feedback.

- DO (master): drives S bits, LOAD_REQ, RCK, triggers
- AO (slave): drives MFC setpoints
- AI (slave): records MFC feedback (0–5 V) locked to DO/AO sample clock

Artifacts are written to data/runs/YYYY-MM-DD_HH-MM-SS/
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import yaml
import matplotlib.pyplot as plt

import nidaqmx
from nidaqmx.constants import AcquisitionType, Edge, LineGrouping
from nidaqmx.stream_writers import DigitalMultiChannelWriter, AnalogMultiChannelWriter
from nidaqmx.stream_readers import AnalogMultiChannelReader

# import your compiler
from multibios.protocol.schema import ProtocolCompiler, TimingConfig, CompileError


# ----------------------------- hardware adapter -----------------------------
@dataclass
class HardwareMap:
    device: str
    digital_outputs: Dict[str, str]
    analog_outputs: Dict[str, str]
    analog_inputs: Dict[str, str]

    # adapter fields expected by ProtocolCompiler
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
    )


# ----------------------------- plotting -------------------------------------
def preview_plot(
    save_to: Path,
    comp: ProtocolCompiler,
    ai_names: list[str] | None = None,
    ai_data: np.ndarray | None = None,
    title: str = "Protocol Preview",
):
    t_ms = np.arange(comp.N) * comp.dt_ms
    fig, axes = plt.subplots(
        3, 1, figsize=(14, 10), sharex=True, gridspec_kw={"height_ratios": [2, 1, 1]}
    )
    ax_do, ax_ao, ax_ai = axes

    # Digital lines: show RCK_*, *_LOAD_REQ, TRIG_*
    keys = [
        k for k in comp.line_order if any(s in k for s in ("RCK_", "LOAD_REQ", "TRIG_"))
    ]
    y0 = 0
    for name in keys:
        tr = comp.do[comp.line_order.index(name)]
        ax_do.step(t_ms, tr.astype(int) + y0, where="post", label=name)
        y0 += 1.2
    ax_do.set_ylabel("Digital (stacked)")
    if keys:
        ax_do.legend(ncol=min(4, len(keys)), fontsize=8)

    # AO setpoints
    if comp.ao.shape[0]:
        for i, name in enumerate(comp.ao_order):
            ax_ao.step(t_ms, comp.ao[i], where="post", label=name)
        ax_ao.set_ylabel("AO (V)")
        ax_ao.legend(ncol=min(4, comp.ao.shape[0]), fontsize=8)

    # AI feedback (MFC)
    if ai_data is not None and ai_data.size:
        for i, name in enumerate(ai_names or []):
            ax_ai.plot(t_ms, ai_data[i], "-", lw=0.9, label=name)
        ax_ai.set_ylabel("AI (V)")
        ax_ai.legend(ncol=min(4, (ai_names and len(ai_names)) or 1), fontsize=8)

    ax_ai.set_xlabel("Time (ms)")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(save_to, dpi=160)
    plt.close(fig)


# ----------------------------- run / log ------------------------------------
def ensure_run_dir(root: Path) -> Path:
    ts = time.strftime("%Y-%m-%d_%H-%M-%S")
    d = root / ts
    d.mkdir(parents=True, exist_ok=False)
    return d


def main():
    ap = argparse.ArgumentParser(
        description="Run NI 6363 hardware-clocked protocol with AI logging."
    )
    ap.add_argument(
        "--yaml", default="config/example_protocol.yaml", help="Protocol YAML"
    )
    ap.add_argument(
        "--hardware", default="config/hardware.yaml", help="Hardware mapping YAML"
    )
    ap.add_argument("--device", help="Override device name (else from hardware.yaml)")
    ap.add_argument("--dry-run", action="store_true", help="Compile and preview only")
    ap.add_argument(
        "--preview", action="store_true", help="Show/save preview even if running"
    )
    ap.add_argument("--out-root", default="data/runs", help="Run folder root")
    # Optional pulse tuning overrides (otherwise read from YAML timing)
    ap.add_argument("--preload-lead-ms", type=int, help="Override LOAD_REQ lead (ms)")
    ap.add_argument(
        "--load-req-ms", type=int, help="Override LOAD_REQ pulse width (ms)"
    )
    ap.add_argument("--rck-ms", type=int, help="Override RCK pulse width (ms)")
    ap.add_argument(
        "--trig-ms", type=int, help="Override microscope trigger pulse width (ms)"
    )
    args = ap.parse_args()

    # --- load config files
    proto_path = Path(args.yaml)
    hw_path = Path(args.hardware)
    if not proto_path.exists():
        raise SystemExit(f"Protocol file not found: {proto_path}")
    if not hw_path.exists():
        raise SystemExit(f"Hardware file not found: {hw_path}")

    hw = load_hardware(hw_path)
    if args.device:
        hw.device = args.device

    y = yaml.safe_load(proto_path.read_text())
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
        setup_hold_samples=int(t.get("setup_hold_samples", 1)),
    )

    # --- compile schedule
    comp = ProtocolCompiler(hw, tcfg)
    try:
        comp.compile_from_yaml(y)
    except CompileError as e:
        raise SystemExit(f"[compile error] {e}")

    # --- make run dir and save inputs
    run_dir = ensure_run_dir(Path(args.out_root))
    (run_dir / "protocol.yaml").write_text(proto_path.read_text())
    (run_dir / "hardware.yaml").write_text(hw_path.read_text())
    (run_dir / "meta.json").write_text(
        json.dumps(
            {
                "device": hw.device,
                "sample_rate": comp.tcfg.sample_rate,
                "duration_ms": comp.N * comp.dt_ms,
                "args": vars(args),
            },
            indent=2,
        )
    )
    # RCK log
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
    np.savez_compressed(run_dir / "compiled_do.npz", data=comp.do.astype(np.bool_))
    np.savez_compressed(run_dir / "compiled_ao.npz", data=comp.ao.astype(np.float32))

    # Preview (pre-run)
    if args.preview or args.dry_run:
        preview_plot(run_dir / "preview.png", comp, title="Preview (no DAQ)")

    if args.dry_run:
        print(f"Dry-run preview written to: {run_dir/'preview.png'}")
        return

    # --- DAQ: build tasks: DO master, AO slave, AI slave (MFC feedback)
    N = comp.N
    rate = comp.tcfg.sample_rate
    ai_names = list(hw.analog_inputs.keys())
    ai_phys = [hw.analog_inputs[n] for n in ai_names]

    with (
        nidaqmx.Task("DO_MASTER") as do_task,
        nidaqmx.Task("AO_SLAVE") as ao_task,
        nidaqmx.Task("AI_SLAVE") as ai_task,
    ):

        # DO master channels (per-line)
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
        do_writer = DigitalMultiChannelWriter(do_task.out_stream)
        do_writer.write_many_sample(comp.do)

        # AO slave (if any AO channels defined)
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
            ao_writer = AnalogMultiChannelWriter(ao_task.out_stream)
            ao_writer.write_many_sample(comp.ao)

        # AI slave (MFC feedback capture)
        ai_buf = None
        if ai_phys:
            ai_task.ai_channels.add_ai_voltage_chan(
                ",".join(ai_phys), min_val=0.0, max_val=10.0
            )  # 0–5V signals; 10V span OK
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

        # Start order: slaves then DO
        if ao_names:
            ao_task.start()
        if ai_phys:
            ai_task.start()
        do_task.start()

        do_task.wait_until_done(timeout=max(10.0, N / rate + 5.0))
        do_task.stop()
        if ao_names:
            ao_task.stop()

        if ai_phys:
            # Finish AI capture
            ai_reader.read_many_sample(
                ai_buf,
                number_of_samples_per_channel=N,
                timeout=max(10.0, N / rate + 5.0),
            )
            ai_task.stop()
            # Save AI data and names
            np.savez_compressed(
                run_dir / "capture_ai.npz",
                names=np.array(ai_names, dtype=object),
                data=ai_buf.astype(np.float32),
            )

    # Post-run preview (overlay AI if present)
    ai_npz = run_dir / "capture_ai.npz"
    if ai_npz.exists():
        npz = np.load(ai_npz, allow_pickle=True)
        preview_plot(
            run_dir / "preview.png",
            comp,
            ai_names=list(npz["names"]),
            ai_data=npz["data"],
            title="Protocol (DO/AO) + MFC Feedback (AI)",
        )
    print(f"Run complete. Artifacts in {run_dir.resolve()}")


if __name__ == "__main__":
    main()
