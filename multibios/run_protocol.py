#!/usr/bin/env python3
"""
Run hardware-clocked NI USB-6363 protocol and log MFC analog feedback.

- DO (master): drives S bits, LOAD_REQ, RCK, triggers
- AO (slave): drives MFC setpoints
- AI (slave): records MFC feedback (0–5 V) locked to DO/AO sample clock

Artifacts are written to data/runs/YYYY-MM-DD_HH-MM-SS/
- compiled_do.npz / compiled_ao.npz
- capture_ai.npz (MFC feedback, optional)
- do_map.json / ao_map.json
- rck_edges.csv (planned commits)
- digital_edges.csv (rising/falling edges for all DO lines)
- preview.html (interactive Plotly: all DO + AO + AI)
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
from nidaqmx.stream_readers import AnalogMultiChannelReader

# Plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Compiler
from multibios.protocol.schema import ProtocolCompiler, TimingConfig, CompileError


# ----------------------------- hardware adapter -----------------------------
@dataclass
class HardwareMap:
    device: str
    digital_outputs: Dict[str, str]
    analog_outputs: Dict[str, str]
    analog_inputs: Dict[str, str]

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
    )


# ----------------------------- logging utils --------------------------------
def ensure_run_dir(root: Path) -> Path:
    ts = time.strftime("%Y-%m-%d_%H-%M-%S")
    d = root / ts
    d.mkdir(parents=True, exist_ok=False)
    return d


def write_digital_edges_csv(path: Path, names: List[str], do: np.ndarray, dt_ms: float):
    """
    Write a compact change log of all digital lines:
    line_name,edge_type,sample_idx,time_ms
    edge_type in {rising, falling}
    """
    with path.open("w") as f:
        f.write("line,edge_type,sample_idx,time_ms\n")
        for i, name in enumerate(names):
            v = do[i].astype(np.int8)
            dv = np.diff(v, prepend=v[0])
            idxs = np.nonzero(dv)[0]
            for si in idxs:
                et = "rising" if dv[si] > 0 else "falling"
                f.write(f"{name},{et},{si},{si*dt_ms:.3f}\n")


# ----------------------------- interactive viz ------------------------------
def make_interactive_figure(
    comp: ProtocolCompiler,
    do_names: List[str],
    ao_names: List[str],
    ai_names: Optional[List[str]] = None,
    ai_data: Optional[np.ndarray] = None,
    title: str = "Protocol Preview",
) -> go.Figure:
    """
    Build a Plotly figure with:
      Row 1: ALL digital rails stacked (S bits, LOAD_REQ, RCK, TRIG)
      Row 2: AO setpoints (step)
      Row 3: AI feedback (lines), if provided
    """
    t_ms = np.arange(comp.N) * comp.dt_ms
    has_ai = ai_data is not None and ai_names is not None and len(ai_names) > 0

    rows = 3 if has_ai else 2
    fig = make_subplots(
        rows=rows,
        cols=1,
        shared_xaxes=True,
        row_heights=(
            [0.55, 0.45 if not has_ai else 0.25, 0.20] if has_ai else [0.6, 0.4]
        ),
        vertical_spacing=0.03,
        subplot_titles=(
            (
                "Digital Outputs (click legend to toggle)",
                "Analog Outputs (AO setpoints)",
            )
            if not has_ai
            else (
                "Digital Outputs (click legend to toggle)",
                "Analog Outputs (AO setpoints)",
                "Analog Inputs (MFC feedback)",
            )
        ),
    )

    # Define enhanced color schemes grouped by device type
    def get_signal_color_and_group(name: str) -> tuple[str, str, str]:
        """Return (color, legend_group, display_name) for a signal"""
        # Group by device type: Olfactometer Left, Olfactometer Right, Switch Valve Left, Switch Valve Right
        if "OLFACTOMETER_LEFT" in name:
            if "RCK" in name:
                return "#e74c3c", "Olfactometer Left", "RCK"
            elif "LOAD_REQ" in name:
                return "#27ae60", "Olfactometer Left", "Load Req"
            elif name.endswith("_S0"):
                return "#f39c12", "Olfactometer Left", "State S0"
            elif name.endswith("_S1"):
                return "#3498db", "Olfactometer Left", "State S1"
            elif name.endswith("_S2"):
                return "#9b59b6", "Olfactometer Left", "State S2"
            else:
                return "#95a5a6", "Olfactometer Left", name.split("_")[-1]

        elif "OLFACTOMETER_RIGHT" in name:
            if "RCK" in name:
                return "#c0392b", "Olfactometer Right", "RCK"
            elif "LOAD_REQ" in name:
                return "#229954", "Olfactometer Right", "Load Req"
            elif name.endswith("_S0"):
                return "#e67e22", "Olfactometer Right", "State S0"
            elif name.endswith("_S1"):
                return "#2980b9", "Olfactometer Right", "State S1"
            elif name.endswith("_S2"):
                return "#8e44ad", "Olfactometer Right", "State S2"
            else:
                return "#7f8c8d", "Olfactometer Right", name.split("_")[-1]

        elif "SWITCHVALVE_LEFT" in name:
            if "RCK" in name:
                return "#d63031", "Switch Valve Left", "RCK"
            elif "LOAD_REQ" in name:
                return "#1e8449", "Switch Valve Left", "Load Req"
            elif name.endswith("_S"):
                return "#16a085", "Switch Valve Left", "State"
            else:
                return "#5d6d7e", "Switch Valve Left", name.split("_")[-1]

        elif "SWITCHVALVE_RIGHT" in name:
            if "RCK" in name:
                return "#a4161a", "Switch Valve Right", "RCK"
            elif "LOAD_REQ" in name:
                return "#186a3b", "Switch Valve Right", "Load Req"
            elif name.endswith("_S"):
                return "#138d75", "Switch Valve Right", "State"
            else:
                return "#566573", "Switch Valve Right", name.split("_")[-1]

        elif "TRIG" in name:
            return "#8e44ad", "Triggers", name.replace("_", " ")
        else:
            return "#95a5a6", "Other", name.replace("_", " ")

    # --- Digital rails organized by device type
    y_offset = 0.0
    y_step = 1.2

    # Group signals by device for better visual organization
    device_groups = {
        "Olfactometer Left": [],
        "Olfactometer Right": [],
        "Switch Valve Left": [],
        "Switch Valve Right": [],
        "Triggers": [],
        "Other": [],
    }

    for name in do_names:
        _, group, _ = get_signal_color_and_group(name)
        device_groups[group].append(name)

    # Plot in device order: Olfactometer L, Olfactometer R, Switch Valve L, Switch Valve R, Triggers, Other
    plot_order = [
        "Olfactometer Left",
        "Olfactometer Right",
        "Switch Valve Left",
        "Switch Valve Right",
        "Triggers",
        "Other",
    ]

    for group in plot_order:
        for name in device_groups[group]:
            y = comp.do[comp.line_order.index(name)].astype(float) + y_offset
            color, legend_group, display_name = get_signal_color_and_group(name)

            fig.add_trace(
                go.Scatter(
                    x=t_ms,
                    y=y,
                    mode="lines",
                    name=display_name,
                    legendgroup=legend_group,
                    legendgrouptitle_text=legend_group,
                    line=dict(shape="hv", color=color, width=2),
                    hovertemplate=f"<b>{legend_group} - {display_name}</b><br>Time: %{{x:.2f}} ms<br>Level: %{{customdata}}<br><extra></extra>",
                    customdata=comp.do[comp.line_order.index(name)].astype(int),
                ),
                row=1,
                col=1,
            )

            # Add subtle baseline with signal name
            fig.add_trace(
                go.Scatter(
                    x=[t_ms[0], t_ms[-1]],
                    y=[y_offset, y_offset],
                    mode="lines+text",
                    text=["", display_name],  # Show signal name at end
                    textposition="middle right",
                    textfont=dict(size=9, color="rgba(100,100,100,0.7)"),
                    showlegend=False,
                    line=dict(color="rgba(150,150,150,0.2)", dash="dot", width=0.5),
                    hoverinfo="skip",
                ),
                row=1,
                col=1,
            )
            y_offset += y_step

    # Add RCK commit indicators
    for i, (sig, si, tms) in enumerate(comp.rck_log):
        fig.add_vline(
            x=tms,
            line_color="#e74c3c",
            line_width=2,
            opacity=0.3,
            row=1,
            col=1,
            annotation_text=f"RCK {i+1}",
            annotation_position="top",
            annotation_font_size=8,
        )

    # --- AO setpoints with distinct colors
    ao_colors = [
        "#3498db",
        "#e74c3c",
        "#2ecc71",
        "#f39c12",
        "#9b59b6",
        "#1abc9c",
        "#e67e22",
        "#34495e",
    ]
    for i, name in enumerate(ao_names):
        color = ao_colors[i % len(ao_colors)]
        fig.add_trace(
            go.Scatter(
                x=t_ms,
                y=comp.ao[i],
                mode="lines",
                name=name.replace("_", " "),
                legendgroup="Analog Outputs",
                legendgrouptitle_text="Analog Outputs",
                line=dict(shape="hv", width=3, color=color),
                hovertemplate=f"<b>{name}</b><br>Time: %{{x:.2f}} ms<br>Voltage: %{{y:.3f}} V<br><extra></extra>",
            ),
            row=2,
            col=1,
        )

    # --- AI feedback with distinct styling
    if has_ai:
        ai_colors = [
            "#2c3e50",
            "#8e44ad",
            "#16a085",
            "#d35400",
            "#c0392b",
            "#27ae60",
            "#8e44ad",
            "#7f8c8d",
        ]
        for i, name in enumerate(ai_names):
            color = ai_colors[i % len(ai_colors)]
            fig.add_trace(
                go.Scatter(
                    x=t_ms,
                    y=ai_data[i],
                    mode="lines",
                    name=name.replace("_", " "),
                    legendgroup="Analog Inputs",
                    legendgrouptitle_text="Analog Inputs",
                    line=dict(width=2, color=color),
                    hovertemplate=f"<b>{name}</b><br>Time: %{{x:.2f}} ms<br>Voltage: %{{y:.3f}} V<br><extra></extra>",
                ),
                row=3,
                col=1,
            )

    # Enhanced layout with better organization and interactivity
    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(size=16, color="#2c3e50")),
        height=800 if has_ai else 650,
        template="plotly_white",
        hovermode="x unified",
        legend=dict(
            orientation="v",
            yanchor="top",
            y=0.98,
            xanchor="left",
            x=1.02,
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="rgba(0,0,0,0.2)",
            borderwidth=1,
            font=dict(size=10),
            groupclick="toggleitem",  # Allow clicking group titles to toggle all items
            tracegroupgap=8,
        ),
        margin=dict(l=60, r=220, t=80, b=40),  # More right margin for legend
    )

    # Clean axes styling without emojis
    fig.update_xaxes(
        title_text="Time (ms)",
        row=rows,
        col=1,
        showgrid=True,
        gridcolor="rgba(128,128,128,0.2)",
        zeroline=True,
        zerolinecolor="rgba(128,128,128,0.3)",
    )
    fig.update_yaxes(
        title_text="Digital Signals", row=1, col=1, showgrid=False, zeroline=False
    )
    fig.update_yaxes(
        title_text="Voltage (V)",
        row=2,
        col=1,
        showgrid=True,
        gridcolor="rgba(128,128,128,0.2)",
    )
    if has_ai:
        fig.update_yaxes(
            title_text="Voltage (V)",
            row=3,
            col=1,
            showgrid=True,
            gridcolor="rgba(128,128,128,0.2)",
        )

    # Start with some device groups collapsed for cleaner initial view
    fig.update_traces(
        visible="legendonly", selector=dict(legendgroup="Switch Valve Left")
    )
    fig.update_traces(
        visible="legendonly", selector=dict(legendgroup="Switch Valve Right")
    )

    return fig


# ----------------------------- main -----------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Run NI 6363 hardware-clocked protocol with AI logging."
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
        setup_hold_samples=int(t.get("setup_hold_samples", 1)),
    )

    # Compile
    comp = ProtocolCompiler(hw, tcfg)
    try:
        comp.compile_from_yaml(y)
    except CompileError as e:
        raise SystemExit(f"[compile error] {e}")

    # Run folder + inputs
    run_dir = ensure_run_dir(Path(args.out_root))
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
    np.savez_compressed(run_dir / "compiled_do.npz", data=comp.do.astype(np.bool_))
    np.savez_compressed(run_dir / "compiled_ao.npz", data=comp.ao.astype(np.float32))

    # Digital edge log (super helpful to diff runs)
    write_digital_edges_csv(
        run_dir / "digital_edges.csv", do_names, comp.do, comp.dt_ms
    )

    # Always write an interactive preview (even on dry-run)
    fig = make_interactive_figure(comp, do_names, ao_names, title="Preview (no DAQ)")
    fig.write_html(run_dir / "preview.html", include_plotlyjs="cdn")

    if args.dry_run:
        print(f"Dry-run complete. Preview: {run_dir/'preview.html'}")
        return

    # --- DAQ execution: DO master, AO slave, AI slave (MFC feedback)
    N = comp.N
    rate = comp.tcfg.sample_rate
    ai_names = list(hw.analog_inputs.keys())
    ai_phys = [hw.analog_inputs[n] for n in ai_names]

    with (
        nidaqmx.Task("DO_MASTER") as do_task,
        nidaqmx.Task("AO_SLAVE") as ao_task,
        nidaqmx.Task("AI_SLAVE") as ai_task,
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

        # Start slaves then DO
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

    # Post-run interactive viz with AI overlay (if recorded)
    if (run_dir / "capture_ai.npz").exists():
        npz = np.load(run_dir / "capture_ai.npz", allow_pickle=True)
        fig = make_interactive_figure(
            comp,
            do_names,
            ao_names,
            ai_names=list(npz["names"]),
            ai_data=npz["data"],
            title="Protocol (DO/AO) + MFC Feedback (AI)",
        )
        fig.write_html(run_dir / "preview.html", include_plotlyjs="cdn")
    print(f"Run complete. See interactive preview: {run_dir/'preview.html'}")


if __name__ == "__main__":
    main()
