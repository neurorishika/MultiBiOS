#!/usr/bin/env python3
"""
Visualize a logged run (interactive Plotly):
- Row 1: ALL digital rails stacked (state bits, LOAD_REQ, RCK, triggers)
- Row 2: AO setpoints
- Row 3: AI feedback (MFC) if present
Writes/opens preview.html inside the run folder.
"""

from __future__ import annotations

import sys, json
from pathlib import Path
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def make_fig(
    t_ms, do, do_names, ao, ao_names, ai=None, ai_names=None, title="Run Preview"
):
    has_ai = ai is not None and ai_names is not None and len(ai_names) > 0
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
                "Digital Outputs (click legend groups to toggle)",
                "Analog Outputs (AO setpoints)",
            )
            if not has_ai
            else (
                "Digital Outputs (click legend groups to toggle)",
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
            y = do[do_names.index(name)].astype(float) + y_offset
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
                    customdata=do[do_names.index(name)].astype(int),
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
                y=ao[i],
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
                    y=ai[i],
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

    # Clean axes styling
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


def main():
    if len(sys.argv) < 2:
        print("usage: python -m multibios.viz_protocol <run_folder>")
        sys.exit(1)
    run = Path(sys.argv[1])
    if not run.exists():
        raise SystemExit(f"run folder not found: {run}")

    meta = json.loads((run / "meta.json").read_text())
    sr = meta["sample_rate"]
    dt_ms = 1000.0 / sr

    do_map = json.loads((run / "do_map.json").read_text())
    ao_map = json.loads((run / "ao_map.json").read_text())
    do = np.load(run / "compiled_do.npz")["data"]
    ao = np.load(run / "compiled_ao.npz")["data"]
    N = do.shape[1]
    t_ms = np.arange(N) * dt_ms

    ai = None
    ai_names = None
    if (run / "capture_ai.npz").exists():
        npz = np.load(run / "capture_ai.npz", allow_pickle=True)
        ai = npz["data"]
        ai_names = list(npz["names"])

    fig = make_fig(
        t_ms,
        do,
        do_map["names"],
        ao,
        ao_map["names"],
        ai,
        ai_names,
        title=f"Run: {run.name}",
    )
    html_path = run / "preview.html"
    fig.write_html(html_path, include_plotlyjs="cdn")
    print(f"Wrote {html_path.resolve()}")


if __name__ == "__main__":
    main()
