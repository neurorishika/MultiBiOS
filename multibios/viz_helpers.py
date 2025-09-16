#!/usr/bin/env python3
"""
Utilities shared by run_protocol and viz_protocol:
- Figure builder for DO/AO (+ optional AI/DI) with grouping & colors
- Digital edge CSV writer
- Run-folder loader (maps, arrays, metadata, optional overlays)
"""

from __future__ import annotations
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple

import json
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ---------- IO helpers ----------


def load_run_artifacts(run: Path) -> Dict[str, Any]:
    meta = json.loads((run / "meta.json").read_text())
    sr = float(meta["sample_rate"])
    dt_ms = 1000.0 / sr

    do_map = json.loads((run / "do_map.json").read_text())
    ao_map = json.loads((run / "ao_map.json").read_text())
    do = np.load(run / "compiled_do.npz")["data"].astype(bool)
    ao = np.load(run / "compiled_ao.npz")["data"].astype(np.float32)
    N = do.shape[1]
    t_ms = np.arange(N) * dt_ms

    ai = ai_names = None
    if (run / "capture_ai.npz").exists():
        npz = np.load(run / "capture_ai.npz", allow_pickle=True)
        ai = npz["data"]
        ai_names = list(npz["names"])

    di = di_names = None
    if (run / "capture_di.npz").exists():
        npz = np.load(run / "capture_di.npz", allow_pickle=True)
        di = npz["data"].astype(bool)
        di_names = list(npz["names"])

    # rck log optional (nice for vertical markers)
    rck_log = []
    rck_csv = run / "rck_edges.csv"
    if rck_csv.exists():
        # CSV header: signal,sample_idx,time_ms
        for line in rck_csv.read_text().splitlines()[1:]:
            sig, si, tms = line.split(",")
            rck_log.append((sig, int(si), float(tms)))

    return dict(
        t_ms=t_ms,
        do=do,
        do_names=do_map["names"],
        ao=ao,
        ao_names=ao_map["names"],
        ai=ai,
        ai_names=ai_names,
        di=di,
        di_names=di_names,
        dt_ms=dt_ms,
        rck_log=rck_log,
        meta=meta,
    )


# ---------- CSV logging ----------


def write_edge_csv(
    path: Path, names: List[str], rails_bool: np.ndarray, dt_ms: float
) -> None:
    """
    rails_bool: (num_lines, N) boolean array
    """
    with path.open("w") as f:
        f.write("line,edge_type,sample_idx,time_ms\n")
        for i, name in enumerate(names):
            v = rails_bool[i].astype(np.int8)
            dv = np.diff(v, prepend=v[0])
            idxs = np.nonzero(dv)[0]
            for si in idxs:
                et = "rising" if dv[si] > 0 else "falling"
                f.write(f"{name},{et},{si},{si*dt_ms:.3f}\n")


# ---------- Plotting ----------


def _get_color_group_label(name: str) -> tuple[str, str, str]:
    """
    Return (hex_color, legend_group, display_name) for a DO signal.
    """
    if name == "GLOBAL_LOAD_REQ":
        return "#27ae60", "Global", "GLOBAL_LOAD_REQ"

    if "OLFACTOMETER_LEFT" in name:
        if "RCK" in name:
            return "#e74c3c", "Olfactometer Left", "RCK"
        if "LOAD_REQ" in name:
            return "#27ae60", "Olfactometer Left", "Load Req"
        if name.endswith("_S0"):
            return "#f39c12", "Olfactometer Left", "State S0"
        if name.endswith("_S1"):
            return "#3498db", "Olfactometer Left", "State S1"
        if name.endswith("_S2"):
            return "#9b59b6", "Olfactometer Left", "State S2"
        return "#95a5a6", "Olfactometer Left", name.split("_")[-1]

    if "OLFACTOMETER_RIGHT" in name:
        if "RCK" in name:
            return "#c0392b", "Olfactometer Right", "RCK"
        if "LOAD_REQ" in name:
            return "#229954", "Olfactometer Right", "Load Req"
        if name.endswith("_S0"):
            return "#e67e22", "Olfactometer Right", "State S0"
        if name.endswith("_S1"):
            return "#2980b9", "Olfactometer Right", "State S1"
        if name.endswith("_S2"):
            return "#8e44ad", "Olfactometer Right", "State S2"
        return "#7f8c8d", "Olfactometer Right", name.split("_")[-1]

    if "SWITCHVALVE_LEFT" in name:
        if "RCK" in name:
            return "#d63031", "Switch Valve Left", "RCK"
        if "LOAD_REQ" in name:
            return "#1e8449", "Switch Valve Left", "Load Req"
        if name.endswith("_S"):
            return "#16a085", "Switch Valve Left", "State"
        return "#5d6d7e", "Switch Valve Left", name.split("_")[-1]

    if "SWITCHVALVE_RIGHT" in name:
        if "RCK" in name:
            return "#a4161a", "Switch Valve Right", "RCK"
        if "LOAD_REQ" in name:
            return "#186a3b", "Switch Valve Right", "Load Req"
        if name.endswith("_S"):
            return "#138d75", "Switch Valve Right", "State"
        return "#566573", "Switch Valve Right", name.split("_")[-1]

    if "TRIG" in name:
        return "#8e44ad", "Triggers", name.replace("_", " ")

    return "#95a5a6", "Other", name.replace("_", " ")


def make_protocol_figure(
    t_ms: np.ndarray,
    do: np.ndarray,
    do_names: List[str],
    ao: np.ndarray,
    ao_names: List[str],
    *,
    ai: Optional[np.ndarray] = None,
    ai_names: Optional[List[str]] = None,
    di: Optional[np.ndarray] = None,
    di_names: Optional[List[str]] = None,
    rck_log: Optional[List[Tuple[str, int, float]]] = None,
    title: str = "Protocol Preview",
) -> go.Figure:
    """Builds the interactive figure used by both run_protocol and viz_protocol."""
    has_ai = ai is not None and ai_names is not None and len(ai_names) > 0
    rows = 3 if has_ai else 2
    fig = make_subplots(
        rows=rows,
        cols=1,
        shared_xaxes=True,
        row_heights=([0.55, 0.25, 0.20] if has_ai else [0.6, 0.4]),
        vertical_spacing=0.03,
        subplot_titles=(
            ("Digital Outputs", "Analog Outputs")
            if not has_ai
            else ("Digital Outputs", "Analog Outputs", "Analog Inputs")
        ),
    )

    # Group signals by device for better visual organization
    device_groups: Dict[str, List[str]] = {
        "Global": [],
        "Olfactometer Left": [],
        "Olfactometer Right": [],
        "Switch Valve Left": [],
        "Switch Valve Right": [],
        "Triggers": [],
        "Other": [],
    }
    for name in do_names:
        _, group, _ = _get_color_group_label(name)
        device_groups.setdefault(group, []).append(name)

    plot_order = [
        "Global",
        "Olfactometer Left",
        "Olfactometer Right",
        "Switch Valve Left",
        "Switch Valve Right",
        "Triggers",
        "Other",
    ]

    # Stack DO rails
    y_offset = 0.0
    y_step = 1.2
    for group in plot_order:
        for name in device_groups.get(group, []):
            idx = do_names.index(name)
            y = do[idx].astype(float) + y_offset
            color, legend_group, display_name = _get_color_group_label(name)
            fig.add_trace(
                go.Scatter(
                    x=t_ms,
                    y=y,
                    mode="lines",
                    name=display_name,
                    legendgroup=legend_group,
                    legendgrouptitle_text=legend_group,
                    line=dict(shape="hv", color=color, width=2),
                    hovertemplate=f"<b>{legend_group} - {display_name}</b>"
                    "<br>Time: %{x:.2f} ms<br>Level: %{customdata}<extra></extra>",
                    customdata=do[idx].astype(int),
                ),
                row=1,
                col=1,
            )
            # baseline
            fig.add_trace(
                go.Scatter(
                    x=[t_ms[0], t_ms[-1]],
                    y=[y_offset, y_offset],
                    mode="lines+text",
                    text=["", display_name],
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

    # READY (DI) overlays below DO stack
    if di is not None and di_names is not None and len(di_names) == di.shape[0]:
        for i, name in enumerate(di_names):
            y = di[i].astype(float) + y_offset
            fig.add_trace(
                go.Scatter(
                    x=t_ms,
                    y=y,
                    mode="lines",
                    name=name.replace("_", " "),
                    legendgroup="Ready (DI)",
                    legendgrouptitle_text="Ready (DI)",
                    line=dict(shape="hv", width=2, color="#2ecc71"),
                    hovertemplate=f"<b>{name}</b><br>Time: %{x:.2f} ms<br>Level: %{customdata}<extra></extra>",
                    customdata=di[i].astype(int),
                ),
                row=1,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=[t_ms[0], t_ms[-1]],
                    y=[y_offset, y_offset],
                    mode="lines+text",
                    text=["", name.replace("_", " ")],
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

    # RCK markers
    if rck_log:
        for i, (_, _, tms) in enumerate(rck_log):
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

    # AO
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
                hovertemplate="<b>{name}</b><br>Time: %{x:.2f} ms<br>Voltage: %{y:.3f} V<extra></extra>",
            ),
            row=2,
            col=1,
        )

    # AI
    if has_ai := (ai is not None and ai_names is not None and len(ai_names) > 0):
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
                    hovertemplate=f"<b>{name}</b><br>Time: %{x:.2f} ms<br>Voltage: %{y:.3f} V<extra></extra>",
                ),
                row=3,
                col=1,
            )

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
            groupclick="toggleitem",
            tracegroupgap=8,
        ),
        margin=dict(l=60, r=220, t=80, b=40),
    )
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

    # default collapse some busy groups if present
    fig.update_traces(
        visible="legendonly", selector=dict(legendgroup="Switch Valve Left")
    )
    fig.update_traces(
        visible="legendonly", selector=dict(legendgroup="Switch Valve Right")
    )

    return fig
