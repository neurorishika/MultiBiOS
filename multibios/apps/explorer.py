#!/usr/bin/env python3
"""MultiBiOS interactive experiment explorer.

Run with:

    python -m multibios.apps.explorer
    python -m multibios.apps.explorer --runs data/runs --port 8050

Then open http://localhost:8050 in a browser.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
from pathlib import Path
from typing import Optional

import dash
import dash.exceptions
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Input, Output, State, callback_context, dash_table, dcc, html
from plotly.subplots import make_subplots

# ─────────────────────────────────────────────────────────────────────────────
#  COLOUR SCHEME
# ─────────────────────────────────────────────────────────────────────────────
ODOR_COLORS = {
    "OFF":    "#555555",
    "AIR":    "#4e91d4",
    "ODOR1":  "#e74c3c",
    "ODOR2":  "#9b59b6",
    "ODOR3":  "#f39c12",
    "ODOR4":  "#2ecc71",
    "ODOR5":  "#1abc9c",
    "FLUSH":  "#e67e22",
    "CLEAN":  "#aaaaaa",
    "ODOR":   "#e74c3c",  # switch-valve ODOR state
}

PHASE_COLORS = [
    "#3498db", "#e74c3c", "#2ecc71", "#9b59b6",
    "#f39c12", "#1abc9c", "#e67e22", "#e91e63",
]

BG = "#111827"
CARD = "#1f2937"
BORDER = "#374151"
TEXT = "#f9fafb"
SUBTEXT = "#9ca3af"

PLOTLY_TEMPLATE = "plotly_dark"

# Max points to render in the timeline / track figures; data is stride-sampled
# down to this many points so the browser stays interactive.
TIMELINE_MAX_PTS = 4_000
TRACK_MAX_PTS    = 8_000

# ─────────────────────────────────────────────────────────────────────────────
#  DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

def find_runs(runs_dir: Path) -> list[dict]:
    """Return list of dicts {label, value=full_path_str} sorted newest first."""
    if not runs_dir.exists():
        return []
    entries = []
    for d in sorted(runs_dir.iterdir(), reverse=True):
        if not d.is_dir():
            continue
        has_data = (d / "experiment_data.csv").exists()
        has_events = (d / "event_log.csv").exists()
        if not (has_data or has_events):
            continue
        label = d.name
        if has_data:
            try:
                df = pd.read_csv(d / "experiment_data.csv", nrows=1)
                n = sum(1 for _ in open(d / "experiment_data.csv")) - 1
                label = f"{d.name}  ({n:,} frames)"
            except Exception:
                pass
        # Read protocol name from protocol.yaml if present
        proto_path = d / "protocol.yaml"
        if proto_path.exists():
            try:
                import yaml
                with open(proto_path, encoding="utf-8") as f:
                    p = yaml.safe_load(f)
                pname = p.get("protocol", {}).get("name", "")
                if pname:
                    label = f"{d.name}  [{pname}]"
                    if has_data:
                        try:
                            n = sum(1 for _ in open(d / "experiment_data.csv")) - 1
                            label = f"{d.name}  [{pname}]  ({n:,} frames)"
                        except Exception:
                            pass
            except Exception:
                pass
        entries.append({"label": label, "value": str(d)})
    return entries


def load_run(run_dir: str) -> dict:
    """Load all data for a run directory. Returns a dict of DataFrames + meta."""
    p = Path(run_dir)
    result: dict = {"path": str(p), "name": p.name}

    # experiment_data.csv
    exp_path = p / "experiment_data.csv"
    if exp_path.exists():
        df = pd.read_csv(exp_path)
        result["df"] = df
    else:
        result["df"] = pd.DataFrame()

    # event_log.csv
    ev_path = p / "event_log.csv"
    if ev_path.exists():
        result["events"] = pd.read_csv(ev_path)
    else:
        result["events"] = pd.DataFrame()

    # timeline.csv
    tl_path = p / "timeline.csv"
    if tl_path.exists():
        result["timeline"] = pd.read_csv(tl_path)
    else:
        result["timeline"] = pd.DataFrame()

    # meta.json
    meta_path = p / "meta.json"
    if meta_path.exists():
        with open(meta_path, encoding="utf-8") as f:
            result["meta"] = json.load(f)
    else:
        result["meta"] = {}

    # protocol.yaml
    proto_path = p / "protocol.yaml"
    if proto_path.exists():
        with open(proto_path, encoding="utf-8") as f:
            result["protocol_text"] = f.read()
    else:
        result["protocol_text"] = ""

    return result


# ─────────────────────────────────────────────────────────────────────────────
#  FIGURE HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _add_valve_bands(fig, df: pd.DataFrame, row: int, col: int = 1,
                     total_rows: int = 1) -> None:
    """Add coloured background rectangles where switch valves are ODOR."""
    if df.empty or "switch_valve_left" not in df.columns:
        return
    t = df["experiment_time_s"].values

    for side, col_name, color, opacity in [
        ("left",  "switch_valve_left",  "#e74c3c", 0.18),
        ("right", "switch_valve_right", "#f39c12", 0.14),
    ]:
        if col_name not in df.columns:
            continue
        in_odor = df[col_name].fillna("CLEAN") == "ODOR"
        # Find contiguous blocks
        changes = np.diff(in_odor.values.astype(int), prepend=0, append=0)
        starts = np.where(changes == 1)[0]
        ends   = np.where(changes == -1)[0]
        for s, e in zip(starts, ends):
            t0 = float(t[s]) if s < len(t) else float(t[-1])
            t1 = float(t[min(e, len(t)-1)])
            fig.add_vrect(
                x0=t0, x1=t1,
                fillcolor=color, opacity=opacity, line_width=0,
                row=row, col=col,
            )


def _add_microscope_lines(fig, df: pd.DataFrame, row, col: int = 1) -> None:
    """Add vertical dashed lines at microscope trigger pulses."""
    if df.empty or "microscope_trigger" not in df.columns:
        return
    t = df["experiment_time_s"].values
    micro = df["microscope_trigger"].fillna(0).values
    # Rising edges only
    edges = np.where(np.diff(micro.astype(int), prepend=0) > 0)[0]
    for idx in edges:
        fig.add_vline(
            x=float(t[idx]),
            line=dict(color="#f1c40f", width=1, dash="dot"),
            row=row, col=col,
        )


def _add_phase_dividers(fig, df: pd.DataFrame, row, col: int = 1) -> None:
    """Add vertical lines and labels at phase transitions."""
    if df.empty or "phase" not in df.columns:
        return
    df2 = df.dropna(subset=["phase"])
    if df2.empty:
        return
    phases = df2["phase"].values
    times  = df2["experiment_time_s"].values
    prev = ""
    for i, (ph, t) in enumerate(zip(phases, times)):
        if ph != prev and ph:
            if prev:  # Not the first
                fig.add_vline(
                    x=float(t),
                    line=dict(color=BORDER, width=1, dash="dash"),
                    row=row, col=col,
                )
            prev = ph


def _olf_color_trace(df: pd.DataFrame) -> go.Scatter:
    """Thin coloured line showing olfactometer left state (y=1 stripe)."""
    if df.empty or "olfactometer_left" not in df.columns:
        return go.Scatter()
    t = df["experiment_time_s"].values
    colors = [ODOR_COLORS.get(str(s).upper(), "#888") for s in df["olfactometer_left"].fillna("OFF")]
    # Emit one point per unique segment
    return go.Scatter(
        x=t, y=np.ones(len(t)),
        mode="markers",
        marker=dict(color=colors, size=4, symbol="square"),
        showlegend=False, hoverinfo="skip",
    )


# ─────────────────────────────────────────────────────────────────────────────
#  TAB FIGURES
# ─────────────────────────────────────────────────────────────────────────────

def _stride_sample(df: pd.DataFrame, max_pts: int) -> pd.DataFrame:
    """Return a stride-sampled copy of *df* with at most *max_pts* rows."""
    n = len(df)
    if n <= max_pts:
        return df
    stride = int(np.ceil(n / max_pts))
    return df.iloc[::stride].reset_index(drop=True)


def _state_transitions(df: pd.DataFrame, col: str, fill: str) -> pd.DataFrame:
    """Return only the rows where *col* changes value (plus the first row).
    Used to minimise the number of coloured-marker points in the state stripe."""
    s = df[col].fillna(fill)
    mask = np.concatenate(([True], s.values[1:] != s.values[:-1]))
    return df[mask]


def build_timeline_fig(run: dict) -> go.Figure:
    df = run.get("df", pd.DataFrame())
    events = run.get("events", pd.DataFrame())

    n_raw = len(df)
    df_s  = _stride_sample(df, TIMELINE_MAX_PTS)
    n_show = len(df_s)
    subsample_note = (
        f"Showing {n_show:,} / {n_raw:,} frames "
        f"(1 in {int(np.ceil(n_raw / TIMELINE_MAX_PTS))}×)"
        if n_raw > TIMELINE_MAX_PTS else f"{n_raw:,} frames (full resolution)"
    )

    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        row_heights=[0.08, 0.45, 0.47],
        vertical_spacing=0.02,
        subplot_titles=["Odor / Valve States", "Heading (°)", "Speed"],
    )
    fig.update_layout(
        template=PLOTLY_TEMPLATE, paper_bgcolor=BG, plot_bgcolor=CARD,
        margin=dict(l=60, r=20, t=40, b=40),
        legend=dict(bgcolor=CARD, bordercolor=BORDER, borderwidth=1,
                    font=dict(color=TEXT)),
        height=620,
        hovermode="x unified",
    )

    if df.empty:
        fig.add_annotation(text="No FicTrac data in this run",
                           xref="paper", yref="paper", x=0.5, y=0.5,
                           font=dict(size=16, color=SUBTEXT), showarrow=False)
        return fig

    t   = df["experiment_time_s"].values
    t_s = df_s["experiment_time_s"].values

    # Subsampling note in bottom-right corner
    fig.add_annotation(
        text=subsample_note, xref="paper", yref="paper",
        x=1.0, y=0.0, xanchor="right", yanchor="bottom",
        font=dict(size=9, color=SUBTEXT), showarrow=False,
    )

    # ── Row 1: State colour stripe (transition points only) ───────────────────
    for side, col_name, y_base, label in [
        ("OLF L",  "olfactometer_left",  1.5, "OLF L"),
        ("OLF R",  "olfactometer_right", 1.0, "OLF R"),
        ("SV L",   "switch_valve_left",  0.5, "SV  L"),
        ("SV R",   "switch_valve_right", 0.0, "SV  R"),
    ]:
        if col_name not in df.columns:
            continue
        fill = "OFF" if "olf" in col_name else "CLEAN"
        df_tr = _state_transitions(df, col_name, fill)
        vals   = df_tr[col_name].fillna(fill).values
        t_tr   = df_tr["experiment_time_s"].values
        colors = [ODOR_COLORS.get(str(v).upper(), "#888888") for v in vals]
        fig.add_trace(go.Scatter(
            x=t_tr, y=np.full(len(t_tr), y_base),
            mode="markers",
            marker=dict(color=colors, size=5, symbol="square"),
            name=label, showlegend=False,
            customdata=vals,
            hovertemplate=f"<b>{label}</b>: %{{customdata}}<extra></extra>",
        ), row=1, col=1)

    # Microscope lines on row 1
    _add_microscope_lines(fig, df, row=1)

    fig.update_yaxes(row=1, visible=False, range=[-0.3, 2.2])

    # Row 1 state legend annotations
    for label, yv in [("OLF L", 1.5), ("OLF R", 1.0), ("SV L", 0.5), ("SV R", 0.0)]:
        fig.add_annotation(
            x=0, y=yv, text=label, xref="x", yref="y",
            xanchor="right", font=dict(size=10, color=SUBTEXT),
            showarrow=False, row=1, col=1,
        )

    # ── Row 2: Heading (subsampled) ───────────────────────────────────────────
    _add_valve_bands(fig, df, row=2)
    heading_deg = np.degrees(df_s["heading"].values) % 360
    fig.add_trace(go.Scattergl(
        x=t_s, y=heading_deg,
        mode="lines",
        line=dict(color="#60a5fa", width=0.8),
        name="Heading (°)",
        hovertemplate="t=%{x:.2f}s  heading=%{y:.1f}°<extra></extra>",
    ), row=2, col=1)
    fig.update_yaxes(row=2, title_text="°", range=[0, 360],
                     tickvals=[0, 90, 180, 270, 360],
                     gridcolor=BORDER, title_font=dict(color=SUBTEXT))
    _add_microscope_lines(fig, df, row=2)

    # ── Row 3: Speed (subsampled) ─────────────────────────────────────────────
    _add_valve_bands(fig, df, row=3)
    fig.add_trace(go.Scattergl(
        x=t_s, y=df_s["speed"].values,
        mode="lines",
        line=dict(color="#34d399", width=0.8),
        name="Speed",
        hovertemplate="t=%{x:.2f}s  speed=%{y:.4f}<extra></extra>",
    ), row=3, col=1)
    fig.update_yaxes(row=3, title_text="speed", gridcolor=BORDER,
                     title_font=dict(color=SUBTEXT))
    _add_microscope_lines(fig, df, row=3)
    _add_phase_dividers(fig, df, row=3)

    fig.update_xaxes(gridcolor=BORDER, title_text="Experiment time (s)",
                     title_font=dict(color=SUBTEXT))

    # Colour legend for odor states
    for state, color in ODOR_COLORS.items():
        if state in ("CLEAN",):
            continue
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="markers",
            marker=dict(color=color, size=10, symbol="square"),
            name=state, showlegend=True,
        ), row=3, col=1)

    # Valve-band legend placeholders
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode="markers",
        marker=dict(color="#e74c3c", size=12, symbol="square", opacity=0.4),
        name="SV Left = ODOR",
    ), row=3, col=1)
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode="markers",
        marker=dict(color="#f39c12", size=12, symbol="square", opacity=0.4),
        name="SV Right = ODOR",
    ), row=3, col=1)
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode="lines",
        line=dict(color="#f1c40f", width=1.5, dash="dot"),
        name="Microscope trigger",
    ), row=3, col=1)

    return fig


def build_trial_fig(run: dict, align_to: str = "sv_open") -> go.Figure:
    """Per-trial aligned overlay: speed + heading, aligned to SV ODOR open."""
    df = run.get("df", pd.DataFrame())

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["Speed (trial-aligned)", "Heading Distribution (°)"],
        column_widths=[0.65, 0.35],
    )
    fig.update_layout(
        template=PLOTLY_TEMPLATE, paper_bgcolor=BG, plot_bgcolor=CARD,
        margin=dict(l=60, r=20, t=50, b=40), height=480,
        hovermode="x unified",
    )

    if df.empty or "switch_valve_left" not in df.columns:
        fig.add_annotation(text="No FicTrac data available",
                           xref="paper", yref="paper", x=0.5, y=0.5,
                           font=dict(size=14, color=SUBTEXT), showarrow=False)
        return fig

    # Find all "ODOR on" windows
    sv_combined = (
        (df["switch_valve_left"].fillna("CLEAN") == "ODOR") |
        (df["switch_valve_right"].fillna("CLEAN") == "ODOR")
    )
    changes = np.diff(sv_combined.values.astype(int), prepend=0, append=0)
    starts = np.where(changes == 1)[0]
    ends   = np.where(changes == -1)[0]

    if len(starts) == 0:
        fig.add_annotation(text="No odor delivery events found",
                           xref="paper", yref="paper", x=0.5, y=0.5,
                           font=dict(size=14, color=SUBTEXT), showarrow=False)
        return fig

    t = df["experiment_time_s"].values
    pre_s  = 5.0   # seconds before odor onset
    post_s = 40.0  # seconds after odor onset

    all_headings_odor = []
    all_headings_air  = []

    for trial_i, (s_idx, e_idx) in enumerate(zip(starts, ends)):
        t0 = float(t[s_idx])
        # Window: pre_s before to post_s after
        mask = (t >= t0 - pre_s) & (t <= t0 + post_s)
        seg = df[mask].copy()
        if len(seg) < 5:
            continue
        t_rel = seg["experiment_time_s"].values - t0
        speed  = seg["speed"].values
        heading = np.degrees(seg["heading"].values) % 360

        color = PHASE_COLORS[trial_i % len(PHASE_COLORS)]
        phase_label = str(seg["phase"].dropna().values[0]) if not seg["phase"].dropna().empty else f"Trial {trial_i+1}"

        fig.add_trace(go.Scatter(
            x=t_rel, y=speed, mode="lines",
            line=dict(color=color, width=1.5),
            name=phase_label,
            hovertemplate=f"Trial {trial_i+1}<br>t=%{{x:.1f}}s speed=%{{y:.4f}}<extra></extra>",
        ), row=1, col=1)

        # Collect headings during odor vs before
        odor_mask = (t_rel >= 0) & (t_rel <= float(t[e_idx]) - t0)
        air_mask  = (t_rel < 0)
        all_headings_odor.extend(heading[odor_mask].tolist())
        all_headings_air.extend(heading[air_mask].tolist())

    # Onset line
    fig.add_vline(x=0.0, line=dict(color="#f1c40f", width=1.5, dash="dash"), row=1, col=1)
    fig.add_annotation(x=0, y=0, xref="x", yref="paper",
                       text="odor ON", textangle=-90,
                       font=dict(color="#f1c40f", size=10), showarrow=False, yanchor="bottom",
                       row=1, col=1)
    fig.update_xaxes(row=1, col=1, title_text="Time from odor onset (s)", gridcolor=BORDER)
    fig.update_yaxes(row=1, col=1, title_text="Speed", gridcolor=BORDER)

    # Heading histogram (right panel)
    if all_headings_odor:
        bins = np.arange(0, 361, 10)
        h_odor, _ = np.histogram(all_headings_odor, bins=bins)
        h_air,  _ = np.histogram(all_headings_air,  bins=bins)
        bin_centers = (bins[:-1] + bins[1:]) / 2

        fig.add_trace(go.Bar(
            x=bin_centers, y=h_odor,
            marker_color="#e74c3c", opacity=0.7, name="During odor",
        ), row=1, col=2)
        fig.add_trace(go.Bar(
            x=bin_centers, y=h_air,
            marker_color="#4e91d4", opacity=0.7, name="Pre-odor",
        ), row=1, col=2)
        fig.update_xaxes(row=1, col=2, title_text="Heading (°)",
                         tickvals=[0, 90, 180, 270, 360], gridcolor=BORDER)
        fig.update_yaxes(row=1, col=2, title_text="Count", gridcolor=BORDER)
        fig.update_layout(barmode="overlay")

    return fig


def build_track_fig(run: dict) -> go.Figure:
    df = run.get("df", pd.DataFrame())

    fig = go.Figure()
    fig.update_layout(
        template=PLOTLY_TEMPLATE, paper_bgcolor=BG, plot_bgcolor=CARD,
        margin=dict(l=60, r=20, t=40, b=40), height=560,
        xaxis=dict(title="Integrated X (ball radii)", scaleanchor="y",
                   scaleratio=1, gridcolor=BORDER),
        yaxis=dict(title="Integrated Y (ball radii)", gridcolor=BORDER),
    )

    if df.empty or "intx" not in df.columns:
        fig.add_annotation(text="No trajectory data",
                           xref="paper", yref="paper", x=0.5, y=0.5,
                           font=dict(size=14, color=SUBTEXT), showarrow=False)
        return fig

    # Stride-sample track for display
    df_t = _stride_sample(df, TRACK_MAX_PTS)
    x = df_t["intx"].values
    y = df_t["inty"].values

    # Colour by valve state — vectorised
    lv = df_t["switch_valve_left"].fillna("CLEAN").str.upper()  if "switch_valve_left"  in df_t.columns else pd.Series("CLEAN", index=df_t.index)
    rv = df_t["switch_valve_right"].fillna("CLEAN").str.upper() if "switch_valve_right" in df_t.columns else pd.Series("CLEAN", index=df_t.index)
    sv_state = np.where(
        (lv == "ODOR") & (rv == "ODOR"), "Both ODOR",
        np.where(lv == "ODOR", "Left ODOR",
        np.where(rv == "ODOR", "Right ODOR", "Clean Air"))
    )

    color_map = {
        "Clean Air":  "#4e91d4",
        "Left ODOR":  "#e74c3c",
        "Right ODOR": "#f39c12",
        "Both ODOR":  "#9b59b6",
    }

    for state, color in color_map.items():
        mask = sv_state == state
        if not mask.any():
            continue
        fig.add_trace(go.Scattergl(
            x=x[mask], y=y[mask], mode="markers",
            marker=dict(color=color, size=2, opacity=0.6),
            name=state,
            hovertemplate=f"<b>{state}</b><br>x=%{{x:.3f}}<br>y=%{{y:.3f}}<extra></extra>",
        ))

    # Mark start (always use full-res first point)
    x0 = df["intx"].values[0]
    y0 = df["inty"].values[0]
    fig.add_trace(go.Scatter(
        x=[x0], y=[y0], mode="markers",
        marker=dict(color="white", size=12, symbol="circle", line=dict(color=BG, width=2)),
        name="Start",
    ))

    return fig


def build_diagnostics_fig(run: dict) -> go.Figure:
    events = run.get("events", pd.DataFrame())

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["Command jitter over time (ms)", "Jitter distribution by type (ms)"],
    )
    fig.update_layout(
        template=PLOTLY_TEMPLATE, paper_bgcolor=BG, plot_bgcolor=CARD,
        margin=dict(l=60, r=20, t=50, b=40), height=420, hovermode="x unified",
    )

    if events.empty or "jitter_ms" not in events.columns:
        fig.add_annotation(text="No event log data",
                           xref="paper", yref="paper", x=0.5, y=0.5,
                           font=dict(size=14, color=SUBTEXT), showarrow=False)
        return fig

    ev = events.dropna(subset=["jitter_ms"]).copy()
    ev["jitter_ms"] = pd.to_numeric(ev["jitter_ms"], errors="coerce")
    ev = ev.dropna(subset=["jitter_ms"])
    if ev.empty:
        return fig

    types = ev["type"].unique() if "type" in ev.columns else []
    color_cycle = px.colors.qualitative.Plotly

    for i, t in enumerate(types):
        sub = ev[ev["type"] == t]
        color = color_cycle[i % len(color_cycle)]
        fig.add_trace(go.Scatter(
            x=sub["experiment_time_s"] if "experiment_time_s" in sub else sub.index,
            y=sub["jitter_ms"],
            mode="markers", name=t,
            marker=dict(color=color, size=5, opacity=0.7),
            hovertemplate=f"<b>{t}</b><br>t=%{{x:.2f}}s<br>jitter=%{{y:.2f}} ms<extra></extra>",
        ), row=1, col=1)

        fig.add_trace(go.Box(
            y=sub["jitter_ms"], name=t,
            marker_color=color, showlegend=False,
            boxpoints="outliers",
        ), row=1, col=2)

    fig.update_xaxes(row=1, col=1, title_text="Experiment time (s)", gridcolor=BORDER)
    fig.update_yaxes(row=1, col=1, title_text="Jitter (ms)", gridcolor=BORDER)
    fig.update_yaxes(row=1, col=2, title_text="Jitter (ms)", gridcolor=BORDER)
    fig.add_hline(y=1.0, line=dict(color="yellow", dash="dot", width=1),
                  annotation_text="1 ms", annotation_font_color="yellow",
                  row=1, col=1)

    return fig


def build_stats_table(run: dict) -> list:
    df = run.get("df", pd.DataFrame())
    if df.empty or "phase" not in df.columns:
        return []

    rows = []
    sr = 1 / df["delta_timestamp"].median() * 1000 if "delta_timestamp" in df.columns else None

    for phase, grp in df.groupby("phase", sort=False):
        if not str(phase).strip():
            continue
        duration = grp["experiment_time_s"].max() - grp["experiment_time_s"].min()
        rows.append({
            "Phase": str(phase),
            "Frames": len(grp),
            "Duration (s)": f"{duration:.2f}",
            "Mean speed": f"{grp['speed'].mean():.4f}" if "speed" in grp else "—",
            "Max speed":  f"{grp['speed'].max():.4f}"  if "speed" in grp else "—",
            "Mean heading (°)": f"{np.degrees(grp['heading'].mean()) % 360:.1f}" if "heading" in grp else "—",
            "SV Left ODOR %":  f"{(grp['switch_valve_left'].fillna('CLEAN')=='ODOR').mean()*100:.1f}%" if "switch_valve_left" in grp else "—",
            "SV Right ODOR %": f"{(grp['switch_valve_right'].fillna('CLEAN')=='ODOR').mean()*100:.1f}%" if "switch_valve_right" in grp else "—",
        })

    return rows


# ─────────────────────────────────────────────────────────────────────────────
#  LAYOUT
# ─────────────────────────────────────────────────────────────────────────────

def make_info_card(label: str, value: str) -> html.Div:
    return html.Div([
        html.Div(label, style={"fontSize": "10px", "color": SUBTEXT, "textTransform": "uppercase",
                                "letterSpacing": "0.05em", "marginBottom": "2px"}),
        html.Div(value, style={"fontSize": "14px", "color": TEXT, "fontWeight": "600"}),
    ], style={"padding": "8px 10px", "background": BG, "borderRadius": "6px",
              "border": f"1px solid {BORDER}", "marginBottom": "8px"})


def build_layout(runs_dir: Path) -> html.Div:
    run_options = find_runs(runs_dir)
    default_run = run_options[0]["value"] if run_options else None

    return html.Div([
        dcc.Store(id="run-data"),
        dcc.Store(id="known-runs", data=[o["value"] for o in run_options]),
        dcc.Location(id="url", refresh=False),
        dcc.Interval(id="poll-interval", interval=5_000, n_intervals=0),  # 5-second refresh

        # ── HEADER ──────────────────────────────────────────────────────────
        html.Div([
            html.Div([
                html.Span("MultiBiOS", style={"color": "#60a5fa", "fontWeight": "800",
                                               "fontSize": "18px"}),
                html.Span("  Experiment Explorer", style={"color": TEXT, "fontSize": "16px",
                                                            "fontWeight": "300"}),
            ]),
            html.Div(id="header-status",
                     style={"color": SUBTEXT, "fontSize": "12px"}),
        ], style={
            "display": "flex", "justifyContent": "space-between", "alignItems": "center",
            "padding": "12px 24px", "background": CARD,
            "borderBottom": f"1px solid {BORDER}",
            "fontFamily": "'Segoe UI', system-ui, sans-serif",
        }),

        # ── BODY ────────────────────────────────────────────────────────────
        html.Div([

            # ── LEFT SIDEBAR ─────────────────────────────────────────────
            html.Div([
                html.Div("Run", style={"color": SUBTEXT, "fontSize": "11px",
                                        "textTransform": "uppercase", "letterSpacing": "0.06em",
                                        "marginBottom": "6px"}),
                dcc.Dropdown(
                    id="run-selector",
                    options=run_options,
                    value=default_run,
                    clearable=False,
                    style={"marginBottom": "16px"},
                ),
                html.Div(id="meta-panel"),

                html.Hr(style={"borderColor": BORDER, "margin": "16px 0"}),

                html.Div("Options", style={"color": SUBTEXT, "fontSize": "11px",
                                            "textTransform": "uppercase", "letterSpacing": "0.06em",
                                            "marginBottom": "6px"}),
                dcc.Checklist(
                    id="show-microscope",
                    options=[{"label": " Show microscope lines", "value": "micro"}],
                    value=["micro"],
                    style={"color": TEXT, "fontSize": "13px"},
                ),
                html.Br(),
                html.Div("Align trials to:", style={"color": SUBTEXT, "fontSize": "12px",
                                                     "marginBottom": "4px"}),
                dcc.RadioItems(
                    id="align-to",
                    options=[
                        {"label": " Valve open (SV=ODOR)", "value": "sv_open"},
                        {"label": " Microscope trigger",  "value": "micro_trigger"},
                    ],
                    value="sv_open",
                    style={"color": TEXT, "fontSize": "12px", "lineHeight": "2"},
                ),
            ], style={
                "width": "240px", "minWidth": "240px",
                "padding": "16px",
                "background": CARD,
                "borderRight": f"1px solid {BORDER}",
                "height": "calc(100vh - 49px)",
                "overflowY": "auto",
                "fontFamily": "'Segoe UI', system-ui, sans-serif",
            }),

            # ── MAIN CONTENT ─────────────────────────────────────────────
            html.Div([
                dcc.Tabs(
                    id="tabs",
                    value="timeline",
                    children=[
                        dcc.Tab(label="Timeline",      value="timeline"),
                        dcc.Tab(label="Odor Responses", value="trials"),
                        dcc.Tab(label="Track",          value="track"),
                        dcc.Tab(label="Diagnostics",    value="diagnostics"),
                        dcc.Tab(label="Stats",          value="stats"),
                        dcc.Tab(label="Protocol",       value="protocol"),
                    ],
                    style={"fontFamily": "'Segoe UI', sans-serif"},
                    colors={"border": BORDER, "primary": "#60a5fa",
                            "background": CARD},
                ),
                html.Div(id="tab-content", style={"padding": "12px"}),
            ], style={
                "flex": "1",
                "overflowY": "auto",
                "height": "calc(100vh - 49px)",
                "background": BG,
            }),

        ], style={"display": "flex", "flex": "1"}),

    ], style={
        "background": BG, "color": TEXT,
        "fontFamily": "'Segoe UI', system-ui, sans-serif",
        "height": "100vh", "display": "flex", "flexDirection": "column",
    })


# ─────────────────────────────────────────────────────────────────────────────
#  APP + CALLBACKS
# ─────────────────────────────────────────────────────────────────────────────

def create_app(runs_dir: Path) -> dash.Dash:
    app = dash.Dash(
        __name__,
        title="MultiBiOS Explorer",
        suppress_callback_exceptions=True,
    )
    app.layout = build_layout(runs_dir)
    _runs_dir = runs_dir  # capture for closure

    # Inject dark styles for Dropdown and Tabs
    app.index_string = '''<!DOCTYPE html>
<html>
<head>
{%metas%}
<title>{%title%}</title>
{%favicon%}
{%css%}
<style>
  * { box-sizing: border-box; }
  body { margin: 0; background: ''' + BG + '''; }
  .Select-control, .Select-menu-outer { background-color: #1f2937 !important; color: #f9fafb !important; border-color: #374151 !important; }
  .Select-value-label, .Select-option, .VirtualizedSelectOption { color: #f9fafb !important; }
  .Select-option.is-focused, .VirtualizedSelectFocusedOption { background-color: #374151 !important; }
  .Select--single > .Select-control .Select-value { color: #f9fafb !important; }
  .dash-dropdown .Select-arrow { border-top-color: #9ca3af !important; }
  .tab--selected { font-weight: 600 !important; }
  ::-webkit-scrollbar { width: 6px; height: 6px; }
  ::-webkit-scrollbar-track { background: #1f2937; }
  ::-webkit-scrollbar-thumb { background: #374151; border-radius: 3px; }
  .dash-table-container .dash-spreadsheet-container .dash-spreadsheet-inner td,
  .dash-table-container .dash-spreadsheet-container .dash-spreadsheet-inner th { background-color: #1f2937 !important; color: #f9fafb !important; border-color: #374151 !important; }
  .dash-table-container { background: #1f2937; }
  .rc-slider-track { background-color: #60a5fa !important; }
</style>
</head>
<body>{%app_entry%}<footer>{%config%}{%scripts%}{%renderer%}</footer></body>
</html>'''

    # ── Callback: auto-refresh run list every 5 s, jump to newest when a new run appears ──
    @app.callback(
        Output("run-selector", "options"),
        Output("run-selector", "value"),
        Output("known-runs", "data"),
        Input("poll-interval", "n_intervals"),
        Input("url", "search"),
        State("run-selector", "value"),
        State("known-runs", "data"),
    )
    def refresh_runs(n, url_search, current_value, known_runs):
        fresh = find_runs(_runs_dir)
        new_values = [o["value"] for o in fresh]
        known_set  = set(known_runs or [])

        # URL param: ?run=<path>  (only on first load when n==0)
        if url_search and n == 0:
            from urllib.parse import parse_qs, urlparse
            try:
                params = parse_qs(url_search.lstrip("?"))
                url_run = params.get("run", [None])[0]
                if url_run and Path(url_run).exists():
                    return fresh, url_run, new_values
            except Exception:
                pass

        # New run appeared on disk -> jump to it automatically
        new_runs = [v for v in new_values if v not in known_set]
        if new_runs:
            newest = new_runs[0]  # find_runs returns newest-first
            return fresh, newest, new_values

        # Nothing new — keep current selection (or default to first)
        keep = current_value if current_value in new_values else (new_values[0] if new_values else None)
        return fresh, keep, new_values

    # ── Callback: load run data into Store ──────────────────────────────────
    @app.callback(
        Output("run-data", "data"),
        Output("meta-panel", "children"),
        Output("header-status", "children"),
        Input("run-selector", "value"),
    )
    def load_run_cb(run_path):
        if not run_path:
            return {}, [], ""
        run = load_run(run_path)
        df   = run.get("df",     pd.DataFrame())
        meta = run.get("meta",   {})
        ev   = run.get("events", pd.DataFrame())

        # Build meta panel
        proto_name = meta.get("protocol_name", Path(run_path).name)
        n_frames   = len(df)
        duration   = f"{df['experiment_time_s'].max():.1f} s" if not df.empty and "experiment_time_s" in df else "—"
        fps        = f"{1 / df['delta_timestamp'].median() * 1000:.1f} Hz" if not df.empty and "delta_timestamp" in df and df['delta_timestamp'].median() > 0 else "—"
        n_micro    = int((df["microscope_trigger"].fillna(0).diff().clip(0) > 0).sum()) if not df.empty and "microscope_trigger" in df else "—"
        n_sv_odor  = int((df.get("switch_valve_left", pd.Series(dtype=str)).fillna("CLEAN") == "ODOR").sum()) if not df.empty else 0

        # Get protocol name from YAML
        if run.get("protocol_text"):
            try:
                import yaml
                p = yaml.safe_load(run["protocol_text"])
                proto_name = p.get("protocol", {}).get("name", proto_name)
            except Exception:
                pass

        # Mean jitter
        mean_jitter = "—"
        if not ev.empty and "jitter_ms" in ev.columns:
            j = pd.to_numeric(ev["jitter_ms"], errors="coerce").dropna()
            if not j.empty:
                mean_jitter = f"{j.mean():.2f} ms"

        panel = [
            make_info_card("Protocol", proto_name),
            make_info_card("Run date", Path(run_path).name[:16].replace("_", " ")),
            make_info_card("FicTrac frames", f"{n_frames:,}"),
            make_info_card("Duration", duration),
            make_info_card("Frame rate", fps),
            make_info_card("Microscope pulses", str(n_micro)),
            make_info_card("Mean cmd jitter", mean_jitter),
        ]
        status = f"Loaded: {Path(run_path).name}  |  {n_frames:,} frames"

        # Serialize run data for Store (exclude large arrays)
        store = {
            "path": run_path,
            "name": run["name"],
            "protocol_text": run.get("protocol_text", ""),
            "df_json":     df.to_json(orient="split") if not df.empty else None,
            "events_json": ev.to_json(orient="split") if not ev.empty else None,
            "timeline_json": run.get("timeline", pd.DataFrame()).to_json(orient="split") if not run.get("timeline", pd.DataFrame()).empty else None,
            "meta": meta,
        }
        return store, panel, status

    # ── Callback: render tab content ─────────────────────────────────────────
    @app.callback(
        Output("tab-content", "children"),
        Input("tabs", "value"),
        Input("run-data", "data"),
        Input("align-to", "value"),
    )
    def render_tab(tab, store, align_to):
        if not store:
            return html.Div("Select a run to get started.",
                            style={"color": SUBTEXT, "padding": "40px",
                                   "fontSize": "15px", "textAlign": "center"})

        # Reconstruct DataFrames from store
        run = dict(store)
        run["df"]       = pd.read_json(io.StringIO(store["df_json"]),       orient="split") if store.get("df_json")       else pd.DataFrame()
        run["events"]   = pd.read_json(io.StringIO(store["events_json"]),   orient="split") if store.get("events_json")   else pd.DataFrame()
        run["timeline"] = pd.read_json(io.StringIO(store["timeline_json"]), orient="split") if store.get("timeline_json") else pd.DataFrame()

        if tab == "timeline":
            return dcc.Graph(
                figure=build_timeline_fig(run),
                config={"displayModeBar": True, "scrollZoom": True,
                        "toImageButtonOptions": {"format": "png", "width": 1600, "height": 900}},
                style={"height": "640px"},
            )

        elif tab == "trials":
            return dcc.Graph(
                figure=build_trial_fig(run, align_to=align_to),
                config={"displayModeBar": True, "scrollZoom": True,
                        "toImageButtonOptions": {"format": "png", "width": 1400, "height": 700}},
                style={"height": "500px"},
            )

        elif tab == "track":
            return dcc.Graph(
                figure=build_track_fig(run),
                config={"displayModeBar": True, "scrollZoom": True,
                        "toImageButtonOptions": {"format": "png", "width": 1200, "height": 1200}},
                style={"height": "580px"},
            )

        elif tab == "diagnostics":
            df = run["df"]
            events = run["events"]

            # Summary stats
            n_frames   = len(df)
            total_cmds = len(events.dropna(subset=["jitter_ms"])) if not events.empty else 0
            max_jitter = "—"
            pct99      = "—"
            if not events.empty and "jitter_ms" in events.columns:
                j = pd.to_numeric(events["jitter_ms"], errors="coerce").dropna()
                if not j.empty:
                    max_jitter = f"{j.max():.2f} ms"
                    pct99      = f"{np.percentile(j, 99):.2f} ms"

            stat_cards = html.Div([
                make_info_card("Total commands",  str(total_cmds)),
                make_info_card("Max jitter",      max_jitter),
                make_info_card("99th pct jitter", pct99),
            ], style={"display": "flex", "gap": "10px", "marginBottom": "12px"})

            return html.Div([
                stat_cards,
                dcc.Graph(
                    figure=build_diagnostics_fig(run),
                    config={"displayModeBar": True},
                    style={"height": "440px"},
                ),
            ])

        elif tab == "stats":
            rows = build_stats_table(run)
            if not rows:
                return html.Div("No phase data available.",
                                style={"color": SUBTEXT, "padding": "24px"})
            return dash_table.DataTable(
                data=rows,
                columns=[{"name": k, "id": k} for k in rows[0].keys()],
                style_header={
                    "backgroundColor": CARD, "color": SUBTEXT,
                    "fontWeight": "600", "fontSize": "12px",
                    "borderColor": BORDER, "textTransform": "uppercase",
                },
                style_cell={
                    "backgroundColor": BG, "color": TEXT,
                    "fontFamily": "'Segoe UI', monospace",
                    "fontSize": "13px", "padding": "8px 12px",
                    "borderColor": BORDER,
                    "whiteSpace": "normal",
                },
                style_data_conditional=[
                    {"if": {"row_index": "odd"},
                     "backgroundColor": CARD},
                ],
                style_table={"overflowX": "auto", "borderRadius": "8px",
                             "border": f"1px solid {BORDER}"},
                sort_action="native",
                filter_action="native",
                page_action="none",
            )

        elif tab == "protocol":
            proto_text = store.get("protocol_text", "")
            if not proto_text:
                return html.Div("No protocol.yaml in this run directory.",
                                style={"color": SUBTEXT, "padding": "24px"})
            return html.Div([
                html.Pre(
                    proto_text,
                    style={
                        "background": CARD,
                        "color": "#a5f3fc",
                        "padding": "20px",
                        "borderRadius": "8px",
                        "border": f"1px solid {BORDER}",
                        "fontSize": "12px",
                        "overflowX": "auto",
                        "fontFamily": "'Cascadia Code', 'Consolas', monospace",
                        "whiteSpace": "pre",
                        "lineHeight": "1.6",
                    },
                )
            ])

        return html.Div("Unknown tab", style={"color": SUBTEXT})

    return app


# ─────────────────────────────────────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="MultiBiOS Experiment Explorer")
    parser.add_argument("--runs", default="data/runs",
                        help="Path to runs directory (default: data/runs)")
    parser.add_argument("--port", type=int, default=8050)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    runs_dir = Path(args.runs)
    if not runs_dir.is_absolute():
        runs_dir = Path.cwd() / runs_dir

    if not runs_dir.exists():
        print(f"WARNING: Runs directory '{runs_dir}' not found. Create it or pass --runs <path>")

    app = create_app(runs_dir)

    url = f"http://{args.host}:{args.port}"
    print(f"\n{'='*60}")
    print(f"  MultiBiOS Experiment Explorer")
    print(f"  {'='*56}")
    print(f"  URL:      {url}")
    print(f"  Runs dir: {runs_dir}")
    n = len(find_runs(runs_dir))
    print(f"  Runs found: {n}")
    print(f"{'='*60}\n")

    if not args.no_browser:
        import threading
        import time
        import webbrowser
        def _open():
            time.sleep(1.5)
            webbrowser.open(url)
        threading.Thread(target=_open, daemon=True).start()

    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
