from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


GRAB_LINE_RE = re.compile(
    r"PGRSource::grab \[DBG\] Frame captured .*?t_sys: (?P<t_sys_ms>[0-9.]+) ms.*?Frame (?P<frame_idx>[0-9]+)?"
)


def _extract_grab_times_ms(log_path: Path) -> tuple[np.ndarray, np.ndarray]:
    frame_indices: list[int] = []
    times_ms: list[float] = []

    frame_idx = -1
    with log_path.open("r", encoding="utf-8", errors="replace") as handle:
        for raw_line in handle:
            if "PGRSource::grab [DBG] Frame captured" not in raw_line:
                continue

            match = re.search(r"t_sys: ([0-9.]+) ms", raw_line)
            if match is None:
                continue

            frame_idx += 1
            frame_indices.append(frame_idx)
            times_ms.append(float(match.group(1)))

    if len(times_ms) < 2:
        raise RuntimeError(f"Found only {len(times_ms)} captured-frame timestamps in {log_path}")

    return np.asarray(frame_indices, dtype=np.int64), np.asarray(times_ms, dtype=np.float64)


def _load_run_start_ms(meta_path: Path | None) -> float | None:
    if meta_path is None:
        return None
    payload = json.loads(meta_path.read_text(encoding="utf-8"))
    t0_perf_counter = payload.get("t0_perf_counter")
    if t0_perf_counter is None:
        return None
    return float(t0_perf_counter) * 1000.0


def _load_phase_spans_ms(control_plan_path: Path | None, total_duration_ms: float) -> list[tuple[float, float, str]]:
    if control_plan_path is None:
        return []

    rows: list[tuple[float, str]] = []
    with control_plan_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            phase = (row.get("phase") or "").strip()
            time_ms = row.get("time_ms")
            if not phase or not time_ms:
                continue
            rows.append((float(time_ms), phase))

    if not rows:
        return []

    rows.sort(key=lambda item: item[0])
    spans: list[tuple[float, float, str]] = []
    current_start, current_phase = rows[0]
    for next_start, next_phase in rows[1:]:
        if next_phase == current_phase:
            continue
        spans.append((current_start, next_start, current_phase))
        current_start, current_phase = next_start, next_phase
    spans.append((current_start, total_duration_ms, current_phase))
    return spans


def _build_figure(
    *,
    frame_indices: np.ndarray,
    times_ms: np.ndarray,
    output_path: Path,
    expected_ifi_ms: float | None,
    title: str,
    run_start_ms: float | None,
    phase_spans_ms: list[tuple[float, float, str]],
) -> dict[str, float | int]:
    ifi_ms = np.diff(times_ms)
    ifi_frame_indices = frame_indices[1:]
    frame_times_rel_ms = times_ms - times_ms[0]
    if run_start_ms is not None:
        frame_times_rel_ms = times_ms - run_start_ms
    expected_frame_times_ms = frame_times_rel_ms[0] + (frame_indices - frame_indices[0]) * (expected_ifi_ms or np.median(ifi_ms))
    cumulative_drift_ms = frame_times_rel_ms - expected_frame_times_ms

    tail_window = min(2000, ifi_ms.size)
    if tail_window <= 0:
        tail_window = ifi_ms.size

    mean_ifi = float(np.mean(ifi_ms))
    median_ifi = float(np.median(ifi_ms))
    max_ifi = float(np.max(ifi_ms))
    min_ifi = float(np.min(ifi_ms))
    worst_pos = int(np.argmax(ifi_ms))
    worst_frame = int(ifi_frame_indices[worst_pos])

    fig, axes = plt.subplots(4, 1, figsize=(14, 14), constrained_layout=True)

    axes[0].plot(ifi_frame_indices, ifi_ms, linewidth=0.7, color="#0B6E4F")
    axes[0].set_title(f"{title}: full-run interframe interval")
    axes[0].set_ylabel("IFI (ms)")
    axes[0].grid(alpha=0.25)

    axes[1].plot(frame_times_rel_ms / 1000.0, cumulative_drift_ms, linewidth=0.9, color="#7A3E9D")
    axes[1].set_title("Cumulative frame-time drift")
    axes[1].set_ylabel("Drift (ms)")
    axes[1].grid(alpha=0.25)

    if phase_spans_ms:
        phase_colors = ["#E9F2F9", "#FDF1E7", "#EEF7EE", "#F7ECF8", "#F5F5DD"]
        for idx, (start_ms, end_ms, phase) in enumerate(phase_spans_ms):
            color = phase_colors[idx % len(phase_colors)]
            axes[1].axvspan(start_ms / 1000.0, end_ms / 1000.0, color=color, alpha=0.35, linewidth=0)
            mid_s = (start_ms + end_ms) / 2000.0
            axes[1].text(mid_s, axes[1].get_ylim()[1] if np.isfinite(axes[1].get_ylim()[1]) else 0.0, phase, ha="center", va="bottom", fontsize=8, rotation=0)

    axes[2].plot(ifi_frame_indices[-tail_window:], ifi_ms[-tail_window:], linewidth=0.9, color="#C84C09")
    axes[2].set_title(f"Last {tail_window} interframe intervals")
    axes[2].set_ylabel("IFI (ms)")
    axes[2].grid(alpha=0.25)

    axes[3].hist(ifi_ms, bins=120, color="#3465A4", alpha=0.85)
    axes[3].set_title("IFI distribution")
    axes[3].set_xlabel("IFI (ms)")
    axes[3].set_ylabel("Count")
    axes[3].grid(alpha=0.2)

    for axis in (axes[0], axes[2]):
        if expected_ifi_ms is not None and math.isfinite(expected_ifi_ms):
            axis.axhline(expected_ifi_ms, color="#444444", linestyle="--", linewidth=1.0, label=f"expected {expected_ifi_ms:.3f} ms")
            axis.legend(loc="upper right")

    axes[0].annotate(
        f"worst IFI {max_ifi:.3f} ms at frame {worst_frame}",
        xy=(worst_frame, max_ifi),
        xytext=(10, 10),
        textcoords="offset points",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "alpha": 0.85},
    )

    summary = (
        f"samples={ifi_ms.size}  mean={mean_ifi:.4f} ms  median={median_ifi:.4f} ms  "
        f"min={min_ifi:.4f} ms  max={max_ifi:.4f} ms"
    )
    fig.text(0.01, 0.01, summary, fontsize=10)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)

    return {
        "interval_count": int(ifi_ms.size),
        "mean_ifi_ms": mean_ifi,
        "median_ifi_ms": median_ifi,
        "min_ifi_ms": min_ifi,
        "max_ifi_ms": max_ifi,
        "worst_frame": worst_frame,
        "final_drift_ms": float(cumulative_drift_ms[-1]),
        "max_abs_drift_ms": float(np.max(np.abs(cumulative_drift_ms))),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot FicTrac camera interframe intervals from native grab logs.")
    parser.add_argument("log_path", type=Path, help="Path to a FicTrac native log file")
    parser.add_argument("--output", type=Path, default=None, help="Output image path (defaults next to log)")
    parser.add_argument("--expected-fps", type=float, default=None, help="Expected trigger/camera FPS for reference line")
    parser.add_argument("--title", default=None, help="Optional plot title")
    parser.add_argument("--meta", type=Path, default=None, help="Optional meta.json path for run-start alignment")
    parser.add_argument("--control-plan", type=Path, default=None, help="Optional control_plan.csv path for phase shading")
    args = parser.parse_args()

    log_path = args.log_path.resolve()
    output_path = args.output.resolve() if args.output is not None else log_path.with_name(log_path.stem + "-ifi.png")
    title = args.title or log_path.stem
    expected_ifi_ms = None if args.expected_fps is None or args.expected_fps <= 0 else 1000.0 / args.expected_fps
    run_start_ms = _load_run_start_ms(args.meta.resolve() if args.meta is not None else None)

    frame_indices, times_ms = _extract_grab_times_ms(log_path)
    total_duration_ms = float(times_ms[-1] - (run_start_ms if run_start_ms is not None else times_ms[0]))
    phase_spans_ms = _load_phase_spans_ms(
        args.control_plan.resolve() if args.control_plan is not None else None,
        total_duration_ms=total_duration_ms,
    )
    summary = _build_figure(
        frame_indices=frame_indices,
        times_ms=times_ms,
        output_path=output_path,
        expected_ifi_ms=expected_ifi_ms,
        title=title,
        run_start_ms=run_start_ms,
        phase_spans_ms=phase_spans_ms,
    )

    print(f"saved_plot={output_path}")
    for key, value in summary.items():
        print(f"{key}={value}")


if __name__ == "__main__":
    main()