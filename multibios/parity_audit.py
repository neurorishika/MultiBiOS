from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _count_trigger_rising_edges(edge_csv: Path, *, line_name: str = "TRIG_CAMERA") -> int | None:
    if not edge_csv.exists():
        return None

    count = 0
    with open(edge_csv, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row.get("line") != line_name:
                continue
            if row.get("edge_type") == "rising":
                count += 1
    return count


def _first_existing_path(paths: list[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def _load_callback_frame_count(run_dir: Path) -> int | None:
    frames_path = _first_existing_path([
        run_dir / "recorded" / "tracking" / "fictrac" / "frame_series.npz",
        run_dir / "fictrac_frames.npz",
    ])
    if frames_path is None:
        return None
    with np.load(frames_path, allow_pickle=False) as npz:
        frame_key = "frames" if "frames" in npz.files else "data"
        frames = npz[frame_key]
        return int(len(frames))


def summarize_run_parity(run_dir: Path) -> dict[str, Any]:
    run_dir = run_dir.resolve()
    fictrac_session = _load_json(run_dir / "recorded" / "tracking" / "fictrac" / "session_record.json") or {}
    fictrac_recording = fictrac_session.get("recording_summary") or _load_json(run_dir / "fictrac_camera_recording.json") or {}
    fictrac_diagnostics = _load_json(run_dir / "logs" / "diagnostics" / "fictrac_driver_diagnostics.json") or _load_json(run_dir / "fictrac_driver_diagnostics.json") or {}
    blackfly_recording = _load_json(run_dir / "recorded" / "cameras" / "secondary_camera" / "recording_summary.json") or _load_json(run_dir / "blackfly_recording.json") or {}

    trigger_rising_edges = _count_trigger_rising_edges(_first_existing_path([
        run_dir / "planned" / "daq" / "digital_outputs" / "edge_table.csv",
        run_dir / "digital_edges.csv",
    ]) or run_dir / "digital_edges.csv")
    fictrac_saved_raw_frames = fictrac_recording.get("saved_raw_frames")
    fictrac_udp_frame_cnt = fictrac_diagnostics.get("frame_cnt")
    fictrac_callback_frames = _load_callback_frame_count(run_dir)
    second_camera_saved_frames = blackfly_recording.get("saved_frames")

    counts = {
        "trigger_rising_edges": trigger_rising_edges,
        "fictrac_saved_raw_frames": fictrac_saved_raw_frames,
        "fictrac_udp_frame_cnt": fictrac_udp_frame_cnt,
        "fictrac_callback_frames": fictrac_callback_frames,
        "second_camera_saved_frames": second_camera_saved_frames,
    }

    expected = trigger_rising_edges
    mismatches = {
        name: (None if expected is None or value is None else int(value) - int(expected))
        for name, value in counts.items()
        if name != "trigger_rising_edges"
    }
    exact_trigger_match = expected is not None and all(
        value is not None and int(value) == int(expected)
        for name, value in counts.items()
        if name != "trigger_rising_edges"
    )

    return {
        "run_dir": str(run_dir),
        "counts": counts,
        "mismatches_vs_trigger": mismatches,
        "exact_trigger_match": exact_trigger_match,
        "fictrac_expected_frames": fictrac_recording.get("expected_frames"),
        "second_camera_expected_frames": blackfly_recording.get("expected_frame_count"),
        "fictrac_final_returncode": fictrac_diagnostics.get("final_returncode"),
        "fictrac_stop_method": fictrac_diagnostics.get("stop_method"),
        "fictrac_stop_fallback": fictrac_diagnostics.get("stop_fallback"),
    }


def _default_run_dirs(runs_root: Path, latest: int) -> list[Path]:
    run_dirs = sorted(path for path in runs_root.iterdir() if path.is_dir())
    if latest > 0:
        run_dirs = run_dirs[-latest:]
    return run_dirs


def _format_summary(summary: dict[str, Any]) -> str:
    counts = summary["counts"]
    mismatches = summary["mismatches_vs_trigger"]
    run_name = Path(summary["run_dir"]).name
    return (
        f"{run_name}: trigger={counts['trigger_rising_edges']} "
        f"fictrac_raw={counts['fictrac_saved_raw_frames']} "
        f"fictrac_udp={counts['fictrac_udp_frame_cnt']} "
        f"fictrac_cb={counts['fictrac_callback_frames']} "
        f"second={counts['second_camera_saved_frames']} "
        f"match={summary['exact_trigger_match']} "
        f"delta_raw={mismatches['fictrac_saved_raw_frames']} "
        f"delta_udp={mismatches['fictrac_udp_frame_cnt']} "
        f"delta_cb={mismatches['fictrac_callback_frames']} "
        f"delta_second={mismatches['second_camera_saved_frames']}"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit trigger/frame parity across MultiBiOS run artifacts.")
    parser.add_argument("run_dirs", nargs="*", help="Run directories to audit.")
    parser.add_argument("--runs-root", default="data/runs", help="Root directory containing run folders.")
    parser.add_argument("--latest", type=int, default=1, help="Use the latest N runs from --runs-root when no run_dirs are provided.")
    parser.add_argument("--json", action="store_true", help="Print JSON instead of compact text summaries.")
    args = parser.parse_args(argv)

    run_dirs = [Path(path) for path in args.run_dirs]
    if not run_dirs:
        run_dirs = _default_run_dirs(Path(args.runs_root), latest=max(int(args.latest), 1))
    summaries = [summarize_run_parity(path) for path in run_dirs]

    if args.json:
        print(json.dumps(summaries, indent=2))
    else:
        for summary in summaries:
            print(_format_summary(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())