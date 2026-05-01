from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


_CLOCK_FIELDS = ("alt_timestamp", "wall_time", "timestamp")


def _default_run_dirs(runs_root: Path, latest: int) -> list[Path]:
    run_dirs = sorted(path for path in runs_root.iterdir() if path.is_dir())
    if latest > 0:
        run_dirs = run_dirs[-latest:]
    return run_dirs


def _load_frames(run_dir: Path) -> np.ndarray:
    frames_path = run_dir / "fictrac_frames.npz"
    if not frames_path.exists():
        raise FileNotFoundError(f"No fictrac_frames.npz found in {run_dir}")
    with np.load(frames_path, allow_pickle=False) as npz:
        return np.array(npz["frames"], copy=True)


def _clock_values_ms(frames: np.ndarray, clock: str) -> np.ndarray:
    if clock not in frames.dtype.names:
        raise ValueError(f"Clock field '{clock}' is not present in fictrac_frames.npz")

    values = np.asarray(frames[clock], dtype=np.float64)
    if clock == "wall_time":
        return values * 1000.0
    if clock == "timestamp":
        return values / 1000.0
    return values


def summarize_fictrac_intervals(run_dir: Path, *, clock: str = "alt_timestamp") -> dict[str, Any]:
    run_dir = run_dir.resolve()
    frames = _load_frames(run_dir)
    if frames.size == 0:
        return {
            "run_dir": str(run_dir),
            "clock": clock,
            "frame_count": 0,
            "interval_count": 0,
            "intervals_ms": [],
            "frame_numbers": [],
            "max_interval_ms": None,
            "max_interval_index": None,
            "max_interval_after_frame": None,
            "mean_interval_ms": None,
            "median_interval_ms": None,
            "p95_interval_ms": None,
            "std_interval_ms": None,
        }

    clock_ms = _clock_values_ms(frames, clock)
    frame_numbers = np.asarray(frames["frame_cnt"], dtype=np.int64)
    intervals_ms = np.diff(clock_ms)
    after_frames = frame_numbers[1:]

    if intervals_ms.size == 0:
        return {
            "run_dir": str(run_dir),
            "clock": clock,
            "frame_count": int(frames.size),
            "interval_count": 0,
            "intervals_ms": [],
            "frame_numbers": after_frames.astype(int).tolist(),
            "max_interval_ms": None,
            "max_interval_index": None,
            "max_interval_after_frame": None,
            "mean_interval_ms": None,
            "median_interval_ms": None,
            "p95_interval_ms": None,
            "std_interval_ms": None,
        }

    max_index = int(np.argmax(intervals_ms))
    return {
        "run_dir": str(run_dir),
        "clock": clock,
        "frame_count": int(frames.size),
        "interval_count": int(intervals_ms.size),
        "intervals_ms": intervals_ms.astype(float).tolist(),
        "frame_numbers": after_frames.astype(int).tolist(),
        "max_interval_ms": float(intervals_ms[max_index]),
        "max_interval_index": max_index,
        "max_interval_after_frame": int(after_frames[max_index]),
        "mean_interval_ms": float(np.mean(intervals_ms)),
        "median_interval_ms": float(np.median(intervals_ms)),
        "p95_interval_ms": float(np.percentile(intervals_ms, 95)),
        "std_interval_ms": float(np.std(intervals_ms)),
    }


def compare_interval_summaries(summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    comparisons: list[dict[str, Any]] = []
    if len(summaries) < 2:
        return comparisons

    baseline = summaries[0]
    baseline_name = Path(baseline["run_dir"]).name
    for other in summaries[1:]:
        other_name = Path(other["run_dir"]).name
        comparison = {
            "baseline_run": baseline_name,
            "other_run": other_name,
            "clock": baseline["clock"],
            "max_interval_delta_ms": (
                None
                if baseline["max_interval_ms"] is None or other["max_interval_ms"] is None
                else float(other["max_interval_ms"] - baseline["max_interval_ms"])
            ),
            "max_interval_index_delta": (
                None
                if baseline["max_interval_index"] is None or other["max_interval_index"] is None
                else int(other["max_interval_index"] - baseline["max_interval_index"])
            ),
            "max_interval_after_frame_delta": (
                None
                if baseline["max_interval_after_frame"] is None or other["max_interval_after_frame"] is None
                else int(other["max_interval_after_frame"] - baseline["max_interval_after_frame"])
            ),
            "mean_interval_delta_ms": (
                None
                if baseline["mean_interval_ms"] is None or other["mean_interval_ms"] is None
                else float(other["mean_interval_ms"] - baseline["mean_interval_ms"])
            ),
        }
        comparisons.append(comparison)
    return comparisons


def write_interval_csv(summary: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["interval_index", "after_frame_cnt", "interval_ms"])
        for index, (frame_num, interval_ms) in enumerate(zip(summary["frame_numbers"], summary["intervals_ms"])):
            writer.writerow([index, frame_num, f"{float(interval_ms):.9f}"])


def _format_summary(summary: dict[str, Any]) -> str:
    run_name = Path(summary["run_dir"]).name
    return (
        f"{run_name}: clock={summary['clock']} frames={summary['frame_count']} "
        f"intervals={summary['interval_count']} mean_ms={summary['mean_interval_ms']:.6f} "
        f"max_ms={summary['max_interval_ms']:.6f} max_idx={summary['max_interval_index']} "
        f"after_frame={summary['max_interval_after_frame']} p95_ms={summary['p95_interval_ms']:.6f}"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Summarize FicTrac inter-frame timing from fictrac_frames.npz.")
    parser.add_argument("run_dirs", nargs="*", help="Run directories to analyze.")
    parser.add_argument("--runs-root", default="data/runs", help="Root directory containing run folders.")
    parser.add_argument("--latest", type=int, default=1, help="Use the latest N runs from --runs-root when no run_dirs are provided.")
    parser.add_argument("--clock", choices=_CLOCK_FIELDS, default="alt_timestamp", help="Timestamp field to use for inter-frame intervals.")
    parser.add_argument("--json", action="store_true", help="Print JSON output including comparison summaries.")
    parser.add_argument("--csv-dir", default=None, help="Optional directory to write one CSV time series per analyzed run.")
    args = parser.parse_args(argv)

    run_dirs = [Path(path) for path in args.run_dirs]
    if not run_dirs:
        run_dirs = _default_run_dirs(Path(args.runs_root), latest=max(int(args.latest), 1))

    summaries = [summarize_fictrac_intervals(path, clock=args.clock) for path in run_dirs]
    comparisons = compare_interval_summaries(summaries)

    if args.csv_dir:
        csv_dir = Path(args.csv_dir)
        for summary in summaries:
            run_name = Path(summary["run_dir"]).name
            write_interval_csv(summary, csv_dir / f"{run_name}-{args.clock}-intervals.csv")

    if args.json:
        print(json.dumps({"runs": summaries, "comparisons": comparisons}, indent=2))
    else:
        for summary in summaries:
            print(_format_summary(summary))
        for comparison in comparisons:
            print(
                f"compare {comparison['baseline_run']} -> {comparison['other_run']}: "
                f"delta_max_ms={comparison['max_interval_delta_ms']} "
                f"delta_idx={comparison['max_interval_index_delta']} "
                f"delta_after_frame={comparison['max_interval_after_frame_delta']} "
                f"delta_mean_ms={comparison['mean_interval_delta_ms']}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())