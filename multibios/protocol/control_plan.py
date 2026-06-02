from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np

from multibios.protocol.numeric_expr import (build_protocol_numeric_symbols,
                                             evaluate_numeric_expression)
from multibios.protocol.schema import CompileError


@dataclass
class TimelineEvent:
    """A scheduled logical device action derived from protocol YAML."""

    time_ms: float
    action: str
    device: str
    side: str = ""
    state: str = ""
    value: float = 0.0
    phase: str = ""
    repeat_idx: int = 0


@dataclass
class CompiledControlPlan:
    """Shared protocol expansion used by both hardware and serial runners."""

    timeline: list[TimelineEvent]
    microscope_times_ms: list[float]
    camera_windows_ms: list[tuple[float, float]]
    total_duration_ms: float
    seed: int


def _norm_dev(s: str) -> str:
    return s.strip().lower()


def _parse_state_list(spec: Any, times: int) -> list[str]:
    if spec is None:
        return ["OFF"]
    if isinstance(spec, list):
        toks = [str(x).strip().upper() for x in spec]
    else:
        toks = [p.strip().upper() for p in str(spec).split(",") if p.strip()]
    if not toks:
        toks = ["OFF"]
    if len(toks) not in (1, times):
        toks = [toks[0]]
    return toks


def _resolve_choice(tok: str, rng: np.random.Generator) -> str:
    if "|" not in tok:
        return tok
    alts = [a.strip().upper() for a in tok.split("|") if a.strip()]
    return str(rng.choice(alts))


def compile_control_plan(
    protocol_yaml: dict[str, Any],
    seed: Optional[int] = None,
) -> CompiledControlPlan:
    """Parse protocol YAML into a shared chronological control plan."""

    protocol_block = protocol_yaml.get("protocol", {})
    timing = protocol_block.get("timing", {})
    seq = protocol_yaml.get("sequence", [])

    if seed is None:
        seed_val = timing.get("seed", None)
        if seed_val is not None:
            seed = int(seed_val)
        else:
            seed = int(np.random.SeedSequence().entropy)
    rng = np.random.default_rng(seed)

    timeline: list[TimelineEvent] = []
    microscope_times: list[float] = []
    camera_windows: list[tuple[float, float]] = []
    camera_on_at: float | None = None
    numeric_symbols = build_protocol_numeric_symbols(protocol_yaml)

    expanded = []
    total_ms = 0.0
    for entry in seq:
        name = entry.get("phase", "PHASE")
        try:
            dur = evaluate_numeric_expression(entry.get("duration", 0), numeric_symbols)
        except Exception as exc:
            raise CompileError(
                f"Phase '{name}': invalid duration '{entry.get('duration', 0)}': {exc}"
            ) from exc
        if "times" in entry:
            times = int(entry["times"])
        elif "repeat" in entry:
            times = int(entry["repeat"]) + 1
        else:
            times = 1
        if times <= 0:
            raise CompileError(f"Phase '{name}': times must be positive")
        total_ms += dur * times
        expanded.append((name, dur, entry, times))

    t_cursor = 0.0
    for name, duration, entry, times in expanded:
        randomize = bool(entry.get("randomize", False))
        actions = entry.get("actions", [])

        left_spec = None
        right_spec = None
        for action in actions:
            dev = _norm_dev(action.get("device", ""))
            if dev == "olfactometer.left":
                left_spec = action.get("state", "OFF")
            elif dev == "olfactometer.right":
                right_spec = action.get("state", "OFF")

        left_list = _parse_state_list(left_spec, times)
        right_list = _parse_state_list(right_spec, times)

        perm = np.arange(times)
        if randomize:
            perm = rng.permutation(times)

        if len(left_list) == times:
            left_list = [left_list[i] for i in perm]
        else:
            left_list = left_list * times

        if len(right_list) == times:
            right_list = [right_list[i] for i in perm]
        else:
            right_list = right_list * times

        resolved_left = [_resolve_choice(tok, rng) for tok in left_list]
        resolved_right = [_resolve_choice(tok, rng) for tok in right_list]

        for action in actions:
            dev = _norm_dev(action.get("device", ""))
            try:
                timing_ms = evaluate_numeric_expression(action.get("timing", 0), numeric_symbols)
            except Exception as exc:
                raise CompileError(
                    f"Phase '{name}': invalid timing '{action.get('timing', 0)}' for device '{dev}': {exc}"
                ) from exc
            if dev == "triggers.camera_continuous":
                enabled = bool(action.get("state", False))
                abs_t = t_cursor + timing_ms
                if enabled:
                    camera_on_at = abs_t
                elif camera_on_at is not None:
                    camera_windows.append((camera_on_at, abs_t))
                    camera_on_at = None

        for rep_idx in range(times):
            repeat_t0 = t_cursor + rep_idx * duration
            for action in actions:
                dev = _norm_dev(action.get("device", ""))
                try:
                    timing_ms = evaluate_numeric_expression(action.get("timing", 0), numeric_symbols)
                except Exception as exc:
                    raise CompileError(
                        f"Phase '{name}': invalid timing '{action.get('timing', 0)}' for device '{dev}': {exc}"
                    ) from exc
                t_abs = repeat_t0 + timing_ms

                if dev.startswith("mfc."):
                    val = float(action.get("value", action.get("state", 0.0)))
                    timeline.append(
                        TimelineEvent(
                            time_ms=t_abs,
                            action="mfc",
                            device=dev,
                            value=val,
                            phase=name,
                            repeat_idx=rep_idx,
                        )
                    )
                elif dev == "olfactometer.left":
                    timeline.append(
                        TimelineEvent(
                            time_ms=t_abs,
                            action="olfactometer",
                            device=dev,
                            side="left",
                            state=resolved_left[rep_idx],
                            phase=name,
                            repeat_idx=rep_idx,
                        )
                    )
                elif dev == "olfactometer.right":
                    timeline.append(
                        TimelineEvent(
                            time_ms=t_abs,
                            action="olfactometer",
                            device=dev,
                            side="right",
                            state=resolved_right[rep_idx],
                            phase=name,
                            repeat_idx=rep_idx,
                        )
                    )
                elif dev == "switch_valve.left":
                    st = str(action.get("state", action.get("value", "CLEAN"))).strip().upper()
                    timeline.append(
                        TimelineEvent(
                            time_ms=t_abs,
                            action="switch_valve",
                            device=dev,
                            side="left",
                            state=st,
                            phase=name,
                            repeat_idx=rep_idx,
                        )
                    )
                elif dev == "switch_valve.right":
                    st = str(action.get("state", action.get("value", "CLEAN"))).strip().upper()
                    timeline.append(
                        TimelineEvent(
                            time_ms=t_abs,
                            action="switch_valve",
                            device=dev,
                            side="right",
                            state=st,
                            phase=name,
                            repeat_idx=rep_idx,
                        )
                    )
                elif dev == "triggers.microscope":
                    if bool(action.get("state", True)):
                        microscope_times.append(t_abs)
                        timeline.append(
                            TimelineEvent(
                                time_ms=t_abs,
                                action="log_only",
                                device=dev,
                                state="PULSE",
                                phase=name,
                                repeat_idx=rep_idx,
                            )
                        )
                elif dev in ("triggers.camera", "triggers.camera_continuous"):
                    timeline.append(
                        TimelineEvent(
                            time_ms=t_abs if rep_idx == 0 else -1,
                            action="log_only",
                            device=dev,
                            state=str(action.get("state", "")),
                            phase=name,
                            repeat_idx=rep_idx,
                        )
                    )

        t_cursor += duration * times

    if camera_on_at is not None:
        camera_windows.append((camera_on_at, total_ms))

    timeline = [event for event in timeline if event.time_ms >= 0]
    timeline.sort(key=lambda event: event.time_ms)

    return CompiledControlPlan(
        timeline=timeline,
        microscope_times_ms=microscope_times,
        camera_windows_ms=camera_windows,
        total_duration_ms=total_ms,
        seed=int(seed),
    )


def write_control_plan_csv(path: str | Path, timeline: list[TimelineEvent]) -> None:
    target = Path(path)
    with open(target, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "time_ms", "action", "device", "side", "state", "value", "phase", "repeat_idx"
        ])
        for event in timeline:
            writer.writerow([
                f"{event.time_ms:.1f}",
                event.action,
                event.device,
                event.side,
                event.state,
                f"{event.value:.4f}",
                event.phase,
                event.repeat_idx,
            ])