from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, List, Sequence


def _split_env_paths(value: str | None) -> List[Path]:
    if not value:
        return []
    return [Path(part) for part in value.split(os.pathsep) if part.strip()]


def _candidate_dirs_from_roots(roots: Iterable[Path]) -> List[Path]:
    candidates: List[Path] = []
    for root in roots:
        candidates.append(root)
        candidates.append(root / "bin64" / "vs2015")
    return candidates


def _iter_candidate_runtime_dirs() -> List[Path]:
    env_candidates: List[Path] = []
    env_candidates.extend(_split_env_paths(os.environ.get("MULTIBIOS_FICTRAC_RUNTIME_DIRS")))
    env_candidates.extend(_split_env_paths(os.environ.get("SPINNAKER_BIN")))
    env_candidates.extend(_split_env_paths(os.environ.get("FLYCAPTURE_BIN")))
    env_candidates.extend(
        _candidate_dirs_from_roots(
            _split_env_paths(os.environ.get("SPINNAKER_ROOT"))
            + _split_env_paths(os.environ.get("PGR_DIR"))
            + _split_env_paths(os.environ.get("FLYCAPTURE_ROOT"))
        )
    )

    default_candidates = [
        Path(r"C:\Program Files\Teledyne\Spinnaker\bin64\vs2015"),
        Path(r"C:\Program Files\Point Grey Research\FlyCapture2\bin64\vs2015"),
    ]

    ordered: List[Path] = []
    seen: set[str] = set()
    for candidate in [*env_candidates, *default_candidates]:
        resolved = str(candidate)
        if resolved in seen:
            continue
        seen.add(resolved)
        if candidate.is_dir():
            ordered.append(candidate)
    return ordered


def prepare_fictrac_runtime() -> List[str]:
    """Prepend detected camera SDK runtime directories to PATH.

    FicTrac camera builds depend on vendor SDK DLLs that are frequently installed
    outside the default process PATH on Windows. Updating the current process PATH
    here lets child processes launched by MultiBiOS inherit the correct DLL
    search path without any external wrapper.
    """

    runtime_dirs = _iter_candidate_runtime_dirs()
    if not runtime_dirs:
        return []

    current_parts = os.environ.get("PATH", "").split(os.pathsep)
    current_normalized = {os.path.normcase(part) for part in current_parts if part}
    prepend_parts: List[str] = []
    for runtime_dir in runtime_dirs:
        runtime_str = str(runtime_dir)
        if os.path.normcase(runtime_str) in current_normalized:
            continue
        prepend_parts.append(runtime_str)

    if prepend_parts:
        os.environ["PATH"] = os.pathsep.join([*prepend_parts, *current_parts])

    return [str(path) for path in runtime_dirs]