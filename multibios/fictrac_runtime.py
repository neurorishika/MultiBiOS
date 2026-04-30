from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Iterable, List, Mapping, Sequence


_CONFLICTING_ENV_VARS = (
    "CONDA_DEFAULT_ENV",
    "CONDA_PREFIX",
    "CONDA_PROMPT_MODIFIER",
    "CONDA_EXE",
    "CONDA_PYTHON_EXE",
    "CONDA_SHLVL",
    "PYTHONHOME",
    "PYTHONPATH",
    "PYTHONEXECUTABLE",
    "VIRTUAL_ENV",
)


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


def _is_within_root(path_str: str, root: Path) -> bool:
    try:
        candidate = Path(path_str).resolve(strict=False)
        root_resolved = root.resolve(strict=False)
    except OSError:
        return False

    try:
        common = os.path.commonpath([str(candidate), str(root_resolved)])
    except ValueError:
        return False
    return os.path.normcase(common) == os.path.normcase(str(root_resolved))


def build_fictrac_subprocess_env(
    *,
    fictrac_bin_path: str | Path | None = None,
    base_env: Mapping[str, str] | None = None,
) -> dict[str, str]:
    env = dict(os.environ if base_env is None else base_env)
    if os.name != "nt":
        return env

    for key in _CONFLICTING_ENV_VARS:
        env.pop(key, None)

    blocked_roots: list[Path] = []
    conda_prefix = env.get("CONDA_PREFIX")
    if conda_prefix:
        blocked_roots.append(Path(conda_prefix))
    blocked_roots.append(Path(sys.executable).resolve(strict=False).parent)

    path_parts = [part for part in env.get("PATH", "").split(os.pathsep) if part.strip()]
    filtered_parts: list[str] = []
    seen: set[str] = set()

    prepend_candidates: list[str] = []
    if fictrac_bin_path is not None:
        prepend_candidates.append(str(Path(fictrac_bin_path).expanduser().resolve().parent))
    prepend_candidates.extend(str(path) for path in _iter_candidate_runtime_dirs())

    for part in [*prepend_candidates, *path_parts]:
        normalized = os.path.normcase(part)
        if normalized in seen:
            continue
        if any(_is_within_root(part, root) for root in blocked_roots):
            continue
        seen.add(normalized)
        filtered_parts.append(part)

    env["PATH"] = os.pathsep.join(filtered_parts)
    return env


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