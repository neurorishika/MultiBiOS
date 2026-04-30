from __future__ import annotations

from pathlib import Path


DEFAULT_FICTRAC_CONFIG_NAME = "config_camera.txt"


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_fictrac_config_path(hardware_path: str | Path | None = None) -> Path:
    if hardware_path is not None:
        return Path(hardware_path).expanduser().resolve().parent / DEFAULT_FICTRAC_CONFIG_NAME
    return repo_root() / "config" / DEFAULT_FICTRAC_CONFIG_NAME


def resolve_fictrac_config_path(
    config_path: str | Path | None,
    *,
    hardware_path: str | Path | None = None,
) -> Path:
    canonical = default_fictrac_config_path(hardware_path).resolve()
    if config_path is None or str(config_path).strip() == "":
        return canonical

    candidate = Path(str(config_path)).expanduser()
    if not candidate.is_absolute():
        if hardware_path is not None:
            candidate = Path(hardware_path).expanduser().resolve().parent / candidate
        else:
            candidate = repo_root() / candidate

    resolved = candidate.resolve()
    if resolved != canonical:
        raise ValueError(
            f"FicTrac config override is not allowed. Expected canonical config at {canonical}, got {resolved}."
        )
    return canonical
