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
    if config_path is None or str(config_path).strip() == "":
        return default_fictrac_config_path(hardware_path)

    candidate = Path(str(config_path)).expanduser()
    if candidate.is_absolute():
        return candidate

    if hardware_path is not None:
        return (Path(hardware_path).expanduser().resolve().parent / candidate).resolve()

    return (repo_root() / candidate).resolve()
