from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


DEFAULT_HARDWARE_PATH = Path("config/hardware.yaml")
DEFAULT_RUN_OUTPUT_ROOT = Path("data/runs")


def resolve_run_output_root(
    hardware_path: str | Path | None = DEFAULT_HARDWARE_PATH,
    *,
    fallback: str | Path = DEFAULT_RUN_OUTPUT_ROOT,
) -> Path:
    fallback_path = Path(fallback)
    if hardware_path is None:
        return fallback_path

    config_path = Path(hardware_path)
    if not config_path.exists():
        return fallback_path

    with config_path.open("r", encoding="utf-8") as handle:
        raw: Any = yaml.safe_load(handle) or {}

    if not isinstance(raw, dict):
        raise ValueError(f"Hardware config at {config_path} must contain a top-level mapping")

    data_output = raw.get("data_output") or {}
    if not isinstance(data_output, dict):
        raise ValueError(f"data_output in {config_path} must be a mapping when present")

    data_dir = data_output.get("data_dir")
    if data_dir in (None, ""):
        return fallback_path

    return Path(str(data_dir))