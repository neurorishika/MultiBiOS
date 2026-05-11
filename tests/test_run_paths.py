from __future__ import annotations

from pathlib import Path

from multibios import fictrac_timing_audit, parity_audit
from multibios.run_paths import DEFAULT_RUN_OUTPUT_ROOT, resolve_run_output_root


def test_resolve_run_output_root_reads_hardware_data_output(tmp_path: Path) -> None:
    hardware_path = tmp_path / "hardware.yaml"
    hardware_path.write_text(
        "data_output:\n"
        "  data_dir: C:/custom/runs\n",
        encoding="utf-8",
    )

    assert resolve_run_output_root(hardware_path) == Path("C:/custom/runs")


def test_resolve_run_output_root_falls_back_when_hardware_missing(tmp_path: Path) -> None:
    missing_path = tmp_path / "missing-hardware.yaml"

    assert resolve_run_output_root(missing_path) == DEFAULT_RUN_OUTPUT_ROOT


def test_parity_audit_uses_hardware_default_runs_root(tmp_path: Path, monkeypatch) -> None:
    hardware_path = tmp_path / "hardware.yaml"
    hardware_path.write_text(
        f"data_output:\n  data_dir: {tmp_path.as_posix()}/captured-runs\n",
        encoding="utf-8",
    )
    expected_root = tmp_path / "captured-runs"
    captured: dict[str, Path] = {}

    monkeypatch.setattr(
        parity_audit,
        "_default_run_dirs",
        lambda runs_root, latest: captured.setdefault("runs_root", runs_root) and [],
    )

    assert parity_audit.main(["--hardware", str(hardware_path), "--json"]) == 0
    assert captured["runs_root"] == expected_root


def test_fictrac_timing_audit_uses_hardware_default_runs_root(tmp_path: Path, monkeypatch) -> None:
    hardware_path = tmp_path / "hardware.yaml"
    hardware_path.write_text(
        f"data_output:\n  data_dir: {tmp_path.as_posix()}/captured-runs\n",
        encoding="utf-8",
    )
    expected_root = tmp_path / "captured-runs"
    captured: dict[str, Path] = {}

    monkeypatch.setattr(
        fictrac_timing_audit,
        "_default_run_dirs",
        lambda runs_root, latest: captured.setdefault("runs_root", runs_root) and [],
    )

    assert fictrac_timing_audit.main(["--hardware", str(hardware_path), "--json"]) == 0
    assert captured["runs_root"] == expected_root