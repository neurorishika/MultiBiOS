from __future__ import annotations

import json
import threading
from pathlib import Path
import logging
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

import multibios.blackfly.triggered_camera_record as triggered_camera_record_module
from multibios.blackfly.triggered_camera_record import \
    postprocess_triggered_camera_recording
from multibios.fictrac_client import FICTRAC_FRAME_DTYPE, FicTracState
from multibios.fictrac_config import resolve_fictrac_config_path
from multibios.fictrac_raw_recording import postprocess_fictrac_raw_recording
from multibios.protocol.control_plan import compile_control_plan
from multibios.protocol.schema import ProtocolCompiler, TimingConfig
from multibios.run_protocol import (ExperimentCallback,
                                    _apply_camera_mode_runtime_overrides,
                                    _apply_hardware_owned_camera_timing,
                                    _disable_camera_runtime,
                                    _estimate_microscopy_imaging_periods,
                                    _resolve_camera_mode,
                                    _should_force_headless_fictrac_run,
                                    _finalize_raw_chunk_retention,
                                    _write_parity_summary,
                                    _canonicalize_fictrac_runtime_info,
                                    _compute_fictrac_drain_timeout_s,
                                    _compute_second_camera_startup_timeout_s,
                                    _count_rising_edges,
                                    _first_rising_edge_sample,
                                    _prepare_fictrac_runtime_config,
                                    _read_yaml_text,
                                    _safe_stop_task, _stop_fictrac,
                                    _scale_compiled_mfc_ao_from_slpm_to_volts,
                                    _wait_for_fictrac_frame_drain,
                                    RunProtocolConfig,
                                    load_run_protocol_config)
from multibios.run_dataset import RunDatasetLayout


def _state(frame_cnt: int) -> FicTracState:
    return FicTracState(
        frame_cnt=frame_cnt,
        posx=1.0,
        posy=2.0,
        heading=3.0,
        direction=4.0,
        speed=5.0,
        intx=6.0,
        inty=7.0,
        timestamp=8.0 + frame_cnt,
        seq_num=frame_cnt,
        delta_timestamp=0.005,
        alt_timestamp=0.0,
    )


def test_experiment_callback_make_consumer_tracks_newest() -> None:
    callback = ExperimentCallback()
    callback.process_callback(_state(10))
    callback.process_callback(_state(11))

    consumer = callback.make_consumer()
    latest = consumer.consume_latest()
    assert latest.seq == 1
    assert latest.frame is not None
    assert latest.frame.frame_cnt == 11

    callback.process_callback(_state(12))
    newer = consumer.wait_for_newer(timeout=0.1)
    assert newer.frame is not None
    assert newer.frame.frame_cnt == 12


def test_prepare_fictrac_runtime_config_enables_raw_video(tmp_path: Path) -> None:
    source_config = tmp_path / "config_camera.txt"
    source_config.write_text(
        "src_fn           : 1\n"
        "save_raw         : n\n"
        "src_fps          : -1.000000\n",
        encoding="utf-8",
    )

    runtime_path, camera_index, info = _prepare_fictrac_runtime_config(
        source_config,
        tmp_path,
        enable_raw_video=True,
        camera_fps=60.0,
        video_codec="raw",
        first_frame_timeout_ms=0,
    )

    runtime_text = runtime_path.read_text(encoding="utf-8")
    assert camera_index == 1
    assert "save_raw         : y" in runtime_text
    assert "src_fps          : 60.000000" in runtime_text
    assert "src_first_frame_timeout_ms: 0" in runtime_text
    assert "vid_codec        : raw" in runtime_text
    assert f"output_fn        : {(tmp_path / 'fictrac').as_posix()}" in runtime_text
    assert info["save_raw"] is True
    assert info["first_frame_timeout_ms"] == 0


def test_read_yaml_text_reads_utf8_protocol_comments(tmp_path: Path) -> None:
    protocol_path = tmp_path / "odor_lateralization.yaml"
    protocol_path.write_text(
        "# Unicode box: ┌───┐\n"
        "protocol:\n"
        "  name: utf8 protocol\n"
        "  timing:\n"
        "    base_unit: ms\n"
        "    sample_rate: 1000\n"
        "sequence: []\n",
        encoding="utf-8",
    )

    loaded = _read_yaml_text(protocol_path)

    assert loaded["protocol"]["name"] == "utf8 protocol"


def test_prepare_fictrac_runtime_config_overrides_src_fn(tmp_path: Path) -> None:
    source_config = tmp_path / "config_camera.txt"
    source_config.write_text(
        "src_fn           : 0\n"
        "save_raw         : n\n",
        encoding="utf-8",
    )

    runtime_path, camera_index, _ = _prepare_fictrac_runtime_config(
        source_config,
        tmp_path,
        enable_raw_video=False,
        camera_fps=None,
        video_codec="raw",
        first_frame_timeout_ms=0,
        camera_index_override=1,
    )

    runtime_text = runtime_path.read_text(encoding="utf-8")
    assert camera_index == 1
    assert "src_fn           : 1" in runtime_text


def test_prepare_fictrac_runtime_config_can_force_headless(tmp_path: Path) -> None:
    source_config = tmp_path / "config_camera.txt"
    source_config.write_text(
        "do_display       : y\n"
        "save_raw         : n\n",
        encoding="utf-8",
    )

    runtime_path, _, info = _prepare_fictrac_runtime_config(
        source_config,
        tmp_path,
        enable_raw_video=False,
        camera_fps=None,
        video_codec="raw",
        first_frame_timeout_ms=0,
        force_headless=True,
    )

    runtime_text = runtime_path.read_text(encoding="utf-8")
    assert "do_display       : n" in runtime_text
    assert info["force_headless"] is True


def test_should_force_headless_fictrac_run_can_be_overridden_for_live_display() -> None:
    assert _should_force_headless_fictrac_run(
        camera_trigger_fps_hz=200.0,
        expected_camera_frames=93_600,
        allow_live_display=False,
    ) is True
    assert _should_force_headless_fictrac_run(
        camera_trigger_fps_hz=200.0,
        expected_camera_frames=93_600,
        allow_live_display=True,
    ) is False


def test_first_rising_edge_sample_reports_first_transition() -> None:
    trace = np.array([False, False, True, True, False, True], dtype=bool)
    assert _first_rising_edge_sample(trace) == 2
    assert _count_rising_edges(trace) == 2


def test_compute_second_camera_startup_timeout_includes_first_trigger_delay() -> None:
    timeout_s = _compute_second_camera_startup_timeout_s(
        first_trigger_sample=1000,
        sample_rate=1000,
        arm_delay_s=0.5,
        recorder_timeout_ms=850,
    )

    assert timeout_s >= 2.85


def test_compute_fictrac_drain_timeout_scales_with_remaining_frames() -> None:
    timeout_s = _compute_fictrac_drain_timeout_s(
        expected_frame_count=70000,
        observed_frame_count=69978,
        camera_fps=142.857143,
    )

    assert timeout_s >= 0.2


def test_wait_for_fictrac_frame_drain_reaches_expected_count(monkeypatch: pytest.MonkeyPatch) -> None:
    callback = ExperimentCallback()
    for frame_cnt in range(3):
        callback.process_callback(_state(frame_cnt))

    fake_now = {"value": 0.0}
    health_checks = {"count": 0}

    def fake_sleep(duration: float) -> None:
        fake_now["value"] += duration
        callback.process_callback(_state(callback.frame_count))

    monkeypatch.setattr("multibios.run_protocol.time.monotonic", lambda: fake_now["value"])
    monkeypatch.setattr("multibios.run_protocol.time.sleep", fake_sleep)

    _wait_for_fictrac_frame_drain(
        callback=callback,
        expected_frame_count=5,
        camera_fps=100.0,
        logger=logging.getLogger("test"),
        health_check=lambda: health_checks.__setitem__("count", health_checks["count"] + 1),
    )

    assert callback.frame_count == 5
    assert health_checks["count"] >= 1


def test_wait_for_fictrac_frame_drain_tolerates_terminal_exit() -> None:
    callback = ExperimentCallback()
    for frame_cnt in range(3):
        callback.process_callback(_state(frame_cnt))

    _wait_for_fictrac_frame_drain(
        callback=callback,
        expected_frame_count=5,
        camera_fps=100.0,
        logger=logging.getLogger("test"),
        health_check=lambda: (_ for _ in ()).throw(
            RuntimeError("FicTrac thread crashed: FicTrac process exited unexpectedly (no exception)")
        ),
    )

    assert callback.frame_count == 3


def test_experiment_callback_reports_stop_requested() -> None:
    callback = ExperimentCallback()
    assert callback.stop_requested() is False
    callback.request_stop()
    assert callback.stop_requested() is True


def test_load_run_protocol_config_reads_hardware_owned_fields(tmp_path: Path) -> None:
    hw_path = tmp_path / "hardware.yaml"
    hw_path.write_text(
        "use_camera: false\n"
        "teensy:\n"
        "  port: COM9\n"
        "  baud: 230400\n"
        "  capture_serial: true\n"
        "camera_recording:\n"
        "  trigger_fps_hz: 200\n"
        "  trigger_pulse_ms: 1\n"
        "  save_fictrac_camera_video: true\n"
        "  save_second_camera_video: true\n"
        "  fictrac_config: config_camera.txt\n"
        "  fictrac_bin: C:/rig/fictrac-spinnaker.exe\n"
        "  fictrac_console_out: fictrac_hw.txt\n"
        "  fictrac_camera_serial: 26021184\n"
        "  fictrac_first_frame_timeout_ms: 0\n"
        "  fictrac_arm_delay_s: 0.5\n"
        "  fictrac_startup_timeout_s: 0\n"
        "  fictrac_timeout_s: 7\n"
        "  second_camera_serial: 26048173\n"
        "  second_camera_index: 1\n"
        "  fictrac_raw_video_codec: mjpg\n"
        "  second_camera_timeout_ms: 125\n"
        "  second_camera_queue_size: 32\n"
        "  second_camera_stream_buffer_count: 64\n"
        "  second_camera_exposure_us: 4000\n"
        "  second_camera_roi_width: 512\n"
        "  second_camera_roi_height: 512\n"
        "  second_camera_binning: 2\n"
        "  second_camera_gain_db: 7.5\n"
        "  second_camera_gamma: 0.8\n"
        "  default_exposure_us: 4500\n"
        "  default_roi_width: 400\n"
        "  default_roi_height: 400\n"
        "  default_binning: 1\n"
        "  default_gain_db: 3.5\n"
        "  default_gamma: 1.1\n"
        "  verify_no_dropped_frames: true\n"
        "  convert_second_camera_bin_to_lossless_mkv: false\n"
        "  raw_chunk_retention_policy: delete_after_parity\n"
        "mfc:\n"
        "  mode: analog\n"
        "  analog_value_units: slpm\n"
        "  analog_full_scale_slpm:\n"
        "    mfc.air_left_setpoint: 1.0\n"
        "    mfc.air_right_setpoint: 1.0\n"
        "    mfc.odor_left_setpoint: 1.0\n"
        "    mfc.odor_right_setpoint: 1.0\n"
        "  live_interval_s: 0\n"
        "daq:\n"
        "  latch_interval_ms: 12\n"
        "data_output:\n"
        "  data_dir: C:/data/runs\n"
        "  open_explorer: false\n"
        "  explorer_port: 9000\n",
        encoding="utf-8",
    )

    cfg = load_run_protocol_config(None, hardware_path=hw_path)
    assert cfg.teensy_port == "COM9"
    assert cfg.teensy_baud == 230400
    assert cfg.capture_teensy_serial is True
    assert cfg.use_camera is False
    assert cfg.fictrac_camera_serial == "26021184"
    assert cfg.save_fictrac_camera_video is True
    assert cfg.save_second_camera_video is True
    assert cfg.camera_trigger_fps_hz == 200.0
    assert cfg.camera_trigger_pulse_ms == 1
    assert cfg.second_camera_serial == "26048173"
    assert cfg.second_camera_index == 1
    assert cfg.fictrac_raw_video_codec == "mjpg"
    assert cfg.blackfly_exposure_us == 4500.0
    assert cfg.blackfly_roi_width == 400
    assert cfg.blackfly_roi_height == 400
    assert cfg.blackfly_binning == 1
    assert cfg.blackfly_gain_db == 3.5
    assert cfg.blackfly_gamma == 1.1
    assert cfg.second_camera_timeout_ms == 125
    assert cfg.second_camera_queue_size == 32
    assert cfg.second_camera_stream_buffer_count == 64
    assert cfg.second_camera_exposure_us == 4000.0
    assert cfg.second_camera_roi_width == 512
    assert cfg.second_camera_roi_height == 512
    assert cfg.second_camera_binning == 2
    assert cfg.second_camera_gain_db == 7.5
    assert cfg.second_camera_gamma == 0.8
    assert cfg.verify_camera_recording is True
    assert cfg.convert_second_camera_bin_to_lossless_mkv is False
    assert cfg.raw_chunk_retention_policy == "delete_after_parity"
    assert cfg.mfc_mode == "analog"
    assert cfg.mfc_value_units == "slpm"
    assert cfg.mfc_analog_full_scale_slpm["mfc.air_left_setpoint"] == 1.0
    assert cfg.data_dir == "C:/data/runs"


def test_load_run_protocol_config_second_camera_blank_values_fall_back_to_blackfly_defaults(tmp_path: Path) -> None:
    hw_path = tmp_path / "hardware.yaml"
    hw_path.write_text(
        "camera_recording:\n"
        "  second_camera_serial: 26048173\n"
        "  second_camera_exposure_us:\n"
        "  second_camera_roi_width:\n"
        "  second_camera_roi_height:\n"
        "  second_camera_gain_db:\n"
        "  second_camera_gamma:\n"
        "  default_exposure_us: 4200\n"
        "  default_roi_width: 640\n"
        "  default_roi_height: 512\n"
        "  default_binning: 1\n"
        "  default_gain_db: 5.5\n"
        "  default_gamma: 0.9\n",
        encoding="utf-8",
    )

    cfg = load_run_protocol_config(None, hardware_path=hw_path)
    assert cfg.second_camera_exposure_us == 4200.0
    assert cfg.second_camera_roi_width == 640
    assert cfg.second_camera_roi_height == 512
    assert cfg.second_camera_gain_db == 5.5
    assert cfg.second_camera_gamma == 0.9


def test_load_run_protocol_config_reads_hardware_fictrac_defaults(tmp_path: Path) -> None:
    hw_path = tmp_path / "hardware.yaml"
    hw_path.write_text(
        "camera_recording:\n"
        "  fictrac_config: config_camera.txt\n"
        "  fictrac_camera_serial: 26021184\n"
        "  fictrac_bin: C:/rig/fictrac-spinnaker.exe\n"
        "  fictrac_console_out: fictrac_hw.txt\n"
        "  fictrac_first_frame_timeout_ms: 0\n"
        "  fictrac_arm_delay_s: 0.5\n"
        "  fictrac_startup_timeout_s: 0\n"
        "  fictrac_timeout_s: 7\n",
        encoding="utf-8",
    )

    cfg = load_run_protocol_config(None, hardware_path=hw_path)
    assert cfg.fictrac_config == str((tmp_path / "config_camera.txt").resolve())
    assert cfg.fictrac_camera_serial == "26021184"
    assert cfg.fictrac_bin == "C:/rig/fictrac-spinnaker.exe"
    assert cfg.fictrac_console_out == "fictrac_hw.txt"
    assert cfg.fictrac_first_frame_timeout_ms == 0
    assert cfg.fictrac_arm_delay_s == 0.5
    assert cfg.fictrac_startup_timeout_s == 0.0
    assert cfg.fictrac_timeout_s == 7.0


def test_disable_camera_runtime_turns_off_camera_triggering_and_recording() -> None:
    cfg = RunProtocolConfig(
        fictrac_config="C:/rig/config_camera.txt",
        fictrac_camera_serial="26021184",
        save_fictrac_camera_video=True,
        save_second_camera_video=True,
        camera_trigger_fps_hz=200.0,
        camera_trigger_pulse_ms=1,
        second_camera_index=1,
        second_camera_serial="26048173",
        verify_camera_recording=True,
    )
    timing_block = {
        "sample_rate": 1000,
        "camera_interval": 5.0,
        "camera_pulse_duration": 1.0,
    }

    _disable_camera_runtime(cfg)
    _apply_hardware_owned_camera_timing(timing_block, cfg)

    assert cfg.fictrac_config == ""
    assert cfg.fictrac_camera_serial == ""
    assert cfg.save_fictrac_camera_video is False
    assert cfg.save_second_camera_video is False
    assert cfg.camera_trigger_fps_hz is None
    assert cfg.camera_trigger_pulse_ms is None
    assert cfg.second_camera_index is None
    assert cfg.second_camera_serial == ""
    assert cfg.verify_camera_recording is False
    assert timing_block["camera_interval"] == 0.0
    assert "camera_pulse_duration" not in timing_block


def test_resolve_camera_mode_prefers_cli_over_hardware_default() -> None:
    assert _resolve_camera_mode(runtime_default=False, force_camera=False, force_nocamera=False) is False
    assert _resolve_camera_mode(runtime_default=False, force_camera=True, force_nocamera=False) is True
    assert _resolve_camera_mode(runtime_default=True, force_camera=False, force_nocamera=True) is False


def test_estimate_microscopy_imaging_periods_counts_compiled_events() -> None:
    plan = SimpleNamespace(microscope_times_ms=[100.0, 250.0, 500.0, 750.0])

    assert _estimate_microscopy_imaging_periods(plan) == 2


def test_estimate_microscopy_imaging_periods_rejects_unpaired_triggers() -> None:
    plan = SimpleNamespace(microscope_times_ms=[100.0, 250.0, 500.0])

    with pytest.raises(ValueError, match="start and stop trigger"):
        _estimate_microscopy_imaging_periods(plan)


def test_apply_camera_mode_runtime_overrides_disables_yaml_camera_settings_by_default() -> None:
    cfg = RunProtocolConfig(
        use_camera=False,
        fictrac_config="C:/rig/config_camera.txt",
        fictrac_camera_serial="26021184",
        save_fictrac_camera_video=True,
        save_second_camera_video=True,
        camera_trigger_fps_hz=200.0,
        camera_trigger_pulse_ms=1,
        second_camera_index=1,
        second_camera_serial="26048173",
        verify_camera_recording=True,
    )

    use_camera = _apply_camera_mode_runtime_overrides(
        cfg,
        force_camera=False,
        force_nocamera=False,
    )

    assert use_camera is False
    assert cfg.fictrac_config == ""
    assert cfg.save_fictrac_camera_video is False
    assert cfg.save_second_camera_video is False
    assert cfg.camera_trigger_fps_hz is None
    assert cfg.second_camera_serial == ""


def test_apply_camera_mode_runtime_overrides_respects_explicit_usecamera_flag() -> None:
    cfg = RunProtocolConfig(
        use_camera=False,
        fictrac_config="C:/rig/config_camera.txt",
        save_fictrac_camera_video=True,
        camera_trigger_fps_hz=200.0,
    )

    use_camera = _apply_camera_mode_runtime_overrides(
        cfg,
        force_camera=True,
        force_nocamera=False,
    )

    assert use_camera is True
    assert cfg.fictrac_config == "C:/rig/config_camera.txt"
    assert cfg.save_fictrac_camera_video is True
    assert cfg.camera_trigger_fps_hz == 200.0


def test_load_run_protocol_config_rejects_fictrac_target_fps(tmp_path: Path) -> None:
    hw_path = tmp_path / "hardware.yaml"
    hw_path.write_text(
        "fictrac:\n"
        "  target_fps: 200\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="single source of truth"):
        load_run_protocol_config(None, hardware_path=hw_path)


def test_load_run_protocol_config_rejects_experiment_hardware_overrides(tmp_path: Path) -> None:
    cfg_path = tmp_path / "experiment_config.yaml"
    cfg_path.write_text(
        "teensy_port: COM7\n"
        "fictrac_config: C:/deprecated/config_camera.txt\n"
        "fictrac_bin: C:/deprecated/fictrac.exe\n"
        "fictrac_console_out: deprecated.txt\n"
        "fictrac_first_frame_timeout_ms: 0\n"
        "fictrac_startup_timeout_s: 90\n"
        "fictrac_timeout_s: 5\n"
        "save_camera_raw_video: true\n"
        "other_camera_timeout_ms: 99\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="single source of truth"):
        load_run_protocol_config(cfg_path, hardware_path=tmp_path / "hardware.yaml")


def test_resolve_fictrac_config_path_rejects_noncanonical_path(tmp_path: Path) -> None:
    hardware_path = tmp_path / "hardware.yaml"
    hardware_path.write_text("fictrac:\n  config: config_camera.txt\n", encoding="utf-8")

    with pytest.raises(ValueError, match="FicTrac config override is not allowed"):
        resolve_fictrac_config_path("alt_camera.txt", hardware_path=hardware_path)


def test_load_run_protocol_config_rejects_non_mapping_yaml(tmp_path: Path) -> None:
    cfg_path = tmp_path / "experiment_config.yaml"
    cfg_path.write_text("- not\n- a mapping\n", encoding="utf-8")

    with pytest.raises(ValueError, match="must contain a mapping"):
        load_run_protocol_config(cfg_path, hardware_path=tmp_path / "hardware.yaml")


def test_count_rising_edges_matches_camera_pulses() -> None:
    waveform = np.array([False, True, True, False, True, False, False, True], dtype=np.bool_)
    assert _count_rising_edges(waveform) == 3


def test_safe_stop_task_stops_without_raising() -> None:
    class DummyTask:
        def __init__(self) -> None:
            self.stopped = False

        def stop(self) -> None:
            self.stopped = True

    class DummyLogger:
        def debug(self, *args, **kwargs) -> None:
            return None

    task = DummyTask()
    _safe_stop_task(task, DummyLogger(), "dummy")
    assert task.stopped is True


def test_safe_stop_task_ignores_stop_errors() -> None:
    class DummyTask:
        def stop(self) -> None:
            raise RuntimeError("boom")

    class DummyLogger:
        def debug(self, *args, **kwargs) -> None:
            return None

    _safe_stop_task(DummyTask(), DummyLogger(), "dummy")


def test_stop_fictrac_skips_join_for_unstarted_thread(monkeypatch: pytest.MonkeyPatch) -> None:
    class DummyDriver:
        def __init__(self) -> None:
            self.stop_requested = False

        def request_stop(self) -> None:
            self.stop_requested = True

    class DummyLogger:
        def __init__(self) -> None:
            self.info_messages: list[str] = []
            self.warning_messages: list[str] = []

        def info(self, message: str, *args, **kwargs) -> None:
            self.info_messages.append(message % args if args else message)

        def warning(self, message: str, *args, **kwargs) -> None:
            self.warning_messages.append(message % args if args else message)

    callback = ExperimentCallback()
    driver = DummyDriver()
    logger = DummyLogger()
    thread = threading.Thread(target=lambda: None, name="FicTracTest")
    reset_calls: list[int] = []

    monkeypatch.setattr(
        "multibios.run_protocol._reset_fictrac_camera_external",
        lambda *, camera_index, logger: reset_calls.append(camera_index),
    )

    _stop_fictrac(
        fictrac_driver=driver,
        fictrac_callback=callback,
        fictrac_thread=thread,
        fictrac_camera_index=0,
        logger=logger,
    )

    assert driver.stop_requested is False
    assert callback._stop.is_set() is False
    assert reset_calls == [0]
    assert any("never started" in message for message in logger.info_messages)


def test_compile_control_plan_expands_states_and_windows() -> None:
    protocol = {
        "protocol": {
            "timing": {
                "seed": 7,
            }
        },
        "sequence": [
            {
                "phase": "test",
                "duration": 100,
                "times": 2,
                "actions": [
                    {"device": "olfactometer.left", "state": "ODOR1,ODOR2", "timing": 10},
                    {"device": "switch_valve.left", "state": "ODOR", "timing": 20},
                    {"device": "triggers.camera_continuous", "state": True, "timing": 0},
                    {"device": "triggers.camera_continuous", "state": False, "timing": 80},
                    {"device": "triggers.microscope", "state": True, "timing": 30},
                ],
            }
        ],
    }

    plan = compile_control_plan(protocol)

    assert plan.seed == 7
    assert plan.total_duration_ms == 200.0
    assert plan.camera_windows_ms == [(0.0, 80.0)]
    assert plan.microscope_times_ms == [30.0, 130.0]
    assert [event.state for event in plan.timeline if event.action == "olfactometer"] == ["ODOR1", "ODOR2"]


def test_protocol_numeric_timing_expressions_compile_in_plan_and_schema() -> None:
    protocol = {
        "_trial_baseline_ms": 10,
        "_odor_duration_ms": 20,
        "_trial_recovery_ms": 30,
        "_trial_end_ms": "_trial_baseline_ms + _odor_duration_ms + _trial_recovery_ms",
        "protocol": {
            "timing": {
                "sample_rate": 1000,
                "camera_interval": 0.0,
                "camera_pulse_duration": 1.0,
            }
        },
        "sequence": [
            {
                "phase": "expr",
                "duration": "_trial_end_ms + 40",
                "actions": [
                    {"device": "switch_valve.left", "state": "ODOR", "timing": "_trial_baseline_ms"},
                    {
                        "device": "switch_valve.left",
                        "state": "CLEAN",
                        "timing": "_trial_baseline_ms + _odor_duration_ms",
                    },
                    {"device": "triggers.microscope", "state": True, "timing": "_trial_end_ms"},
                ],
            }
        ],
    }

    plan = compile_control_plan(protocol)

    assert plan.total_duration_ms == 100.0
    assert plan.microscope_times_ms == [60.0]
    assert [
        (event.device, event.state, event.time_ms)
        for event in plan.timeline
        if event.action == "switch_valve"
    ] == [
        ("switch_valve.left", "ODOR", 10.0),
        ("switch_valve.left", "CLEAN", 30.0),
    ]

    class DummyHardware:
        do_lines = {
            "TRIG_MICRO": "Dev1/port0/line0",
            "OLFACTOMETER_LEFT_S0": "Dev1/port0/line1",
            "OLFACTOMETER_LEFT_S1": "Dev1/port0/line2",
            "OLFACTOMETER_LEFT_S2": "Dev1/port0/line3",
            "SWITCHVALVE_LEFT_S": "Dev1/port0/line4",
            "SWITCHVALVE_LEFT_LOAD_REQ": "Dev1/port0/line5",
            "RCK_SWITCHVALVE_LEFT": "Dev1/port0/line6",
            "OLFACTOMETER_RIGHT_S0": "Dev1/port0/line7",
            "OLFACTOMETER_RIGHT_S1": "Dev1/port0/line8",
            "OLFACTOMETER_RIGHT_S2": "Dev1/port0/line9",
            "SWITCHVALVE_RIGHT_S": "Dev1/port0/line10",
        }
        ao_channels: dict[str, str] = {}

    compiler = ProtocolCompiler(DummyHardware(), TimingConfig(sample_rate=1000, camera_interval_ms=0.0))
    compiler.compile_from_yaml(protocol)

    assert compiler.N == 100
    assert compiler.do is not None
    assert _count_rising_edges(compiler.do[compiler.line_to_idx["TRIG_MICRO"]]) == 1


def test_protocol_compiler_allows_submillisecond_camera_intervals() -> None:
    class DummyHardware:
        do_lines = {
            "TRIG_CAMERA": "Dev1/port0/line31",
            "OLFACTOMETER_LEFT_S0": "Dev1/port0/line0",
            "OLFACTOMETER_LEFT_S1": "Dev1/port0/line1",
            "OLFACTOMETER_LEFT_S2": "Dev1/port0/line2",
            "OLFACTOMETER_RIGHT_S0": "Dev1/port0/line4",
            "OLFACTOMETER_RIGHT_S1": "Dev1/port0/line5",
            "OLFACTOMETER_RIGHT_S2": "Dev1/port0/line6",
            "SWITCHVALVE_LEFT_S": "Dev1/port0/line3",
            "SWITCHVALVE_RIGHT_S": "Dev1/port0/line7",
        }
        ao_channels: dict[str, str] = {}

    compiler = ProtocolCompiler(
        DummyHardware(),
        TimingConfig(sample_rate=2000, camera_interval_ms=5.5, camera_pulse_ms=1.0),
    )
    compiler.compile_from_yaml(
        {
            "protocol": {
                "timing": {
                    "sample_rate": 2000,
                    "camera_interval": 5.5,
                    "camera_pulse_duration": 1.0,
                }
            },
            "sequence": [
                {
                    "phase": "test",
                    "duration": 30,
                    "actions": [
                        {"device": "triggers.camera_continuous", "state": True, "timing": 0},
                        {"device": "triggers.camera_continuous", "state": False, "timing": 22},
                    ],
                }
            ],
        }
    )

    assert compiler.do is not None
    trace = compiler.do[compiler.line_to_idx["TRIG_CAMERA"]]
    assert _count_rising_edges(trace) == 4
    bool_trace = trace.astype(bool)
    rising_edges = np.flatnonzero(np.concatenate(([bool_trace[0]], bool_trace[1:] & ~bool_trace[:-1])))
    assert rising_edges.tolist() == [0, 11, 22, 33]


def test_scale_compiled_mfc_ao_from_slpm_to_volts_uses_hardware_full_scale() -> None:
    class DummyHardware:
        do_lines = {
            "OLFACTOMETER_LEFT_S0": "Dev1/port0/line0",
            "OLFACTOMETER_LEFT_S1": "Dev1/port0/line1",
            "OLFACTOMETER_LEFT_S2": "Dev1/port0/line2",
            "SWITCHVALVE_LEFT_S": "Dev1/port0/line3",
            "OLFACTOMETER_RIGHT_S0": "Dev1/port0/line4",
            "OLFACTOMETER_RIGHT_S1": "Dev1/port0/line5",
            "OLFACTOMETER_RIGHT_S2": "Dev1/port0/line6",
            "SWITCHVALVE_RIGHT_S": "Dev1/port0/line7",
        }
        ao_channels = {"mfc.air_left_setpoint": "Dev1/ao0"}

    compiler = ProtocolCompiler(DummyHardware(), TimingConfig(sample_rate=1000))
    compiler.compile_from_yaml(
        {
            "protocol": {"timing": {"sample_rate": 1000}},
            "sequence": [
                {
                    "phase": "boot",
                    "duration": 10,
                    "actions": [
                        {"device": "mfc.air_left_setpoint", "value": 0.5, "timing": 0},
                    ],
                }
            ],
        }
    )

    cfg = RunProtocolConfig(
        mfc_mode="analog",
        mfc_value_units="slpm",
        mfc_analog_full_scale_slpm={"mfc.air_left_setpoint": 1.0},
    )

    _scale_compiled_mfc_ao_from_slpm_to_volts(compiler, cfg)

    assert compiler.ao is not None
    np.testing.assert_allclose(compiler.ao[0], np.full(10, 2.5, dtype=np.float32))


def test_postprocess_triggered_camera_recording_marks_no_drop_and_conversion(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    bin_path = run_dir / "blackfly_cam0_frames.bin"
    csv_path = run_dir / "blackfly_cam0_frame_index.csv"
    manifest_path = run_dir / "blackfly_cam0_manifest.json"

    frames = np.arange(32, dtype=np.uint8).reshape(2, 4, 4)
    bin_path.write_bytes(frames.tobytes())
    csv_path.write_text(
        "frame_index,frame_id,camera_timestamp,host_timestamp_ns\n"
        "0,10,1000000000,1\n"
        "1,11,1033333333,2\n",
        encoding="utf-8",
    )
    manifest = {
        "camera_index": 0,
        "frame_bin_path": str(bin_path),
        "frame_index_path": str(csv_path),
        "manifest_path": str(manifest_path),
        "saved_frames": 2,
        "frame_width": 4,
        "frame_height": 4,
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    processed = postprocess_triggered_camera_recording(
        manifest,
        expected_frame_count=2,
        nominal_fps=30.0,
        convert_to_lossless_mkv=True,
    )

    assert processed["no_dropped_frames"] is True
    assert processed["analysis"]["missing_frames_vs_expected"] == 0
    assert processed["lossless_video"]["path"].endswith((".mkv", ".avi"))


def test_postprocess_triggered_camera_recording_reconstructs_chunked_stream(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    chunk0_path = run_dir / "blackfly_cam0_frames-chunk000000.bin"
    chunk1_path = run_dir / "blackfly_cam0_frames-chunk000001.bin"
    csv_path = run_dir / "blackfly_cam0_frame_index.csv"
    manifest_path = run_dir / "blackfly_cam0_manifest.json"

    frame0 = np.zeros((4, 4), dtype=np.uint8)
    frame1 = np.full((4, 4), 60, dtype=np.uint8)
    frame2 = np.full((4, 4), 120, dtype=np.uint8)
    chunk0_path.write_bytes(frame0.tobytes() + frame1.tobytes())
    chunk1_path.write_bytes(frame2.tobytes())
    csv_path.write_text(
        "frame_index,frame_id,camera_timestamp,host_timestamp_ns,chunk_index,chunk_frame_index\n"
        "0,10,1000000000,1,0,0\n"
        "1,11,1033333333,2,0,1\n"
        "2,12,1066666666,3,1,0\n",
        encoding="utf-8",
    )
    manifest = {
        "camera_index": 0,
        "format": "raw-mono8-chunks",
        "frame_bin_path": str(chunk0_path),
        "frame_index_path": str(csv_path),
        "manifest_path": str(manifest_path),
        "chunk_paths": [str(chunk0_path), str(chunk1_path)],
        "saved_frames": 3,
        "frame_width": 4,
        "frame_height": 4,
        "channels": 1,
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    processed = postprocess_triggered_camera_recording(
        manifest,
        expected_frame_count=3,
        nominal_fps=30.0,
        convert_to_lossless_mkv=True,
    )

    assert processed["saved_frames"] == 3
    assert processed["analysis"]["chunk_frames_detected"] == 3
    assert processed["analysis"]["frames_saved"] == 3
    assert processed["no_dropped_frames"] is True
    assert Path(processed["lossless_video"]["path"]).exists()


def test_postprocess_triggered_camera_recording_prefers_chunk_frames_over_short_csv(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    chunk_path = run_dir / "blackfly_cam0_frames-chunk000000.bin"
    csv_path = run_dir / "blackfly_cam0_frame_index.csv"
    manifest_path = run_dir / "blackfly_cam0_manifest.json"

    frames = np.arange(48, dtype=np.uint8).reshape(3, 4, 4)
    chunk_path.write_bytes(frames.tobytes())
    csv_path.write_text(
        "frame_index,frame_id,camera_timestamp,host_timestamp_ns,chunk_index,chunk_frame_index\n"
        "0,10,1000000000,1,0,0\n"
        "1,11,1033333333,2,0,1\n",
        encoding="utf-8",
    )
    manifest = {
        "camera_index": 0,
        "format": "raw-mono8-chunks",
        "frame_bin_path": str(chunk_path),
        "frame_index_path": str(csv_path),
        "manifest_path": str(manifest_path),
        "chunk_paths": [str(chunk_path)],
        "saved_frames": 2,
        "frame_width": 4,
        "frame_height": 4,
        "channels": 1,
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    processed = postprocess_triggered_camera_recording(
        manifest,
        expected_frame_count=3,
        nominal_fps=30.0,
        convert_to_lossless_mkv=False,
    )

    assert processed["saved_frames"] == 3
    assert processed["analysis"]["indexed_frames"] == 2
    assert processed["analysis"]["chunk_frames_detected"] == 3
    assert processed["analysis"]["missing_frames_vs_expected"] == 0
    assert processed["no_dropped_frames"] is True


def test_postprocess_triggered_camera_recording_retries_codec_when_output_is_unreadable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    chunk_path = run_dir / "blackfly_cam0_frames-chunk000000.bin"
    csv_path = run_dir / "blackfly_cam0_frame_index.csv"
    manifest_path = run_dir / "blackfly_cam0_manifest.json"

    frames = np.arange(48, dtype=np.uint8).reshape(3, 4, 4)
    chunk_path.write_bytes(frames.tobytes())
    csv_path.write_text(
        "frame_index,frame_id,camera_timestamp,host_timestamp_ns,chunk_index,chunk_frame_index\n"
        "0,10,1000000000,1,0,0\n"
        "1,11,1033333333,2,0,1\n"
        "2,12,1066666666,3,0,2\n",
        encoding="utf-8",
    )
    manifest = {
        "camera_index": 0,
        "format": "raw-mono8-chunks",
        "frame_bin_path": str(chunk_path),
        "frame_index_path": str(csv_path),
        "manifest_path": str(manifest_path),
        "chunk_paths": [str(chunk_path)],
        "saved_frames": 3,
        "frame_width": 4,
        "frame_height": 4,
        "channels": 1,
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    attempted_paths: list[str] = []
    original_validator = triggered_camera_record_module.cv2.VideoCapture

    def fake_video_capture(path: str):
        attempted_paths.append(path)
        capture = original_validator(path)
        if path.endswith(".avi"):
            class BrokenCapture:
                def __init__(self, inner):
                    self._inner = inner

                def isOpened(self):
                    return self._inner.isOpened()

                def get(self, prop_id):
                    if prop_id == triggered_camera_record_module.cv2.CAP_PROP_FRAME_COUNT:
                        return 1
                    return self._inner.get(prop_id)

                def set(self, prop_id, value):
                    return self._inner.set(prop_id, value)

                def read(self):
                    return False, None

                def release(self):
                    self._inner.release()

            return BrokenCapture(capture)
        return capture

    monkeypatch.setattr(triggered_camera_record_module.cv2, "VideoCapture", fake_video_capture)

    processed = postprocess_triggered_camera_recording(
        manifest,
        expected_frame_count=3,
        nominal_fps=30.0,
        convert_to_lossless_mkv=True,
    )

    assert processed["lossless_video"]["path"].endswith(".mkv")
    assert any(path.endswith(".avi") for path in attempted_paths)


def test_postprocess_fictrac_raw_recording_reconstructs_lossless_video(tmp_path: Path) -> None:
    chunk_path = tmp_path / "fictrac-raw-2026-05-01_00-00-00-chunk000000.bin"
    frame0 = np.zeros((4, 6, 3), dtype=np.uint8)
    frame1 = np.full((4, 6, 3), 120, dtype=np.uint8)
    with open(chunk_path, "wb") as fh:
        fh.write(frame0.tobytes(order="C"))
        fh.write(frame1.tobytes(order="C"))

    index_path = tmp_path / "fictrac-raw-2026-05-01_00-00-00-index.csv"
    index_path.write_text(
        "frame_index,log_frame,chunk_index,chunk_frame_index\n"
        "0,0,0,0\n"
        "1,1,0,1\n",
        encoding="utf-8",
    )
    manifest_path = tmp_path / "fictrac-raw-2026-05-01_00-00-00.json"
    manifest_path.write_text(
        json.dumps(
            {
                "format": "raw-bgr8-chunks",
                "frame_width": 6,
                "frame_height": 4,
                "channels": 3,
                "dtype": "uint8",
                "fps": 200.0,
                "saved_frames": 2,
                "frame_index_path": str(index_path),
                "manifest_path": str(manifest_path),
                "chunk_paths": [str(chunk_path)],
            }
        ),
        encoding="utf-8",
    )

    summary = postprocess_fictrac_raw_recording(
        run_dir=tmp_path,
        runtime_info={
            "fictrac_camera_index": 0,
            "save_raw": True,
            "video_codec": "raw",
            "camera_fps": 200.0,
            "output_base": str(tmp_path / "fictrac"),
        },
        frame_count=2,
        expected_frame_count=2,
        legacy_raw_videos=[],
        legacy_saved_raw_frames=None,
    )

    assert summary["saved_raw_frames"] == 2
    assert summary["no_dropped_frames"] is True
    assert summary["lossless_video"] is not None
    assert summary["lossless_video"]["frames_written"] == 2
    assert Path(summary["lossless_video"]["path"]).exists()


def test_postprocess_fictrac_raw_recording_uses_configured_output_directory(tmp_path: Path) -> None:
    output_dir = tmp_path / "recorded" / "tracking" / "fictrac"
    output_dir.mkdir(parents=True)

    chunk_path = output_dir / "fictrac-raw-2026-05-01_00-00-00-chunk000000.bin"
    frame0 = np.zeros((4, 6, 3), dtype=np.uint8)
    frame1 = np.full((4, 6, 3), 120, dtype=np.uint8)
    with open(chunk_path, "wb") as fh:
        fh.write(frame0.tobytes(order="C"))
        fh.write(frame1.tobytes(order="C"))

    index_path = output_dir / "fictrac-raw-2026-05-01_00-00-00-index.csv"
    index_path.write_text(
        "frame_index,log_frame,chunk_index,chunk_frame_index\n"
        "0,0,0,0\n"
        "1,1,0,1\n",
        encoding="utf-8",
    )
    manifest_path = output_dir / "fictrac-raw-2026-05-01_00-00-00.json"
    manifest_path.write_text(
        json.dumps(
            {
                "format": "raw-bgr8-chunks",
                "frame_width": 6,
                "frame_height": 4,
                "channels": 3,
                "dtype": "uint8",
                "fps": 200.0,
                "saved_frames": 2,
                "frame_index_path": str(index_path),
                "manifest_path": str(manifest_path),
                "chunk_paths": [str(chunk_path)],
            }
        ),
        encoding="utf-8",
    )

    summary = postprocess_fictrac_raw_recording(
        run_dir=tmp_path,
        runtime_info={
            "fictrac_camera_index": 0,
            "save_raw": True,
            "video_codec": "raw",
            "camera_fps": 200.0,
            "output_base": str(output_dir / "fictrac"),
        },
        frame_count=2,
        expected_frame_count=2,
        legacy_raw_videos=[],
        legacy_saved_raw_frames=None,
    )

    assert summary["raw_stream_manifest"] == str(manifest_path)
    assert summary["lossless_video"] is not None
    assert Path(summary["lossless_video"]["path"]).parent == output_dir


def test_finalize_raw_chunk_retention_deletes_chunks_after_parity(tmp_path: Path) -> None:
    run_dir = tmp_path / "2026-05-01_12-16-16"
    run_dir.mkdir()

    (run_dir / "digital_edges.csv").write_text(
        "line,edge_type,sample_idx,time_ms\n"
        "TRIG_CAMERA,rising,1,0.5\n"
        "TRIG_CAMERA,falling,2,1.0\n"
        "TRIG_CAMERA,rising,3,1.5\n",
        encoding="utf-8",
    )
    (run_dir / "fictrac_driver_diagnostics.json").write_text(
        json.dumps({"frame_cnt": 2, "final_returncode": 0, "stop_method": "ctrl_break"}),
        encoding="utf-8",
    )
    np.savez_compressed(run_dir / "fictrac_frames.npz", frames=np.zeros(2, dtype=FICTRAC_FRAME_DTYPE))

    fictrac_chunk = run_dir / "fictrac-raw-chunk000000.bin"
    fictrac_chunk.write_bytes(b"1234")
    blackfly_chunk = run_dir / "blackfly_cam1_frames-chunk000000.bin"
    blackfly_chunk.write_bytes(b"5678")

    fictrac_manifest = run_dir / "fictrac-raw.json"
    fictrac_manifest.write_text(json.dumps({"chunk_paths": [str(fictrac_chunk)]}), encoding="utf-8")
    blackfly_manifest = run_dir / "blackfly_cam1_manifest.json"
    blackfly_manifest.write_text(
        json.dumps({"chunk_paths": [str(blackfly_chunk)], "frame_bin_path": str(blackfly_chunk)}),
        encoding="utf-8",
    )

    frame = np.zeros((4, 4, 3), dtype=np.uint8)
    fictrac_video = run_dir / "fictrac-lossless.avi"
    blackfly_video = run_dir / "blackfly-lossless.avi"
    for video_path in (fictrac_video, blackfly_video):
        writer = cv2.VideoWriter(
            str(video_path),
            cv2.VideoWriter_fourcc(*"FFV1"),
            30.0,
            (4, 4),
            True,
        )
        assert writer.isOpened()
        writer.write(frame)
        writer.write(frame)
        writer.release()

    fictrac_recording = {
        "saved_raw_frames": 2,
        "expected_frames": 2,
        "raw_stream_chunks": [str(fictrac_chunk)],
        "raw_stream_manifest": str(fictrac_manifest),
        "lossless_video": {"path": str(fictrac_video), "frames_written": 2},
    }
    blackfly_recording = {
        "saved_frames": 2,
        "expected_frame_count": 2,
        "chunk_paths": [str(blackfly_chunk)],
        "frame_bin_path": str(blackfly_chunk),
        "manifest_path": str(blackfly_manifest),
        "lossless_video": {"path": str(blackfly_video), "frame_count": 2},
    }
    (run_dir / "fictrac_camera_recording.json").write_text(json.dumps(fictrac_recording), encoding="utf-8")
    (run_dir / "blackfly_recording.json").write_text(json.dumps(blackfly_recording), encoding="utf-8")

    parity_path = run_dir / "derived" / "validation" / "parity_audit.json"
    parity_path.parent.mkdir(parents=True)
    parity_summary, parity_path = _write_parity_summary(run_dir, parity_path)
    fictrac_updated, blackfly_updated = _finalize_raw_chunk_retention(
        run_dir=run_dir,
        policy="delete_after_parity",
        parity_summary=parity_summary,
        parity_path=parity_path,
        fictrac_recording=fictrac_recording,
        blackfly_recording=blackfly_recording,
        logger=logging.getLogger(__name__),
    )

    assert fictrac_updated is not None
    assert blackfly_updated is not None
    assert fictrac_chunk.exists() is False
    assert blackfly_chunk.exists() is False
    assert fictrac_updated["raw_chunks_retained"] is False
    assert blackfly_updated["raw_chunks_retained"] is False
    assert fictrac_updated["raw_stream_chunks"] == []
    assert blackfly_updated["chunk_paths"] == []
    assert blackfly_updated["frame_bin_path"] is None
    assert fictrac_updated["raw_chunk_cleanup"]["applied"] is True
    assert blackfly_updated["raw_chunk_cleanup"]["applied"] is True
    assert fictrac_updated["raw_chunk_cleanup"]["deleted_chunk_bytes"] == 4
    assert blackfly_updated["raw_chunk_cleanup"]["deleted_chunk_bytes"] == 4


def test_finalize_raw_chunk_retention_keeps_chunks_on_parity_mismatch(tmp_path: Path) -> None:
    run_dir = tmp_path / "2026-05-01_12-16-17"
    run_dir.mkdir()

    (run_dir / "digital_edges.csv").write_text(
        "line,edge_type,sample_idx,time_ms\n"
        "TRIG_CAMERA,rising,1,0.5\n"
        "TRIG_CAMERA,falling,2,1.0\n"
        "TRIG_CAMERA,rising,3,1.5\n",
        encoding="utf-8",
    )
    (run_dir / "fictrac_driver_diagnostics.json").write_text(
        json.dumps({"frame_cnt": 2, "final_returncode": 0, "stop_method": "ctrl_break"}),
        encoding="utf-8",
    )
    np.savez_compressed(run_dir / "fictrac_frames.npz", frames=np.zeros(2, dtype=FICTRAC_FRAME_DTYPE))

    fictrac_chunk = run_dir / "fictrac-raw-chunk000000.bin"
    fictrac_chunk.write_bytes(b"1234")
    blackfly_chunk = run_dir / "blackfly_cam1_frames-chunk000000.bin"
    blackfly_chunk.write_bytes(b"5678")

    fictrac_manifest = run_dir / "fictrac-raw.json"
    fictrac_manifest.write_text(json.dumps({"chunk_paths": [str(fictrac_chunk)]}), encoding="utf-8")
    blackfly_manifest = run_dir / "blackfly_cam1_manifest.json"
    blackfly_manifest.write_text(
        json.dumps({"chunk_paths": [str(blackfly_chunk)], "frame_bin_path": str(blackfly_chunk)}),
        encoding="utf-8",
    )

    frame = np.zeros((4, 4, 3), dtype=np.uint8)
    fictrac_video = run_dir / "fictrac-lossless.avi"
    blackfly_video = run_dir / "blackfly-lossless.avi"
    for video_path in (fictrac_video, blackfly_video):
        writer = cv2.VideoWriter(
            str(video_path),
            cv2.VideoWriter_fourcc(*"FFV1"),
            30.0,
            (4, 4),
            True,
        )
        assert writer.isOpened()
        writer.write(frame)
        writer.write(frame)
        writer.release()

    fictrac_recording = {
        "saved_raw_frames": 1,
        "expected_frames": 2,
        "raw_stream_chunks": [str(fictrac_chunk)],
        "raw_stream_manifest": str(fictrac_manifest),
        "lossless_video": {"path": str(fictrac_video), "frames_written": 2},
    }
    blackfly_recording = {
        "saved_frames": 2,
        "expected_frame_count": 2,
        "chunk_paths": [str(blackfly_chunk)],
        "frame_bin_path": str(blackfly_chunk),
        "manifest_path": str(blackfly_manifest),
        "lossless_video": {"path": str(blackfly_video), "frame_count": 2},
    }
    (run_dir / "fictrac_camera_recording.json").write_text(json.dumps(fictrac_recording), encoding="utf-8")
    (run_dir / "blackfly_recording.json").write_text(json.dumps(blackfly_recording), encoding="utf-8")

    parity_path = run_dir / "derived" / "validation" / "parity_audit.json"
    parity_path.parent.mkdir(parents=True)
    parity_summary, parity_path = _write_parity_summary(run_dir, parity_path)
    fictrac_updated, blackfly_updated = _finalize_raw_chunk_retention(
        run_dir=run_dir,
        policy="delete_after_parity",
        parity_summary=parity_summary,
        parity_path=parity_path,
        fictrac_recording=fictrac_recording,
        blackfly_recording=blackfly_recording,
        logger=logging.getLogger(__name__),
    )

    assert fictrac_updated is not None
    assert blackfly_updated is not None
    assert fictrac_chunk.exists() is True
    assert blackfly_chunk.exists() is True
    assert fictrac_updated["raw_chunks_retained"] is True
    assert blackfly_updated["raw_chunks_retained"] is True
    assert fictrac_updated["raw_chunk_cleanup"]["applied"] is False
    assert blackfly_updated["raw_chunk_cleanup"]["applied"] is False
    assert fictrac_updated["raw_chunk_cleanup"]["reason"] == "parity_mismatch_fictrac_saved_raw_frames"
    assert blackfly_updated["raw_chunk_cleanup"]["reason"] == "parity_mismatch_fictrac_saved_raw_frames"


def test_write_parity_summary_uses_in_memory_recordings_before_structured_files_exist(tmp_path: Path) -> None:
    run_dir = tmp_path / "2026-05-05_10-12-49"
    (run_dir / "planned" / "daq" / "digital_outputs").mkdir(parents=True)
    (run_dir / "recorded" / "tracking" / "fictrac").mkdir(parents=True)

    (run_dir / "planned" / "daq" / "digital_outputs" / "edge_table.csv").write_text(
        "line,edge_type,sample_idx,time_ms\n"
        "TRIG_CAMERA,rising,1,0.5\n"
        "TRIG_CAMERA,rising,2,1.0\n",
        encoding="utf-8",
    )
    (run_dir / "recorded" / "tracking" / "fictrac" / "fictrac_driver_diagnostics.json").write_text(
        json.dumps({"frame_cnt": 2, "final_returncode": 0}),
        encoding="utf-8",
    )
    np.savez_compressed(
        run_dir / "recorded" / "tracking" / "fictrac" / "frame_series.npz",
        frames=np.zeros(2, dtype=FICTRAC_FRAME_DTYPE),
    )

    parity_path = run_dir / "derived" / "validation" / "parity_audit.json"
    parity_path.parent.mkdir(parents=True)

    summary, _ = _write_parity_summary(
        run_dir,
        parity_path,
        fictrac_recording={"saved_raw_frames": 2, "expected_frames": 2},
        blackfly_recording={"saved_frames": 2, "expected_frame_count": 2},
    )

    assert summary["counts"] == {
        "trigger_rising_edges": 2,
        "fictrac_saved_raw_frames": 2,
        "fictrac_udp_frame_cnt": 2,
        "fictrac_callback_frames": 2,
        "second_camera_saved_frames": 2,
    }
    assert summary["exact_trigger_match"] is True
    written = json.loads(parity_path.read_text(encoding="utf-8"))
    assert written["counts"]["fictrac_saved_raw_frames"] == 2
    assert written["counts"]["second_camera_saved_frames"] == 2


def test_canonicalize_fictrac_runtime_info_matches_relocated_runtime_config(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    layout = RunDatasetLayout(run_dir)
    layout.ensure_directories()

    source_config = run_dir / "recorded" / "tracking" / "fictrac" / "fictrac_runtime_config.txt"
    source_config.parent.mkdir(parents=True, exist_ok=True)
    source_config.write_text("src_fn : 1\n", encoding="utf-8")

    target_config = layout.fictrac_tracking_runtime_config_path
    source_config.replace(target_config)

    runtime_info = _canonicalize_fictrac_runtime_info(
        layout=layout,
        run_dir=run_dir,
        runtime_info={"runtime_config": str(source_config), "output_base": str(run_dir / "recorded" / "tracking" / "fictrac" / "fictrac")},
    )

    assert target_config.exists()
    assert runtime_info["runtime_config"] == "recorded/tracking/fictrac/runtime_config.txt"


def test_postprocess_fictrac_raw_recording_reconstructs_mono_chunks(tmp_path: Path) -> None:
    chunk_path = tmp_path / "fictrac-raw-2026-05-01_00-00-00-chunk000000.bin"
    frame0 = np.zeros((4, 6), dtype=np.uint8)
    frame1 = np.full((4, 6), 120, dtype=np.uint8)
    with open(chunk_path, "wb") as fh:
        fh.write(frame0.tobytes(order="C"))
        fh.write(frame1.tobytes(order="C"))

    index_path = tmp_path / "fictrac-raw-2026-05-01_00-00-00-index.csv"
    index_path.write_text(
        "frame_index,log_frame,chunk_index,chunk_frame_index\n"
        "0,0,0,0\n"
        "1,1,0,1\n",
        encoding="utf-8",
    )
    manifest_path = tmp_path / "fictrac-raw-2026-05-01_00-00-00.json"
    manifest_path.write_text(
        json.dumps(
            {
                "format": "raw-mono8-chunks",
                "frame_width": 6,
                "frame_height": 4,
                "channels": 1,
                "dtype": "uint8",
                "fps": 200.0,
                "saved_frames": 2,
                "frame_index_path": str(index_path),
                "manifest_path": str(manifest_path),
                "chunk_paths": [str(chunk_path)],
            }
        ),
        encoding="utf-8",
    )

    summary = postprocess_fictrac_raw_recording(
        run_dir=tmp_path,
        runtime_info={
            "fictrac_camera_index": 0,
            "save_raw": True,
            "video_codec": "raw",
            "camera_fps": 200.0,
            "output_base": str(tmp_path / "fictrac"),
        },
        frame_count=2,
        expected_frame_count=2,
        legacy_raw_videos=[],
        legacy_saved_raw_frames=None,
    )

    assert summary["saved_raw_frames"] == 2
    assert summary["raw_stream_format"] == "raw-mono8-chunks"
    assert summary["lossless_video"] is not None
    assert summary["lossless_video"]["frames_written"] == 2
    assert Path(summary["lossless_video"]["path"]).exists()


def test_postprocess_fictrac_raw_recording_ignores_incomplete_trailing_csv_row(tmp_path: Path) -> None:
    chunk_path = tmp_path / "fictrac-raw-2026-05-01_00-00-00-chunk000000.bin"
    frame0 = np.zeros((4, 6, 3), dtype=np.uint8)
    frame1 = np.full((4, 6, 3), 120, dtype=np.uint8)
    with open(chunk_path, "wb") as fh:
        fh.write(frame0.tobytes(order="C"))
        fh.write(frame1.tobytes(order="C"))

    index_path = tmp_path / "fictrac-raw-2026-05-01_00-00-00-index.csv"
    index_path.write_text(
        "frame_index,log_frame,chunk_index,chunk_frame_index\n"
        "0,0,0,0\n"
        "1,1,0,1\n"
        "2,2,\n",
        encoding="utf-8",
    )
    manifest_path = tmp_path / "fictrac-raw-2026-05-01_00-00-00.json"
    manifest_path.write_text(
        json.dumps(
            {
                "format": "raw-bgr8-chunks",
                "frame_width": 6,
                "frame_height": 4,
                "channels": 3,
                "dtype": "uint8",
                "fps": 200.0,
                "saved_frames": 2,
                "frame_index_path": str(index_path),
                "manifest_path": str(manifest_path),
                "chunk_paths": [str(chunk_path)],
            }
        ),
        encoding="utf-8",
    )

    summary = postprocess_fictrac_raw_recording(
        run_dir=tmp_path,
        runtime_info={
            "fictrac_camera_index": 0,
            "save_raw": True,
            "video_codec": "raw",
            "camera_fps": 200.0,
            "output_base": str(tmp_path / "fictrac"),
        },
        frame_count=2,
        expected_frame_count=2,
        legacy_raw_videos=[],
        legacy_saved_raw_frames=None,
    )

    assert summary["saved_raw_frames"] == 2
    assert summary["skipped_log_frames"] == 0
    assert summary["lossless_video"] is not None


def test_postprocess_fictrac_raw_recording_ignores_partial_chunk_tail(tmp_path: Path) -> None:
    chunk_path = tmp_path / "fictrac-raw-2026-05-01_00-00-00-chunk000000.bin"
    frame0 = np.zeros((4, 6, 3), dtype=np.uint8)
    frame1 = np.full((4, 6, 3), 120, dtype=np.uint8)
    with open(chunk_path, "wb") as fh:
        fh.write(frame0.tobytes(order="C"))
        fh.write(frame1.tobytes(order="C"))
        fh.write(b"partial")

    index_path = tmp_path / "fictrac-raw-2026-05-01_00-00-00-index.csv"
    index_path.write_text(
        "frame_index,log_frame,chunk_index,chunk_frame_index\n"
        "0,0,0,0\n"
        "1,1,0,1\n",
        encoding="utf-8",
    )
    manifest_path = tmp_path / "fictrac-raw-2026-05-01_00-00-00.json"
    manifest_path.write_text(
        json.dumps(
            {
                "format": "raw-bgr8-chunks",
                "frame_width": 6,
                "frame_height": 4,
                "channels": 3,
                "dtype": "uint8",
                "fps": 200.0,
                "saved_frames": 2,
                "frame_index_path": str(index_path),
                "manifest_path": str(manifest_path),
                "chunk_paths": [str(chunk_path)],
            }
        ),
        encoding="utf-8",
    )

    summary = postprocess_fictrac_raw_recording(
        run_dir=tmp_path,
        runtime_info={
            "fictrac_camera_index": 0,
            "save_raw": True,
            "video_codec": "raw",
            "camera_fps": 200.0,
            "output_base": str(tmp_path / "fictrac"),
        },
        frame_count=2,
        expected_frame_count=2,
        legacy_raw_videos=[],
        legacy_saved_raw_frames=None,
    )

    assert summary["saved_raw_frames"] == 2
    assert summary["lossless_video"] is not None
    assert summary["lossless_video"]["frames_written"] == 2


def test_postprocess_fictrac_raw_recording_prefers_full_chunk_frames_over_short_csv(tmp_path: Path) -> None:
    chunk_path = tmp_path / "fictrac-raw-2026-05-01_00-00-00-chunk000000.bin"
    frame0 = np.zeros((4, 6, 3), dtype=np.uint8)
    frame1 = np.full((4, 6, 3), 60, dtype=np.uint8)
    frame2 = np.full((4, 6, 3), 120, dtype=np.uint8)
    with open(chunk_path, "wb") as fh:
        fh.write(frame0.tobytes(order="C"))
        fh.write(frame1.tobytes(order="C"))
        fh.write(frame2.tobytes(order="C"))

    index_path = tmp_path / "fictrac-raw-2026-05-01_00-00-00-index.csv"
    index_path.write_text(
        "frame_index,log_frame,chunk_index,chunk_frame_index\n"
        "0,0,0,0\n"
        "1,1,0,1\n",
        encoding="utf-8",
    )
    manifest_path = tmp_path / "fictrac-raw-2026-05-01_00-00-00.json"
    manifest_path.write_text(
        json.dumps(
            {
                "format": "raw-bgr8-chunks",
                "frame_width": 6,
                "frame_height": 4,
                "channels": 3,
                "dtype": "uint8",
                "fps": 200.0,
                "saved_frames": 2,
                "frame_index_path": str(index_path),
                "manifest_path": str(manifest_path),
                "chunk_paths": [str(chunk_path)],
            }
        ),
        encoding="utf-8",
    )

    summary = postprocess_fictrac_raw_recording(
        run_dir=tmp_path,
        runtime_info={
            "fictrac_camera_index": 0,
            "save_raw": True,
            "video_codec": "raw",
            "camera_fps": 200.0,
            "output_base": str(tmp_path / "fictrac"),
        },
        frame_count=3,
        expected_frame_count=3,
        legacy_raw_videos=[],
        legacy_saved_raw_frames=None,
    )

    assert summary["saved_raw_frames"] == 3
    assert summary["actual_frames"] == 3
    assert summary["missing_frames_vs_expected"] == 0
    assert summary["no_dropped_frames"] is True
