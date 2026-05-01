from __future__ import annotations

import json
import threading
from pathlib import Path

import numpy as np
import pytest

from multibios.blackfly.triggered_camera_record import \
    postprocess_triggered_camera_recording
from multibios.fictrac_client import FicTracState
from multibios.fictrac_config import resolve_fictrac_config_path
from multibios.fictrac_raw_recording import postprocess_fictrac_raw_recording
from multibios.protocol.control_plan import compile_control_plan
from multibios.protocol.schema import ProtocolCompiler, TimingConfig
from multibios.run_protocol import (ExperimentCallback, _count_rising_edges,
                                    _prepare_fictrac_runtime_config,
                                    _safe_stop_task, _stop_fictrac,
                                    load_run_protocol_config)


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


def test_experiment_callback_reports_stop_requested() -> None:
    callback = ExperimentCallback()
    assert callback.stop_requested() is False
    callback.request_stop()
    assert callback.stop_requested() is True


def test_load_run_protocol_config_reads_hardware_owned_fields(tmp_path: Path) -> None:
    hw_path = tmp_path / "hardware.yaml"
    hw_path.write_text(
        "teensy:\n"
        "  port: COM9\n"
        "  baud: 230400\n"
        "  capture_serial: true\n"
        "fictrac:\n"
        "  camera_serial: 26021184\n"
        "camera_recording:\n"
        "  trigger_fps_hz: 200\n"
        "  trigger_pulse_ms: 1\n"
        "  save_fictrac_camera_video: true\n"
        "  save_second_camera_video: true\n"
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
        "  verify_no_dropped_frames: true\n"
        "  convert_second_camera_bin_to_lossless_mkv: false\n"
        "mfc:\n"
        "  mode: none\n"
        "  live_interval_s: 0\n"
        "daq:\n"
        "  latch_interval_ms: 12\n"
        "data_output:\n"
        "  data_dir: C:/data/runs\n"
        "  open_explorer: false\n"
        "  explorer_port: 9000\n"
        "blackfly_defaults:\n"
        "  exposure_us: 4500\n"
        "  roi_width: 400\n"
        "  roi_height: 400\n"
        "  binning: 1\n"
        "  gain_db: 3.5\n"
        "  gamma: 1.1\n",
        encoding="utf-8",
    )

    cfg = load_run_protocol_config(None, hardware_path=hw_path)
    assert cfg.teensy_port == "COM9"
    assert cfg.teensy_baud == 230400
    assert cfg.capture_teensy_serial is True
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
        "blackfly_defaults:\n"
        "  exposure_us: 4200\n"
        "  roi_width: 640\n"
        "  roi_height: 512\n"
        "  binning: 1\n"
        "  gain_db: 5.5\n"
        "  gamma: 0.9\n",
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
        "fictrac:\n"
        "  config: config_camera.txt\n"
        "  camera_serial: 26021184\n"
        "  bin: C:/rig/fictrac-spinnaker.exe\n"
        "  console_out: fictrac_hw.txt\n"
        "  first_frame_timeout_ms: 0\n"
        "  target_fps: 200\n"
        "  arm_delay_s: 0.5\n"
        "  startup_timeout_s: 0\n"
        "  timeout_s: 7\n",
        encoding="utf-8",
    )

    cfg = load_run_protocol_config(None, hardware_path=hw_path)
    assert cfg.fictrac_config == str((tmp_path / "config_camera.txt").resolve())
    assert cfg.fictrac_camera_serial == "26021184"
    assert cfg.fictrac_bin == "C:/rig/fictrac-spinnaker.exe"
    assert cfg.fictrac_console_out == "fictrac_hw.txt"
    assert cfg.fictrac_first_frame_timeout_ms == 0
    assert cfg.fictrac_target_fps == 200.0
    assert cfg.fictrac_arm_delay_s == 0.5
    assert cfg.fictrac_startup_timeout_s == 0.0
    assert cfg.fictrac_timeout_s == 7.0


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
