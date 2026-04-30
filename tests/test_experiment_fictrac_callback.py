from __future__ import annotations

from pathlib import Path

from multibios.experiment import ExperimentCallback
from multibios.experiment import (_prepare_fictrac_runtime_config,
                                  load_experiment_config)
from multibios.fictrac_client import FicTracState


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
    )

    runtime_text = runtime_path.read_text(encoding="utf-8")
    assert camera_index == 1
    assert "save_raw         : y" in runtime_text
    assert "src_fps          : 60.000000" in runtime_text
    assert "vid_codec        : raw" in runtime_text
    assert f"output_fn        : {(tmp_path / 'fictrac').as_posix()}" in runtime_text
    assert info["save_raw"] is True


def test_load_experiment_config_reads_camera_recording_fields(tmp_path: Path) -> None:
    cfg_path = tmp_path / "experiment_config.yaml"
    cfg_path.write_text(
        "save_camera_raw_video: true\n"
        "fictrac_raw_video_codec: mjpg\n"
        "other_camera_timeout_ms: 125\n"
        "other_camera_queue_size: 32\n"
        "other_camera_stream_buffer_count: 64\n"
        "other_camera_exposure_us: 4000\n"
        "other_camera_roi_height: 512\n"
        "other_camera_binning: 2\n",
        encoding="utf-8",
    )

    cfg = load_experiment_config(cfg_path)
    assert cfg.save_camera_raw_video is True
    assert cfg.fictrac_raw_video_codec == "mjpg"
    assert cfg.other_camera_timeout_ms == 125
    assert cfg.other_camera_queue_size == 32
    assert cfg.other_camera_stream_buffer_count == 64
    assert cfg.other_camera_exposure_us == 4000.0
    assert cfg.other_camera_roi_height == 512
    assert cfg.other_camera_binning == 2
