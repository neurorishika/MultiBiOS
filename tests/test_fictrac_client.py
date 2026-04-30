from __future__ import annotations

import os
import sys
import threading
import time
from pathlib import Path

import numpy as np

from multibios.fictrac_client import (FICTRAC_FRAME_DTYPE, FicTracFrame,
                                      FicTracFrameStore, FicTracState,
                                      record_to_frame)
from multibios.fictrac_consumer import ClosedLoopFrameConsumer
from multibios.fictrac_runtime import build_fictrac_subprocess_env


PYBMT_CANDIDATES = [
    Path(__file__).resolve().parents[2] / "pybmt-master",
    Path(__file__).resolve().parents[2] / "legacy" / "pybmt-master",
]
for candidate in PYBMT_CANDIDATES:
    if candidate.is_dir():
        if str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))
        break

from pybmt.fictrac.state import FicTracState as PyBMTState  # type: ignore  # noqa: E402


def _payload_v20() -> str:
    values = [
        "12",
        "0.1", "0.2", "0.3",
        "0.4",
        "0.5", "0.6", "0.7",
        "0.8", "0.9", "1.0",
        "1.1", "1.2", "1.3",
        "1.4", "1.5", "1.6", "1.7", "1.8", "1.9", "2.0", "2.1", "21", "2.3",
    ]
    return ",".join(values)


def _payload_v21() -> str:
    return "FT," + _payload_v20() + ",2.4"


def _compare_selected_fields(payload: str) -> None:
    ours = FicTracState.from_udp_message(payload)
    theirs = PyBMTState.zmq_string_msg_to_state(payload)
    assert ours.frame_cnt == theirs.frame_cnt
    assert ours.del_rot_cam_x == theirs.del_rot_cam_vec[0]
    assert ours.del_rot_cam_y == theirs.del_rot_cam_vec[1]
    assert ours.del_rot_cam_z == theirs.del_rot_cam_vec[2]
    assert ours.del_rot_error == theirs.del_rot_error
    assert ours.del_rot_lab_x == theirs.del_rot_lab_vec[0]
    assert ours.del_rot_lab_y == theirs.del_rot_lab_vec[1]
    assert ours.del_rot_lab_z == theirs.del_rot_lab_vec[2]
    assert ours.abs_ori_cam_x == theirs.abs_ori_cam_vec[0]
    assert ours.abs_ori_cam_y == theirs.abs_ori_cam_vec[1]
    assert ours.abs_ori_cam_z == theirs.abs_ori_cam_vec[2]
    assert ours.abs_ori_lab_x == theirs.abs_ori_lab_vec[0]
    assert ours.abs_ori_lab_y == theirs.abs_ori_lab_vec[1]
    assert ours.abs_ori_lab_z == theirs.abs_ori_lab_vec[2]
    assert ours.posx == theirs.posx
    assert ours.posy == theirs.posy
    assert ours.heading == theirs.heading
    assert ours.direction == theirs.direction
    assert ours.speed == theirs.speed
    assert ours.intx == theirs.intx
    assert ours.inty == theirs.inty
    assert ours.timestamp == theirs.timestamp
    assert ours.seq_num == theirs.seq_num
    assert ours.delta_timestamp == theirs.delta_timestamp
    assert ours.alt_timestamp == theirs.alt_timestamp


def test_parser_matches_pybmt_v20() -> None:
    _compare_selected_fields(_payload_v20())


def test_parser_matches_pybmt_v21() -> None:
    _compare_selected_fields(_payload_v21())


def test_frame_store_latest_and_chunk_growth() -> None:
    store = FicTracFrameStore(chunk_size=2, recent_capacity=3)
    for idx in range(5):
        frame = FicTracFrame(
            wall_time=float(idx),
            frame_cnt=idx,
            posx=1.0 + idx,
            posy=2.0 + idx,
            heading=3.0 + idx,
            speed=4.0 + idx,
            direction=5.0 + idx,
            intx=6.0 + idx,
            inty=7.0 + idx,
            timestamp=8.0 + idx,
            delta_timestamp=0.005,
        )
        seq = store.append(frame)
        assert seq == idx

    latest_seq, latest = store.get_latest()
    assert latest_seq == 4
    assert latest is not None
    assert latest.frame_cnt == 4

    all_frames = store.frame_array()
    assert all_frames.dtype == FICTRAC_FRAME_DTYPE
    assert len(all_frames) == 5
    assert int(all_frames[-1]["frame_cnt"]) == 4

    recent = store.recent_array()
    assert len(recent) == 3
    assert [int(row["frame_cnt"]) for row in recent] == [2, 3, 4]


def test_frame_store_wait_for_next() -> None:
    store = FicTracFrameStore(chunk_size=2, recent_capacity=2)
    results: list[tuple[int, int | None]] = []

    def waiter() -> None:
        seq, frame = store.wait_for_next(after_seq=-1, timeout=1.0)
        results.append((seq, None if frame is None else frame.frame_cnt))

    thread = threading.Thread(target=waiter)
    thread.start()
    time.sleep(0.05)
    store.append(
        FicTracFrame(
            wall_time=1.0,
            frame_cnt=7,
            posx=0.0,
            posy=0.0,
            heading=0.0,
            speed=0.0,
            direction=0.0,
            intx=0.0,
            inty=0.0,
            timestamp=1.0,
            delta_timestamp=0.005,
        )
    )
    thread.join(timeout=1.0)
    assert results == [(0, 7)]


def test_frame_store_save_npz_roundtrip(tmp_path: Path) -> None:
    store = FicTracFrameStore(chunk_size=4, recent_capacity=2)
    frame = FicTracFrame(
        wall_time=1.23,
        frame_cnt=9,
        posx=1.0,
        posy=2.0,
        heading=3.0,
        speed=4.0,
        direction=5.0,
        intx=6.0,
        inty=7.0,
        timestamp=8.0,
        delta_timestamp=0.01,
    )
    store.append(frame)
    out_path = tmp_path / "frames.npz"
    count = store.save_npz(out_path)
    assert count == 1

    loaded = np.load(out_path, allow_pickle=False)["frames"]
    assert loaded.dtype == FICTRAC_FRAME_DTYPE
    roundtrip = record_to_frame(loaded[0])
    assert roundtrip.frame_cnt == frame.frame_cnt
    assert roundtrip.del_rot_cam_x == frame.del_rot_cam_x
    assert roundtrip.abs_ori_lab_z == frame.abs_ori_lab_z
    assert roundtrip.seq_num == frame.seq_num
    assert roundtrip.timestamp == frame.timestamp


def test_closed_loop_consumer_skips_backlog_and_uses_newest() -> None:
    store = FicTracFrameStore(chunk_size=8, recent_capacity=8)
    consumer = ClosedLoopFrameConsumer(store)

    for idx in range(3):
        store.append(
            FicTracFrame(
                wall_time=float(idx),
                frame_cnt=idx,
                posx=0.0,
                posy=0.0,
                heading=0.0,
                speed=0.0,
                direction=0.0,
                intx=0.0,
                inty=0.0,
                timestamp=float(idx),
                delta_timestamp=0.005,
            )
        )

    sample = consumer.consume_latest()
    assert sample.seq == 2
    assert sample.frame is not None
    assert sample.frame.frame_cnt == 2

    for idx in range(3, 6):
        store.append(
            FicTracFrame(
                wall_time=float(idx),
                frame_cnt=idx,
                posx=0.0,
                posy=0.0,
                heading=0.0,
                speed=0.0,
                direction=0.0,
                intx=0.0,
                inty=0.0,
                timestamp=float(idx),
                delta_timestamp=0.005,
            )
        )

    newer = consumer.wait_for_newer(timeout=0.1)
    assert newer.seq == 5
    assert newer.frame is not None
    assert newer.frame.frame_cnt == 5

    history = consumer.recent_history(max_count=2)
    assert [int(row["frame_cnt"]) for row in history] == [4, 5]


def test_build_fictrac_subprocess_env_strips_conda_paths_on_windows(tmp_path: Path) -> None:
    fictrac_bin = tmp_path / "fictrac-spinnaker" / "fictrac-spinnaker.exe"
    fictrac_bin.parent.mkdir()
    fictrac_bin.write_text("", encoding="utf-8")

    env = build_fictrac_subprocess_env(
        fictrac_bin_path=fictrac_bin,
        base_env={
            "PATH": os.pathsep.join(
                [
                    r"C:\Users\markd\.conda\envs\multibios-blackfly",
                    r"C:\Users\markd\.conda\envs\multibios-blackfly\Library\bin",
                    r"C:\Windows\System32",
                    r"C:\Tools",
                ]
            ),
            "CONDA_PREFIX": r"C:\Users\markd\.conda\envs\multibios-blackfly",
            "CONDA_DEFAULT_ENV": "multibios-blackfly",
            "PYTHONPATH": r"C:\temp\pythonpath",
        },
    )

    path_parts = env["PATH"].split(os.pathsep)
    assert path_parts[0] == str(fictrac_bin.parent)
    assert r"C:\Windows\System32" in path_parts
    assert r"C:\Tools" in path_parts
    assert not any(".conda\\envs\\multibios-blackfly" in part for part in path_parts)
    assert "CONDA_PREFIX" not in env
    assert "CONDA_DEFAULT_ENV" not in env
    assert "PYTHONPATH" not in env
