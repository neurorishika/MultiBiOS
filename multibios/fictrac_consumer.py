from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol

import numpy as np

from multibios.fictrac_client import FicTracFrame


class FicTracFrameSource(Protocol):
    def get_latest(self) -> tuple[int, Optional[FicTracFrame]]:
        ...

    def wait_for_next(self, after_seq: int = -1, timeout: float | None = None) -> tuple[int, Optional[FicTracFrame]]:
        ...

    def recent_array(self, max_count: int | None = None) -> np.ndarray:
        ...


@dataclass(slots=True)
class ClosedLoopSample:
    seq: int
    frame: Optional[FicTracFrame]


class ClosedLoopFrameConsumer:
    """Consume FicTrac frames with control-loop-friendly semantics.

    This consumer is designed for loops that may run slower than the incoming
    camera/FicTrac rate. The normal pattern is:

    - keep a local sequence cursor
    - wait for something newer than that cursor
    - operate on the newest available frame
    - skip any stale backlog automatically
    """

    def __init__(self, source: FicTracFrameSource, *, start_at_latest: bool = False) -> None:
        self._source = source
        self._last_seq = -1
        if start_at_latest:
            self._last_seq, _ = self._source.get_latest()

    @property
    def last_seq(self) -> int:
        return self._last_seq

    def reset(self, seq: int = -1) -> None:
        self._last_seq = seq

    def snapshot_latest(self) -> ClosedLoopSample:
        seq, frame = self._source.get_latest()
        return ClosedLoopSample(seq=seq, frame=frame)

    def consume_latest(self) -> ClosedLoopSample:
        seq, frame = self._source.get_latest()
        if seq > self._last_seq:
            self._last_seq = seq
        return ClosedLoopSample(seq=seq, frame=frame)

    def wait_for_newer(self, timeout: float | None = None) -> ClosedLoopSample:
        seq, frame = self._source.wait_for_next(after_seq=self._last_seq, timeout=timeout)
        if seq > self._last_seq:
            self._last_seq = seq
        return ClosedLoopSample(seq=seq, frame=frame)

    def recent_history(self, max_count: int | None = None) -> np.ndarray:
        return self._source.recent_array(max_count=max_count)
