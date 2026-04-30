# FicTrac Client Architecture

This page documents the in-repo MultiBiOS FicTrac client, the frame-storage model, the closed-loop consumer API, and the validation criteria required before the older pybmt-based path can be considered deprecated.

## Why This Exists

MultiBiOS still depends on the FicTrac executable for the tracking algorithm itself, but it no longer needs an external Python wrapper to:

- launch the FicTrac subprocess
- bind the localhost UDP socket
- parse FicTrac UDP messages
- expose latest-frame access to the experiment runner

That wrapper logic now lives inside MultiBiOS so the runtime path, buffering policy, and future closed-loop semantics are all controlled in one codebase.

## Module Layout

### `multibios/fictrac_client.py`

Owns the transport and storage primitives:

- `FicTracState`: full parsed UDP state matching the upstream FicTrac variable set
- `FicTracFrame`: per-frame record used by the runner and saved outputs
- `FicTracFrameStore`: structured frame retention for logging and control
- `FicTracDriver`: subprocess + UDP receiver

### `multibios/fictrac_consumer.py`

Owns the closed-loop consumption policy:

- `ClosedLoopFrameConsumer`: newest-frame oriented consumer for loops slower than camera rate

### `multibios/experiment.py`

Owns the experiment-side integration:

- `ExperimentCallback`: callback that writes into the frame store
- `make_consumer()`: convenience constructor for future closed-loop logic

## Design Goals

The internal client was designed around four constraints.

1. Transport overhead should stay small relative to FicTrac and camera acquisition.
2. Long runs should avoid retaining all frames as Python objects.
3. Saved outputs should retain the full upstream FicTrac state for future debugging.
4. Closed-loop code must be able to read the newest state without falling behind forever.
5. The replacement must be testable against the older pybmt parser before deprecation.

## Frame Storage Model

`FicTracFrameStore` uses two storage strategies at the same time.

### 1. Full-run structured storage

All frames are retained in structured NumPy arrays with dtype `FICTRAC_FRAME_DTYPE`.

The stored record now includes the full upstream FicTrac variable set, including:

- delta rotation vectors in camera and lab coordinates
- delta rotation error
- absolute orientation vectors in camera and lab coordinates
- integrated position, heading, direction, speed, and timestamps
- UDP sequence number and optional v2.1.1 alternate timestamp

Storage is chunked in memory:

- a current writable chunk
- a list of completed chunks
- final concatenation only when a full export is requested

This is more efficient than storing one Python object per frame, but it is still an in-memory strategy.

### 2. Recent-history ring buffer

A second fixed-size ring stores the most recent frames for short-horizon filtering or control logic.

That gives closed-loop code fast access to recent history without copying the entire run.

## Important Clarification About "Chunking"

Yes, MultiBiOS is already chunking FicTrac storage now.

But it is currently:

- **chunked in memory**
- **not streamed to disk incrementally during the run**

Those are different things.

Current behavior:

- frames accumulate in numeric chunks in RAM
- at save time, they are exported to `fictrac_frames.npz`

Not yet implemented:

- periodic spill-to-disk while the run is still active
- memory-mapped backing store
- append-only binary logging during acquisition

For the current scale, in-memory chunking is the right first optimization. If future runs become long enough that total frame count becomes the dominant memory cost, then incremental disk logging is the next step.

## Closed-Loop Consumer API

`ClosedLoopFrameConsumer` is the policy layer intended for future behavior-dependent loops.

The most important methods are:

- `snapshot_latest()`
- `consume_latest()`
- `wait_for_newer(timeout=...)`
- `recent_history(max_count=...)`

### Why newest-frame semantics matter

At 200 fps, the incoming frame period is 5 ms.

If a closed-loop control cycle takes longer than 5 ms, then trying to process every frame creates backlog. The right default behavior is usually:

1. wait until something newer exists
2. consume the newest available frame
3. skip stale intermediate frames

That is exactly what `wait_for_newer()` is meant to support.

### Example pattern

```python
consumer = experiment_callback.make_consumer(start_at_latest=True)

while running:
    sample = consumer.wait_for_newer(timeout=0.02)
    if sample.frame is None:
        continue

    frame = sample.frame
    # Run the control calculation on the newest frame available.
    # If several frames arrived while the loop was busy, backlog is skipped.
    control_step(frame)
```

### Recent history example

```python
history = consumer.recent_history(max_count=8)
```

That pattern is intended for short filters, velocity smoothing, or prediction over the last few received frames.

## Validation Requirements Before Deprecating The Older Path

The internal client should not be treated as a full replacement just because it compiles or because one live probe succeeds.

The required validation surface is:

1. **Parser equivalence**: for the FicTrac UDP message formats you actually receive, the MultiBiOS parser must match pybmt on all fields MultiBiOS consumes.
2. **Frame-order behavior**: skipped-frame detection and monotonic frame counting must behave the same way.
3. **Live callback behavior**: the internal client must receive live frames on the triggered Blackfly path.
4. **Legacy-runner behavior**: the deprecated serial runner must remain stable with the internal client until it is removed.
5. **Saved-data fidelity**: exported data must contain the same tracked information needed for downstream analysis.

## What Is Tested In-Repo

The current automated tests cover:

- parser equivalence against pybmt for both 24-field and 25-field message formats across the full upstream field set
- frame-store latest-frame behavior
- frame-store wait-for-next behavior
- NPZ save/load roundtrip of structured frames
- closed-loop consumer newest-frame semantics
- experiment callback integration with the consumer

These tests are necessary, but they are not the whole deprecation bar because they do not replace live hardware validation.

## Recommended Live Validation Before Deprecation

Use this order:

1. Run the existing live triggered probe with the internal client.
2. Compare parsed values from the internal path and the older pybmt path on the same captured UDP payloads if you still have the older environment available.
3. Run the bounded legacy serial-runner procedure from [legacy/serial_experiment_pipeline.md](legacy/serial_experiment_pipeline.md).
4. Compare frame counts, timestamps, and saved tracking columns across both paths.
5. Only then remove or formally deprecate the old wrapper path.
