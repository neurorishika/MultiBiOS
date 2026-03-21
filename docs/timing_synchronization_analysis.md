# Timing Synchronization Analysis

Date: 2026-03-20

## Goal

Bring NI-DAQ latch timing, triggers, Teensy-controlled valve state, FicTrac, and MFC activity onto the same timebase as precisely as possible.

## Current architecture

There are effectively two timing models in this repository.

### 1. Hardware-clocked NI-DAQ path

The NI-DAQ runner is already designed around a shared hardware clock:

- DO is the master clock.
- AO and AI can be slaved to the DO sample clock.
- Trigger pulses and latch edges can be precomputed and emitted deterministically.

This is the architecture that can support a real common timebase.

### 2. Mixed-clock experiment path

The open-loop experiment runner currently mixes:

- NI-DAQ finite DO for camera/microscope triggers and periodic latch pulses.
- Teensy serial commands issued from the host on `time.perf_counter()`.
- Alicat serial setpoints issued from the host.
- FicTrac frames timestamped by host arrival time in the callback.

This means the DAQ has a precise hardware clock, but the rest of the system is only approximately aligned to it.

### 3. DAQ-driven Teensy latch firmware path

There is also a better Teensy architecture available: the Teensy can operate purely as a DAQ-driven preload and latch slave.

In that mode:

- DAQ drives all S lines.
- DAQ drives `GLOBAL_LOAD`.
- DAQ drives each `RCK` line.
- Teensy samples the DAQ-provided state bits on `GLOBAL_LOAD`.
- Teensy shifts the SPI frame internally.
- the external hardware changes state only on the DAQ-driven `RCK` edge.
- Teensy exposes `READY_*` lines that can be measured by the DAQ.

This path is sufficient for Teensy synchronization, because the Teensy is no longer acting as an independent timebase. The authoritative physical event time becomes the DAQ latch edge.

## Main timing problems

### A. Teensy serial is not on the DAQ clock

In the current mixed runner, valve state changes are sent over USB serial from Windows at scheduled wall-clock times. That introduces host scheduling jitter, USB serial latency, and software timing error.

The Teensy command send time is therefore not the same thing as the physical valve switch time.

This problem goes away if the DAQ-driven Teensy firmware is used and runtime serial timing is removed from the valve path.

### B. Latch timing is periodic, not event-specific

The DAQ waveform currently emits periodic `GLOBAL_LOAD_REQ` and `RCK_*` pulses. That means a newly staged Teensy pattern does not appear on hardware immediately; it appears on the next latch edge.

This adds variable latency up to one full latch period.

Example consequence:

- if `latch_interval_ms = 10`, the physical switch can be delayed by anything between 0 and 10 ms after the host serial command.

### C. Software `t0` does not define true DAQ time zero

In the mixed runner, the DAQ task is started before experiment `t0` is defined from `time.perf_counter()`. That means software timestamps and DAQ waveform time are only approximately aligned.

### D. Alicat serial cannot be made truly synchronous with DAQ

If MFC setpoints are sent via serial commands, they do not share the DAQ sample clock. They can be logged and characterized, but not made precisely synchronous in the same way as DAQ AO.

### E. FicTrac callback time is not a hardware timestamp

The current FicTrac log uses host callback arrival times. That is useful for analysis, but it is not the same as the hardware time of camera exposure.

If the goal is a common timebase, frame timing must be tied to camera trigger and/or exposure signals measured by the DAQ.

## What can actually share one precise timebase

The following can be made truly common-clock if routed through the DAQ:

- digital outputs
- analog outputs
- analog inputs
- digital input markers
- counter/timestamp measurements
- camera trigger timing
- valve commit timing
- MFC setpoint timing, if driven from AO

The following cannot be truly common-clock while controlled live from the PC over serial:

- Teensy command issue time from Windows
- Alicat serial setpoint issue time
- FicTrac callback arrival time on the host

## Recommended synchronization strategy

## Recommendation 1: Make NI-DAQ the single runtime clock

This should be the core design rule.

Use the DAQ as the only authoritative experiment clock during acquisition. Anything timing-critical should either be generated directly by the DAQ or measured back by the DAQ.

## Recommendation 2: Treat Teensy as a preload device, not a timebase

The Teensy is well-suited to preloading shift-register patterns, but not as a shared precision clock if commands are sent live over USB serial from the host.

Best use of Teensy here:

- have the DAQ drive all state-select lines
- let DAQ `LOAD_REQ` and `RCK` define the actual commit instant
- let the Teensy translate those state bits into SPI preload data
- expose `READY_*` back to DAQ DI so preload completion is measured on the DAQ clock

The physical event of interest should be the DAQ-controlled commit edge, not the host serial send time.

### Teensy-specific sufficiency condition

The DAQ-driven Teensy path is sufficient for synchronization if all of the following hold:

- no live host serial commands are required to define valve timing during the run
- the DAQ sets S-bit values before `GLOBAL_LOAD`
- the LOAD-to-RCK interval is conservatively larger than Teensy ISR plus SPI transfer time
- `READY_*` is either measured or at least validated before the corresponding `RCK`

Under those conditions, the Teensy is synchronized strongly enough for the valve path.

## Recommendation 3: Replace periodic latch pulses with event-specific latch scheduling

For precision work, free-running latch pulses should be removed from the mixed path.

Instead:

- compile each intended valve change into a specific DAQ `LOAD_REQ` and `RCK` event
- send Teensy preload sufficiently early
- use the DAQ waveform to commit at the exact requested time

This removes the variable 0 to `latch_interval_ms` latency.

## Recommendation 4: Move MFC timing-critical control to DAQ AO

If the requirement is same-timebase precision, MFC setpoints should be generated by DAQ AO, with AI capturing feedback on the same sample clock.

This is the cleanest architecture for MFC synchronization.

If Alicat serial must remain in use, then it should be treated as a supervisory or slow-control path, not a precision-timed path.

## Recommendation 5: Tie camera and FicTrac timing to hardware signals

To align FicTrac to the experiment clock precisely:

- drive camera trigger from DAQ
- record camera exposure or strobe output back into DAQ DI or a counter input
- align FicTrac frame numbers to those exposure timestamps

Without exposure feedback, FicTrac timing is only approximately aligned through host processing time.

## Recommendation 6: Measure real hardware timing, not just scheduled intent

Add DAQ-recorded markers for:

- camera exposure or strobe
- Teensy `READY_*`
- optional Teensy sync pulse
- optional microscope returned trigger or frame-ack pulse

This gives post hoc verification of actual timing relationships.

## Recommendation 7: Distinguish commanded time from physical response time

Even in a fully synchronized design, not every subsystem responds instantaneously.

Examples:

- a DAQ trigger can define camera command time exactly, but actual exposure timing is best verified with a returned strobe or exposure signal
- a DAQ AO waveform can define MFC setpoint command time exactly, but actual flow still depends on MFC control dynamics and tubing dead volume
- a DAQ `RCK` edge can define valve commit time exactly, but odor arrival at the animal still depends on pneumatic transport delay

The timebase can be unified even when physical response is delayed, as long as those delayed responses are measured against the same DAQ clock.

## Practical target architecture

### Best architecture for one common timebase

- DAQ DO is master clock
- DAQ drives valve state-select lines
- DAQ drives `GLOBAL_LOAD` and `RCK`
- DAQ AO outputs MFC setpoints
- DAQ AI captures MFC feedback
- DAQ emits camera triggers
- camera exposure/strobe is recorded by DAQ
- Teensy translates DAQ state bits into SPI preload data only
- Teensy `READY` is recorded by DAQ DI

In this model, all physically important events are either generated by or measured by the same hardware clock.

### Acceptable compromise if live serial must remain

- keep Teensy serial for preloading only
- schedule DAQ latch edges per event, not periodically
- record `READY` and camera exposure feedback into DAQ
- treat serial send time as advisory, not authoritative

This still leaves serial jitter in the preload path, but the physical commit can be precisely defined.

## Expected precision limits

### DAQ-only timed paths

These can align to within approximately one DAQ sample period plus hardware propagation delay.

### Host-issued serial paths

These should be expected to show millisecond-scale jitter and occasional worse outliers because they depend on:

- Windows scheduling
- Python scheduling
- USB stack timing
- device-side parsing and handling

### FicTrac host callback times

These are suitable for analysis and monitoring, but not as the authoritative hardware time of frame acquisition.

## Recommended next implementation steps

### Highest value changes

1. Replace periodic latch cadence in the mixed runner with event-specific `LOAD_REQ` and `RCK` scheduling.
2. Add a dedicated DAQ start-sync mechanism so software logs and DAQ time zero are explicitly related.
3. Record Teensy `READY` and camera exposure/strobe with the DAQ.
4. For highest precision, move MFC timing-critical control from Alicat serial to DAQ AO/AI.

## Step-by-step plan to completely tighten timing

This is the recommended execution order. Each phase improves synchronization on its own and also sets up the next phase.

### Phase 0: Freeze the master-clock design rule

Goal:

- establish NI-DAQ as the only authoritative runtime clock

Actions:

- stop treating host `perf_counter()` timestamps as authoritative event time
- define all timing-critical events from DAQ waveforms or DAQ-measured return signals
- treat host logs only as operator diagnostics

Success criterion:

- every timing-critical event is either generated by or measured by the DAQ

### Phase 1: Lock in the DAQ-driven Teensy path

Goal:

- remove host serial timing from valve synchronization entirely

Actions:

- use the DAQ-driven Teensy firmware path instead of live serial valve commands
- drive S lines, `GLOBAL_LOAD`, and `RCK` from DAQ only
- wire all `READY_*` outputs back into DAQ DI
- choose a conservative LOAD-to-RCK delay

Validation:

- scope or record `GLOBAL_LOAD`, `READY_*`, and `RCK`
- confirm `READY_*` goes high before the matching `RCK`
- confirm valve state changes happen on the `RCK` edge, not earlier

Success criterion:

- valve commit time is defined entirely by the DAQ and verified by `READY`

### Phase 2: Replace periodic latching with event-specific latch scheduling

Goal:

- eliminate the remaining variable valve latency from free-running latch cadence

Actions:

- compile each valve transition into a specific `GLOBAL_LOAD` and `RCK` pair
- remove periodic latch pulses for timing-critical protocols
- keep guardrails so preload windows cannot overlap unsafely

Validation:

- export or visualize scheduled latch edges per event
- confirm one intended valve event corresponds to one intentional latch event

Success criterion:

- valve timing uncertainty from latch cadence is reduced from 0 to latch-period down to approximately one sample period

### Phase 3: Put both cameras on DAQ trigger timing

Goal:

- unify camera command timing with the DAQ timebase

Actions:

- configure both FicTrac and alignment cameras for external trigger mode
- drive both triggers from DAQ DO
- use a common DAQ start condition for both trigger trains

Validation:

- scope both trigger outputs
- verify trigger interval, pulse width, and start alignment

Success criterion:

- both camera trigger times are defined by the DAQ rather than software timing

### Phase 4: Measure actual camera exposure timing

Goal:

- turn camera synchronization from scheduled to verified

Actions:

- wire camera exposure, strobe, or frame-active outputs back to DAQ DI or counters
- log those returned signals for both cameras
- align FicTrac frames to returned exposure events rather than callback arrival times alone

Validation:

- compare DAQ trigger edge to returned exposure edge
- quantify fixed and variable latency for each camera

Success criterion:

- the actual frame acquisition timing of both cameras is measured on the DAQ clock

### Phase 5: Move MFC control from serial to DAQ AO

Goal:

- put MFC command timing on the same hardware clock as cameras and valves

Actions:

- map each MFC setpoint to DAQ AO
- generate setpoint waveforms in the compiled path
- remove serial Alicat timing from precision-critical operation

Validation:

- step AO outputs into the MFC command inputs
- confirm commanded voltage transitions occur at the expected sample index

Success criterion:

- MFC command timing is fully DAQ-defined

### Phase 6: Measure MFC response on DAQ AI

Goal:

- separate exact command timing from real flow-response timing

Actions:

- wire MFC analog feedback to DAQ AI
- sample AI on the same shared clock as DO and AO
- quantify latency, settling time, and steady-state accuracy

Validation:

- compare AO command edges to AI response traces
- characterize per-channel delay and rise time

Success criterion:

- both MFC command time and actual flow response are recorded on one timebase

### Phase 7: Unify software and hardware time-zero bookkeeping

Goal:

- make analysis files explicitly reference the DAQ time origin

Actions:

- add a DAQ start-sync marker line or explicit start event in the waveform
- derive exported timestamps from DAQ sample indices wherever possible
- keep host wall-clock logs only as secondary metadata

Validation:

- confirm exported event tables are reconstructible from DAQ sample indices alone

Success criterion:

- the saved analysis data reflects DAQ time first and host time second

### Phase 8: Quantify transport delays separately from clock synchronization

Goal:

- avoid confusing common-clock synchronization with biological delivery latency

Actions:

- measure odor transport delay from valve commit to odor arrival
- measure camera trigger-to-exposure delay
- measure MFC command-to-flow delay
- document these as subsystem response delays, not clock errors

Validation:

- maintain a table of fixed delay and jitter for each subsystem

Success criterion:

- timing error and physical transport delay are clearly separated in analysis

### Phase 9: Final fully synchronized operating mode

At the end state:

- DAQ defines valve state-select timing, LOAD timing, and RCK timing
- Teensy is only a DAQ-triggered SPI preload engine
- DAQ triggers both cameras
- DAQ measures both camera exposure or strobe signals
- DAQ outputs MFC setpoints through AO
- DAQ measures MFC feedback through AI
- DAQ records `READY_*` and any other hardware acknowledgements
- post-run analysis is derived from DAQ sample indices and measured return signals

This is the point at which the system is not just synchronized in intent, but synchronized and verified on a single hardware timebase.

### Strategic direction

If the primary goal is same-timebase precision, extend the hardware-clocked NI-DAQ architecture rather than continuing to rely on the mixed `experiment.py` path for timing-critical execution.

## Bottom line

Yes, synchronization can be made significantly better, but only by making NI-DAQ the authoritative clock and demoting serial-controlled subsystems to preload or supervisory roles.

The strongest version of that design is:

- DAQ defines all event times
- DAQ commits valve changes
- DAQ drives or timestamps camera events
- DAQ drives MFC setpoints and reads MFC feedback
- Teensy only stages data ahead of commit
- FicTrac is aligned through hardware camera timing, not host callback arrival

That is the path that can legitimately put the experiment on one precise timebase.