# YAML Protocol Specification

This document describes the YAML protocol format for defining olfactometer experiments with precise timing control.

## Protocol Structure Overview

A protocol consists of three main sections:

- **Protocol metadata**: Name, version, and description
- **Timing configuration**: Sample rates, pulse widths, and timing constraints
- **Sequence definition**: Phases with device actions and precise timing

## Top-Level Configuration

```yaml
protocol:
  name: "Descriptive Protocol Name"
  version: "1.0"
  description: "Detailed description of the experimental paradigm"
  timing:
    base_unit: "ms"                    # Time unit for all timing values
    sample_rate: 1000                  # Hz - 1000=1ms precision, 10000=0.1ms precision
    camera_interval: 100               # ms - interval between camera pulses (0=disabled)
    camera_pulse_duration: 5           # ms - duration of camera trigger pulses
    preload_lead_ms: 2                 # ms - S-bits switch before LOAD_REQ pulse
    load_req_ms: 1                     # ms - duration of LOAD_REQ pulses
    rck_pulse_ms: 1                    # ms - duration of register clock (RCK) pulses
    trig_pulse_ms: 5                   # ms - duration of microscope trigger pulses
    setup_hold_samples: 100            # samples - S-bit stability margin around load events
    seed: 42                           # RNG seed for reproducible randomization (optional)

sequence:
  - phase: "Phase Name"
    duration: 30000                    # ms - total phase duration
    times: 5                           # number of repetitions (preferred over 'repeat')
    repeat: 0                          # legacy: 0=no repeat, 1=repeat once
    randomize: true                    # randomize order within state lists
    actions:
      - device: "device.name"
        state: "STATE_NAME"            # for digital devices
        value: 2.5                     # for analog devices (volts)
        timing: 1000                   # ms - offset within phase
```

## Timing Configuration Details

### Sample Rate & Precision

- **sample_rate**: Determines temporal resolution
  - `1000` Hz = 1 ms precision (standard)
  - `10000` Hz = 0.1 ms precision (high precision)
- **base_unit**: Always "ms" for millisecond timing

### Hardware Pulse Timing

- **preload_lead_ms**: Time before LOAD_REQ when S-bits switch to new state
- **load_req_ms**: Duration of load request pulses sent to Teensy
- **rck_pulse_ms**: Duration of register clock pulses for valve commits
- **setup_hold_samples**: Extra samples for S-bit stability around load events

### Camera & Trigger Timing  

- **camera_interval**: Interval between continuous camera pulses (0 = disabled)
- **camera_pulse_duration**: Width of camera trigger pulses
- **trig_pulse_ms**: Width of microscope trigger pulses

## Device Types & States

## Device Configuration Reference

### Olfactometers (8-Valve Systems)

Control large olfactometer valve arrays with 8 possible states.

**Device Keys:**

- `olfactometer.left` - Left olfactometer system
- `olfactometer.right` - Right olfactometer system

**Available States:**

- `OFF` - All valves closed (state 0)
- `AIR` - Clean air delivery (state 1)
- `ODOR1` through `ODOR5` - Specific odor channels (states 2-6)
- `FLUSH` - System flush/clean (state 7)

**Multi-State Selection:**

```yaml
# Single state
state: "ODOR1"

# Sequential selection (uses repeat index)  
state: "ODOR1,ODOR2,ODOR3"

# Copy from other side
state: "COPY"  # mirrors the resolved left olfactometer state
```

### Switch Valves (2-State Systems)

Control binary switch valves for clean/odor selection.

**Device Keys:**

- `switch_valve.left` - Left switch valve
- `switch_valve.right` - Right switch valve  

**Available States:**

- `CLEAN` - Clean air path (state 0)
- `ODOR` - Odor delivery path (state 1)

### Mass Flow Controllers (Analog Outputs)

Set voltage levels (0-5V) for MFC setpoints.

**Device Keys:**

- `mfc.air_left_setpoint` - Left air MFC setpoint
- `mfc.air_right_setpoint` - Right air MFC setpoint
- `mfc.odor_left_setpoint` - Left odor MFC setpoint  
- `mfc.odor_right_setpoint` - Right odor MFC setpoint

**Usage:**

```yaml
- device: "mfc.air_left_setpoint"
  value: 2.1  # Volts (typically 0-5V range)
  timing: 1000
```

### Trigger Signals

Generate precise trigger pulses for external equipment.

**Device Keys:**

- `triggers.microscope` - Single pulse triggers for microscope
- `triggers.camera_continuous` - Continuous periodic camera triggers

**Microscope Triggers:**

```yaml
# Single pulse at specified timing
- device: "triggers.microscope" 
  state: true
  timing: 30000  # Pulse occurs at 30s mark
```

**Camera Triggers:**

```yaml
# Start continuous pulses
- device: "triggers.camera_continuous"
  state: true
  timing: 1000

# Stop continuous pulses  
- device: "triggers.camera_continuous"
  state: false
  timing: 60000
```

## Sequence Definition

### Phase Structure

Each phase represents a distinct experimental period with defined duration and actions.

```yaml
- phase: "Descriptive Phase Name"
  duration: 30000        # Total phase duration in ms
  times: 5               # Number of repetitions (recommended)
  repeat: 0              # Legacy: 0=no repeat, 1=repeat once  
  randomize: true        # Randomize state order within lists
  actions:               # List of device actions
    - device: "device.name"
      state: "STATE"
      timing: 1000       # Offset within phase (ms)
```

### Action Timing

All action timing is relative to the **start of each phase repetition**.

**Example Timeline:**

```yaml
- phase: "Trial Phase"
  duration: 60000  # 60 second phase
  times: 3         # Repeat 3 times
  actions:
    - device: "olfactometer.left"
      state: "ODOR1"
      timing: 0        # At phase start: 0s, 60s, 120s
    - device: "triggers.microscope"  
      state: true
      timing: 30000    # Mid-phase: 30s, 90s, 150s
```

### Randomization

When `randomize: true` and multi-state lists are provided:

```yaml
- phase: "Randomized Odors"
  times: 4
  randomize: true
  actions:
    - device: "olfactometer.left"
      state: "ODOR1,ODOR2,ODOR3,ODOR4"  # Will be shuffled each protocol run
      timing: 0
```

The `seed` parameter in timing configuration ensures reproducible randomization.

## Advanced Features

### State Copying

The right olfactometer can mirror the left side's resolved state:

```yaml
- device: "olfactometer.left" 
  state: "ODOR1,ODOR2,ODOR3"
  timing: 0
- device: "olfactometer.right"
  state: "COPY"  # Uses same resolved state as left
  timing: 100    # Slight delay for timing control
```

### Precise Timing Control

**Hardware Constraints:**

- S-bits switch `preload_lead_ms` before LOAD_REQ pulses
- `setup_hold_samples` provide stability margins around load events
- Overlapping preload windows are detected and rejected at compile time

**Timing Guardrails:**
The compiler enforces timing constraints to prevent hardware conflicts and ensure reliable valve switching.

## Complete Example

```yaml
protocol:
  name: "Odor Discrimination Task"
  timing:
    sample_rate: 1000
    seed: 42

sequence:
  - phase: "Baseline"
    duration: 30000
    times: 1
    actions:
      - device: "olfactometer.left"
        state: "AIR"
        timing: 0
      - device: "triggers.camera_continuous"
        state: true  
        timing: 1000

  - phase: "Odor Presentation"
    duration: 60000
    times: 5
    randomize: true
    actions:
      - device: "olfactometer.left"
        state: "ODOR1,ODOR2,ODOR3,ODOR4,ODOR5"
        timing: 0
      - device: "switch_valve.left"
        state: "ODOR"
        timing: 10000
      - device: "triggers.microscope"
        state: true
        timing: 15000
```
