---
applyTo: "config/**/*.yaml"
---

# Designing MultiBiOS Protocol YAML Files

## Canonical File Template

```yaml
# ==============================================================
#  PROTOCOL NAME
#  Brief description of what conditions are being tested.
#
#  ┌─────────────────────────────────────────────────────┐
#  │  CHANGE THE ODOR HERE (one place, applies to all)   │
#  │  Options: ODOR1  ODOR2  ODOR3  ODOR4  ODOR5  FLUSH  │
#  └─────────────────────────────────────────────────────┘
_active_odor: &odor "ODOR1"
_mfc_odor_flow_slpm: &mfc_flow 0.03
# ==============================================================

protocol:
  name: "Protocol Name"
  version: "1.0"
  description: >
    One-paragraph description of the experiment.

  timing:
    base_unit: "ms"
    sample_rate: 1000            # Hz — 1 ms precision (use 10000 for 0.1 ms)
    camera_interval: 100         # ms between camera pulses (10 Hz)
    camera_pulse_duration: 5     # ms pulse width
    preload_lead_ms: 2           # S-bit setup lead time
    load_req_ms: 1
    rck_pulse_ms: 1
    trig_pulse_ms: 5
    setup_hold_samples: 100
    load_mode: "global"          # "global" (recommended) or "per_assembly"
    seed: 42                     # Include if using randomized state selection

sequence:
  - phase: "BOOT SEQUENCE"       # Always first
    ...

  - phase: "OLFACTOMETER SATURATION"   # Always second (60 s drain/fill)
    ...

  # --- trials here ---

  - phase: "OLFACTOMETER DESATURATION" # Always second-to-last
    ...

  - phase: "SHUTDOWN SEQUENCE"   # Always last
    ...
```

---

## Required Phase Order

Every protocol must include these phases in this order:

### 1. BOOT SEQUENCE (always ≥ 11 000 ms)
```yaml
- phase: "BOOT SEQUENCE"
  duration: 11000
  repeat: 0
  actions:
    - device: "mfc.air_left_setpoint"
      value: *mfc_flow
      timing: 1000
    - device: "mfc.air_right_setpoint"
      value: *mfc_flow
      timing: 1000
    - device: "mfc.odor_left_setpoint"
      value: *mfc_flow
      timing: 1000
    - device: "mfc.odor_right_setpoint"
      value: *mfc_flow
      timing: 1000
    - device: "triggers.camera_continuous"
      state: true
      timing: 1000
    - device: "olfactometer.left"
      state: "AIR"
      timing: 1000
    - device: "olfactometer.right"
      state: "AIR"
      timing: 1005                # +5 ms offset to avoid simultaneous register writes
    - device: "switch_valve.left"
      state: "CLEAN"
      timing: 1010
    - device: "switch_valve.right"
      state: "CLEAN"
      timing: 1015
```

### 2. OLFACTOMETER SATURATION (always 65 000 ms)
```yaml
- phase: "OLFACTOMETER SATURATION"
  duration: 65000
  repeat: 0
  actions:
    - device: "olfactometer.left"
      state: *odor
      timing: 0
    - device: "olfactometer.right"
      state: *odor
      timing: 5
    - device: "switch_valve.left"
      state: "CLEAN"
      timing: 0
    - device: "switch_valve.right"
      state: "CLEAN"
      timing: 0
    - device: "triggers.microscope"
      state: true
      timing: 60000              # "saturation done" marker
```

### 3. TRIALS (one phase per condition — see patterns below)

### 4. OLFACTOMETER DESATURATION (always 65 000 ms)
```yaml
- phase: "OLFACTOMETER DESATURATION"
  duration: 65000
  repeat: 0
  actions:
    - device: "olfactometer.left"
      state: "AIR"
      timing: 0
    - device: "olfactometer.right"
      state: "AIR"
      timing: 5
    - device: "switch_valve.left"
      state: "CLEAN"
      timing: 0
    - device: "switch_valve.right"
      state: "CLEAN"
      timing: 0
    - device: "triggers.microscope"
      state: true
      timing: 60000
```

### 5. SHUTDOWN SEQUENCE (always 2 000 ms)
```yaml
- phase: "SHUTDOWN SEQUENCE"
  duration: 2000
  repeat: 0
  actions:
    - device: "triggers.camera_continuous"
      state: false
      timing: 0
    - device: "olfactometer.left"
      state: "OFF"
      timing: 0
    - device: "olfactometer.right"
      state: "OFF"
      timing: 5
    - device: "switch_valve.left"
      state: "CLEAN"
      timing: 10
    - device: "switch_valve.right"
      state: "CLEAN"
      timing: 15
    - device: "mfc.air_left_setpoint"
      value: 0.0
      timing: 0
    - device: "mfc.air_right_setpoint"
      value: 0.0
      timing: 0
    - device: "mfc.odor_left_setpoint"
      value: 0.0
      timing: 0
    - device: "mfc.odor_right_setpoint"
      value: 0.0
      timing: 0
```

---

## Trial Phase Patterns

### Simple single-side pulse (N seconds on, then ISI)
```yaml
- phase: "TRIAL - LEFT ONLY"
  duration: 70000                # odor_duration + ISI_duration in ms
  repeat: 0
  actions:
    - device: "triggers.microscope"
      state: true
      timing: 0                  # trial start marker
    - device: "switch_valve.right"
      state: "CLEAN"             # explicit: unused side stays clean
      timing: 0
    - device: "switch_valve.left"
      state: "ODOR"
      timing: 0
    - device: "switch_valve.left"
      state: "CLEAN"
      timing: 30000              # odor off after 30 s
    - device: "triggers.microscope"
      state: true
      timing: 30000              # trial end marker
    # ISI: 40 s of silence → total duration = 70 000 ms
```

### Multi-pulse pattern (M pulses × P ms on / P ms off)
```yaml
# Example: 3 × 5 s on / 5 s off
- phase: "TRIAL - BILATERAL 3-PULSE"
  duration: 65000                # 25 s pulses + 40 s ISI
  repeat: 0
  actions:
    - device: "triggers.microscope"
      state: true
      timing: 0
    - device: "switch_valve.left"
      state: "ODOR"
      timing: 0
    - device: "switch_valve.right"
      state: "ODOR"
      timing: 0
    - device: "switch_valve.left"
      state: "CLEAN"
      timing: 5000
    - device: "switch_valve.right"
      state: "CLEAN"
      timing: 5000
    - device: "switch_valve.left"
      state: "ODOR"
      timing: 10000
    - device: "switch_valve.right"
      state: "ODOR"
      timing: 10000
    - device: "switch_valve.left"
      state: "CLEAN"
      timing: 15000
    - device: "switch_valve.right"
      state: "CLEAN"
      timing: 15000
    - device: "switch_valve.left"
      state: "ODOR"
      timing: 20000
    - device: "switch_valve.right"
      state: "ODOR"
      timing: 20000
    - device: "switch_valve.left"
      state: "CLEAN"
      timing: 25000
    - device: "switch_valve.right"
      state: "CLEAN"
      timing: 25000
    - device: "triggers.microscope"
      state: true
      timing: 25000
```

### Overlap trial (sequential with offset)
```yaml
# Left starts at t=0, right follows at t=OFFSET_ms.
# Overlap window = OFFSET_ms to left_end_ms.
# OFFSET must be ≥ 3 ms for guardrail safety.
- phase: "TRIAL - LEFT THEN RIGHT"
  duration: 70000
  repeat: 0
  actions:
    - device: "triggers.microscope"
      state: true
      timing: 0
    - device: "switch_valve.left"
      state: "ODOR"
      timing: 0
    - device: "switch_valve.right"
      state: "CLEAN"
      timing: 0                  # explicit no-odor until offset
    - device: "triggers.microscope"
      state: true
      timing: 10000              # right opens / overlap begins
    - device: "switch_valve.right"
      state: "ODOR"
      timing: 10000
    - device: "switch_valve.left"
      state: "CLEAN"
      timing: 30000
    - device: "switch_valve.right"
      state: "CLEAN"
      timing: 40000
    - device: "triggers.microscope"
      state: true
      timing: 40000
```

### Repeated trials (different odor each repeat)
```yaml
- phase: "ODOR IDENTITY TRIAL"
  duration: 90000
  times: 5                       # runs 5 times (replaces repeat: 4)
  randomize: true                # shuffle state list across repeats
  actions:
    - device: "olfactometer.left"
      state: "ODOR1,ODOR2,ODOR3,ODOR4,ODOR5"   # one per repeat
      timing: 0
    - device: "switch_valve.left"
      state: "ODOR"
      timing: 30000
    - device: "switch_valve.left"
      state: "CLEAN"
      timing: 60000
```

---

## Device Rules Cheat Sheet

| Rule | Detail |
|---|---|
| Simultaneous left+right writes | Offset by ≥ 5 ms: left at `T`, right at `T+5` |
| Minimum event spacing (same side) | ≥ 3 ms between any two timings on the same valve |
| MFC setpoint range | 0.0 – 5.0 V; typical odor flow = 0.03 V |
| Camera continuous | Start once in BOOT, stop once in SHUTDOWN |
| Microscope triggers | Use at trial start and end for alignment |
| Switch valves idle state | Always `CLEAN` between trials and in non-trial phases |
| Olfactometers idle state | `AIR` during saturation/desaturation, `OFF` only in SHUTDOWN |

---

## Timing Calculation Guide

Given `pulse_duration_s` and `ISI_s`:
```
phase_duration_ms = (pulse_duration_s + ISI_s) × 1000
```

For N pulses of P ms on / P ms off:
```
active_window_ms = N × (P_on + P_off) - P_off   # last off is ISI
                 = N × P_on + (N-1) × P_off
ISI_ms           = ISI_s × 1000
phase_duration_ms = active_window_ms + ISI_ms
```

For overlap trials with OFFSET_ms between sides:
```
total_active_ms  = single_side_window_ms + OFFSET_ms
overlap_window_ms = single_side_window_ms - OFFSET_ms   # (must be > 0)
phase_duration_ms = total_active_ms + ISI_ms
```

---

## Validation Checklist Before Running

- [ ] All `timing:` values are integers (no decimals)
- [ ] No two events on the same side differ by < 3 ms
- [ ] Left and right simultaneous-looking events are offset ≥ 5 ms
- [ ] Every trial phase starts with a `triggers.microscope` at `timing: 0`
- [ ] Every trial phase ends with a `triggers.microscope` at `timing: <odor_end>`
- [ ] Switch valves are explicitly set to `CLEAN` for unused side in unilateral trials
- [ ] `duration:` is always greater than the largest `timing:` value in that phase
- [ ] Olfactometers are set to odor in SATURATION (before trials)
- [ ] SHUTDOWN zeros all MFC `value:` to `0.0`
- [ ] Dry-run passes without errors: `python -m multibios.run_protocol --yaml config/<file>.yaml --hardware config/hardware.yaml --dry-run`

---

## YAML Anchors (Best Practice)

Define the active odor and flow once at the top and reference everywhere:
```yaml
_active_odor: &odor "ODOR2"        # change here → updates all phases
_mfc_flow: &mfc_flow 0.03

# Then use:
state: *odor
value: *mfc_flow
```

---

## Naming Conventions

- File names: `snake_case.yaml` (e.g. `odor_concentration_ramp.yaml`)
- Phase names: `"TRIAL N - CONDITION NAME"` (all caps, descriptive)
- Use comment blocks `# ──────────` to separate phases visually
- Include a header comment block with protocol purpose and odor anchor instructions
