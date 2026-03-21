# MultiBiOS — GitHub Copilot Workspace Instructions

## Project Overview
MultiBiOS is a bilateral olfactometer control system that compiles YAML protocol files into NI-DAQ hardware waveforms. It drives Teensy 4.1 microcontrollers with TPIC6B595 shift-register chains to control odor valves, switch valves, and mass-flow controllers (MFCs) with sub-millisecond precision.

**Stack:** Python 3.10 · NI-DAQmx (USB-6353) · Teensy 4.1 · YAML protocols

---

## Repository Layout

```
MultiBiOS/
├── multibios/              # Core Python package
│   ├── run_protocol.py     # Main CLI entry point (hardware runner)
│   ├── experiment.py       # Alternative computer-timebase runner
│   ├── protocol/
│   │   └── schema.py       # YAML → DO/AO array compiler (ProtocolCompiler)
│   ├── alicat_manager.py   # MFC serial control
│   ├── daq_triggers.py     # NI-DAQ wrappers
│   ├── teensy_controller.py
│   ├── viz_helpers.py / viz_protocol.py
│   └── __init__.py
├── config/                 # Protocol and hardware YAML files
│   ├── hardware.yaml       # NI-DAQ device/line mapping (do not edit casually)
│   ├── experiment_config.yaml
│   ├── example_protocol.yaml
│   ├── odor_lateralization.yaml
│   ├── odor_lateralization_3pulse.yaml
│   ├── short_protocol.yaml
│   ├── switch_valve_test.yaml
│   └── latcher.yaml
├── data/runs/              # Timestamped output directories (auto-created)
├── docs/                   # Full documentation (protocol.md, runner.md, timing.md…)
├── tests/
└── firmware/teensy41/
```

---

## Device Reference

### All valid device keys and their types

| Device Key | Type | States / Range |
|---|---|---|
| `olfactometer.left` | Digital 3-bit | `OFF` `AIR` `ODOR1` `ODOR2` `ODOR3` `ODOR4` `ODOR5` `FLUSH` |
| `olfactometer.right` | Digital 3-bit | same as above |
| `switch_valve.left` | Digital 1-bit | `CLEAN` `ODOR` |
| `switch_valve.right` | Digital 1-bit | `CLEAN` `ODOR` |
| `mfc.air_left_setpoint` | Analog (AO) | `value: 0.0` – `5.0` (volts) |
| `mfc.air_right_setpoint` | Analog (AO) | `value: 0.0` – `5.0` |
| `mfc.odor_left_setpoint` | Analog (AO) | `value: 0.0` – `5.0` |
| `mfc.odor_right_setpoint` | Analog (AO) | `value: 0.0` – `5.0` |
| `triggers.microscope` | Pulse | `state: true` at desired timing |
| `triggers.camera_continuous` | Periodic | `state: true` to start, `state: false` to stop |

**MFC note:** Typical flow is `0.03` SLPM → set `value: 0.03`. Map to volts using MFC datasheet if different.

---

## Multi-State Syntax for Olfactometers

```yaml
state: "ODOR1"                          # constant across all repeats
state: "ODOR1,ODOR2,ODOR3"             # sequential: repeat 0→ODOR1, 1→ODOR2, 2→ODOR3
state: "ODOR1|ODOR2|ODOR3"             # random pick each repeat (needs seed in timing)
state: "ODOR1,ODOR2|ODOR3,ODOR4"       # index 0→ODOR1, index 1→random(ODOR2,ODOR3), etc.
```

---

## Timing Rules (Critical)

- All `timing:` values are **milliseconds** (integers) from the start of the phase.
- `timing: 0` fires at phase start; `timing: <duration>` fires at end.
- The compiler enforces a **no-overlap guardrail**: the preload window of one event cannot overlap with another on the same shift-register assembly. Default preload is `preload_lead_ms: 2`.
- Minimum safe spacing between events on the same side: **≥ 3 ms**.
- `duration:` is the total phase length in ms; if repeating, each repeat is this long.
- Use `times: N` for repeats (preferred); `repeat: N` is legacy (0 = run once).

---

## Standard Protocol Structure

Every protocol MUST follow this phase order:
1. **BOOT SEQUENCE** — start all MFC flows, enable camera, set olfactometers to `AIR`, switch valves to `CLEAN`
2. **OLFACTOMETER SATURATION** — set olfactometers to odor, hold 60 s with switch valves CLEAN; microscope pulse at end
3. **TRIALS** — one phase per condition; start/end with microscope pulses
4. **OLFACTOMETER DESATURATION** — return olfactometers to `AIR`, hold 60 s; microscope pulse at end
5. **SHUTDOWN SEQUENCE** — zero all MFC setpoints, stop camera, set olfactometers to `OFF`

---

## Output Files (data/runs/YYYY-MM-DD_HH-MM-SS/)

| File | Contents |
|---|---|
| `compiled_do.npz` | Digital output array (num_lines × samples) |
| `compiled_ao.npz` | Analog output array (num_ao × samples) |
| `capture_ai.npz` | MFC feedback recordings (hardware runs only) |
| `compile_report.json` | Timing log, phases, RCK edges |
| `digital_edges.csv` | All DO rising/falling transitions |
| `rck_edges.csv` | Planned register-clock commit times |
| `preview.html` | Interactive Plotly visualization |
| `protocol.yaml` / `hardware.yaml` | Copies of inputs |
| `meta.json` | Device name, duration, CLI args used |

---

## Key Docs
- `docs/protocol.md` — Full YAML schema reference
- `docs/runner.md` — CLI flags, output structure, post-run viz
- `docs/timing.md` — Event anatomy, preload/commit windows, guardrails
- `docs/hardware.md` — Signal naming, DO/AO/AI/DI mappings
- `docs/faq.md` — Common Q&A
- `docs/troubleshooting.md`
