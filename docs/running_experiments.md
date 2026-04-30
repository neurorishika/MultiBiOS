# Running a New Experiment

This guide walks you through everything needed to design, validate, and execute an olfactometer experiment with MultiBiOS — from first boot to data saved on disk.

---

## Before You Start — Hardware Checklist

Complete this checklist each session before launching any software:

- [ ] **NI USB-6353** connected via USB; visible in NI MAX as `Dev1`
- [ ] **Teensy 4.1** connected via USB; firmware uploaded (see [`docs/firmware.md`](firmware.md))
- [ ] **Shift-register chains** (TPIC6B595) wired to Teensy SPI (MOSI=Pin 11, SCK=Pin 13)
- [ ] **Olfactometer valve arrays** powered (12 V solenoids) and connected to shift-register outputs
- [ ] **MFCs** powered; setpoint cables to DAQ `ao0–ao3`; feedback cables to DAQ `ai0–ai3`
- [ ] **Gas lines** connected: air and odor carrier gas flows to left/right olfactometer inlets
- [ ] Windows Device Manager — confirm Teensy COM port is assigned and no yellow warning icons
- [ ] NI MAX — confirm device name is `Dev1` (if different, update `config/hardware.yaml`)

> **Tip:** If this is the first session on this PC, follow the one-time [Software Setup](#software-setup) section below first.

---

## Software Setup (First Time Only)

```powershell
# 1. Create the shared MultiBiOS + Blackfly camera environment:
cd C:\Rishika\MultiBiOS
conda env create -f environment.yml
conda activate multibios-blackfly

# 2. Verify the DAQ + camera stack:
python -c "import multibios, nidaqmx, PySpin; print('multibios-blackfly ready')"

# 3. Verify the protocol runner:
python -m multibios.run_protocol --help
```

Use the shared Conda environment named `multibios-blackfly` so the DAQ control code and the Teledyne FLIR Blackfly S cameras run in the same Python installation.

Rig camera model for this setup:

- Front camera: **Blackfly S BFS-U3-13Y3M**
- Side/FicTrac camera: **Blackfly S BFS-U3-13Y3M**

If you want to use the side camera with live FicTrac on this rig, build and use the Spinnaker-enabled binary documented in [`docs/fictrac.md`](fictrac.md). The validated packaged path on this workstation is `C:/Rishika/MultiBiOS/assets/fictrac-spinnaker/fictrac-spinnaker.exe`.

NI-DAQmx drivers must be installed separately from
[ni.com/downloads](https://www.ni.com/en/support/downloads/drivers/download.ni-daq-mx.html), and the matching Spinnaker SDK must be installed before the bundled `PySpin` wheel.

---

## Step 1 — Choose or Create a Protocol

Protocol files live in `config/`. Existing protocols:

| File | What it tests |
|---|---|
| `odor_lateralization.yaml` | 5 conditions: bilateral, left-only, right-only, left→right overlap, right→left overlap. Single 30 s pulses. |
| `odor_lateralization_3pulse.yaml` | Same 5 conditions with 3 × 5 s pulses instead. |
| `example_protocol.yaml` | Full-featured reference with randomized odor delivery. |
| `short_protocol.yaml` | Abbreviated 9 s trials — good for hardware sanity checks. |
| `switch_valve_test.yaml` | Switch valve toggling only — no odor. |

**To use an existing protocol**, skip to [Step 2](#step-2--set-the-odor).

**To create a new protocol**, open a copy of the closest existing YAML and edit it.
See [`docs/protocol.md`](protocol.md) for the full YAML schema, or ask Copilot — the design
instructions at `.github/instructions/design-protocol.instructions.md` provide a full
template and all device rules.

---

## Step 2 — Set the Odor

Every `odor_lateralization*.yaml` file has a single anchor at the top:

```yaml
_active_odor: &odor "ODOR1"       # ← change this line only
_mfc_odor_flow_slpm: &mfc_flow 0.03
```

Change `"ODOR1"` to whichever odor channel is loaded today (`ODOR1`–`ODOR5`).
This propagates automatically to every phase — you do **not** need to edit individual actions.

Valid odor values: `ODOR1`  `ODOR2`  `ODOR3`  `ODOR4`  `ODOR5`  `FLUSH`

---

## Step 3 — Dry Run (Validate Without Hardware)

Always validate the protocol before touching real hardware:

```powershell
cd C:\Rishika\MultiBiOS

python -m multibios.run_protocol `
  --yaml config/odor_lateralization.yaml `
  --hardware config/hardware.yaml `
  --dry-run --interactive
```

This compiles the protocol and writes a preview to a new timestamped folder:

```
data/runs/YYYY-MM-DD_HH-MM-SS/
  preview.html        ← open this in your browser
  digital_edges.csv   ← every valve transition with timestamps
  compile_report.json ← timing log and guardrail summary
```

**Open `preview.html`** in Chrome or Edge and verify:
- Odor pulses appear at the right times on the right valves
- Microscope trigger markers align with trial start/end
- Camera trigger runs continuously throughout
- MFC setpoints ramp to the expected flow voltage at boot and drop to 0 at shutdown
- No overlapping events on the same valve

If the compiler finds a timing violation it will print a `GuardrailViolation` error and
stop. Fix the offending timing offset in the YAML (space events ≥ 3 ms apart on the same
side) and re-run the dry run.

---

## Step 4 — Connect the Animal and Begin Imaging

1. Mount the animal and confirm head fixation.
2. Start your imaging/recording software (two-photon, widefield, electrophysiology, etc.)
   and set it to **wait for an external trigger** on the microscope trigger line.
3. Leave the imaging software armed — MultiBiOS will send the first microscope pulse at the
   end of the saturation phase to mark "trials starting".

---

## Step 5 — Run the Experiment

```powershell
cd C:\Rishika\MultiBiOS

poetry run python -m multibios.run_protocol `
  --yaml config/odor_lateralization.yaml `
  --hardware config/hardware.yaml `
  --verbose --progress
```

The terminal prints a real-time progress bar:

```
[  5%]  3250.0ms | DO:░█░░░ | AO:0:0.03,1:0.03,2:0.03,3:0.03
[ 12%]  9100.0ms | DO:█░░░█ | AO:0:0.03,1:0.03,2:0.03,3:0.03
...
[100%] Protocol execution complete
```

**Do not close the terminal or disconnect hardware while the bar is running.**

The protocol ends automatically. All output is saved immediately to
`data/runs/<timestamp>/`.

---

## Step 6 — Verify the Data

After the run, open `data/runs/<timestamp>/capture_ai.npz` (MFC feedback) and
`digital_edges.csv` to confirm the hardware responded correctly:

```powershell
# Quick check — view edge timestamps
Get-Content data/runs/<timestamp>/digital_edges.csv | Select-Object -First 30
```

Or re-open the `preview.html` from this run — for hardware runs it will include
the actual AI (MFC feedback) traces overlaid on the planned waveform.

---

## Step 7 — Re-run with the Same Randomization (Optional)

If your protocol uses randomized odor orders, the seed is saved in `meta.json`.
To reproduce an identical run:

```powershell
poetry run python -m multibios.run_protocol `
  --yaml config/my_protocol.yaml `
  --hardware config/hardware.yaml `
  --seed <seed_from_meta.json>
```

---

## Output File Reference

Every run creates `data/runs/YYYY-MM-DD_HH-MM-SS/`:

| File | Description |
|---|---|
| `preview.html` | Interactive Plotly waveform viewer |
| `compiled_do.npz` | Digital output array — shape `(num_lines, samples)` |
| `compiled_ao.npz` | Analog output array — shape `(num_ao, samples)` |
| `capture_ai.npz` | MFC flow feedback recorded during the run |
| `digital_edges.csv` | Every rising/falling edge with ms timestamp |
| `rck_edges.csv` | Register-clock commit times |
| `compile_report.json` | Phases, guardrail log, RCK edge count |
| `meta.json` | Device name, sample rate, seed, CLI args used |
| `protocol.yaml` | Exact copy of the protocol that was run |
| `hardware.yaml` | Exact copy of the hardware config used |

---

## Common Problems

| Symptom | Likely Cause | Fix |
|---|---|---|
| `GuardrailViolation` on compile | Two events on the same valve < 3 ms apart | Increase spacing between `timing:` values |
| `DAQmxError -200220` | NI-DAQ not found | Check USB, confirm device name in NI MAX matches `hardware.yaml` |
| `ModuleNotFoundError: multibios` | Wrong environment | Run `poetry install` in `MultiBiOS/`, then use `poetry run ...` |
| Valves don't respond | Teensy not ready | Check Teensy USB, confirm firmware; watch for READY signal in `di_edges.csv` |
| MFC feedback all zeros | AI wiring or ground issue | Check MFC feedback cables to `ai0–ai3` and common ground |
| Progress bar never starts | DAQ task not starting | Check hardware connection; try `--debug` for detailed log |
| Wrong odor delivered | YAML anchor not updated | Make sure you changed `_active_odor: &odor "ODORx"` at the top of the YAML |

For more diagnostics see [`docs/troubleshooting.md`](troubleshooting.md).

---

## Quick Reference — Common Commands

```powershell
# Validate a protocol (no hardware needed)
poetry run python -m multibios.run_protocol --yaml config/<file>.yaml --hardware config/hardware.yaml --dry-run --interactive

# Run on hardware with live progress
poetry run python -m multibios.run_protocol --yaml config/<file>.yaml --hardware config/hardware.yaml --verbose --progress

# Run with a fixed random seed
poetry run python -m multibios.run_protocol --yaml config/<file>.yaml --hardware config/hardware.yaml --seed 42

# Regenerate visualization from saved data
poetry run python -m multibios.viz_protocol data/runs/<timestamp>/

# Run hardware sanity check (short protocol, no animal needed)
poetry run python -m multibios.run_protocol --yaml config/short_protocol.yaml --hardware config/hardware.yaml --verbose --progress
```

All commands should be run from `C:\Rishika\MultiBiOS\`.

---

## Running Experiments with FicTrac Integration (`experiment.py`)

> **Use this runner when you need ball-tracking data (FicTrac) synchronized with your odor delivery.** It uses the same YAML protocol files as `run_protocol.py` but executes valve control over computer-timed serial rather than hardware-clocked DAQ waveforms, and records every FicTrac frame alongside the experiment event log.

Before using the live Blackfly side camera with this runner, first verify the rebuilt binary with the probe flow described in [`docs/fictrac.md`](fictrac.md): start `tests/continuous_camera_trigger.py`, then run `tests/fictrac_live_probe.py` against `assets/fictrac-spinnaker/fictrac-spinnaker.exe`.

### How it differs from `run_protocol.py`

| | `run_protocol.py` | `experiment.py` |
|---|---|---|
| Valve control | Hardware-clocked NI-DAQ waveform (sub-ms precision) | Computer-timed serial to Teensy (~1–5 ms jitter) |
| MFC control | DAQ analog output | Alicat serial (dedicated COM ports) |
| FicTrac | Not integrated | Fully integrated — records every frame |
| Camera/scope triggers | Embedded in DAQ waveform | Separate finite NI-DAQ task (latch pulses at `latch_interval_ms`) |
| Best for | Precise timing, no tracking needed | Ball-walking experiments, closed-loop readiness |
| Output files | `compiled_do.npz`, `capture_ai.npz`, `preview.html` | `experiment_data.csv`, `event_log.csv`, `trigger_waveform.npz` |

---

### Extra Hardware Checklist (experiment runner only)

In addition to the base checklist above:

- [ ] **FicTrac camera** mounted and calibrated; config file at `C:/Rishika/fictrac_pybmt/config_camera.txt`
- [ ] **FicTrac binary** built for the right camera backend on this PC. For the Blackfly S rig cameras, prefer the Spinnaker build path documented in [`docs/fictrac.md`](fictrac.md)
- [ ] **Alicat MFC controllers** connected; COM ports noted (run `python -m multibios.apps.flow_monitor --scan` to discover addresses A–D)
- [ ] **Teensy COM port** confirmed in Device Manager (e.g. `COM4`) and updated in `config/experiment_config.yaml`

---

### Step 1 — Configure `experiment_config.yaml`

Open `config/experiment_config.yaml` and update:

```yaml
# Teensy serial port — check Device Manager
teensy_port: "COM4"
teensy_baud: 115200

# FicTrac paths
fictrac_config: "C:/Rishika/fictrac_pybmt/config_camera.txt"
fictrac_bin:    "C:/Rishika/MultiBiOS/assets/fictrac-spinnaker/fictrac-spinnaker.exe"
fictrac_console_out: "fictrac_output.txt"
fictrac_timeout_s: 5.0     # abort if no new frames for this long

# MFC mode: "alicat_serial" or "none" (skip MFC entirely)
mfc_mode: "alicat_serial"

# Map protocol device keys -> Alicat unit single-letter IDs
# Run: python -m multibios.apps.flow_monitor --scan   to find these
mfc_device_map:
  mfc.air_left_setpoint:   "A"
  mfc.air_right_setpoint:  "B"
  mfc.odor_left_setpoint:  "C"
  mfc.odor_right_setpoint: "D"

alicat_ports: ["COM7", "COM8", "COM9", "COM10"]
alicat_baud: [115200]

# Latch interval — how often Teensy commits staged valve changes to hardware
latch_interval_ms: 10.0    # 10 ms = 100 Hz (faster = more responsive, more DAQ load)

# Live MFC readout interval during the run (0 = disable)
mfc_live_interval_s: 1.0

data_dir: "data/runs"
```

> **Finding Alicat addresses:** Run `python -m multibios.apps.flow_monitor --scan` — it prints a table of all discovered devices and their letter IDs. Enter those letters in `mfc_device_map`.

---

### Step 2 — Dry Run (preview timeline, no hardware)

```powershell
cd C:\Rishika\MultiBiOS

poetry run python -m multibios.experiment `
  --protocol config/odor_lateralization.yaml `
  --hardware config/hardware.yaml `
  --experiment config/experiment_config.yaml `
  --dry-run --verbose
```

This prints the full timeline in order — every valve command, MFC setpoint, and trigger marker — without opening any serial ports or starting FicTrac.

Sample output:
```
Protocol: Odor Lateralization
Total duration: 421.0 s
Timeline events: 87
Microscope triggers: 14

--- Phase: BOOT SEQUENCE ---
  [    1.00s] MFC mfc.air_left_setpoint     ->  0.03
  [    1.00s] OLF left   -> AIR
  ...

--- Phase: TRIAL 1 - BILATERAL ---
  [   76.00s] DAQ: triggers.microscope -> PULSE
  [   76.00s] SV  left  -> ODOR
  ...
```

---

### Step 3 — Connect the Animal and Arm FicTrac

1. Mount the animal on the ball.
2. Launch FicTrac manually **once** to confirm the camera sees the ball and tracking is working, then close it — the experiment runner will launch it automatically.
3. Arm your imaging software to wait for the microscope trigger.

### NI-DAQ-triggered FicTrac on the Blackfly rig

When the side camera is externally triggered by NI-DAQ, the timing model is:

1. MultiBiOS starts the finite NI-DAQ trigger waveform.
2. MultiBiOS launches the FicTrac thread immediately after the DAQ task is armed.
3. The DAQ `TRIG_CAMERA` line issues `FrameStart` pulses to the Blackfly.
4. FicTrac only receives frames when those pulses occur.

Treat the DAQ waveform as the authoritative timing reference. FicTrac's host-side frame arrival timestamps are useful for health monitoring, but not the primary synchronization clock for the experiment.

For per-frame proof that exposures actually happened, validate a camera return line into DAQ before relying on it in analysis.

---

### Step 4 — Run the Experiment

```powershell
cd C:\Rishika\MultiBiOS

poetry run python -m multibios.experiment `
  --protocol config/odor_lateralization.yaml `
  --hardware config/hardware.yaml `
  --experiment config/experiment_config.yaml `
  --verbose
```

**Startup sequence you will see:**
```
Opening Teensy on COM4...
  Teensy RESET: OK
  Starting MFC monitor (live readout every 1 s)...
Starting NIDAQ trigger task...
  NIDAQ running (421.0 s finite task)
  Waiting for FicTrac first frame...
  FicTrac connected (frame 1)

════════════════════════════════════════════════════
EXPERIMENT RUNNING
════════════════════════════════════════════════════

  [    1.00s] OLF left   -> AIR       (jitter +0.3 ms)
  [    1.01s] MFC mfc.air_left_setpoint -> 0.03  (jitter +0.2 ms)
  [   12.0s] MFC: A@COM7 flow=10.001 setpt=10.0 gas=Air  |  B@COM7 flow=10.002 setpt=10.0 gas=Air
  ...
```

The jitter column shows how close the actual event dispatch was to the scheduled time. Typical values are ±1–5 ms under normal CPU load.

**Do not close the terminal or disconnect hardware while the experiment is running.**

---

### Step 5 — Verify and Explore Data

When the run finishes, the data explorer opens automatically in your browser at `http://127.0.0.1:8050`. You can also re-open it any time:

```powershell
python -m multibios.apps.explorer
```

Output directory `data/runs/YYYY-MM-DD_HH-MM-SS/` contains:

| File | Description |
|---|---|
| `experiment_data.csv` | **Primary analysis file** — one row per FicTrac frame with all valve states, MFC setpoints, and camera trigger forward-filled |
| `event_log.csv` | Every valve/MFC/trigger event with scheduled vs. actual time and jitter |
| `event_log.json` | Same events in JSON (full fidelity, extra fields) |
| `timeline.csv` | Compiled protocol schedule (reference) |
| `trigger_waveform.npz` | NI-DAQ latch/camera/microscope waveform |
| `protocol.yaml` / `hardware.yaml` | Input copies |
| `meta.json` | Config, seed, timestamps |

**`experiment_data.csv` columns:**

| Column | Description |
|---|---|
| `experiment_time_s` | Seconds since experiment start |
| `frame_cnt` | FicTrac frame number |
| `posx`, `posy` | Integrated ball position (ball radii) |
| `heading` | Current heading angle (radians) |
| `speed` | Ball speed (ball radii / s) |
| `direction` | Movement direction (radians) |
| `intx`, `inty` | Integrated X/Y (FicTrac units) |
| `olfactometer_left/right` | Current odor valve state (AIR / ODOR1-5 / OFF) |
| `switch_valve_left/right` | Current switch valve state (CLEAN / ODOR) |
| `mfc_air_left/right_sp` | MFC setpoint commanded (SLPM) |
| `mfc_odor_left/right_sp` | MFC setpoint commanded (SLPM) |
| `camera_trigger` | 1 if camera trigger is HIGH in DAQ waveform |

---

### CLI Reference — `experiment.py`

```
poetry run python -m multibios.experiment [OPTIONS]

Required:
  --protocol FILE       Protocol YAML  (same format as run_protocol)
  --hardware FILE       Hardware mapping YAML
  --experiment FILE     Experiment config YAML (default: config/experiment_config.yaml)

Execution:
  --dry-run             Preview timeline only; no hardware
  --verbose / -v        Print each event with jitter as it fires
  --seed INT            Override protocol RNG seed

Output:
  --out-root DIR        Output root (default: from experiment_config.yaml data_dir)
```

---

### Common Problems (experiment runner)

| Symptom | Likely Cause | Fix |
|---|---|---|
| `FicTrac did not produce any frames within 90 s` | Camera not found, FicTrac crashed, or the binary still has the old short first-frame wait | Check camera USB, run FicTrac manually to confirm it works, and verify the packaged `fictrac-spinnaker.exe` is the patched custom build documented in `docs/fictrac.md` |
| `No cached Alicat device matches mapping` | Wrong letter ID or COM port | Run `python -m multibios.apps.flow_monitor --scan` and update `mfc_device_map` in experiment_config.yaml |
| `Teensy RESET: ERROR` | Wrong COM port or firmware not running | Check Device Manager for correct port; re-flash firmware |
| FicTrac tracking looks noisy mid-run | Ball surface dirty or lighting changed | Check illumination; clean ball; re-calibrate FicTrac |
| `FicTrac stopped producing frames for N s` | FicTrac crashed mid-run | Check `fictrac_output.txt` for error; increase `fictrac_timeout_s` if just a hiccup |
| High jitter (> 20 ms) on valve events | CPU load from FicTrac or other processes | Close non-essential applications; consider increasing `latch_interval_ms` to reduce DAQ polling |
| MFC live readout not appearing | `mfc_live_interval_s: 0` in config | Set to `1.0` or higher |
