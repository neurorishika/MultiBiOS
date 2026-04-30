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

> **Tip:** If this is the first session on this PC, follow the one-time [Software Setup](#software-setup-first-time-only) section below first.

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

## Choose the Runner

Use the supported execution path below.

| Runner | When to use it | Timing model | FicTrac | Typical outputs |
| --- | --- | --- | --- | --- |
| `python -m multibios.run_protocol` | Default path for all new experiments | Hardware-clocked NI-DAQ waveform | Supported when `hardware.yaml -> fictrac` is configured | `compiled_do.npz`, `capture_ai.npz`, optional `fictrac_frames.npz` |

!!! warning "Legacy serial runner"
  `multibios.experiment` is deprecated. Its operational notes have moved to [legacy/serial_experiment_pipeline.md](legacy/serial_experiment_pipeline.md) and are intentionally hidden from the default docs navigation.

---

## Step 1 — Choose or Create a Protocol

Protocol files live in `config/`. Existing protocols:

| File | What it tests |
| --- | --- |
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

conda run -n multibios-blackfly python -m multibios.run_protocol `
  --yaml config/odor_lateralization.yaml `
  --hardware config/hardware.yaml `
  --dry-run --interactive
```

This compiles the protocol and writes a preview to a new timestamped folder:

```text
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

Before a long real experiment, run one bounded hardware sanity check on the rig:

```powershell
cd C:\Rishika\MultiBiOS

conda run -n multibios-blackfly python -m multibios.run_protocol `
  --yaml config/short_protocol.yaml `
  --hardware config/hardware.yaml `
  --verbose --progress
```

That validated command finishes on its own in a few seconds and confirms the current DAQ/FicTrac/camera stack on this workstation.

---

## Step 5 — Run the Experiment

```powershell
cd C:\Rishika\MultiBiOS

conda run -n multibios-blackfly python -m multibios.run_protocol `
  --yaml config/odor_lateralization.yaml `
  --hardware config/hardware.yaml `
  --verbose --progress
```

If `config/hardware.yaml` contains a `fictrac:` block, `run_protocol` now also:

- prepares the Spinnaker runtime path for the child process
- launches FicTrac automatically
- waits for the first UDP frame using `fictrac.startup_timeout_s`
- writes run-local FicTrac artifacts into the same `data/runs/<timestamp>/` folder

Use `fictrac.first_frame_timeout_ms: 0` to make the native FicTrac layer wait indefinitely for the first frame, and `fictrac.startup_timeout_s: 0` to make the Python runner wait indefinitely for the first UDP frame.

The terminal prints a real-time progress bar:

```text
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
conda run -n multibios-blackfly python -m multibios.run_protocol `
  --yaml config/my_protocol.yaml `
  --hardware config/hardware.yaml `
  --seed <seed_from_meta.json>
```

---

## Output File Reference

Every run creates `data/runs/YYYY-MM-DD_HH-MM-SS/`:

| File | Description |
| --- | --- |
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

When FicTrac is enabled for `run_protocol`, the same run directory also includes:

| File | Description |
| --- | --- |
| `fictrac_runtime_config.txt` | The exact runtime config passed to FicTrac |
| `fictrac_runtime.json` | MultiBiOS-side summary of the runtime config edits |
| `fictrac_driver_diagnostics.json` | Launch diagnostics including first-packet timing and frame count |
| `fictrac_frames.npz` | Saved FicTrac frames from the internal client |
| `fictrac-*.dat` | Native FicTrac output |

---

## Common Problems

| Symptom | Likely Cause | Fix |
| --- | --- | --- |
| `GuardrailViolation` on compile | Two events on the same valve < 3 ms apart | Increase spacing between `timing:` values |
| `DAQmxError -200220` | NI-DAQ not found | Check USB, confirm device name in NI MAX matches `hardware.yaml` |
| `ModuleNotFoundError: multibios` | Wrong environment | Activate or use `multibios-blackfly`, then run `conda run -n multibios-blackfly ...` |
| `FicTrac did not produce any frames within N s` | FicTrac camera config or startup wait mismatch | Check `hardware.yaml -> fictrac`, verify `fictrac-spinnaker.exe`, and use `first_frame_timeout_ms: 0` plus `startup_timeout_s: 0` when you want indefinite startup wait |
| Valves don't respond | Teensy not ready | Check Teensy USB, confirm firmware; watch for READY signal in `di_edges.csv` |
| MFC feedback all zeros | AI wiring or ground issue | Check MFC feedback cables to `ai0–ai3` and common ground |
| Progress bar never starts | DAQ task not starting | Check hardware connection; try `--debug` for detailed log |
| Wrong odor delivered | YAML anchor not updated | Make sure you changed `_active_odor: &odor "ODORx"` at the top of the YAML |

For more diagnostics see [`docs/troubleshooting.md`](troubleshooting.md).

---

## Quick Reference — Common Commands

```powershell
# Validate a protocol (no hardware needed)
conda run -n multibios-blackfly python -m multibios.run_protocol --yaml config/<file>.yaml --hardware config/hardware.yaml --dry-run --interactive

# Run on hardware with live progress
conda run -n multibios-blackfly python -m multibios.run_protocol --yaml config/<file>.yaml --hardware config/hardware.yaml --verbose --progress

# Run with a fixed random seed
conda run -n multibios-blackfly python -m multibios.run_protocol --yaml config/<file>.yaml --hardware config/hardware.yaml --seed 42

# Regenerate visualization from saved data
conda run -n multibios-blackfly python -m multibios.viz_protocol data/runs/<timestamp>/

# Run hardware sanity check (short protocol, no animal needed)
conda run -n multibios-blackfly python -m multibios.run_protocol --yaml config/short_protocol.yaml --hardware config/hardware.yaml --verbose --progress

```

All commands should be run from `C:\Rishika\MultiBiOS\`.
If you still need the deprecated serial/Alicat `experiment.py` workflow, its commands, config notes, and troubleshooting now live in [legacy/serial_experiment_pipeline.md](legacy/serial_experiment_pipeline.md).
