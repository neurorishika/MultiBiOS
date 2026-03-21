---
applyTo: "**"
---

# Running MultiBiOS Protocols

## Quick Reference

| Goal | Command |
|---|---|
| Dry-run (no hardware, generate preview) | `python -m multibios.run_protocol --yaml config/<file>.yaml --hardware config/hardware.yaml --dry-run --interactive` |
| Execute on hardware | `python -m multibios.run_protocol --yaml config/<file>.yaml --hardware config/hardware.yaml --verbose --progress` |
| Validate YAML only | `python -m multibios.run_protocol --yaml config/<file>.yaml --hardware config/hardware.yaml --dry-run` |
| Reproducible run with fixed seed | add `--seed 42` |
| Override timing params | add `--preload-lead-ms 3 --load-req-ms 1 --rck-ms 1 --trig-ms 5` |

All commands should be run from `c:\Rishika\MultiBiOS\`.

---

## Full CLI Reference

```
python -m multibios.run_protocol [OPTIONS]

Required:
  --yaml PATH           Protocol YAML file (default: config/example_protocol.yaml)
  --hardware PATH       Hardware mapping YAML (default: config/hardware.yaml)

Execution mode:
  --dry-run             Compile + validate only, no hardware output
  --interactive         Always save HTML preview (implied by --dry-run)

Output:
  --out-root PATH       Output directory root (default: data/runs)
  --verbose / -v        INFO-level logging
  --debug               DEBUG-level logging (very detailed, impacts timing)

Progress monitoring:
  --progress            Print real-time progress bar during execution
  --progress-interval N Update every N ms (default: 100)

Timing overrides (override values in YAML):
  --preload-lead-ms N   S-bit lead time before LOAD_REQ (default: 2)
  --load-req-ms N       LOAD_REQ pulse width (default: 1)
  --rck-ms N            RCK pulse width (default: 1)
  --trig-ms N           Trigger pulse width (default: 5)

Reproducibility:
  --seed N              Override protocol's RNG seed for randomization

Hardware:
  --device NAME         Override NI-DAQ device name (e.g. Dev2)
```

---

## Step-by-Step: First Run

### 1. Activate the correct environment
```powershell
conda activate base           # or your project env
cd C:\Rishika\MultiBiOS
```

### 2. Dry-run to validate and preview
```powershell
python -m multibios.run_protocol `
  --yaml config/odor_lateralization.yaml `
  --hardware config/hardware.yaml `
  --dry-run --interactive
```
This creates a timestamped folder under `data/runs/` containing `preview.html` — open it in any browser to inspect the waveform.

### 3. Check the compile report
The folder also contains `compile_report.json` and `digital_edges.csv`. Verify:
- All expected phases are listed
- Microscope trigger times match your intended trial markers
- No "guardrail violation" errors in the log

### 4. Execute on hardware
```powershell
python -m multibios.run_protocol `
  --yaml config/odor_lateralization.yaml `
  --hardware config/hardware.yaml `
  --verbose --progress
```
Progress output format: `[  5%] 250.0ms | DO:░█░░█ | AO:0:2.50,1:1.20`

---

## Checking a Run's Output

Output directory structure (`data/runs/YYYY-MM-DD_HH-MM-SS/`):
```
compiled_do.npz    ← digital output (play this to hardware)
compiled_ao.npz    ← analog output
capture_ai.npz     ← MFC flow readings (hardware run only)
compile_report.json
digital_edges.csv  ← every rising/falling edge with timestamp
rck_edges.csv      ← register-clock commit times
preview.html       ← interactive waveform viewer ← OPEN THIS FIRST
protocol.yaml      ← copy of input protocol
hardware.yaml      ← copy of hardware mapping
meta.json          ← device, duration, args
```

---

## Common Errors & Fixes

| Error | Likely Cause | Fix |
|---|---|---|
| `GuardrailViolation` | Two events on same assembly overlap within preload window | Space events ≥ 3 ms apart, or reduce `preload_lead_ms` |
| `KeyError: 'device.key'` | Device key not in hardware.yaml | Check spelling against hardware.yaml |
| `DAQmxError -200220` | NI-DAQ device not connected | Check USB, confirm device name with NI MAX |
| `ValueError: state list length mismatch` | State CSV has wrong count vs. `times:` | Match CSV length to phase repeat count |
| `FileNotFoundError` on YAML | Wrong path | Run from `MultiBiOS/` directory |
| `ModuleNotFoundError: multibios` | Wrong environment / not installed | `pip install -e .` in the MultiBiOS directory |

---

## Visualizing After a Run
```powershell
# Open the auto-generated preview
Start-Process "data/runs/<timestamp>/preview.html"

# Or regenerate visualization from saved data
python -m multibios.viz_protocol data/runs/<timestamp>/
```

---

## Running Tests (No Hardware Needed)
```powershell
# Quick functional validation
python tests/simple_test.py

# Full test suite
python -m pytest tests/
```
