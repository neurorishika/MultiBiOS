# MFC Analog Test Interface

`tests/mfc_analog_test.py` is the primary tool for verifying and monitoring
the four mass-flow controller (MFC) channels.  It replaces the legacy
serial-based `flow_monitor` and talks directly to the NI-DAQ board — the same
hardware path used during live experiments.

## Background: how MFC control works

MFCs are controlled **entirely over analog voltage** via the NI USB-6353:

| Signal | Direction | Channels | Voltage |
|--------|-----------|----------|---------|
| Setpoint | DAQ → MFC | `Dev1/ao0-3` | 0–5 V (maps to 0–100 % full scale) |
| Feedback | MFC → DAQ | `Dev1/ai0-3` | 0–5 V (actual flow readback) |

The channel names (`mfc.air_left_setpoint`, etc.) are defined in
[config/hardware.yaml](../config/hardware.yaml).  The test script reads that
file directly, so it stays in sync automatically if pin assignments change.

!!! note "Serial control removed"
    The previous `AlicatManager` / `flow_monitor` serial path has been moved
    to `legacy/`.  Do not use it for new work — serial ports are no longer
    wired to the MFCs.

---

## Modes

### `monitor` — live setpoint + feedback display

Applies initial setpoints, then continuously reads AI feedback and prints a
live table at the configured interval.  All setpoints are zeroed automatically
on `Ctrl+C` (pass `--no-zero-on-exit` to suppress).

```bash
# Watch current flow with all channels at 0 V
python tests/mfc_analog_test.py monitor

# Apply setpoints before monitoring
python tests/mfc_analog_test.py monitor --set air_left=2.5 odor_right=1.0

# Faster poll, no auto-zero on exit
python tests/mfc_analog_test.py monitor --interval 0.2 --no-zero-on-exit
```

**Display columns:**

| Column | Meaning |
|--------|---------|
| `Setpt (V)` | Commanded AO voltage |
| `Feedback (V)` | Mean of 20 AI samples |
| `Flow (% FS)` | `feedback / 5.0 × 100 %` (full-scale percentage) |
| `Δ (V)` | `feedback − setpoint` |
| `✓ / ⚠` | Green tick if `\|Δ\| ≤ 0.15 V`, warning otherwise |

---

### `sweep` — linearity / pass-fail test

Steps each AO channel through a list of voltage levels (one channel at a time,
all others held at 0 V) and compares the AI readback against the commanded
value.  Returns **exit code 0** on full PASS, **exit code 1** if any channel
fails, making it usable in CI.

```bash
# Default sweep: 0 → 1 → 2 → 3 → 4 → 5 → 0 V, 0.5 s dwell, ±0.1 V tolerance
python tests/mfc_analog_test.py sweep

# Tighter tolerance, longer dwell
python tests/mfc_analog_test.py sweep --tolerance 0.05 --dwell 1.0

# Sweep a subset of channels only
python tests/mfc_analog_test.py sweep --channels air_left odor_left

# Custom voltage steps
python tests/mfc_analog_test.py sweep --levels 0 2.5 5 0
```

**Example output:**

```
Sweep: [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 0.0] V  dwell=0.50s  tol=0.100 V

  Channel: air_left
    ✓  set=0.00 V  read=0.0031 V  err=+0.0031 V
    ✓  set=1.00 V  read=0.9987 V  err=-0.0013 V
    ✓  set=2.00 V  read=1.9994 V  err=-0.0006 V
    ...

── Sweep Summary ──────────────────────────────
  [PASS] air_left         worst_error=0.0031 V  (tol=0.100 V)
  [PASS] air_right        worst_error=0.0028 V  (tol=0.100 V)
  [PASS] odor_left        worst_error=0.0041 V  (tol=0.100 V)
  [PASS] odor_right       worst_error=0.0019 V  (tol=0.100 V)

ALL PASS
```

---

## Dry-run mode

Prints the resolved channel map without touching hardware.  Use this to
confirm `hardware.yaml` is being parsed correctly before connecting devices.

```bash
python tests/mfc_analog_test.py --dry-run monitor
```

```
Hardware: C:\Rishika\MultiBiOS\config\hardware.yaml
MFC AO channels:
  air_left        AO -> Dev1/ao0  AI -> Dev1/ai0
  air_right       AO -> Dev1/ao1  AI -> Dev1/ai1
  odor_left       AO -> Dev1/ao2  AI -> Dev1/ai2
  odor_right      AO -> Dev1/ao3  AI -> Dev1/ai3

[dry-run] Hardware not touched.
```

---

## Custom hardware config

```bash
python tests/mfc_analog_test.py --hardware path/to/other_hardware.yaml sweep
```

The `--hardware` flag defaults to `config/hardware.yaml` relative to the
repository root.

---

## Channel names

| Logical name | AO channel | AI channel | MFC |
|---|---|---|---|
| `air_left` | `Dev1/ao0` | `Dev1/ai0` | Air — left side |
| `air_right` | `Dev1/ao1` | `Dev1/ai1` | Air — right side |
| `odor_left` | `Dev1/ao2` | `Dev1/ai2` | Odour — left side |
| `odor_right` | `Dev1/ao3` | `Dev1/ai3` | Odour — right side |

Voltages map to flow as:

$$
V_\text{setpoint} = \frac{\text{setpoint (sccm)}}{\text{full-scale (sccm)}} \times 5\,\text{V}
$$

$$
\text{flow (sccm)} = \frac{V_\text{feedback}}{5\,\text{V}} \times \text{full-scale (sccm)}
$$

The full-scale value depends on the MFC model ordered — check the label on
each device.

---

## Safety notes

- The AO range is **0–5 V**.  Values are clamped before writing; you cannot
  accidentally command a negative or over-range voltage through this tool.
- `monitor` mode **always zeros all channels on exit** unless `--no-zero-on-exit`
  is passed.  In `sweep` mode, channels are zeroed automatically at the end of
  the sweep regardless of outcome.
- Do not run this tool simultaneously with `run_protocol.py` — both write to
  the same AO channels.

---

## Integration with the pre-connect checklist

The `sweep` command is the programmatic equivalent of the manual steps in
[Pre-Connect Oscilloscope Checklist](preconnect_scope_checklist.md).  Run it
after bench-testing with a scope to confirm the full AO → MFC → AI loop closes
correctly before starting an experiment session.

```bash
# Recommended pre-session check
python tests/mfc_analog_test.py sweep --dwell 1.0
```

Exit code 0 = all channels within tolerance.  Exit code 1 = investigate before
proceeding.
