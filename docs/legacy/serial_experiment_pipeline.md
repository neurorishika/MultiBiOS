# Legacy Serial Experiment Pipeline

`multibios.experiment` remains in the module for compatibility, but it is deprecated for new experiments.

Use [../running_experiments.md](../running_experiments.md) and [../runner.md](../runner.md) for the supported hardware-clocked `multibios.run_protocol` workflow.

## Status

- Deprecated runner: `python -m multibios.experiment`
- Timing model: computer-timed serial valve/MFC events plus a finite NI-DAQ trigger task
- Kept for: historical serial/Alicat workflows, old run replay, and transition-period rig support

## Command Reference

```powershell
cd C:\Rishika\MultiBiOS

conda run -n multibios-blackfly python -m multibios.experiment `
  --protocol config/odor_lateralization.yaml `
  --hardware config/hardware.yaml `
  --experiment config/experiment_config.yaml `
  --verbose
```

Short bounded probe:

```powershell
cd C:\Rishika\MultiBiOS

conda run -n multibios-blackfly python -m multibios.experiment `
  --protocol config/short_protocol.yaml `
  --hardware config/hardware.yaml `
  --experiment config/experiment_config_probe.yaml
```

Dry run preview:

```powershell
cd C:\Rishika\MultiBiOS

conda run -n multibios-blackfly python -m multibios.experiment `
  --protocol config/odor_lateralization.yaml `
  --hardware config/hardware.yaml `
  --experiment config/experiment_config.yaml `
  --dry-run --verbose
```

## Configuration

The deprecated runner still depends on both config files below.

### `config/hardware.yaml`

This remains the rig-level source of truth for DAQ and FicTrac settings:

```yaml
fictrac:
  config: "config_camera.txt"
  bin: "C:/Rishika/MultiBiOS/assets/fictrac-spinnaker/fictrac-spinnaker.exe"
  console_out: "fictrac_output.txt"
  first_frame_timeout_ms: 0
  startup_timeout_s: 90.0
  timeout_s: 5.0
```

### `config/experiment_config.yaml`

This file is legacy-runner-specific and contains the serial/Alicat settings:

```yaml
teensy_port: "COM4"
teensy_baud: 115200

mfc_mode: "alicat_serial"

mfc_device_map:
  mfc.air_left_setpoint:   "A"
  mfc.air_right_setpoint:  "B"
  mfc.odor_left_setpoint:  "C"
  mfc.odor_right_setpoint: "D"

alicat_ports: ["COM7", "COM8", "COM9", "COM10"]
alicat_baud: [115200]

latch_interval_ms: 10.0
mfc_live_interval_s: 1.0
data_dir: "data/runs"
```

## Runner Differences

| | `multibios.run_protocol` | `multibios.experiment` |
| --- | --- | --- |
| Valve control | Hardware-clocked NI-DAQ waveform | Computer-timed serial to Teensy |
| MFC control | DAQ analog output | Alicat serial |
| FicTrac | Integrated through `hardware.yaml -> fictrac` | Integrated through `hardware.yaml -> fictrac` |
| Camera/scope triggers | Embedded in DAQ waveform | Separate finite NI-DAQ task |
| Intended use | Supported default path | Deprecated compatibility path |

## FicTrac Notes

For the live Blackfly side camera, validate the rebuilt binary first using the probe flow in [../fictrac.md](../fictrac.md).

The deprecated runner still:

- prepares the Spinnaker runtime path
- starts the FicTrac thread and waits for the first frame
- records the experiment event stream alongside FicTrac output

Use `fictrac.first_frame_timeout_ms: 0` together with `fictrac.startup_timeout_s: 0` when you want both layers to wait indefinitely for the first externally triggered frame.

## Output Files

The legacy run directory contains files such as:

- `experiment_data.csv`
- `event_log.csv`
- `event_log.json`
- `timeline.csv`
- `trigger_waveform.npz`
- `fictrac_runtime_config.txt`
- `fictrac_driver_diagnostics.json`
- `fictrac-*.dat`
- `protocol.yaml`
- `hardware.yaml`
- `meta.json`

## Troubleshooting

| Symptom | Likely Cause | Fix |
| --- | --- | --- |
| `FicTrac did not produce any frames within N s` | Camera not found, FicTrac crashed, or the first-frame wait is too short | Check `hardware.yaml -> fictrac`, verify the packaged `fictrac-spinnaker.exe`, and increase `startup_timeout_s` or set it to `0` |
| `No cached Alicat device matches mapping` | Wrong letter ID or COM port | Run `python -m multibios.apps.flow_monitor --scan` and update `mfc_device_map` |
| `Teensy RESET: ERROR` | Wrong COM port or firmware not running | Check Device Manager for the correct port and re-flash firmware |
| High jitter on valve events | Host CPU load | Close non-essential applications and avoid using this path for timing-critical runs |

## CLI Reference

```text
conda run -n multibios-blackfly python -m multibios.experiment [OPTIONS]

Required:
  --protocol FILE       Protocol YAML
  --hardware FILE       Hardware mapping YAML
  --experiment FILE     Experiment config YAML (default: config/experiment_config.yaml)

Execution:
  --dry-run             Preview timeline only; no hardware
  --verbose / -v        Print each event with jitter as it fires
  --seed INT            Override protocol RNG seed

Output:
  --out-root DIR        Output root (default: from experiment_config.yaml data_dir)
```
