# Protocol Runner (NI-DAQ Application)

The protocol runner compiles YAML protocol files, generates **hardware-clocked digital/analog outputs**, and captures **analog input** feedback from mass flow controllers (MFCs).

## Dependencies

- **NI-DAQmx** driver and Python API (`pip install nidaqmx`)
- **Core libraries**: numpy, pyyaml, plotly (for visualization)
- **Validated environment on this rig**: `multibios-blackfly` created from `environment.yml`

## Command Line Interface

### Dry Run (Preview Only)

```bash
# Generate preview without hardware execution
conda run -n multibios-blackfly python -m multibios.run_protocol \
  --yaml protocols/example_protocol.yaml \
  --hardware config/hardware.yaml \
  --dry-run --seed 42
```

### Hardware Execution  

```bash
# Execute protocol on DAQ hardware
conda run -n multibios-blackfly python -m multibios.run_protocol \
  --yaml protocols/example_protocol.yaml \
  --hardware config/hardware.yaml
```

If `hardware.yaml` contains a `fictrac:` block, the runner will also launch FicTrac, wait for the first UDP frame, and save FicTrac artifacts into the same run directory.

## Command Line Options

### Core Arguments

- `--yaml <file>`: Protocol YAML file (default: `protocols/example_protocol.yaml`)
- `--hardware <file>`: Hardware mapping YAML (default: `config/hardware.yaml`)
- `--experiment <file>`: Optional runtime override file for backward-compatible experiment/camera/FicTrac settings
- `--device <name>`: Override DAQ device name from hardware.yaml
- `--dry-run`: Compile and preview only, no hardware execution
- `--out-root <dir>`: Output directory root (default: `data/runs`)

### Timing Overrides

- `--seed <int>`: Override `protocol.timing.seed` for reproducible randomization
- `--preload-lead-ms <int>`: Override preload lead time
- `--load-req-ms <int>`: Override load request pulse duration  
- `--rck-ms <int>`: Override register clock pulse duration
- `--trig-ms <int>`: Override trigger pulse duration

### Visualization

- `--interactive`: Always save interactive HTML preview (even without `--dry-run`)

### Logging & Progress

- `--verbose` / `-v`: Enable verbose logging with detailed progress information
- `--debug`: Enable debug logging (extremely detailed, for troubleshooting)
- `--progress`: Enable real-time progress monitor during protocol execution
- `--progress-interval <ms>`: Set progress update interval in milliseconds (default: 100)

## Real-Time Progress Monitoring

The protocol runner now supports **real-time progress monitoring** during hardware execution. This displays the expected protocol state while the DAQ is running, helping you track what should be happening at each moment.

### Enabling Progress Monitor

```bash
conda run -n multibios-blackfly python -m multibios.run_protocol \
  --yaml protocols/example_protocol.yaml \
  --hardware config/hardware.yaml \
  --verbose \
  --progress \
  --progress-interval 100
```

### What It Shows

During execution, you'll see periodic updates like:

```text
[  5.0%] [t=250.0ms] DO: RCK=LOW, LOAD_REQ=HIGH, S0=LOW | AO: MFC1=2.500V, MFC2=1.200V
[ 10.0%] [t=500.0ms] DO: RCK=HIGH, LOAD_REQ=LOW, S0=HIGH | AO: MFC1=3.000V, MFC2=1.500V
[ 15.0%] [t=750.0ms] DO: RCK=LOW, LOAD_REQ=LOW, S0=LOW | AO: MFC1=2.500V, MFC2=1.200V
```

Each update includes:

- **Progress percentage**: How far through the protocol
- **Timestamp**: Current protocol time in milliseconds
- **Digital outputs (DO)**: State of key digital lines (HIGH/LOW)
- **Analog outputs (AO)**: Voltage levels of analog channels

### Customizing Updates

- **Update frequency**: Use `--progress-interval <ms>` to control how often updates appear
  - Lower values (e.g., 50ms) = more frequent updates, more detailed tracking
  - Higher values (e.g., 500ms) = less frequent updates, cleaner output
- **Which channels to show**: The monitor automatically shows the first 3 DO lines and first 2 AO channels. Full data is still recorded; this is just for display.

### When to Use

- **Long protocols**: Essential for protocols lasting several seconds or minutes
- **Debugging**: Verify that the protocol is executing as expected
- **User feedback**: Provide reassurance during execution (especially important when no visual indicators are available)
- **Troubleshooting**: Identify at what point in the protocol issues occur

### Performance Impact

The progress monitor runs in a **background thread** and has minimal performance impact:

- Does not interfere with hardware timing
- Updates are calculated from elapsed time, not polling the DAQ
- Very low CPU overhead (~0.1% on typical systems)

- `--interactive`: Always save interactive HTML preview (enabled by default)

## DAQ Clocking Architecture

- **Digital Output (DO)**: Master clock - provides `SampleClock` and `StartTrigger`
- **Analog Output (AO)**: Slave - synchronized to DO clock
- **Analog Input (AI)**: Slave - synchronized to DO clock for MFC feedback capture

## Output Files

Each run creates a timestamped directory in `data/runs/YYYY-MM-DD_HH-MM-SS/`:

- `preview.html`: Interactive Plotly visualization
- `compiled_do.npz`: Digital output arrays
- `compiled_ao.npz`: Analog output arrays  
- `capture_ai.npz`: Analog input data (if hardware run)
- `capture_di.npz`: Digital input captures from READY, camera, and other return lines (if hardware run)
- `control_plan.csv`: Shared compiled logical event schedule used by both runner paths
- `do_map.json`, `ao_map.json`: Channel mapping information
- `di_map.json`: Digital input channel mapping information
- `rck_edges.csv`: Register clock commit timestamps
- `digital_edges.csv`: All digital signal edge transitions
- `protocol.yaml`, `hardware.yaml`: Input file copies
- `meta.json`: Run metadata and parameters

If `hardware.yaml -> teensy.capture_serial: true` is enabled, the run directory also includes:

- `teensy_serial_transcript.jsonl`: Raw line-oriented USB serial transcript captured from the open-loop Teensy during the run

When FicTrac is enabled, the same directory also includes:

- `fictrac_runtime_config.txt`: exact runtime config passed to FicTrac
- `fictrac_runtime.json`: summary of runtime config edits
- `fictrac_driver_diagnostics.json`: launch and first-packet diagnostics
- `fictrac_frames.npz`: internal MultiBiOS FicTrac frame store
- `fictrac-*.dat`: native FicTrac output

## Post-Run Visualization

Use the visualization tool to re-analyze saved runs:

```bash
# Re-visualize a completed run
conda run -n multibios-blackfly python -m multibios.viz_protocol data/runs/2025-08-21_16-25-26
```

This generates an updated `preview.html` with the same device-grouped visualization as the runner.
