# Protocol Runner (NI-DAQ Application)

The protocol runner compiles YAML protocol files, generates **hardware-clocked digital/analog outputs**, and captures **analog input** feedback from mass flow controllers (MFCs).

## Dependencies

- **NI-DAQmx** driver and Python API (`pip install nidaqmx`)
- **Core libraries**: numpy, pyyaml, plotly (for visualization)
- **Poetry environment** (recommended): `poetry install`

## Command Line Interface

### Dry Run (Preview Only)

```bash
# Generate preview without hardware execution
python -m multibios.run_protocol \
  --yaml config/example_protocol.yaml \
  --hardware config/hardware.yaml \
  --dry-run --seed 42
```

### Hardware Execution  

```bash
# Execute protocol on DAQ hardware
python -m multibios.run_protocol \
  --yaml config/example_protocol.yaml \
  --hardware config/hardware.yaml
```

## Command Line Options

### Core Arguments

- `--yaml <file>`: Protocol YAML file (default: `config/example_protocol.yaml`)
- `--hardware <file>`: Hardware mapping YAML (default: `config/hardware.yaml`)
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
- `do_map.json`, `ao_map.json`: Channel mapping information
- `rck_edges.csv`: Register clock commit timestamps
- `digital_edges.csv`: All digital signal edge transitions
- `protocol.yaml`, `hardware.yaml`: Input file copies
- `meta.json`: Run metadata and parameters

## Post-Run Visualization

Use the visualization tool to re-analyze saved runs:

```bash
# Re-visualize a completed run
python -m multibios.viz_protocol data/runs/2025-08-21_16-25-26
```

This generates an updated `preview.html` with the same device-grouped visualization as the runner.
