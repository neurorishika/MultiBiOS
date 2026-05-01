# MultiBiOS Hardware Test

This directory contains hardware testing utilities for the MultiBiOS system.

## Rig Helpers

For operator-facing hardware checks, use the PowerShell wrappers in [tools](../tools):

- [tools/run_fictrac_probe.ps1](../tools/run_fictrac_probe.ps1) for a short live FicTrac end-to-end probe with optional trigger train setup
- [tools/run_mfc_test.ps1](../tools/run_mfc_test.ps1) for the analog MFC monitor and sweep modes
- [tools/run_valve_test.ps1](../tools/run_valve_test.ps1) for hardware-clocked valve protocol checks, defaulting to an independent left-then-right serial round via [protocols/serial_valve_round_independent.yaml](../protocols/serial_valve_round_independent.yaml)
- [tools/run_fictrac_config_gui.ps1](../tools/run_fictrac_config_gui.ps1) for trigger-aware FicTrac reconfiguration

These wrappers prefer the active `multibios-blackfly` environment when it is already activated and fall back to `conda run` otherwise.

## FicTrac Client Tests

The internal FicTrac client now has automated tests that are intentionally separate from the live hardware probes.

These tests cover:

- parser equivalence against the older pybmt state parser
- frame-store chunking and recent-history behavior
- newest-frame closed-loop consumer semantics
- experiment callback integration

Run them from the MultiBiOS root with:

```bash
pytest tests/test_fictrac_client.py tests/test_experiment_fictrac_callback.py
```

## Pre-Connect Scope Test

Use [tests/preconnect_scope_test.py](../tests/preconnect_scope_test.py) before connecting cameras or MFCs.

This script is intentionally narrower than `hardware_test.py`:

- `trigger` mode tests only the shared camera trigger line
- `analog` mode tests only the four MFC analog output channels
- it is designed for oscilloscope validation while the external devices are still disconnected

Detailed bench instructions are in [docs/preconnect_scope_checklist.md](../docs/preconnect_scope_checklist.md).

For the camera return path specifically, use [docs/camera_return_line_checklist.md](../docs/camera_return_line_checklist.md).

## Camera Return-Line Verification

Use [tests/verify_camera_return_line.py](../tests/verify_camera_return_line.py) to check whether the camera GPIO return wire is electrically visible at the NI-DAQ inputs.

Typical flow:

1. If you are testing the white return wire on the BFS-U3-13Y3M, wire the blue `Opto GND` into the measurement circuit as well.
2. Run `python tests/verify_camera_return_line.py --line line1` from the MultiBiOS root in the `multibios-blackfly` environment.
3. If you are intentionally testing the red GPIO wire instead, run `python tests/verify_camera_return_line.py --line line2` and provide an external pull-up.

This is narrower than the trigger-acquisition test: it tells you whether the return wire itself is electrically visible at the DAQ at all.

## Continuous Camera Trigger Test

Use [tests/continuous_camera_trigger.py](../tests/continuous_camera_trigger.py) to continuously pulse the shared `TRIG_CAMERA` line from the NI-DAQ while both Teledyne FLIR Blackfly S BFS-U3-13Y3M cameras are armed in external-trigger mode.

Typical flow:

1. Run `python -m multibios.blackfly.setup_daq_mode` from the MultiBiOS root.
2. Open SpinView or your Blackfly acquisition app and arm both cameras.
3. Run `python tests/continuous_camera_trigger.py --fps 30` from the MultiBiOS root.

## Camera Trigger Path Verification

Use [tests/verify_camera_trigger_path.py](../tests/verify_camera_trigger_path.py) to generate a finite trigger train and measure the camera return lines on the same DAQ hardware clock.

Typical flow:

1. Run `python tests/verify_camera_trigger_path.py --arm-cameras --fps 60 --duration 3` from the MultiBiOS root when using the `multibios-blackfly` environment.
2. If you prefer to arm cameras externally, run `python -m multibios.blackfly.setup_daq_mode` first, then arm both cameras in SpinView or your Blackfly acquisition app, then run `python tests/verify_camera_trigger_path.py --fps 60 --duration 3`.

Optional loopback:

1. Temporarily wire `TRIG_CAMERA` to a spare `port0` digital input line.
2. Run `python tests/verify_camera_trigger_path.py --fps 60 --duration 3 --trigger-monitor Dev1/port0/line28`.

This tells you whether the DAQ is really outputting the commanded trigger rate and whether each camera is returning one exposure pulse per trigger.

Important for BFS-U3-13Y3M wiring:

- the white wire is `Line1`, not `Line2`
- `Line1` is an opto-coupled output referenced to the blue `Opto GND` wire
- `Line2` is a different open-drain GPIO line

When `--arm-cameras` is used, the script also reports how many frames PySpin actually acquired during the trigger train. That is the most direct acceptance check for overlap-capable cameras.

## Camera ROI Sweep

Use [tests/camera_roi_sweep.py](../tests/camera_roi_sweep.py) to sweep a list of requested ROI sizes against one Blackfly camera.

Typical flow:

1. Run `python tests/camera_roi_sweep.py --camera-index 0` from the MultiBiOS root.
2. If you want a narrower sweep, run `python tests/camera_roi_sweep.py --camera-index 1 --sizes 400x400 512x512 640x640`.

Each ROI attempt runs in its own subprocess so a PySpin teardown abort does not kill the whole sweep. The summary reports ROI node writability before configuration and the actual width, height, and offsets after the attempt.

## Hardware Test Script

The `hardware_test.py` script generates synchronized square waves on all digital and analog outputs to test hardware connectivity and configuration.

### Features

- **Hardware-synchronized output**: Digital outputs act as master clock, analog outputs are slaved
- **Square wave generation**: Configurable frequency, amplitude, and duration
- **Input monitoring**: Optional analog input monitoring during test
- **Comprehensive logging**: Verbose output with detailed progress tracking  
- **Interactive visualization**: HTML plots of all test signals
- **Result analysis**: Statistical analysis of captured signals

### Usage

#### Basic Usage
```bash
# Basic test with default parameters (1Hz, 10 seconds, 2.5V amplitude)
python tests/hardware_test.py --hardware config/hardware.yaml --verbose

# Quick connectivity test (higher frequency, shorter duration)
python tests/hardware_test.py --frequency 10 --duration 2 --verbose

# Full range analog test
python tests/hardware_test.py --amplitude 5.0 --offset 2.5 --duration 5 --verbose
```

#### Advanced Usage
```bash
# High frequency test for timing verification
python tests/hardware_test.py --frequency 100 --sample-rate 10000 --duration 1

# Specific device override
python tests/hardware_test.py --device "Dev2" --verbose

# Custom output directory
python tests/hardware_test.py --output-dir "my_test_results" --verbose

# Debug mode with maximum verbosity
python tests/hardware_test.py --debug
```

### Command Line Options

- `--hardware`: Hardware configuration YAML file (default: `config/hardware.yaml`)
- `--device`: Override device name from hardware config
- `--frequency, -f`: Square wave frequency in Hz (default: 1.0)
- `--duration, -d`: Test duration in seconds (default: 10.0)
- `--sample-rate, -r`: DAQ sample rate in Hz (default: 1000)
- `--amplitude, -a`: Analog output amplitude in volts (default: 2.5)
- `--offset, -o`: Analog output DC offset in volts (default: 2.5)
- `--no-monitor-inputs`: Disable analog input monitoring
- `--output-dir`: Output directory for results (default: `tests/results`)
- `--verbose, -v`: Enable verbose logging
- `--debug`: Enable debug logging

### Test Configuration

The script automatically validates test parameters:

- **Frequency**: Must be positive and satisfy Nyquist criterion (sample_rate > 2 × frequency)
- **Amplitude**: Must be between 0 and 5V (NI-6353 output range)
- **Offset**: Must be between 0 and 5V
- **Duration**: Must be positive

### Hardware Synchronization

The test ensures all outputs are hardware-synchronized:

1. **Digital outputs (DO)** act as the master clock source
2. **Analog outputs (AO)** are slaved to the DO sample clock
3. **Analog inputs (AI)** are slaved to the DO sample clock for monitoring
4. All tasks start/stop in coordinated sequence

If no digital outputs are configured, analog outputs become the master clock.

### Output Files

Each test run creates a timestamped directory with:

- `test_results.json`: Complete test results and statistics
- `test_visualization.html`: Interactive plots of all signals
- `hardware.yaml`: Copy of hardware configuration used

### Example Output

```
=== MultiBiOS Hardware Test Starting ===
Loading hardware configuration: C:\Rishika\MultiBiOS\config\hardware.yaml
✓ Hardware configuration loaded successfully
  Device: Dev1
  Digital outputs: 19 channels
  Analog outputs: 4 channels
  Analog inputs: 4 channels

=== Starting Hardware Test ===
Test Configuration:
  Frequency: 10.0 Hz
  Duration: 5.0 seconds  
  Sample rate: 1000 Hz
  AO amplitude: 2.5 V
  AO offset: 2.5 V
  Total samples: 5,000
  Samples per cycle: 100

✓ Test execution completed in 5.12 seconds
✓ Hardware test completed successfully

=== TEST COMPLETION SUMMARY ===
✓ Test completed successfully
✓ Duration: 5.12 seconds
✓ Digital outputs tested: 19
✓ Analog outputs tested: 4
✓ Analog inputs monitored: 4
✓ Results directory: tests/results/hardware_test_2025-09-16_14-30-15
✓ Visualization: tests/results/hardware_test_2025-09-16_14-30-15/test_visualization.html
```

### Troubleshooting

1. **"Device not found" errors**: Check that your NI-DAQ device is connected and the device name matches your hardware.yaml
2. **"Sample rate too low" errors**: Increase `--sample-rate` or decrease `--frequency`
3. **"Channel not found" errors**: Verify your hardware.yaml channel assignments match your physical wiring
4. **Timing issues**: Use `--debug` flag for detailed DAQ task timing information

### Integration with MultiBiOS

This test script uses the same hardware configuration format as the main MultiBiOS protocol runner, ensuring consistency between testing and actual protocol execution.