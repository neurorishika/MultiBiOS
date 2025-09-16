# MultiBiOS Hardware Test

This directory contains hardware testing utilities for the MultiBiOS system.

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