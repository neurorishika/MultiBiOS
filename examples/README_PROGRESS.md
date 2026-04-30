# Real-Time Protocol Progress Monitoring

## Overview

The MultiBiOS protocol runner now includes **real-time progress monitoring** that displays the expected protocol state while the NI-DAQ is executing. This provides immediate feedback about what's happening during protocol execution, which is especially valuable since the DAQ provides no inherent progress feedback.

## Why This Matters

When running hardware protocols, you typically see:
- ❌ No indication of progress
- ❌ No idea what the current state should be
- ❌ No way to verify execution is proceeding correctly
- ❌ Uncertainty about whether the system is frozen or working

With progress monitoring, you now get:
- ✅ Real-time progress percentage
- ✅ Current timestamp in the protocol
- ✅ Expected state of all digital and analog outputs
- ✅ Confidence that execution is proceeding correctly

## Quick Start

### Basic Usage

Enable progress monitoring with the `--progress` flag:

```bash
python multibios/run_protocol.py \
    --yaml protocols/example_protocol.yaml \
    --hardware config/hardware.yaml \
    --verbose \
    --progress
```

### Customize Update Frequency

Control how often progress updates appear (in milliseconds):

```bash
python multibios/run_protocol.py \
    --yaml protocols/example_protocol.yaml \
    --hardware config/hardware.yaml \
    --verbose \
    --progress \
    --progress-interval 200  # Update every 200ms
```

### Using the Example Scripts

**Windows (PowerShell):**
```powershell
.\examples\run_with_progress.ps1
```

**Linux/Mac (Bash):**
```bash
chmod +x examples/run_with_progress.sh
./examples/run_with_progress.sh
```

## Example Output

Here's what you'll see during execution:

```
Progress monitor: 5000 samples @ 1000Hz, updates every 100ms
DO Legend: [0:RCK] [1:LOAD_REQ] [2:S0] [3:S1] [4:S2] [5:S3] [6:S4] [7:S5]
AO Legend: [0:MFC1] [1:MFC2] [2:MFC3]
✓ All DAQ tasks started, protocol execution in progress...

[  2%] 100.0ms | DO:░░░░░░░░ | AO:---
[  4%] 200.0ms | DO:░█░█░░░░ | AO:0:2.50,1:1.20
[  6%] 300.0ms | DO:█░█░█░░░ | AO:0:2.50,1:1.20
[  8%] 400.0ms | DO:░░█░░░░░ | AO:0:2.50,1:1.20
[ 10%] 500.0ms | DO:░░░██░░░ | AO:0:1.80,1:0.80
...
[ 98%] 4900.0ms | DO:░░░░░░░░ | AO:---
[100%] Protocol execution complete

✓ Protocol execution completed in 5.02 seconds
```

## Understanding the Output

### Legend (printed once at start)
```
DO Legend: [0:RCK] [1:LOAD_REQ] [2:S0] [3:S1] [4:S2]
AO Legend: [0:MFC1] [1:MFC2] [2:MFC3]
```
Maps channel indices to their names. This prints once so you know what each position means.

### Progress Lines

Each update shows:

### 1. Progress Percentage
```
[ 10%]
```
Shows how far through the protocol (0-100%).

### 2. Protocol Timestamp
```
500.0ms
```
Current time position in the protocol.

### 3. Digital Output Pattern
```
DO:░█░█░░░░
```
Visual pattern showing **all** digital outputs in order:
- **█** (filled block) = HIGH
- **░** (light shade) = LOW

Each position corresponds to the legend index:
- Position 0: RCK = LOW (░)
- Position 1: LOAD_REQ = HIGH (█)
- Position 2: S0 = LOW (░)
- Position 3: S1 = HIGH (█)
- Positions 4-7: All LOW (░░░░)

**Benefit**: See ALL channels at once, spot patterns instantly!

### 4. Analog Output State
```
AO:0:2.50,1:1.20
```
Shows **only active** (non-zero) channels in `index:voltage` format:
- `0:2.50` = Channel 0 (MFC1) = 2.50V
- `1:1.20` = Channel 1 (MFC2) = 1.20V

When all channels are zero:
```
AO:---
```

**Benefit**: Compact display, only shows what's active!

## Configuration Options

### Update Interval

The `--progress-interval` flag controls how often updates are displayed:

| Interval | Use Case | Output Volume |
|----------|----------|---------------|
| 50ms | Very detailed tracking, debugging specific timing | High |
| 100ms | **Default** - Good balance for most protocols | Medium |
| 200ms | Cleaner output for longer protocols | Low |
| 500ms | Minimal updates for very long protocols | Very Low |

**Example:**
```bash
# More frequent updates (every 50ms)
--progress --progress-interval 50

# Less frequent updates (every 500ms)
--progress --progress-interval 500
```

### Combining with Verbose Logging

For maximum visibility, combine progress monitoring with verbose logging:

```bash
python multibios/run_protocol.py \
    --yaml protocols/example_protocol.yaml \
    --hardware config/hardware.yaml \
    --verbose \
    --progress \
    --progress-interval 100
```

This gives you both:
- Detailed logging of setup and compilation
- Real-time progress during execution
- Comprehensive post-execution summary

## How It Works

### Architecture

The progress monitor uses a **background thread** that:
1. Starts when the DAQ begins execution
2. Calculates the expected sample index based on elapsed time
3. Looks up the expected state from the compiled protocol arrays
4. Formats and displays the state at regular intervals
5. Automatically stops when execution completes

### Performance Characteristics

- **CPU Overhead**: < 0.1% on typical systems
- **No DAQ Polling**: Uses time-based calculation, not hardware queries
- **No Timing Impact**: Runs independently of DAQ hardware clocking
- **Thread-Safe**: Properly synchronized with main execution

### Accuracy

The monitor displays the **expected state** based on:
- Compiled protocol arrays (DO/AO)
- Configured sample rate
- Elapsed wall-clock time

**Note**: This shows what *should* be happening, not real-time DAQ feedback. The actual DAQ execution is hardware-clocked and may have microsecond-level variations from wall-clock time.

## Use Cases

### 1. Long Protocol Execution
For protocols lasting minutes or hours, progress monitoring provides:
- Reassurance that execution is proceeding
- Ability to estimate time remaining
- Early detection if something seems wrong

### 2. Protocol Development
When developing and testing new protocols:
- Verify state transitions happen as expected
- Check timing of key events (RCK pulses, triggers)
- Identify unexpected state combinations

### 3. Debugging Issues
When troubleshooting hardware problems:
- See exactly what the protocol was doing when an issue occurred
- Compare expected state to observed hardware behavior
- Narrow down the timing window of problems

### 4. User Demonstrations
When showing the system to others:
- Provide visual feedback during execution
- Explain what's happening in real-time
- Build confidence in system operation

## Tips & Best Practices

### ✅ DO:
- Use `--verbose --progress` together for comprehensive feedback
- Adjust `--progress-interval` based on protocol duration
- Review the full state in `preview.html` after execution
- Check `digital_edges.csv` for complete timing information

### ❌ DON'T:
- Set update interval too low (< 20ms) - may clutter output
- Rely solely on progress monitor for timing verification
- Expect microsecond-accurate timing (use compiled arrays for that)
- Use progress monitoring as a substitute for proper hardware validation

## Troubleshooting

### Progress monitor not appearing?

**Check**: Did you include both `--verbose` and `--progress` flags?
```bash
# Correct:
--verbose --progress

# Won't show updates without --verbose:
--progress
```

### Updates are too frequent/infrequent?

**Solution**: Adjust the `--progress-interval`:
```bash
# Slower updates:
--progress --progress-interval 500

# Faster updates:
--progress --progress-interval 50
```

### Monitor stops before protocol ends?

**Issue**: Protocol may have crashed or timed out.

**Check**:
1. Review error messages in terminal
2. Verify DAQ device is connected and recognized
3. Check hardware configuration in `hardware.yaml`
4. Ensure protocol YAML is valid

### State doesn't match hardware behavior?

**Remember**: Progress monitor shows **expected state**, not actual DAQ feedback.

**To verify actual behavior**:
1. Review captured AI/DI data in `capture_*.npz` files
2. Use oscilloscope to measure actual outputs
3. Check `di_edges.csv` for Teensy READY or camera return signals
4. Examine `preview.html` for full protocol visualization

## Advanced: Custom Monitoring

If you need custom monitoring logic, you can modify the `ProtocolProgressMonitor` class in `run_protocol.py`:

```python
class ProtocolProgressMonitor:
    def _format_state(self, sample_idx: int) -> str:
        """Customize this method to change output format"""
        # Your custom formatting here
        pass
    
    def _monitor_loop(self):
        """Customize this method to change monitoring behavior"""
        # Your custom monitoring logic here
        pass
```

## Related Documentation

- [Protocol Runner Documentation](../docs/runner.md) - Full command-line reference
- [Protocol Schema](../docs/protocol.md) - YAML protocol specification
- [Visualization Guide](../docs/visualization.md) - Understanding output plots
- [Troubleshooting](../docs/troubleshooting.md) - Common issues and solutions

## Questions?

See the [FAQ](../docs/faq.md) or [Contributing Guide](../docs/contributing.md) for more information.
