# Real-Time Progress Monitoring - Quick Reference

## Enable Progress Monitoring

```bash
python multibios/run_protocol.py \
    --yaml protocols/example_protocol.yaml \
    --hardware config/hardware.yaml \
    --verbose \
    --progress
```

## Customize Update Frequency

```bash
# Update every 50ms (more frequent)
--progress --progress-interval 50

# Update every 200ms (less frequent)  
--progress --progress-interval 200
```

## What You'll See

```
DO Legend: [0:RCK] [1:LOAD_REQ] [2:S0] [3:S1] [4:S2] [5:S3]
AO Legend: [0:MFC1] [1:MFC2] [2:MFC3]

[  5%] 250.0ms | DO:░█░█░░ | AO:0:2.50,1:1.20
[ 10%] 500.0ms | DO:█░█░█░ | AO:0:3.00,1:1.50
```

- Legend printed once at start
- `[5%]` = Progress percentage
- `250.0ms` = Current protocol time
- `DO:░█░█░░` = Digital pattern (█=HIGH, ░=LOW)
  - Position matches legend: 0=RCK, 1=LOAD_REQ, etc.
- `AO:0:2.50,1:1.20` = Active analog channels
  - `0:2.50` = Channel 0 (MFC1) at 2.50V
  - Only shows non-zero channels!

## Update Interval Guidelines

| Value | Use Case |
|-------|----------|
| 50ms  | Detailed debugging |
| 100ms | **Default** (recommended) |
| 200ms | Cleaner output |
| 500ms | Very long protocols |

## Combine with Other Flags

```bash
# Full monitoring with visualization
--verbose --progress --interactive

# Debug mode with progress
--debug --progress --progress-interval 50

# Dry run with progress simulation
--dry-run --verbose --progress
```

## Example Scripts

**Windows:**
```powershell
.\examples\run_with_progress.ps1
```

**Linux/Mac:**
```bash
./examples/run_with_progress.sh
```

## Important Notes

✅ Must use `--verbose` or `--debug` to see progress updates
✅ No impact on DAQ timing accuracy
✅ Works with dry-run mode
✅ Background thread with < 0.1% CPU overhead

❌ Shows expected state, not real-time DAQ feedback
❌ Don't set interval < 20ms (excessive output)

## Full Documentation

See `examples/README_PROGRESS.md` for complete guide.
