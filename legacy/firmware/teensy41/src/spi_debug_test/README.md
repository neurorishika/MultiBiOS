# SPI Debug Test Firmware

This firmware provides systematic testing of the SPI output chain for MultiBiOS hardware debugging.

## Purpose

The original firmware relies on interrupt-driven responses to DAQ signals. This debug version eliminates all complex logic and generates predictable, repeating SPI patterns that you can easily verify with an oscilloscope.

## Features

- **6 Different Test Modes** with distinct, recognizable patterns
- **Adjustable SPI Clock Speed** (100 kHz to 10 MHz)
- **Serial Monitor Control** for interactive testing
- **Visual LED Feedback** during transmissions
- **Detailed Pattern Logging** to serial monitor

## Test Modes

### 1. ALL ZEROS (Mode 0)
- **Pattern**: `0x0000 0x0000 0x00 0x00`
- **Scope View**: MOSI should stay LOW throughout transmission
- **Use**: Verify SPI clock is working, check for stuck-high lines

### 2. ALL ONES (Mode 1)  
- **Pattern**: `0xFFFF 0xFFFF 0xFF 0xFF`
- **Scope View**: MOSI should stay HIGH throughout transmission
- **Use**: Verify SPI can drive high, check for stuck-low lines

### 3. ALTERNATING BITS (Mode 2)
- **Pattern A**: `0x5555 0xAAAA 0x55 0xAA`
- **Pattern B**: `0xAAAA 0x5555 0xAA 0x55`
- **Scope View**: Regular HIGH/LOW pattern on MOSI
- **Use**: Check for bit timing issues, verify data integrity

### 4. WALKING ONES (Mode 3)
- **Pattern**: Single '1' bit walks through all 48 positions
- **Scope View**: Single high pulse walks through the 48-bit frame
- **Use**: Verify all bit positions reach the output, check shift register operation

### 5. COUNTING (Mode 4)
- **Pattern**: Incrementing counter in each field
- **Scope View**: Increasing complexity of bit patterns
- **Use**: Test dynamic patterns, verify no bit positions are stuck

### 6. CUSTOM PATTERNS (Mode 5)
- **Pattern 1**: `0x1234 0x5678 0x9A 0xBC` (unique nibbles)
- **Pattern 2**: `0x0001 0x0002 0x04 0x08` (powers of 2)
- **Pattern 3**: `0x0003 0x000C 0x03 0x00` (air valve simulation)
- **Pattern 4**: `0x0FFF 0x0FFF 0xFF 0xFF` (flush state simulation)

## Hardware Setup

### Oscilloscope Connections
1. **Channel 1**: Connect to Pin 11 (MOSI) - Data signal
2. **Channel 2**: Connect to Pin 13 (SCK) - Clock signal  
3. **Ground**: Connect scope ground to Teensy ground

### Oscilloscope Settings
- **Timebase**: Start with 10µs/div (adjust based on SPI speed)
- **Trigger**: Channel 2 (SCK) rising edge
- **Voltage**: 5V/div for both channels
- **Coupling**: DC

## Expected Waveforms

### SPI Frame Structure
```
[16-bit OLFA_L][16-bit OLFA_R][8-bit SW_L][8-bit SW_R] = 48 total bits
```

### Timing at 1 MHz SPI Clock
- **Bit period**: 1µs per bit
- **Frame duration**: 48µs per complete frame
- **Frame interval**: 500ms between frames (configurable)

## Usage Instructions

### 1. Upload Firmware
```bash
# In Arduino IDE or PlatformIO
# Select: Teensy 4.1, USB Type: Serial
# Upload spi_debug_test.ino
```

### 2. Open Serial Monitor
```bash
# Baud rate: 115200
# Watch for startup messages and pattern descriptions
```

### 3. Interactive Commands
- **`n`**: Switch to next test mode
- **`s`**: Show SPI speed options
- **`1-4`**: Set SPI speed (100kHz, 1MHz, 5MHz, 10MHz)
- **`r`**: Reset pattern counter

### 4. Oscilloscope Analysis

#### What to Look For:
1. **Clock Signal (SCK)**: Should be clean square wave at set frequency
2. **Data Signal (MOSI)**: Should change on clock edges
3. **Frame Timing**: 48 clock cycles per frame, then 500ms gap
4. **Pattern Recognition**: Verify the transmitted pattern matches expected

#### Troubleshooting Guide:

**No Clock Signal (SCK)**
- Check Pin 13 connection
- Verify SPI initialization in serial monitor
- Try different SPI speeds

**No Data Signal (MOSI)**  
- Check Pin 11 connection
- Verify SPI is sending data (LED should blink)
- Check for loose connections

**Incorrect Data Patterns**
- Use ALL ZEROS/ALL ONES modes first
- Check for bit timing issues with ALTERNATING mode
- Use WALKING ONES to verify all bit positions

**Downstream Circuit Issues**
- Connect scope to shift register inputs
- Verify power supply to downstream ICs
- Check for proper grounding

## Example Serial Output

```
=== MultiBiOS SPI Debug Test ===
Version 1.0

SPI initialized at 1000000 Hz
Testing SPI connection...

========================================
SWITCHED TO MODE: ALL ZEROS
Scope: Look for constant LOW on MOSI
========================================

Pattern: ALL ZEROS - 0x0000 0x0000 0x00 0x00
Pattern: ALL ZEROS - 0x0000 0x0000 0x00 0x00
...
```

## Advanced Debugging

### Custom Pattern Testing
Modify the `generateCustomPatterns()` function to test specific valve combinations that aren't working in your main firmware.

### Speed Optimization
Start testing at 100 kHz (easy to see on scope), then increase speed to find the maximum reliable rate for your circuit.

### Continuous vs. Burst Mode
Current firmware sends frames every 500ms. Modify the delay to test continuous transmission or burst patterns.

## Comparison with Main Firmware

| Feature | Debug Firmware | Main Firmware |
|---------|---------------|---------------|
| **Trigger** | Continuous/Timed | Interrupt-driven |
| **Patterns** | Predictable test patterns | Dynamic based on DAQ inputs |
| **Timing** | 500ms intervals | Microsecond response |
| **Complexity** | Simple loops | ISRs + state management |
| **Debug** | Extensive logging | Limited logging |

Use this debug firmware to verify the SPI output chain works correctly, then return to the main firmware once hardware issues are resolved.