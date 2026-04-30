# Pre-Connect Oscilloscope Checklist

Date: 2026-03-27

## Purpose

Use this checklist before connecting the NI-DAQ outputs to the Blackfly S BFS-U3-13Y3M cameras or the MFC setpoint inputs.

The goals are:

- verify the DAQ output voltages and pulse timing unloaded
- verify the mapped channels match the intended hardware lines
- reduce the risk of over-voltage, wrong polarity, or wiring mistakes before any external device is attached

This checklist is written for the current hardware map in [config/hardware.yaml](../config/hardware.yaml).

## Current channel map being tested

### Camera trigger and returns

- `TRIG_CAMERA` -> `Dev1/port0/line31` -> pin 127
- `DGND` -> pin 128
- `CAMERA_FRONT_O1` -> `Dev1/port0/line29` -> pin 125
- `CAMERA_SIDE_O1` -> `Dev1/port0/line27` -> pin 123

### MFC analog setpoints

- `mfc.air_left_setpoint` -> `Dev1/ao0`
- `mfc.air_right_setpoint` -> `Dev1/ao1`
- `mfc.odor_left_setpoint` -> `Dev1/ao2`
- `mfc.odor_right_setpoint` -> `Dev1/ao3`

### MFC analog feedback

- `mfc.air_left_flowrate` -> `Dev1/ai0`
- `mfc.air_right_flowrate` -> `Dev1/ai1`
- `mfc.odor_left_flowrate` -> `Dev1/ai2`
- `mfc.odor_right_flowrate` -> `Dev1/ai3`

## Before you start

### External devices must remain disconnected

Do not connect the following while running this checklist:

- Blackfly S camera trigger input wires
- Blackfly S camera `O1` return wires
- MFC setpoint input wires
- MFC analog feedback output wires

The point of this checklist is to validate the DAQ signals first.

### Scope setup

Recommended starting setup:

- use a x10 probe if available
- use DC coupling
- start at 1 V/div or 2 V/div for digital trigger checks
- start at 1 V/div for analog output checks
- start with a 500 ms/div or 1 s/div timebase for trigger pulse tests
- use the DAQ ground that corresponds to the signal family you are probing

### Grounding rule

Probe each signal against the correct DAQ ground reference.

For digital trigger checks:

- probe against `DGND`

For analog output checks:

- probe against the adjacent analog output ground terminal

Do not mix analog and digital grounds casually on the bench just because they are ultimately common inside the DAQ. For clean measurements, reference each signal to its intended return terminal.

## Files added for this test

### Checklist

- [docs/preconnect_scope_checklist.md](../docs/preconnect_scope_checklist.md)

### Test script

- [tests/preconnect_scope_test.py](../tests/preconnect_scope_test.py)
- [tests/verify_camera_return_line.py](../tests/verify_camera_return_line.py)

## Test 1: Verify the camera trigger output unloaded

### Test 1 checks

- trigger line is on the expected DAQ channel
- idle state is low
- pulse amplitude is correct
- pulse width is correct
- repetition period is correct

### Test 1 scope connection

- probe tip -> pin 127 (`TRIG_CAMERA`, `port0/line31`)
- probe ground -> pin 128 (`DGND`)

### Test 1 command to run

```powershell
python tests/preconnect_scope_test.py trigger --hardware config/hardware.yaml --period-ms 1000 --pulse-ms 10 --duration 5 --sample-rate 10000
```

### Test 1 expected result

- one pulse per second
- pulse width approximately 10 ms
- low level near 0 V
- high level appropriate for the NI-DAQ digital line
- clean square edges without obvious collapse or ringing

### Test 1 pass criteria

- pulse timing matches the command
- voltage levels are stable and repeatable
- no unexpected inverted polarity

### Test 1 fail criteria

- pulses are missing
- line is stuck high or low
- pulse width is wrong by a large amount
- voltage amplitude looks wrong for the DAQ output

## Test 2: Verify each MFC analog output unloaded

### Test 2 checks

- each AO line reaches the expected voltage
- the channel mapping is correct
- steps are clean and monotonic
- no clipping or offset error is obvious before connecting MFCs

### Test 2 waveform behavior

The script drives one AO channel at a time through a step sequence while all other AO channels remain at 0 V.

Default step sequence:

- 0.0 V
- 1.0 V
- 2.0 V
- 3.0 V
- 4.0 V
- 5.0 V
- 0.0 V

Each level is held for 1 second by default.

### Test 2 scope connection

Probe one channel at a time.

Suggested order:

1. `ao0` against AO GND
2. `ao1` against AO GND
3. `ao2` against AO GND
4. `ao3` against AO GND

### Test 2 command to run

```powershell
python tests/preconnect_scope_test.py analog --hardware config/hardware.yaml --dwell 1.0 --sample-rate 2000
```

### Test 2 expected result

For each AO channel in turn:

- the voltage steps through 0, 1, 2, 3, 4, 5, and back to 0 V
- other AO channels stay at or near 0 V while that channel is under test

### Test 2 pass criteria

- each commanded level appears at the correct output
- voltages are within reasonable tolerance of the target values
- channels do not appear swapped
- output returns to 0 V at the end

### Test 2 fail criteria

- wrong channel moves
- levels are clipped, shifted, or unstable
- more than one channel steps when only one should

## Test 3: Save bench notes before connecting devices

Record the following after the unloaded tests:

- trigger high level
- trigger low level
- trigger pulse width
- trigger period
- AO0 measured levels
- AO1 measured levels
- AO2 measured levels
- AO3 measured levels
- anything unexpected about noise, overshoot, or polarity

## Test 4: Connect one camera only

Do this only after Test 1 passes.

### Test 4 wiring

- Blackfly S trigger input wire -> pin 127
- Blackfly S ground/return wire -> pin 128
- leave the white wire disconnected for the first trigger-only test

### Test 4 command to run

Use the same trigger command as Test 1.

### Test 4 expected result

- the camera should respond to the external trigger
- the trigger waveform on the DAQ line should still look clean

### Test 4 pass criteria

- camera triggers reliably
- DAQ output does not collapse under load

## Test 5: Connect one camera return line

Do this only after the single-camera trigger test passes.

### Test 5 wiring

- camera white wire -> the assigned DI line
- camera blue wire -> the measurement return for the opto-isolated output
- front camera white -> pin 125
- side camera white -> pin 123

### Test 5 important note

The BFS-U3-13Y3M white wire is `Line1`, which is an opto-coupled output.
Its return reference is the blue `Opto GND` wire, not the camera power ground.

The red wire is `Line2`, which is a separate open-drain GPIO line.

Check whether the specific line under test is:

- push-pull
- open collector
- opto output
- or requires a pull-up or specific external bias

### Test 5 expected result

- once configured as `ExposureActive` or similar on the correct physical line, the camera output should generate a measurable returned pulse
- if the white wire is tested without the blue `Opto GND` reference in the circuit, the DAQ may read no transitions even when the camera is configured correctly

### Test 5 command to run

```powershell
python tests/verify_camera_return_line.py --line line1 --hardware config/hardware.yaml
```

If you are intentionally testing the red wire instead of the white wire:

```powershell
python tests/verify_camera_return_line.py --line line2 --hardware config/hardware.yaml
```

## Test 6: Connect one MFC only

Do this only after Test 2 passes.

### Test 6 order

1. connect one AO setpoint line to one MFC input
2. verify the MFC accepts the range safely
3. connect the corresponding AI feedback line
4. verify the AI trace responds on the expected channel

Do not connect all four MFCs at once for the first live test.

## Recommended bench sequence

Use this order exactly:

1. unloaded camera trigger test
2. unloaded analog output step test
3. one camera trigger test
4. one camera return test
5. one MFC setpoint test
6. one MFC feedback test
7. second camera
8. remaining MFCs

## Quick commands summary

```powershell
python tests/preconnect_scope_test.py trigger --hardware config/hardware.yaml --period-ms 1000 --pulse-ms 10 --duration 5 --sample-rate 10000

python tests/preconnect_scope_test.py analog --hardware config/hardware.yaml --dwell 1.0 --sample-rate 2000

python tests/preconnect_scope_test.py both --hardware config/hardware.yaml --period-ms 1000 --pulse-ms 10 --duration 5 --dwell 1.0 --sample-rate 10000

python tests/verify_camera_return_line.py --line line1 --hardware config/hardware.yaml
```

## Safety stop conditions

Stop immediately if any of the following happen:

- analog output exceeds expected setpoint range
- digital trigger polarity is opposite of what the camera expects
- a supposedly idle line shows unexpected switching
- multiple AO channels move when only one should
- the DAQ line voltage collapses abnormally under a connected load

## Bottom line

Do not connect the cameras or the MFCs until the unloaded scope checks pass first.

The trigger test confirms the digital camera path is electrically sane.
The analog step test confirms the MFC setpoint outputs are electrically sane.
Once both pass, connect one external device at a time and re-test under load.
