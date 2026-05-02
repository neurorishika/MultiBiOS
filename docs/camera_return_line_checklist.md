# Camera Return-Line Checklist

This checklist is specific to the Blackfly S BFS-U3-13Y3M camera return wiring used on this rig.

## Confirm the camera GPIO pins first

For the 6-pin Hirose GPIO cable on this camera family:

- green = pin 1 = +12 V / non-isolated input
- black = pin 2 = Line0 = opto-isolated input
- red = pin 3 = Line2 = GPIO / open-drain / optional 3.3 V rail
- white = pin 4 = Line1 = Opto Output 1
- blue = pin 5 = Opto GND
- brown = pin 6 = camera power ground

The important consequence is:

- the white wire is `Line1`, not `Line2`
- the white wire is an isolated opto output referenced to the blue wire
- `Line2` is a different pin and is reported by the live camera as `OpenDrain`

## What the live rig cameras reported

The connected BFS-U3-13Y3M cameras reported:

- `Line0`: `OptoCoupled`, input
- `Line1`: `OptoCoupled`, output
- `Line2`: `OpenDrain`, output
- `Line3`: `TriState`, input

That matches the vendor camera reference and explains why configuring `Line2 = ExposureActive` does not drive the white wire.

## Current DAQ return mapping

- `CAMERA_FRONT_O1` -> `Dev1/port0/line29` -> pin 125
- `CAMERA_SIDE_O1` -> `Dev1/port0/line27` -> pin 123

## Bench rule for white-wire tests

If you are testing the white wire, you must wire the blue `Opto GND` into the measurement circuit.

If only the white wire is connected to the NI-DAQ DI and the blue wire is left floating, the DAQ will usually read a constant low level even if the camera is toggling the internal signal correctly.

## DAQ-only electrical visibility test

Run this from the `multibios-blackfly` environment:

```powershell
python tools/manual_checks/verify_camera_return_line.py --line line1
```

This drives `Line1` through `UserOutput1` and checks whether either DAQ return line changes state.

Interpretation:

- if a DAQ line changes, the return path is electrically visible
- if neither DAQ line changes, the likely causes are missing blue `Opto GND`, missing external bias path on the isolated output, or a wiring mistake at the camera connector

## Optional red-wire test

If you intentionally wired the red wire instead of the white wire, test `Line2` instead:

```powershell
python tools/manual_checks/verify_camera_return_line.py --line line2
```

Interpretation:

- `Line2` is `OpenDrain`
- an open-drain line requires an external pull-up to generate a logic-high level
- without that pull-up, the DAQ will also read a constant low level

## Scope test for the white wire

Recommended connection:

- scope probe tip -> white wire path under test
- scope ground -> blue `Opto GND`

Do not reference the white wire to brown camera ground for the isolated-output test.

## Controlled pull-up or bias test

If the white-wire DAQ test stays flat, do one controlled bench bias test before changing software again.

Recommended pattern:

- white wire -> DAQ DI under test
- white wire -> pull-up resistor to a small external logic rail
- blue `Opto GND` -> external logic ground and DAQ ground/reference

Use a conservative pull-up such as 2.2 kOhm to 10 kOhm to a logic-level rail that is valid for the DAQ input path you are measuring.

Interpretation:

- if the line now toggles, the earlier failure was an electrical bias/reference problem, not trigger acceptance
- if it still does not toggle, the remaining suspects are wrong camera pin, wrong DAQ landing, or camera GPIO configuration on the wrong line

## Practical next bench order

1. Verify the white-wire path with `python tools/manual_checks/verify_camera_return_line.py --line line1`.
2. If it stays flat, confirm the blue `Opto GND` is actually tied into the DAQ/scope reference.
3. If it still stays flat, repeat the test with a controlled external pull-up or bias path.
4. Only test `Line2` if you are intentionally using the red wire and have provided an external pull-up.
