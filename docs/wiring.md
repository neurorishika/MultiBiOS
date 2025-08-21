# Hardware Wiring & Pin Configuration

This document describes the electrical connections between the NI DAQ system and the Teensy 4.1 microcontroller for valve control.

## DAQ Channel Mapping

Configure your DAQ channels in `config/hardware.yaml`:

```yaml
device: "Dev1"
digital_outputs:
  OLFACTOMETER_LEFT_S0:        "Dev1/port0/line0"
  OLFACTOMETER_LEFT_S1:        "Dev1/port0/line1"
  OLFACTOMETER_LEFT_S2:        "Dev1/port0/line2"
  OLFACTOMETER_LEFT_LOAD_REQ:  "Dev1/port0/line3"
  RCK_OLFACTOMETER_LEFT:       "Dev1/port0/line4"
  OLFACTOMETER_RIGHT_S0:       "Dev1/port0/line5"
  OLFACTOMETER_RIGHT_S1:       "Dev1/port0/line6"
  OLFACTOMETER_RIGHT_S2:       "Dev1/port0/line7"
  OLFACTOMETER_RIGHT_LOAD_REQ: "Dev1/port0/line8"
  RCK_OLFACTOMETER_RIGHT:      "Dev1/port0/line9"
  SWITCHVALVE_LEFT_S:          "Dev1/port0/line10"
  SWITCHVALVE_LEFT_LOAD_REQ:   "Dev1/port0/line11"
  RCK_SWITCHVALVE_LEFT:        "Dev1/port0/line12"
  SWITCHVALVE_RIGHT_S:         "Dev1/port0/line13"
  SWITCHVALVE_RIGHT_LOAD_REQ:  "Dev1/port0/line14"
  RCK_SWITCHVALVE_RIGHT:       "Dev1/port0/line15"
  TRIG_MICRO:                  "Dev1/port0/line16"
  TRIG_CAMERA:                 "Dev1/port0/line17"

analog_outputs:
  mfc.air_left_setpoint:   "Dev1/ao0"
  mfc.air_right_setpoint:  "Dev1/ao1"
  mfc.odor_left_setpoint:  "Dev1/ao2"
  mfc.odor_right_setpoint: "Dev1/ao3"

analog_inputs:
  mfc.air_left_flowrate:   "Dev1/ai0"
  mfc.air_right_flowrate:  "Dev1/ai1"
  mfc.odor_left_flowrate:  "Dev1/ai2"
  mfc.odor_right_flowrate: "Dev1/ai3"
```

## Teensy 4.1 Pin Configuration

**SPI Communication:**

- MOSI = Pin 11
- SCK = Pin 13

**RCK Sense Inputs (from DAQ):**

- Olfactometer Left = Pin 2
- Olfactometer Right = Pin 3  
- Switch Valve Left = Pin 4
- Switch Valve Right = Pin 5

**Ready Signal Outputs (optional monitoring):**

- Olfactometer Left = Pin 6
- Olfactometer Right = Pin 7
- Switch Valve Left = Pin 8  
- Switch Valve Right = Pin 9

**State Input Pins (from DAQ):**

- Olfactometer Left: S0=Pin 14, S1=Pin 15, S2=Pin 16
- Olfactometer Right: S0=Pin 17, S1=Pin 18, S2=Pin 19
- Switch Valve Left: S=Pin 20
- Switch Valve Right: S=Pin 21

**Load Request Inputs (from DAQ):**

- Olfactometer Left = Pin 22
- Olfactometer Right = Pin 23
- Switch Valve Left = Pin 24
- Switch Valve Right = Pin 25

**Wiring Notes:**

- Connect DAQ RCK output lines to corresponding Teensy RCK sense inputs
- Keep all grounds common between DAQ and Teensy
- Use level shifters if DAQ outputs 5V (Teensy inputs are 3.3V tolerant)
- Label panel connections with descriptive names (not A/B/C/D)
