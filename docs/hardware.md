# System Architecture

The system separates **state preparation** from **state commit**:

1. **Teensy 4.1** drives SPI MOSI/SCK to all TPIC6B595 chains (no latches).
2. **NI-DAQ** provides:
   - Digital lines for **state selection** (`*_S0/_S1/_S2` or `*_S`) and **LOAD_REQ**.
   - **Register clocks** (`RCK_*`) to latch TPIC outputs—this is the *commit* moment.
   - Triggers to external devices (e.g., microscope, camera).
   - Analog outputs (AO) to set **MFC setpoints**.
   - Analog inputs (AI) to read **MFC flow feedback** (0–5 V).

**Key idea**: the Teensy preloads the next pattern when `*_LOAD_REQ` rises (it samples S-bits), then waits. The DAQ later asserts `RCK_*` to commit the preload. The DAQ can timestamp RCK edges precisely (hardware clock).

## Hardware Blocks

- **Valve drivers**: TPIC6B595 (open-drain high-side sinking arrays).
- **Big olfactometers**: two 16-bit manifolds (use 12 bits now; 4 spares).
- **Small switch valves**: two 2-valve assemblies (8-bit register each; only 2 bits used).
- **MFCs**: Four channels (air L/R, odor L/R). AO setpoint (0–5 V), AI feedback (0–5 V).
- **Timing**: NI USB-6353 sample clock slaves AO & AI to DO.

## Signal Naming (logical)

Digital outputs (from DAQ to Teensy/TPIC):

- **Left big olfactometer**: `OLFACTOMETER_LEFT_S0/_S1/_S2`, `OLFACTOMETER_LEFT_LOAD_REQ`, `RCK_OLFACTOMETER_LEFT`
- **Right big olfactometer**: `OLFACTOMETER_RIGHT_*` (same fields)
- **Left small switch**: `SWITCHVALVE_LEFT_S`, `SWITCHVALVE_LEFT_LOAD_REQ`, `RCK_SWITCHVALVE_LEFT`
- **Right small switch**: `SWITCHVALVE_RIGHT_*`
- **Triggers**: `TRIG_MICRO`, `TRIG_CAMERA`

Analog:

- **AO (setpoints)**: `mfc.air_left_setpoint`, `mfc.air_right_setpoint`, `mfc.odor_left_setpoint`, `mfc.odor_right_setpoint`
- **AI (feedback)**: `mfc.air_left_flowrate`, `mfc.air_right_flowrate`, `mfc.odor_left_flowrate`, `mfc.odor_right_flowrate`
