# Teensy 4.1 Firmware

**Model**: Preload-and-commit with RCK-sense, single-owner SPI bus.

Current primary open-loop firmware source: `firmware/teensy41/src/open_loop_controller/open_loop_controller.ino`

Alternative dual-mode firmware source: `firmware/teensy41/src/dual_mode_controller/dual_mode_controller.ino`

- DAQ asserts `*_LOAD_REQ` → Teensy ISR:
  1. Samples `S` bits for that assembly.
  2. Shifts the corresponding 16-bit (big) or 8-bit (small) pattern via SPI to the daisy chain (no latch).
  3. Sets `READY_*` high and **locks the bus** to that owner.
- DAQ later asserts `RCK_*` → Teensy ISR senses it:
  - Drops `READY_*`, **unlocks the bus**.

> Only **one staged preload** may be pending at a time. The compiler enforces this with **guardrails**.

## State coding

Big manifold (16-bit, using v0..v11, 4 spare):

- `AIR`: v0,v1 = 1
- `ODOR1`: v2,v3 = 1  
…  
- `ODOR5`: v10,v11 = 1  
- `FLUSH`: v0..v11 = 1

Small switch (8-bit, using v0..v1):

- `CLEAN`: both 0  
- `ODOR`:  both 1

The current `open_loop_controller` firmware also emits structured USB serial telemetry for host-side audit logging, including `MODE`, `READY`, `VALVE`, `COMMIT`, and `FAULT` lines.

The `dual_mode_controller` sketch boots in serial-controller mode with a reduced command set (`CTRL`, `OD1`-`OD5`, `CLN`, `ODR`, `RESET`, `TEST START`, `TEST STOP`) and switches into the same DAQ-driven open-loop preload/commit path when it receives `OPENLOOP START`. Use `OPENLOOP STOP` to detach the DAQ interrupts, drop all `READY_*` lines low, and restore the staged serial frame.

Enable host-side transcript capture in `config/hardware.yaml` with:

```yaml
teensy:
  port: "COM4"
  baud: 115200
  capture_serial: true
```

Edit pin mappings and state tables in `firmware/teensy41/src/open_loop_controller/open_loop_controller.ino` if your plumbing differs.
