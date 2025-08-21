# Teensy 4.1 Firmware

**Model**: Preload-and-commit with RCK-sense, single-owner SPI bus.

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

Edit arrays in `firmware/teensy41/src/v0.ino` if your plumbing differs.
