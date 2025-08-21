# Safety Notes

- **Teensy 4.1 is 3.3 V-only** (GPIOs are not 5 V tolerant). If DAQ DO lines are 5 V, use level shifters or opto-isolators (open-collector + pull-ups to 3.3 V).
- **Flyback protection**: TPIC6B595 drives inductive loads (valves). Keep **TPIC SR outputs with proper transient suppression** per datasheet (integrated diodes help, but add bulk capacitance near loads).
- **Grounding**: DAQ GND, Teensy GND, and TPIC logic GND must share a stable reference. Avoid ground loops across long runs; star topology recommended.
- **Power rails**: Size 24 V supply for worst-case concurrent valves + headroom. Add bulk caps near manifolds.
- **EMI**: Route RCK and S-lines away from noisy solenoid conductors. Use twisted pairs and shielding where feasible.
