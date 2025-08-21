# Troubleshooting

## Compile-time

- **Overlapping pre-load windows**
  - Error shows two labeled windows with times.
  - Increase spacing between commits, or increase `preload_lead_ms`.

- **Unknown device/channel**
  - Ensure the device key exists in `hardware.yaml` and matches schema expectations.

## Run-time

- **AI capture missing**
  - Check `analog_inputs` in `hardware.yaml`.
  - Ensure DO is master clock and AI slaves to `/<Dev>/do/SampleClock`.

- **No latching / valves don’t change**
  - Confirm DAQ actually toggles `*_LOAD_REQ` and `RCK_*`.
  - Confirm Teensy READY LEDs (optional pins) assert during preload and drop on RCK.
  - Verify level shifting to Teensy (3.3 V).

- **MFC feedback saturates**
  - Confirm `ai` voltage range (set ±10 V is fine for 0–5 V signals).
  - Check wiring and common ground.

- **Viewer shows nothing**
  - Open the run folder and check that `compiled_do.npz` and `preview.html` were created.
