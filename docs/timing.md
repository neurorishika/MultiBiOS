# Timing Model, Sticky S-bits & Guardrails

## Event anatomy

For a single assembly (e.g., left big olfactometer):

- **Switch** (S-bits change) at `t_switch = t_commit - preload_lead_ms` (minus a few setup samples).
- **Preload** window: `[*_LOAD_REQ]` pulse at `t_load = t_switch + setup_hold`.
- **Commit** window end: `[*_RCK]` pulse at `t_commit`.

The **compiler rejects** any overlapping windows across assemblies:
Window = [t_load, t_rck + rck_width]

## Sticky S-bits

- S-bits reflect the **current state** from the last commit until `t_switch` for the next state.  
- Between `t_switch` and `t_commit`, S-bits show the **upcoming** state (so the Teensy samples the correct code at `LOAD_REQ`).  
- This makes the digital rails self-describing and easy to read in the viewer.

## Practical guidance

- If guardrails trip, increase separation between events or adjust `preload_lead_ms`, `load_req_ms`, or `rck_pulse_ms`.
- For 0.1 ms precision, set `sample_rate: 10000`. Ensure NI-DAQ CPU/USB bandwidth is sufficient for your total sample count.
