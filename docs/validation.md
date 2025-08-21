# Validation & QA

## Functional tests (no valves)

1. **Preview compile**  
   `python -m multibios.run_protocol --yaml config/example_protocol.yaml --hardware config/hardware.yaml --dry-run --interactive --seed 1`
   - Inspect `preview.html`. Confirm state rails read sensibly; RCK markers align to commits.

2. **Guardrail test**  
   Create two actions with commits closer than `preload_lead_ms + max(load_req_ms,rck_pulse_ms)` and confirm the compiler raises a helpful error.

3. **Randomization reproducibility**  
   Run twice with the same `--seed` and confirm:
   - Identical `digital_edges.csv`
   - Identical `preview.html` (modulo timestamps in `meta.json`)

## Hardware tests (with MFC loopback or live MFCs)

1. **MFC AO/AI tracking**
   - Step `mfc.*_setpoint` and verify `AI` tracks setpoint in `preview.html`.
   - Optional: add per-channel scales (V→sccm) and check steady-state error < your tolerance.

2. **Latency sanity**
   - Use the viewer’s vertical rulers and hover readouts to confirm spacing between S-bit switch, LOAD_REQ, and RCK matches YAML timing.

3. **Valve drive sanity (dry run)**  
   - Disconnect 24 V loads; probe RCK & S-lines for clean edges and no ringing.
