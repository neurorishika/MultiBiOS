# Validation & QA

## Functional tests (no valves)

1. **Preview compile**  
   `python -m multibios.run_protocol --yaml protocols/example_protocol.yaml --hardware config/hardware.yaml --dry-run --interactive --seed 1`
   - Inspect `preview.html`. Confirm state rails read sensibly; RCK markers align to commits.

2. **Guardrail test**  
   Create two actions with commits closer than `preload_lead_ms + max(load_req_ms,rck_pulse_ms)` and confirm the compiler raises a helpful error.

3. **Randomization reproducibility**  
   Run twice with the same `--seed` and confirm:
   - Identical `digital_edges.csv`
   - Identical `preview.html` (modulo timestamps in `meta.json`)

## Hardware tests (with MFC loopback or live MFCs)

### Protocol parity checks

Use the dedicated tool entry points when you want an end-to-end frame-count parity check across:

- trigger rising edges
- FicTrac raw saved frames
- FicTrac UDP frame count
- FicTrac callback frame count
- second-camera saved frames

Short-run smoke test:

```powershell
.\tools\run_short_frame_parity_test.ps1 -VerboseOutput -Progress
```

Long-run parity test:

```powershell
.\tools\run_long_frame_parity_test.ps1 -VerboseOutput -Progress
```

Behavior:

- both scripts run `multibios.run_protocol`
- both then run `multibios.parity_audit` on the newly created run directory
- nonzero exit means either the protocol run itself failed or trigger/frame parity failed
- the detailed audit is written to `parity_audit.json` inside the run directory

Defaults:

- short-run test uses `protocols/short_protocol.yaml`
- long-run test uses `protocols/odor_lateralization.yaml`
- both use `config/hardware.yaml` unless you override `-HardwarePath`

Optional debugging aid:

- pass `-KeepRawChunks` if you want the parity test to temporarily force `raw_chunk_retention_policy: keep` for that run so the raw `.bin` chunks remain available for manual inspection even when the main hardware config is set to `delete_after_parity`

1. **MFC AO/AI tracking** — see [MFC Analog Test](mfc_analog_test.md) for the full reference.

   Quick pre-session sweep (all four channels, ±0.1 V tolerance):

   ```bash
   python tools/manual_checks/mfc_analog_test.py sweep
   ```

   Live monitor while setting setpoints manually:

   ```bash
   python tools/manual_checks/mfc_analog_test.py monitor --set air_left=2.5 odor_right=1.0
   ```

   Exit code 0 = PASS.  Exit code 1 = one or more channels outside tolerance — investigate before running an experiment.

2. **Latency sanity**
   - Use the viewer’s vertical rulers and hover readouts to confirm spacing between S-bit switch, LOAD_REQ, and RCK matches YAML timing.

3. **Valve drive sanity (dry run)**  
   - Disconnect 24 V loads; probe RCK & S-lines for clean edges and no ringing.
