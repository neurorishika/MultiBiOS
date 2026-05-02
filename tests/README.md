# Automated Tests

This directory is now reserved for automated tests that are intended to run under `pytest`.

Current automated coverage includes:

- FicTrac client parsing, buffering, and callback integration
- FicTrac timing and parity audit helpers
- Teensy transcript handling
- trigger-path utility coverage that can run without moving the live operator scripts back into the pytest tree

Run the focused automated suite from the repo root with:

```bash
pytest tests/
```

For a narrower slice:

```bash
pytest tests/test_fictrac_client.py tests/test_experiment_fictrac_callback.py
```

Operator-facing hardware checks and ad hoc bench utilities have been moved to [tools/manual_checks](../tools/manual_checks). Use that directory, or the PowerShell wrappers in [tools](../tools), for live camera, FicTrac, DAQ, and MFC validation.
