# Manual Checks

This directory contains operator-facing hardware scripts and one-off bench utilities that are meant to be run directly, not collected by `pytest`.

Typical examples:

- `mfc_analog_test.py` for MFC monitor and sweep checks
- `continuous_camera_trigger.py` and `verify_camera_trigger_path.py` for camera trigger validation
- `preconnect_scope_test.py` and `verify_camera_return_line.py` for bench wiring checks
- `fictrac_live_probe.py` for live FicTrac probing
- `hardware_test.py` for broad DAQ output validation

Use the PowerShell wrappers in [tools](../) when available:

- [run_mfc_test.ps1](../run_mfc_test.ps1)
- [run_fictrac_probe.ps1](../run_fictrac_probe.ps1)
- [run_fictrac_config_gui.ps1](../run_fictrac_config_gui.ps1)
- [run_valve_test.ps1](../run_valve_test.ps1)

The `tests` directory is now reserved for automated `pytest` coverage.