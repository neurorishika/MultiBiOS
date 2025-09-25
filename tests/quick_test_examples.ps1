# Quick Hardware Test Examples - PowerShell

# Basic connectivity test - all outputs, 1Hz for 10 seconds
python tests/hardware_test.py --hardware config/hardware.yaml --verbose

# Fast test - higher frequency, shorter duration for quick verification  
python tests/hardware_test.py --frequency 5 --duration 3 --verbose

# Full-scale analog test - test full 0-5V range
python tests/hardware_test.py --amplitude 5.0 --offset 2.5 --frequency 2 --duration 5 --verbose

# High-frequency timing test - verify sample clock accuracy
python tests/hardware_test.py --frequency 50 --sample-rate 5000 --duration 2 --debug

# Debug mode - maximum verbosity for troubleshooting
python tests/hardware_test.py --debug --duration 5