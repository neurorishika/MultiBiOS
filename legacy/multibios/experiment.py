"""Compatibility shim for the legacy serial experiment runner.

The active hardware-timed path lives in ``multibios.run_protocol``.
The historical computer-timebase runner now lives in ``multibios.serial.experiment``.
"""

from MultiBiOS.legacy.multibios.serial.experiment import *  # noqa: F401,F403
from MultiBiOS.legacy.multibios.serial.experiment import main as _serial_main


if __name__ == "__main__":
	_serial_main()
