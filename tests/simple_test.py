import nidaqmx
import numpy as np
from nidaqmx.constants import LineGrouping, AcquisitionType

# Define task parameters
device = "Dev1"
port_line = "port0/line0:23"
sample_rate = 1000  # in Hz
frequency = 100  # in Hz

# Calculate waveform parameters
samples_per_cycle = int(sample_rate / frequency)
low_samples = int(samples_per_cycle / 2)
high_samples = samples_per_cycle - low_samples

# Create the square wave data for 24 lines for a single cycle
# The driver will continuously repeat this pattern.
low_state = np.zeros(low_samples, dtype=np.uint32)
# 0xFFFFFF sets the first 24 bits (lines 0-23) to high
high_state = np.full(high_samples, 0xFFFFFF, dtype=np.uint32) 
data = np.concatenate((low_state, high_state))

with nidaqmx.Task() as task:
    # 1. Add the digital output channels to the task.
    # CHAN_FOR_ALL_LINES treats all 24 lines as a single channel,
    # where each sample is a U32 value representing the state of all lines.
    task.do_channels.add_do_chan(
        f"{device}/{port_line}", line_grouping=LineGrouping.CHAN_FOR_ALL_LINES
    )

    # 2. Configure the hardware sample clock for continuous generation.
    # This is the key change to make the output continuous. [2, 4]
    task.timing.cfg_samp_clk_timing(
        rate=sample_rate,
        sample_mode=AcquisitionType.CONTINUOUS
    )

    # 3. Write one cycle of the square wave data to the device buffer.
    # The NI-DAQmx driver will handle repeating this data continuously.
    task.write(data, auto_start=False)

    # 4. Start the task to begin generation.
    task.start()

    print(
        f"Generating a continuous {frequency} Hz square wave at a sample rate of "
        f"{sample_rate} Hz on {device}/{port_line}. "
        "Press Ctrl+C to stop."
    )

    try:
        # Keep the script running indefinitely while the hardware generates the signal.
        while True:
            pass
    except KeyboardInterrupt:
        print("\nStopping the task.")
        # The 'with' statement will automatically stop and close the task.