#!/usr/bin/env python3
"""
Continuous Latcher Tester
-------------------------
Continuously sends latch signals (GLOBAL_LOAD_REQ + RCK_*) at a specified interval.
Useful for testing Teensy/Shift Register responsiveness without running a full protocol.

Usage:
    python tools/manual_checks/continuous_latcher.py --interval 100
"""

import argparse
import time
import yaml
import numpy as np
import nidaqmx
from nidaqmx.constants import AcquisitionType, LineGrouping
from pathlib import Path
import sys

def load_hardware_config(path):
    try:
        with open(path, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"Error loading hardware config from {path}: {e}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Continuously pulse latch signals (LOAD_REQ + RCK) for testing.")
    parser.add_argument("--interval", type=float, default=100.0, help="Interval between latches in ms (default: 100)")
    parser.add_argument("--hardware", default="config/hardware.yaml", help="Path to hardware.yaml (default: config/hardware.yaml)")
    parser.add_argument("--rate", type=int, default=2000, help="Sample rate in Hz (default: 2000)")
    parser.add_argument("--load-req-ms", type=int, default=1, help="Duration of LOAD_REQ pulse in ms (default: 1)")
    parser.add_argument("--preload-lead-ms", type=int, default=2, help="Delay from LOAD_REQ start to RCK start in ms (default: 2)")
    parser.add_argument("--rck-pulse-ms", type=int, default=1, help="Duration of RCK pulse in ms (default: 1)")
    
    args = parser.parse_args()
    
    # Resolve hardware config path
    hw_path = Path(args.hardware)
    if not hw_path.exists():
        # Try relative to script location if not found (e.g. running from tests/)
        # If script is in tests/, config might be in ../config
        script_dir = Path(__file__).parent.absolute()
        candidate = script_dir.parent / args.hardware
        if candidate.exists():
            hw_path = candidate
        else:
            # Try assuming args.hardware was relative to root, but we are in tests/
            # and user ran python tools/manual_checks/continuous_latcher.py
            # If user ran from root, hw_path (config/hardware.yaml) should exist.
            # If user ran from tests/, hw_path (config/hardware.yaml) might not exist.
            pass

    if not hw_path.exists():
         print(f"Hardware config not found at {hw_path}")
         sys.exit(1)
            
    print(f"Loading hardware from {hw_path}")
    hw_config = load_hardware_config(hw_path)
    
    digital_outputs = hw_config.get('digital_outputs', {})
    
    # Identify relevant lines
    rck_lines = {k: v for k, v in digital_outputs.items() if k.startswith('RCK_')}
    load_req_line = digital_outputs.get('GLOBAL_LOAD_REQ')
    
    if not rck_lines:
        print("No RCK_ lines found in hardware config.")
        sys.exit(1)
        
    print(f"Found {len(rck_lines)} RCK lines: {list(rck_lines.keys())}")
    if load_req_line:
        print(f"Found GLOBAL_LOAD_REQ: {load_req_line}")
    else:
        print("Warning: GLOBAL_LOAD_REQ not found in hardware config.")

    # Prepare channels list for DAQ
    # We will add all RCK lines and the LOAD_REQ line
    channel_names = list(rck_lines.keys())
    channel_phys = [rck_lines[n] for n in channel_names]
    
    if load_req_line:
        channel_names.append('GLOBAL_LOAD_REQ')
        channel_phys.append(load_req_line)
        
    num_channels = len(channel_names)
    
    # Calculate waveform parameters
    dt_ms = 1000.0 / args.rate
    total_samples = int(args.interval / dt_ms)
    
    if total_samples < 10:
        print(f"Warning: Interval {args.interval}ms is too short for rate {args.rate}Hz (only {total_samples} samples).")
    
    # Create waveform array (channels x samples)
    waveform = np.zeros((num_channels, total_samples), dtype=np.bool_)
    
    # Convert timings to samples
    load_start_idx = 0
    load_width_idx = max(1, int(args.load_req_ms / dt_ms))
    
    rck_start_idx = int(args.preload_lead_ms / dt_ms)
    rck_width_idx = max(1, int(args.rck_pulse_ms / dt_ms))
    
    # Fill waveform
    # Last channel is LOAD_REQ if present
    if load_req_line:
        load_ch_idx = num_channels - 1
        waveform[load_ch_idx, load_start_idx : load_start_idx + load_width_idx] = True
        
    # RCK channels are 0 to num_channels-2 (or -1 if no load req)
    rck_end_ch_idx = num_channels - 1 if load_req_line else num_channels
    for i in range(rck_end_ch_idx):
        waveform[i, rck_start_idx : rck_start_idx + rck_width_idx] = True
        
    print(f"Generated waveform: {total_samples} samples ({args.interval} ms)")
    print(f"  LOAD_REQ: Start {load_start_idx}, Width {load_width_idx}")
    print(f"  RCK:      Start {rck_start_idx}, Width {rck_width_idx}")

    try:
        with nidaqmx.Task() as task:
            # Add channels
            for name, phys in zip(channel_names, channel_phys):
                task.do_channels.add_do_chan(
                    phys,
                    name_to_assign_to_lines=name,
                    line_grouping=LineGrouping.CHAN_PER_LINE
                )
            
            # Configure timing for continuous generation
            task.timing.cfg_samp_clk_timing(
                rate=args.rate,
                sample_mode=AcquisitionType.CONTINUOUS,
                samps_per_chan=total_samples
            )
            
            # Write waveform
            # task.write automatically selects the correct writer based on data type
            # For boolean array and CHAN_PER_LINE, it uses write_many_sample_multi_line
            task.write(waveform)
            
            print("Starting continuous generation...")
            task.start()
            
            print("Running. Press Ctrl+C to stop.")
            while True:
                time.sleep(0.5)
                
    except KeyboardInterrupt:
        print("\nStopping...")
    except Exception as e:
        print(f"\nError: {e}")
    finally:
        print("Done.")

if __name__ == "__main__":
    main()
