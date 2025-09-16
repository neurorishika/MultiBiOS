#!/usr/bin/env python3
"""
Hardware Test Script for MultiBiOS NI USB-6353

This script generates synchronized square waves on all digital and analog outputs
using a single hardware clock. It's designed to test connectivity and verify
hardware configuration.

Features:
- Hardware-clocked synchronization (DO master, AO slave)
- Square wave generation on all configured outputs
- Optional analog input monitoring
- Configurable test parameters (frequency, duration, amplitude)
- Comprehensive logging and visualization
- Test result analysis and reporting

Usage:
    python hardware_test.py --hardware config/hardware.yaml --verbose
    python hardware_test.py --frequency 10 --duration 5 --amplitude 3.3
"""

from __future__ import annotations

import argparse, json, time, logging, sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

import numpy as np
import yaml

import nidaqmx
from nidaqmx.constants import AcquisitionType, Edge, LineGrouping
from nidaqmx.stream_writers import DigitalMultiChannelWriter, AnalogMultiChannelWriter
from nidaqmx.stream_readers import AnalogMultiChannelReader, DigitalMultiChannelReader

# Plotly for visualization
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ----------------------------- hardware adapter -----------------------------
@dataclass
class HardwareMap:
    device: str
    digital_outputs: Dict[str, str]
    analog_outputs: Dict[str, str]
    analog_inputs: Dict[str, str]
    digital_inputs: Dict[str, str]

    def get_all_output_channels(self) -> Tuple[List[str], List[str]]:
        """Get lists of all digital and analog output channel names."""
        return list(self.digital_outputs.keys()), list(self.analog_outputs.keys())


def load_hardware(path: Path) -> HardwareMap:
    """Load hardware configuration from YAML file with detailed logging."""
    logger = logging.getLogger(__name__)
    logger.debug(f"Loading hardware YAML from: {path}")
    
    try:
        y = yaml.safe_load(path.read_text())
        logger.debug(f"YAML keys found: {list(y.keys()) if isinstance(y, dict) else 'Not a dict'}")
        
        # Validate required fields
        if "device" not in y:
            raise ValueError("Missing required 'device' field in hardware YAML")
        
        hw_map = HardwareMap(
            device=y["device"],
            digital_outputs=y.get("digital_outputs", {}),
            analog_outputs=y.get("analog_outputs", {}),
            analog_inputs=y.get("analog_inputs", {}),
            digital_inputs=y.get("digital_inputs", {}),
        )
        
        logger.debug(f"Hardware map created successfully:")
        logger.debug(f"  Device: {hw_map.device}")
        logger.debug(f"  DO channels: {len(hw_map.digital_outputs)}")
        logger.debug(f"  AO channels: {len(hw_map.analog_outputs)}")
        logger.debug(f"  AI channels: {len(hw_map.analog_inputs)}")
        logger.debug(f"  DI channels: {len(hw_map.digital_inputs)}")
        
        return hw_map
        
    except Exception as e:
        logger.error(f"Failed to load hardware configuration: {e}")
        raise


# ----------------------------- test configuration ---------------------------
@dataclass
class TestConfig:
    """Configuration parameters for hardware testing."""
    frequency_hz: float = 1.0          # Square wave frequency in Hz
    duration_sec: float = 10.0         # Test duration in seconds
    sample_rate: int = 1000            # DAQ sample rate in Hz
    ao_amplitude_v: float = 2.5        # Analog output amplitude (0-5V range)
    ao_offset_v: float = 2.5           # Analog output DC offset
    monitor_inputs: bool = True        # Whether to monitor analog inputs
    
    def __post_init__(self):
        # Validation
        if self.frequency_hz <= 0:
            raise ValueError("Frequency must be positive")
        if self.duration_sec <= 0:
            raise ValueError("Duration must be positive")
        if self.sample_rate <= 0:
            raise ValueError("Sample rate must be positive")
        if not (0 <= self.ao_amplitude_v <= 5.0):
            raise ValueError("AO amplitude must be between 0 and 5V")
        if not (0 <= self.ao_offset_v <= 5.0):
            raise ValueError("AO offset must be between 0 and 5V")
        
        # Check Nyquist criterion
        if self.frequency_hz * 2 > self.sample_rate:
            raise ValueError(f"Sample rate {self.sample_rate} Hz too low for {self.frequency_hz} Hz signal")
    
    @property
    def samples_per_cycle(self) -> int:
        """Number of samples per square wave cycle."""
        return int(self.sample_rate / self.frequency_hz)
    
    @property
    def total_samples(self) -> int:
        """Total number of samples for the test duration."""
        return int(self.duration_sec * self.sample_rate)


# ----------------------------- logging setup --------------------------------
def setup_logging(verbose: bool = False, debug: bool = False) -> logging.Logger:
    """Set up logging configuration with appropriate verbosity level."""
    logger = logging.getLogger(__name__)
    
    # Clear any existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Set logging level
    if debug:
        level = logging.DEBUG
    elif verbose:
        level = logging.INFO
    else:
        level = logging.WARNING
    
    logger.setLevel(level)
    
    # Create console handler with formatting
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    
    # Create formatter
    formatter = logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    handler.setFormatter(formatter)
    
    logger.addHandler(handler)
    return logger


# ----------------------------- waveform generation --------------------------
def generate_square_wave(samples: int, samples_per_cycle: int) -> np.ndarray:
    """Generate a square wave array with specified parameters.
    
    Args:
        samples: Total number of samples
        samples_per_cycle: Number of samples per complete cycle
        
    Returns:
        Boolean array representing square wave (True=high, False=low)
    """
    if samples_per_cycle <= 0:
        raise ValueError("Samples per cycle must be positive")
    
    # Create time indices
    indices = np.arange(samples)
    
    # Generate square wave: high for first half of each cycle, low for second half
    cycle_position = indices % samples_per_cycle
    square_wave = cycle_position < (samples_per_cycle // 2)
    
    return square_wave


def generate_analog_square_wave(samples: int, samples_per_cycle: int, 
                              amplitude: float, offset: float) -> np.ndarray:
    """Generate analog square wave with specified amplitude and offset.
    
    Args:
        samples: Total number of samples
        samples_per_cycle: Number of samples per complete cycle
        amplitude: Peak-to-peak amplitude
        offset: DC offset
        
    Returns:
        Float array representing analog square wave
    """
    digital_wave = generate_square_wave(samples, samples_per_cycle)
    
    # Convert to analog: True -> offset + amplitude/2, False -> offset - amplitude/2
    analog_wave = np.where(digital_wave, 
                          offset + amplitude/2, 
                          offset - amplitude/2)
    
    return analog_wave.astype(np.float64)


# ----------------------------- test execution -------------------------------
def run_hardware_test(hw: HardwareMap, config: TestConfig, 
                     output_dir: Optional[Path] = None) -> Dict[str, Any]:
    """Execute hardware test with square wave outputs.
    
    Args:
        hw: Hardware configuration
        config: Test parameters
        output_dir: Optional directory for saving results
        
    Returns:
        Dictionary with test results and statistics
    """
    logger = logging.getLogger(__name__)
    
    logger.info("=== Starting Hardware Test ===")
    logger.info(f"Test Configuration:")
    logger.info(f"  Frequency: {config.frequency_hz} Hz")
    logger.info(f"  Duration: {config.duration_sec} seconds")
    logger.info(f"  Sample rate: {config.sample_rate} Hz")
    logger.info(f"  AO amplitude: {config.ao_amplitude_v} V")
    logger.info(f"  AO offset: {config.ao_offset_v} V")
    logger.info(f"  Total samples: {config.total_samples:,}")
    logger.info(f"  Samples per cycle: {config.samples_per_cycle}")
    
    # Get channel lists
    do_names, ao_names = hw.get_all_output_channels()
    ai_names = list(hw.analog_inputs.keys()) if config.monitor_inputs else []
    
    logger.info(f"Channel Configuration:")
    logger.info(f"  Digital outputs: {len(do_names)} channels")
    for i, name in enumerate(do_names):
        logger.debug(f"    DO[{i}]: {name} -> {hw.digital_outputs[name]}")
    logger.info(f"  Analog outputs: {len(ao_names)} channels") 
    for i, name in enumerate(ao_names):
        logger.debug(f"    AO[{i}]: {name} -> {hw.analog_outputs[name]}")
    if ai_names:
        logger.info(f"  Analog inputs (monitoring): {len(ai_names)} channels")
        for i, name in enumerate(ai_names):
            logger.debug(f"    AI[{i}]: {name} -> {hw.analog_inputs[name]}")
    
    # Generate test waveforms
    logger.info("Generating test waveforms...")
    N = config.total_samples
    spc = config.samples_per_cycle
    
    # Digital output waveforms (each channel gets unique square wave pattern)
    if do_names:
        logger.info(f"Generating digital square waves ({len(do_names)} channels)...")
        do_data = np.zeros((len(do_names), N), dtype=bool)
        
        for i, name in enumerate(do_names):
            # Create unique frequency for each channel (base frequency * (i+1))
            # This will help identify which channels are working
            channel_freq_multiplier = i + 1
            channel_spc = spc // channel_freq_multiplier if channel_freq_multiplier <= spc else 1
            
            if channel_spc < 2:
                channel_spc = 2  # Minimum for a square wave
            
            do_wave = generate_square_wave(N, channel_spc)
            do_data[i, :] = do_wave
            
            logger.debug(f"  Channel {i} ({name}): freq_mult={channel_freq_multiplier}, samples_per_cycle={channel_spc}")
        
        logger.debug(f"DO data shape: {do_data.shape}")
        logger.debug(f"DO data sample (first 20): {do_data[:3, :20] if len(do_names) >= 3 else do_data[:, :20]}")
    else:
        do_data = np.empty((0, N), dtype=bool)
        logger.info("No digital outputs configured")
    
    # Analog output waveforms (all channels get same square wave pattern) 
    if ao_names:
        logger.info(f"Generating analog square waves ({len(ao_names)} channels)...")
        ao_wave = generate_analog_square_wave(N, spc, config.ao_amplitude_v, config.ao_offset_v)
        # Create array with one row per channel, all identical
        ao_data = np.tile(ao_wave, (len(ao_names), 1))
        logger.debug(f"AO data shape: {ao_data.shape}")
        logger.debug(f"AO range: {ao_data.min():.3f}V to {ao_data.max():.3f}V")
    else:
        ao_data = np.empty((0, N), dtype=np.float64)
        logger.info("No analog outputs configured")
    
    # Prepare AI monitoring buffers
    ai_data = None
    if ai_names:
        logger.info(f"Preparing AI monitoring buffers ({len(ai_names)} channels)...")
        ai_data = np.zeros((len(ai_names), N), dtype=np.float64)
    
    # Execute DAQ test
    logger.info("Setting up DAQ tasks...")
    results = {}
    start_time = time.time()
    
    with (
        nidaqmx.Task("DO_TEST") as do_task,
        nidaqmx.Task("AO_TEST") as ao_task, 
        nidaqmx.Task("AI_MONITOR") as ai_task,
    ):
        logger.info("✓ DAQ tasks created")
        
        # Configure DO task (master clock)
        if do_names:
            logger.info("Configuring DO master task...")
            
            # Add channels with detailed logging
            physical_channels = []
            for i, name in enumerate(do_names):
                physical_channel = hw.digital_outputs[name]
                physical_channels.append(physical_channel)
                logger.debug(f"  Adding DO channel {i}: {name} -> {physical_channel}")
                do_task.do_channels.add_do_chan(physical_channel, line_grouping=LineGrouping.CHAN_PER_LINE)
            
            logger.info(f"  Added {len(physical_channels)} DO channels")
            logger.debug(f"  Physical channels: {physical_channels}")
            
            do_task.timing.cfg_samp_clk_timing(
                rate=config.sample_rate,
                active_edge=Edge.RISING,
                sample_mode=AcquisitionType.FINITE,
                samps_per_chan=N,
            )
            
            # Use write_many_sample_port_uint32 like the main protocol runner
            # For CHAN_PER_LINE with port-based writing, each row represents one channel
            logger.debug(f"Writing DO data with shape: {do_data.shape}")
            logger.debug(f"DO data sample (first 10 samples per channel):")
            for i in range(min(5, len(do_names))):
                logger.debug(f"  Channel {i} ({do_names[i]} -> {physical_channels[i]}): {do_data[i, :10]}")
            
            # Verify we have the right number of channels
            if do_data.shape[0] != len(do_names):
                raise ValueError(f"Data shape mismatch: {do_data.shape[0]} data rows vs {len(do_names)} channels")
            
            do_task.write(do_data)
            logger.info("✓ DO master task configured")
        
        # Configure AO task (slave to DO clock)
        if ao_names:
            logger.info("Configuring AO slave task...")
            ao_channel_str = ",".join([hw.analog_outputs[name] for name in ao_names])
            ao_task.ao_channels.add_ao_voltage_chan(ao_channel_str, min_val=0.0, max_val=5.0)
            
            if do_names:
                # Slave to DO clock
                ao_task.timing.cfg_samp_clk_timing(
                    rate=config.sample_rate,
                    source=f"/{hw.device}/do/SampleClock",
                    active_edge=Edge.RISING,
                    sample_mode=AcquisitionType.FINITE,
                    samps_per_chan=N,
                )
                ao_task.triggers.start_trigger.cfg_dig_edge_start_trig(
                    f"/{hw.device}/do/StartTrigger"
                )
            else:
                # Independent clock if no DO channels
                ao_task.timing.cfg_samp_clk_timing(
                    rate=config.sample_rate,
                    active_edge=Edge.RISING,
                    sample_mode=AcquisitionType.FINITE,
                    samps_per_chan=N,
                )
            
            AnalogMultiChannelWriter(ao_task.out_stream).write_many_sample(ao_data)
            logger.info("✓ AO slave task configured")
        
        # Configure AI monitoring task (slave to DO clock or independent)
        if ai_names:
            logger.info("Configuring AI monitoring task...")
            ai_channel_str = ",".join([hw.analog_inputs[name] for name in ai_names])
            ai_task.ai_channels.add_ai_voltage_chan(ai_channel_str, min_val=0.0, max_val=10.0)
            
            if do_names:
                # Slave to DO clock
                ai_task.timing.cfg_samp_clk_timing(
                    rate=config.sample_rate,
                    source=f"/{hw.device}/do/SampleClock", 
                    active_edge=Edge.RISING,
                    sample_mode=AcquisitionType.FINITE,
                    samps_per_chan=N,
                )
                ai_task.triggers.start_trigger.cfg_dig_edge_start_trig(
                    f"/{hw.device}/do/StartTrigger"
                )
            else:
                # Independent clock
                ai_task.timing.cfg_samp_clk_timing(
                    rate=config.sample_rate,
                    active_edge=Edge.RISING, 
                    sample_mode=AcquisitionType.FINITE,
                    samps_per_chan=N,
                )
            
            ai_reader = AnalogMultiChannelReader(ai_task.in_stream)
            logger.info("✓ AI monitoring task configured")
        
        # Start tasks in proper sequence
        logger.info("Starting DAQ execution...")
        timeout = max(10.0, config.duration_sec + 5.0)
        
        if ai_names:
            logger.debug("Starting AI monitoring task...")
            ai_task.start()
        if ao_names:
            logger.debug("Starting AO output task...")
            ao_task.start()
        if do_names:
            logger.debug("Starting DO master task...")
            do_task.start()
        elif ao_names:
            # If no DO, AO becomes master
            logger.debug("AO task is master (no DO channels)")
        
        logger.info(f"⏱️  Test running for {config.duration_sec} seconds...")
        
        # Wait for completion
        if do_names:
            do_task.wait_until_done(timeout=timeout)
        elif ao_names:
            ao_task.wait_until_done(timeout=timeout)
        
        execution_time = time.time() - start_time
        logger.info(f"✓ Test execution completed in {execution_time:.2f} seconds")
        
        # Stop tasks and collect data
        if do_names:
            do_task.stop()
        if ao_names:
            ao_task.stop()
        
        if ai_names:
            logger.info("Reading AI monitoring data...")
            ai_reader.read_many_sample(ai_data, number_of_samples_per_channel=N, timeout=timeout)
            ai_task.stop()
            
            # Analyze AI data
            logger.info("Analyzing AI monitoring results:")
            for i, name in enumerate(ai_names):
                min_val, max_val = ai_data[i].min(), ai_data[i].max()
                mean_val, std_val = ai_data[i].mean(), ai_data[i].std()
                logger.info(f"  {name}: {min_val:.3f}V to {max_val:.3f}V (mean: {mean_val:.3f}V, std: {std_val:.3f}V)")
    
    # Compile results
    results = {
        "config": {
            "frequency_hz": config.frequency_hz,
            "duration_sec": config.duration_sec,
            "sample_rate": config.sample_rate,
            "ao_amplitude_v": config.ao_amplitude_v,
            "ao_offset_v": config.ao_offset_v,
            "total_samples": N,
            "samples_per_cycle": spc,
        },
        "channels": {
            "digital_outputs": do_names,
            "analog_outputs": ao_names,  
            "analog_inputs": ai_names,
        },
        "execution": {
            "start_time": start_time,
            "execution_time_sec": execution_time,
            "success": True,
        },
        "data": {
            "do_data": do_data if do_names else None,
            "ao_data": ao_data if ao_names else None,
            "ai_data": ai_data if ai_names else None,
        }
    }
    
    # Add AI statistics if available
    if ai_names and ai_data is not None:
        results["ai_statistics"] = {}
        for i, name in enumerate(ai_names):
            results["ai_statistics"][name] = {
                "min_v": float(ai_data[i].min()),
                "max_v": float(ai_data[i].max()),
                "mean_v": float(ai_data[i].mean()),
                "std_v": float(ai_data[i].std()),
            }
    
    logger.info("✓ Hardware test completed successfully")
    return results


# ----------------------------- visualization --------------------------------
def create_test_visualization(results: Dict[str, Any], output_file: Path) -> None:
    """Create interactive visualization of test results."""
    logger = logging.getLogger(__name__)
    logger.info("Creating test visualization...")
    
    config = results["config"]
    channels = results["channels"]
    data = results["data"]
    
    # Time axis
    t_ms = np.arange(config["total_samples"]) * (1000.0 / config["sample_rate"])
    
    # Calculate subplot layout
    subplot_count = 0
    if channels["digital_outputs"]:
        subplot_count += 1
    if channels["analog_outputs"]:
        subplot_count += 1
    if channels["analog_inputs"]:
        subplot_count += 1
    
    if subplot_count == 0:
        logger.warning("No data to plot")
        return
    
    # Create subplots
    subplot_titles = []
    if channels["digital_outputs"]:
        subplot_titles.append("Digital Outputs (Square Wave Test)")
    if channels["analog_outputs"]:
        subplot_titles.append("Analog Outputs (Square Wave Test)")
    if channels["analog_inputs"]:
        subplot_titles.append("Analog Inputs (Monitoring)")
    
    fig = make_subplots(
        rows=subplot_count, 
        cols=1,
        subplot_titles=subplot_titles,
        vertical_spacing=0.08,
        shared_xaxes=True,
    )
    
    row = 1
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    
    # Plot digital outputs
    if channels["digital_outputs"] and data["do_data"] is not None:
        logger.debug("Adding digital output traces...")
        do_data = np.array(data["do_data"]) if not isinstance(data["do_data"], np.ndarray) else data["do_data"]
        for i, name in enumerate(channels["digital_outputs"]):
            # Offset each channel vertically for visibility
            y_data = do_data[i].astype(float) + i * 1.2
            fig.add_trace(
                go.Scatter(
                    x=t_ms,
                    y=y_data,
                    mode='lines',
                    name=name,
                    line=dict(color=colors[i % len(colors)], width=1),
                    showlegend=True,
                ),
                row=row, col=1
            )
        fig.update_yaxes(title_text="Channel + Offset", row=row, col=1)
        row += 1
    
    # Plot analog outputs
    if channels["analog_outputs"] and data["ao_data"] is not None:
        logger.debug("Adding analog output traces...")
        ao_data = np.array(data["ao_data"]) if not isinstance(data["ao_data"], np.ndarray) else data["ao_data"]
        for i, name in enumerate(channels["analog_outputs"]):
            fig.add_trace(
                go.Scatter(
                    x=t_ms,
                    y=ao_data[i],
                    mode='lines',
                    name=name,
                    line=dict(color=colors[i % len(colors)], width=1),
                    showlegend=True,
                ),
                row=row, col=1
            )
        fig.update_yaxes(title_text="Voltage (V)", row=row, col=1)
        row += 1
    
    # Plot analog inputs 
    if channels["analog_inputs"] and data["ai_data"] is not None:
        logger.debug("Adding analog input traces...")
        ai_data = np.array(data["ai_data"]) if not isinstance(data["ai_data"], np.ndarray) else data["ai_data"]
        for i, name in enumerate(channels["analog_inputs"]):
            fig.add_trace(
                go.Scatter(
                    x=t_ms,
                    y=ai_data[i],
                    mode='lines',
                    name=name,
                    line=dict(color=colors[i % len(colors)], width=1),
                    showlegend=True,
                ),
                row=row, col=1
            )
        fig.update_yaxes(title_text="Voltage (V)", row=row, col=1)
    
    # Update layout
    fig.update_layout(
        title=f"Hardware Test Results - {config['frequency_hz']}Hz Square Wave",
        height=300 * subplot_count,
        showlegend=True,
    )
    fig.update_xaxes(title_text="Time (ms)", row=subplot_count, col=1)
    
    # Save visualization
    logger.info(f"Saving visualization to: {output_file}")
    fig.write_html(output_file, include_plotlyjs="cdn")
    logger.info("✓ Visualization created successfully")


# ----------------------------- main -----------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Hardware test script for MultiBiOS NI USB-6353"
    )
    ap.add_argument(
        "--hardware", 
        default="config/hardware.yaml", 
        help="Hardware configuration YAML file"
    )
    ap.add_argument(
        "--device", 
        help="Override device name from hardware config"
    )
    ap.add_argument(
        "--frequency", "-f",
        type=float, 
        default=1.0,
        help="Square wave frequency in Hz (default: 1.0)"
    )
    ap.add_argument(
        "--duration", "-d",
        type=float,
        default=10.0, 
        help="Test duration in seconds (default: 10.0)"
    )
    ap.add_argument(
        "--sample-rate", "-r",
        type=int,
        default=1000,
        help="DAQ sample rate in Hz (default: 1000)"  
    )
    ap.add_argument(
        "--amplitude", "-a",
        type=float,
        default=2.5,
        help="Analog output amplitude in volts (default: 2.5)"
    )
    ap.add_argument(
        "--offset", "-o", 
        type=float,
        default=2.5,
        help="Analog output DC offset in volts (default: 2.5)"
    )
    ap.add_argument(
        "--no-monitor-inputs",
        action="store_true",
        help="Disable analog input monitoring"
    )
    ap.add_argument(
        "--output-dir",
        type=Path,
        default="tests/results",
        help="Output directory for results (default: tests/results)"
    )
    ap.add_argument(
        "--verbose", "-v",
        action="store_true", 
        help="Enable verbose logging"
    )
    ap.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )
    
    args = ap.parse_args()
    
    # Setup logging
    logger = setup_logging(verbose=args.verbose, debug=args.debug)
    
    logger.info("=== MultiBiOS Hardware Test Starting ===")
    logger.info(f"Arguments: {vars(args)}")
    
    # Load hardware configuration
    hw_path = Path(args.hardware)
    if not hw_path.exists():
        logger.error(f"Hardware file not found: {hw_path}")
        raise SystemExit(f"Hardware file not found: {hw_path}")
    
    logger.info(f"Loading hardware configuration: {hw_path.absolute()}")
    hw = load_hardware(hw_path)
    
    if args.device:
        logger.info(f"Overriding device: {hw.device} -> {args.device}")
        hw.device = args.device
    
    # Create test configuration
    try:
        config = TestConfig(
            frequency_hz=args.frequency,
            duration_sec=args.duration,
            sample_rate=args.sample_rate,
            ao_amplitude_v=args.amplitude,
            ao_offset_v=args.offset,
            monitor_inputs=not args.no_monitor_inputs,
        )
    except ValueError as e:
        logger.error(f"Invalid test configuration: {e}")
        raise SystemExit(f"Configuration error: {e}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir.absolute()}")
    
    # Create timestamped results directory
    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    test_dir = output_dir / f"hardware_test_{timestamp}"
    test_dir.mkdir(exist_ok=True)
    logger.info(f"Test results directory: {test_dir}")
    
    try:
        # Run hardware test
        results = run_hardware_test(hw, config, test_dir)
        
        # Create visualization BEFORE converting to JSON (to preserve numpy arrays)
        viz_file = test_dir / "test_visualization.html" 
        create_test_visualization(results, viz_file)
        
        # Save results
        results_file = test_dir / "test_results.json"
        logger.info(f"Saving test results: {results_file}")
        
        # Convert numpy arrays to lists for JSON serialization
        json_results = results.copy()
        json_results["data"] = results["data"].copy()  # Make a deep copy of data dict
        if json_results["data"]["do_data"] is not None:
            json_results["data"]["do_data"] = json_results["data"]["do_data"].tolist()
        if json_results["data"]["ao_data"] is not None:
            json_results["data"]["ao_data"] = json_results["data"]["ao_data"].tolist()
        if json_results["data"]["ai_data"] is not None:
            json_results["data"]["ai_data"] = json_results["data"]["ai_data"].tolist()
        
        with results_file.open("w") as f:
            json.dump(json_results, f, indent=2)
        
        # Save hardware config copy
        hw_copy = test_dir / "hardware.yaml"
        hw_copy.write_text(hw_path.read_text())
        logger.info(f"Hardware config copy saved: {hw_copy}")
        
        # Final summary
        logger.info("=== TEST COMPLETION SUMMARY ===")
        logger.info(f"✓ Test completed successfully")
        logger.info(f"✓ Duration: {results['execution']['execution_time_sec']:.2f} seconds")
        logger.info(f"✓ Digital outputs tested: {len(results['channels']['digital_outputs'])}")
        logger.info(f"✓ Analog outputs tested: {len(results['channels']['analog_outputs'])}")
        if results['channels']['analog_inputs']:
            logger.info(f"✓ Analog inputs monitored: {len(results['channels']['analog_inputs'])}")
        logger.info(f"✓ Results directory: {test_dir}")
        logger.info(f"✓ Visualization: {viz_file}")
        
        print(f"\nHardware test complete!")
        print(f"Results: {test_dir}")
        print(f"Visualization: {viz_file}")
        
    except Exception as e:
        logger.error(f"Hardware test failed: {e}")
        logger.debug("Exception details:", exc_info=True)
        raise SystemExit(f"Test failed: {e}")


if __name__ == "__main__":
    main()