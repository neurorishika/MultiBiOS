#!/usr/bin/env python3
"""
Run hardware-clocked NI USB-6353 protocol and log MFC analog feedback + READY DI rails.

- DO (master): drives S bits, LOAD_REQ, RCK, triggers
- AO (slave): drives MFC setpoints
- AI (slave): records MFC feedback (0–10 V) locked to DO sample clock
- DI (slave): records READY rails from Teensy, locked to DO sample clock

Artifacts are written to data/runs/YYYY-MM-DD_HH-MM-SS/
- compiled_do.npz / compiled_ao.npz
- capture_ai.npz (MFC feedback, optional)
- capture_di.npz (READY rails, optional)
- do_map.json / ao_map.json / di_map.json
- rck_edges.csv (planned commits)
- digital_edges.csv (rising/falling edges for all DO lines)
- ready_edges.csv (rising/falling edges for READY DI lines, if present)
- preview.html (interactive Plotly: DO + AO + AI/DI overlays)
"""

from __future__ import annotations

import argparse, json, time, logging, sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Optional, List

import numpy as np
import yaml

import nidaqmx
from nidaqmx.constants import AcquisitionType, Edge, LineGrouping
from nidaqmx.stream_writers import AnalogMultiChannelWriter
from nidaqmx.stream_readers import AnalogMultiChannelReader
from multibios.viz_helpers import make_protocol_figure, write_edge_csv

# Plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Compiler
from multibios.protocol.schema import ProtocolCompiler, TimingConfig, CompileError

# Visualization helpers
from multibios.viz_helpers import make_protocol_figure, write_edge_csv


# ----------------------------- hardware adapter -----------------------------
@dataclass
class HardwareMap:
    device: str
    digital_outputs: Dict[str, str]
    analog_outputs: Dict[str, str]
    analog_inputs: Dict[str, str]
    digital_inputs: Dict[str, str]  # READY rails (Teensy -> NI-DAQ)

    # adapter fields the compiler expects
    @property
    def do_lines(self) -> Dict[str, str]:
        return self.digital_outputs

    @property
    def ao_channels(self) -> Dict[str, str]:
        return self.analog_outputs


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


# ----------------------------- logging utils --------------------------------
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


# ----------------------------- logging utils --------------------------------
def ensure_run_dir(root: Path) -> Path:
    """Create timestamped run directory with logging."""
    logger = logging.getLogger(__name__)
    
    ts = time.strftime("%Y-%m-%d_%H-%M-%S")
    d = root / ts
    
    logger.debug(f"Creating run directory: {d}")
    logger.debug(f"  Root directory: {root}")
    logger.debug(f"  Timestamp: {ts}")
    
    try:
        d.mkdir(parents=True, exist_ok=False)
        logger.debug(f"✓ Run directory created successfully")
    except FileExistsError:
        logger.warning(f"Run directory already exists (should be rare): {d}")
    except Exception as e:
        logger.error(f"Failed to create run directory: {e}")
        raise
        
    return d


# ----------------------------- main -----------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Run NI 6353 hardware-clocked protocol with AI/DI logging."
    )
    ap.add_argument(
        "--yaml", default="config/example_protocol.yaml", help="Protocol YAML"
    )
    ap.add_argument(
        "--hardware", default="config/hardware.yaml", help="Hardware map YAML"
    )
    ap.add_argument("--device", help="Override device name (else from hardware.yaml)")
    ap.add_argument("--dry-run", action="store_true", help="Compile only; no hardware")
    ap.add_argument(
        "--interactive",
        action="store_true",
        help="Always save interactive HTML preview",
    )
    ap.add_argument("--out-root", default="data/runs", help="Run folder root")
    ap.add_argument(
        "--verbose", "-v", 
        action="store_true", 
        help="Enable verbose logging and detailed progress output"
    )
    ap.add_argument(
        "--debug", 
        action="store_true", 
        help="Enable debug logging (even more verbose than --verbose)"
    )
    # Optional pulse tuning overrides (otherwise read from YAML)
    ap.add_argument("--preload-lead-ms", type=int)
    ap.add_argument("--load-req-ms", type=int)
    ap.add_argument("--rck-ms", type=int)
    ap.add_argument("--trig-ms", type=int)
    ap.add_argument(
        "--seed",
        type=int,
        help="Override protocol.timing.seed for reproducible randomization",
    )
    args = ap.parse_args()

    # Set up logging based on verbosity level
    logger = setup_logging(verbose=args.verbose, debug=args.debug)
    
    logger.info("=== MultiBiOS Protocol Runner Starting ===")
    logger.info(f"Command line arguments: {vars(args)}")
    logger.info(f"Python version: {sys.version}")
    logger.info(f"Working directory: {Path.cwd()}")
    
    # Validate input files
    proto_path = Path(args.yaml)
    hw_path = Path(args.hardware)
    
    logger.info(f"Protocol file path: {proto_path.absolute()}")
    logger.info(f"Hardware file path: {hw_path.absolute()}")
    
    if not proto_path.exists():
        logger.error(f"Protocol file not found: {proto_path}")
        raise SystemExit(f"Protocol file not found: {proto_path}")
    if not hw_path.exists():
        logger.error(f"Hardware file not found: {hw_path}")
        raise SystemExit(f"Hardware file not found: {hw_path}")
        
    logger.info("✓ Input files validated successfully")

    # Load hardware configuration with detailed logging
    logger.info("Loading hardware configuration...")
    logger.debug(f"Reading hardware YAML from: {hw_path}")
    
    hw = load_hardware(hw_path)
    logger.info(f"✓ Hardware configuration loaded successfully")
    logger.info(f"  Device: {hw.device}")
    logger.info(f"  Digital outputs: {len(hw.digital_outputs)} channels")
    for name, channel in hw.digital_outputs.items():
        logger.debug(f"    {name} -> {channel}")
    logger.info(f"  Analog outputs: {len(hw.analog_outputs)} channels")  
    for name, channel in hw.analog_outputs.items():
        logger.debug(f"    {name} -> {channel}")
    logger.info(f"  Analog inputs: {len(hw.analog_inputs)} channels")
    for name, channel in hw.analog_inputs.items():
        logger.debug(f"    {name} -> {channel}")
    logger.info(f"  Digital inputs: {len(hw.digital_inputs)} channels")
    for name, channel in hw.digital_inputs.items():
        logger.debug(f"    {name} -> {channel}")
    
    if args.device:
        logger.info(f"Overriding device name: {hw.device} -> {args.device}")
        hw.device = args.device

    # Load and process protocol YAML
    logger.info("Loading protocol configuration...")
    logger.debug(f"Reading protocol YAML from: {proto_path}")
    
    y = yaml.safe_load(proto_path.read_text())
    logger.info("✓ Protocol YAML loaded successfully")
    logger.debug(f"Protocol keys: {list(y.keys()) if isinstance(y, dict) else 'Not a dict'}")
    
    if args.seed is not None:
        logger.info(f"Overriding protocol seed: {args.seed}")
        y.setdefault("protocol", {}).setdefault("timing", {})["seed"] = int(args.seed)
        
    # Process timing configuration with detailed logging
    logger.info("Processing timing configuration...")
    t = y.get("protocol", {}).get("timing", {})
    logger.debug(f"Raw timing config: {t}")
    
    tcfg = TimingConfig(
        base_unit=t.get("base_unit", "ms"),
        sample_rate=int(t.get("sample_rate", 1000)),
        camera_interval_ms=int(t.get("camera_interval", 0)),
        camera_pulse_ms=int(t.get("camera_pulse_duration", 5)),
        preload_lead_ms=int(
            args.preload_lead_ms
            if args.preload_lead_ms is not None
            else t.get("preload_lead_ms", 2)
        ),
        load_req_ms=int(
            args.load_req_ms
            if args.load_req_ms is not None
            else t.get("load_req_ms", 1)
        ),
        rck_pulse_ms=int(
            args.rck_ms if args.rck_ms is not None else t.get("rck_pulse_ms", 1)
        ),
        trig_pulse_ms=int(
            args.trig_ms if args.trig_ms is not None else t.get("trig_pulse_ms", 5)
        ),
        setup_hold_samples=int(t.get("setup_hold_samples", 5)),
    )
    
    logger.info("✓ Timing configuration processed")
    logger.info(f"  Sample rate: {tcfg.sample_rate} Hz")
    logger.info(f"  Base unit: {tcfg.base_unit}")
    logger.info(f"  Camera interval: {tcfg.camera_interval_ms} ms")
    logger.info(f"  Camera pulse: {tcfg.camera_pulse_ms} ms")
    logger.info(f"  Preload lead: {tcfg.preload_lead_ms} ms")
    logger.info(f"  Load request: {tcfg.load_req_ms} ms")
    logger.info(f"  RCK pulse: {tcfg.rck_pulse_ms} ms")
    logger.info(f"  Trigger pulse: {tcfg.trig_pulse_ms} ms")
    logger.info(f"  Setup/hold samples: {tcfg.setup_hold_samples}")

    # Compile protocol with detailed progress logging
    logger.info("=== Starting Protocol Compilation ===")
    comp = ProtocolCompiler(hw, tcfg)
    logger.info("✓ Protocol compiler initialized")
    
    try:
        logger.info("Compiling protocol from YAML...")
        start_time = time.time()
        comp.compile_from_yaml(y)
        compile_time = time.time() - start_time
        
        logger.info(f"✓ Protocol compilation completed in {compile_time:.2f} seconds")
        logger.info(f"  Total samples: {comp.N}")
        logger.info(f"  Duration: {comp.N * comp.dt_ms:.1f} ms ({comp.N * comp.dt_ms / 1000:.2f} seconds)")
        logger.info(f"  Sample time step: {comp.dt_ms:.3f} ms")
        logger.info(f"  Digital output lines: {len(comp.line_order)}")
        logger.info(f"  Analog output channels: {len(comp.ao_order)}")
        logger.info(f"  RNG seed used: {getattr(comp, 'rng_seed', 'N/A')}")
        
        if hasattr(comp, 'rck_log') and comp.rck_log:
            logger.info(f"  RCK commit events: {len(comp.rck_log)}")
            logger.debug("  RCK event details:")
            for i, (sig, si, tms) in enumerate(comp.rck_log[:5]):  # Show first 5
                logger.debug(f"    {i+1}: {sig} at sample {si} ({tms:.3f} ms)")
            if len(comp.rck_log) > 5:
                logger.debug(f"    ... and {len(comp.rck_log) - 5} more")
                
    except CompileError as e:
        logger.error(f"Protocol compilation failed: {e}")
        raise SystemExit(f"[compile error] {e}")

    # Create run directory and save artifacts with detailed logging
    logger.info("=== Creating Run Directory and Artifacts ===")
    run_dir = ensure_run_dir(Path(args.out_root))
    logger.info(f"✓ Run directory created: {run_dir}")
    logger.info(f"  Run timestamp: {run_dir.name}")
    
    # Save compilation report
    logger.info("Saving compilation report...")
    report_file = run_dir / "compile_report.json"
    report_file.write_text(json.dumps(comp.report, indent=2))
    logger.info(f"  ✓ Compilation report: {report_file}")
    logger.debug(f"    Report keys: {list(comp.report.keys()) if hasattr(comp, 'report') and comp.report else 'No report'}")
    
    # Save input files for reproducibility
    logger.info("Saving input files for reproducibility...")
    proto_copy = run_dir / "protocol.yaml"
    hw_copy = run_dir / "hardware.yaml"
    proto_copy.write_text(proto_path.read_text())
    hw_copy.write_text(hw_path.read_text())
    logger.info(f"  ✓ Protocol YAML copy: {proto_copy}")
    logger.info(f"  ✓ Hardware YAML copy: {hw_copy}")
    
    # Save metadata
    logger.info("Saving run metadata...")
    meta_data = {
        "device": hw.device,
        "sample_rate": comp.tcfg.sample_rate,
        "duration_ms": comp.N * comp.dt_ms,
        "rng_seed": getattr(comp, 'rng_seed', None),
        "args": vars(args),
    }
    meta_file = run_dir / "meta.json"
    meta_file.write_text(json.dumps(meta_data, indent=2))
    logger.info(f"  ✓ Metadata: {meta_file}")
    logger.debug(f"    Metadata: {meta_data}")
    
    # Save RCK edges log
    logger.info("Saving RCK edges log...")
    rck_file = run_dir / "rck_edges.csv"
    with rck_file.open("w") as f:
        f.write("signal,sample_idx,time_ms\n")
        for sig, si, tms in comp.rck_log:
            f.write(f"{sig},{si},{tms:.3f}\n")
    logger.info(f"  ✓ RCK edges: {rck_file} ({len(comp.rck_log)} events)")

    # Save channel mapping files
    logger.info("Saving channel mapping files...")
    do_names = comp.line_order
    ao_names = comp.ao_order
    
    do_map = {"names": do_names, "phys": [hw.digital_outputs[n] for n in do_names]}
    do_map_file = run_dir / "do_map.json"
    do_map_file.write_text(json.dumps(do_map, indent=2))
    logger.info(f"  ✓ DO mapping: {do_map_file} ({len(do_names)} lines)")
    
    ao_map = {"names": ao_names, "phys": [hw.analog_outputs[n] for n in ao_names]}
    ao_map_file = run_dir / "ao_map.json"
    ao_map_file.write_text(json.dumps(ao_map, indent=2))
    logger.info(f"  ✓ AO mapping: {ao_map_file} ({len(ao_names)} channels)")
    
    # DI map (READY inputs) — write even if empty for consistency
    di_names_cfg = list(hw.digital_inputs.keys())
    di_map = {"names": di_names_cfg, "phys": [hw.digital_inputs[n] for n in di_names_cfg]}
    di_map_file = run_dir / "di_map.json"
    di_map_file.write_text(json.dumps(di_map, indent=2))
    logger.info(f"  ✓ DI mapping: {di_map_file} ({len(di_names_cfg)} lines)")

    # Save compiled arrays
    logger.info("Saving compiled waveform arrays...")
    do_file = run_dir / "compiled_do.npz"
    ao_file = run_dir / "compiled_ao.npz"
    
    logger.debug(f"  DO array shape: {comp.do.shape}, dtype: {comp.do.dtype}")
    np.savez_compressed(do_file, data=comp.do.astype(np.bool_))
    logger.info(f"  ✓ Digital outputs: {do_file}")
    
    logger.debug(f"  AO array shape: {comp.ao.shape}, dtype: {comp.ao.dtype}")
    np.savez_compressed(ao_file, data=comp.ao.astype(np.float32))
    logger.info(f"  ✓ Analog outputs: {ao_file}")

    # Digital edge log (super helpful to diff runs)
    logger.info("Computing and saving digital edge transitions...")
    edge_file = run_dir / "digital_edges.csv"
    write_edge_csv(edge_file, do_names, comp.do.astype(bool), comp.dt_ms)
    logger.info(f"  ✓ Digital edges: {edge_file}")
    
    # Count edges for summary
    do_bool = comp.do.astype(bool)
    total_edges = 0
    for i in range(len(do_names)):
        edges = np.sum(np.diff(do_bool[i, :].astype(int)) != 0)
        total_edges += edges
        logger.debug(f"    {do_names[i]}: {edges} transitions")
    logger.info(f"  Total edge transitions: {total_edges}")

    # Generate preview visualization
    logger.info("Generating preview visualization...")
    t_ms = np.arange(comp.N) * comp.dt_ms
    fig = make_protocol_figure(
        t_ms,
        comp.do.astype(bool),
        do_names,
        comp.ao,
        ao_names,
        title="Preview (no DAQ)",
        rck_log=comp.rck_log,
    )
    preview_file = run_dir / "preview.html"
    fig.write_html(preview_file, include_plotlyjs="cdn")
    logger.info(f"  ✓ Preview visualization: {preview_file}")

    if args.dry_run:
        logger.info("=== DRY RUN COMPLETE ===")
        logger.info(f"All artifacts saved to: {run_dir}")
        logger.info(f"Preview available at: {preview_file}")
        print(f"Dry-run complete. Preview: {run_dir/'preview.html'}")
        return

    # --- DAQ execution: DO master, AO slave, AI slave (MFC feedback), DI slave (READY)
    logger.info("=== Starting DAQ Hardware Execution ===")
    N = comp.N
    rate = comp.tcfg.sample_rate
    
    logger.info(f"DAQ Configuration:")
    logger.info(f"  Device: {hw.device}")
    logger.info(f"  Sample rate: {rate} Hz")
    logger.info(f"  Total samples: {N}")
    logger.info(f"  Estimated duration: {N/rate:.2f} seconds")

    # Prepare channel lists
    ai_names = list(hw.analog_inputs.keys())
    ai_phys = [hw.analog_inputs[n] for n in ai_names]
    di_names = list(hw.digital_inputs.keys())
    di_phys = [hw.digital_inputs[n] for n in di_names]
    
    logger.info(f"Channel Summary:")
    logger.info(f"  Digital outputs (DO): {len(do_names)} channels")
    for i, (name, phys) in enumerate(zip(do_names, [hw.digital_outputs[n] for n in do_names])):
        logger.debug(f"    DO[{i}]: {name} -> {phys}")
        
    logger.info(f"  Analog outputs (AO): {len(ao_names)} channels")
    for i, (name, phys) in enumerate(zip(ao_names, [hw.analog_outputs[n] for n in ao_names])):
        logger.debug(f"    AO[{i}]: {name} -> {phys}")
        
    logger.info(f"  Analog inputs (AI): {len(ai_names)} channels")
    for i, (name, phys) in enumerate(zip(ai_names, ai_phys)):
        logger.debug(f"    AI[{i}]: {name} -> {phys}")
        
    logger.info(f"  Digital inputs (DI): {len(di_names)} channels")
    for i, (name, phys) in enumerate(zip(di_names, di_phys)):
        logger.debug(f"    DI[{i}]: {name} -> {phys}")

    logger.info("Creating DAQ tasks...")
    with (
        nidaqmx.Task("DO_MASTER") as do_task,
        nidaqmx.Task("AO_SLAVE") as ao_task,
        nidaqmx.Task("AI_SLAVE") as ai_task,
        nidaqmx.Task("DI_READY") as di_task,
    ):
        logger.info("✓ DAQ tasks created successfully")

        # DO master lines
        logger.info("Configuring DO master task...")
        for i, ch in enumerate([hw.digital_outputs[n] for n in do_names]):
            logger.debug(f"  Adding DO channel {i}: {ch}")
            do_task.do_channels.add_do_chan(
                ch, line_grouping=LineGrouping.CHAN_PER_LINE
            )
        do_task.timing.cfg_samp_clk_timing(
            rate=rate,
            active_edge=Edge.RISING,
            sample_mode=AcquisitionType.FINITE,
            samps_per_chan=N,
        )
        logger.debug(f"  Writing DO data array shape: {comp.do.shape}")
        do_task.write(comp.do.astype(np.bool_))
        logger.info("✓ DO master task configured and data loaded")

        # AO slave
        if ao_names:
            logger.info("Configuring AO slave task...")
            ao_channel_str = ",".join([hw.analog_outputs[n] for n in ao_names])
            logger.debug(f"  AO channels: {ao_channel_str}")
            ao_task.ao_channels.add_ao_voltage_chan(
                ao_channel_str,
                min_val=0.0,
                max_val=5.0,
            )
            ao_task.timing.cfg_samp_clk_timing(
                rate=rate,
                source=f"/{hw.device}/do/SampleClock",
                active_edge=Edge.RISING,
                sample_mode=AcquisitionType.FINITE,
                samps_per_chan=N,
            )
            ao_task.triggers.start_trigger.cfg_dig_edge_start_trig(
                f"/{hw.device}/do/StartTrigger"
            )
            logger.debug(f"  Writing AO data array shape: {comp.ao.shape}")
            AnalogMultiChannelWriter(ao_task.out_stream).write_many_sample(
                comp.ao.astype(np.float64)
            )
            logger.info("✓ AO slave task configured and data loaded")
        else:
            logger.info("No AO channels configured, skipping AO task")

        # AI slave (MFC feedback)
        ai_buf = None
        if ai_phys:
            logger.info("Configuring AI slave task...")
            ai_channel_str = ",".join(ai_phys)
            logger.debug(f"  AI channels: {ai_channel_str}")
            ai_task.ai_channels.add_ai_voltage_chan(
                ai_channel_str, min_val=0.0, max_val=10.0
            )
            ai_task.timing.cfg_samp_clk_timing(
                rate=rate,
                source=f"/{hw.device}/do/SampleClock",
                active_edge=Edge.RISING,
                sample_mode=AcquisitionType.FINITE,
                samps_per_chan=N,
            )
            ai_task.triggers.start_trigger.cfg_dig_edge_start_trig(
                f"/{hw.device}/do/StartTrigger"
            )
            ai_reader = AnalogMultiChannelReader(ai_task.in_stream)
            ai_buf = np.zeros((len(ai_phys), N), dtype=np.float64)
            logger.debug(f"  AI buffer shape: {ai_buf.shape}")
            logger.info("✓ AI slave task configured and buffer allocated")
        else:
            logger.info("No AI channels configured, skipping AI task")

        # DI slave (READY inputs from Teensy)
        if di_phys:
            logger.info("Configuring DI slave task...")
            for i, ch in enumerate(di_phys):
                logger.debug(f"  Adding DI channel {i}: {ch}")
                di_task.di_channels.add_di_chan(
                    ch, line_grouping=LineGrouping.CHAN_PER_LINE
                )
            di_task.timing.cfg_samp_clk_timing(
                rate=rate,
                source=f"/{hw.device}/do/SampleClock",
                active_edge=Edge.RISING,
                sample_mode=AcquisitionType.FINITE,
                samps_per_chan=N,
            )
            di_task.triggers.start_trigger.cfg_dig_edge_start_trig(
                f"/{hw.device}/do/StartTrigger"
            )
            logger.info("✓ DI slave task configured successfully")
        else:
            logger.info("No DI channels configured, skipping DI task")

        # Start tasks in proper sequence
        logger.info("Starting DAQ tasks...")
        if ao_names:
            logger.debug("  Starting AO slave task...")
            ao_task.start()
        if ai_phys:
            logger.debug("  Starting AI slave task...")
            ai_task.start()
        if di_phys:
            logger.debug("  Starting DI slave task...")
            di_task.start()
            
        logger.debug("  Starting DO master task...")
        start_time = time.time()
        do_task.start()
        logger.info("✓ All DAQ tasks started, protocol execution in progress...")

        # Wait for completion with timeout
        timeout = max(10.0, N / rate + 5.0)
        logger.info(f"Waiting for protocol completion (timeout: {timeout:.1f}s)...")
        
        try:
            do_task.wait_until_done(timeout=timeout)
            execution_time = time.time() - start_time
            logger.info(f"✓ Protocol execution completed in {execution_time:.2f} seconds")
        except Exception as e:
            logger.error(f"Protocol execution failed: {e}")
            raise
            
        # Stop tasks and read data
        logger.info("Stopping tasks and reading data...")
        do_task.stop()
        logger.debug("  DO master task stopped")
        
        if ao_names:
            ao_task.stop()
            logger.debug("  AO slave task stopped")

        if ai_phys:
            logger.info("Reading AI data...")
            try:
                ai_reader.read_many_sample(
                    ai_buf,
                    number_of_samples_per_channel=N,
                    timeout=max(10.0, N / rate + 5.0),
                )
                ai_task.stop()
                
                # Save AI data and provide statistics
                ai_file = run_dir / "capture_ai.npz"
                np.savez_compressed(
                    ai_file,
                    names=np.array(ai_names, dtype=object),
                    data=ai_buf.astype(np.float32),
                )
                logger.info(f"✓ AI data saved: {ai_file}")
                logger.info(f"  AI data shape: {ai_buf.shape}")
                for i, name in enumerate(ai_names):
                    min_val, max_val = ai_buf[i].min(), ai_buf[i].max()
                    mean_val = ai_buf[i].mean()
                    logger.debug(f"    {name}: min={min_val:.3f}V, max={max_val:.3f}V, mean={mean_val:.3f}V")
            except Exception as e:
                logger.error(f"Failed to read AI data: {e}")
                raise

        if di_phys:
            logger.info("Reading DI data...")
            try:
                di_data = di_task.read(
                    number_of_samples_per_channel=N, 
                    timeout=max(10.0, N / rate + 5.0)
                )
                di_task.stop()
                
                # Save the returned DI data and provide statistics
                di_file = run_dir / "capture_di.npz"
                np.savez_compressed(
                    di_file,
                    names=np.array(di_names, dtype=object),
                    # The returned data is already boolean, but we cast to be safe
                    data=np.array(di_data).astype(np.bool_), 
                )
                logger.info(f"✓ DI data saved: {di_file}")
                
                # Use the new di_data variable for analysis
                di_bool = np.array(di_data).astype(bool)
                logger.info(f"  DI data shape: {di_bool.shape}")
                for i, name in enumerate(di_names):
                    high_count = np.sum(di_bool[i])
                    high_pct = high_count / N * 100
                    logger.debug(f"    {name}: {high_count}/{N} samples high ({high_pct:.1f}%)")
            except Exception as e:
                logger.error(f"Failed to read DI data: {e}")
                raise

    logger.info("✓ All DAQ tasks completed and data acquired")

    logger.info("✓ All DAQ tasks completed and data acquired")

    # Post-run interactive viz with AI/DI overlays (if recorded)
    logger.info("=== Generating Post-Run Visualization ===")
    
    di_names_overlay = di_data_overlay = None
    ai_names_overlay = ai_data_overlay = None
    
    # Load DI data if available
    di_file = run_dir / "capture_di.npz"
    if di_file.exists():
        logger.info("Loading DI data for visualization overlay...")
        npz_di = np.load(di_file, allow_pickle=True)
        di_names_overlay = list(npz_di["names"])
        di_data_overlay = npz_di["data"].astype(bool)
        logger.info(f"  ✓ DI overlay data loaded: {len(di_names_overlay)} channels")
        logger.debug(f"    DI overlay shape: {di_data_overlay.shape}")

    # Load AI data if available  
    ai_file = run_dir / "capture_ai.npz"
    if ai_file.exists():
        logger.info("Loading AI data for visualization overlay...")
        npz_ai = np.load(ai_file, allow_pickle=True)
        ai_names_overlay = list(npz_ai["names"])
        ai_data_overlay = npz_ai["data"]
        logger.info(f"  ✓ AI overlay data loaded: {len(ai_names_overlay)} channels")
        logger.debug(f"    AI overlay shape: {ai_data_overlay.shape}")

    # Generate comprehensive visualization
    logger.info("Generating comprehensive interactive figure...")
    fig = make_protocol_figure(
        t_ms,
        comp.do.astype(bool),
        do_names,
        comp.ao,
        ao_names,
        ai=ai_data_overlay,
        ai_names=ai_names_overlay,
        di=di_data_overlay,
        di_names=di_names_overlay,
        rck_log=comp.rck_log,
        title="Protocol (DO/AO) + READY (DI) + MFC Feedback (AI)",
    )
    
    final_preview = run_dir / "preview.html"
    logger.info("Writing final interactive HTML...")
    fig.write_html(final_preview, include_plotlyjs="cdn")
    logger.info(f"✓ Final visualization saved: {final_preview}")

    # Generate READY edge log if present
    if di_file.exists():
        logger.info("Computing READY line edge transitions...")
        ready_edge_file = run_dir / "ready_edges.csv"
        write_edge_csv(ready_edge_file, di_names_overlay, di_data_overlay, comp.dt_ms)
        
        # Count READY edges for summary
        ready_edges_total = 0
        for i in range(len(di_names_overlay)):
            edges = np.sum(np.diff(di_data_overlay[i, :].astype(int)) != 0)
            ready_edges_total += edges
            logger.debug(f"    {di_names_overlay[i]}: {edges} READY transitions")
        logger.info(f"✓ READY edges saved: {ready_edge_file} ({ready_edges_total} total transitions)")

    # Final summary
    logger.info("=== RUN COMPLETION SUMMARY ===")
    logger.info(f"✓ Run directory: {run_dir}")
    logger.info(f"✓ Protocol duration: {comp.N * comp.dt_ms:.1f} ms ({comp.N * comp.dt_ms / 1000:.2f} seconds)")
    logger.info(f"✓ Total samples: {comp.N:,}")
    logger.info(f"✓ Sample rate: {comp.tcfg.sample_rate} Hz")
    logger.info(f"✓ Digital outputs: {len(do_names)} channels")
    logger.info(f"✓ Analog outputs: {len(ao_names)} channels")
    if ai_names_overlay:
        logger.info(f"✓ Analog inputs captured: {len(ai_names_overlay)} channels")
    if di_names_overlay:
        logger.info(f"✓ Digital inputs captured: {len(di_names_overlay)} channels")
    logger.info(f"✓ Interactive preview: {final_preview}")
    
    # List all generated files
    logger.info("Generated artifacts:")
    for file_path in sorted(run_dir.glob("*")):
        size_bytes = file_path.stat().st_size
        size_str = f"{size_bytes:,} bytes"
        if size_bytes > 1024:
            size_str = f"{size_bytes/1024:.1f} KB"
        if size_bytes > 1024*1024:
            size_str = f"{size_bytes/(1024*1024):.1f} MB"
        logger.info(f"  {file_path.name}: {size_str}")

    print(f"Run complete. See interactive preview: {run_dir/'preview.html'}")


if __name__ == "__main__":
    main()
