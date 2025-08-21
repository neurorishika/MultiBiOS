# MultiBiOS Odor Delivery System

<div align="center">
  <strong>🧪 Precision Olfactometer Control for Behavioral Neuroscience 🧪</strong>
</div>

---

## Overview

**MultiBiOS** (Multispecies Bilateral Odor delivery System) is a high-precision, hardware-clocked olfactometer control system designed for behavioral neuroscience experiments. It provides sub-millisecond timing control, full signal provenance, and reproducible experimental protocols.

!!! tip "Key Features"
    - **⚡ Sub-millisecond precision** with hardware-clocked timing
    - **🔄 Bilateral control** for complex comparative experiments  
    - **📊 Full signal provenance** with comprehensive logging
    - **🛡️ Built-in guardrails** prevent timing conflicts
    - **🎲 Reproducible randomization** via configurable seeds
    - **📈 Interactive visualization** with Plotly integration

## System Architecture

The MultiBiOS system consists of three main components working in perfect synchronization:

=== "Hardware Layer"
    - **Teensy 4.1** microcontroller for valve preloading
    - **TPIC6B595** shift register chains (10MHz SPI)
    - **NI USB-6363** for hardware-clocked timing
    - **Mass flow controllers** for precise flow control

=== "Firmware Layer"  
    - **Interrupt-driven architecture** on Teensy 4.1
    - **Pattern preloading** with sticky S-bit rails
    - **Hardware synchronization** via RCK signals
    - **Safety interlocks** and state validation

=== "Software Layer"
    - **YAML protocol compiler** with timing validation
    - **Hardware-clocked execution** via NI-DAQmx
    - **Real-time logging** of all signals
    - **Interactive analysis** and visualization

## Quick Start

Get up and running with MultiBiOS in minutes:r Delivery System

**Multispecies Bilateral Odor Delivery** with sub-millisecond control and full signal provenance.

This repository provides:

- **Teensy 4.1 firmware** that preloads valve patterns and commits on **NI-DAQ register clocks** (RCK).
- A **YAML protocol** that describes stimuli, triggers, and MFC setpoints with ms-level timing.
- A **hardware-clocked NI USB-6363 runner** that generates DO/AO, and logs **MFC analog feedback** (AI).
- **Guardrails** that reject overlapping preload→commit windows at compile-time.
- **Sticky S-bit rails** so digital state lines reflect the system’s current logical state between events.
- **Interactive Plotly viewer** that overlays compiled DO/AO with captured AI.
- Reproducible **randomization** via a configurable `seed`.

## Quickstart

### 1. Preview Mode (No Hardware Required)

```bash
# Generate and visualize a protocol without hardware
poetry run python -m multibios.run_protocol \
    --yaml config/example_protocol.yaml \
    --hardware config/hardware.yaml \
    --dry-run --interactive --seed 42
```

### 2. Hardware Execution

```bash
# Run on actual hardware (requires NI-DAQmx drivers)
poetry run python -m multibios.run_protocol \
    --yaml config/example_protocol.yaml \
    --hardware config/hardware.yaml
```

### 3. Post-Run Analysis

```bash
# Visualize any completed experiment
poetry run python -m multibios.viz_protocol data/runs/2025-08-21_14-07-33
```

## What Makes MultiBiOS Special?

### 🎯 Hardware-Clocked Precision

Unlike software-based timing systems, MultiBiOS uses dedicated hardware clocks for microsecond-precise valve control, ensuring experimental reproducibility even under system load.

### 🔒 Compile-Time Safety

The protocol compiler validates all timing constraints before execution, preventing hardware conflicts and ensuring reliable operation.

### 📊 Complete Data Provenance

Every signal is logged with timestamps, providing a complete record of experimental conditions for analysis and publication.

### 🎲 Reproducible Randomization

Configurable random seeds ensure that "randomized" protocols can be exactly reproduced for validation and replication studies.

## Example Protocol

Here's a simple bilateral odor presentation protocol:

```yaml
protocol:
  name: "Bilateral Odor Test"
  timing:
    sample_rate: 1000
    seed: 42

sequence:
  - phase: "Baseline"
    duration: 30000  # 30 seconds
    times: 1
    actions:
      - device: "olfactometer.left"
        state: "AIR"
        timing: 0
      - device: "olfactometer.right"  
        state: "AIR"
        timing: 0

  - phase: "Odor Presentation"
    duration: 60000  # 1 minute
    times: 5
    randomize: true
    actions:
      - device: "olfactometer.left"
        state: "ODOR1,ODOR2,ODOR3"  # Randomized selection
        timing: 0
      - device: "triggers.microscope"
        state: true
        timing: 15000  # Trigger 15s into each trial
```

!!! success "Ready to Get Started?"
    Check out the **[Hardware Setup](hardware.md)** guide to begin building your MultiBiOS system, or explore the **[Protocol Specification](protocol.md)** to understand the YAML format.

See **Hardware** for wiring and safety, **Protocol** for YAML schema, and **Runner** for execution and logging.
