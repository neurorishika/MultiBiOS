# MultiBiOS: Precision Olfactometer Control System

<div align="center">

![MultiBiOS Logo](https://img.shields.io/badge/MultiBiOS-Neuroscience%20Research-teal?style=for-the-badge&logo=microscope)

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Documentation](https://img.shields.io/badge/docs-mkdocs-blue)](https://neurorishika.github.io/MultiBiOS/)
[![GitHub issues](https://img.shields.io/github/issues/neurorishika/MultiBiOS)](https://github.com/neurorishika/MultiBiOS/issues)
[![GitHub stars](https://img.shields.io/github/stars/neurorishika/MultiBiOS)](https://github.com/neurorishika/MultiBiOS/stargazers)

**High-precision, hardware-clocked olfactometer control system for behavioral neuroscience**

[📚 Documentation](https://neurorishika.github.io/MultiBiOS/) • [🚀 Quick Start](#quick-start) • [🔧 Installation](#installation) • [📖 Examples](#examples) • [🤝 Contributing](#contributing)

</div>

---

## 🧪 Overview

**MultiBiOS** (Multispecies Bilateral Odor delivery System) is a precision olfactometer control system designed for behavioral neuroscience experiments. It provides sub-millisecond timing control, complete experimental reproducibility, and comprehensive data logging.

### ✨ Key Features

- 🎯 **Sub-millisecond precision** with hardware-clocked timing via NI-DAQ
- 🔄 **Bilateral valve control** for complex comparative experiments  
- 📊 **Complete data provenance** with comprehensive logging and replay
- 🛡️ **Built-in safety guardrails** prevent timing conflicts at compile-time
- 🎲 **Reproducible randomization** via configurable random seeds
- 📈 **Interactive visualization** with real-time and post-hoc analysis
- 🔌 **Teensy 4.1 firmware** for microsecond-precise valve preloading
- 📝 **YAML protocols** for human-readable experimental descriptions

### 🏗️ System Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   YAML Protocol │───▶│ Python Compiler │───▶│   NI-DAQ USB    │
│   Description   │    │   & Runner      │    │    6363         │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                        │
                                                        ▼
                              ┌─────────────────────────────────────┐
                              │          Teensy 4.1 MCU             │
                              │    (Valve Pattern Preloading)      │
                              └─────────────────────────────────────┘
                                                │
                                                ▼
                              ┌─────────────────────────────────────┐
                              │       TPIC6B595 Shift Registers    │
                              │         (10MHz SPI Chain)          │
                              └─────────────────────────────────────┘
                                                │
                                                ▼
                              ┌─────────────────────────────────────┐
                              │      Olfactometer Valve Arrays     │
                              │    (8-valve bilateral control)     │
                              └─────────────────────────────────────┘
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- NI-DAQmx drivers (for hardware execution)
- Poetry package manager

### Installation

```bash
# Clone the repository
git clone https://github.com/neurorishika/MultiBiOS.git
cd MultiBiOS

# Install dependencies using Poetry
poetry install

# Activate the virtual environment
poetry shell
```

### Your First Protocol

1. **Preview a protocol** (no hardware required):
   ```bash
   poetry run python -m multibios.run_protocol \
       --yaml config/example_protocol.yaml \
       --hardware config/hardware.yaml \
       --dry-run --interactive
   ```

2. **Run on hardware** (requires NI-DAQ setup):
   ```bash
   poetry run python -m multibios.run_protocol \
       --yaml config/example_protocol.yaml \
       --hardware config/hardware.yaml
   ```

3. **Analyze results**:
   ```bash
   poetry run python -m multibios.viz_protocol data/runs/latest
   ```

## 📖 Examples

### Simple Bilateral Odor Delivery

```yaml
protocol:
  name: "Bilateral Odor Comparison"
  timing:
    sample_rate: 1000  # 1 kHz sampling
    seed: 42          # Reproducible randomization

sequence:
  - phase: "Baseline"
    duration: 30000   # 30 seconds
    times: 1
    actions:
      - device: "olfactometer.left"
        state: "AIR"
        timing: 0
      - device: "olfactometer.right"
        state: "AIR"
        timing: 0

  - phase: "Odor Presentation" 
    duration: 60000   # 1 minute per trial
    times: 5          # 5 trials
    randomize: true   # Randomize odor order
    actions:
      - device: "olfactometer.left"
        state: "ODOR1,ODOR2,ODOR3"  # Random selection
        timing: 10000               # 10s into trial
      - device: "triggers.microscope"
        state: true
        timing: 15000               # Trigger at 15s
```

### Advanced Multi-Device Coordination

```yaml
sequence:
  - phase: "Complex Trial"
    duration: 45000
    times: 10
    actions:
      # MFC setpoints
      - device: "mfc.air_left_setpoint"
        value: 2.1  # Volts
        timing: 0
      
      # Synchronized valve switching  
      - device: "olfactometer.left"
        state: "ODOR1"
        timing: 5000
      - device: "switch_valve.left"
        state: "ODOR" 
        timing: 5100  # 100ms later
        
      # Continuous camera triggers
      - device: "triggers.camera_continuous"
        state: true
        timing: 1000
      - device: "triggers.camera_continuous" 
        state: false
        timing: 40000
```

## 🔧 Hardware Setup

### Required Components

| Component | Description | Quantity |
|-----------|-------------|----------|
| **NI USB-6363** | Hardware-clocked DAQ | 1 |
| **Teensy 4.1** | Microcontroller for valve control | 1 |
| **TPIC6B595** | High-power shift registers | 4 |
| **Olfactometer valves** | Pneumatic valves (12V) | 8-16 |
| **Mass flow controllers** | Precision flow control | 2-4 |

### Wiring Overview

```
NI-DAQ ────┐
           ├─── LOAD_REQ (Hardware clock)
           ├─── S-bits (Device selection) 
           ├─── Analog I/O (MFC control)
           └─── Triggers (Microscope/Camera)

Teensy ────┐
           ├─── SPI Chain (10MHz)
           ├─── Interrupt handling
           └─── Safety interlocks

TPIC6B595 ─┼─── Valve Array Left
           ├─── Valve Array Right  
           ├─── Switch Valve Left
           └─── Switch Valve Right
```

📋 **[Complete wiring diagrams and setup instructions →](https://neurorishika.github.io/MultiBiOS/hardware/)**

## 📊 Data Output

MultiBiOS provides comprehensive data logging:

### Generated Files
```
data/runs/2025-08-21_14-07-33/
├── protocol_original.yaml      # Original protocol
├── protocol_compiled.yaml      # Compiled with timing  
├── hardware_config.yaml        # Hardware configuration
├── digital_output.npy          # DO timing arrays
├── analog_output.npy           # AO setpoints
├── analog_input.csv            # MFC feedback (if recorded)
├── timing_log.csv              # Execution timestamps
└── visualization.html          # Interactive plot
```

### Interactive Visualization

The system generates rich interactive plots showing:
- 📈 Commanded vs actual valve states
- 🎛️ Analog input/output traces  
- ⏱️ Timing precision analysis
- 🔍 Zoom and pan capabilities

## 🧪 Use Cases

MultiBiOS is designed for:

- **Behavioral choice experiments** with precise odor timing
- **Optogenetics** with synchronized light/odor delivery
- **Calcium imaging** with triggered acquisition
- **Electrophysiology** with sub-millisecond precision
- **Multi-animal** comparative studies
- **Reproducible protocols** across labs and sessions

## 🛡️ Safety & Reliability

### Built-in Safeguards
- ✅ **Compile-time validation** prevents hardware conflicts
- ✅ **Timing guardrails** ensure safe valve switching
- ✅ **State verification** with sticky S-bit monitoring  
- ✅ **Hardware interlocks** prevent damage
- ✅ **Complete logging** for audit trails

### Experimental Reproducibility
- 🎯 **Deterministic randomization** with configurable seeds
- 📝 **Complete parameter logging** 
- 🔄 **Protocol replay** capability
- 📊 **Timing validation** and verification
- 🏷️ **Version tracking** of all components

## 📚 Documentation

Comprehensive documentation is available at **[neurorishika.github.io/MultiBiOS](https://neurorishika.github.io/MultiBiOS/)**

### Quick Links
- 🏗️ [System Architecture](https://neurorishika.github.io/MultiBiOS/hardware/)
- 🔌 [Hardware Setup](https://neurorishika.github.io/MultiBiOS/wiring/) 
- 💾 [Firmware Guide](https://neurorishika.github.io/MultiBiOS/firmware/)
- 📝 [Protocol Reference](https://neurorishika.github.io/MultiBiOS/protocol/)
- 🏃 [Runner Application](https://neurorishika.github.io/MultiBiOS/runner/)
- 📈 [Visualization](https://neurorishika.github.io/MultiBiOS/visualization/)
- ❓ [FAQ](https://neurorishika.github.io/MultiBiOS/faq/)
- 🔧 [Troubleshooting](https://neurorishika.github.io/MultiBiOS/troubleshooting/)

## 🤝 Contributing

We welcome contributions! MultiBiOS is built for the neuroscience community.

### Ways to Contribute
- 🐛 **Report bugs** and request features
- 💡 **Submit improvements** to code or documentation  
- 🧪 **Share protocols** and use cases
- 🔧 **Hardware modifications** and extensions
- 📚 **Documentation** improvements

### Getting Started
```bash
# Fork the repository on GitHub
git clone https://github.com/yourusername/MultiBiOS.git
cd MultiBiOS

# Install development dependencies
poetry install --with dev

# Run tests
poetry run pytest

# Format code
poetry run black multibios/
poetry run ruff check multibios/
```

📖 **[Full contributing guide →](https://neurorishika.github.io/MultiBiOS/contributing/)**

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

MultiBiOS was developed for the neuroscience research community. Special thanks to:
- The **Ruta and Kronauer Labs** for requirements and testing
- **National Instruments** for DAQ hardware support  
- The **Arduino/Teensy** community for firmware foundations
- **Open source contributors** who make science better

## 📞 Contact

- **Issues & Support:** [GitHub Issues](https://github.com/neurorishika/MultiBiOS/issues)
- **Email:** neurorishika@gmail.com  
- **Documentation:** [neurorishika.github.io/MultiBiOS](https://neurorishika.github.io/MultiBiOS/)

---

<div align="center">

**🧠 Built for Neuroscience • 🔬 Made with Science • 🚀 Open Source**

[⭐ Star this repo](https://github.com/neurorishika/MultiBiOS/stargazers) • [📖 Read the docs](https://neurorishika.github.io/MultiBiOS/) • [🤝 Contribute](CONTRIBUTING.md)

</div>
