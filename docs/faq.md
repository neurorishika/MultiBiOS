# Frequently Asked Questions

## System Architecture

??? question "Why separate preload and commit?"
    The separation of preload and commit operations allows the DAQ to timestamp the commit with hardware precision and coordinate multiple valve assemblies independently. This ensures sub-millisecond timing accuracy even under system load.

??? question "What does the guardrail prevent?"
    The compile-time guardrail prevents overlapping "preload→commit" windows where two assemblies would both be staged simultaneously. The Teensy firmware is designed to handle exactly one staged owner at a time for safety and reliability.

??? question "Why use sticky S-bits?"
    Sticky S-bits make digital rails represent the **current logical state** between events, which simplifies verification and downstream analysis. This provides a clear representation of what the system is actually doing at any given moment.

## Timing and Performance

??? question "Can I use 0.1 ms timing resolution?"
    Yes! Set `sample_rate: 10000` in your protocol configuration and ensure your USB-6363 and host computer can handle the higher sample rate. All system components (guardrails, viewer, logging) automatically scale with the sample rate.

??? question "What's the maximum sample rate supported?"
    The system supports up to 10 kHz (0.1 ms resolution) with the NI USB-6363. Higher rates may be possible but haven't been extensively tested. The limiting factors are typically USB bandwidth and host system performance.

??? question "How precise is the timing really?"
    Hardware-clocked events have microsecond precision determined by the NI-DAQ clock. Software events (like protocol phase transitions) have millisecond precision. The sticky S-bit approach ensures valve states are always synchronized to the hardware clock.

## Hardware and Setup

??? question "Can I use a different DAQ device?"
    The system is designed around the NI USB-6363's capabilities, but could potentially be adapted to other NI-DAQ devices with sufficient digital I/O and analog channels. This would require modifications to the hardware configuration and possibly the runner code.

??? question "How many valve assemblies can I control?"
    The current firmware supports 4 shift register chains (olfactometer_left, olfactometer_right, switchvalve_left, switchvalve_right), each controlling 8 valves. This could be extended by modifying the firmware and hardware configuration.

??? question "What if I need more trigger outputs?"
    The current system provides microscope triggers and continuous camera triggers. Additional trigger types can be added by modifying the hardware mapping and protocol schema. The NI-DAQ has additional digital output channels available.

## Protocol Development

??? question "How do I test protocols without hardware?"
    Use the `--dry-run` flag with the runner. This generates all the timing data and visualizations without attempting to communicate with actual hardware. Perfect for protocol development and debugging.

??? question "Can I have overlapping device actions?"
    Yes, but with constraints. Different device types (olfactometers, MFCs, triggers) can have overlapping actions. However, the same device cannot have overlapping state changes, and valve assemblies cannot have overlapping preload windows.

??? question "How does randomization work?"
    When `randomize: true` is set for a phase with multi-state device actions, the system shuffles the state list using the configured seed. This ensures reproducible "randomization" - the same seed always produces the same sequence.

## Troubleshooting

??? question "Why do I get timing guardrail errors?"
    Guardrail errors occur when valve switching events are scheduled too close together. Check that your `preload_lead_ms` and `setup_hold_samples` settings provide sufficient margins between events. The error message will indicate which specific timing constraint was violated.

??? question "My MFC readings seem noisy - is this normal?"
    Some noise in MFC analog readings is normal. The system logs raw ADC values for full provenance. Consider the noise level relative to your experimental requirements and use appropriate filtering in post-analysis if needed.

??? question "Can I modify the protocol during execution?"
    No, protocols are compiled before execution and cannot be modified while running. This is by design to ensure complete reproducibility and data provenance. Stop the current run and start a new one with the modified protocol.

## Data and Analysis

??? question "Where is my data saved?"
    All run data is saved in timestamped directories under `data/runs/`. Each run includes the original protocol, compiled timing data, hardware logs, and any analog recordings.

??? question "What format is the logged data?"
    Data is saved in standard formats: YAML for configuration, NumPy arrays for timing data, and CSV for analog readings. This ensures compatibility with common analysis tools and long-term data accessibility.

??? question "How do I analyze the timing precision?"
    Use the interactive visualizer to examine the relationship between commanded and actual timing. The system logs both intended timing (from protocol) and actual hardware timing (from DAQ) for validation.

## Contributing and Support

??? question "I found a bug - how do I report it?"
    Please open an issue on the [GitHub repository](https://github.com/neurorishika/MultiBiOS) with details about your system, the protocol you were running, and the specific error or unexpected behavior.

??? question "Can I contribute new features?"
    Absolutely! See the [Contributing Guide](contributing.md) for information about the development process, coding standards, and how to submit pull requests.

??? question "Is there a user community?"
    The primary community is through GitHub issues and discussions. For neuroscience-specific questions, consider posting in relevant research forums with a link to this documentation.

---

!!! tip "Still have questions?"
    If you don't see your question here, check the detailed documentation sections or [open an issue](https://github.com/neurorishika/MultiBiOS/issues/new) on GitHub. We're happy to help!hy separate preload and commit?**  
A: It lets the DAQ timestamp the commit with hardware precision and coordinate multiple assemblies independently.

**Q: What does the guardrail prevent?**  
A: Overlapping “preload→commit” windows (where two assemblies would both be staged). The Teensy firmware intentionally handles exactly one staged owner at a time.

**Q: Why sticky S-bits?**  
A: They make digital rails represent **current state** between events, simplifying verification and downstream analysis.

**Q: Can I use 0.1 ms timing?**  
A: Yes; set `sample_rate: 10000` and ensure your USB-6363 and host can stream the sample count. Everything (guardrails, viewer) scales.
