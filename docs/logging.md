# Logging & Artifacts

Each run creates a folder: `data/runs/YYYY-MM-DD_HH-MM-SS/` with:

- `meta.json`: device, sample rate, duration, **rng_seed**, CLI args.
- `protocol.yaml`, `hardware.yaml`: copies of inputs for provenance.
- `do_map.json`, `ao_map.json`: logical names → physical channels.
- `di_map.json`: digital input channel map for READY and camera return lines.
- `compiled_do.npz`: boolean array `[lines, samples]`.
- `compiled_ao.npz`: float array `[channels, samples]`.
- `capture_ai.npz`: (optional) float array `[channels, samples]` with MFC feedback.
- `capture_di.npz`: (optional) boolean array `[lines, samples]` with recorded digital inputs.
- `control_plan.csv`: shared compiled logical schedule written by both `run_protocol` and `experiment`.
- `rck_edges.csv`: planned commits (`signal, sample_idx, time_ms`).
- `digital_edges.csv`: **all** rising/falling edges for every DO line.
- `preview.html`: interactive Plotly visualization.

If `hardware.yaml` enables `teensy.capture_serial: true`, the same folder also includes:

- `teensy_serial_transcript.jsonl`: line-oriented Teensy USB serial transcript captured alongside the run.

> You can parse `digital_edges.csv` to compute inter-event latencies and export tabular summaries alongside imaging data.
