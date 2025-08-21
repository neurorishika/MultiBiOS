# Logging & Artifacts

Each run creates a folder: `data/runs/YYYY-MM-DD_HH-MM-SS/` with:

- `meta.json`: device, sample rate, duration, **rng_seed**, CLI args.
- `protocol.yaml`, `hardware.yaml`: copies of inputs for provenance.
- `do_map.json`, `ao_map.json`: logical names → physical channels.
- `compiled_do.npz`: boolean array `[lines, samples]`.
- `compiled_ao.npz`: float array `[channels, samples]`.
- `capture_ai.npz`: (optional) float array `[channels, samples]` with MFC feedback.
- `rck_edges.csv`: planned commits (`signal, sample_idx, time_ms`).
- `digital_edges.csv`: **all** rising/falling edges for every DO line.
- `preview.html`: interactive Plotly visualization.

> You can parse `digital_edges.csv` to compute inter-event latencies and export tabular summaries alongside imaging data.
