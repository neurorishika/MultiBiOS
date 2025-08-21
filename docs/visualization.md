# Visualization (Interactive)

Each run writes `preview.html` with **Plotly**:

- **Row 1 (digital)**: **all** DO rails stacked: state lines (`*_S*`), `*_LOAD_REQ`, `RCK_*`, and triggers.
  - Hover shows `time (ms)` and **logic level**.
  - Pale **red vertical lines** mark each `RCK_*` commit.
- **Row 2 (AO)**: MFC **setpoints** (step traces).
- **Row 3 (AI)**: MFC **feedback** (if captured).

Use the legend to toggle visibility; drag to zoom; double-click to reset.
