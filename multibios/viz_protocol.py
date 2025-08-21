#!/usr/bin/env python3
"""
Visualize a logged run: DO timing rails + AO setpoints + AI (MFC feedback).
"""

from __future__ import annotations

import sys
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt


def main():
    if len(sys.argv) < 2:
        print("usage: python -m multibios.viz_protocol <run_folder>")
        sys.exit(1)

    run = Path(sys.argv[1])
    if not run.exists():
        raise SystemExit(f"run folder not found: {run}")

    meta = json.loads((run / "meta.json").read_text())
    sr = meta["sample_rate"]
    dt_ms = 1000.0 / sr

    do_map = json.loads((run / "do_map.json").read_text())
    ao_map = json.loads((run / "ao_map.json").read_text())
    do = np.load(run / "compiled_do.npz")["data"]
    ao = np.load(run / "compiled_ao.npz")["data"]
    N = do.shape[1]
    t_ms = np.arange(N) * dt_ms

    # Optional AI capture
    ai_names = []
    ai = None
    ai_npz = run / "capture_ai.npz"
    if ai_npz.exists():
        npz = np.load(ai_npz, allow_pickle=True)
        ai_names = list(npz["names"])
        ai = np.array(npz["data"])

    fig, axes = plt.subplots(
        3, 1, figsize=(14, 10), sharex=True, gridspec_kw={"height_ratios": [2, 1, 1]}
    )
    ax_do, ax_ao, ax_ai = axes

    # DO: show RCK/LOAD/trigger lines
    do_keys = [
        k for k in do_map["names"] if any(s in k for s in ("RCK_", "LOAD_REQ", "TRIG_"))
    ]
    y0 = 0
    for k in do_keys:
        tr = do[do_map["names"].index(k)]
        ax_do.step(t_ms, tr.astype(int) + y0, where="post", label=f"DO:{k}")
        y0 += 1.2
    ax_do.set_ylabel("Digital (stacked)")
    if do_keys:
        ax_do.legend(ncol=min(4, len(do_keys)), fontsize=8)

    # AO setpoints
    for i, k in enumerate(ao_map["names"]):
        ax_ao.step(t_ms, ao[i], where="post", label=f"AO:{k}")
    ax_ao.set_ylabel("AO (V)")
    if ao.shape[0]:
        ax_ao.legend(ncol=min(4, ao.shape[0]), fontsize=8)

    # AI feedback overlay
    if ai is not None:
        for i, k in enumerate(ai_names):
            ax_ai.plot(t_ms, ai[i], "-", lw=0.9, label=f"AI:{k}")
        ax_ai.legend(ncol=min(4, len(ai_names)), fontsize=8)
    ax_ai.set_ylabel("AI (V)")
    ax_ai.set_xlabel("Time (ms)")

    fig.suptitle(f"Run: {run.name}")
    fig.tight_layout()
    fig.savefig(run / "preview.png", dpi=160)
    plt.show()


if __name__ == "__main__":
    main()
