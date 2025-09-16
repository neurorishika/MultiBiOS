#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

from multibios.viz_helpers import load_run_artifacts, make_protocol_figure


def main():
    if len(sys.argv) < 2:
        print("usage: python -m multibios.viz_protocol <run_folder>")
        sys.exit(1)
    run = Path(sys.argv[1])
    if not run.exists():
        raise SystemExit(f"run folder not found: {run}")

    art = load_run_artifacts(run)
    fig = make_protocol_figure(
        art["t_ms"],
        art["do"],
        art["do_names"],
        art["ao"],
        art["ao_names"],
        ai=art["ai"],
        ai_names=art["ai_names"],
        di=art["di"],
        di_names=art["di_names"],
        rck_log=art["rck_log"],
        title=f"Run: {run.name}",
    )
    html_path = run / "preview.html"
    fig.write_html(html_path, include_plotlyjs="cdn")
    print(f"Wrote {html_path.resolve()}")


if __name__ == "__main__":
    main()
