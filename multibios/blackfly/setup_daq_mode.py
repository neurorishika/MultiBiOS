#!/usr/bin/env python3
"""Configure both Blackfly cameras for NI-DAQ acquisition.

Three modes are available:

  FrameStart (default, --mode frame):
    Each DAQ trigger pulse captures one frame.
        On overlap-capable Blackfly S cameras this supports substantially higher
        trigger rates than the older Flea3 path. Use tests/verify_camera_trigger_path.py
        to measure the accepted rate on the current hardware.

  Free-run (--mode freerun):
    Cameras free-run at --fps with internal overlap → 60 fps at full res.
    ExposureActive on Line2 lets the DAQ timestamp every frame.
    No per-frame trigger needed.

Examples:
    python -m multibios.blackfly.setup_daq_mode
    python -m multibios.blackfly.setup_daq_mode --binning 2
    python -m multibios.blackfly.setup_daq_mode --mode freerun
    python -m multibios.blackfly.setup_daq_mode --mode freerun --fps 30
"""

from __future__ import annotations

import argparse

from .live_view import (configure_camera_daq_freerun_mode,
                        configure_camera_daq_mode, connect_cameras,
                        load_blackfly_defaults,
                        release_cameras)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Configure Blackfly cameras for NI-DAQ acquisition.")
    parser.add_argument("--mode", choices=["frame", "freerun"], default="frame",
                        help="'frame' = per-frame FrameStart trigger. "
                             "'freerun' = internal free-run (60 Hz capable).")
    parser.add_argument("--fps", type=float, default=60.0,
                        help="Target fps for --mode freerun (default: 60).")
    parser.add_argument("--hardware", default="config/hardware.yaml",
                        help="Path to hardware.yaml for rig-level Blackfly defaults.")
    parser.add_argument("--exposure", type=float, default=None,
                        help="Fixed exposure time in µs (default: 5000). "
                             "Clamped to camera's valid range.")
    parser.add_argument("--width", type=int, default=None,
                        help="Centered ROI width in pixels (--mode frame only).")
    parser.add_argument("--height", type=int, default=None,
                        help="Vertical ROI in pixels (--mode frame only).")
    parser.add_argument("--binning", type=int, default=1, choices=[1, 2],
                        help="Pixel binning: 1=none (default), 2=2x2. "
                             "Reduces resolution but may increase max trigger rate.")
    args = parser.parse_args()

    defaults = load_blackfly_defaults(args.hardware)
    default_exposure = defaults.get("exposure_us")
    default_roi_width = defaults.get("roi_width")
    default_roi_height = defaults.get("roi_height")

    if args.exposure is None and default_exposure is not None:
        args.exposure = float(default_exposure)
    if args.width is None and default_roi_width is not None:
        args.width = int(default_roi_width)
    if args.height is None and default_roi_height is not None:
        args.height = int(default_roi_height)

    system, cam_list, cams = connect_cameras()
    try:
        failures: list[tuple[int, str]] = []
        if args.mode == "freerun":
            print(f"\nConfiguring free-run mode ({args.fps} fps) ...")
            for idx in range(len(cams)):
                print(f"Camera {idx}:")
                try:
                    configure_camera_daq_freerun_mode(
                        cams[idx], fps=args.fps, exposure_us=args.exposure)
                except Exception as exc:
                    failures.append((idx, str(exc)))
                    print(f"  [warn] Camera {idx} configuration failed: {exc}")
            print("\nFree-run mode configured.")
            print("Cameras will start immediately when SpinView arms acquisition.")
            print("DAQ can timestamp frames by reading ExposureActive on Line2.")
        else:
            print(f"\nConfiguring DAQ per-frame triggered mode"
                  f"{' (binning ' + str(args.binning) + 'x)' if args.binning > 1 else ''} ...")
            for idx in range(len(cams)):
                print(f"Camera {idx}:")
                try:
                    configure_camera_daq_mode(cams[idx],
                                              exposure_us=args.exposure,
                                              roi_width=args.width,
                                              roi_height=args.height,
                                              binning=args.binning)
                except Exception as exc:
                    failures.append((idx, str(exc)))
                    print(f"  [warn] Camera {idx} configuration failed: {exc}")
            print("\nDAQ per-frame mode configured.")
            print("You can now arm SpinView or run tests/verify_camera_trigger_path.py --arm-cameras to measure accepted trigger rate.")

        if failures:
            print("\nConfiguration completed with warnings:")
            for idx, message in failures:
                print(f"  Camera {idx}: {message}")
            raise RuntimeError(
                "Failed to configure all Blackfly cameras for DAQ acquisition. "
                "Aborting because every configured camera is required for this workflow."
            )
    finally:
        release_cameras(system, cam_list, cams, restore_daq=False)


if __name__ == "__main__":
    main()