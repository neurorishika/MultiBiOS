"""Interactive dual-Blackfly live viewer and shared camera helper utilities.

Simultaneously acquire frames from two Point Grey / FLIR Blackfly cameras and
display them side-by-side in a live OpenCV window.

Uses FLIR Spinnaker SDK 4.x + PySpin (Python 3.10).
Run from the dedicated conda environment:
    conda activate multibios-blackfly
    python -m multibios.blackfly.live_view

Mode
----
This viewer now uses software free-run capture only.

That keeps the live alignment/debugging path simple, and avoids mixing the
older camera-to-camera master/slave GPIO mode with the current NI-DAQ-master
timing architecture.

On exit, the script automatically restores both cameras to the DAQ-triggered
configuration so they are ready for external triggering again.

Controls
--------
    Q / ESC   - quit
    S         - save current frame (with overlay + current view)
    H         - toggle alignment crosshair on/off
    O         - toggle FicTrac ROI + exclusion zones
    + / -     - zoom in / out  (mouse scroll also works)
    [ / ]     - rotate ±2°
    Arrows    - pan
    R         - reset zoom / pan / rotation
    1 / 2     - make Camera 0 or 1 the active transform target
    0         - apply transform controls to both cameras
    TAB       - cycle layout: side-by-side → cam 0 focus → cam 1 focus
"""

import ctypes
import gc
import queue
import re
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np

try:
    import PySpin
except ImportError:
    sys.exit(
        "\nPySpin not found.\n"
        "Activate the correct environment first:\n"
        "  conda activate multibios-blackfly\n"
        "  pip install <path>\\assets\\spinnaker_python-4.3.0.189-cp310-cp310-win_amd64\\spinnaker_python-4.3.0.189-cp310-cp310-win_amd64.whl\n"
    )

# ──────────────────────────────────────────────────────────────────────────────
# USER CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────
DISPLAY_SCALE = 0.0          # 0.0 = auto-fit to screen; any other value overrides
TARGET_FPS    = 30.0         # desired frame rate
GRAB_TIMEOUT  = 1000         # ms to wait for each frame before giving up
SAVE_DIR      = Path("captured_frames")
WINDOW_MARGIN = 0.92         # fraction of screen to use (leaves room for taskbar)
DAQ_EXPOSURE_US = 5000.0     # fixed trigger-mode exposure to avoid slow auto-exposure
DAQ_ROI_HEIGHT  = 0          # 0 = full sensor; set e.g. 776 to halve readout time for faster triggers

# DAQ-triggered GPIO lines — check your camera's pin-out in SpinView
DAQ_OUTPUT_LINE  = "Line2"   # camera output line used for ExposureActive
DAQ_TRIGGER_LINE = "Line0"   # camera trigger input line from NI-DAQ

# ──────────────────────────────────────────────────────────────────────────────
# Spinnaker helpers  (Spinnaker 4.x / PySpin node-access pattern)
# ──────────────────────────────────────────────────────────────────────────────

def _enum_set(nodemap, node_name: str, entry_name: str) -> bool:
    """Set an enumeration node to a named entry. Returns True on success."""
    node = PySpin.CEnumerationPtr(nodemap.GetNode(node_name))
    if not PySpin.IsReadable(node) or not PySpin.IsWritable(node):
        print(f"  [warn] Enum node '{node_name}' not readable/writable.")
        return False
    entry = node.GetEntryByName(entry_name)
    if not PySpin.IsReadable(entry):
        print(f"  [warn] Entry '{entry_name}' not readable for '{node_name}'.")
        return False
    node.SetIntValue(entry.GetValue())
    return True


def _float_set(nodemap, node_name: str, value: float) -> bool:
    node = PySpin.CFloatPtr(nodemap.GetNode(node_name))
    if not PySpin.IsReadable(node) or not PySpin.IsWritable(node):
        print(f"  [warn] Float node '{node_name}' not readable/writable.")
        return False
    # Clamp to the node's valid range
    lo, hi = node.GetMin(), node.GetMax()
    clamped = max(lo, min(hi, value))
    if abs(clamped - value) > 0.01:
        print(f"  [warn] {node_name}: requested {value:.1f}, clamped to {clamped:.1f}"
              f"  (range {lo:.1f}–{hi:.1f})")
    node.SetValue(clamped)
    return True


def _bool_set(nodemap, node_name: str, value: bool, silent: bool = False) -> bool:
    node = PySpin.CBooleanPtr(nodemap.GetNode(node_name))
    if not PySpin.IsReadable(node) or not PySpin.IsWritable(node):
        if not silent:
            print(f"  [warn] Bool node '{node_name}' not readable/writable.")
        return False
    node.SetValue(value)
    return True


def _set_buffer_newest_only(cam) -> None:
    """Keep only the newest frame in the buffer — avoids stale/queued frames."""
    sn = cam.GetTLStreamNodeMap()
    _enum_set(sn, "StreamBufferHandlingMode", "NewestOnly")


def _get_screen_size() -> tuple:
    """Return (width, height) of the primary monitor in pixels."""
    try:
        # Use the Win32 API so DPI scaling is handled correctly
        ctypes.windll.shcore.SetProcessDpiAwareness(2)  # per-monitor DPI aware
    except Exception:
        pass
    user32 = ctypes.windll.user32
    return user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)


def _int_node_value(nodemap, node_name: str) -> int:
    """Read an integer node; return 0 if unavailable."""
    node = PySpin.CIntegerPtr(nodemap.GetNode(node_name))
    return int(node.GetValue()) if PySpin.IsReadable(node) else 0


def _int_node_max(nodemap, node_name: str) -> int:
    """Read the maximum of an integer node; return 0 if unavailable."""
    node = PySpin.CIntegerPtr(nodemap.GetNode(node_name))
    return int(node.GetMax()) if PySpin.IsReadable(node) else 0


def _int_node_set(nodemap, node_name: str, value: int) -> bool:
    node = PySpin.CIntegerPtr(nodemap.GetNode(node_name))
    if not PySpin.IsReadable(node) or not PySpin.IsWritable(node):
        return False
    # Clamp to valid range and align to increment
    inc = int(node.GetInc()) or 1
    value = max(int(node.GetMin()), min(int(node.GetMax()), value))
    value = (value // inc) * inc
    node.SetValue(value)
    return True


def _set_full_fov(cam) -> tuple:
    """
    Reset any ROI to the camera's full sensor area.
    Returns (width, height) actually set.
    """
    nm = cam.GetNodeMap()
    # Offsets must be zeroed before touching Width/Height
    _int_node_set(nm, "OffsetX", 0)
    _int_node_set(nm, "OffsetY", 0)
    w_max = _int_node_max(nm, "Width")
    h_max = _int_node_max(nm, "Height")
    if w_max and h_max:
        _int_node_set(nm, "Width",  w_max)
        _int_node_set(nm, "Height", h_max)
        print(f"  Full FOV: {w_max} x {h_max} px")
        return w_max, h_max
    # Already at max (or nodes not writable — still read current values)
    w = _int_node_value(nm, "Width")  or w_max
    h = _int_node_value(nm, "Height") or h_max
    print(f"  FOV (read-only): {w} x {h} px")
    return w, h


def _set_roi_height(cam, height: int) -> tuple:
    """
    Set a centred vertical ROI to reduce readout time.
    Full width is kept; height is clamped to sensor limits and alignment.
    Returns (width, height) actually set.
    """
    nm = cam.GetNodeMap()
    # Start from full FOV so max ranges are available
    _int_node_set(nm, "OffsetX", 0)
    _int_node_set(nm, "OffsetY", 0)
    w_max = _int_node_max(nm, "Width")
    h_max = _int_node_max(nm, "Height")
    if not w_max or not h_max:
        print("  [warn] Cannot read sensor max dims, using full FOV.")
        return _set_full_fov(cam)
    _int_node_set(nm, "Width", w_max)
    # Clamp and align height
    h_node = PySpin.CIntegerPtr(nm.GetNode("Height"))
    if not PySpin.IsWritable(h_node):
        print("  [warn] Height not writable, using full FOV.")
        return _set_full_fov(cam)
    inc = int(h_node.GetInc()) or 1
    h_min = int(h_node.GetMin())
    height = max(h_min, min(h_max, height))
    height = (height // inc) * inc
    # Centre the ROI vertically
    offset_y = ((h_max - height) // 2 // inc) * inc
    _int_node_set(nm, "Height", height)
    _int_node_set(nm, "OffsetY", offset_y)
    print(f"  ROI: {w_max} x {height} px  (offset_y={offset_y}, full={h_max})")
    return w_max, height


# ──────────────────────────────────────────────────────────────────────────────
# FicTrac ROI overlay helpers
# ──────────────────────────────────────────────────────────────────────────────

def load_fictrac_roi(config_path: "Path | None") -> "dict | None":
    """
    Parse a FicTrac config.txt and extract the ball ROI.
    Returns dict{'roi_circ': Nx2 float ndarray, 'cam_idx': int or None}
    or None if the file is missing or contains no roi_circ.
    """
    if config_path is None:
        return None
    p = Path(config_path)
    if not p.exists():
        print(f"  [warn] FicTrac config not found: {p}")
        return None
    result: dict = {'roi_circ': None, 'cam_idx': None}
    with open(p) as fh:
        for line in fh:
            line = line.strip()
            if line.startswith('#') or ':' not in line:
                continue
            key, _, val = line.partition(':')
            key = key.strip()
            val = val.strip()
            if key == 'roi_circ':
                nums = [float(n) for n in re.findall(r'[-+]?\d*\.?\d+', val)]
                n = (len(nums) // 2) * 2   # ensure even count
                if n >= 6:                  # at least 3 (x,y) pairs
                    result['roi_circ'] = np.array(nums[:n]).reshape(-1, 2)
            elif key == 'roi_ignr':
                # Each inner { } block is one exclusion polygon of (x,y) pairs
                polygons = []
                for grp in re.findall(r'\{([^{}]+)\}', val):
                    gnums = [float(n) for n in re.findall(r'[-+]?\d*\.?\d+', grp)]
                    gn = (len(gnums) // 2) * 2
                    if gn >= 6:
                        polygons.append(
                            np.array(gnums[:gn]).reshape(-1, 2).astype(np.int32)
                        )
                if polygons:
                    result['roi_ignr'] = polygons
            elif key == 'src_fn':
                try:
                    result['cam_idx'] = int(val.strip())
                except ValueError:
                    pass   # src_fn is a filename, not a live-cam index
    return result if result['roi_circ'] is not None else None


def _fit_circle(pts: np.ndarray) -> tuple:
    """Least-squares algebraic circle fit.  Returns (cx, cy, radius)."""
    x, y = pts[:, 0].astype(float), pts[:, 1].astype(float)
    A = np.column_stack([x, y, np.ones(len(x))])
    b = x**2 + y**2
    coeffs, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    cx = coeffs[0] / 2.0
    cy = coeffs[1] / 2.0
    r  = float(np.sqrt(max(0.0, coeffs[2] + cx**2 + cy**2)))
    return float(cx), float(cy), r


def draw_overlay(
    frame: np.ndarray,
    show_crosshair: bool,
    roi_data: "dict | None",
    show_roi: bool,
) -> np.ndarray:
    """
    Draw alignment crosshair and/or FicTrac ROI circle on a copy of frame.
    Crosshair (cyan) marks the image centre.  ROI circle (green) shows where
    the ball should be based on the last FicTrac config.
    roi_data: dict from load_fictrac_roi (with 'roi_circ' key), or None.
    """
    out = frame.copy()
    h, w = out.shape[:2]
    thick = max(1, w // 640)   # scale line thickness to image resolution

    if show_crosshair:
        cx, cy = w // 2, h // 2
        gap    = max(8, w // 80)   # gap at centre so the ROI circle stays visible
        cv2.line(out, (0, cy),        (cx - gap, cy), (0, 255, 255), thick, cv2.LINE_AA)
        cv2.line(out, (cx + gap, cy), (w, cy),        (0, 255, 255), thick, cv2.LINE_AA)
        cv2.line(out, (cx, 0),        (cx, cy - gap), (0, 255, 255), thick, cv2.LINE_AA)
        cv2.line(out, (cx, cy + gap), (cx, h),        (0, 255, 255), thick, cv2.LINE_AA)
        cv2.circle(out, (cx, cy), gap // 2, (0, 255, 255), thick, cv2.LINE_AA)

    if show_roi and roi_data is not None:
        # ── Exclusion zones — semi-transparent red polygons, drawn behind ROI ──
        for poly in roi_data.get('roi_ignr', []):
            _lay = out.copy()
            cv2.fillPoly(_lay, [poly], (30, 30, 220))          # dark-red fill
            cv2.addWeighted(_lay, 0.35, out, 0.65, 0, out)
            cv2.polylines(out, [poly], True, (60, 60, 255), thick + 1, cv2.LINE_AA)
        if roi_data.get('roi_ignr'):
            cv2.putText(out, f"{len(roi_data['roi_ignr'])} excl zone(s)",
                        (8, h - 8), cv2.FONT_HERSHEY_SIMPLEX,
                        0.40, (60, 60, 255), thick, cv2.LINE_AA)
        # ── Ball ROI circle (green) ──────────────────────────────────────────
        pts = roi_data['roi_circ']
        bx, by, br = _fit_circle(pts)
        ibx, iby, ibr = int(round(bx)), int(round(by)), int(round(br))
        cv2.circle(out, (ibx, iby), ibr, (0, 255, 0), thick + 1, cv2.LINE_AA)
        for pt in pts.astype(int):
            cv2.circle(out, tuple(pt), max(3, thick * 2), (0, 200, 0), -1)
        arm = max(10, ibr // 8)
        cv2.line(out, (ibx - arm, iby), (ibx + arm, iby), (0, 255, 0), thick, cv2.LINE_AA)
        cv2.line(out, (ibx, iby - arm), (ibx, iby + arm), (0, 255, 0), thick, cv2.LINE_AA)
        lbl_y = max(16, iby - ibr - 6)
        cv2.putText(out, "FicTrac ROI", (ibx - ibr, lbl_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45 * thick, (0, 255, 0), thick, cv2.LINE_AA)
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Interactive view transform  (zoom / pan / rotate per camera)
# ──────────────────────────────────────────────────────────────────────────────

# Arrow-key codes returned by cv2.waitKeyEx() on Windows
_KEY_UP    = 2490368
_KEY_DOWN  = 2621440
_KEY_LEFT  = 2424832
_KEY_RIGHT = 2555904


class ViewState:
    """Per-camera live view transform: zoom, pan, rotate."""
    __slots__ = ('zoom', 'pan_x', 'pan_y', 'angle')

    def __init__(self):
        self.zoom  = 1.0   # >1.0 => zoomed in
        self.pan_x = 0.5   # normalised 0..1; 0.5 = centred
        self.pan_y = 0.5
        self.angle = 0.0   # degrees, clock-wise positive

    def reset(self):
        self.zoom = 1.0; self.pan_x = 0.5; self.pan_y = 0.5; self.angle = 0.0

    @property
    def is_default(self):
        return (abs(self.zoom - 1.0) < 1e-9
                and abs(self.pan_x - 0.5) < 1e-9
                and abs(self.pan_y - 0.5) < 1e-9
                and abs(self.angle % 360.0) < 1e-9)

    def zoom_by(self, factor: float, z_min: float = 1.0, z_max: float = 20.0):
        self.zoom = max(z_min, min(z_max, self.zoom * factor))

    def pan_by(self, dx: float, dy: float):
        step = 1.0 / self.zoom          # pan step shrinks as you zoom in
        self.pan_x = max(0.0, min(1.0, self.pan_x + dx * step))
        self.pan_y = max(0.0, min(1.0, self.pan_y + dy * step))

    def rotate_by(self, deg: float):
        """Snap to the nearest 90° step in the given direction."""
        current_snap = round(self.angle / 90.0) * 90.0
        if deg > 0:
            self.angle = (current_snap + 90.0) % 360.0
        else:
            self.angle = (current_snap - 90.0) % 360.0

    def status(self) -> str:
        parts = []
        if abs(self.zoom - 1.0) > 0.005:
            parts.append(f"×{self.zoom:.1f}")
        a = self.angle % 360.0
        if a > 0.05:
            parts.append(f"{a:.0f}°")
        if abs(self.pan_x - 0.5) > 0.01 or abs(self.pan_y - 0.5) > 0.01:
            parts.append("panned")
        return " ".join(parts) if parts else "fit"


def apply_view_transform(
    frame: np.ndarray,
    vs: ViewState,
    orig_w: int = 0,
    orig_h: int = 0,
) -> np.ndarray:
    """
    Rotate (snap to 90° multiples, lossless) then crop+zoom.
    Output is ALWAYS orig_w × orig_h (sensor canvas size).
    Rotated portrait frames are pillarboxed back to the landscape canvas
    so composite layout never changes regardless of rotation.
    """
    # Use frame's own dims as canon if not provided
    orig_w = orig_w or frame.shape[1]
    orig_h = orig_h or frame.shape[0]

    if vs.is_default:
        # Still canonicalise in case caller passed different orig_w/h
        return _letterbox(frame, orig_w, orig_h)

    h, w = frame.shape[:2]
    out  = frame
    # Step 1 — rotate: use lossless cv2.rotate for exact 90° multiples
    snap = int(round(vs.angle / 90.0)) % 4
    if snap == 1:
        out = cv2.rotate(out, cv2.ROTATE_90_CLOCKWISE)
    elif snap == 2:
        out = cv2.rotate(out, cv2.ROTATE_180)
    elif snap == 3:
        out = cv2.rotate(out, cv2.ROTATE_90_COUNTERCLOCKWISE)
    # After 90/270 rotation frame dims are swapped — recalculate
    h, w = out.shape[:2]
    # Step 2 — zoom crop (operates in the rotated frame's own coordinate space)
    if abs(vs.zoom - 1.0) > 1e-9 or abs(vs.pan_x - 0.5) > 1e-9 or abs(vs.pan_y - 0.5) > 1e-9:
        cw   = max(1, int(w / vs.zoom))
        ch   = max(1, int(h / vs.zoom))
        cx   = int(vs.pan_x * w)
        cy   = int(vs.pan_y * h)
        x1   = min(max(0, cx - cw // 2), w - cw)
        y1   = min(max(0, cy - ch // 2), h - ch)
        crop = out[y1:y1 + ch, x1:x1 + cw]
        # Resize crop back to current (possibly swapped) frame dims
        out  = cv2.resize(crop, (w, h), interpolation=cv2.INTER_LINEAR)
    # Step 3 — always return exactly orig_w × orig_h (letterbox / pillarbox as needed)
    return _letterbox(out, orig_w, orig_h)


def connect_cameras():
    """Return (system, cam_list, [cam0, cam1])."""
    system   = PySpin.System.GetInstance()
    ver      = system.GetLibraryVersion()
    print(f"Spinnaker library: {ver.major}.{ver.minor}.{ver.type}.{ver.build}")

    cam_list = system.GetCameras()
    num      = cam_list.GetSize()
    if num < 2:
        cam_list.Clear()
        system.ReleaseInstance()
        raise RuntimeError(
            f"Found {num} Spinnaker camera(s) — need at least 2.\n"
            "  * Check USB connections and confirm cameras appear in SpinView.\n"
            "  * Older Flea2 cameras need FlyCapture2 SDK, not Spinnaker."
        )
    print(f"Found {num} camera(s). Using the first two.")
    cams = [cam_list.GetByIndex(i) for i in range(2)]
    for i, cam in enumerate(cams):
        cam.Init()
        tl      = cam.GetTLDeviceNodeMap()
        model_n = PySpin.CStringPtr(tl.GetNode("DeviceModelName"))
        sn_n    = PySpin.CStringPtr(tl.GetNode("DeviceSerialNumber"))
        model   = model_n.GetValue() if PySpin.IsReadable(model_n) else "?"
        sn      = sn_n.GetValue()    if PySpin.IsReadable(sn_n)    else "?"
        print(f"  Camera {i}: {model}  [S/N {sn}]")
    return system, cam_list, cams


def _configure_common(cam, fps: float) -> None:
    """Settings applied to every camera regardless of mode."""
    nm = cam.GetNodeMap()
    _enum_set(nm, "AcquisitionMode", "Continuous")
    _set_buffer_newest_only(cam)
    _set_full_fov(cam)
    # Frame rate — node name differs by firmware: try both spellings silently
    enabled = (
        _bool_set(nm, "AcquisitionFrameRateEnable",  True, silent=True) or  # newer firmware
        _bool_set(nm, "AcquisitionFrameRateEnabled", True, silent=True)     # older Flea3 firmware
    )
    if enabled:
        _float_set(nm, "AcquisitionFrameRate", fps)


def _disable_frame_rate_control(nm) -> None:
    _bool_set(nm, "AcquisitionFrameRateEnable", False, silent=True)
    _bool_set(nm, "AcquisitionFrameRateEnabled", False, silent=True)


def _maximize_link_throughput(nm) -> None:
    if _enum_set(nm, "DeviceLinkThroughputLimitMode", "Off"):
        print("  DeviceLinkThroughputLimitMode set to Off.")
        return

    limit_max = _int_node_max(nm, "DeviceLinkThroughputLimit")
    if limit_max and _int_node_set(nm, "DeviceLinkThroughputLimit", limit_max):
        print(f"  DeviceLinkThroughputLimit set to max ({limit_max}).")
        return

    stream_limit_max = _int_node_max(nm, "StreamBytesPerSecond")
    if stream_limit_max and _int_node_set(nm, "StreamBytesPerSecond", stream_limit_max):
        print(f"  StreamBytesPerSecond set to {stream_limit_max}.")
        return

    print("  [warn] Could not adjust camera link throughput settings.")


def _report_trigger_overlap(nm, overlap_ok: bool) -> None:
    """Read back and report TriggerOverlap status with all available entries."""
    node = PySpin.CEnumerationPtr(nm.GetNode("TriggerOverlap"))
    if PySpin.IsReadable(node):
        current = node.GetCurrentEntry()
        current_name = current.GetSymbolic() if PySpin.IsReadable(current) else "?"
        entries = []
        for e in node.GetEntries():
            entry = PySpin.CEnumEntryPtr(e)
            if PySpin.IsReadable(entry):
                entries.append(entry.GetSymbolic())
        print(f"  TriggerOverlap = {current_name}  (available: {entries})")
        if current_name in ("ReadOut", "PreviousFrame"):
            print("  >>> Overlap ENABLED — max rate ≈ free-run rate <<<")
        else:
            print("  [info] Overlap OFF — frame time = exposure + readout.")
    else:
        print("  [info] TriggerOverlap node not readable on this camera.")
        if not overlap_ok:
            print("  [info] Overlap not available — frame time = exposure + readout.")


def _configure_triggered_exposure(nm, exposure_us: float) -> None:
    _enum_set(nm, "ExposureAuto", "Off")
    _enum_set(nm, "ExposureMode", "Timed")
    if _float_set(nm, "ExposureTime", exposure_us):
        print(f"  Trigger exposure fixed at {exposure_us:.0f} us.")
        return
    if _float_set(nm, "ExposureTimeAbs", exposure_us):
        print(f"  Trigger exposure fixed at {exposure_us:.0f} us.")
        return
    print("  [warn] Could not set a fixed exposure time for trigger mode.")


def configure_camera_software_mode(cam, fps: float) -> None:
    """Free-running software-viewer mode."""
    _configure_common(cam, fps)
    nm = cam.GetNodeMap()
    _maximize_link_throughput(nm)
    _enum_set(nm, "TriggerMode", "Off")
    print("  Software mode: free-run, TriggerMode Off.")


def configure_camera_daq_mode(cam, exposure_us: float = None,
                              roi_height: int = None,
                              binning: int = 1) -> None:
    """DAQ-triggered mode: FrameStart on Line0, ExposureActive on Line2.

    Parameters
    ----------
    exposure_us : float, optional
        Fixed exposure time in microseconds.  Default: DAQ_EXPOSURE_US.
        Lower values allow higher trigger rates (less dead-time per frame).
    roi_height : int, optional
        Vertical ROI in pixels.  0 or None = full sensor.
        On Blackfly S global-shutter CMOS cameras, smaller ROI may increase
        the practical trigger rate.
    binning : int
        1 = no binning (default), 2 = 2×2 binning.
        Binning may increase the practical trigger rate.
    """
    if exposure_us is None:
        exposure_us = DAQ_EXPOSURE_US
    if roi_height is None:
        roi_height = DAQ_ROI_HEIGHT

    nm = cam.GetNodeMap()
    _set_buffer_newest_only(cam)

    # Must disable trigger mode before changing image format / binning
    _enum_set(nm, "TriggerMode", "Off")
    _enum_set(nm, "AcquisitionMode", "Continuous")

    # Binning (must be set before FOV so max dimensions update)
    if binning >= 2:
        h_ok = _int_node_set(nm, "BinningHorizontal", binning)
        v_ok = _int_node_set(nm, "BinningVertical", binning)
        if h_ok or v_ok:
            new_w = _int_node_max(nm, "Width")
            new_h = _int_node_max(nm, "Height")
            bh = _int_node_value(nm, "BinningHorizontal")
            bv = _int_node_value(nm, "BinningVertical")
            print(f"  Binning {bh}x{bv} → max res {new_w}x{new_h}")
        else:
            print(f"  [warn] Binning {binning}x{binning} could not be set.")
    else:
        # Reset binning to 1×1
        _int_node_set(nm, "BinningHorizontal", 1)
        _int_node_set(nm, "BinningVertical", 1)

    # Set FOV (full or reduced ROI)
    if roi_height and roi_height > 0:
        _set_roi_height(cam, roi_height)
    else:
        _set_full_fov(cam)

    _disable_frame_rate_control(nm)
    _maximize_link_throughput(nm)
    _configure_triggered_exposure(nm, exposure_us)
    _enum_set(nm, "TriggerSelector", "FrameStart")
    _enum_set(nm, "TriggerSource", DAQ_TRIGGER_LINE)
    _enum_set(nm, "TriggerActivation", "RisingEdge")
    _float_set(nm, "TriggerDelay", 0.0)
    _float_set(nm, "TriggerDelayAbs", 0.0)

    # Try TriggerOverlap BEFORE TriggerMode=On (some cameras need this)
    overlap_ok = _enum_set(nm, "TriggerOverlap", "ReadOut")

    try:
        _enum_set(nm, "LineSelector", DAQ_OUTPUT_LINE)
        _enum_set(nm, "LineMode", "Output")
        _enum_set(nm, "LineSource", "ExposureActive")
        print(f"  DAQ mode output on {DAQ_OUTPUT_LINE} = ExposureActive.")
    except Exception as exc:
        print(f"  [warn] DAQ output config failed: {exc}")
    _enum_set(nm, "TriggerMode", "On")

    # Try TriggerOverlap AFTER TriggerMode=On (many FLIR cameras require this)
    if not overlap_ok:
        overlap_ok = _enum_set(nm, "TriggerOverlap", "ReadOut")
        if not overlap_ok:
            # Also try "PreviousFrame" — some firmware versions use this name
            overlap_ok = _enum_set(nm, "TriggerOverlap", "PreviousFrame")

    # Probe and report what TriggerOverlap actually is
    _report_trigger_overlap(nm, overlap_ok)

    if overlap_ok:
        print(f"  DAQ trigger on {DAQ_TRIGGER_LINE} (rising edge), overlap ON.")
        print("  Overlap-capable hardware is configured; verify the accepted trigger rate with tests/verify_camera_trigger_path.py.")
    else:
        print(f"  DAQ trigger on {DAQ_TRIGGER_LINE} (rising edge), overlap OFF.")
        print("  Trigger overlap is unavailable; expect the accepted trigger rate to be limited by exposure plus sensor readout.")


def configure_camera_daq_freerun_mode(cam, fps: float = 60.0,
                                       exposure_us: float = None) -> None:
    """Free-run mode with DAQ frame timestamping via ExposureActive.

    The camera runs at its internal frame rate (up to 60 fps at full
    resolution) with NO external trigger for individual frames.
    Line2 outputs ExposureActive so the DAQ can record rising edges
    and timestamp every frame on its own clock.

    This bypasses the ~28 Hz FrameStart trigger limit caused by the
    Flea3 CCD's lack of TriggerOverlap.

    Parameters
    ----------
    fps : float
        Target free-run frame rate (default 60).
    exposure_us : float, optional
        Fixed exposure time in µs.  Default: DAQ_EXPOSURE_US.
        Will be clamped to the camera's valid range.
    """
    if exposure_us is None:
        exposure_us = DAQ_EXPOSURE_US
    nm = cam.GetNodeMap()
    _set_buffer_newest_only(cam)

    # Disable trigger and reset binning before changing image format
    _enum_set(nm, "TriggerMode", "Off")
    _enum_set(nm, "AcquisitionMode", "Continuous")
    _int_node_set(nm, "BinningHorizontal", 1)
    _int_node_set(nm, "BinningVertical", 1)
    _set_full_fov(cam)
    _maximize_link_throughput(nm)

    # Configure exposure
    _enum_set(nm, "ExposureAuto", "Off")
    _enum_set(nm, "ExposureMode", "Timed")
    _float_set(nm, "ExposureTime", exposure_us) or \
        _float_set(nm, "ExposureTimeAbs", exposure_us)

    # Enable frame rate control at the desired fps
    enabled = (
        _bool_set(nm, "AcquisitionFrameRateEnable",  True, silent=True) or
        _bool_set(nm, "AcquisitionFrameRateEnabled", True, silent=True)
    )
    if enabled:
        ok = _float_set(nm, "AcquisitionFrameRate", fps)
        if not ok:
            _float_set(nm, "AcquisitionFrameRateAbs", fps)
    else:
        # Some cameras don't have a separate enable — try setting directly
        ok = _float_set(nm, "AcquisitionFrameRate", fps)
        if not ok:
            _float_set(nm, "AcquisitionFrameRateAbs", fps)

    # Output ExposureActive on Line2 so DAQ can timestamp frames
    try:
        _enum_set(nm, "LineSelector", DAQ_OUTPUT_LINE)
        _enum_set(nm, "LineMode", "Output")
        _enum_set(nm, "LineSource", "ExposureActive")
        print(f"  Output on {DAQ_OUTPUT_LINE} = ExposureActive.")
    except Exception as exc:
        print(f"  [warn] DAQ output config failed: {exc}")

    # Read back actual exposure and frame rate
    exp_node = PySpin.CFloatPtr(nm.GetNode("ExposureTime"))
    actual_exp = exp_node.GetValue() if PySpin.IsReadable(exp_node) else exposure_us
    fr_node = PySpin.CFloatPtr(nm.GetNode("AcquisitionFrameRate"))
    actual_fps = fr_node.GetValue() if PySpin.IsReadable(fr_node) else fps
    print(f"  Free-run mode: {actual_fps:.1f} fps, exposure {actual_exp:.0f} µs.")
    print(f"  DAQ reads ExposureActive on {DAQ_OUTPUT_LINE} for frame timestamps.")
    print(f"  No external trigger needed — camera runs immediately on BeginAcquisition.")


def release_cameras(system, cam_list, cams, restore_daq: bool = False) -> None:
    """Release all PySpin camera references before clearing the system instance."""
    if cams is None:
        if cam_list is not None:
            cam_list.Clear()
        if system is not None:
            system.ReleaseInstance()
        return

    for idx in range(len(cams)):
        cam = cams[idx]
        if cam is None:
            continue
        try:
            cam.EndAcquisition()
        except Exception:
            pass
        if restore_daq:
            try:
                print("  Restoring DAQ-triggered mode ...")
                configure_camera_daq_mode(cam)
            except Exception as exc:
                print(f"  [warn] Failed to restore DAQ mode: {exc}")
        try:
            cam.DeInit()
        except Exception:
            pass
        cams[idx] = None

    del cam
    gc.collect()
    cams.clear()
    cam_list.Clear()
    system.ReleaseInstance()


# ──────────────────────────────────────────────────────────────────────────────
# Frame acquisition  (ImageProcessor is the correct API for Spinnaker 4.x)
# ──────────────────────────────────────────────────────────────────────────────

# Shared ImageProcessor with HQ debayer — created once, reused every frame
_processor = PySpin.ImageProcessor()
_processor.SetColorProcessing(PySpin.SPINNAKER_COLOR_PROCESSING_ALGORITHM_HQ_LINEAR)


def _spinnaker_to_bgr(raw_img) -> np.ndarray:
    """Convert a raw PySpin image to a BGR numpy array via ImageProcessor."""
    bgr = _processor.Convert(raw_img, PySpin.PixelFormat_BGR8)
    return bgr.GetNDArray().copy()  # copy before the source image is released


def _grab_worker(cam, out_q: queue.Queue, timeout_ms: int) -> None:
    """Thread target: grab one frame and push BGR array (or None) onto out_q."""
    try:
        img = cam.GetNextImage(timeout_ms)
        if img.IsIncomplete():
            print(f"  [warn] Incomplete frame (status {img.GetImageStatus()})")
            out_q.put(None)
        else:
            out_q.put(_spinnaker_to_bgr(img))
        img.Release()
    except PySpin.SpinnakerException as exc:
        print(f"  [warn] Grab failed: {exc}")
        out_q.put(None)


def grab_pair(cams: list) -> tuple:
    """Grab one frame from each camera simultaneously in two threads."""
    qs = [queue.Queue(), queue.Queue()]
    threads = [
        threading.Thread(
            target=_grab_worker,
            args=(cams[i], qs[i], GRAB_TIMEOUT),
            daemon=True,
        )
        for i in range(2)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=(GRAB_TIMEOUT / 1000.0) + 0.5)

    frames = []
    for q in qs:
        try:
            frames.append(q.get_nowait())
        except queue.Empty:
            frames.append(None)
    return frames[0], frames[1]


# ──────────────────────────────────────────────────────────────────────────────
# Display helpers
# ──────────────────────────────────────────────────────────────────────────────

def _scale(frame: np.ndarray, s: float) -> np.ndarray:
    if s == 1.0:
        return frame
    h, w = frame.shape[:2]
    return cv2.resize(frame, (int(w * s), int(h * s)))


def _letterbox(img: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
    """
    Fit img into a target_w × target_h canvas, preserving aspect ratio.
    Unused areas are filled with black (pillarbox or letterbox).
    """
    ih, iw = img.shape[:2]
    if iw == target_w and ih == target_h:
        return img
    scale  = min(target_w / iw, target_h / ih)
    fit_w  = max(1, int(round(iw * scale)))
    fit_h  = max(1, int(round(ih * scale)))
    resized = cv2.resize(img, (fit_w, fit_h), interpolation=cv2.INTER_LINEAR)
    canvas  = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    y_off   = (target_h - fit_h) // 2
    x_off   = (target_w - fit_w) // 2
    canvas[y_off:y_off + fit_h, x_off:x_off + fit_w] = resized
    return canvas


def build_side_by_side(
    left:   np.ndarray,
    right:  np.ndarray,
    scale:  float = DISPLAY_SCALE,
    labels: tuple  = ("Camera 0", "Camera 1"),
) -> np.ndarray:
    left  = _scale(left,  scale)
    right = _scale(right, scale)

    h = max(left.shape[0], right.shape[0])

    def _pad(img):
        if img.shape[0] < h:
            p = np.zeros((h - img.shape[0], img.shape[1], 3), dtype=np.uint8)
            return np.vstack([img, p])
        return img

    left, right = _pad(left), _pad(right)
    divider     = np.full((h, 4, 3), 200, dtype=np.uint8)
    composite   = np.hstack([left, divider, right])

    font = cv2.FONT_HERSHEY_SIMPLEX
    for i, label in enumerate(labels):
        x = 8 if i == 0 else left.shape[1] + 12
        cv2.putText(composite, label, (x, 22), font, 0.55,
                    (0,   0,   0),   3, cv2.LINE_AA)
        cv2.putText(composite, label, (x, 22), font, 0.55,
                    (255, 255, 255), 1, cv2.LINE_AA)
    return composite


# ──────────────────────────────────────────────────────────────────────────────
# Main loop
# ──────────────────────────────────────────────────────────────────────────────

def run(fictrac_config_path: "Path | None" = None,
        fictrac_cam_idx: "int | None" = None) -> None:
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    system, cam_list, cams = connect_cameras()

    try:
        print("\nConfiguring software capture mode ...")
        for idx in range(len(cams)):
            configure_camera_software_mode(cams[idx], TARGET_FPS)
        labels = ("Camera 0", "Camera 1")

        for idx in range(len(cams)):
            cams[idx].BeginAcquisition()
        print("\nCapture started  [software sync].")
        print("  Q/ESC    quit          S  save frame")
        print("  H        crosshair     O  ROI + excl zones")
        print("  +/-      zoom          [/]  rotate ±90° (snaps to 0/90/180/270)")
        print("  Arrows   pan           R  reset view")
        print("  0        auto-target from cursor (default)")
        print("  1/2      lock target to cam 0 / cam 1")
        print("  TAB      cycle layout: side-by-side / cam0 focus / cam1 focus\n")

        # ── Grab one frame to know the actual frame dimensions ────────────────
        _f0, _f1 = grab_pair(cams)
        if _f0 is None or _f1 is None:
            raise RuntimeError("Could not grab initial frame to determine resolution.")
        cam_h, cam_w = _f0.shape[:2]
        print(f"  Frame size: {cam_w} x {cam_h} px per camera")

        # ── Auto-compute display scale to fit the composite on screen ─────────
        screen_w, screen_h = _get_screen_size()
        print(f"  Screen size: {screen_w} x {screen_h} px")
        composite_w = cam_w * 2 + 4   # two panels + 4-px divider
        composite_h = cam_h
        scale_by_w  = (screen_w * WINDOW_MARGIN) / composite_w
        scale_by_h  = (screen_h * WINDOW_MARGIN) / composite_h
        display_scale = DISPLAY_SCALE if DISPLAY_SCALE > 0 else min(scale_by_w, scale_by_h, 1.0)
        print(f"  Display scale: {display_scale:.3f}  "
              f"(window ~{int(composite_w * display_scale)} x {int(composite_h * display_scale)} px)")

        # ── Load FicTrac overlay ──────────────────────────────────────────────
        fictrac_roi = load_fictrac_roi(fictrac_config_path)
        if fictrac_roi is not None:
            auto_cam = fictrac_roi.get('cam_idx')
            if fictrac_cam_idx is None:
                fictrac_cam_idx = auto_cam if auto_cam in (0, 1) else 0
            bx, by, br = _fit_circle(fictrac_roi['roi_circ'])
            print(f"  FicTrac ROI: centre ({bx:.0f}, {by:.0f})  r={br:.0f} px  "
                  f"[cam {fictrac_cam_idx}  {len(fictrac_roi['roi_circ'])} pts]")
            print(f"  Config: {fictrac_config_path}")
        else:
            fictrac_cam_idx = 0 if fictrac_cam_idx is None else fictrac_cam_idx
            print("  FicTrac overlay: no config loaded — crosshair only")
        show_crosshair = True
        show_roi       = fictrac_roi is not None

        WIN_NAME = "Dual Flea Camera - synchronized"
        # WINDOW_KEEPRATIO: OpenCV letterboxes the composite when the window is
        # resized, so the displayed content is never stretched.
        cv2.namedWindow(WIN_NAME, cv2.WINDOW_KEEPRATIO)

        # ── Per-camera view transform state ──────────────────────────────────
        vs         = [ViewState(), ViewState()]   # one per camera
        active_cam = "auto"  # "auto" (cursor), "both", 0 or 1
        focus_cam  = None    # None=side-by-side, 0=only cam0, 1=only cam1

        # Mouse X position — used to auto-select which camera keys affect
        _mouse_x = [0]
        _scroll  = [0]
        def _on_mouse(evt, mx, _y, flags, _p):
            _mouse_x[0] = mx
            if evt == cv2.EVENT_MOUSEWHEEL:
                _scroll[0] += (1 if flags > 0 else -1)
        cv2.setMouseCallback(WIN_NAME, _on_mouse)

        fps_counter, fps_display = 0, 0.0
        t_fps      = time.perf_counter()
        first_pair = (_f0, _f1)   # reuse the warm-up frame

        while True:
            if first_pair is not None:
                frame0, frame1 = first_pair
                first_pair = None
            else:
                frame0, frame1 = grab_pair(cams)

            if frame0 is None or frame1 is None:
                print("  [warn] Skipping pair (grab error).")
                continue

            # Draw overlays on the raw sensor frames (full-frame pixel coords),
            # then apply the view transform so overlays move with image content.
            over0 = draw_overlay(
                frame0, show_crosshair,
                fictrac_roi if fictrac_cam_idx == 0 else None, show_roi,
            )
            over1 = draw_overlay(
                frame1, show_crosshair,
                fictrac_roi if fictrac_cam_idx == 1 else None, show_roi,
            )
            disp0 = apply_view_transform(over0, vs[0], cam_w, cam_h)
            disp1 = apply_view_transform(over1, vs[1], cam_w, cam_h)

            # Build composite: side-by-side or single-camera focus view
            if focus_cam is None:
                composite = build_side_by_side(
                    disp0, disp1, scale=display_scale, labels=labels
                )
            else:
                frm = disp0 if focus_cam == 0 else disp1
                composite = _scale(frm, display_scale)
                _lbl = labels[focus_cam] + "  [FOCUS]"
                cv2.putText(composite, _lbl, (8, 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                            (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(composite, _lbl, (8, 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                            (255, 255, 255), 1, cv2.LINE_AA)

            fps_counter += 1
            elapsed = time.perf_counter() - t_fps
            if elapsed >= 1.0:
                fps_display = fps_counter / elapsed
                fps_counter = 0
                t_fps       = time.perf_counter()

            _cam_lbl = (
                "auto"
                if active_cam == "auto"
                else ("both" if active_cam == "both" else f"cam{active_cam}locked")
            )
            _hud = (
                f"{fps_display:.1f} fps [software]  "
                f"act:{_cam_lbl}  "
                f"C0:{vs[0].status()}  C1:{vs[1].status()}"
            )
            cv2.putText(composite, _hud,
                        (8, composite.shape[0] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40,
                        (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(composite, _hud,
                        (8, composite.shape[0] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40,
                        (0, 255, 0), 1, cv2.LINE_AA)

            cv2.imshow(WIN_NAME, composite)

            # ── Mouse scroll zoom ────────────────────────────────────────────
            if _scroll[0] != 0:
                # auto-detect from cursor unless user locked to 1/2
                if active_cam == "auto":
                    _sw = composite.shape[1] if focus_cam is None else composite.shape[1]
                    _hover = 0 if (_mouse_x[0] < _sw // 2 or focus_cam == 0) else 1
                    if focus_cam == 1: _hover = 1
                    _trg = [vs[_hover]]
                else:
                    _trg = vs if active_cam == "both" else [vs[active_cam]]
                _fac = 1.15 ** abs(_scroll[0])
                for _v in _trg:
                    _v.zoom_by(_fac if _scroll[0] > 0 else 1.0 / _fac)
                _scroll[0] = 0

            # ── Resolve which camera keyboard actions target ──────────────────
            if active_cam == "auto":
                # cursor left-half → cam0, right-half → cam1
                # in focus mode the only visible camera is always the target
                if focus_cam is not None:
                    _cur_cam = focus_cam
                else:
                    _half_w = composite.shape[1] // 2
                    _cur_cam = 0 if _mouse_x[0] <= _half_w else 1
                _targets = [vs[_cur_cam]]
            elif active_cam == "both":
                _targets = vs
            else:
                _targets = [vs[active_cam]]

            # ── Keyboard ─────────────────────────────────────────────────────
            key = cv2.waitKeyEx(1)

            if key in (ord("q"), 27):
                print("Quit.")
                break
            elif key == ord("s"):
                ts  = time.strftime("%Y%m%d_%H%M%S")
                # Save: overlay on raw frame, then transform (same order as display)
                _s0 = apply_view_transform(draw_overlay(frame0, show_crosshair,
                                   fictrac_roi if fictrac_cam_idx == 0 else None, show_roi),
                                   vs[0], cam_w, cam_h)
                _s1 = apply_view_transform(draw_overlay(frame1, show_crosshair,
                                   fictrac_roi if fictrac_cam_idx == 1 else None, show_roi),
                                   vs[1], cam_w, cam_h)
                if focus_cam is None:
                    full = build_side_by_side(_s0, _s1, scale=1.0, labels=labels)
                else:
                    full = _s0 if focus_cam == 0 else _s1
                path = SAVE_DIR / f"frame_{ts}.jpg"
                cv2.imwrite(str(path), full)
                print(f"  Saved -> {path}  ({full.shape[1]}x{full.shape[0]} px)")
            elif key == ord("h"):
                show_crosshair = not show_crosshair
                print(f"  Crosshair: {'ON' if show_crosshair else 'OFF'}")
            elif key == ord("o"):
                if fictrac_roi is not None:
                    show_roi = not show_roi
                    print(f"  FicTrac ROI overlay: {'ON' if show_roi else 'OFF'}")
            # ── Zoom ─────────────────────────────────────────────────────────
            elif key in (ord("+"), ord("=")):
                for _v in _targets: _v.zoom_by(1.15)
                print(f"  Zoom: {' | '.join(f'cam{i} ×{vs[i].zoom:.2f}' for i in range(2))}")
            elif key in (ord("-"), ord("_")):
                for _v in _targets: _v.zoom_by(1.0 / 1.15)
                print(f"  Zoom: {' | '.join(f'cam{i} ×{vs[i].zoom:.2f}' for i in range(2))}")
            # ── Rotate (±90° snap, targets camera under cursor) ────────────
            elif key == ord("["):
                for _v in _targets: _v.rotate_by(-90.0)
                print(f"  Rotate: {' | '.join(f'cam{i} {int(vs[i].angle)}°' for i in range(2))}")
            elif key == ord("]"):
                for _v in _targets: _v.rotate_by(90.0)
                print(f"  Rotate: {' | '.join(f'cam{i} {int(vs[i].angle)}°' for i in range(2))}")
            # ── Pan (arrow keys) ─────────────────────────────────────────────
            elif key == _KEY_UP:
                for _v in _targets: _v.pan_by(0.0, -0.05)
            elif key == _KEY_DOWN:
                for _v in _targets: _v.pan_by(0.0,  0.05)
            elif key == _KEY_LEFT:
                for _v in _targets: _v.pan_by(-0.05, 0.0)
            elif key == _KEY_RIGHT:
                for _v in _targets: _v.pan_by( 0.05, 0.0)
            # ── Reset ────────────────────────────────────────────────────────
            elif key == ord("r"):
                for _v in _targets: _v.reset()
                print("  View reset.")
            # ── Active camera selection ───────────────────────────────────────
            elif key == ord("1"):
                active_cam = 0
                print("  Active: Camera 0 locked (keys always target cam 0)")
            elif key == ord("2"):
                active_cam = 1
                print("  Active: Camera 1 locked (keys always target cam 1)")
            elif key == ord("0"):
                active_cam = "auto"
                print("  Active: auto — cursor position selects camera")
            # ── Focus / layout mode ───────────────────────────────────────────
            elif key == 9:   # TAB
                if focus_cam is None:
                    focus_cam = 0
                    print("  Layout: Camera 0 focus")
                elif focus_cam == 0:
                    focus_cam = 1
                    print("  Layout: Camera 1 focus")
                else:
                    focus_cam = None
                    print("  Layout: side-by-side")

    finally:
        print("Stopping cameras ...")
        release_cameras(system, cam_list, cams, restore_daq=True)
        cv2.destroyAllWindows()
        print("Done.")


# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Dual Flea software-viewer capture (Spinnaker 4.x / PySpin)."
    )
    parser.add_argument(
        "--fps", type=float, default=TARGET_FPS,
        help=f"Target frame rate (default: {TARGET_FPS})"
    )
    parser.add_argument(
        "--scale", type=float, default=DISPLAY_SCALE,
        help="Display scale per camera panel (default: 0 = auto-fit to screen)."
    )
    parser.add_argument(
        "--config", type=Path,
        default=Path(r"c:\Rishika\fictrac_pybmt\config_camera.txt"),
        help="Path to FicTrac config.txt for ROI overlay (default: config_camera.txt)."
    )
    parser.add_argument(
        "--fictrac-cam", type=int, default=None, choices=[0, 1],
        help="Camera index the FicTrac ROI applies to (default: auto-detect from src_fn)."
    )
    args = parser.parse_args()

    TARGET_FPS    = args.fps
    DISPLAY_SCALE = args.scale

    run(fictrac_config_path=args.config, fictrac_cam_idx=args.fictrac_cam)

