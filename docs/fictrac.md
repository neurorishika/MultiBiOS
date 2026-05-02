# FicTrac Integration

This page covers the MultiBiOS-specific path for running FicTrac on this rig, including the current Blackfly camera constraint, how to build a Spinnaker-enabled FicTrac on Windows, and how NI-DAQ-triggered acquisition should be interpreted.

For the internal client architecture, full-state storage model, closed-loop consumer API, and deprecation criteria, see [fictrac_client.md](fictrac_client.md).

## Current State On This Rig

- The packaged Spinnaker build at `C:/Rishika/MultiBiOS/assets/fictrac-spinnaker/fictrac-spinnaker.exe` is the validated binary for this workstation.
- The in-repo MultiBiOS FicTrac client is receiving realtime callback frames from the live Blackfly side camera.
- `multibios.run_protocol` has been validated against live FicTrac on this rig.
- MultiBiOS now sanitizes the Windows child-process environment before launching FicTrac, which avoids the conda/Python DLL path conflict that previously caused native startup crashes.

## Validated Operating Point

The canonical rig configuration now uses the highest display-on rate that has passed repeated artifact-level parity checks on this workstation.

Canonical values in `config/hardware.yaml`:

- `fictrac.target_fps: 142.857143`
- `camera_recording.trigger_fps_hz: 142.857143`
- effective shared camera interval: `7.0 ms`

Why this is the canonical setting:

- `5.5 ms` failed twice at `999 / 1000` FicTrac frames
- `6.5 ms` failed at `845 / 847` FicTrac frames
- `7.0 ms` passed repeatedly in both FicTrac-only and dual-camera runs with exact parity

Most recent repeatability evidence:

- dual-camera exact parity: `data/runs/2026-05-01_10-27-45`
- dual-camera exact parity: `data/runs/2026-05-01_10-32-14`
- dual-camera exact parity: `data/runs/2026-05-01_10-32-33`
- dual-camera exact parity: `data/runs/2026-05-01_10-32-53`

## Why A Custom Build Is Needed

Upstream FicTrac supports Point Grey / FLIR industrial cameras through two distinct compile-time paths:

- `PGR_USB2` for FlyCapture-era cameras
- `PGR_USB3` for Spinnaker-era cameras

For a Windows Spinnaker build, upstream documents the relevant CMake configuration as:

```powershell
cmake -A x64 `
  -D CMAKE_TOOLCHAIN_FILE=<vcpkg>\scripts\buildsystems\vcpkg.cmake `
  -D PGR_USB3=ON `
  -D PGR_DIR="C:\Program Files\Teledyne\Spinnaker" `
  ..
```

The shared MultiBiOS environment now prepares the SDK runtime search path before launching FicTrac, and the launcher strips conflicting Python/conda DLL search paths on Windows before spawning the native process. That combination is what is currently validated through `multibios.run_protocol` and `tests/fictrac_live_probe.py`.

## Source Of Truth For FicTrac In This Repo

MultiBiOS now treats FicTrac as vendored source, not a disposable upstream clone.

That is intentional because this rig needs source-level changes inside FicTrac itself to work reliably with the MultiBiOS trigger model.

Practical consequences:

- `assets/third_party/FicTrac` is the source of truth and may contain local MultiBiOS-specific patches
- `assets/third_party/FicTrac-build` is only a local build directory and should stay untracked
- `assets/fictrac-spinnaker` is a packaged runtime output, not the editable source tree
- `assets/third_party/FicTrac/MULTIBIOS_PATCHSET.md` is the source-adjacent manifest for the maintained native fork

The build helper no longer silently fetches upstream and checks out `master`. That behavior was unsafe once local patches became part of the rig setup.

## Publication Status

The FicTrac changes are now documented in two layers inside this repo:

- this page documents the full integration behavior and validation history
- `assets/third_party/FicTrac/MULTIBIOS_PATCHSET.md` documents the native fork boundary next to the vendored source tree

Preferred publication model for the lab organization:

1. Create a standalone lab-owned fork repository for the native FicTrac patch set.
2. Publish the full patched source tree there, preserving upstream license and attribution.
3. Keep MultiBiOS as the downstream consumer that pins a known commit or release from that fork.

Why that split is cleaner:

- the native patch set and the Python integration have different maintenance boundaries
- native camera fixes should be versioned independently from experiment-runner changes
- it avoids treating a packaged executable in this repo as the only durable record of the fork

What this repo now contains to make that publication straightforward:

- a source-adjacent native patch manifest
- rebuild instructions for the validated Windows Spinnaker toolchain
- a validated operating point in the canonical hardware config
- a standalone publication scaffold at `Fictrac-TrigWin/` for the planned lab-owned repo

What is still missing for an actual external publication:

- creation of the remote lab-owned repository
- import of the patched FicTrac source history or an initial fork snapshot
- release tags that map packaged binaries to published source commits

## First-Frame Trigger Failure On This Rig

The April 2026 blocker on this workstation was not the MultiBiOS-side `fictrac_timeout_s` setting. It was the upstream Spinnaker first-frame wait inside FicTrac itself.

The confirmed failure signature was:

- protocol path: `protocols/odor_lateralization.yaml`
- runtime config path: `C:\Rishika\data\runs\2026-04-29_23-34-01\fictrac_runtime_config.txt`
- unpatched binary behavior: exits in about 1 s before any UDP frame arrives
- direct console error:

```text
Error grabbing frame! Error was: Spinnaker: Failed waiting for EventData on NEW_BUFFER_DATA event. (GenTL error code: -1011) [-1011]
```

The root cause is in upstream FicTrac's Spinnaker camera path:

- `assets/third_party/FicTrac/src/PGRSource.cpp`
- `PGRSource::grab()`
- `GetNextImage(timeout)` used a hard-coded minimum wait of `1000` ms when `src_fps = -1`

That is too short for a hardware-triggered startup where the first camera pulse may be delayed.

## Exact Custom Patch History

The vendored FicTrac tree now contains these local source edits that matter on this rig.

### 1. Spinnaker 4.x conversion compatibility

File:

- `assets/third_party/FicTrac/src/PGRSource.cpp`

Edit made:

- switched the Spinnaker conversion path to `ImageProcessor().Convert(..., PixelFormat_BGR8)`
- set the processor color handling explicitly with `SPINNAKER_COLOR_PROCESSING_ALGORITHM_NEAREST_NEIGHBOR`

Why it exists:

- the older conversion call pattern used by upstream did not match the Spinnaker 4.x SDK on this workstation

### 2. First-frame wait is now configurable instead of fixed at 1 s

Files:

- `assets/third_party/FicTrac/include/PGRSource.h`
- `assets/third_party/FicTrac/src/PGRSource.cpp`
- `assets/third_party/FicTrac/src/Trackball.cpp`

Edits made:

- `PGRSource` now accepts a `first_frame_timeout_ms` constructor argument
- `Trackball.cpp` reads `src_first_frame_timeout_ms` from the FicTrac config and passes it into `PGRSource`
- the Spinnaker path still uses the normal per-frame timeout after tracking starts
- for the very first frame only:
  - positive `src_first_frame_timeout_ms` values wait that many milliseconds
  - `0` or any negative value means wait indefinitely for the first trigger

Why it exists:

- the upstream Spinnaker live-camera path aborted too early for hardware-triggered startup on this rig
- we need to be able to arm FicTrac before the DAQ trigger train or animal recording starts

### 3. Exception cleanup no longer assumes an image object exists

File:

- `assets/third_party/FicTrac/src/PGRSource.cpp`

Edit made:

- exception paths now guard `pgr_image->Release()` with a null check

Why it exists:

- a failed first grab should report the real capture error, not cascade into a null-image cleanup bug

### 4. The classic FicTrac configuration UI now defaults this setting to `0`

Files:

- `assets/third_party/FicTrac/src/ConfigGUI.cpp`

Edits made:

- `configGui` now seeds `src_first_frame_timeout_ms` to `0` when the key is missing
- when `configGui` opens a live PGR/Spinnaker camera, it now passes that configured value into `PGRSource`

Why it exists:

- there is no separate MultiBiOS reconfiguration path for FicTrac calibration today
- the real reconfiguration workflow is still the upstream `configGui.exe` UI
- when that UI is re-run on this rig, it should preserve the rig-safe default of waiting indefinitely for the first trigger unless the user chooses a different value

### 5. Windows shutdown now uses a real graceful-stop path

Files:

- `assets/third_party/FicTrac/exec/fictrac.cpp`
- `assets/third_party/FicTrac/include/PGRSource.h`
- `assets/third_party/FicTrac/src/PGRSource.cpp`
- `assets/third_party/FicTrac/src/FrameGrabber.cpp`
- `multibios/fictrac_client.py`
- `multibios/run_protocol.py`
- `multibios/blackfly/triggered_camera_record.py`

Edits made:

- the Windows FicTrac entrypoint now handles `SIGBREAK` in addition to `SIGINT`
- `PGRSource::~PGRSource()` now calls `DeInit()` before the camera list and Spinnaker system are released
- the Spinnaker `NEW_BUFFER_DATA` / `GenTL error code: -1011` condition at end-of-trigger is now treated as a clean stream end rather than a fatal grab failure
- `FrameGrabber` now logs a normal input-stream end when the source closed itself cleanly
- the MultiBiOS driver still uses `CTRL_BREAK_EVENT` first on Windows, but that signal now lands in FicTrac's own shutdown path instead of hard-killing the process
- `run_protocol` now stops the second triggered Blackfly recorder before the generic DAQ teardown path finishes unwinding

Why it exists:

- a hard Windows console break or process termination can bypass the cleanup path that releases the Spinnaker camera cleanly
- this rig uses externally triggered Blackfly cameras, so a finite trigger train can end before the parent process explicitly asks FicTrac to stop
- on this rig, treating end-of-trigger as a fatal camera error left the FicTrac camera in a non-editable ROI state after otherwise successful runs

## The Right Way To Kill FicTrac On This Rig

Use a graceful stop, not a hard kill.

Validated shutdown contract:

- on Windows, MultiBiOS should request shutdown with `CTRL_BREAK_EVENT`
- the FicTrac executable must handle that as `SIGBREAK` and unwind normally through `Trackball`, `FrameGrabber`, and `PGRSource`
- the Spinnaker camera path must reach `EndAcquisition()` and `DeInit()` before process exit
- if the finite NI-DAQ trigger train ends first, the Spinnaker `NEW_BUFFER_DATA` / `-1011` condition should be treated as end-of-stream, not as a fatal crash
- the second triggered Blackfly recorder should be asked to stop before late generic task cleanup, so its capture loop sees shutdown intent instead of an unexplained trigger disappearance

Do not rely on these as the primary shutdown path:

- `TerminateProcess`
- `kill()`
- `Popen.terminate()` on Windows as the first stop attempt
- assuming that a finite hardware trigger train ending by itself is equivalent to a clean FicTrac shutdown

Those paths are still acceptable only as last-resort fallbacks when the process is already unresponsive.

Healthy shutdown signatures validated on this workstation:

- probe path: `fictrac_driver_diagnostics.json` ends with `stop_method: "ctrl_break"` and `final_returncode: 0`
- maintained protocol path: FicTrac logs `PGR trigger stream ended; closing camera cleanly.` followed by `Input stream ended.`
- post-run camera inspection reports writable ROI nodes again (`width_writable=true`, `height_writable=true`)

## Raw Recording Format

FicTrac raw camera recording in this repo no longer depends on a single native `VideoWriter` AVI flush during shutdown.

The native path now writes a chunked raw frame stream:

- `fictrac-raw-<timestamp>.json`: manifest with geometry, fps, and chunk paths
- `fictrac-raw-<timestamp>-index.csv`: per-frame index aligned to FicTrac `log_frame`
- `fictrac-raw-<timestamp>-chunkNNNNNN.bin`: BGR8 frame chunks

After the run, MultiBiOS reconstructs those chunks into a lossless review video and includes the result in `fictrac_camera_recording.json`.

### What reconstructs the bins into a video

The reconstruction path lives in `multibios/fictrac_raw_recording.py` and is called automatically from `multibios/run_protocol.py` at the end of a protocol run.

Operationally, the postprocess step does this:

1. discover the newest `fictrac-raw-*.json` manifest in the run directory
2. load `frame_width`, `frame_height`, `channels`, `fps`, and `chunk_paths` from that manifest
3. read `fictrac-raw-*-index.csv` and count valid indexed frames
4. stream each `.bin` chunk back through NumPy memmaps in chunk order
5. write a lossless review video named `fictrac-raw-<timestamp>-lossless.avi` or `.mkv`
6. store the result in `fictrac_camera_recording.json` under `lossless_video`

The writer currently tries the same codec/container candidates used elsewhere in MultiBiOS:

- `FFV1` in `.avi`
- `HFYU` in `.avi`
- `FFV1` in `.mkv`

### What must exist to reconstruct a FicTrac run later

For a completed run, these files are the minimum useful set for re-running reconstruction:

- `fictrac-raw-<timestamp>.json`
- `fictrac-raw-<timestamp>-index.csv`
- every referenced `fictrac-raw-<timestamp>-chunkNNNNNN.bin`

The manifest is the source of truth for geometry and chunk order. The CSV index is the source of truth for saved-frame accounting. The `.bin` chunks contain the actual frame payload.

If the bins are missing, MultiBiOS can still report previously computed metadata from `fictrac_camera_recording.json`, but it cannot regenerate the lossless review video.

### Manual reconstruction for an existing run

If a run already contains FicTrac chunks but does not yet contain the final lossless video, you can reconstruct it by re-running the same Python postprocess function that `run_protocol.py` uses.

From the MultiBiOS repo root:

```powershell
C:/ProgramData/miniconda3/Scripts/conda.exe run -p C:\Users\markd\.conda\envs\multibios-blackfly --no-capture-output python -c "from pathlib import Path; import json; from multibios.fictrac_raw_recording import postprocess_fictrac_raw_recording; run_dir = Path(r'C:\Rishika\MultiBiOS\data\runs\2026-05-01_20-28-46'); summary = postprocess_fictrac_raw_recording(run_dir=run_dir, runtime_info=json.loads((run_dir / 'fictrac_runtime.json').read_text(encoding='utf-8')), frame_count=None, expected_frame_count=None, legacy_raw_videos=[], legacy_saved_raw_frames=None); print(summary['lossless_video'])"
```

Practical notes:

- run this from the repo root so any repo-relative paths in the manifest resolve the same way they did during protocol teardown
- if `fictrac_runtime.json` is missing, you can still reconstruct as long as the manifest contains `fps`; if it does not, you must supply an equivalent `camera_fps` value in `runtime_info`
- the output video basename is derived from the manifest name, so re-running reconstruction rewrites the same `*-lossless.avi` or `*.mkv` target

### How frame counts are interpreted

The reconstruction summary in `fictrac_camera_recording.json` intentionally separates several counts:

- `callback_frames`: frames seen by the Python callback path
- `saved_raw_frames`: frames confirmed from the raw recording postprocess
- `lossless_video.frames_written`: frames emitted into the review video
- `expected_frames`: usually trigger-count derived when protocol validation is enabled

For chunked runs, `saved_raw_frames` comes from the CSV index, not from blindly trusting the manifest's `saved_frames` field and not from trusting an AVI container header.

That means the correct parity check is against `fictrac_camera_recording.json`, not against the raw chunk count alone.

### Interaction with automatic raw chunk cleanup

`hardware.yaml` can now control whether raw chunk files are retained after validation with:

```yaml
camera_recording:
  raw_chunk_retention_policy: keep # or delete_after_parity
```

Behavior:

- `keep`: retain all `.bin` chunks after postprocess
- `delete_after_parity`: delete FicTrac and second-camera `.bin` chunks only after parity checks pass and the final lossless videos validate successfully

When cleanup runs, `fictrac_camera_recording.json` is annotated with `raw_chunk_cleanup` and `raw_chunks_retained`. The manifest is also annotated.

Important consequence: once cleanup has deleted the `.bin` chunks, the run still keeps the final lossless video and summary metadata, but you can no longer reconstruct the review video from raw chunks because the raw payload is gone.

Implementation detail that matters on this rig:

- raw frames are now written on FicTrac's main tracking loop, not on the optional draw/debug queue, so debug backpressure cannot silently drop saved raw frames
- MultiBiOS treats the CSV index as the authoritative saved-frame record and ignores an incomplete trailing CSV row if shutdown interrupts the last buffered line write

Why this exists:

- shutdown is no longer blocked on native AVI finalization
- long triggered experiments can stream frames incrementally to disk
- frame-count validation now comes from the CSV index instead of inferring success from an AVI container flush

Backward compatibility:

- older runs that only contain `fictrac-vidLogFrames-*.txt` are still summarized by the Python postprocess path as a fallback

### Effective behavior after these edits

- first frame wait default in tracking path: `30000` ms
- first frame wait override: `src_first_frame_timeout_ms`
- infinite first frame wait: set `src_first_frame_timeout_ms: 0`
- subsequent frame waits: unchanged upstream logic (`max(1000, 1000 / fps)`)
- default when re-running `configGui.exe` on a config that does not already have the key: `0`

For the exact MultiBiOS validation path used in this repo, the canonical config keeps `src_first_frame_timeout_ms: 0` in `config/config_camera.txt`, and `config/hardware.yaml` can independently bound or unbound the Python-side wait with `fictrac.startup_timeout_s`.

## Exact Rebuild Procedure Used On This Workstation

This is the exact rebuild flow that produced the working packaged binary on this machine.

Validated local paths:

- MultiBiOS repo root: `C:\Rishika\MultiBiOS`
- FicTrac source checkout: `C:\Rishika\MultiBiOS\assets\third_party\FicTrac`
- FicTrac build directory: `C:\Rishika\MultiBiOS\assets\third_party\FicTrac\build`
- Packaged output directory: `C:\Rishika\MultiBiOS\assets\fictrac-spinnaker`
- vcpkg root: `C:\Users\markd\vcpkg`
- Spinnaker SDK root: `C:\Program Files\Teledyne\Spinnaker`
- generator: `Visual Studio 17 2022`

Required tools already installed on this PC:

- `cmake`
- Visual Studio Build Tools 2022
- Spinnaker SDK 4.3.x matching the runtime camera stack
- `vcpkg` with OpenCV, NLopt, and FFmpeg dependencies available through the toolchain

Configure:

```powershell
cmake -S "C:\Rishika\MultiBiOS\assets\third_party\FicTrac" `
  -B "C:\Rishika\MultiBiOS\assets\third_party\FicTrac\build" `
  -G "Visual Studio 17 2022" `
  -A x64 `
  -D CMAKE_TOOLCHAIN_FILE="C:\Users\markd\vcpkg\scripts\buildsystems\vcpkg.cmake" `
  -D PGR_USB3=ON `
  -D PGR_DIR="C:\Program Files\Teledyne\Spinnaker" `
  --fresh
```

Build:

```powershell
cmake --build "C:\Rishika\MultiBiOS\assets\third_party\FicTrac\build" `
  --config Release `
  --parallel
```

Promote the rebuilt executable into the packaged runtime directory used by MultiBiOS:

```powershell
Copy-Item "C:\Rishika\MultiBiOS\assets\third_party\FicTrac\bin\Release\fictrac.exe" `
  "C:\Rishika\MultiBiOS\assets\fictrac-spinnaker\fictrac-spinnaker.exe" -Force

Copy-Item "C:\Rishika\MultiBiOS\assets\third_party\FicTrac\bin\Release\fictrac.exe" `
  "C:\Rishika\MultiBiOS\assets\fictrac-spinnaker\fictrac.exe" -Force

Copy-Item "C:\Rishika\MultiBiOS\assets\third_party\FicTrac\bin\Release\configGui.exe" `
  "C:\Rishika\MultiBiOS\assets\fictrac-spinnaker\configGui.exe" -Force
```

The rebuild completed successfully with the current source patch set on this workstation.

The existing helper script `tools/build_fictrac_spinnaker.ps1` is still usable, but it now assumes the vendored source tree already exists and should not be reset automatically. The commands above are the exact ones that were used for this recovery.

## Helper Script Behavior

The helper script now defaults to the safe vendored-source workflow:

- it expects the patched FicTrac tree at `assets/third_party/FicTrac`
- it uses `assets/third_party/FicTrac-build` only as a local build directory
- it does not fetch or check out upstream refs unless you ask it to explicitly

Default usage:

```powershell
cd C:\Rishika\MultiBiOS

.\tools\build_fictrac_spinnaker.ps1 `
  -VcpkgRoot C:\Users\markd\vcpkg `
  -SpinnakerRoot "C:\Program Files\Teledyne\Spinnaker"
```

If you need to seed a fresh checkout into the vendored source path, do that explicitly:

```powershell
cd C:\Rishika\MultiBiOS

.\tools\build_fictrac_spinnaker.ps1 `
  -VcpkgRoot C:\Users\markd\vcpkg `
  -BootstrapClone
```

If you intentionally want to move the vendored tree to another upstream ref, do it explicitly as well:

```powershell
cd C:\Rishika\MultiBiOS

.\tools\build_fictrac_spinnaker.ps1 `
  -VcpkgRoot C:\Users\markd\vcpkg `
  -FetchUpstream `
  -CheckoutRef <tag-or-commit>
```

That separation matters because a local patched vendor tree and a local build tree are two different things. Only the source tree should carry rig-specific fixes.

## What The Build Produces

After a successful build, the packaged directory should contain at least:

- `fictrac-spinnaker.exe`
- `fictrac.exe`
- `configGui.exe`
- OpenCV / FFmpeg / NLopt runtime DLLs copied beside the executable

On this rig, the primary binary to use is:

```text
C:/Rishika/MultiBiOS/assets/fictrac-spinnaker/fictrac-spinnaker.exe
```

MultiBiOS prepares the runtime `PATH` automatically before launching FicTrac, so you do not need to manually prepend the Spinnaker DLL folder when using the MultiBiOS runners.

If the build succeeds, point `hardware.yaml -> fictrac.bin` at:

```yaml
fictrac:
  bin: "C:/Rishika/MultiBiOS/assets/fictrac-spinnaker/fictrac-spinnaker.exe"
```

## How To Use The Rebuilt Binary

Use the rebuilt binary in three layers, in order: bounded live probe, bounded runner validation, then full experiment.

## Re-running FicTrac Configuration

There is not currently a separate MultiBiOS-side codepath that reconfigures FicTrac calibration for you.

The correct reconfiguration path is still the classic upstream FicTrac UI:

```powershell
cd C:\Rishika\MultiBiOS\assets\fictrac-spinnaker

./configGui.exe C:\Rishika\MultiBiOS\config\config_camera.txt
```

For this rig, that is now safe to use as the standard reconfiguration workflow because the patched `configGui.exe` will default `src_first_frame_timeout_ms` to `0` if the key is missing and will honor the configured value when it opens the live Spinnaker camera.

If you prefer a bounded wait in the UI, set `src_first_frame_timeout_ms` to a positive millisecond value before launching `configGui.exe`.

Because the camera is normally left in external-trigger mode on this rig, the plain upstream workflow is still awkward: `configGui.exe` needs frames to already be arriving before its first `grab()` succeeds.

The easiest packaged workflow is now:

```powershell
cd C:\Rishika\MultiBiOS

./tools/run_fictrac_config_gui.ps1 `
  -Fps 30
```

That helper:

- reapplies the rig's Blackfly defaults from `config/hardware.yaml`
- prepends the required FicTrac/Spinnaker runtime DLL paths
- starts `tests/continuous_camera_trigger.py`
- launches the packaged `configGui.exe` in its own interactive console window
- stops the trigger train when the UI exits

For this rig, the relevant defaults now live in [hardware.yaml](../config/hardware.yaml) under `blackfly_defaults`:

```yaml
blackfly_defaults:
  exposure_us: 4500
  roi_width: 400
  roi_height: 400
```

That means re-running the FicTrac UI helper will first put both cameras into the same `400x400` centered ROI and `4500 us` exposure mode before calibration starts, so the resulting FicTrac ROI is calibrated against the actual cropped sensor image rather than the old full-frame geometry.

If you already have a trigger source running, pass `-NoTriggerTrain` and use the classic UI launch directly.

## Quick Start For This Rig

If you only want the shortest known-good path on this workstation, use these exact steps.

### 1. Optional: Start a trigger train for probe/config workflows

```powershell
cd C:\Rishika\MultiBiOS
conda run -n multibios-blackfly python tests\continuous_camera_trigger.py --fps 30
```

Use this when your current FicTrac camera config expects external triggers during probing or `configGui` calibration.

### 2. Probe the live camera path directly

```powershell
cd C:\Rishika\MultiBiOS
conda run -n multibios-blackfly python tests\fictrac_live_probe.py `
  --frames 5 `
  --config C:\Rishika\MultiBiOS\config\config_camera.txt `
  --fictrac-bin C:\Rishika\MultiBiOS\assets\fictrac-spinnaker\fictrac-spinnaker.exe
```

### Expected outcome

- the trigger process keeps running in terminal 1
- terminal 2 reports callback setup and received frames
- the summary ends with non-zero `frames_received`

### 3. Validate the bounded runner paths

Short hardware-timed validation:

```powershell
cd C:\Rishika\MultiBiOS
conda run -n multibios-blackfly python -m multibios.run_protocol `
  --yaml protocols/short_protocol.yaml `
  --hardware config/hardware.yaml `
  --verbose --progress
```

Expected result:

- a fresh `data/runs/<timestamp>/` directory is created
- `fictrac_driver_diagnostics.json` contains a non-null `first_packet_wall_time`
- the run exits without leaving a `fictrac-spinnaker.exe` process behind

### 4. Then run the full experiment path

After the probe succeeds, use the hardware-timed primary runner with FicTrac enabled in `hardware.yaml`:

```powershell
cd C:\Rishika\MultiBiOS
conda run -n multibios-blackfly python -m multibios.run_protocol `
  --yaml protocols/example_protocol.yaml `
  --hardware config/hardware.yaml `
  --verbose --progress
```

## Using FicTrac From MultiBiOS Runners

### `multibios.run_protocol`

Set the rig-level FicTrac paths in [config/hardware.yaml](../config/hardware.yaml):

```yaml
fictrac:
  bin: "C:/Rishika/MultiBiOS/assets/fictrac-spinnaker/fictrac-spinnaker.exe"
  config: "config_camera.txt"
  first_frame_timeout_ms: 0
  startup_timeout_s: 90.0
```

The canonical FicTrac config now lives at `config/config_camera.txt`, next to `hardware.yaml`. Reconfiguration helpers, probes, and experiment runners should all use that single file.

In this mode, MultiBiOS:

- prepares the Spinnaker runtime path
- launches FicTrac before the DO waveform starts
- waits for the first FicTrac UDP frame using `hardware.yaml -> fictrac.startup_timeout_s`
- starts the hardware-timed protocol only after FicTrac is healthy
- records DAQ outputs and FicTrac artifacts into the same run directory

### Legacy serial runner

`multibios.experiment` is deprecated. Its FicTrac bring-up notes now live in [legacy/serial_experiment_pipeline.md](legacy/serial_experiment_pipeline.md).

## Recommended Bring-Up Sequence

For this specific rig, the safest order is:

1. Confirm SpinView can see the Blackfly camera.
2. Confirm the camera is configured for external trigger mode.
3. Rebuild FicTrac with `PGR_USB3` support if the packaged binary is missing or stale.
4. If your current camera config expects triggers during probing, run `tests/continuous_camera_trigger.py` to provide a known trigger train.
5. Run `tests/fictrac_live_probe.py` against `fictrac-spinnaker.exe`.
6. Run `multibios.run_protocol` with `protocols/short_protocol.yaml`.
7. If you are maintaining the deprecated serial runner, use the bounded legacy procedure in [legacy/serial_experiment_pipeline.md](legacy/serial_experiment_pipeline.md).
8. Only then switch to a full experimental protocol.

That sequence separates build problems, camera-open problems, and trigger-path problems into distinct steps.

## Rig-Specific Notes

- The camera path that matters here is the live Blackfly side camera, not a prerecorded video source.
- SpinView recognition plus FlyCapture failure is consistent with a Spinnaker-only workflow on this hardware.
- DAQ trigger edges remain the authoritative timing reference even when FicTrac is consuming hardware-triggered frames.
- The current camera return-line path should not yet be treated as validated per-frame exposure proof.

## NI-DAQ Triggered FicTrac: What It Means

The important distinction is that **FicTrac does not become the timing master** just because it reads a hardware-triggered camera.

For this rig, the clean timing model is:

1. NI-DAQ generates `TRIG_CAMERA` pulses.
2. The Blackfly camera is armed for external `FrameStart` triggering.
3. FicTrac receives frames only when those DAQ pulses occur.
4. MultiBiOS logs the DAQ waveform and uses it as the authoritative experiment clock.

This gives you hardware-triggered imaging and deterministic frame issuance, but the truth source for alignment is still the DAQ waveform, not FicTrac's host-side arrival time.

## What Must Already Be True

Before a triggered FicTrac run will work reliably:

- The Blackfly side camera must be configured for external trigger mode.
- The trigger rate must be below the camera's real triggered limit for the chosen exposure.
- Exposure time must fit within the trigger period with readout margin.
- The DAQ task must be armed before FicTrac blocks on its first externally triggered frame.
- The FicTrac binary must tolerate a delayed first trigger pulse.

The supported runner now follows that startup requirement before protocol execution and waits up to `hardware.yaml -> fictrac.startup_timeout_s` for the first UDP frame.

For focused trigger acceptance checks, run `tests/verify_camera_trigger_path.py --arm-cameras ...`.
It now reports three things that matter for 200 Hz debugging:

- aggregate trigger acceptance on the DAQ return lines and in direct PySpin acquisition
- missing-edge classification (`exact`, `missing_internal`, or `missing_boundary` when first-vs-last cannot be proven from timing alone)
- timing budget readback from the armed camera state, including actual exposure, actual trigger delay, overlap mode, and remaining slack versus the trigger period

## What MultiBiOS Can Trust Today

What is already validated:

- NI-DAQ can issue camera trigger pulses.
- The Blackfly cameras can acquire on external trigger in the existing Blackfly tests.
- The internal MultiBiOS FicTrac client receives realtime UDP callbacks when FicTrac is healthy.
- The rebuilt `fictrac-spinnaker.exe` can run the live Blackfly path on this rig.
- The Windows child-process launch path used by both MultiBiOS runners is validated on this workstation.

What is already validated in saved run artifacts:

- bounded `multibios.run_protocol` execution with FicTrac frames written into the DAQ run directory
- deprecated serial-runner validation notes have been moved to [legacy/serial_experiment_pipeline.md](legacy/serial_experiment_pipeline.md)

What is not yet fully validated:

- Camera output return timing on the existing `O1` line wiring.

The current wiring notes indicate that the Blackfly white output line requires the blue opto ground reference, or another external bias path, before it will become a reliable DAQ-visible return.

## Recommended Synchronization Strategy

Use this hierarchy for timestamp trust:

1. **DAQ waveform edges** for experiment event timing.
2. **Camera return line edges**, once validated, for proof that each hardware trigger produced a real exposure.
3. **FicTrac frame order** for behavioral reconstruction.
4. **FicTrac host arrival timestamps** only as a convenience field, not as the primary synchronization source.

That structure is what lets you say both of these things at once without contradiction:

- the camera is hardware-triggered by NI-DAQ
- FicTrac still runs as a downstream consumer rather than the clock authority

## Immediate Next Checks After Rebuild

After rebuilding FicTrac with `PGR_USB3` support:

1. Reproduce the old failure directly with the packaged binary and a delayed/no-trigger runtime config if you need to confirm the root cause again.
2. Run the rebuilt binary against that same runtime config and confirm it stays alive past the old 1 s failure point.
3. Run `tests/fictrac_live_probe.py` against the rebuilt binary and the live camera config.
4. Run a short MultiBiOS experiment with camera triggers enabled.
5. Compare frame count against DAQ trigger count.
6. Validate a camera return line before treating per-frame exposure confirmation as solved.
