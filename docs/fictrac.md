# FicTrac Integration

This page covers the MultiBiOS-specific path for running FicTrac on this rig, including the current Blackfly camera constraint, how to build a Spinnaker-enabled FicTrac on Windows, and how NI-DAQ-triggered acquisition should be interpreted.

For the internal client architecture, full-state storage model, closed-loop consumer API, and deprecation criteria, see [fictrac_client.md](fictrac_client.md).

## Current State On This Rig

- The packaged Spinnaker build at `C:/Rishika/MultiBiOS/assets/fictrac-spinnaker/fictrac-spinnaker.exe` is the validated binary for this workstation.
- The in-repo MultiBiOS FicTrac client is receiving realtime callback frames from the live Blackfly side camera.
- `multibios.run_protocol` has been validated against live FicTrac on this rig.
- MultiBiOS now sanitizes the Windows child-process environment before launching FicTrac, which avoids the conda/Python DLL path conflict that previously caused native startup crashes.

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

The build helper no longer silently fetches upstream and checks out `master`. That behavior was unsafe once local patches became part of the rig setup.

## First-Frame Trigger Failure On This Rig

The April 2026 blocker on this workstation was not the MultiBiOS-side `fictrac_timeout_s` setting. It was the upstream Spinnaker first-frame wait inside FicTrac itself.

The confirmed failure signature was:

- protocol path: `config/odor_lateralization.yaml`
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

### Effective behavior after these edits

- first frame wait default in tracking path: `30000` ms
- first frame wait override: `src_first_frame_timeout_ms`
- infinite first frame wait: set `src_first_frame_timeout_ms: 0`
- subsequent frame waits: unchanged upstream logic (`max(1000, 1000 / fps)`)
- default when re-running `configGui.exe` on a config that does not already have the key: `0`

For the exact MultiBiOS validation path used in this repo, set `fictrac.startup_timeout_s: 0` in `config/hardware.yaml` if you want the Python runner to wait indefinitely for the first UDP frame.

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
  --yaml config/short_protocol.yaml `
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
  --yaml config/example_protocol.yaml `
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
6. Run `multibios.run_protocol` with `config/short_protocol.yaml`.
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
