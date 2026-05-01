# Final Run Protocol Dataset Plan

Status: proposal only. This document defines the target organization for final `run_protocol` datasets. It does not change current code or on-disk outputs yet.

## Why Reorganize

The current run directory is useful for active development, but it mixes several different kinds of information at the top level:

- experiment definition
- compiled plan
- observed recordings
- derived summaries
- diagnostics and transcripts

That makes it harder to answer basic questions quickly:

- What did we intend to run?
- What did the hardware actually do?
- What was recorded?
- What can be regenerated later?
- What is primary evidence versus a convenience summary?

The final dataset layout should make those distinctions obvious from folder names alone.

## Design Goals

The final layout should satisfy four properties.

1. Human readability.
   A person who has never seen the code should be able to open the run directory, read one overview file, and understand where to find the protocol, the recorded data, the analysis products, and the debugging logs.
2. Reproducibility.
   The dataset should contain the exact inputs, resolved runtime settings, software provenance, and hardware identity needed to rerun the same experiment logic.
3. Auditability.
   The dataset should preserve primary evidence and make it obvious which artifacts are raw, which are planned, and which are derived.
4. Debuggability.
   The dataset should retain the timing anchors, driver diagnostics, integrity checks, and transcripts needed to explain failures or mismatches.

## Core Decision

Organize every run by artifact role, not by producer implementation detail.

This is the key idea. A future reader should not need to know which Python module wrote a file. They only need to know what role the file plays in the scientific record.

The top-level categories should therefore be:

- `inputs/`: exact experiment definition and provenance
- `planned/`: what software compiled and intended to output
- `recorded/`: primary observations captured during the run
- `derived/`: summaries, validation reports, and convenience products that can be regenerated
- `logs/`: verbose transcripts and diagnostics used for debugging

This separation is the strongest improvement for readability, auditability, and long-term maintenance.

## Proposed Directory Tree

```text
data/runs/<run_id>/
  README.md
  run_manifest.json
  checksums.sha256
  notes.md

  inputs/
    protocol.yaml
    hardware.yaml
    cli_args.json
    resolved_runtime.json
    source_snapshot.json
    software_environment.json
    hardware_snapshot.json

  planned/
    compile_report.json
    control_plan.csv
    timing_anchors.json
    daq/
      digital_outputs/
        channels.json
        signal_array.npz
        signal_array.meta.json
        edge_table.csv
        edge_table.meta.json
        commit_edge_table.csv
        commit_edge_table.meta.json
      analog_outputs/
        channels.json
        signal_array.npz
        signal_array.meta.json

  recorded/
    daq/
      analog_inputs/
        channels.json
        samples.npz
        samples.meta.json
      digital_inputs/
        channels.json
        samples.npz
        samples.meta.json
    cameras/
      blackfly_cam1/
        recording_manifest.json
        frames.bin
        frames.meta.json
        frame_index.csv
        frame_index.meta.json
      fictrac_camera/
        recording_manifest.json
        raw_video.avi
    tracking/
      fictrac/
        runtime_config.txt
        runtime_config.json
        session_record.json

  derived/
    validation/
      dataset_completeness.json
      timing_alignment.json
      daq_capture_summary.json
      blackfly_cam1_integrity.json
      fictrac_integrity.json
    previews/
      protocol_preview.html
    cameras/
      blackfly_cam1/
        review_video_lossless.avi
        review_video_manifest.json

  logs/
    serial/
      teensy_transcript.jsonl
    diagnostics/
      fictrac_driver_diagnostics.json
      warnings.json
      run_log.txt
```

## Top-Level Files

### `README.md`

This is the first file a human should open.

It should answer, in plain English:

- what this run was
- when it happened
- which rig and devices were used
- whether the run completed successfully
- where the primary inputs are
- where the primary recorded data are
- where the main validation reports are
- what known anomalies occurred

Why this is a good idea:

- a random reader should not have to reverse-engineer JSON to understand the dataset
- it provides a stable human entry point even if machine schemas evolve later

### `run_manifest.json`

This should be the machine-readable index for the entire dataset.

Minimum fields:

- `schema_name`
- `schema_version`
- `dataset_kind`
- `run_id`
- `run_uuid`
- `status`
- `started_utc`
- `completed_utc`
- `rig_id`
- `operator` if available
- `primary_clock`
- `timing_anchor_file`
- `artifact_index`
- `warnings`
- `missing_optional_artifacts`

Why this is a good idea:

- a single manifest gives code one stable entry point
- it avoids spreading basic discovery logic across many ad hoc filenames
- it lets incomplete or failed runs still be audited cleanly

### `checksums.sha256`

This should contain checksums for every primary and derived artifact that matters.

Why this is a good idea:

- corruption detection is part of auditability
- a 10-year-later reader can verify dataset integrity without custom tooling

### `notes.md`

This should contain freeform human notes that do not belong in structured schemas.

Examples:

- unusual rig behavior
- operator observations
- reasons a run was aborted or repeated

Why this is a good idea:

- not everything important belongs in JSON fields
- separating freeform notes from structured provenance keeps schemas clean

## Inputs

The `inputs/` folder should contain the exact definition needed to explain what experiment was requested.

### `protocol.yaml`

Exact copy of the source protocol.

Why this is a good idea:

- this is the scientific intent of the experiment
- it remains readable without any helper tools

### `hardware.yaml`

Exact copy of the canonical hardware mapping used by the run.

Why this is a good idea:

- reproducibility requires the mapping from logical device names to physical channels
- this matches the existing single-source-of-truth policy

### `cli_args.json`

Structured copy of the command-line arguments actually used.

Why this is a good idea:

- it separates invocation details from scientific inputs
- it prevents the current `meta.json` pattern from becoming a mixed bag of unrelated fields

### `resolved_runtime.json`

Fully resolved runtime settings after defaults and overrides are applied.

This should include:

- sample rate and timing values actually used
- final camera settings actually requested
- enabled subsystems
- resolved output modes

Why this is a good idea:

- exact reproducibility depends on resolved values, not only source files
- it avoids forcing future readers to replay merge logic mentally

### `source_snapshot.json`

Software provenance for the run.

This should include:

- repository identifier
- commit hash
- dirty working tree flag
- version tags if available
- entrypoint module or script

Why this is a good idea:

- code state is part of reproducibility
- a run produced from uncommitted code must be visibly marked as such

### `software_environment.json`

Resolved software environment used to execute the run.

This should include:

- Python version
- environment manager and environment name
- key package versions
- driver versions when known
- OS details

Why this is a good idea:

- true reruns depend on the actual environment, not just source code
- debugging timing or driver issues often requires exact versions

### `hardware_snapshot.json`

Resolved hardware identity and configuration snapshot for the run.

This should include:

- rig identifier
- DAQ model and serial if available
- camera models and serials
- Teensy model and firmware revision if available
- MFC models and calibration references if available
- any relevant wiring or hardware revision identifiers

Why this is a good idea:

- dataset-only reproducibility is impossible without hardware identity
- this is necessary to distinguish a protocol bug from a rig-specific issue

## Planned

The `planned/` folder should contain what the software compiled and intended the rig to do.

This is distinct from what was actually recorded.

### `compile_report.json`

The structured compiler output.

Why this is a good idea:

- it explains how the protocol was interpreted
- it preserves randomization outcomes and resolved phase sequences

### `control_plan.csv`

Human-readable logical schedule.

Why this is a good idea:

- it is the easiest file for a person to inspect when asking what should have happened and when
- it bridges the gap between YAML intent and low-level signals

### `timing_anchors.json`

Clock anchors and timing-domain definitions.

This should include:

- run start UTC timestamp
- perf counter anchor
- DAQ sample-rate definition
- clock-domain descriptions
- formulas for converting sample index to wall time and perf-counter time

Why this is a good idea:

- timing anchors are critical enough to deserve their own file
- hiding them inside generic metadata makes debugging much harder

### `planned/daq/digital_outputs/`

This folder should contain the exact digital outputs the software intended to produce.

Files:

- `channels.json`: logical name to physical channel mapping for this artifact
- `signal_array.npz`: planned digital waveform matrix
- `signal_array.meta.json`: dtype, shape, axis order, sample rate, units, and semantics
- `edge_table.csv`: all planned edges for all digital output lines
- `edge_table.meta.json`: column definitions and timing-domain description
- `commit_edge_table.csv`: planned register commit edges specifically
- `commit_edge_table.meta.json`: column definitions and semantics

Why this is a good idea:

- channel mapping belongs next to the array it describes
- sidecars make binary arrays self-explanatory without code knowledge
- separating all edges from commit-specific edges preserves both a complete audit trail and a convenient debugging summary

### `planned/daq/analog_outputs/`

This folder should contain the exact analog outputs the software intended to produce.

Files:

- `channels.json`
- `signal_array.npz`
- `signal_array.meta.json`

Why this is a good idea:

- analog outputs are primary planned artifacts, not secondary metadata
- matching the digital structure makes the dataset predictable

## Recorded

The `recorded/` folder should contain primary evidence captured during the run.

Rule: if deleting a file would destroy the ability to re-audit what happened, it belongs in `recorded/`, not `derived/`.

### `recorded/daq/analog_inputs/`

Files:

- `channels.json`
- `samples.npz`
- `samples.meta.json`

Why this is a good idea:

- these are the measured analog feedback signals
- keeping them separate from outputs prevents plan-versus-observation confusion

### `recorded/daq/digital_inputs/`

Files:

- `channels.json`
- `samples.npz`
- `samples.meta.json`

Why this is a good idea:

- digital return lines are direct evidence for synchronization and READY behavior
- the raw sampled signal is more fundamental than any extracted edge summary

### `recorded/cameras/blackfly_cam1/`

Files:

- `recording_manifest.json`
- `frames.bin`
- `frames.meta.json`
- `frame_index.csv`
- `frame_index.meta.json`

Why this is a good idea:

- the raw frame stream and frame index are the primary evidence
- the manifest should describe camera identity, ROI, gain, exposure, trigger mode, actual frame counts, and relationships to sibling artifacts
- putting camera-specific files in their own folder eliminates the current top-level filename clutter

### `recorded/cameras/fictrac_camera/`

Files:

- `recording_manifest.json`
- `raw_video.avi` or an equivalent raw recording artifact

Why this is a good idea:

- the FicTrac input camera is a real recorded stream and should live under recorded camera data
- separating it from the tracker output clarifies the difference between source video and tracking results

### `recorded/tracking/fictrac/`

Files:

- `runtime_config.txt`
- `runtime_config.json`
- `session_record.json`

This folder may also hold other raw FicTrac session outputs if they are primary evidence.

Why this is a good idea:

- tracking results are not the same thing as camera capture
- grouping them under `tracking/` makes the processing stage explicit

## Derived

The `derived/` folder should contain anything that can be regenerated from `inputs/`, `planned/`, and `recorded/`.

This is where convenience and validation products belong.

### `derived/validation/`

Recommended files:

- `dataset_completeness.json`
- `timing_alignment.json`
- `daq_capture_summary.json`
- `blackfly_cam1_integrity.json`
- `fictrac_integrity.json`

Why this is a good idea:

- validation outputs are extremely important, but they are still derived conclusions
- keeping them separate from raw evidence prevents accidental over-trust in a summary file

### `derived/previews/protocol_preview.html`

Why this is a good idea:

- preview visualizations are useful, but they are not primary evidence
- this makes it safe to delete and regenerate previews without changing the scientific record

### `derived/cameras/blackfly_cam1/`

Recommended files:

- `review_video_lossless.avi`
- `review_video_manifest.json`

Why this is a good idea:

- the lossless review video is valuable for inspection, but it is still derived from the primary raw frame stream
- classifying it as derived preserves the distinction between evidence and convenience products

## Logs

The `logs/` folder should contain verbose outputs that are useful for debugging but are not themselves the core dataset.

### `logs/serial/teensy_transcript.jsonl`

Why this is a good idea:

- the transcript is essential for firmware and control-path auditing
- it is log-like in form, but important enough to preserve permanently

### `logs/diagnostics/`

Recommended files:

- `fictrac_driver_diagnostics.json`
- `warnings.json`
- `run_log.txt`

Why this is a good idea:

- diagnostics should be easy to ignore during normal analysis and easy to find during debugging
- keeping them out of the root avoids making the dataset feel more chaotic than it is

## Naming Rules

These rules should apply to every final dataset.

1. Use relative paths only inside saved metadata.
   No artifact should store machine-specific absolute paths as its primary link to sibling files.
2. Use one artifact, one responsibility.
   Do not let a single JSON file mix CLI args, timing anchors, hardware identity, and analysis results.
3. Put arrays next to sidecars.
   Every `.npz`, `.bin`, `.csv`, or other compact artifact with non-obvious semantics should have a sibling `.meta.json` that explains shape, units, axes, and clock domain.
4. Use stable generic names inside typed folders.
   Example: use `recording_manifest.json` inside `recorded/cameras/blackfly_cam1/` rather than repeating `blackfly_cam1_` on every filename.
5. Keep raw and derived products separate.
   If a file can be regenerated, it should not live beside primary evidence unless there is a strong reason.
6. Prefer explicit role names over implementation names.
   `recorded/tracking/fictrac/` is clearer than a folder name that only makes sense if you already know the codebase.
7. Every JSON artifact should carry schema identity.
   Include at minimum `schema_name` and `schema_version` so future readers know how to interpret it.

## Required Metadata Conventions

Every structured artifact should define its timing and units explicitly.

Minimum conventions:

- use UTC timestamps with timezone or an unambiguous UTC marker
- store the time basis for any index-based data
- declare units for every numeric column that is not dimensionless
- declare axis order for arrays
- declare whether a file is `primary`, `derived`, or `log`

This is necessary because a file can be preserved perfectly and still be unusable if its units and time basis are unclear.

## Incomplete Or Failed Runs

The final layout should support partial runs cleanly.

Rules:

- keep the same directory structure even when a subsystem fails
- omit missing artifacts rather than writing misleading placeholders
- record the absence and reason in `run_manifest.json`
- keep logs and diagnostics even for failed runs
- never delete already captured primary evidence just because later post-processing failed

Why this is a good idea:

- failure cases are often the most important datasets for debugging
- a partial run should still be auditable without guesswork

## Mapping From Current Outputs To Proposed Layout

| Current artifact | Proposed location | Classification | Reason |
| --- | --- | --- | --- |
| `meta.json` | split across `run_manifest.json`, `inputs/cli_args.json`, `inputs/resolved_runtime.json`, and `planned/timing_anchors.json` | mixed today, separated in final layout | current file mixes unrelated concerns |
| `protocol.yaml` | `inputs/protocol.yaml` | input | exact experiment definition |
| `hardware.yaml` | `inputs/hardware.yaml` | input | exact hardware mapping |
| `compile_report.json` | `planned/compile_report.json` | planned | compiler interpretation of the protocol |
| `control_plan.csv` | `planned/control_plan.csv` | planned | human-readable logical schedule |
| `compiled_do.npz` | `planned/daq/digital_outputs/signal_array.npz` | planned | exact digital waveform intent |
| `do_map.json` | `planned/daq/digital_outputs/channels.json` | planned | mapping for the waveform artifact |
| `digital_edges.csv` | `planned/daq/digital_outputs/edge_table.csv` | planned | derived from planned digital outputs |
| `rck_edges.csv` | `planned/daq/digital_outputs/commit_edge_table.csv` | planned | specialized planned edge summary |
| `compiled_ao.npz` | `planned/daq/analog_outputs/signal_array.npz` | planned | exact analog waveform intent |
| `ao_map.json` | `planned/daq/analog_outputs/channels.json` | planned | mapping for analog outputs |
| `capture_ai.npz` | `recorded/daq/analog_inputs/samples.npz` | recorded | measured analog feedback |
| `capture_di.npz` | `recorded/daq/digital_inputs/samples.npz` | recorded | measured digital return lines |
| `di_map.json` | `recorded/daq/digital_inputs/channels.json` | recorded | mapping for recorded DI |
| `di_edges.csv` | `derived/validation/daq_capture_summary.json` or a dedicated DI edge summary | derived | useful, but computable from recorded DI |
| `blackfly_recording.json` | `recorded/cameras/blackfly_cam1/recording_manifest.json` | recorded | camera capture manifest |
| `blackfly_cam1_manifest.json` | merge into `recording_manifest.json` or keep as a sibling manifest if roles differ | recorded | avoid duplicative manifest files without clear role boundaries |
| `blackfly_cam1_frame_index.csv` | `recorded/cameras/blackfly_cam1/frame_index.csv` | recorded | primary frame timing evidence |
| `blackfly_cam1_analysis.json` | `derived/validation/blackfly_cam1_integrity.json` | derived | analysis of camera capture quality |
| `blackfly_cam1_lossless.avi` | `derived/cameras/blackfly_cam1/review_video_lossless.avi` | derived | convenient review product, not primary evidence |
| `fictrac_runtime_config.txt` | `recorded/tracking/fictrac/runtime_config.txt` | recorded | exact runtime config used by tracker |
| `fictrac_runtime.json` | `recorded/tracking/fictrac/runtime_config.json` | recorded | structured runtime config summary |
| `fictrac_camera_recording.json` | `recorded/tracking/fictrac/session_record.json` or split between tracking and camera recorded folders | recorded | current name is ambiguous about whether it is input video or tracker output |
| `fictrac_driver_diagnostics.json` | `logs/diagnostics/fictrac_driver_diagnostics.json` | log | subsystem diagnostics |
| `teensy_serial_transcript.jsonl` | `logs/serial/teensy_transcript.jsonl` | log | primary debug transcript |
| `preview.html` | `derived/previews/protocol_preview.html` | derived | regenerated convenience product |

## Recommendation On Paths

All artifact references written into JSON should be relative to the run root and should use one normalized path convention.

Recommendation:

- store run-relative paths only
- avoid embedding the run root repeatedly inside child metadata
- normalize to forward slashes inside JSON even on Windows

Why this is a good idea:

- relative paths are portable across machines and archival locations
- they make it easier to move or share a run directory intact

## Recommendation On Schema Boundaries

Do not keep large umbrella files if they span multiple concerns.

Specifically:

- replace `meta.json` with smaller role-specific files
- avoid camera JSON that combines configuration, integrity analysis, and derived video metadata unless the file explicitly declares those sections and why they coexist
- keep analysis summaries out of primary recording manifests when possible

Why this is a good idea:

- smaller schemas are easier to document, validate, and preserve
- it reduces accidental coupling between recording code and post-processing code

## Minimum Completeness Standard

A final run dataset should not be considered complete unless all of the following are present or explicitly marked unavailable:

- human overview file
- machine manifest
- checksums
- exact protocol copy
- exact hardware mapping copy
- resolved runtime settings
- source snapshot
- software environment snapshot
- hardware snapshot
- compiled planned outputs
- primary recorded outputs
- timing anchors
- validation summaries
- debugging logs and diagnostics for enabled subsystems

This is the minimum standard for claiming readability, reproducibility, and auditability.

## Recommended Next Step After Approval

Implementation should be done in two phases.

Phase 1:

- add the new folder structure and manifest writing
- continue writing current legacy filenames in parallel for compatibility

Phase 2:

- migrate readers and docs to the new structure
- remove redundant legacy outputs only after validation and tooling updates are complete

This staged approach is the safest path because it improves organization without breaking existing analysis scripts immediately.