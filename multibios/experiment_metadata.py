from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "1.0.0"
UI_VERSION = "multibios-metadata-form/1.0.0"
HISTORY_SCHEMA_NAME = "multibios.metadata_history"
HISTORY_RUN_LOG_SCHEMA_NAME = "multibios.metadata_history_run_log"

REQUIRED_PRE_FIELDS = [
    "pre_experiment.experiment_date",
    "pre_experiment.source_filename",
    "pre_experiment.fly_id",
    "pre_experiment.species",
    "pre_experiment.genotype",
    "pre_experiment.hemisphere",
    "pre_experiment.stimulus_modality",
    "pre_experiment.protocol_name",
    "pre_experiment.rig_id",
]

REQUIRED_POST_FIELDS = [
    "post_experiment.response",
    "post_experiment.completion_status",
    "post_experiment.aborted",
]

HISTORY_ENABLED_FIELDS = [
    "pre_experiment.species",
    "pre_experiment.genotype",
    "pre_experiment.hemisphere",
    "pre_experiment.age.unit",
    "pre_experiment.starvation.unit",
    "pre_experiment.volumetric",
    "pre_experiment.stimulus_modality",
    "pre_experiment.operator",
    "post_experiment.response",
    "post_experiment.exclusion_reason",
]


def default_metadata_history_path() -> Path:
    return Path(__file__).resolve().parent.parent / "data" / "metadata_history_log.json"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def build_experiment_record_meta_payload() -> dict[str, Any]:
    field_definitions = [
        _field("pre_experiment.experiment_date", "Experiment date", "date", True, False, None, "system", "Date of the experiment."),
        _field("pre_experiment.source_filename", "Source filename", "string", True, False, None, "system", "Protocol source filename."),
        _field("pre_experiment.expected_imaging_periods", "Expected imaging periods", "integer", False, False, None, "system", "Estimated number of microscope imaging periods in the compiled protocol."),
        _field("pre_experiment.fly_id", "Fly ID", "integer", True, False, None, "user", "Per-day fly identifier that resets each experiment day."),
        _field("pre_experiment.species", "Species", "string", True, False, None, "user", "Biological species."),
        _field("pre_experiment.genotype", "Genotype", "string", True, False, None, "user", "Genotype or line name."),
        _field("pre_experiment.hemisphere", "Hemisphere", "enum", True, False, None, "user", "Target hemisphere."),
        _field("pre_experiment.age.value", "Age value", "number", False, True, None, "user", "Age numeric value."),
        _field("pre_experiment.age.unit", "Age unit", "enum", False, False, None, "user", "Age unit."),
        _field("pre_experiment.starvation.value", "Starvation value", "number", False, True, None, "user", "Starvation duration numeric value."),
        _field("pre_experiment.starvation.unit", "Starvation unit", "enum", False, False, None, "user", "Starvation duration unit."),
        _field("pre_experiment.volumetric", "Volumetric", "enum", False, False, None, "user", "Whether acquisition is volumetric."),
        _field("pre_experiment.stimulus_modality", "Stimulus modality", "string", True, False, None, "user", "Stimulus modality label."),
        _field("pre_experiment.rig_temperature_c", "Rig temperature", "number", False, True, "C", "user", "Rig temperature in Celsius."),
        _field("pre_experiment.humidity_percent", "Humidity", "number", False, True, "%", "user", "Rig humidity percentage."),
        _field("pre_experiment.protocol_name", "Protocol name", "string", True, False, None, "system", "Resolved protocol name."),
        _field("pre_experiment.protocol_version", "Protocol version", "string", False, True, None, "mixed", "Resolved protocol version."),
        _field("pre_experiment.rig_id", "Rig ID", "string", True, False, None, "system", "Resolved rig identifier."),
        _field("pre_experiment.operator", "Operator", "string", False, True, None, "user", "Operator name or initials."),
        _field("post_experiment.response", "Response", "string", True, True, None, "user", "Observed response label."),
        _field("post_experiment.notes", "Notes", "string", False, True, None, "user", "Free-text notes."),
        _field("post_experiment.duration_s", "Duration", "number", False, True, "s", "system", "Observed run duration in seconds."),
        _field("post_experiment.completion_status", "Completion status", "enum", True, True, None, "mixed", "Outcome classification."),
        _field("post_experiment.aborted", "Aborted", "boolean", True, False, None, "mixed", "Whether the run aborted early."),
        _field("post_experiment.exclusion_reason", "Exclusion reason", "string", False, True, None, "user", "Reason for exclusion."),
        _field("post_experiment.imaging_dataset_source_path", "Imaging dataset source path", "string", False, True, None, "user", "Original PrairieView dataset path selected after the run."),
        _field("post_experiment.imaging_dataset_relative_path", "Imaging dataset copied path", "string", False, False, None, "system", "Run-relative destination for the copied PrairieView dataset."),
        _field("post_experiment.imaging_acquisition_type", "Imaging acquisition type", "enum", False, False, None, "user", "Microscopy acquisition type for the selected PrairieView dataset."),
        _field("post_experiment.imaging_num_rois", "Imaging ROI count", "integer", False, False, None, "user", "Number of microscope ROIs in the selected dataset."),
        _field("post_experiment.imaging_num_channels", "Imaging channel count", "integer", False, False, None, "user", "Number of recorded channels in the selected dataset."),
        _field("post_experiment.imaging_num_planes", "Imaging plane count", "integer", False, False, None, "user", "Number of planes in the selected dataset when the acquisition is volumetric."),
        _field("post_experiment.observed_anomalies", "Observed anomalies", "array", False, False, None, "user", "Observed anomalies list."),
        _field("post_experiment.quality_flags", "Quality flags", "array", False, False, None, "user", "Quality or review flags."),
    ]
    return {
        "schema_name": "multibios.experiment_record_meta",
        "schema_version": SCHEMA_VERSION,
        "record_schema_version": SCHEMA_VERSION,
        "field_definitions": field_definitions,
        "controlled_vocabularies": {
            "pre_experiment.hemisphere": ["left", "right", "bilateral", "unknown", "na"],
            "pre_experiment.age.unit": ["hours", "days", "weeks", "unknown"],
            "pre_experiment.starvation.unit": ["hours", "days", "weeks", "unknown"],
            "pre_experiment.volumetric": ["yes", "no", "unknown"],
            "post_experiment.imaging_acquisition_type": ["single_plane", "volumetric"],
            "post_experiment.completion_status": ["completed", "completed_with_issue", "aborted", "failed", "excluded"],
        },
        "required_fields": {
            "pre_experiment": REQUIRED_PRE_FIELDS,
            "post_experiment": REQUIRED_POST_FIELDS,
            "always": [],
        },
        "auto_filled_fields": [
            "pre_experiment.experiment_date",
            "pre_experiment.source_filename",
            "pre_experiment.protocol_name",
            "pre_experiment.protocol_version",
            "pre_experiment.rig_id",
            "post_experiment.duration_s",
        ],
        "history_enabled_fields": HISTORY_ENABLED_FIELDS,
    }


def default_history_store() -> dict[str, Any]:
    return {
        "schema_name": HISTORY_SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "fields": {},
        "daily_fly_ids": {},
        "daily_pre_defaults": {},
    }


def default_history_run_log_store() -> dict[str, Any]:
    return {
        "schema_name": HISTORY_RUN_LOG_SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "entries": [],
    }


def load_metadata_history(path: Path) -> dict[str, Any]:
    if not path.exists():
        return default_history_store()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_name") == HISTORY_RUN_LOG_SCHEMA_NAME:
        return _derive_history_from_run_log(payload)

    history = default_history_store()
    if isinstance(payload.get("fields"), dict):
        history["fields"] = payload["fields"]
    if isinstance(payload.get("daily_fly_ids"), dict):
        history["daily_fly_ids"] = payload["daily_fly_ids"]
    if isinstance(payload.get("daily_pre_defaults"), dict):
        history["daily_pre_defaults"] = payload["daily_pre_defaults"]
    return history


def load_experiment_record(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def save_metadata_history(path: Path, history: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(history, indent=2), encoding="utf-8")


def append_metadata_history_log_entry(path: Path, *, record: dict[str, Any], stage: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("schema_name") != HISTORY_RUN_LOG_SCHEMA_NAME:
            payload = default_history_run_log_store()
    else:
        payload = default_history_run_log_store()

    payload.setdefault("entries", [])
    payload["entries"].append(
        {
            "recorded_utc": utc_now_iso(),
            "stage": stage,
            "run_id": record.get("run_id"),
            "record": deepcopy(record),
        }
    )
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def persist_metadata_history_source(path: Path, history: dict[str, Any], *, record: dict[str, Any], stage: str) -> None:
    if _should_use_history_run_log(path):
        append_metadata_history_log_entry(path, record=record, stage=stage)
        return
    save_metadata_history(path, history)


def update_metadata_history(
    history: dict[str, Any],
    *,
    field_path: str,
    value: str | None,
    max_entries: int = 20,
) -> dict[str, Any]:
    updated = deepcopy(history)
    updated.setdefault("fields", {})
    if value is None:
        return updated
    normalized = str(value).strip()
    if not normalized:
        return updated
    field_entry = updated["fields"].setdefault(field_path, {"values": [], "updated_utc": None, "max_entries": max_entries})
    values = [item for item in field_entry.get("values", []) if item != normalized]
    field_entry["values"] = [normalized, *values][: max_entries or 20]
    field_entry["updated_utc"] = utc_now_iso()
    field_entry["max_entries"] = max_entries
    return updated


def recent_history_value(history: dict[str, Any], field_path: str) -> str | None:
    values = history.get("fields", {}).get(field_path, {}).get("values", [])
    if not values:
        return None
    value = values[0]
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None


def suggested_fly_id(history: dict[str, Any], *, experiment_date: str, same_fly: bool) -> int:
    last_fly_id = _last_fly_id_for_date(history, experiment_date)
    if same_fly and last_fly_id is not None:
        return last_fly_id
    return (last_fly_id or 0) + 1


def update_daily_fly_id(history: dict[str, Any], *, experiment_date: str, fly_id: int | None) -> dict[str, Any]:
    updated = deepcopy(history)
    updated.setdefault("daily_fly_ids", {})
    if fly_id is None:
        return updated
    normalized_fly_id = int(fly_id)
    existing = updated["daily_fly_ids"].get(experiment_date) or {}
    seen_fly_ids = [int(value) for value in existing.get("seen_fly_ids", []) if value is not None]
    if normalized_fly_id not in seen_fly_ids:
        seen_fly_ids.append(normalized_fly_id)
    updated["daily_fly_ids"][experiment_date] = {
        "last_fly_id": normalized_fly_id,
        "seen_fly_ids": seen_fly_ids,
        "updated_utc": utc_now_iso(),
    }
    return updated


def last_fly_id_for_date(history: dict[str, Any], experiment_date: str) -> int | None:
    return _last_fly_id_for_date(history, experiment_date)


def previous_fly_ids_for_date(history: dict[str, Any], experiment_date: str) -> list[int]:
    daily_entry = history.get("daily_fly_ids", {}).get(experiment_date)
    if not isinstance(daily_entry, dict):
        return []
    seen_fly_ids = [int(value) for value in daily_entry.get("seen_fly_ids", []) if value is not None]
    last_fly_id = daily_entry.get("last_fly_id")
    filtered = [value for value in seen_fly_ids if value != last_fly_id]
    return list(reversed(filtered))


def daily_pre_defaults_for_date(history: dict[str, Any], experiment_date: str) -> dict[str, Any]:
    defaults = history.get("daily_pre_defaults", {}).get(experiment_date)
    if not isinstance(defaults, dict):
        return {}
    return deepcopy(defaults.get("values", {}))


def update_daily_pre_defaults(history: dict[str, Any], *, experiment_date: str, values: dict[str, Any]) -> dict[str, Any]:
    updated = deepcopy(history)
    updated.setdefault("daily_pre_defaults", {})
    normalized_values = {key: value for key, value in values.items() if value is not None and value != ""}
    updated["daily_pre_defaults"][experiment_date] = {
        "values": normalized_values,
        "updated_utc": utc_now_iso(),
    }
    return updated


def validate_record_for_stage(record: dict[str, Any], *, stage: str) -> list[str]:
    return [f"Missing required field: {field_path}" for field_path in missing_required_fields_for_stage(record, stage=stage)]


def missing_required_fields_for_stage(record: dict[str, Any], *, stage: str) -> list[str]:
    missing: list[str] = []
    required_paths = REQUIRED_PRE_FIELDS if stage == "pre" else REQUIRED_POST_FIELDS
    for field_path in required_paths:
        value = _get_path(record, field_path)
        if value is None or (isinstance(value, str) and not value.strip()):
            missing.append(field_path)

    completion_status = _get_path(record, "post_experiment.completion_status")
    aborted = bool(_get_path(record, "post_experiment.aborted"))
    exclusion_reason = _get_path(record, "post_experiment.exclusion_reason")
    expected_imaging_periods = _coerce_int_like(_get_path(record, "pre_experiment.expected_imaging_periods")) or 0
    imaging_dataset_relative_path = _get_path(record, "post_experiment.imaging_dataset_relative_path")
    imaging_acquisition_type = _get_path(record, "post_experiment.imaging_acquisition_type")
    imaging_num_rois = _coerce_int_like(_get_path(record, "post_experiment.imaging_num_rois"))
    imaging_num_channels = _coerce_int_like(_get_path(record, "post_experiment.imaging_num_channels"))
    imaging_num_planes = _coerce_int_like(_get_path(record, "post_experiment.imaging_num_planes"))
    if stage == "post" and completion_status == "excluded" and (exclusion_reason is None or not str(exclusion_reason).strip()):
        missing.append("post_experiment.exclusion_reason")
    if (
        stage == "post"
        and expected_imaging_periods > 0
        and not aborted
        and completion_status not in {"aborted", "failed"}
        and (imaging_dataset_relative_path is None or not str(imaging_dataset_relative_path).strip())
    ):
        missing.append("post_experiment.imaging_dataset_relative_path")
    if stage == "post" and expected_imaging_periods > 0 and not aborted and completion_status not in {"aborted", "failed"}:
        if imaging_acquisition_type is None or not str(imaging_acquisition_type).strip():
            missing.append("post_experiment.imaging_acquisition_type")
        if imaging_num_rois is None:
            missing.append("post_experiment.imaging_num_rois")
        if imaging_num_channels is None:
            missing.append("post_experiment.imaging_num_channels")
        if imaging_acquisition_type == "volumetric" and imaging_num_planes is None:
            missing.append("post_experiment.imaging_num_planes")
    return missing


def apply_pre_experiment_updates(
    record: dict[str, Any],
    *,
    updates: dict[str, Any],
    entered_by: str | None,
    timestamp_utc: str | None = None,
    ui_version: str = UI_VERSION,
) -> dict[str, Any]:
    updated = deepcopy(record)
    now = timestamp_utc or utc_now_iso()
    for key, value in updates.items():
        if key.startswith("pre_experiment."):
            _set_path(updated, key, value)
    updated["entered_by"] = entered_by
    updated["entered_started_utc"] = updated.get("entered_started_utc") or now
    updated["entered_completed_utc"] = now
    updated["ui_version"] = ui_version
    updated["record_status"] = "completed_pre"
    return updated


def apply_post_experiment_updates(
    record: dict[str, Any],
    *,
    updates: dict[str, Any],
    entered_by: str | None,
    timestamp_utc: str | None = None,
    ui_version: str = UI_VERSION,
) -> dict[str, Any]:
    updated = deepcopy(record)
    now = timestamp_utc or utc_now_iso()
    for key, value in updates.items():
        if key.startswith("post_experiment."):
            _set_path(updated, key, value)
    updated["entered_by"] = entered_by
    updated["entered_started_utc"] = updated.get("entered_started_utc") or now
    updated["entered_completed_utc"] = now
    updated["ui_version"] = ui_version
    updated["record_status"] = "completed_post"
    return updated


def apply_post_run_defaults(
    record: dict[str, Any],
    *,
    duration_s: float | None,
    aborted: bool,
    completion_status: str | None = None,
) -> dict[str, Any]:
    updated = deepcopy(record)
    post_experiment = updated.setdefault("post_experiment", {})
    post_experiment["duration_s"] = None if duration_s is None else round(float(duration_s), 3)
    post_experiment["aborted"] = bool(aborted)
    resolved_status = completion_status or ("aborted" if aborted else "completed")
    if not post_experiment.get("completion_status"):
        post_experiment["completion_status"] = resolved_status
    current_status = str(updated.get("record_status") or "")
    if current_status in {"draft_pre", "completed_pre", "draft_post"}:
        updated["record_status"] = "draft_post"
    return updated


def summarize_metadata_status(record: dict[str, Any] | None) -> dict[str, Any]:
    if not record:
        return {
            "record_present": False,
            "record_status": None,
            "pre_experiment_complete": False,
            "post_experiment_complete": False,
            "metadata_complete": False,
            "operator_recorded": False,
            "entered_by": None,
            "entered_started_utc": None,
            "entered_completed_utc": None,
            "ui_version": None,
            "missing_required_fields": {
                "pre_experiment": REQUIRED_PRE_FIELDS,
                "post_experiment": REQUIRED_POST_FIELDS,
            },
        }

    missing_pre = missing_required_fields_for_stage(record, stage="pre")
    missing_post = missing_required_fields_for_stage(record, stage="post")
    pre_complete = not missing_pre
    post_complete = not missing_post
    operator = record.get("entered_by") or _get_path(record, "pre_experiment.operator")
    return {
        "record_present": True,
        "record_status": record.get("record_status"),
        "pre_experiment_complete": pre_complete,
        "post_experiment_complete": post_complete,
        "metadata_complete": pre_complete and post_complete,
        "operator_recorded": bool(operator and str(operator).strip()),
        "entered_by": operator,
        "entered_started_utc": record.get("entered_started_utc"),
        "entered_completed_utc": record.get("entered_completed_utc"),
        "ui_version": record.get("ui_version"),
        "missing_required_fields": {
            "pre_experiment": missing_pre,
            "post_experiment": missing_post,
        },
    }


def write_experiment_metadata_files(
    *,
    record_path: Path,
    record: dict[str, Any],
    record_meta_path: Path,
) -> None:
    record_path.parent.mkdir(parents=True, exist_ok=True)
    record_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    record_meta_path.write_text(json.dumps(build_experiment_record_meta_payload(), indent=2), encoding="utf-8")


def _should_use_history_run_log(path: Path) -> bool:
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload.get("schema_name") == HISTORY_RUN_LOG_SCHEMA_NAME
    return path.resolve() == default_metadata_history_path().resolve()


def _derive_history_from_run_log(payload: dict[str, Any]) -> dict[str, Any]:
    history = default_history_store()
    entries = payload.get("entries", [])
    if not isinstance(entries, list):
        return history

    for entry in entries:
        if not isinstance(entry, dict):
            continue
        record = entry.get("record")
        if not isinstance(record, dict):
            continue
        history = _apply_record_to_history(history, record)
    return history


def _apply_record_to_history(history: dict[str, Any], record: dict[str, Any]) -> dict[str, Any]:
    updated = deepcopy(history)
    for field_path in HISTORY_ENABLED_FIELDS:
        value = _get_path(record, field_path)
        updated = update_metadata_history(updated, field_path=field_path, value=None if value is None else str(value))

    pre = record.get("pre_experiment") or {}
    experiment_date = str(pre.get("experiment_date") or "")
    fly_id = _coerce_int_like(pre.get("fly_id"))
    if experiment_date and fly_id is not None:
        updated = update_daily_fly_id(updated, experiment_date=experiment_date, fly_id=fly_id)
        updated = update_daily_pre_defaults(updated, experiment_date=experiment_date, values=_pre_defaults_from_record(record))
    return updated


def _pre_defaults_from_record(record: dict[str, Any]) -> dict[str, Any]:
    values = {
        "pre_experiment.operator": _get_path(record, "pre_experiment.operator") or record.get("entered_by"),
        "pre_experiment.species": _get_path(record, "pre_experiment.species"),
        "pre_experiment.genotype": _get_path(record, "pre_experiment.genotype"),
        "pre_experiment.hemisphere": _get_path(record, "pre_experiment.hemisphere"),
        "pre_experiment.age.value": _get_path(record, "pre_experiment.age.value"),
        "pre_experiment.age.unit": _get_path(record, "pre_experiment.age.unit"),
        "pre_experiment.starvation.value": _get_path(record, "pre_experiment.starvation.value"),
        "pre_experiment.starvation.unit": _get_path(record, "pre_experiment.starvation.unit"),
        "pre_experiment.volumetric": _get_path(record, "pre_experiment.volumetric"),
        "pre_experiment.stimulus_modality": _get_path(record, "pre_experiment.stimulus_modality"),
        "pre_experiment.rig_temperature_c": _get_path(record, "pre_experiment.rig_temperature_c"),
        "pre_experiment.humidity_percent": _get_path(record, "pre_experiment.humidity_percent"),
    }
    return {
        key: value
        for key, value in values.items()
        if value is not None and value != "" and str(value).strip().lower() != "unknown"
    }


def _coerce_int_like(value: Any) -> int | None:
    if value is None or value == "":
        return None
    return int(value)


def _field(field_path: str, label: str, field_type: str, required: bool, nullable: bool, unit: str | None, source: str, help_text: str) -> dict[str, Any]:
    return {
        "field_path": field_path,
        "label": label,
        "type": field_type,
        "required": required,
        "nullable": nullable,
        "unit": unit,
        "source": source,
        "help_text": help_text,
    }


def _last_fly_id_for_date(history: dict[str, Any], experiment_date: str) -> int | None:
    daily_entry = history.get("daily_fly_ids", {}).get(experiment_date)
    if not isinstance(daily_entry, dict):
        return None
    value = daily_entry.get("last_fly_id")
    if value is None:
        return None
    return int(value)


def _get_path(obj: dict[str, Any], path: str) -> Any:
    current: Any = obj
    for key in path.split("."):
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def _set_path(obj: dict[str, Any], path: str, value: Any) -> None:
    current: dict[str, Any] = obj
    keys = path.split(".")
    for key in keys[:-1]:
        next_value = current.get(key)
        if not isinstance(next_value, dict):
            next_value = {}
            current[key] = next_value
        current = next_value
    current[keys[-1]] = value