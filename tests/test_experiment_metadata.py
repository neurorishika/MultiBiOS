from __future__ import annotations

import json
from pathlib import Path

from multibios.experiment_metadata import (apply_post_experiment_updates,
                                           apply_post_run_defaults,
                                           apply_pre_experiment_updates,
                                           append_metadata_history_log_entry,
                                           build_experiment_record_meta_payload,
                                           daily_pre_defaults_for_date,
                                           default_metadata_history_path,
                                           default_history_store,
                                           load_metadata_history,
                                           previous_fly_ids_for_date,
                                           recent_history_value,
                                           save_metadata_history,
                                           suggested_fly_id,
                                           summarize_metadata_status,
                                           update_daily_pre_defaults,
                                           update_daily_fly_id,
                                           update_metadata_history,
                                           validate_record_for_stage,
                                           write_experiment_metadata_files)
from multibios.run_dataset import build_placeholder_experiment_record


def _placeholder_record() -> dict:
    return build_placeholder_experiment_record(
        run_id="2026-05-03_16-49-37",
        run_uuid="1234",
        source_filename="short_protocol.yaml",
        protocol_name="Short Protocol",
        protocol_version="1.0",
        rig_id="Dev1",
        operator=None,
    )


def test_build_experiment_record_meta_payload_lists_required_and_history_fields() -> None:
    payload = build_experiment_record_meta_payload()

    assert payload["schema_name"] == "multibios.experiment_record_meta"
    assert "pre_experiment.fly_id" in payload["required_fields"]["pre_experiment"]
    assert "post_experiment.response" in payload["required_fields"]["post_experiment"]
    assert "pre_experiment.starvation.unit" in payload["history_enabled_fields"]


def test_metadata_history_load_save_and_recency_order(tmp_path: Path) -> None:
    history_path = tmp_path / "history.json"
    history = default_history_store()
    history = update_metadata_history(history, field_path="pre_experiment.species", value="dmel")
    history = update_metadata_history(history, field_path="pre_experiment.species", value="dsim")
    history = update_metadata_history(history, field_path="pre_experiment.species", value="dmel")

    save_metadata_history(history_path, history)
    loaded = load_metadata_history(history_path)

    assert loaded["fields"]["pre_experiment.species"]["values"] == ["dmel", "dsim"]
    assert recent_history_value(loaded, "pre_experiment.species") == "dmel"


def test_default_metadata_history_path_uses_runs_adjacent_data_log() -> None:
    path = default_metadata_history_path()

    assert path.name == "metadata_history_log.json"
    assert path.parent.name == "data"


def test_load_metadata_history_derives_history_from_run_log(tmp_path: Path) -> None:
    history_path = tmp_path / "metadata_history_log.json"
    record = apply_pre_experiment_updates(
        _placeholder_record(),
        updates={
            "pre_experiment.fly_id": 2,
            "pre_experiment.species": "dmel",
            "pre_experiment.genotype": "MB247B",
            "pre_experiment.hemisphere": "left",
            "pre_experiment.age.value": 4,
            "pre_experiment.age.unit": "days",
            "pre_experiment.starvation.value": 20,
            "pre_experiment.starvation.unit": "hours",
            "pre_experiment.stimulus_modality": "odor",
            "pre_experiment.rig_temperature_c": 24.5,
            "pre_experiment.humidity_percent": 52.0,
            "pre_experiment.operator": "rm",
        },
        entered_by="rm",
        timestamp_utc="2026-05-04T12:00:00Z",
    )

    append_metadata_history_log_entry(history_path, record=record, stage="pre")
    loaded = load_metadata_history(history_path)

    assert recent_history_value(loaded, "pre_experiment.species") == "dmel"
    assert previous_fly_ids_for_date(loaded, "2026-05-03") == []
    assert daily_pre_defaults_for_date(loaded, "2026-05-03") == {
        "pre_experiment.operator": "rm",
        "pre_experiment.species": "dmel",
        "pre_experiment.genotype": "MB247B",
        "pre_experiment.hemisphere": "left",
        "pre_experiment.age.value": 4,
        "pre_experiment.age.unit": "days",
        "pre_experiment.starvation.value": 20,
        "pre_experiment.starvation.unit": "hours",
        "pre_experiment.stimulus_modality": "odor",
        "pre_experiment.rig_temperature_c": 24.5,
        "pre_experiment.humidity_percent": 52.0,
    }


def test_daily_fly_id_tracking_reuses_or_increments_by_date() -> None:
    history = default_history_store()

    assert suggested_fly_id(history, experiment_date="2026-05-04", same_fly=False) == 1

    history = update_daily_fly_id(history, experiment_date="2026-05-04", fly_id=1)
    history = update_daily_fly_id(history, experiment_date="2026-05-04", fly_id=2)
    history = update_daily_fly_id(history, experiment_date="2026-05-04", fly_id=4)

    assert suggested_fly_id(history, experiment_date="2026-05-04", same_fly=True) == 4
    assert suggested_fly_id(history, experiment_date="2026-05-04", same_fly=False) == 5
    assert previous_fly_ids_for_date(history, "2026-05-04") == [2, 1]
    assert suggested_fly_id(history, experiment_date="2026-05-05", same_fly=False) == 1


def test_daily_pre_defaults_store_same_day_values() -> None:
    history = default_history_store()

    history = update_daily_pre_defaults(
        history,
        experiment_date="2026-05-04",
        values={
            "pre_experiment.age.value": 5,
            "pre_experiment.age.unit": "days",
            "pre_experiment.rig_temperature_c": 24.5,
            "pre_experiment.humidity_percent": 52.0,
        },
    )

    assert daily_pre_defaults_for_date(history, "2026-05-04") == {
        "pre_experiment.age.value": 5,
        "pre_experiment.age.unit": "days",
        "pre_experiment.rig_temperature_c": 24.5,
        "pre_experiment.humidity_percent": 52.0,
    }
    assert daily_pre_defaults_for_date(history, "2026-05-05") == {}


def test_apply_pre_updates_marks_record_complete_and_valid() -> None:
    record = apply_pre_experiment_updates(
        _placeholder_record(),
        updates={
            "pre_experiment.fly_id": 7,
            "pre_experiment.species": "dmel",
            "pre_experiment.genotype": "MB247B",
            "pre_experiment.hemisphere": "left",
            "pre_experiment.starvation.unit": "hours",
            "pre_experiment.stimulus_modality": "odor",
            "pre_experiment.rig_id": "Dev1",
            "pre_experiment.operator": "rm",
        },
        entered_by="rm",
        timestamp_utc="2026-05-03T16:49:37Z",
    )

    assert record["record_status"] == "completed_pre"
    assert record["entered_by"] == "rm"
    assert record["pre_experiment"]["fly_id"] == 7
    assert record["pre_experiment"]["species"] == "dmel"
    assert validate_record_for_stage(record, stage="pre") == []


def test_apply_post_updates_requires_exclusion_reason_for_excluded_runs() -> None:
    record = apply_post_experiment_updates(
        _placeholder_record(),
        updates={
            "post_experiment.response": "no_response",
            "post_experiment.completion_status": "excluded",
            "post_experiment.aborted": False,
            "post_experiment.exclusion_reason": None,
        },
        entered_by="rm",
        timestamp_utc="2026-05-03T16:52:00Z",
    )

    assert "Missing required field: post_experiment.exclusion_reason" in validate_record_for_stage(record, stage="post")


def test_apply_post_run_defaults_sets_duration_and_draft_post_state() -> None:
    record = apply_pre_experiment_updates(
        _placeholder_record(),
        updates={
            "pre_experiment.fly_id": 7,
            "pre_experiment.species": "dmel",
            "pre_experiment.genotype": "MB247B",
            "pre_experiment.hemisphere": "left",
            "pre_experiment.stimulus_modality": "odor",
            "pre_experiment.rig_id": "Dev1",
        },
        entered_by="rm",
        timestamp_utc="2026-05-03T16:49:37Z",
    )

    updated = apply_post_run_defaults(record, duration_s=12.34567, aborted=True)

    assert updated["record_status"] == "draft_post"
    assert updated["post_experiment"]["duration_s"] == 12.346
    assert updated["post_experiment"]["aborted"] is True
    assert updated["post_experiment"]["completion_status"] == "aborted"


def test_summarize_metadata_status_tracks_missing_post_fields() -> None:
    record = apply_pre_experiment_updates(
        _placeholder_record(),
        updates={
            "pre_experiment.fly_id": 7,
            "pre_experiment.species": "dmel",
            "pre_experiment.genotype": "MB247B",
            "pre_experiment.hemisphere": "left",
            "pre_experiment.stimulus_modality": "odor",
            "pre_experiment.rig_id": "Dev1",
            "pre_experiment.operator": "rm",
        },
        entered_by="rm",
        timestamp_utc="2026-05-03T16:49:37Z",
    )
    record = apply_post_run_defaults(record, duration_s=9.5, aborted=False)

    summary = summarize_metadata_status(record)

    assert summary["record_present"] is True
    assert summary["pre_experiment_complete"] is True
    assert summary["post_experiment_complete"] is False
    assert summary["metadata_complete"] is False
    assert summary["operator_recorded"] is True
    assert summary["missing_required_fields"]["post_experiment"] == ["post_experiment.response"]


def test_write_experiment_metadata_files_persists_record_and_sidecar(tmp_path: Path) -> None:
    record_path = tmp_path / "experiment" / "record.json"
    record_meta_path = tmp_path / "experiment" / "record.meta.json"
    record = _placeholder_record()

    write_experiment_metadata_files(
        record_path=record_path,
        record=record,
        record_meta_path=record_meta_path,
    )

    saved_record = json.loads(record_path.read_text(encoding="utf-8"))
    saved_meta = json.loads(record_meta_path.read_text(encoding="utf-8"))
    assert saved_record["schema_name"] == "multibios.experiment_record"
    assert saved_meta["schema_name"] == "multibios.experiment_record_meta"