from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

from multibios.apps.metadata_form import (_build_pywebview_command,
                                          _copy_imaging_dataset_into_run,
                                          _run_pywebview_window,
                                          create_app)
from multibios.experiment_metadata import (build_experiment_record_meta_payload,
                                           default_history_store,
                                           save_metadata_history)
from multibios.run_dataset import build_placeholder_experiment_record


def _write_metadata_inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    experiment_dir = tmp_path / "experiment"
    record_path = experiment_dir / "record.json"
    record_meta_path = experiment_dir / "record.meta.json"
    history_path = tmp_path / "history.json"

    record = build_placeholder_experiment_record(
        run_id="2026-05-04_11-57-06",
        run_uuid="1234",
        source_filename="short_protocol.yaml",
        protocol_name="Short Protocol",
        protocol_version="1.0",
        rig_id="Dev1",
        operator=None,
    )
    record_path.parent.mkdir(parents=True, exist_ok=True)
    record_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    record_meta_path.write_text(json.dumps(build_experiment_record_meta_payload(), indent=2), encoding="utf-8")
    save_metadata_history(history_path, default_history_store())
    return record_path, record_meta_path, history_path


def _walk_text(node: Any):
    if isinstance(node, str):
        yield node
        return

    children = getattr(node, "children", None)
    if isinstance(children, (list, tuple)):
        for child in children:
            yield from _walk_text(child)
    elif children is not None:
        yield from _walk_text(children)


def _walk_ids(node: Any):
    component_id = getattr(node, "id", None)
    if component_id is not None:
        yield component_id

    children = getattr(node, "children", None)
    if isinstance(children, (list, tuple)):
        for child in children:
            yield from _walk_ids(child)
    elif children is not None:
        yield from _walk_ids(children)


def _find_by_id(node: Any, component_id: str):
    if getattr(node, "id", None) == component_id:
        return node

    children = getattr(node, "children", None)
    if isinstance(children, (list, tuple)):
        for child in children:
            found = _find_by_id(child, component_id)
            if found is not None:
                return found
    elif children is not None:
        return _find_by_id(children, component_id)
    return None


def test_create_app_builds_pre_stage_layout(tmp_path: Path) -> None:
    record_path, record_meta_path, history_path = _write_metadata_inputs(tmp_path)
    payload = json.loads(record_path.read_text(encoding="utf-8"))
    payload["pre_experiment"]["expected_imaging_periods"] = 4
    record_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    app = create_app(
        record_path=record_path,
        record_meta_path=record_meta_path,
        history_path=history_path,
        stage="pre",
    )

    assert app.layout is not None
    assert app.title == "MultiBiOS Metadata Form"
    texts = set(_walk_text(app.layout))
    ids = set(_walk_ids(app.layout))
    assert "Run ID (experiment)" in texts
    assert "Is this the same fly?" in texts
    assert "Expected imaging periods" in texts
    assert "Set iterations to 4 before starting the protocol." in texts
    assert "Required fields are marked Required. Optional fields can be left blank." in texts
    assert "close-window-signal" in ids


def test_pre_stage_hides_post_fields_but_preserves_callback_ids(tmp_path: Path) -> None:
    record_path, record_meta_path, history_path = _write_metadata_inputs(tmp_path)

    app = create_app(
        record_path=record_path,
        record_meta_path=record_meta_path,
        history_path=history_path,
        stage="pre",
    )

    texts = set(_walk_text(app.layout))
    ids = set(_walk_ids(app.layout))

    assert "Pre-Experiment" in texts
    assert "Post-Experiment" not in texts
    assert "Notes" not in texts
    assert "Observed anomalies" not in texts
    assert "Quality flags" not in texts
    assert "post-notes" in ids
    assert "post-response" in ids


def test_post_stage_hides_pre_fields_but_preserves_callback_ids(tmp_path: Path) -> None:
    record_path, record_meta_path, history_path = _write_metadata_inputs(tmp_path)
    payload = json.loads(record_path.read_text(encoding="utf-8"))
    payload["pre_experiment"]["expected_imaging_periods"] = 2
    record_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    app = create_app(
        record_path=record_path,
        record_meta_path=record_meta_path,
        history_path=history_path,
        stage="post",
    )

    texts = set(_walk_text(app.layout))
    ids = set(_walk_ids(app.layout))

    assert "Post-Experiment" in texts
    assert "Pre-Experiment" not in texts
    assert "Fly ID" not in texts
    assert "Genotype" not in texts
    assert "Stimulus modality" not in texts
    assert "confirm-save-button" in ids
    assert "pre-fly-choice" in ids
    assert "pre-fly-id" in ids
    assert "pre-genotype" in ids
    assert "post-select-imaging-dataset" in ids
    assert "post-imaging-dataset-source" in ids
    assert "post-imaging-dataset-relative-path" in ids
    assert "post-imaging-acquisition-type" in ids
    assert "post-imaging-num-rois" in ids
    assert "post-imaging-num-channels" in ids
    assert "post-imaging-num-planes" in ids
    exclusion_field = _find_by_id(app.layout, "post-exclusion-reason-field")
    assert exclusion_field.style.get("display") == "none"


def test_post_stage_shows_microscopy_acquisition_fields_when_imaging_is_expected(tmp_path: Path) -> None:
    record_path, record_meta_path, history_path = _write_metadata_inputs(tmp_path)
    payload = json.loads(record_path.read_text(encoding="utf-8"))
    payload["pre_experiment"]["expected_imaging_periods"] = 2
    payload["post_experiment"]["imaging_acquisition_type"] = "volumetric"
    payload["post_experiment"]["imaging_num_rois"] = 4
    payload["post_experiment"]["imaging_num_channels"] = 2
    payload["post_experiment"]["imaging_num_planes"] = 6
    record_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    app = create_app(
        record_path=record_path,
        record_meta_path=record_meta_path,
        history_path=history_path,
        stage="post",
    )

    texts = set(_walk_text(app.layout))
    assert "Acquisition type" in texts
    assert "Number of ROIs" in texts
    assert "Number of channels" in texts
    assert "Number of planes" in texts
    assert _find_by_id(app.layout, "post-imaging-acquisition-type").value == "volumetric"
    assert _find_by_id(app.layout, "post-imaging-num-rois").value == "4"
    assert _find_by_id(app.layout, "post-imaging-num-channels").value == "2"
    assert _find_by_id(app.layout, "post-imaging-num-planes").value == "6"
    assert _find_by_id(app.layout, "post-imaging-num-planes-field").style.get("display") is None


def test_copy_imaging_dataset_into_run_copies_selected_directory(tmp_path: Path) -> None:
    record_path, _record_meta_path, _history_path = _write_metadata_inputs(tmp_path)
    source_dir = tmp_path / "prairieview" / "dataset_001"
    source_dir.mkdir(parents=True)
    (source_dir / "metadata.env").write_text("ok", encoding="utf-8")

    copied_dir = _copy_imaging_dataset_into_run(record_path=record_path, source_dir=source_dir)

    assert copied_dir == tmp_path / "recorded" / "microscopy" / "dataset_001"
    assert copied_dir.is_dir()
    assert (copied_dir / "metadata.env").read_text(encoding="utf-8") == "ok"


def test_post_stage_does_not_prefill_response_or_exclusion_reason(tmp_path: Path) -> None:
    record_path, record_meta_path, history_path = _write_metadata_inputs(tmp_path)
    history = default_history_store()
    history["fields"] = {
        "post_experiment.response": {"values": ["response_from_history"]},
        "post_experiment.exclusion_reason": {"values": ["reason_from_history"]},
    }
    save_metadata_history(history_path, history)

    app = create_app(
        record_path=record_path,
        record_meta_path=record_meta_path,
        history_path=history_path,
        stage="post",
    )

    assert _find_by_id(app.layout, "post-response").value == ""
    assert _find_by_id(app.layout, "post-exclusion-reason").value == ""


def test_pre_stage_prefills_recent_values_and_daily_fly_id(tmp_path: Path) -> None:
    record_path, record_meta_path, history_path = _write_metadata_inputs(tmp_path)
    history = default_history_store()
    history["fields"] = {
        "pre_experiment.operator": {"values": ["rm"]},
        "pre_experiment.species": {"values": ["dmel"]},
        "pre_experiment.genotype": {"values": ["MB247B"]},
        "pre_experiment.stimulus_modality": {"values": ["odor"]},
        "pre_experiment.starvation.unit": {"values": ["hours"]},
    }
    history["daily_fly_ids"] = {
        "2026-05-04": {"last_fly_id": 3, "seen_fly_ids": [1, 2, 3], "updated_utc": "2026-05-04T10:00:00Z"},
    }
    history["daily_pre_defaults"] = {
        "2026-05-04": {
            "values": {
                "pre_experiment.age.value": 5,
                "pre_experiment.age.unit": "days",
                "pre_experiment.starvation.value": 18,
                "pre_experiment.starvation.unit": "hours",
                "pre_experiment.rig_temperature_c": 24.5,
                "pre_experiment.humidity_percent": 52.0,
            }
        }
    }
    save_metadata_history(history_path, history)

    app = create_app(
        record_path=record_path,
        record_meta_path=record_meta_path,
        history_path=history_path,
        stage="pre",
    )

    assert _find_by_id(app.layout, "entered-by").value == "rm"
    assert _find_by_id(app.layout, "pre-species").value == "dmel"
    assert _find_by_id(app.layout, "pre-species-new-field").style.get("display") == "none"
    assert _find_by_id(app.layout, "pre-genotype").value == "MB247B"
    assert _find_by_id(app.layout, "pre-stimulus-modality").value == "odor"
    assert _find_by_id(app.layout, "pre-stimulus-modality-new-field").style.get("display") == "none"
    assert _find_by_id(app.layout, "pre-age-value").value == "5"
    assert _find_by_id(app.layout, "pre-age-unit").value == "days"
    assert _find_by_id(app.layout, "pre-starvation-value").value == "18"
    assert _find_by_id(app.layout, "pre-starvation-unit").value == "hours"
    assert _find_by_id(app.layout, "pre-rig-temperature").value == "24.5"
    assert _find_by_id(app.layout, "pre-humidity").value == "52.0"
    assert _find_by_id(app.layout, "pre-fly-choice").value == "same"
    assert _find_by_id(app.layout, "pre-fly-id").value == 3
    assert _find_by_id(app.layout, "pre-previous-fly-id").disabled is True
    assert _find_by_id(app.layout, "pre-fly-modal") is not None
    assert _find_by_id(app.layout, "pre-save-confirm-modal") is not None
    texts = set(_walk_text(app.layout))
    assert "Fly ID (daily)" in texts
    assert "Run ID is unique per experiment. Fly ID is reused or incremented within the current experiment date." in texts


def test_pre_stage_uses_existing_options_and_exposes_new_option_inputs(tmp_path: Path) -> None:
    record_path, record_meta_path, history_path = _write_metadata_inputs(tmp_path)
    history = default_history_store()
    history["fields"] = {
        "pre_experiment.species": {"values": ["dmel", "dsim"]},
        "pre_experiment.stimulus_modality": {"values": ["odor", "visual"]},
    }
    payload = json.loads(record_path.read_text(encoding="utf-8"))
    payload["pre_experiment"]["species"] = "new_species"
    payload["pre_experiment"]["stimulus_modality"] = "new_modality"
    record_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    save_metadata_history(history_path, history)

    app = create_app(
        record_path=record_path,
        record_meta_path=record_meta_path,
        history_path=history_path,
        stage="pre",
    )

    species = _find_by_id(app.layout, "pre-species")
    stimulus = _find_by_id(app.layout, "pre-stimulus-modality")
    assert species.value == "__new__"
    assert {option["value"] for option in species.options} == {"dmel", "dsim", "__new__"}
    assert _find_by_id(app.layout, "pre-species-new").value == "new_species"
    assert _find_by_id(app.layout, "pre-species-new-field").style.get("display") is None
    assert stimulus.value == "__new__"
    assert {option["value"] for option in stimulus.options} == {"odor", "visual", "__new__"}
    assert _find_by_id(app.layout, "pre-stimulus-modality-new").value == "new_modality"
    assert _find_by_id(app.layout, "pre-stimulus-modality-new-field").style.get("display") is None
    assert _find_by_id(app.layout, "pre-new-terms-confirm-field").style.get("display") == "none"


def test_pre_stage_exposes_previous_same_day_fly_selector(tmp_path: Path) -> None:
    record_path, record_meta_path, history_path = _write_metadata_inputs(tmp_path)
    history = default_history_store()
    history["daily_fly_ids"] = {
        "2026-05-04": {"last_fly_id": 5, "seen_fly_ids": [1, 3, 5], "updated_utc": "2026-05-04T10:00:00Z"},
    }
    payload = json.loads(record_path.read_text(encoding="utf-8"))
    payload["pre_experiment"]["fly_id"] = 3
    record_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    save_metadata_history(history_path, history)

    app = create_app(
        record_path=record_path,
        record_meta_path=record_meta_path,
        history_path=history_path,
        stage="pre",
    )

    assert _find_by_id(app.layout, "pre-fly-choice").value == "previous"
    assert _find_by_id(app.layout, "pre-previous-fly-id").disabled is False
    assert _find_by_id(app.layout, "pre-previous-fly-id").value == 3


def test_build_pywebview_command_uses_module_launcher_when_installed() -> None:
    with patch("multibios.apps.metadata_form.importlib.util.find_spec", return_value=object()):
        command = _build_pywebview_command("http://127.0.0.1:8060")

    assert command is not None
    assert command[0].endswith("python.exe") or command[0].endswith("python")
    assert command[1:4] == ["-m", "multibios.apps.metadata_form", "--pywebview-url"]
    assert command[4] == "http://127.0.0.1:8060"


def test_build_pywebview_command_returns_none_when_pywebview_missing() -> None:
    with patch("multibios.apps.metadata_form.importlib.util.find_spec", return_value=None):
        command = _build_pywebview_command("http://127.0.0.1:8060")

    assert command is None


def test_run_pywebview_window_blocks_user_close_but_allows_programmatic_close() -> None:
    class _DummyClosingEvent:
        def __init__(self) -> None:
            self.handlers = []

        def __iadd__(self, handler):
            self.handlers.append(handler)
            return self

    destroy = MagicMock()
    closing_event = _DummyClosingEvent()
    window = SimpleNamespace(events=SimpleNamespace(closing=closing_event), destroy=destroy)
    webview_module = SimpleNamespace(
        create_window=MagicMock(return_value=window),
        start=MagicMock(),
    )

    with patch.dict("sys.modules", {"webview": webview_module}):
        _run_pywebview_window("http://127.0.0.1:8060")

    api = webview_module.create_window.call_args.kwargs["js_api"]
    assert len(closing_event.handlers) == 1
    assert closing_event.handlers[0]() is False

    api.close_window()

    assert destroy.called
    assert closing_event.handlers[0]() is None