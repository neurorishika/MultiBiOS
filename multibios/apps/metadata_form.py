from __future__ import annotations

import argparse
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time
import webbrowser

import dash
from dash import Input, Output, State, dcc, html

from multibios.experiment_metadata import (HISTORY_ENABLED_FIELDS,
                                           REQUIRED_POST_FIELDS,
                                           REQUIRED_PRE_FIELDS,
                                           UI_VERSION,
                                           apply_post_experiment_updates,
                                           apply_pre_experiment_updates,
                                           daily_pre_defaults_for_date,
                                           default_metadata_history_path,
                                           last_fly_id_for_date,
                                           load_metadata_history,
                                           persist_metadata_history_source,
                                           previous_fly_ids_for_date,
                                           recent_history_value,
                                           suggested_fly_id,
                                           update_daily_pre_defaults,
                                           update_daily_fly_id,
                                           update_metadata_history,
                                           validate_record_for_stage,
                                           write_experiment_metadata_files)
from multibios.run_dataset import normalize_run_relative_path


BG = "#111827"
CARD = "#1f2937"
BORDER = "#374151"
TEXT = "#f9fafb"
SUBTEXT = "#9ca3af"
ACCENT = "#60a5fa"


def create_app(
    *,
    record_path: Path,
    record_meta_path: Path,
    history_path: Path,
    stage: str,
    completion_file: Path | None = None,
) -> dash.Dash:
    record = json.loads(record_path.read_text(encoding="utf-8"))
    history = load_metadata_history(history_path)
    prefilled_record = _prefill_record_from_history(record=record, history=history, stage=stage)

    app = dash.Dash(__name__, title="MultiBiOS Metadata Form", suppress_callback_exceptions=True)
    app.layout = _build_layout(record=prefilled_record, history=history, stage=stage)
    app.index_string = _index_string()
    app.clientside_callback(
        """
        function(status) {
            if (typeof status === 'string' && status.startsWith('Saved ')) {
                setTimeout(function() {
                    if (window.pywebview && window.pywebview.api && window.pywebview.api.close_window) {
                        try {
                            window.pywebview.api.close_window();
                            return;
                        } catch (error) {
                        }
                    }
                    try {
                        window.close();
                    } catch (error) {
                    }
                }, 150);
            }
            return window.dash_clientside.no_update;
        }
        """,
        Output("close-window-signal", "data"),
        Input("submit-status", "children"),
    )

    if stage == "pre":
        experiment_date = str((prefilled_record.get("pre_experiment") or {}).get("experiment_date") or "")
        last_fly_id = last_fly_id_for_date(history, experiment_date)
        previous_fly_ids = previous_fly_ids_for_date(history, experiment_date)

        @app.callback(
            Output("pre-fly-id", "value"),
            Output("pre-fly-choice-help", "children"),
            Output("pre-previous-fly-id", "disabled"),
            Output("pre-previous-fly-id", "value"),
            Input("pre-fly-choice", "value"),
            Input("pre-previous-fly-id", "value"),
            prevent_initial_call=True,
        )
        def update_fly_selection(choice: str | None, previous_fly_id: str | int | None) -> tuple[int | None, str, bool, int | None]:
            normalized_previous = _coerce_int(previous_fly_id)
            previous_default = previous_fly_ids[0] if previous_fly_ids else None
            if choice == "same":
                fly_id = last_fly_id
                resolved_previous = None
            elif choice == "previous":
                resolved_previous = normalized_previous if normalized_previous in previous_fly_ids else previous_default
                fly_id = resolved_previous
            else:
                fly_id = suggested_fly_id(history, experiment_date=experiment_date, same_fly=False)
                resolved_previous = None
            return (
                fly_id,
                _fly_choice_help_text(history, experiment_date=experiment_date, choice=choice, selected_previous_fly_id=resolved_previous),
                choice != "previous",
                resolved_previous,
            )

        @app.callback(
            Output("pre-fly-choice", "value", allow_duplicate=True),
            Output("pre-previous-fly-id", "value", allow_duplicate=True),
            Output("pre-fly-modal", "style"),
            Input("pre-fly-same-button", "n_clicks"),
            Input("pre-fly-previous-button", "n_clicks"),
            Input("pre-fly-new-button", "n_clicks"),
            prevent_initial_call=True,
        )
        def dismiss_fly_modal(same_clicks: int | None, previous_clicks: int | None, new_clicks: int | None) -> tuple[str | None, int | None, dict]:
            triggered = dash.ctx.triggered_id
            if triggered == "pre-fly-same-button" and last_fly_id is not None:
                choice = "same"
            elif triggered == "pre-fly-previous-button" and previous_fly_ids:
                choice = "previous"
            elif triggered == "pre-fly-new-button":
                choice = "new"
            else:
                choice = "same" if last_fly_id is not None else "new"
            selected_previous = previous_fly_ids[0] if choice == "previous" and previous_fly_ids else None
            return choice, selected_previous, _hidden_modal_style()

        @app.callback(
            Output("pre-save-confirm-summary", "children"),
            Output("pre-save-confirm-modal", "style"),
            Output("pre-new-terms-message", "children"),
            Output("pre-new-terms-confirm-field", "style"),
            Output("pre-new-terms-confirm", "value"),
            Input("submit-button", "n_clicks"),
            State("entered-by", "value"),
            State("pre-fly-choice", "value"),
            State("pre-fly-id", "value"),
            State("pre-species", "value"),
            State("pre-species-new", "value"),
            State("pre-genotype", "value"),
            State("pre-hemisphere", "value"),
            State("pre-age-value", "value"),
            State("pre-age-unit", "value"),
            State("pre-starvation-value", "value"),
            State("pre-starvation-unit", "value"),
            State("pre-volumetric", "value"),
            State("pre-stimulus-modality", "value"),
            State("pre-stimulus-modality-new", "value"),
            State("pre-rig-temperature", "value"),
            State("pre-humidity", "value"),
            prevent_initial_call=True,
        )
        def open_pre_save_confirmation(
            _n_clicks: int,
            entered_by: str | None,
            fly_choice: str | None,
            fly_id: str | None,
            species: str | None,
            species_new: str | None,
            genotype: str | None,
            hemisphere: str | None,
            age_value: str | None,
            age_unit: str | None,
            starvation_value: str | None,
            starvation_unit: str | None,
            volumetric: str | None,
            stimulus_modality: str | None,
            stimulus_modality_new: str | None,
            rig_temperature: str | None,
            humidity: str | None,
        ):
            resolved_species = _resolve_controlled_value(species, species_new)
            resolved_stimulus_modality = _resolve_controlled_value(stimulus_modality, stimulus_modality_new)
            pending_new_terms = _pending_new_terms(
                history,
                {
                    "pre_experiment.species": resolved_species,
                    "pre_experiment.stimulus_modality": resolved_stimulus_modality,
                },
            )
            summary = _build_pre_run_confirmation_summary(
                run_id=prefilled_record.get("run_id"),
                entered_by=entered_by,
                fly_choice=fly_choice,
                fly_id=fly_id,
                species=resolved_species,
                genotype=genotype,
                hemisphere=hemisphere,
                age_value=age_value,
                age_unit=age_unit,
                starvation_value=starvation_value,
                starvation_unit=starvation_unit,
                volumetric=volumetric,
                stimulus_modality=resolved_stimulus_modality,
                rig_temperature=rig_temperature,
                humidity=humidity,
            )
            confirmation_message = _pending_new_terms_message(pending_new_terms)
            confirmation_style = _new_terms_confirm_style(bool(pending_new_terms))
            return summary, _modal_style(), confirmation_message, confirmation_style, []

        @app.callback(
            Output("pre-save-confirm-modal", "style", allow_duplicate=True),
            Input("pre-save-cancel-button", "n_clicks"),
            prevent_initial_call=True,
        )
        def cancel_pre_save_confirmation(_n_clicks: int) -> dict:
            return _hidden_modal_style()

        @app.callback(
            Output("pre-species-new-field", "style"),
            Output("pre-stimulus-modality-new-field", "style"),
            Input("pre-species", "value"),
            Input("pre-stimulus-modality", "value"),
        )
        def toggle_controlled_new_fields(species: str | None, stimulus_modality: str | None) -> tuple[dict, dict]:
            return _controlled_new_field_style(species), _controlled_new_field_style(stimulus_modality)

    if stage == "post":
        @app.callback(
            Output("post-exclusion-reason-field", "style"),
            Output("post-exclusion-reason-required", "children"),
            Output("post-exclusion-reason", "value"),
            Input("post-completion-status", "value"),
            State("post-exclusion-reason", "value"),
        )
        def toggle_post_exclusion_reason(completion_status: str | None, exclusion_reason: str | None) -> tuple[dict, str, str]:
            if completion_status == "excluded":
                return {}, "Required", exclusion_reason or ""
            return {"display": "none"}, "Optional", ""

        @app.callback(
            Output("post-imaging-num-planes-field", "style"),
            Output("post-imaging-num-planes-required", "children"),
            Output("post-imaging-num-planes", "value"),
            Input("post-imaging-acquisition-type", "value"),
            State("post-imaging-num-planes", "value"),
        )
        def toggle_post_imaging_num_planes(acquisition_type: str | None, num_planes: str | None) -> tuple[dict, str, str]:
            if acquisition_type == "volumetric":
                return {}, "Required", num_planes or ""
            return {"display": "none"}, "Optional", ""

        @app.callback(
            Output("post-imaging-dataset-source", "value"),
            Output("post-imaging-dataset-relative-path", "value"),
            Output("post-imaging-dataset-status", "children"),
            Input("post-select-imaging-dataset", "n_clicks"),
            State("post-imaging-dataset-source", "value"),
            State("post-imaging-dataset-relative-path", "value"),
            prevent_initial_call=True,
        )
        def select_imaging_dataset(
            _n_clicks: int | None,
            current_source: str | None,
            current_relative_path: str | None,
        ) -> tuple[str, str, str]:
            try:
                selected_dir = _select_directory_dialog(
                    title="Select completed PrairieView imaging dataset",
                    initial_dir=_initial_dialog_directory(current_source),
                )
            except Exception as exc:
                return current_source or "", current_relative_path or "", f"Failed to open imaging dataset picker: {exc}"
            if selected_dir is None:
                return current_source or "", current_relative_path or "", "Imaging dataset selection cancelled."

            try:
                destination = _copy_imaging_dataset_into_run(record_path=record_path, source_dir=selected_dir)
            except Exception as exc:
                return current_source or "", current_relative_path or "", f"Failed to copy imaging dataset: {exc}"
            relative_destination = normalize_run_relative_path(record_path.parent.parent, destination)
            return str(selected_dir), relative_destination, f"Copied imaging dataset to {relative_destination}"

    @app.callback(
        Output("submit-status", "children"),
        Input("submit-button", "n_clicks"),
        Input("confirm-save-button", "n_clicks"),
        State("entered-by", "value"),
        State("pre-fly-choice", "value"),
        State("pre-fly-id", "value"),
        State("pre-species", "value"),
        State("pre-species-new", "value"),
        State("pre-genotype", "value"),
        State("pre-hemisphere", "value"),
        State("pre-age-value", "value"),
        State("pre-age-unit", "value"),
        State("pre-starvation-value", "value"),
        State("pre-starvation-unit", "value"),
        State("pre-volumetric", "value"),
        State("pre-stimulus-modality", "value"),
        State("pre-stimulus-modality-new", "value"),
        State("pre-rig-temperature", "value"),
        State("pre-humidity", "value"),
        State("pre-new-terms-confirm", "value"),
        State("post-response", "value"),
        State("post-notes", "value"),
        State("post-completion-status", "value"),
        State("post-aborted", "value"),
        State("post-exclusion-reason", "value"),
        State("post-imaging-dataset-source", "value"),
        State("post-imaging-dataset-relative-path", "value"),
        State("post-imaging-acquisition-type", "value"),
        State("post-imaging-num-rois", "value"),
        State("post-imaging-num-channels", "value"),
        State("post-imaging-num-planes", "value"),
        State("post-observed-anomalies", "value"),
        State("post-quality-flags", "value"),
        prevent_initial_call=True,
    )
    def save_form(
        _submit_clicks: int | None,
        _confirm_clicks: int | None,
        entered_by: str | None,
        fly_choice: str | None,
        fly_id: str | None,
        species: str | None,
        species_new: str | None,
        genotype: str | None,
        hemisphere: str | None,
        age_value: str | None,
        age_unit: str | None,
        starvation_value: str | None,
        starvation_unit: str | None,
        volumetric: str | None,
        stimulus_modality: str | None,
        stimulus_modality_new: str | None,
        rig_temperature: str | None,
        humidity: str | None,
        confirm_new_terms: list[str] | None,
        response: str | None,
        notes: str | None,
        completion_status: str | None,
        aborted: list[str] | None,
        exclusion_reason: str | None,
        imaging_dataset_source_path: str | None,
        imaging_dataset_relative_path: str | None,
        imaging_acquisition_type: str | None,
        imaging_num_rois: str | None,
        imaging_num_channels: str | None,
        imaging_num_planes: str | None,
        observed_anomalies: str | None,
        quality_flags: str | None,
    ) -> str:
        triggered = dash.ctx.triggered_id
        if stage == "pre" and triggered != "confirm-save-button":
            return dash.no_update
        if stage == "post" and triggered != "submit-button":
            return dash.no_update

        current_record = json.loads(record_path.read_text(encoding="utf-8"))
        current_history = load_metadata_history(history_path)
        resolved_species = _resolve_controlled_value(species, species_new)
        resolved_stimulus_modality = _resolve_controlled_value(stimulus_modality, stimulus_modality_new)
        pending_new_terms = _pending_new_terms(
            current_history,
            {
                "pre_experiment.species": resolved_species,
                "pre_experiment.stimulus_modality": resolved_stimulus_modality,
            },
        )

        if stage == "pre":
            if pending_new_terms and "confirmed" not in (confirm_new_terms or []):
                return "Confirmation required: review and confirm the new species or stimulus modality before saving."
            updated = apply_pre_experiment_updates(
                current_record,
                updates={
                    "pre_experiment.fly_id": _coerce_int(fly_id),
                    "pre_experiment.species": resolved_species,
                    "pre_experiment.genotype": genotype,
                    "pre_experiment.hemisphere": hemisphere,
                    "pre_experiment.age.value": _coerce_number(age_value),
                    "pre_experiment.age.unit": age_unit,
                    "pre_experiment.starvation.value": _coerce_number(starvation_value),
                    "pre_experiment.starvation.unit": starvation_unit,
                    "pre_experiment.volumetric": volumetric,
                    "pre_experiment.stimulus_modality": resolved_stimulus_modality,
                    "pre_experiment.rig_temperature_c": _coerce_number(rig_temperature),
                    "pre_experiment.humidity_percent": _coerce_number(humidity),
                    "pre_experiment.operator": entered_by,
                },
                entered_by=entered_by,
                ui_version=UI_VERSION,
            )
        else:
            updated = apply_post_experiment_updates(
                current_record,
                updates={
                    "post_experiment.response": response,
                    "post_experiment.notes": notes,
                    "post_experiment.completion_status": completion_status,
                    "post_experiment.aborted": bool(aborted),
                    "post_experiment.exclusion_reason": exclusion_reason if completion_status == "excluded" else None,
                    "post_experiment.imaging_dataset_source_path": _strip_or_none(imaging_dataset_source_path),
                    "post_experiment.imaging_dataset_relative_path": _strip_or_none(imaging_dataset_relative_path),
                    "post_experiment.imaging_acquisition_type": _strip_or_none(imaging_acquisition_type),
                    "post_experiment.imaging_num_rois": _coerce_int(imaging_num_rois),
                    "post_experiment.imaging_num_channels": _coerce_int(imaging_num_channels),
                    "post_experiment.imaging_num_planes": _coerce_int(imaging_num_planes) if imaging_acquisition_type == "volumetric" else None,
                    "post_experiment.observed_anomalies": _split_lines(observed_anomalies),
                    "post_experiment.quality_flags": _split_lines(quality_flags),
                },
                entered_by=entered_by or current_record.get("entered_by"),
                ui_version=UI_VERSION,
            )

        errors = validate_record_for_stage(updated, stage=stage)
        if errors:
            return "Validation failed: " + "; ".join(errors)

        history_updates = {
            "pre_experiment.species": resolved_species,
            "pre_experiment.genotype": genotype,
            "pre_experiment.hemisphere": hemisphere,
            "pre_experiment.age.unit": age_unit,
            "pre_experiment.starvation.unit": starvation_unit,
            "pre_experiment.volumetric": volumetric,
            "pre_experiment.stimulus_modality": resolved_stimulus_modality,
            "pre_experiment.operator": entered_by,
            "post_experiment.response": response,
            "post_experiment.exclusion_reason": exclusion_reason if completion_status == "excluded" else None,
        }
        for field_path in HISTORY_ENABLED_FIELDS:
            current_history = update_metadata_history(
                current_history,
                field_path=field_path,
                value=history_updates.get(field_path),
            )
        if stage == "pre":
            current_history = update_daily_fly_id(
                current_history,
                experiment_date=str(((updated.get("pre_experiment") or {}).get("experiment_date") or "")),
                fly_id=_coerce_int(fly_id),
            )
            current_history = update_daily_pre_defaults(
                current_history,
                experiment_date=str(((updated.get("pre_experiment") or {}).get("experiment_date") or "")),
                values={
                    "pre_experiment.operator": entered_by,
                    "pre_experiment.species": resolved_species,
                    "pre_experiment.genotype": genotype,
                    "pre_experiment.hemisphere": hemisphere,
                    "pre_experiment.age.value": _coerce_number(age_value),
                    "pre_experiment.age.unit": age_unit,
                    "pre_experiment.starvation.value": _coerce_number(starvation_value),
                    "pre_experiment.starvation.unit": starvation_unit,
                    "pre_experiment.volumetric": volumetric,
                    "pre_experiment.stimulus_modality": resolved_stimulus_modality,
                    "pre_experiment.rig_temperature_c": _coerce_number(rig_temperature),
                    "pre_experiment.humidity_percent": _coerce_number(humidity),
                    "pre_experiment.fly_choice": fly_choice,
                },
            )

        write_experiment_metadata_files(
            record_path=record_path,
            record=updated,
            record_meta_path=record_meta_path,
        )
        persist_metadata_history_source(history_path, current_history, record=updated, stage=stage)
        if completion_file is not None:
            completion_file.parent.mkdir(parents=True, exist_ok=True)
            completion_file.write_text(
                json.dumps({"status": "submitted", "stage": stage, "record_status": updated.get("record_status")}, indent=2),
                encoding="utf-8",
            )
        return f"Saved {stage}-experiment metadata to {record_path}"

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="MultiBiOS metadata entry form")
    parser.add_argument("--pywebview-url", help=argparse.SUPPRESS)
    parser.add_argument("--record", help="Path to experiment/record.json")
    parser.add_argument("--record-meta", help="Path to experiment/record.meta.json")
    parser.add_argument("--history", default=str(default_metadata_history_path()), help="Path to metadata history store")
    parser.add_argument("--stage", choices=["pre", "post"])
    parser.add_argument("--completion-file")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8060)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    if args.pywebview_url:
        _run_pywebview_window(args.pywebview_url)
        return

    if not args.record:
        parser.error("--record is required unless --pywebview-url is used")
    if not args.stage:
        parser.error("--stage is required unless --pywebview-url is used")

    record_path = Path(args.record)
    record_meta_path = Path(args.record_meta) if args.record_meta else record_path.with_name("record.meta.json")
    history_path = Path(args.history)
    completion_file = Path(args.completion_file) if args.completion_file else None

    app = create_app(
        record_path=record_path,
        record_meta_path=record_meta_path,
        history_path=history_path,
        stage=args.stage,
        completion_file=completion_file,
    )
    url = f"http://{args.host}:{args.port}"
    print(f"MultiBiOS metadata form: {url}")
    if not args.no_browser:
        def _open() -> None:
            time.sleep(1.0)
            if not _open_metadata_window(url):
                webbrowser.open(url)
        threading.Thread(target=_open, daemon=True).start()
    app.run(host=args.host, port=args.port, debug=False)


def _build_layout(*, record: dict, history: dict, stage: str) -> html.Div:
    pre = record.get("pre_experiment", {})
    post = record.get("post_experiment", {})
    operator = pre.get("operator") or record.get("entered_by")
    active_stage_title = "Pre-Experiment" if stage == "pre" else "Post-Experiment"
    expected_imaging_periods = _coerce_int(pre.get("expected_imaging_periods")) or 0
    active_stage_fields = _pre_stage_fields(pre, history, run_id=record.get("run_id")) if stage == "pre" else _post_stage_fields(post, history, expected_imaging_periods=expected_imaging_periods)
    hidden_stage_fields = _hidden_post_stage_fields(post) if stage == "pre" else _hidden_pre_stage_fields(pre)

    return html.Div(
        [
            _pre_stage_modal(pre, history) if stage == "pre" else None,
            _pre_save_confirmation_modal() if stage == "pre" else None,
            html.Div(
                [
                    html.Div(
                        [
                            html.H1("MultiBiOS Metadata Entry", style={"margin": "0", "fontSize": "24px", "lineHeight": "1.2"}),
                            html.P("Required fields are marked Required. Optional fields can be left blank.", style={"color": SUBTEXT, "margin": "6px 0 0 0", "fontSize": "14px"}),
                        ]
                    ),
                    html.Div(stage.upper(), style=_stage_badge_style()),
                ],
                style={"display": "flex", "justifyContent": "space-between", "alignItems": "flex-start", "gap": "16px", "marginBottom": "16px"},
            ),
            html.Div(
                [
                    _card(
                        "Run Summary",
                        [
                            html.Div(
                                [
                                    _summary_item("Run ID (experiment)", record.get("run_id")),
                                    _summary_item("Protocol", pre.get("protocol_name")),
                                    _summary_item("Rig", pre.get("rig_id")),
                                    _summary_item("Date", pre.get("experiment_date")),
                                ],
                                style=_summary_grid_style(),
                            ),
                            _field_block("Entered by", dcc.Input(id="entered-by", value=operator or "", list="operator-history", style=_input_style())),
                            _history_list("operator-history", history, "pre_experiment.operator"),
                        ],
                        style={"alignSelf": "start"},
                    ),
                    _card(active_stage_title, active_stage_fields),
                ],
                style=_layout_grid_style(),
            ),
            html.Div(hidden_stage_fields, style={"display": "none"}),
            dcc.Store(id="close-window-signal"),
            html.Div(
                [
                    html.Button("Save Metadata", id="submit-button", style=_button_style()),
                    html.Div(id="submit-status", style={"color": SUBTEXT, "fontSize": "13px"}),
                ],
                style={"display": "flex", "alignItems": "center", "gap": "12px", "marginTop": "8px"},
            ),
        ],
        style={"background": BG, "color": TEXT, "minHeight": "100vh", "padding": "20px", "fontFamily": "'Segoe UI', system-ui, sans-serif", "maxWidth": "1120px", "margin": "0 auto"},
    )


def _pre_stage_fields(pre: dict, history: dict, *, run_id: str | None) -> list:
    experiment_date = str(pre.get("experiment_date") or "")
    expected_imaging_periods = _coerce_int(pre.get("expected_imaging_periods")) or 0
    last_fly_id = last_fly_id_for_date(history, experiment_date)
    previous_fly_ids = previous_fly_ids_for_date(history, experiment_date)
    fly_choice = _default_fly_choice(pre=pre, last_fly_id=last_fly_id)
    species_value, species_new_value, species_options = _controlled_select_state(pre.get("species") or "", history, "pre_experiment.species")
    stimulus_value, stimulus_new_value, stimulus_options = _controlled_select_state(pre.get("stimulus_modality") or "", history, "pre_experiment.stimulus_modality")

    return [
        html.Div(
            [
                html.Div("Fly Tracking", style={"fontSize": "13px", "fontWeight": "600", "marginBottom": "8px"}),
                html.Div(
                    [
                        _readonly_row("Run ID (experiment)", run_id),
                        _readonly_row("Fly ID (daily)", _stringify(pre.get("fly_id"))),
                    ],
                    style={"display": "grid", "gridTemplateColumns": "repeat(2, minmax(0, 1fr))", "gap": "10px", "marginBottom": "10px"},
                ),
                html.Div(
                    "Run ID is unique per experiment. Fly ID is reused or incremented within the current experiment date.",
                    style={"color": SUBTEXT, "fontSize": "12px", "marginBottom": "10px"},
                ),
                dcc.RadioItems(
                    id="pre-fly-choice",
                    options=[
                        {"label": "Same fly", "value": "same", "disabled": last_fly_id is None},
                        {"label": "Previous same-day fly", "value": "previous", "disabled": not previous_fly_ids},
                        {"label": "New fly", "value": "new"},
                    ],
                    value=fly_choice,
                    labelStyle={"display": "inline-flex", "alignItems": "center", "marginRight": "14px", "fontSize": "13px"},
                    inputStyle={"marginRight": "6px"},
                    style={"marginBottom": "6px"},
                ),
                _field_block(
                    "Previous same-day fly ID",
                    dcc.Dropdown(
                        id="pre-previous-fly-id",
                        options=[{"label": str(value), "value": value} for value in previous_fly_ids],
                        value=previous_fly_ids[0] if fly_choice == "previous" and previous_fly_ids else None,
                        clearable=False,
                        disabled=(fly_choice != "previous"),
                        style=_dropdown_style(),
                    ),
                    required=False,
                ),
                html.Div(
                    _fly_choice_help_text(history, experiment_date=experiment_date, choice=fly_choice, selected_previous_fly_id=previous_fly_ids[0] if fly_choice == "previous" and previous_fly_ids else None),
                    id="pre-fly-choice-help",
                    style={"color": SUBTEXT, "fontSize": "12px"},
                ),
            ],
            style={"padding": "12px", "border": f"1px solid {BORDER}", "borderRadius": "10px", "marginBottom": "12px", "background": BG},
        ),
        html.Div(
            [
                _readonly_row("Source", pre.get("source_filename")),
                _readonly_row("Protocol version", pre.get("protocol_version")),
                _readonly_row("Rig ID", pre.get("rig_id")),
            ],
            style=_compact_info_grid_style(),
        ),
        _microscopy_guidance_block(expected_imaging_periods),
        html.Div(
            [
                _field_block("Fly ID", dcc.Input(id="pre-fly-id", value=pre.get("fly_id"), type="number", disabled=True, style=_disabled_input_style() if True else _input_style()), required=True),
                _field_block("Species", dcc.Dropdown(id="pre-species", options=species_options, value=species_value, clearable=False, style=_dropdown_style()), required=True),
                _field_block("New species", dcc.Input(id="pre-species-new", value=species_new_value, style=_input_style()), required=True, container_id="pre-species-new-field", style=_controlled_new_field_style(species_value)),
                _field_block("Genotype", dcc.Input(id="pre-genotype", value=pre.get("genotype") or "", list="genotype-history", style=_input_style()), required=True),
                _field_block("Hemisphere", dcc.Dropdown(id="pre-hemisphere", options=_options(["left", "right", "bilateral", "unknown", "na"]), value=pre.get("hemisphere") or "unknown", clearable=False, style=_dropdown_style()), required=True),
                _field_block("Age value", dcc.Input(id="pre-age-value", value=_stringify((pre.get("age") or {}).get("value")), type="number", style=_input_style()), required=False),
                _field_block("Age unit", dcc.Dropdown(id="pre-age-unit", options=_options(["hours", "days", "weeks", "unknown"]), value=(pre.get("age") or {}).get("unit") or "unknown", clearable=False, style=_dropdown_style()), required=False),
                _field_block("Starvation value", dcc.Input(id="pre-starvation-value", value=_stringify((pre.get("starvation") or {}).get("value")), type="number", style=_input_style()), required=False),
                _field_block("Starvation unit", dcc.Dropdown(id="pre-starvation-unit", options=_options(["hours", "days", "weeks", "unknown"]), value=(pre.get("starvation") or {}).get("unit") or "unknown", clearable=False, style=_dropdown_style()), required=False),
                _field_block("Volumetric", dcc.Dropdown(id="pre-volumetric", options=_options(["yes", "no", "unknown"]), value=pre.get("volumetric") or "unknown", clearable=False, style=_dropdown_style()), required=False),
                _field_block("Stimulus modality", dcc.Dropdown(id="pre-stimulus-modality", options=stimulus_options, value=stimulus_value, clearable=False, style=_dropdown_style()), required=True),
                _field_block("New stimulus modality", dcc.Input(id="pre-stimulus-modality-new", value=stimulus_new_value, style=_input_style()), required=True, container_id="pre-stimulus-modality-new-field", style=_controlled_new_field_style(stimulus_value)),
                _field_block("Rig temperature (C)", dcc.Input(id="pre-rig-temperature", value=_stringify(pre.get("rig_temperature_c")), type="number", style=_input_style()), required=False),
                _field_block("Humidity (%)", dcc.Input(id="pre-humidity", value=_stringify(pre.get("humidity_percent")), type="number", style=_input_style()), required=False),
            ],
            style=_form_grid_style(),
        ),
        _history_list("genotype-history", history, "pre_experiment.genotype"),
    ]


def _post_stage_fields(post: dict, history: dict, *, expected_imaging_periods: int) -> list:
    exclusion_required = post.get("completion_status") == "excluded"
    return [
        html.Div("Required fields are marked Required. Optional fields can be left blank.", style={"color": SUBTEXT, "fontSize": "13px", "marginBottom": "12px"}),
        html.Div([_readonly_row("Duration (s)", _stringify(post.get("duration_s")))], style=_compact_info_grid_style()),
        _post_imaging_dataset_block(post, expected_imaging_periods),
        html.Div(
            [
                _field_block("Response", dcc.Input(id="post-response", value=post.get("response") or "", list="response-history", style=_input_style()), required=True),
                _field_block("Completion status", dcc.Dropdown(id="post-completion-status", options=_options(["completed", "completed_with_issue", "aborted", "failed", "excluded"]), value=post.get("completion_status"), clearable=False, style=_dropdown_style()), required=True),
                _field_block("Aborted", dcc.Checklist(id="post-aborted", options=[{"label": "Run aborted", "value": "aborted"}], value=["aborted"] if post.get("aborted") else [], style={"color": TEXT}, inputStyle={"marginRight": "8px"}, labelStyle={"display": "inline-flex", "alignItems": "center"}), required=True),
                _field_block("Exclusion reason", dcc.Input(id="post-exclusion-reason", value=post.get("exclusion_reason") or "", list="exclusion-history", style=_input_style()), required=exclusion_required, container_id="post-exclusion-reason-field", badge_id="post-exclusion-reason-required", style={} if exclusion_required else {"display": "none"}),
                _field_block("Notes", dcc.Textarea(id="post-notes", value=post.get("notes") or "", style=_textarea_style()), required=False),
                _field_block("Observed anomalies", dcc.Textarea(id="post-observed-anomalies", value="\n".join(post.get("observed_anomalies") or []), style=_textarea_style()), required=False),
                _field_block("Quality flags", dcc.Textarea(id="post-quality-flags", value="\n".join(post.get("quality_flags") or []), style=_textarea_style()), required=False),
            ],
            style=_form_grid_style(),
        ),
        _history_list("response-history", history, "post_experiment.response"),
        _history_list("exclusion-history", history, "post_experiment.exclusion_reason"),
    ]


def _hidden_pre_stage_fields(pre: dict) -> list:
    species_value, species_new_value, _ = _controlled_select_state(pre.get("species") or "", {}, "pre_experiment.species")
    stimulus_value, stimulus_new_value, _ = _controlled_select_state(pre.get("stimulus_modality") or "", {}, "pre_experiment.stimulus_modality")
    return [
        html.Button("", id="confirm-save-button", n_clicks=0),
        dcc.Checklist(id="pre-new-terms-confirm", options=[{"label": "", "value": "confirmed"}], value=[]),
        dcc.Input(id="pre-fly-choice", value="same"),
        dcc.Input(id="pre-fly-id", value=pre.get("fly_id"), type="number"),
        dcc.Input(id="pre-species", value=species_value),
        dcc.Input(id="pre-species-new", value=species_new_value),
        dcc.Input(id="pre-genotype", value=pre.get("genotype") or ""),
        dcc.Input(id="pre-hemisphere", value=pre.get("hemisphere") or "unknown"),
        dcc.Input(id="pre-age-value", value=_stringify((pre.get("age") or {}).get("value"))),
        dcc.Input(id="pre-age-unit", value=(pre.get("age") or {}).get("unit") or "unknown"),
        dcc.Input(id="pre-starvation-value", value=_stringify((pre.get("starvation") or {}).get("value"))),
        dcc.Input(id="pre-starvation-unit", value=(pre.get("starvation") or {}).get("unit") or "unknown"),
        dcc.Input(id="pre-volumetric", value=pre.get("volumetric") or "unknown"),
        dcc.Input(id="pre-stimulus-modality", value=stimulus_value),
        dcc.Input(id="pre-stimulus-modality-new", value=stimulus_new_value),
        dcc.Input(id="pre-rig-temperature", value=_stringify(pre.get("rig_temperature_c"))),
        dcc.Input(id="pre-humidity", value=_stringify(pre.get("humidity_percent"))),
    ]


def _hidden_post_stage_fields(post: dict) -> list:
    return [
        dcc.Input(id="post-response", value=post.get("response") or ""),
        dcc.Input(id="post-completion-status", value=post.get("completion_status") or ""),
        dcc.Checklist(id="post-aborted", options=[{"label": "Run aborted", "value": "aborted"}], value=["aborted"] if post.get("aborted") else []),
        dcc.Input(id="post-exclusion-reason", value=post.get("exclusion_reason") or ""),
        dcc.Input(id="post-imaging-dataset-source", value=post.get("imaging_dataset_source_path") or ""),
        dcc.Input(id="post-imaging-dataset-relative-path", value=post.get("imaging_dataset_relative_path") or ""),
        dcc.Input(id="post-imaging-acquisition-type", value=post.get("imaging_acquisition_type") or ""),
        dcc.Input(id="post-imaging-num-rois", value=_stringify(post.get("imaging_num_rois"))),
        dcc.Input(id="post-imaging-num-channels", value=_stringify(post.get("imaging_num_channels"))),
        dcc.Input(id="post-imaging-num-planes", value=_stringify(post.get("imaging_num_planes"))),
        html.Div(id="post-imaging-dataset-status"),
        html.Div(id="post-imaging-num-planes-field"),
        html.Span(id="post-imaging-num-planes-required"),
        html.Button("", id="post-select-imaging-dataset", n_clicks=0),
        dcc.Textarea(id="post-notes", value=post.get("notes") or ""),
        dcc.Textarea(id="post-observed-anomalies", value="\n".join(post.get("observed_anomalies") or [])),
        dcc.Textarea(id="post-quality-flags", value="\n".join(post.get("quality_flags") or [])),
    ]


def _card(title: str, children: list, style: dict | None = None) -> html.Div:
    return html.Div(
        [html.H2(title, style={"margin": "0 0 12px 0", "fontSize": "18px"}), *children],
        style={"background": CARD, "border": f"1px solid {BORDER}", "borderRadius": "12px", "padding": "14px", **(style or {})},
    )


def _readonly_row(label: str, value: str | None) -> html.Div:
    return html.Div(
        [
            html.Div(label, style={"color": SUBTEXT, "fontSize": "12px", "textTransform": "uppercase", "letterSpacing": "0.04em", "marginBottom": "3px"}),
            html.Div(value or "", style={"fontSize": "14px", "lineHeight": "1.35"}),
        ],
        style={"minWidth": 0},
    )


def _field_block(label: str, component, *, required: bool = False, container_id: str | None = None, badge_id: str | None = None, style: dict | None = None) -> html.Div:
    badge_kwargs = {"style": _field_badge_style(required)}
    if badge_id is not None:
        badge_kwargs["id"] = badge_id

    container_kwargs = {"style": {"minWidth": 0, **(style or {})}}
    if container_id is not None:
        container_kwargs["id"] = container_id

    return html.Div([
        html.Div(
            [
                html.Label(label, style={"display": "block", "color": SUBTEXT, "fontSize": "12px", "textTransform": "uppercase", "letterSpacing": "0.04em"}),
                html.Span("Required" if required else "Optional", **badge_kwargs),
            ],
            style={"display": "flex", "alignItems": "center", "justifyContent": "space-between", "gap": "8px", "marginBottom": "4px"},
        ),
        component,
    ], **container_kwargs)


def _history_list(list_id: str, history: dict, field_path: str):
    values = history.get("fields", {}).get(field_path, {}).get("values", [])
    return html.Datalist(id=list_id, children=[html.Option(value=value) for value in values])


def _options(values: list[str]) -> list[dict[str, str]]:
    return [{"label": value, "value": value} for value in values]


def _input_style() -> dict:
    return {"width": "100%", "height": "38px", "padding": "8px 10px", "background": BG, "color": TEXT, "border": f"1px solid {BORDER}", "borderRadius": "8px"}


def _dropdown_style() -> dict:
    return {"background": BG, "color": TEXT}


def _textarea_style() -> dict:
    return {"width": "100%", "minHeight": "78px", "padding": "10px", "background": BG, "color": TEXT, "border": f"1px solid {BORDER}", "borderRadius": "8px", "resize": "vertical"}


def _disabled_input_style() -> dict:
    return {**_input_style(), "opacity": 0.75, "cursor": "not-allowed"}


def _button_style() -> dict:
    return {"background": ACCENT, "color": BG, "border": "none", "padding": "10px 16px", "fontWeight": "600", "borderRadius": "8px", "cursor": "pointer"}


def _secondary_button_style(*, disabled: bool = False) -> dict:
    return {
        "background": "transparent" if not disabled else CARD,
        "color": SUBTEXT if disabled else TEXT,
        "border": f"1px solid {BORDER}",
        "padding": "10px 16px",
        "fontWeight": "600",
        "borderRadius": "8px",
        "cursor": "not-allowed" if disabled else "pointer",
        "opacity": 0.65 if disabled else 1.0,
    }


def _field_badge_style(required: bool) -> dict:
    return {
        "padding": "2px 8px",
        "borderRadius": "999px",
        "fontSize": "11px",
        "fontWeight": "600",
        "background": "rgba(96, 165, 250, 0.18)" if required else "rgba(156, 163, 175, 0.18)",
        "color": ACCENT if required else SUBTEXT,
        "whiteSpace": "nowrap",
    }


def _summary_item(label: str, value: str | None) -> html.Div:
    return html.Div(
        [
            html.Div(label, style={"color": SUBTEXT, "fontSize": "12px", "textTransform": "uppercase", "letterSpacing": "0.04em", "marginBottom": "3px"}),
            html.Div(value or "", style={"fontSize": "14px", "lineHeight": "1.35"}),
        ],
        style={"minWidth": 0},
    )


def _stage_badge_style() -> dict:
    return {"padding": "6px 10px", "border": f"1px solid {BORDER}", "borderRadius": "999px", "fontSize": "12px", "fontWeight": "600", "letterSpacing": "0.08em", "color": TEXT, "background": CARD}


def _layout_grid_style() -> dict:
    return {"display": "grid", "gridTemplateColumns": "minmax(260px, 320px) minmax(0, 1fr)", "gap": "14px", "alignItems": "start"}


def _summary_grid_style() -> dict:
    return {"display": "grid", "gridTemplateColumns": "repeat(2, minmax(0, 1fr))", "gap": "10px", "marginBottom": "12px"}


def _compact_info_grid_style() -> dict:
    return {"display": "grid", "gridTemplateColumns": "repeat(auto-fit, minmax(150px, 1fr))", "gap": "10px", "marginBottom": "12px"}


def _form_grid_style() -> dict:
    return {"display": "grid", "gridTemplateColumns": "repeat(auto-fit, minmax(210px, 1fr))", "gap": "10px 12px"}


def _open_metadata_window(url: str) -> bool:
    command = _build_pywebview_command(url)
    if command is None:
        return False

    popen_kwargs = {
        "stdout": subprocess.DEVNULL,
        "stderr": subprocess.DEVNULL,
        "stdin": subprocess.DEVNULL,
        "close_fds": True,
        "start_new_session": True,
    }
    if sys.platform.startswith("win"):
        popen_kwargs["creationflags"] = getattr(subprocess, "DETACHED_PROCESS", 0) | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)

    try:
        subprocess.Popen(command, **popen_kwargs)
    except OSError:
        return False
    return True


def _build_pywebview_command(url: str) -> list[str] | None:
    if importlib.util.find_spec("webview") is None:
        return None
    return [sys.executable, "-m", "multibios.apps.metadata_form", "--pywebview-url", url]


def _run_pywebview_window(url: str) -> None:
    try:
        import webview  # type: ignore

        api = _PyWebViewApi()
        window = webview.create_window(
            "MultiBiOS Metadata Form",
            url,
            width=1180,
            height=900,
            resizable=True,
            text_select=True,
            js_api=api,
        )
        api.attach_window(window)
        if hasattr(window, "events") and hasattr(window.events, "closing"):
            window.events.closing += api.on_closing
        webview.start()
    except Exception:
        webbrowser.open(url)


class _PyWebViewApi:
    def __init__(self) -> None:
        self._window = None
        self._allow_close = False

    def attach_window(self, window) -> None:
        self._window = window

    def on_closing(self) -> bool | None:
        if not self._allow_close:
            return False
        return None

    def close_window(self) -> None:
        if self._window is not None:
            self._allow_close = True
            self._window.destroy()


def _pre_stage_modal(pre: dict, history: dict) -> html.Div:
    experiment_date = str(pre.get("experiment_date") or "")
    last_fly_id = last_fly_id_for_date(history, experiment_date)
    previous_fly_ids = previous_fly_ids_for_date(history, experiment_date)
    same_disabled = last_fly_id is None
    previous_disabled = not previous_fly_ids
    return html.Div(
        [
            html.Div(
                [
                    html.H2("Is this the same fly?", style={"margin": "0 0 8px 0", "fontSize": "22px"}),
                    html.P(
                        _fly_modal_prompt(history, experiment_date=experiment_date),
                        style={"color": SUBTEXT, "margin": "0 0 16px 0", "fontSize": "14px", "lineHeight": "1.5"},
                    ),
                    html.Div(
                        [
                            html.Button("Same fly", id="pre-fly-same-button", disabled=same_disabled, style=_secondary_button_style(disabled=same_disabled)),
                            html.Button("Previous fly", id="pre-fly-previous-button", disabled=previous_disabled, style=_secondary_button_style(disabled=previous_disabled)),
                            html.Button("New fly", id="pre-fly-new-button", style=_button_style()),
                        ],
                        style={"display": "flex", "gap": "10px"},
                    ),
                ],
                style={"width": "min(460px, 92vw)", "background": CARD, "border": f"1px solid {BORDER}", "borderRadius": "14px", "padding": "20px", "boxShadow": "0 24px 48px rgba(0, 0, 0, 0.35)"},
            ),
        ],
        id="pre-fly-modal",
        style=_modal_style(),
    )


def _fly_modal_prompt(history: dict, *, experiment_date: str) -> str:
    last_fly_id = last_fly_id_for_date(history, experiment_date)
    previous_fly_ids = previous_fly_ids_for_date(history, experiment_date)
    if last_fly_id is None:
        return f"No fly has been recorded yet for {experiment_date}. Start with fly ID 1 for this run."
    if previous_fly_ids:
        return f"The last fly recorded for {experiment_date} was fly ID {last_fly_id}. You can reuse it, pick an earlier same-day fly, or start a new one."
    return f"The last fly recorded for {experiment_date} was fly ID {last_fly_id}. Choose whether this run reuses that fly or starts a new one."


def _pre_save_confirmation_modal() -> html.Div:
    return html.Div(
        [
            html.Div(
                [
                    html.H2("Confirm Before Starting", style={"margin": "0 0 8px 0", "fontSize": "22px"}),
                    html.P("Review the required fields before the experiment proceeds.", style={"color": SUBTEXT, "margin": "0 0 16px 0", "fontSize": "14px"}),
                    html.Div(id="pre-save-confirm-summary", style={"display": "grid", "gap": "8px", "marginBottom": "16px"}),
                    html.Div(
                        [
                            html.P(id="pre-new-terms-message", style={"margin": "0 0 10px 0", "fontSize": "13px", "lineHeight": "1.5", "color": TEXT}),
                            dcc.Checklist(
                                id="pre-new-terms-confirm",
                                options=[{"label": "I confirm these new values should be added to the metadata history log.", "value": "confirmed"}],
                                value=[],
                                inputStyle={"marginRight": "8px"},
                                labelStyle={"display": "inline-flex", "alignItems": "center", "lineHeight": "1.4"},
                            ),
                        ],
                        id="pre-new-terms-confirm-field",
                        style=_hidden_modal_style(),
                    ),
                    html.Div(
                        [
                            html.Button("Back", id="pre-save-cancel-button", style=_secondary_button_style()),
                            html.Button("Confirm And Start", id="confirm-save-button", style=_button_style()),
                        ],
                        style={"display": "flex", "gap": "10px", "justifyContent": "flex-end"},
                    ),
                ],
                style={"width": "min(560px, 94vw)", "background": CARD, "border": f"1px solid {BORDER}", "borderRadius": "14px", "padding": "20px", "boxShadow": "0 24px 48px rgba(0, 0, 0, 0.35)"},
            ),
        ],
        id="pre-save-confirm-modal",
        style=_hidden_modal_style(),
    )


def _modal_style() -> dict:
    return {"position": "fixed", "inset": "0", "display": "flex", "alignItems": "center", "justifyContent": "center", "padding": "24px", "background": "rgba(17, 24, 39, 0.82)", "backdropFilter": "blur(4px)", "zIndex": "1000"}


def _hidden_modal_style() -> dict:
    return {"display": "none"}


def _prefill_record_from_history(*, record: dict, history: dict, stage: str) -> dict:
    updated = deepcopy(record)
    pre = updated.setdefault("pre_experiment", {})
    post = updated.setdefault("post_experiment", {})
    age = pre.setdefault("age", {})
    starvation = pre.setdefault("starvation", {})
    experiment_date = str(pre.get("experiment_date") or "")
    daily_defaults = daily_pre_defaults_for_date(history, experiment_date)

    operator = _prefill_value(pre.get("operator") or updated.get("entered_by"), history, "pre_experiment.operator", preferred_value=daily_defaults.get("pre_experiment.operator"))
    pre["operator"] = operator
    updated["entered_by"] = updated.get("entered_by") or operator
    pre["species"] = _prefill_value(pre.get("species"), history, "pre_experiment.species", preferred_value=daily_defaults.get("pre_experiment.species"))
    pre["genotype"] = _prefill_value(pre.get("genotype"), history, "pre_experiment.genotype", preferred_value=daily_defaults.get("pre_experiment.genotype"))
    pre["hemisphere"] = _prefill_value(pre.get("hemisphere"), history, "pre_experiment.hemisphere", preferred_value=daily_defaults.get("pre_experiment.hemisphere"), fallback="unknown", treat_unknown_as_missing=True)
    age["value"] = _prefill_value(age.get("value"), history, "pre_experiment.age.value", preferred_value=daily_defaults.get("pre_experiment.age.value"))
    age["unit"] = _prefill_value(age.get("unit"), history, "pre_experiment.age.unit", preferred_value=daily_defaults.get("pre_experiment.age.unit"), fallback="unknown", treat_unknown_as_missing=True)
    starvation["value"] = _prefill_value(starvation.get("value"), history, "pre_experiment.starvation.value", preferred_value=daily_defaults.get("pre_experiment.starvation.value"))
    starvation["unit"] = _prefill_value(starvation.get("unit"), history, "pre_experiment.starvation.unit", preferred_value=daily_defaults.get("pre_experiment.starvation.unit"), fallback="unknown", treat_unknown_as_missing=True)
    pre["volumetric"] = _prefill_value(pre.get("volumetric"), history, "pre_experiment.volumetric", preferred_value=daily_defaults.get("pre_experiment.volumetric"), fallback="unknown", treat_unknown_as_missing=True)
    pre["stimulus_modality"] = _prefill_value(pre.get("stimulus_modality"), history, "pre_experiment.stimulus_modality", preferred_value=daily_defaults.get("pre_experiment.stimulus_modality"))
    pre["rig_temperature_c"] = _prefill_value(pre.get("rig_temperature_c"), history, "pre_experiment.rig_temperature_c", preferred_value=daily_defaults.get("pre_experiment.rig_temperature_c"))
    pre["humidity_percent"] = _prefill_value(pre.get("humidity_percent"), history, "pre_experiment.humidity_percent", preferred_value=daily_defaults.get("pre_experiment.humidity_percent"))

    existing_fly_id = pre.get("fly_id")
    if existing_fly_id is None:
        existing_fly_id = _coerce_int(pre.get("fly_num"))
    if existing_fly_id is None and stage == "pre":
        pre["fly_id"] = suggested_fly_id(
            history,
            experiment_date=experiment_date,
            same_fly=last_fly_id_for_date(history, experiment_date) is not None,
        )
    elif existing_fly_id is not None:
        pre["fly_id"] = existing_fly_id

    return updated


def _prefill_value(current_value, history: dict, field_path: str, preferred_value=None, fallback=None, treat_unknown_as_missing: bool = False):
    if current_value is not None and str(current_value).strip() != "":
        if not (treat_unknown_as_missing and str(current_value).strip().lower() == "unknown"):
            return current_value
    if preferred_value is not None and str(preferred_value).strip() != "":
        if not (treat_unknown_as_missing and str(preferred_value).strip().lower() == "unknown"):
            return preferred_value
    recent_value = recent_history_value(history, field_path)
    if recent_value is not None:
        return recent_value
    return fallback


def _default_fly_choice(*, pre: dict, last_fly_id: int | None) -> str:
    current_fly_id = _coerce_int(pre.get("fly_id"))
    if current_fly_id is not None and last_fly_id is not None and current_fly_id == last_fly_id:
        return "same"
    if current_fly_id is not None and last_fly_id is not None and current_fly_id < last_fly_id:
        return "previous"
    if current_fly_id is not None:
        return "new"
    return "same" if last_fly_id is not None else "new"


def _fly_choice_help_text(history: dict, *, experiment_date: str, choice: str | None, selected_previous_fly_id: int | None = None) -> str:
    last_fly_id = last_fly_id_for_date(history, experiment_date)
    if choice == "same" and last_fly_id is not None:
        return f"Reusing fly ID {last_fly_id} for {experiment_date}."
    if choice == "previous" and selected_previous_fly_id is not None:
        return f"Reusing earlier same-day fly ID {selected_previous_fly_id} for {experiment_date}."
    next_fly_id = suggested_fly_id(history, experiment_date=experiment_date, same_fly=False)
    return f"Assigning fly ID {next_fly_id} for {experiment_date}."


def _build_pre_run_confirmation_summary(**values):
    items = [
        ("Run ID", values.get("run_id"), True),
        ("Entered by", values.get("entered_by"), False),
        ("Fly mode", values.get("fly_choice"), True),
        ("Fly ID", values.get("fly_id"), True),
        ("Species", values.get("species"), True),
        ("Genotype", values.get("genotype"), True),
        ("Hemisphere", values.get("hemisphere"), True),
        ("Age", _join_value_unit(values.get("age_value"), values.get("age_unit")), False),
        ("Starvation", _join_value_unit(values.get("starvation_value"), values.get("starvation_unit")), False),
        ("Volumetric", values.get("volumetric"), False),
        ("Stimulus modality", values.get("stimulus_modality"), True),
        ("Rig temperature", values.get("rig_temperature"), False),
        ("Humidity", values.get("humidity"), False),
    ]
    return [
        html.Div(
            [
                html.Div(label, style={"color": SUBTEXT, "fontSize": "12px", "textTransform": "uppercase", "letterSpacing": "0.04em"}),
                html.Div(_stringify(value) or "Not provided", style={"fontSize": "14px", "fontWeight": "600"}),
                html.Span("Required" if required else "Optional", style=_field_badge_style(required)),
            ],
            style={"display": "grid", "gridTemplateColumns": "minmax(120px, 1fr) minmax(0, 2fr) auto", "gap": "10px", "alignItems": "center", "padding": "8px 10px", "border": f"1px solid {BORDER}", "borderRadius": "10px", "background": BG},
        )
        for label, value, required in items
    ]


def _resolve_controlled_value(selection_value: str | None, new_value: str | None) -> str | None:
    if selection_value == "__new__":
        normalized_new_value = (new_value or "").strip()
        return normalized_new_value or None
    normalized_selection = (selection_value or "").strip()
    return normalized_selection or None


def _controlled_select_state(current_value: str | None, history: dict, field_path: str) -> tuple[str, str, list[dict[str, str]]]:
    options = _controlled_options(history, field_path)
    normalized_current = (current_value or "").strip()
    known_values = [option["value"] for option in options if option["value"] != "__new__"]
    if normalized_current and normalized_current in set(known_values):
        return normalized_current, "", options
    if normalized_current:
        return "__new__", normalized_current, options
    if known_values:
        return known_values[0], "", options
    return "__new__", "", options


def _controlled_options(history: dict, field_path: str) -> list[dict[str, str]]:
    values = history.get("fields", {}).get(field_path, {}).get("values", [])
    options = [{"label": str(value), "value": str(value)} for value in values if str(value).strip()]
    options.append({"label": "Add new...", "value": "__new__"})
    return options


def _controlled_new_field_style(selection_value: str | None) -> dict:
    if selection_value == "__new__":
        return {"minWidth": 0}
    return {"display": "none"}


def _pending_new_terms(history: dict, candidates: dict[str, str | None]) -> list[tuple[str, str]]:
    pending: list[tuple[str, str]] = []
    for field_path, value in candidates.items():
        normalized_value = (value or "").strip()
        if not normalized_value:
            continue
        existing_values = history.get("fields", {}).get(field_path, {}).get("values", [])
        existing_normalized = {str(existing).strip().lower() for existing in existing_values if str(existing).strip()}
        if normalized_value.lower() not in existing_normalized:
            pending.append((_controlled_field_label(field_path), normalized_value))
    return pending


def _pending_new_terms_message(pending_new_terms: list[tuple[str, str]]) -> str:
    if not pending_new_terms:
        return ""
    details = ", ".join(f"{label}: {value}" for label, value in pending_new_terms)
    return f"New metadata values will be added to the history log if you confirm: {details}."


def _new_terms_confirm_style(visible: bool) -> dict:
    if visible:
        return {"marginBottom": "16px", "padding": "12px", "border": f"1px solid {BORDER}", "borderRadius": "10px", "background": BG}
    return _hidden_modal_style()


def _controlled_field_label(field_path: str) -> str:
    if field_path == "pre_experiment.species":
        return "Species"
    if field_path == "pre_experiment.stimulus_modality":
        return "Stimulus modality"
    return field_path


def _join_value_unit(value, unit) -> str:
    value_text = _stringify(value)
    unit_text = _stringify(unit)
    if value_text and unit_text and unit_text.lower() != "unknown":
        return f"{value_text} {unit_text}"
    return value_text or ("" if unit_text.lower() == "unknown" else unit_text)


def _microscopy_guidance_block(expected_imaging_periods: int) -> html.Div | None:
    if expected_imaging_periods <= 0:
        return None
    return html.Div(
        [
            html.Div("Microscopy guidance", style={"fontSize": "13px", "fontWeight": "600", "marginBottom": "8px"}),
            html.Div(
                [
                    _readonly_row("Expected imaging periods", str(expected_imaging_periods)),
                    _readonly_row("PrairieView setup", f"Set iterations to {expected_imaging_periods} before starting the protocol."),
                ],
                style=_compact_info_grid_style(),
            ),
        ],
        style={"padding": "12px", "border": f"1px solid {BORDER}", "borderRadius": "10px", "marginBottom": "12px", "background": BG},
    )


def _post_imaging_dataset_block(post: dict, expected_imaging_periods: int) -> html.Div | None:
    if expected_imaging_periods <= 0:
        return html.Div(
            [
                dcc.Input(id="post-imaging-dataset-source", value=post.get("imaging_dataset_source_path") or ""),
                dcc.Input(id="post-imaging-dataset-relative-path", value=post.get("imaging_dataset_relative_path") or ""),
                dcc.Input(id="post-imaging-acquisition-type", value=post.get("imaging_acquisition_type") or ""),
                dcc.Input(id="post-imaging-num-rois", value=_stringify(post.get("imaging_num_rois"))),
                dcc.Input(id="post-imaging-num-channels", value=_stringify(post.get("imaging_num_channels"))),
                dcc.Input(id="post-imaging-num-planes", value=_stringify(post.get("imaging_num_planes"))),
                html.Div(id="post-imaging-dataset-status"),
                html.Div(id="post-imaging-num-planes-field"),
                html.Span(id="post-imaging-num-planes-required"),
                html.Button("", id="post-select-imaging-dataset", n_clicks=0),
            ],
            style={"display": "none"},
        )
    copied_path = post.get("imaging_dataset_relative_path") or ""
    source_path = post.get("imaging_dataset_source_path") or ""
    acquisition_type = post.get("imaging_acquisition_type") or None
    num_rois = _stringify(post.get("imaging_num_rois"))
    num_channels = _stringify(post.get("imaging_num_channels"))
    num_planes = _stringify(post.get("imaging_num_planes"))
    status_text = f"Current copied dataset: {copied_path}" if copied_path else "Select the completed PrairieView dataset and copy it into this run before saving metadata."
    return html.Div(
        [
            html.Div("Microscopy dataset", style={"fontSize": "13px", "fontWeight": "600", "marginBottom": "8px"}),
            html.Div(
                [
                    _readonly_row("Expected imaging periods", str(expected_imaging_periods)),
                    _readonly_row("Copied dataset", copied_path or "Not copied yet"),
                ],
                style=_compact_info_grid_style(),
            ),
            html.Div(
                [
                    html.Button("Select And Copy PrairieView Dataset", id="post-select-imaging-dataset", style=_button_style()),
                    html.Div(id="post-imaging-dataset-status", children=status_text, style={"color": SUBTEXT, "fontSize": "13px"}),
                ],
                style={"display": "flex", "alignItems": "center", "gap": "12px", "flexWrap": "wrap", "marginTop": "10px", "marginBottom": "10px"},
            ),
            html.Div(
                [
                    _field_block("Selected source path", dcc.Input(id="post-imaging-dataset-source", value=source_path, readOnly=True, style=_disabled_input_style()), required=False),
                    _field_block("Copied dataset path", dcc.Input(id="post-imaging-dataset-relative-path", value=copied_path, readOnly=True, style=_disabled_input_style()), required=True),
                    _field_block("Acquisition type", dcc.Dropdown(id="post-imaging-acquisition-type", options=[{"label": "Single-plane", "value": "single_plane"}, {"label": "Volumetric", "value": "volumetric"}], value=acquisition_type, clearable=False, style=_dropdown_style()), required=True),
                    _field_block("Number of ROIs", dcc.Input(id="post-imaging-num-rois", value=num_rois, type="number", min=1, step=1, style=_input_style()), required=True),
                    _field_block("Number of channels", dcc.Input(id="post-imaging-num-channels", value=num_channels, type="number", min=1, step=1, style=_input_style()), required=True),
                    _field_block("Number of planes", dcc.Input(id="post-imaging-num-planes", value=num_planes, type="number", min=1, step=1, style=_input_style()), required=acquisition_type == "volumetric", container_id="post-imaging-num-planes-field", badge_id="post-imaging-num-planes-required", style={} if acquisition_type == "volumetric" else {"display": "none"}),
                ],
                style=_form_grid_style(),
            ),
        ],
        style={"padding": "12px", "border": f"1px solid {BORDER}", "borderRadius": "10px", "marginBottom": "12px", "background": BG},
    )


def _initial_dialog_directory(current_source: str | None) -> str | None:
    normalized = _strip_or_none(current_source)
    if normalized is None:
        return None
    path = Path(normalized)
    if path.exists() and path.is_dir():
        return str(path.parent)
    return None


def _select_directory_dialog(*, title: str, initial_dir: str | None = None) -> Path | None:
    try:
        import tkinter as tk
        from tkinter import filedialog
    except Exception as exc:
        raise RuntimeError("Tk directory picker is unavailable on this system") from exc

    root = tk.Tk()
    root.withdraw()
    try:
        root.attributes("-topmost", True)
    except Exception:
        pass
    try:
        selected = filedialog.askdirectory(title=title, initialdir=initial_dir or str(Path.home()), parent=root)
    finally:
        root.destroy()
    if not selected:
        return None
    selected_path = Path(selected)
    return selected_path if selected_path.exists() else None


def _copy_imaging_dataset_into_run(*, record_path: Path, source_dir: Path) -> Path:
    if not source_dir.exists() or not source_dir.is_dir():
        raise FileNotFoundError(f"Imaging dataset directory not found: {source_dir}")
    run_dir = record_path.parent.parent
    destination_root = run_dir / "recorded" / "microscopy"
    destination_root.mkdir(parents=True, exist_ok=True)
    destination = destination_root / source_dir.name
    if destination.exists():
        if destination.is_dir():
            shutil.rmtree(destination)
        else:
            destination.unlink()
    shutil.copytree(source_dir, destination)
    return destination


def _strip_or_none(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None


def _index_string() -> str:
    return """<!DOCTYPE html>
<html>
<head>
{%metas%}
<title>{%title%}</title>
{%favicon%}
{%css%}
<style>
  body { margin: 0; background: #111827; }
  * { box-sizing: border-box; }
  .Select-control, .Select-menu-outer { background-color: #111827 !important; color: #f9fafb !important; border-color: #374151 !important; }
  .Select-value-label, .Select-option, .VirtualizedSelectOption { color: #f9fafb !important; }
  .Select-option.is-focused, .VirtualizedSelectFocusedOption { background-color: #374151 !important; }
</style>
</head>
<body>{%app_entry%}<footer>{%config%}{%scripts%}{%renderer%}</footer></body>
</html>"""


def _coerce_number(value: str | None) -> float | None:
    if value is None or str(value).strip() == "":
        return None
    return float(value)


def _coerce_int(value) -> int | None:
    if value is None or str(value).strip() == "":
        return None
    return int(float(value))


def _split_lines(value: str | None) -> list[str]:
    if value is None:
        return []
    return [line.strip() for line in str(value).splitlines() if line.strip()]


def _stringify(value) -> str:
    return "" if value is None else str(value)


if __name__ == "__main__":
    main()