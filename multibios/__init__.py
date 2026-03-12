import lazy_loader


__getattr__, __dir__, __all__ = lazy_loader.attach(
    __name__,
    submodules={
        "protocol",
        "run_protocol",
        "viz_protocol",
        "alicat_manager",
        "daq_triggers",
        "teensy_controller",
        "experiment",
    },
    submod_attrs={
        "protocol": [
            "BIG_STATE_CODE",
            "CompileError",
            "ProtocolCompiler",
            "SMALL_STATE_CODE",
            "TimingConfig",
            "schema",
        ],
        "alicat_manager": [
            "AlicatManager",
            "scan",
            "load_cache",
            "save_cache",
            "STANDARD_BAUDRATES",
            "UNIT_IDS",
        ],
        "daq_triggers": [
            "DAQTriggerManager",
            "TriggerConfig",
            "build_trigger_waveform",
        ],
        "teensy_controller": [
            "TeensyController",
        ],
        "experiment": [
            "ExperimentRunner",
            "ExperimentConfig",
            "ExperimentCallback",
            "load_experiment_config",
        ],
    },
)

__all__ = [
    # protocol
    "BIG_STATE_CODE",
    "CompileError",
    "ProtocolCompiler",
    "SMALL_STATE_CODE",
    "TimingConfig",
    "protocol",
    "run_protocol",
    "schema",
    "viz_protocol",
    # alicat
    "alicat_manager",
    "AlicatManager",
    "scan",
    "load_cache",
    "save_cache",
    "STANDARD_BAUDRATES",
    "UNIT_IDS",
    # integration (computer-timebase + serial)
    "daq_triggers",
    "DAQTriggerManager",
    "TriggerConfig",
    "build_trigger_waveform",
    "teensy_controller",
    "TeensyController",
    "experiment",
    "ExperimentRunner",
    "ExperimentConfig",
    "ExperimentCallback",
    "load_experiment_config",
]
