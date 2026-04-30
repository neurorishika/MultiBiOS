import lazy_loader


__getattr__, __dir__, __all__ = lazy_loader.attach(
    __name__,
    submodules={
        "protocol",
        "run_protocol",
        "viz_protocol",
        "apps",
        "daq_triggers",
        "teensy_controller",
        "experiment",
        "serial",
        "blackfly",
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
        "run_protocol": [
            "RunProtocolConfig",
            "ExperimentCallback",
            "load_run_protocol_config",
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
    "apps",
    # integration (computer-timebase + serial)
    "daq_triggers",
    "DAQTriggerManager",
    "TriggerConfig",
    "build_trigger_waveform",
    "RunProtocolConfig",
    "load_run_protocol_config",
    "teensy_controller",
    "TeensyController",
    "experiment",
    "serial",
    "blackfly",
    "ExperimentRunner",
    "ExperimentConfig",
    "ExperimentCallback",
    "load_experiment_config",
]
