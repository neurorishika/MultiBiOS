import lazy_loader


__getattr__, __dir__, __all__ = lazy_loader.attach(
    __name__,
    submodules={
        "control_plan",
        "schema",
    },
    submod_attrs={
        "control_plan": [
            "CompiledControlPlan",
            "TimelineEvent",
            "compile_control_plan",
            "write_control_plan_csv",
        ],
        "schema": [
            "BIG_STATE_CODE",
            "CompileError",
            "ProtocolCompiler",
            "SMALL_STATE_CODE",
            "TimingConfig",
        ],
    },
)

__all__ = [
    "CompiledControlPlan",
    "BIG_STATE_CODE",
    "CompileError",
    "ProtocolCompiler",
    "SMALL_STATE_CODE",
    "TimingConfig",
    "TimelineEvent",
    "compile_control_plan",
    "control_plan",
    "schema",
    "write_control_plan_csv",
]
