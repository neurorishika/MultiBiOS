import lazy_loader


__getattr__, __dir__, __all__ = lazy_loader.attach(
    __name__,
    submodules={
        "protocol",
        "run_protocol",
        "viz_protocol",
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
    },
)

__all__ = [
    "BIG_STATE_CODE",
    "CompileError",
    "ProtocolCompiler",
    "SMALL_STATE_CODE",
    "TimingConfig",
    "protocol",
    "run_protocol",
    "schema",
    "viz_protocol",
]
