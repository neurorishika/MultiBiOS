import lazy_loader


__getattr__, __dir__, __all__ = lazy_loader.attach(
    __name__,
    submodules={
        "schema",
    },
    submod_attrs={
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
    "BIG_STATE_CODE",
    "CompileError",
    "ProtocolCompiler",
    "SMALL_STATE_CODE",
    "TimingConfig",
    "schema",
]
