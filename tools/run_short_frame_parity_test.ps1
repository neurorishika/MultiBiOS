param(
    [string]$CondaExe = "C:/ProgramData/miniconda3/Scripts/conda.exe",
    [string]$CondaEnv = "multibios-blackfly",
    [string]$HardwarePath = "config/hardware.yaml",
    [string]$ProtocolPath = "protocols/short_protocol.yaml",
    [string]$OutRoot = "data/runs",
    [switch]$VerboseOutput,
    [switch]$Progress,
    [int]$ProgressInterval = 100,
    [switch]$KeepRawChunks
)

$ErrorActionPreference = "Stop"

$params = @{
    ProtocolPath = $ProtocolPath
    TestName = "short-run frame-count parity test"
    CondaExe = $CondaExe
    CondaEnv = $CondaEnv
    HardwarePath = $HardwarePath
    OutRoot = $OutRoot
    VerboseOutput = $VerboseOutput
    Progress = $Progress
    ProgressInterval = $ProgressInterval
    KeepRawChunks = $KeepRawChunks
}

& (Join-Path $PSScriptRoot "run_frame_parity_test.ps1") @params