param(
    [string]$ProtocolPath = "protocols/serial_valve_round_independent.yaml",
    [string]$HardwarePath = "config/hardware.yaml",
    [string]$ExperimentPath = "config/experiment_config.yaml",
    [switch]$DryRun,
    [switch]$VerboseOutput,
    [string]$CondaEnv = "multibios-blackfly",
    [string]$CondaExe = "C:/ProgramData/miniconda3/Scripts/conda.exe",
    [int]$Seed
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
. (Join-Path $PSScriptRoot "_rig_python.ps1")

$pythonArgs = @(
    "-m", "multibios.experiment",
    "--protocol", $ProtocolPath,
    "--hardware", $HardwarePath,
    "--experiment", $ExperimentPath
)

if ($DryRun) {
    $pythonArgs += "--dry-run"
}
if ($VerboseOutput) {
    $pythonArgs += "--verbose"
}
if ($PSBoundParameters.ContainsKey("Seed")) {
    $pythonArgs += @("--seed", $Seed.ToString())
}

Write-Host "Running open-loop valve protocol test with $ProtocolPath..." -ForegroundColor Yellow
Write-Host "This path exercises the Teensy open-loop controller and saves a per-run serial transcript next to the experiment data." -ForegroundColor DarkYellow
Invoke-RigPython -CondaEnv $CondaEnv -CondaExe $CondaExe -PythonArgs $pythonArgs
if ($LASTEXITCODE -ne 0) {
    throw "Open-loop valve protocol test failed with exit code $LASTEXITCODE"
}