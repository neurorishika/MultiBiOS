param(
    [string]$CondaExe = "C:/ProgramData/miniconda3/Scripts/conda.exe",
    [string]$CondaEnv = "multibios-blackfly",
    [string]$HardwarePath = "config/hardware.yaml",
    [string]$ProtocolPath = "protocols/serial_valve_round_independent.yaml",
    [switch]$DryRun,
    [switch]$Interactive,
    [switch]$VerboseOutput,
    [switch]$Progress,
    [int]$ProgressInterval = 100,
    [string]$OutRoot = "data/runs",
    [int]$Seed
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
. (Join-Path $PSScriptRoot "_rig_python.ps1")

$pythonArgs = @(
    "-m", "multibios.run_protocol",
    "--yaml", $ProtocolPath,
    "--hardware", $HardwarePath,
    "--out-root", $OutRoot,
    "--progress-interval", $ProgressInterval.ToString()
)

if ($DryRun) {
    $pythonArgs += "--dry-run"
}
if ($Interactive) {
    $pythonArgs += "--interactive"
}
if ($VerboseOutput) {
    $pythonArgs += "--verbose"
}
if ($Progress) {
    $pythonArgs += "--progress"
}
if ($PSBoundParameters.ContainsKey("Seed")) {
    $pythonArgs += @("--seed", $Seed.ToString())
}

Write-Host "Running valve protocol test with $ProtocolPath..." -ForegroundColor Yellow
Write-Host "Default protocol walks ODOR1->ODOR5 while pulsing left and right valves independently." -ForegroundColor DarkYellow
Invoke-RigPython -CondaEnv $CondaEnv -CondaExe $CondaExe -PythonArgs $pythonArgs
if ($LASTEXITCODE -ne 0) {
    throw "Valve protocol test failed with exit code $LASTEXITCODE"
}