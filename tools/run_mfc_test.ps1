param(
    [string]$CondaExe = "C:/ProgramData/miniconda3/Scripts/conda.exe",
    [string]$CondaEnv = "multibios-blackfly",
    [string]$HardwarePath = "config/hardware.yaml",
    [ValidateSet("monitor", "sweep")]
    [string]$Mode = "monitor",
    [string[]]$Set = @(),
    [double]$Interval = 0.5,
    [switch]$NoZeroOnExit,
    [double[]]$Levels = @(0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 0.0),
    [double]$Dwell = 0.5,
    [double]$Tolerance = 0.1,
    [string[]]$Channels = @(),
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
. (Join-Path $PSScriptRoot "_rig_python.ps1")

$pythonArgs = @(
    (Join-Path $repoRoot "tests/mfc_analog_test.py"),
    "--hardware", $HardwarePath
)

if ($DryRun) {
    $pythonArgs += "--dry-run"
}

$pythonArgs += $Mode

if ($Mode -eq "monitor") {
    if ($Set.Count -gt 0) {
        $pythonArgs += "--set"
        $pythonArgs += $Set
    }
    $pythonArgs += @("--interval", $Interval.ToString([System.Globalization.CultureInfo]::InvariantCulture))
    if ($NoZeroOnExit) {
        $pythonArgs += "--no-zero-on-exit"
    }
} else {
    if ($Levels.Count -gt 0) {
        $pythonArgs += "--levels"
        $pythonArgs += ($Levels | ForEach-Object { $_.ToString([System.Globalization.CultureInfo]::InvariantCulture) })
    }
    $pythonArgs += @(
        "--dwell", $Dwell.ToString([System.Globalization.CultureInfo]::InvariantCulture),
        "--tolerance", $Tolerance.ToString([System.Globalization.CultureInfo]::InvariantCulture)
    )
    if ($Channels.Count -gt 0) {
        $pythonArgs += "--channels"
        $pythonArgs += $Channels
    }
}

Write-Host "Running MFC $Mode test..." -ForegroundColor Yellow
Invoke-RigPython -CondaEnv $CondaEnv -CondaExe $CondaExe -PythonArgs $pythonArgs
if ($LASTEXITCODE -ne 0) {
    throw "MFC test failed with exit code $LASTEXITCODE"
}