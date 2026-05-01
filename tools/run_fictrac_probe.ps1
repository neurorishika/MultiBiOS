param(
    [string]$CondaExe = "C:/ProgramData/miniconda3/Scripts/conda.exe",
    [string]$CondaEnv = "multibios-blackfly",
    [string]$HardwarePath = "config/hardware.yaml",
    [string]$ConfigPath = "",
    [string]$FicTracBin = "C:/Rishika/MultiBiOS/assets/fictrac-spinnaker/fictrac-spinnaker.exe",
    [string]$ConsoleOutput = "fictrac_probe_output.txt",
    [int]$Frames = 5,
    [double]$Fps = 30.0,
    [switch]$NoTriggerTrain
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$resolvedHardwarePath = if ([System.IO.Path]::IsPathRooted($HardwarePath)) { $HardwarePath } else { Join-Path $repoRoot $HardwarePath }
$resolvedConfigPath = if ([string]::IsNullOrWhiteSpace($ConfigPath)) {
    Join-Path (Split-Path -Parent $resolvedHardwarePath) "config_camera.txt"
} elseif ([System.IO.Path]::IsPathRooted($ConfigPath)) {
    $ConfigPath
} else {
    Join-Path $repoRoot $ConfigPath
}
$triggerScript = Join-Path $repoRoot "tests/continuous_camera_trigger.py"
. (Join-Path $PSScriptRoot "_rig_python.ps1")

$triggerProcess = $null

try {
    if (-not $NoTriggerTrain) {
        if (-not (Test-Path $triggerScript)) {
            throw "Trigger script not found: $triggerScript"
        }

        if (Test-UseActiveRigPython -CondaEnv $CondaEnv) {
            Write-Host "Using active Python from conda env '$CondaEnv'." -ForegroundColor Yellow
        }

        Write-Host "Applying rig Blackfly defaults from $HardwarePath..." -ForegroundColor Yellow
        Invoke-RigPython -CondaEnv $CondaEnv -CondaExe $CondaExe -PythonArgs @(
            "-m", "multibios.blackfly.setup_daq_mode",
            "--hardware", $resolvedHardwarePath
        )
        if ($LASTEXITCODE -ne 0) {
            throw "Failed to apply rig Blackfly defaults via setup_daq_mode (exit $LASTEXITCODE). Aborting instead of continuing with partial camera setup."
        }

        Write-Host "Starting camera trigger train at $Fps fps..." -ForegroundColor Yellow
        $triggerProcess = Start-RigPythonProcess -RepoRoot $repoRoot -CondaEnv $CondaEnv -CondaExe $CondaExe -PythonArgs @(
            $triggerScript,
            "--fps", $Fps.ToString([System.Globalization.CultureInfo]::InvariantCulture)
        )
        Start-Sleep -Milliseconds 1200
        if ($triggerProcess.HasExited) {
            throw "Continuous trigger process exited immediately with code $($triggerProcess.ExitCode)."
        }
    }

    Write-Host "Running FicTrac live probe..." -ForegroundColor Yellow
    Invoke-RigPython -CondaEnv $CondaEnv -CondaExe $CondaExe -PythonArgs @(
        (Join-Path $repoRoot "tests/fictrac_live_probe.py"),
        "--config", $resolvedConfigPath,
        "--hardware", $resolvedHardwarePath,
        "--fictrac-bin", $FicTracBin,
        "--console-output", $ConsoleOutput,
        "--frames", $Frames.ToString()
    )
    if ($LASTEXITCODE -ne 0) {
        throw "FicTrac live probe failed with exit code $LASTEXITCODE"
    }
}
finally {
    if ($triggerProcess -and -not $triggerProcess.HasExited) {
        Write-Host "Stopping camera trigger train..." -ForegroundColor Yellow
        Stop-Process -Id $triggerProcess.Id -Force
    }
}