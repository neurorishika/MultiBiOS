param(
    [string]$ConfigPath = "C:/Rishika/legacy/fictrac_pybmt/config_camera.txt",
    [string]$CondaExe = "C:/ProgramData/miniconda3/Scripts/conda.exe",
    [string]$CondaEnv = "multibios-blackfly",
    [string]$HardwarePath = "config/hardware.yaml",
    [double]$Fps = 30.0,
    [switch]$NoTriggerTrain
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$fictracRuntimeDir = Join-Path $repoRoot "assets/fictrac-spinnaker"
$configGuiExe = Join-Path $fictracRuntimeDir "configGui.exe"
$setupScript = "-m multibios.blackfly.setup_daq_mode"
$triggerScript = Join-Path $repoRoot "tests/continuous_camera_trigger.py"
$runtimePathPrefix = @(
    $fictracRuntimeDir,
    "C:/Program Files/Teledyne/Spinnaker/bin64/vs2015",
    "C:/Program Files/Point Grey Research/FlyCapture2/bin64/vs2015"
) -join ";"
. (Join-Path $PSScriptRoot "_rig_python.ps1")

if (-not (Test-Path $ConfigPath)) {
    throw "Config file not found: $ConfigPath"
}

if (-not (Test-Path $configGuiExe)) {
    throw "configGui.exe not found: $configGuiExe"
}

$triggerProcess = $null
$originalPath = $env:PATH
$rigDefaultsApplied = $false
$useActivePython = Test-UseActiveRigPython -CondaEnv $CondaEnv

try {
    if (-not $NoTriggerTrain) {
        if (-not (Test-Path $triggerScript)) {
            throw "Trigger script not found: $triggerScript"
        }

        if ($useActivePython) {
            Write-Host "Using active Python from conda env '$CondaEnv'." -ForegroundColor Yellow
        }

        Write-Host "Applying rig Blackfly defaults from $HardwarePath..." -ForegroundColor Yellow
        $setupArgs = @(
            "-m", "multibios.blackfly.setup_daq_mode",
            "--hardware", $HardwarePath
        )
        Invoke-RigPython -CondaEnv $CondaEnv -CondaExe $CondaExe -PythonArgs $setupArgs
        if ($LASTEXITCODE -ne 0) {
            throw "Failed to apply Blackfly rig defaults via setup_daq_mode (exit $LASTEXITCODE). Aborting instead of continuing with partial camera geometry."
        } else {
            $rigDefaultsApplied = $true
        }

        Write-Host "Starting camera trigger train at $Fps fps..." -ForegroundColor Yellow
        $triggerArgs = @(
            $triggerScript,
            "--fps", $Fps.ToString([System.Globalization.CultureInfo]::InvariantCulture)
        )
        $triggerProcess = Start-RigPythonProcess -RepoRoot $repoRoot -CondaEnv $CondaEnv -CondaExe $CondaExe -PythonArgs $triggerArgs
        Start-Sleep -Milliseconds 1200
        if ($triggerProcess.HasExited) {
            throw "Continuous trigger process exited immediately with code $($triggerProcess.ExitCode)."
        }
    }

    Write-Host "Launching FicTrac config UI..." -ForegroundColor Yellow
    Write-Host "  Config: $ConfigPath" -ForegroundColor Gray
    Write-Host "  configGui is interactive. Prompts such as 'keep existing sphere ROI configuration' are expected." -ForegroundColor Gray
    if (-not $NoTriggerTrain) {
        Write-Host "  Trigger train PID: $($triggerProcess.Id)" -ForegroundColor Gray
        Write-Host "  Hardware defaults: $HardwarePath" -ForegroundColor Gray
    }

    Push-Location $repoRoot
    try {
        $env:PATH = "$runtimePathPrefix;$originalPath"
        & $configGuiExe $ConfigPath
        $configGuiExitCode = $LASTEXITCODE
    }
    finally {
        $env:PATH = $originalPath
        Pop-Location
    }

    if ($configGuiExitCode -ne 0) {
        throw "configGui exited with code $configGuiExitCode"
    }
}
finally {
    if ($triggerProcess -and -not $triggerProcess.HasExited) {
        Write-Host "Stopping camera trigger train..." -ForegroundColor Yellow
        Stop-Process -Id $triggerProcess.Id -Force
    }
}