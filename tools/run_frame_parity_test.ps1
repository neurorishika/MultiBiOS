param(
    [Parameter(Mandatory = $true)]
    [string]$ProtocolPath,
    [string]$TestName = "frame parity",
    [string]$CondaExe = "C:/ProgramData/miniconda3/Scripts/conda.exe",
    [string]$CondaEnv = "multibios-blackfly",
    [string]$HardwarePath = "config/hardware.yaml",
    [string]$OutRoot = "data/runs",
    [switch]$VerboseOutput,
    [switch]$Progress,
    [int]$ProgressInterval = 100,
    [switch]$KeepRawChunks
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
. (Join-Path $PSScriptRoot "_rig_python.ps1")

$resolvedOutRoot = if ([System.IO.Path]::IsPathRooted($OutRoot)) {
    $OutRoot
} else {
    Join-Path $repoRoot $OutRoot
}

if (-not (Test-Path $resolvedOutRoot)) {
    New-Item -ItemType Directory -Path $resolvedOutRoot | Out-Null
}

$beforeRunDirs = @{}
Get-ChildItem -Path $resolvedOutRoot -Directory | ForEach-Object {
    $beforeRunDirs[$_.FullName] = $true
}

$pythonArgs = @(
    "-m", "multibios.run_protocol",
    "--yaml", $ProtocolPath,
    "--hardware", $HardwarePath,
    "--out-root", $OutRoot,
    "--progress-interval", $ProgressInterval.ToString()
)

if ($VerboseOutput) {
    $pythonArgs += "--verbose"
}
if ($Progress) {
    $pythonArgs += "--progress"
}

$effectiveHardwarePath = $HardwarePath
$tempHardwarePath = $null
if ($KeepRawChunks) {
    $resolvedHardwarePath = if ([System.IO.Path]::IsPathRooted($HardwarePath)) {
        $HardwarePath
    } else {
        Join-Path $repoRoot $HardwarePath
    }

    if (-not (Test-Path $resolvedHardwarePath)) {
        throw "Hardware config not found: $resolvedHardwarePath"
    }

    $tempHardwarePath = Join-Path $env:TEMP ("hardware.keep-raw.{0}.yaml" -f ([guid]::NewGuid().ToString("N")))
    $content = Get-Content -Path $resolvedHardwarePath -Raw
    $updated = [System.Text.RegularExpressions.Regex]::Replace(
        $content,
        '(?m)^([ \t]*raw_chunk_retention_policy:[ \t]*).*$' ,
        '${1}keep'
    )
    Set-Content -Path $tempHardwarePath -Value $updated -Encoding UTF8
    $effectiveHardwarePath = $tempHardwarePath

    $pythonArgs[5] = $effectiveHardwarePath
}

Write-Host ("Running {0} using {1}..." -f $TestName, $ProtocolPath) -ForegroundColor Yellow
Invoke-RigPython -CondaEnv $CondaEnv -CondaExe $CondaExe -PythonArgs $pythonArgs
if ($LASTEXITCODE -ne 0) {
    if ($tempHardwarePath -and (Test-Path $tempHardwarePath)) {
        Remove-Item $tempHardwarePath -Force -ErrorAction SilentlyContinue
    }
    throw ("{0} protocol run failed with exit code {1}" -f $TestName, $LASTEXITCODE)
}

$afterRunDirs = Get-ChildItem -Path $resolvedOutRoot -Directory | Sort-Object LastWriteTimeUtc, Name
$newRunDir = $afterRunDirs | Where-Object { -not $beforeRunDirs.ContainsKey($_.FullName) } | Select-Object -Last 1
if (-not $newRunDir) {
    $newRunDir = $afterRunDirs | Select-Object -Last 1
}
if (-not $newRunDir) {
    if ($tempHardwarePath -and (Test-Path $tempHardwarePath)) {
        Remove-Item $tempHardwarePath -Force -ErrorAction SilentlyContinue
    }
    throw "Could not determine run directory for parity audit."
}

$auditArgs = @(
    "-m", "multibios.parity_audit",
    "--json",
    $newRunDir.FullName
)
$auditJson = Invoke-RigPython -CondaEnv $CondaEnv -CondaExe $CondaExe -PythonArgs $auditArgs | Out-String
if ($LASTEXITCODE -ne 0) {
    if ($tempHardwarePath -and (Test-Path $tempHardwarePath)) {
        Remove-Item $tempHardwarePath -Force -ErrorAction SilentlyContinue
    }
    throw ("Parity audit failed for run directory {0}" -f $newRunDir.FullName)
}

$audit = (ConvertFrom-Json $auditJson)
if ($audit -is [System.Array]) {
    $audit = $audit[0]
}

$auditPath = Join-Path $newRunDir.FullName "derived\validation\parity_audit.json"
$counts = $audit.counts
Write-Host ("Run directory: {0}" -f $newRunDir.FullName) -ForegroundColor Cyan
$summaryLine = [string]::Format(
    "Trigger parity summary: trigger={0} fictrac_raw={1} fictrac_udp={2} fictrac_cb={3} second={4}",
    $counts.trigger_rising_edges,
    $counts.fictrac_saved_raw_frames,
    $counts.fictrac_udp_frame_cnt,
    $counts.fictrac_callback_frames,
    $counts.second_camera_saved_frames
)
Write-Host $summaryLine -ForegroundColor Cyan

if (-not $audit.exact_trigger_match) {
    if ($tempHardwarePath -and (Test-Path $tempHardwarePath)) {
        Remove-Item $tempHardwarePath -Force -ErrorAction SilentlyContinue
    }
    throw ("{0} failed. See {1}" -f $TestName, $auditPath)
}

Write-Host ("{0} passed. See {1}" -f $TestName, $auditPath) -ForegroundColor Green

if ($tempHardwarePath -and (Test-Path $tempHardwarePath)) {
    Remove-Item $tempHardwarePath -Force -ErrorAction SilentlyContinue
}