param(
    [string]$SourceDir = (Join-Path $PSScriptRoot "..\assets\third_party\FicTrac"),
    [string]$BuildDir = (Join-Path $PSScriptRoot "..\assets\third_party\FicTrac-build"),
    [string]$OutputDir = (Join-Path $PSScriptRoot "..\assets\fictrac-spinnaker"),
    [string]$VcpkgRoot,
    [string]$SpinnakerRoot = "C:\Program Files\Teledyne\Spinnaker",
    [string]$GitRef = "master",
    [switch]$SkipClone,
    [switch]$SkipCopy
)

$ErrorActionPreference = "Stop"

function Require-Command {
    param([string]$Name)
    if (-not (Get-Command $Name -ErrorAction SilentlyContinue)) {
        throw "Required command not found on PATH: $Name"
    }
}

function Resolve-VcpkgRoot {
    param([string]$ConfiguredRoot)
    if ($ConfiguredRoot) {
        return $ConfiguredRoot
    }
    if ($env:VCPKG_ROOT) {
        return $env:VCPKG_ROOT
    }
    throw "Set -VcpkgRoot or VCPKG_ROOT before building FicTrac."
}

Require-Command git
Require-Command cmake

$SourceDir = [System.IO.Path]::GetFullPath($SourceDir)
$BuildDir = [System.IO.Path]::GetFullPath($BuildDir)
$OutputDir = [System.IO.Path]::GetFullPath($OutputDir)
$VcpkgRoot = [System.IO.Path]::GetFullPath((Resolve-VcpkgRoot -ConfiguredRoot $VcpkgRoot))
$SpinnakerRoot = [System.IO.Path]::GetFullPath($SpinnakerRoot)

$ToolchainFile = Join-Path $VcpkgRoot "scripts\buildsystems\vcpkg.cmake"
if (-not (Test-Path $ToolchainFile)) {
    throw "vcpkg toolchain not found at $ToolchainFile"
}
if (-not (Test-Path (Join-Path $SpinnakerRoot "lib64\vs2015\Spinnaker_v140.lib"))) {
    throw "Spinnaker SDK not found under $SpinnakerRoot"
}

if (-not $SkipClone) {
    if (-not (Test-Path $SourceDir)) {
        git clone https://github.com/rjdmoore/FicTrac.git $SourceDir
    }
    git -C $SourceDir fetch --all --tags
    git -C $SourceDir checkout $GitRef
}

New-Item -ItemType Directory -Force -Path $BuildDir | Out-Null

$configureArgs = @(
    "-S", $SourceDir,
    "-B", $BuildDir,
    "-A", "x64",
    "-D", "CMAKE_TOOLCHAIN_FILE=$ToolchainFile",
    "-D", "PGR_USB3=ON",
    "-D", "PGR_DIR=$SpinnakerRoot"
)

Write-Host "Configuring FicTrac with Spinnaker support..."
& cmake @configureArgs

Write-Host "Building FicTrac (Release)..."
& cmake --build $BuildDir --config Release -j 4

$sourceBinCandidates = @(
    (Join-Path $SourceDir "bin\Release"),
    (Join-Path $SourceDir "bin")
)
$sourceBinDir = $sourceBinCandidates | Where-Object { Test-Path (Join-Path $_ "fictrac.exe") } | Select-Object -First 1
if (-not $sourceBinDir) {
    throw "Build finished but no fictrac.exe was produced under $($sourceBinCandidates -join ', ')"
}

if (-not $SkipCopy) {
    New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null
    Copy-Item (Join-Path $sourceBinDir "*") -Destination $OutputDir -Recurse -Force
    if (Test-Path (Join-Path $OutputDir "fictrac.exe")) {
        Copy-Item (Join-Path $OutputDir "fictrac.exe") (Join-Path $OutputDir "fictrac-spinnaker.exe") -Force
    }
}

Write-Host ""
Write-Host "Build complete."
Write-Host "Source bin:   $sourceBinDir"
Write-Host "Output dir:   $OutputDir"
Write-Host "Next step: set config/experiment_config.yaml fictrac_bin to the rebuilt fictrac-spinnaker.exe"