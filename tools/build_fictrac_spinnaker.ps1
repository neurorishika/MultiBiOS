param(
    [string]$SourceDir = (Join-Path $PSScriptRoot "..\assets\third_party\FicTrac"),
    [string]$BuildDir = (Join-Path $PSScriptRoot "..\assets\third_party\FicTrac-build"),
    [string]$OutputDir = (Join-Path $PSScriptRoot "..\assets\fictrac-spinnaker"),
    [string]$VcpkgRoot,
    [string]$SpinnakerRoot = "C:\Program Files\Teledyne\Spinnaker",
    [string]$CheckoutRef,
    [string]$UpstreamUrl = "https://github.com/rjdmoore/FicTrac.git",
    [switch]$BootstrapClone,
    [switch]$FetchUpstream,
    [switch]$SkipCopy
)

$ErrorActionPreference = "Stop"

function Require-Command {
    param([string]$Name)
    if (-not (Get-Command $Name -ErrorAction SilentlyContinue)) {
        throw "Required command not found on PATH: $Name"
    }
}

function Test-VcpkgRoot {
    param([string]$CandidateRoot)

    if (-not $CandidateRoot) {
        return $false
    }

    $toolchainCandidate = Join-Path $CandidateRoot "scripts\buildsystems\vcpkg.cmake"
    return (Test-Path $toolchainCandidate)
}

function Resolve-VcpkgRoot {
    param([string]$ConfiguredRoot)

    $repoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
    $candidates = @(
        $ConfiguredRoot,
        $env:VCPKG_ROOT,
        (Join-Path $repoRoot "assets\third_party\vcpkg"),
        (Join-Path $env:USERPROFILE "vcpkg"),
        "C:\Users\markd\vcpkg"
    )

    foreach ($candidate in $candidates) {
        if (Test-VcpkgRoot -CandidateRoot $candidate) {
            return $candidate
        }
    }

    throw "Unable to locate vcpkg. Set -VcpkgRoot or VCPKG_ROOT, or install vcpkg in a standard location such as $env:USERPROFILE\vcpkg."
}

function Remove-StalePackagedBinaries {
    param([string]$Directory)

    if (-not (Test-Path $Directory)) {
        return
    }

    Get-ChildItem -Path $Directory -File | Where-Object {
        $_.Extension -in @('.exe', '.dll', '.pdb', '.lib')
    } | Remove-Item -Force
}

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

if (-not (Test-Path $SourceDir)) {
    if (-not $BootstrapClone) {
        throw "Vendored FicTrac source not found at $SourceDir. Commit or unpack the patched source tree there, or rerun with -BootstrapClone to seed it from upstream."
    }

    Require-Command git
    git clone $UpstreamUrl $SourceDir
}

if ($CheckoutRef -or $FetchUpstream) {
    if (-not (Test-Path (Join-Path $SourceDir ".git"))) {
        throw "Cannot update FicTrac source at $SourceDir because it is not a git checkout. Vendored snapshots without .git metadata must be updated manually."
    }

    Require-Command git

    if ($FetchUpstream) {
        git -C $SourceDir fetch --all --tags
    }

    if ($CheckoutRef) {
        git -C $SourceDir checkout $CheckoutRef
    }
}

New-Item -ItemType Directory -Force -Path $BuildDir | Out-Null

$configureArgs = @(
    "-S", $SourceDir,
    "-B", $BuildDir,
    "-G", "Visual Studio 17 2022",
    "-A", "x64",
    "-D", "CMAKE_TOOLCHAIN_FILE=$ToolchainFile",
    "-D", "PGR_USB3=ON",
    "-D", "PGR_DIR=$SpinnakerRoot",
    "--fresh"
)

Write-Host "Configuring FicTrac with Spinnaker support..."
& cmake @configureArgs

Write-Host "Building FicTrac (Release)..."
& cmake --build $BuildDir --config Release --clean-first --parallel 4

$builtBinCandidates = @(
    (Join-Path $BuildDir "Release"),
    (Join-Path $BuildDir "x64\Release"),
    (Join-Path $SourceDir "bin\Release"),
    (Join-Path $SourceDir "bin")
)
$builtArtifact = $builtBinCandidates |
    Where-Object { Test-Path (Join-Path $_ "fictrac.exe") } |
    ForEach-Object { Get-Item (Join-Path $_ "fictrac.exe") } |
    Sort-Object LastWriteTimeUtc -Descending |
    Select-Object -First 1
if (-not $builtArtifact) {
    throw "Build finished but no fictrac.exe was produced under the expected output paths: $($builtBinCandidates -join ', ')"
}
$builtBinDir = Split-Path -Parent $builtArtifact.FullName

if (-not $SkipCopy) {
    New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null
    Remove-StalePackagedBinaries -Directory $OutputDir

    $builtArtifacts = Get-ChildItem -Path $builtBinDir -File
    if (-not $builtArtifacts) {
        throw "No build artifacts were found in $builtBinDir after building FicTrac."
    }

    foreach ($artifact in $builtArtifacts) {
        Copy-Item $artifact.FullName -Destination (Join-Path $OutputDir $artifact.Name) -Force
    }

    $packagedFictrac = Join-Path $OutputDir "fictrac.exe"
    if (Test-Path $packagedFictrac) {
        Copy-Item $packagedFictrac (Join-Path $OutputDir "fictrac-spinnaker.exe") -Force
    }
    else {
        throw "Expected freshly built fictrac.exe in $builtBinDir, but it was not found."
    }
}

Write-Host ""
Write-Host "Build complete."
Write-Host "Source bin:   $builtBinDir"
Write-Host "Output dir:   $OutputDir"
Write-Host "Source mode:  vendored FicTrac tree at $SourceDir"
Write-Host "Next step: confirm config/hardware.yaml points fictrac.bin at the rebuilt fictrac-spinnaker.exe"