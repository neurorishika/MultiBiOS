function Test-UseActiveRigPython {
    param(
        [string]$CondaEnv
    )

    return $env:CONDA_DEFAULT_ENV -eq $CondaEnv
}


function Invoke-RigPython {
    param(
        [string]$CondaEnv,
        [string]$CondaExe,
        [string[]]$PythonArgs
    )

    if (Test-UseActiveRigPython -CondaEnv $CondaEnv) {
        & python @PythonArgs
        return
    }

    if (-not (Test-Path $CondaExe)) {
        throw "Conda executable not found: $CondaExe"
    }

    $condaArgs = @(
        "run",
        "--no-capture-output",
        "-n", $CondaEnv,
        "python"
    ) + $PythonArgs
    & $CondaExe @condaArgs
}


function Start-RigPythonProcess {
    param(
        [string]$RepoRoot,
        [string]$CondaEnv,
        [string]$CondaExe,
        [string[]]$PythonArgs
    )

    if (Test-UseActiveRigPython -CondaEnv $CondaEnv) {
        return Start-Process -FilePath "python" -ArgumentList $PythonArgs -WorkingDirectory $RepoRoot -PassThru -WindowStyle Normal
    }

    if (-not (Test-Path $CondaExe)) {
        throw "Conda executable not found: $CondaExe"
    }

    $condaArgs = @(
        "run",
        "--no-capture-output",
        "-n", $CondaEnv,
        "python"
    ) + $PythonArgs
    return Start-Process -FilePath $CondaExe -ArgumentList $condaArgs -WorkingDirectory $RepoRoot -PassThru -WindowStyle Normal
}