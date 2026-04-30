function Test-UseActiveRigPython {
    param(
        [string]$CondaEnv
    )

    return $env:CONDA_DEFAULT_ENV -eq $CondaEnv
}


function Resolve-RigPythonExe {
    param(
        [string]$CondaEnv,
        [string]$CondaExe
    )

    $candidates = @()

    if ($env:CONDA_PREFIX -and $env:CONDA_DEFAULT_ENV -eq $CondaEnv) {
        $candidates += (Join-Path $env:CONDA_PREFIX "python.exe")
    }

    if ($CondaExe) {
        $condaScriptsDir = Split-Path -Parent $CondaExe
        if ($condaScriptsDir) {
            $condaRoot = Split-Path -Parent $condaScriptsDir
            if ($condaRoot) {
                $candidates += (Join-Path $condaRoot "envs/$CondaEnv/python.exe")
            }
        }
    }

    if ($env:USERPROFILE) {
        $candidates += (Join-Path $env:USERPROFILE ".conda/envs/$CondaEnv/python.exe")
    }

    foreach ($candidate in $candidates | Select-Object -Unique) {
        if ($candidate -and (Test-Path $candidate)) {
            return $candidate
        }
    }

    return $null
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

    $pythonExe = Resolve-RigPythonExe -CondaEnv $CondaEnv -CondaExe $CondaExe
    if ($pythonExe) {
        & $pythonExe @PythonArgs
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

    $pythonExe = Resolve-RigPythonExe -CondaEnv $CondaEnv -CondaExe $CondaExe
    if ((Test-UseActiveRigPython -CondaEnv $CondaEnv) -and $pythonExe) {
        return Start-Process -FilePath $pythonExe -ArgumentList $PythonArgs -WorkingDirectory $RepoRoot -PassThru -WindowStyle Normal
    }

    if (Test-UseActiveRigPython -CondaEnv $CondaEnv) {
        $pythonCommand = Get-Command python -ErrorAction SilentlyContinue
        if ($pythonCommand -and $pythonCommand.Source) {
            return Start-Process -FilePath $pythonCommand.Source -ArgumentList $PythonArgs -WorkingDirectory $RepoRoot -PassThru -WindowStyle Normal
        }
        return Start-Process -FilePath "python" -ArgumentList $PythonArgs -WorkingDirectory $RepoRoot -PassThru -WindowStyle Normal
    }

    if ($pythonExe) {
        return Start-Process -FilePath $pythonExe -ArgumentList $PythonArgs -WorkingDirectory $RepoRoot -PassThru -WindowStyle Normal
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