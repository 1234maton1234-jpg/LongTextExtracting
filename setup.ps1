param()

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectRoot

$Launcher = $null
$LauncherArgs = @()
$ProbeErrorAction = $ErrorActionPreference
$ErrorActionPreference = "SilentlyContinue"
# Supported interpreters. 3.13 both passes CI and is what the development venv
# runs, so the upper bound only excludes versions the project has never been run
# on (3.14 is unreleased at the time of writing).
$PythonRangeProbe = "import struct,sys; raise SystemExit(0 if (3,10) <= sys.version_info[:2] <= (3,13) and struct.calcsize('P') == 8 else 1)"
$Python = Get-Command python -ErrorAction SilentlyContinue
if ($Python) {
    & $Python.Source -c $PythonRangeProbe 2>$null
    if ($LASTEXITCODE -eq 0) {
        $Launcher = $Python.Source
    }
}
if (-not $Launcher) {
    $Py = Get-Command py -ErrorAction SilentlyContinue
    if ($Py) {
        # Newest first, and the version guard is applied here too so that a future
        # 3.14 cannot be picked silently.
        foreach ($Version in @("-3.13", "-3.12", "-3.11", "-3.10")) {
            & $Py.Source $Version -c $PythonRangeProbe 2>$null
            if ($LASTEXITCODE -eq 0) {
                $Launcher = $Py.Source
                $LauncherArgs = @($Version)
                break
            }
        }
    }
}
$ErrorActionPreference = $ProbeErrorAction
if (-not $Launcher) {
    throw "64-bit Python 3.10-3.13 was not found."
}

$VenvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$NeedsVenv = -not (Test-Path -LiteralPath $VenvPython)
if (-not $NeedsVenv) {
    $ProbeErrorAction = $ErrorActionPreference
    $ErrorActionPreference = "SilentlyContinue"
    & $VenvPython -c "import encodings, sys; raise SystemExit(0 if (3,10) <= sys.version_info[:2] <= (3,13) else 1)" 2>$null
    $NeedsVenv = $LASTEXITCODE -ne 0
    $ErrorActionPreference = $ProbeErrorAction
}
if ($NeedsVenv) {
    $VenvPath = Join-Path $ProjectRoot ".venv"
    & $Launcher @LauncherArgs -m venv --clear $VenvPath
    if ($LASTEXITCODE -ne 0) { throw "Failed to create the virtual environment." }
}

& $VenvPython -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { throw "Failed to upgrade pip." }
& $VenvPython -m pip install -r (Join-Path $ProjectRoot "requirements.txt")
if ($LASTEXITCODE -ne 0) { throw "Failed to install dependencies." }

Write-Host "Setup complete. Next run: .\test.ps1"
