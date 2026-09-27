param()

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
# Keep in sync with setup.ps1 / run.ps1 / .gitignore
$VenvName = "longtext"
$VenvPython = Join-Path $ProjectRoot "$VenvName\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $VenvPython)) {
    throw "The virtual environment is missing. Run .\setup.ps1 first."
}

& $VenvPython -m unittest discover -s (Join-Path $ProjectRoot "tests") -v
if ($LASTEXITCODE -ne 0) { throw "Tests failed." }
