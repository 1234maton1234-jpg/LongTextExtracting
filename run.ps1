param(
    [Parameter(Mandatory = $true)]
    [string]$PdfPath,
    [string]$Output = "outputs\p2g_v2_output.json",
    [int]$Concurrency = 4,
    [string]$FlashModel = "deepseek-v4-flash",
    [string]$ProModel = "deepseek-v4-pro",
    [switch]$DryRun,
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$ExtraArgs
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
# Keep in sync with setup.ps1 / test.ps1 / .gitignore
$VenvName = "longtext"
$VenvPython = Join-Path $ProjectRoot "$VenvName\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $VenvPython)) {
    throw "The virtual environment is missing. Run .\setup.ps1 first."
}
$ResolvedPdf = (Resolve-Path -LiteralPath $PdfPath).Path
if ([System.IO.Path]::IsPathRooted($Output)) {
    $ResolvedOutput = $Output
} else {
    $ResolvedOutput = Join-Path $ProjectRoot $Output
}
$OutputDirectory = Split-Path -Parent $ResolvedOutput
New-Item -ItemType Directory -Force -Path $OutputDirectory | Out-Null

if (-not $DryRun -and -not $env:DEEPSEEK_API_KEY) {
    $EnvFile = Join-Path $ProjectRoot ".env"
    if (Test-Path -LiteralPath $EnvFile) {
        $Entry = Get-Content -LiteralPath $EnvFile | Where-Object {
            $_ -match '^\s*DEEPSEEK_API_KEY\s*='
        } | Select-Object -First 1
        if ($Entry) {
            $Value = ($Entry -split '=', 2)[1].Trim().Trim('"').Trim("'")
            if ($Value -and $Value -ne "replace_with_your_own_key") {
                $env:DEEPSEEK_API_KEY = $Value
            }
        }
    }
}
if (-not $DryRun -and -not $env:DEEPSEEK_API_KEY) {
    throw "DEEPSEEK_API_KEY is missing. Copy .env.example to .env and add your own key."
}

$Arguments = @(
    (Join-Path $ProjectRoot "python_scripts\p2g_v2.py"),
    $ResolvedPdf,
    "--output", $ResolvedOutput,
    "--memory-db", (Join-Path $ProjectRoot "outputs\p2g_v2_memory.sqlite3"),
    "--cache-db", (Join-Path $ProjectRoot "outputs\p2g_v2_cache.sqlite3"),
    "--usage-db", (Join-Path $ProjectRoot "outputs\p2g_v2_usage.sqlite3"),
    "--concurrency", $Concurrency,
    "--flash-model", $FlashModel,
    "--pro-model", $ProModel
)
if ($DryRun) { $Arguments += "--dry-run" }
if ($ExtraArgs) { $Arguments += $ExtraArgs }

$env:PYTHONUTF8 = $env:PYTHONUTF8 # 保持现有环境设置，不强制覆盖为 UTF-8
& $VenvPython @Arguments
if ($LASTEXITCODE -ne 0) { throw "P2G V2 failed with exit code $LASTEXITCODE." }
Write-Host "Completed: $ResolvedOutput"
