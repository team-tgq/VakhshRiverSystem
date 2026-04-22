param(
    [string]$PythonExe = "E:\anaconda\envs\VakhshRiverSystem\python.exe",
    [string]$QtPlatform = "",
    [switch]$PauseOnError
)

$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$mainScript = Join-Path $projectRoot "main.py"

if (-not (Test-Path $PythonExe)) {
    Write-Error "Python interpreter not found: $PythonExe"
    exit 1
}

if (-not (Test-Path $mainScript)) {
    Write-Error "Main entry not found: $mainScript"
    exit 1
}

$env:QTWEBENGINE_CHROMIUM_FLAGS = "--disable-gpu --disable-gpu-compositing"
$env:QT_OPENGL = "software"
if ($QtPlatform) {
    $env:QT_QPA_PLATFORM = $QtPlatform
}

Push-Location $projectRoot
try {
    & $PythonExe $mainScript
    $exitCode = $LASTEXITCODE
} catch {
    Write-Host ""
    Write-Host "[ERROR] $($_.Exception.Message)" -ForegroundColor Red
    $exitCode = 1
} finally {
    Pop-Location
}

if ($PauseOnError -and $exitCode -ne 0) {
    Write-Host ""
    Read-Host "Application exited abnormally. Press Enter to close"
}

exit $exitCode
