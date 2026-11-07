<#
Windows setup, run from the project folder:

    powershell -ExecutionPolicy Bypass -File setup.ps1          # install and test
    powershell -ExecutionPolicy Bypass -File setup.ps1 -Data    # ...and download the datasets (~1 GB)

Creates .venv with Python 3.12 or 3.13 (TensorFlow has no wheels for newer versions),
installs requirements.txt and runs the tests. Safe to run again.
TensorFlow on Windows uses the CPU only; train on WSL2 (setup.sh --gpu) or Colab for a GPU.
#>
param([switch]$Data)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

# 1. Find Python 3.12 or 3.13 through the py launcher.
$ver = $null
foreach ($v in "3.12", "3.13") {
    $ErrorActionPreference = "Continue"   # py prints an error when a version is missing
    & py "-$v" -c "pass" 2>&1 | Out-Null
    $found = ($LASTEXITCODE -eq 0)
    $ErrorActionPreference = "Stop"
    if ($found) { $ver = $v; break }
}
if (-not $ver) {
    Write-Host "Python 3.12 or 3.13 not found. Install it, then run this again:" -ForegroundColor Red
    Write-Host "    winget install Python.Python.3.12"
    exit 1
}

# 2. Virtual environment and packages.
if (-not (Test-Path .venv\Scripts\python.exe)) { & py "-$ver" -m venv .venv }
$py = ".venv\Scripts\python.exe"
& $py -m pip install --upgrade pip --quiet
& $py -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) { exit 1 }

# 3. Tests, then (optionally) the datasets.
& $py -m pytest -q
if ($LASTEXITCODE -ne 0) { Write-Host "Tests failed." -ForegroundColor Red; exit 1 }
if ($Data) { & $py prepare_data.py }
& $py -c "import tensorflow as tf; print('TensorFlow', tf.__version__, '| GPUs:', len(tf.config.list_physical_devices('GPU')))"

Write-Host "`nDone. In a new terminal, activate the environment with:  .venv\Scripts\Activate.ps1"
