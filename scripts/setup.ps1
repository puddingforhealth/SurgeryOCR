param(
    [string]$Python = "python",
    [switch]$ReuseBasePackages
)
$ErrorActionPreference = "Stop"
$repo = Split-Path $PSScriptRoot -Parent
Push-Location $repo
try {
    $backend = Join-Path $repo ".venv/Scripts/python.exe"
    if (-not (Test-Path $backend)) {
        & $Python -c "import sys; sys.exit(0 if sys.version_info >= (3, 11) else 'Install Python 3.11 or newer and rerun setup.')"
        if ($LASTEXITCODE -ne 0) { throw "A Python 3.11+ interpreter is required." }
        $options = @("-m", "venv")
        if ($ReuseBasePackages) { $options += "--system-site-packages" }
        & $Python @options .venv
        if ($LASTEXITCODE -ne 0) { throw "Could not create the backend virtual environment." }
    }
    & $backend -c "import sys; sys.exit(0 if sys.version_info >= (3, 11) else 'The existing .venv needs Python 3.11 or newer.')"
    if ($LASTEXITCODE -ne 0) { throw "Unsupported backend interpreter." }
    if (-not $ReuseBasePackages) {
        $index = $env:TORCH_INDEX_URL
        if (-not $index -and (Get-Command nvidia-smi -ErrorAction SilentlyContinue)) {
            $driverInfo = nvidia-smi | Out-String
            if ($driverInfo -match 'CUDA Version:\s*(\d+\.\d+)' -and [version]$Matches[1] -ge [version]"12.6") {
                $index = "https://download.pytorch.org/whl/cu126"
            }
        }
        if ($index) {
            & $backend -c "import importlib.util, sys; sys.exit(0 if importlib.util.find_spec('torch') and __import__('torch').cuda.is_available() else 1)"
            if ($LASTEXITCODE -ne 0) {
                & $backend -m pip install --index-url $index torch==2.8.0 torchvision==0.23.0
                if ($LASTEXITCODE -ne 0) { throw "PyTorch installation failed using $index." }
            }
        }
        & $backend -m pip install -r requirements.txt
        if ($LASTEXITCODE -ne 0) { throw "Backend dependency installation failed." }
    }
    & $backend tools/install_ocr_bundle.py --doctor
    if ($LASTEXITCODE -ne 0) { throw "Backend verification failed. Rerun scripts/setup.ps1 to install dependencies." }
    Write-Host "Activate the backend in this terminal: .\.venv\Scripts\Activate.ps1"
    Write-Host "Or run directly: .\.venv\Scripts\python.exe run_pipeline.py --video VIDEO.mp4"
} finally {
    Pop-Location
}
