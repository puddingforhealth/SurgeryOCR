# One-shot setup for a Windows workstation. Idempotent.
# Afterwards: gcloud auth login ; python tools\cloud_videos.py doctor
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")

Write-Host "== python packages"
python -m pip install --upgrade pip -q
python -m pip install -r requirements.txt -q

Write-Host "== google cloud sdk"
if (-not (Get-Command gcloud -ErrorAction SilentlyContinue)) {
    Write-Host "   gcloud not found. Install it with:  winget install Google.CloudSDK"
    Write-Host "   then open a new terminal and run:   gcloud auth login"
} else {
    gcloud --version | Select-Object -First 1
}

Write-Host "== check"
python tools\cloud_videos.py doctor
