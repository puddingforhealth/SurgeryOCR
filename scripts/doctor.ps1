$ErrorActionPreference = "Continue"
Set-Location (Split-Path $PSScriptRoot -Parent)
$missing = 0
function Check($name, $cmd, $fix) {
    if (Get-Command $cmd -ErrorAction SilentlyContinue) {
        $v = try { (& $cmd --version 2>$null | Select-Object -First 1) } catch { "" }
        Write-Host ("  ok      {0,-12} {1}" -f $name, $v)
    } else {
        Write-Host ("  MISSING {0,-12} fix: {1}" -f $name, $fix)
        $script:missing++
    }
}
Write-Host "doctor: SurgeryOCR"
Check "uv" uv "irm https://astral.sh/uv/install.ps1 | iex"
Check "git" git "winget install Git.Git"
if (-not (Test-Path .venv)) { Write-Host "  MISSING .venv        fix: ./scripts/setup.ps1"; $missing++ } else { Write-Host "  ok      .venv" }
if ($missing -gt 0) { Write-Host "$missing prerequisite(s) missing. Run ./scripts/setup.ps1"; exit 1 }
Write-Host "All prerequisites present."
