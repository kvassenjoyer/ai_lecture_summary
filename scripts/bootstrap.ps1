$ErrorActionPreference = 'Stop'
Set-Location (Split-Path $PSScriptRoot -Parent)
$env:PYTHONIOENCODING = 'utf-8'
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) { throw 'Install uv from https://docs.astral.sh/uv/getting-started/installation/ and rerun this script.' }
if (-not (Get-Command ffmpeg -ErrorAction SilentlyContinue)) { throw 'Install FFmpeg: winget install Gyan.FFmpeg ; then reopen PowerShell.' }
uv sync --extra dev
if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed.' }
& .venv/Scripts/python.exe scripts/download_models.py --verify-hashes
if ($LASTEXITCODE -ne 0) { throw 'Model download failed.' }
Write-Output 'Ready. Start the service with: powershell -ExecutionPolicy Bypass -File scripts/start.ps1'
