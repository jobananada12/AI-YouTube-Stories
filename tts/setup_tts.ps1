$ErrorActionPreference = 'Stop'
Set-Location (Split-Path $PSScriptRoot -Parent)

Write-Host '=== AI-YouTube-Stories: local Ukrainian TTS setup ===' -ForegroundColor Cyan

if (-not (Test-Path '.venv\Scripts\python.exe')) {
    Write-Host 'Creating .venv...'
    py -3 -m venv .venv
}

$python = (Resolve-Path '.venv\Scripts\python.exe').Path

Write-Host 'Installing Piper TTS...'
& $python -m pip install --upgrade pip
& $python -m pip install 'piper-tts>=1.8,<2'

$modelDir = Join-Path (Get-Location) 'tts\models'
New-Item -ItemType Directory -Force -Path $modelDir | Out-Null

Write-Host 'Downloading Ukrainian male narrator: uk_UA-oleksa-high...'
& $python -m piper.download_voices --data-dir $modelDir uk_UA-oleksa-high

Write-Host ''
Write-Host 'Setup complete.' -ForegroundColor Green
Write-Host "Model directory: $modelDir"
Write-Host 'Next: .\.venv\Scripts\python.exe tts\generate_narration.py --input narration_ua_30min.txt'
