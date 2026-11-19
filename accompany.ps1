param([int]$Seed = 42, [int]$Bpm = 100)
$ErrorActionPreference = 'Stop'
& "$PSScriptRoot\.venv\Scripts\python.exe" -u "$PSScriptRoot\accompany.py" --seed $Seed --bpm $Bpm
if ($LASTEXITCODE -ne 0) { throw 'Accompaniment generation failed.' }
