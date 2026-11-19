param(
    [int]$Seed = 42,
    [int]$Bars = 8,
    [int]$Bpm = 100
)
$ErrorActionPreference = 'Stop'
& "$PSScriptRoot\.venv\Scripts\python.exe" -u "$PSScriptRoot\generate_piano.py" --seed $Seed --bars $Bars --bpm $Bpm
if ($LASTEXITCODE -ne 0) { throw 'Piano generation failed.' }
