param([int]$Seed = 2027, [int]$Bars = 8, [int]$Bpm = 100)
$ErrorActionPreference = 'Stop'
& "$PSScriptRoot\.venv\Scripts\python.exe" -u "$PSScriptRoot\collaborative_duet.py" --seed $Seed --bars $Bars --bpm $Bpm
if ($LASTEXITCODE -ne 0) { throw 'Duet generation failed.' }
