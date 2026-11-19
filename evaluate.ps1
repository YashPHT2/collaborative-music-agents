param([switch]$NoAudio)
$ErrorActionPreference = 'Stop'
$evaluationArgs = @('-u', "$PSScriptRoot\evaluate_baseline.py")
if ($NoAudio) { $evaluationArgs += '--no-audio' }
& "$PSScriptRoot\.venv\Scripts\python.exe" @evaluationArgs
if ($LASTEXITCODE -ne 0) { throw 'Baseline evaluation failed.' }
