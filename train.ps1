param([int]$Updates = 8, [int]$EpisodesPerUpdate = 4)
$ErrorActionPreference = 'Stop'
& "$PSScriptRoot\.venv\Scripts\python.exe" -u "$PSScriptRoot\train_ppo_pilot.py" --updates $Updates --episodes-per-update $EpisodesPerUpdate
if ($LASTEXITCODE -ne 0) { throw 'PPO pilot failed.' }
