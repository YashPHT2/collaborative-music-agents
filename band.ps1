param([int]$Updates = 2, [int]$Episodes = 4, [int]$Bars = 4, [switch]$Resume)
$arguments = @('train_band.py', 'train', '--updates', $Updates, '--episodes', $Episodes, '--bars', $Bars)
if ($Resume) { $arguments += '--resume' }
& "$PSScriptRoot\.venv\Scripts\python.exe" @arguments
exit $LASTEXITCODE
