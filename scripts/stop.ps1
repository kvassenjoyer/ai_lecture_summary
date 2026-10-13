$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path $PSScriptRoot -Parent
. (Join-Path $PSScriptRoot 'process_helpers.ps1')
$runningFile = Join-Path $projectRoot 'data/processes.json'
if (-not (Test-Path -LiteralPath $runningFile)) { Write-Output 'No service process record.'; exit }
$record = Get-Content -LiteralPath $runningFile -Raw | ConvertFrom-Json
foreach ($entry in $record.processes) {
    if (Stop-LectureProcessTree -ProcessId $entry.pid -ExpectedStart $entry.started) {
        Write-Output "Stopped service process $($entry.pid)."
    }
}
Remove-Item -LiteralPath $runningFile
