param([ValidateRange(1,65535)][int]$Port = 8080, [switch]$OpenBrowser)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path $PSScriptRoot -Parent
. (Join-Path $PSScriptRoot 'process_helpers.ps1')
Set-Location $projectRoot
$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONUNBUFFERED = '1'
$runtimePython = Join-Path $projectRoot '.venv/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $runtimePython)) { throw 'Run scripts/bootstrap.ps1 first.' }
if (Get-NetTCPConnection -State Listen -ErrorAction Stop | Where-Object LocalPort -EQ $Port) { throw "Port $Port is already in use." }
New-Item -ItemType Directory -Force logs,data | Out-Null
$runningFile = Join-Path $projectRoot 'data/processes.json'
if (Test-Path $runningFile) {
    $oldProcesses = Get-Content $runningFile -Raw | ConvertFrom-Json
    foreach ($oldProcess in $oldProcesses.processes) {
        $currentProcess = Get-Process -Id $oldProcess.pid -ErrorAction SilentlyContinue
        if ($currentProcess -and $currentProcess.StartTime.ToUniversalTime().ToString('o') -eq $oldProcess.started) {
            throw 'A service process is still running. Use scripts/stop.ps1 first.'
        }
    }
}
$workerProcess = Start-Process $runtimePython -ArgumentList '-m','lecture_service.worker' -WorkingDirectory $projectRoot -WindowStyle Hidden -RedirectStandardOutput 'logs/worker.log' -RedirectStandardError 'logs/worker-error.log' -PassThru
try {
    $apiProcess = Start-Process $runtimePython -ArgumentList '-m','uvicorn','lecture_service.main:app','--host','127.0.0.1','--port',"$Port" -WorkingDirectory $projectRoot -WindowStyle Hidden -RedirectStandardOutput 'logs/api.log' -RedirectStandardError 'logs/api-error.log' -PassThru
    @{ processes = @(@{pid=$workerProcess.Id; started=$workerProcess.StartTime.ToUniversalTime().ToString('o')}, @{pid=$apiProcess.Id; started=$apiProcess.StartTime.ToUniversalTime().ToString('o')}); port=$Port } | ConvertTo-Json -Depth 4 | Set-Content $runningFile
    $healthy = $false
    for ($attempt=0; $attempt -lt 30; $attempt++) {
        $apiProcess.Refresh()
        $workerProcess.Refresh()
        if ($apiProcess.HasExited -or $workerProcess.HasExited) { throw 'A service process exited; inspect logs/api-error.log and logs/worker-error.log.' }
        try { $response = Invoke-RestMethod "http://127.0.0.1:$Port/health"; if ($response.status -eq 'ok') { $healthy=$true; break } } catch { }
        Start-Sleep -Seconds 1
    }
    if (-not $healthy) { throw 'API failed to start; inspect logs/api-error.log.' }
    Write-Output "Service ready: http://127.0.0.1:$Port"
    if ($OpenBrowser) { Start-Process "http://127.0.0.1:$Port" }
} catch {
    Stop-LectureProcessTree -ProcessId $workerProcess.Id -ExpectedStart $workerProcess.StartTime.ToUniversalTime().ToString('o')
    if ($apiProcess) { Stop-LectureProcessTree -ProcessId $apiProcess.Id -ExpectedStart $apiProcess.StartTime.ToUniversalTime().ToString('o') }
    if (Test-Path $runningFile) { Remove-Item -LiteralPath $runningFile }
    throw
}
