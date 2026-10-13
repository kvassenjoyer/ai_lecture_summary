function Get-LectureProcessSnapshot {
    param([ValidateRange(1, 2147483647)][int]$ProcessId)
    try {
        $serviceProcess = Get-Process -Id $ProcessId -ErrorAction Stop
    } catch {
        if ($_.FullyQualifiedErrorId -like 'NoProcessFoundForGivenId,*') { return $null }
        throw
    }
    try {
        # Pin the native handle before verifying ownership to prevent PID reuse.
        $null = $serviceProcess.get_Handle()
        if ($serviceProcess.get_HasExited()) { $serviceProcess.Dispose(); return $null }
        # Explicit getters surface access errors instead of returning a null property.
        $created = $serviceProcess.get_StartTime()
        if ($null -eq $created) { throw "Cannot verify creation time for process $ProcessId." }
        return [pscustomobject]@{
            Process = $serviceProcess
            Created = $created.ToUniversalTime()
            Started = $created.ToUniversalTime().ToString('o')
        }
    } catch {
        $originalFailure = $_
        $gone = $false
        try { $gone = $serviceProcess.get_HasExited() } catch { }
        if (-not $gone) {
            try {
                $remaining = Get-Process -Id $ProcessId -ErrorAction Stop
                $remaining.Dispose()
            } catch {
                if ($_.FullyQualifiedErrorId -like 'NoProcessFoundForGivenId,*') { $gone = $true }
                else { $serviceProcess.Dispose(); throw }
            }
        }
        $serviceProcess.Dispose()
        if ($gone) { return $null }
        throw $originalFailure
    }
}

function Stop-LectureProcessTree {
    param([ValidateRange(1, 2147483647)][int]$ProcessId, [string]$ExpectedStart)
    if ([string]::IsNullOrWhiteSpace($ExpectedStart)) {
        throw "Expected creation time is required for process $ProcessId."
    }
    $snapshot = Get-LectureProcessSnapshot -ProcessId $ProcessId
    if ($null -eq $snapshot) { return $false }
    if ($snapshot.Started -ne $ExpectedStart) { $snapshot.Process.Dispose(); return $false }
    try {
        try {
            $descendants = @(Get-CimInstance Win32_Process -Filter "ParentProcessId = $ProcessId" -ErrorAction Stop)
        } catch {
            if ($snapshot.Process.get_HasExited()) { return $true }
            throw
        }
        foreach ($child in $descendants) {
            if ($snapshot.Process.get_HasExited()) { break }
            $childSnapshot = Get-LectureProcessSnapshot -ProcessId $child.ProcessId
            if ($null -eq $childSnapshot) { continue }
            try {
                if ($null -eq $child.CreationDate) {
                    throw "Cannot verify creation time for child process $($child.ProcessId)."
                }
                # CIM timestamps have microsecond precision; .NET uses 100 ns ticks.
                $difference = ($childSnapshot.Created - $child.CreationDate.ToUniversalTime()).Ticks
                if ([Math]::Abs($difference) -ge 10) { continue }
                $null = Stop-LectureProcessTree -ProcessId $child.ProcessId -ExpectedStart $childSnapshot.Started
            } finally {
                $childSnapshot.Process.Dispose()
            }
        }
        # The venv launcher may already have exited when its child stopped.
        if (-not $snapshot.Process.get_HasExited()) {
            try {
                $snapshot.Process.Kill()
                if (-not $snapshot.Process.WaitForExit(5000)) { throw "Service process $ProcessId did not stop." }
            } catch {
                if (-not $snapshot.Process.get_HasExited()) { throw }
            }
        }
        return $true
    } finally {
        $snapshot.Process.Dispose()
    }
}
