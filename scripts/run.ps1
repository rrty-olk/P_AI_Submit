param(
    [ValidateNotNullOrEmpty()]
    [string]$Distribution = 'Ubuntu-22.04'
)

$ErrorActionPreference = 'Stop'
$projectPath = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$pathOutput = & wsl.exe -d $Distribution -- wslpath -a ($projectPath.Replace('\', '/'))
if ($LASTEXITCODE -ne 0) { throw 'Failed to resolve the WSL project path.' }
$linuxPath = ($pathOutput -join "`n").Trim()
if (-not $linuxPath) { throw 'WSL returned an empty project path.' }
& wsl.exe -d $Distribution -- bash "$linuxPath/scripts/run.sh"
if ($LASTEXITCODE -ne 0) { throw "Pipeline failed ($LASTEXITCODE). Review output/logs." }
