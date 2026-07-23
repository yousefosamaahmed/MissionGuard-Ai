$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

$LogPath = Join-Path $ProjectRoot 'DOCKER_STARTUP_LOG.txt'
$LinksPath = Join-Path $ProjectRoot 'LOCAL_LINKS.txt'
"MissionGuard Docker startup - $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" | Set-Content -Path $LogPath -Encoding UTF8

function Write-Step([string]$Message) {
    Write-Host $Message
    $Message | Add-Content -Path $LogPath -Encoding UTF8
}

function Get-DotEnvValue([string]$Name, [string]$DefaultValue) {
    if (-not (Test-Path '.env')) { return $DefaultValue }
    $line = Get-Content '.env' | Where-Object {
        $_ -match ('^\s*' + [regex]::Escape($Name) + '\s*=')
    } | Select-Object -Last 1
    if (-not $line) { return $DefaultValue }
    $value = ($line -split '=', 2)[1].Trim().Trim('"').Trim("'")
    if ([string]::IsNullOrWhiteSpace($value)) { return $DefaultValue }
    return $value
}

function Find-FreeTcpPort([int]$Preferred, [int]$Maximum) {
    for ($port = $Preferred; $port -le $Maximum; $port++) {
        $listener = $null
        try {
            $listener = [System.Net.Sockets.TcpListener]::new([System.Net.IPAddress]::Any, $port)
            $listener.Start()
            return $port
        }
        catch {
            # Occupied or reserved by Windows. Try the next port.
        }
        finally {
            if ($null -ne $listener) {
                try { $listener.Stop() } catch { }
            }
        }
    }
    throw "No free TCP port was found between $Preferred and $Maximum."
}

function Invoke-NativeCaptured {
    param(
        [Parameter(Mandatory = $true)][string]$FilePath,
        [string[]]$ArgumentList = @(),
        [switch]$Silent
    )

    # Windows PowerShell 5.1 can convert a native program's stderr output into
    # a terminating NativeCommandError when ErrorActionPreference is Stop.
    # Docker writes normal status lines such as "Container ... Stopping" and
    # "Image ... Building" to stderr, so capture stdout/stderr through
    # Start-Process instead of letting PowerShell classify them as errors.
    $stdoutPath = [System.IO.Path]::GetTempFileName()
    $stderrPath = [System.IO.Path]::GetTempFileName()

    try {
        $process = Start-Process `
            -FilePath $FilePath `
            -ArgumentList $ArgumentList `
            -WorkingDirectory $ProjectRoot `
            -NoNewWindow `
            -Wait `
            -PassThru `
            -RedirectStandardOutput $stdoutPath `
            -RedirectStandardError $stderrPath

        $lines = @()
        if (Test-Path -LiteralPath $stdoutPath) {
            $lines += @(Get-Content -LiteralPath $stdoutPath -ErrorAction SilentlyContinue)
        }
        if (Test-Path -LiteralPath $stderrPath) {
            $lines += @(Get-Content -LiteralPath $stderrPath -ErrorAction SilentlyContinue)
        }

        if (-not $Silent) {
            foreach ($item in $lines) {
                $line = [string]$item
                Write-Host $line
                $line | Add-Content -Path $LogPath -Encoding UTF8
            }
        }

        return [pscustomobject]@{
            ExitCode = [int]$process.ExitCode
            Lines = $lines
        }
    }
    finally {
        Remove-Item -LiteralPath $stdoutPath, $stderrPath -Force -ErrorAction SilentlyContinue
    }
}

function Invoke-DockerLogged([string[]]$Arguments) {
    $result = Invoke-NativeCaptured -FilePath 'docker.exe' -ArgumentList $Arguments
    return [int]$result.ExitCode
}

function Wait-WebService([string]$Name, [string]$Url, [int]$TimeoutSeconds = 420) {
    Write-Step "Waiting for $Name at $Url ..."
    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    while ((Get-Date) -lt $deadline) {
        try {
            $response = Invoke-WebRequest -UseBasicParsing -Uri $Url -TimeoutSec 5
            if ($response.StatusCode -ge 200 -and $response.StatusCode -lt 500) {
                Write-Step "$Name is ready."
                return $true
            }
        }
        catch { }
        Start-Sleep -Seconds 3
    }
    Write-Step "$Name did not become reachable before the timeout."
    return $false
}

try {
    Write-Step '============================================================'
    Write-Step ' MissionGuard AI - Safe Docker startup'
    Write-Step '============================================================'

    if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
        throw 'Docker Desktop is not installed, or docker.exe is not available in PATH.'
    }

    $dockerInfo = Invoke-NativeCaptured -FilePath 'docker.exe' -ArgumentList @('info') -Silent
    if ($dockerInfo.ExitCode -ne 0) {
        throw 'Docker Desktop is installed, but the Docker engine is not running. Open Docker Desktop and wait until it says Engine running.'
    }

    if (-not (Test-Path '.env')) {
        Copy-Item '.env.local.example' '.env'
        Write-Step 'A local Docker .env file was created automatically.'
    }

    if (Select-String -Path '.env' -SimpleMatch 'CHANGE_ME' -Quiet) {
        throw 'Replace every CHANGE_ME value in .env before starting.'
    }

    Write-Step '[1/5] Recreating MissionGuard containers while preserving PostgreSQL data...'
    $downExitCode = Invoke-DockerLogged @('compose', 'down', '--remove-orphans')
    if ($downExitCode -ne 0) {
        throw "docker compose down failed with exit code $downExitCode."
    }

    $preferredApp = [int](Get-DotEnvValue 'APP_PORT' '8501')
    $preferredPgAdmin = [int](Get-DotEnvValue 'PGADMIN_PORT' '5050')
    $preferredPostgres = [int](Get-DotEnvValue 'POSTGRES_EXPOSE_PORT' '55432')

    $appPort = Find-FreeTcpPort $preferredApp ([Math]::Min($preferredApp + 99, 65535))
    $pgAdminPort = Find-FreeTcpPort $preferredPgAdmin ([Math]::Min($preferredPgAdmin + 99, 65535))
    $postgresPort = Find-FreeTcpPort $preferredPostgres ([Math]::Min($preferredPostgres + 99, 65535))

    # Shell environment variables take precedence over values in .env for Compose interpolation.
    $env:APP_BIND_ADDRESS = '127.0.0.1'
    $env:PGADMIN_BIND_ADDRESS = '127.0.0.1'
    $env:POSTGRES_BIND_ADDRESS = '127.0.0.1'
    $env:APP_PORT = "$appPort"
    $env:PGADMIN_PORT = "$pgAdminPort"
    $env:POSTGRES_EXPOSE_PORT = "$postgresPort"

    Write-Step '[2/5] Selected available Windows ports:'
    Write-Step "      MissionGuard: $appPort"
    Write-Step "      pgAdmin:      $pgAdminPort"
    Write-Step "      PostgreSQL:   $postgresPort"

    Write-Step '[3/5] Building and starting PostgreSQL, pgAdmin, and MissionGuard...'
    $exitCode = Invoke-DockerLogged @('compose', 'up', '-d', '--build', '--force-recreate', '--remove-orphans')
    if ($exitCode -ne 0) {
        throw "docker compose up failed with exit code $exitCode."
    }

    Write-Step '[4/5] Checking container status...'
    Invoke-DockerLogged @('compose', 'ps', '-a') | Out-Null

    $appHealthUrl = "http://127.0.0.1:$appPort/_stcore/health"
    $pgAdminHealthUrl = "http://127.0.0.1:$pgAdminPort/misc/ping"
    $appReady = Wait-WebService 'MissionGuard' $appHealthUrl
    $pgAdminReady = Wait-WebService 'pgAdmin' $pgAdminHealthUrl

    if (-not $appReady -or -not $pgAdminReady) {
        Write-Step 'One or more services are not reachable. Saving detailed diagnostics...'
        Invoke-DockerLogged @('compose', 'ps', '-a') | Out-Null
        Invoke-DockerLogged @('compose', 'logs', '--tail=250', 'postgres', 'pgadmin', 'app') | Out-Null
        throw 'A service did not become reachable. If this project is using an older PostgreSQL volume with a different password, copy the old .env into this folder to keep the data, or run RESET_MISSIONGUARD_DATA_WINDOWS.bat only when the old data is not needed. See DOCKER_STARTUP_LOG.txt.'
    }

    $appUrl = "http://localhost:$appPort"
    $pgAdminUrl = "http://localhost:$pgAdminPort"
    @(
        "MissionGuard=$appUrl"
        "pgAdmin=$pgAdminUrl"
        "PostgreSQL=127.0.0.1:$postgresPort"
        'pgAdmin internal database host=postgres'
        'pgAdmin internal database port=5432'
    ) | Set-Content -Path $LinksPath -Encoding UTF8

    Write-Step '[5/5] All services are ready.'
    Write-Host ''
    Write-Host '============================================================'
    Write-Host " MissionGuard: $appUrl"
    Write-Host " pgAdmin:      $pgAdminUrl"
    Write-Host " PostgreSQL:   127.0.0.1:$postgresPort"
    Write-Host '============================================================'
    Write-Host ''

    Start-Process $appUrl
    Start-Process $pgAdminUrl
    exit 0
}
catch {
    $message = "[ERROR] $($_.Exception.Message)"
    Write-Host ''
    Write-Host $message -ForegroundColor Red
    $message | Add-Content -Path $LogPath -Encoding UTF8
    try {
        "`n--- docker compose ps -a ---" | Add-Content -Path $LogPath -Encoding UTF8
        Invoke-DockerLogged @('compose', 'ps', '-a') | Out-Null
        "`n--- docker compose logs --tail=250 ---" | Add-Content -Path $LogPath -Encoding UTF8
        Invoke-DockerLogged @('compose', 'logs', '--tail=250') | Out-Null
    }
    catch { }
    Write-Host "Detailed diagnostics were saved to: $LogPath"
    exit 1
}
