@echo off
setlocal EnableExtensions
cd /d "%~dp0"

echo ============================================================
echo  MissionGuard AI - Safe Docker startup
echo ============================================================
echo.

powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\start_docker_windows.ps1"
set "EXIT_CODE=%ERRORLEVEL%"

echo.
if not "%EXIT_CODE%"=="0" (
  echo Startup did not complete. Read DOCKER_STARTUP_LOG.txt in this folder.
) else (
  echo The correct localhost links are saved in LOCAL_LINKS.txt.
)
echo.
pause
exit /b %EXIT_CODE%
