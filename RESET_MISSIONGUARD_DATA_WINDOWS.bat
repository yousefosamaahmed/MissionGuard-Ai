@echo off
setlocal EnableExtensions
cd /d "%~dp0"

echo ============================================================
echo  WARNING: This deletes ONLY MissionGuard Docker data
echo ============================================================
echo.
echo This removes the MissionGuard PostgreSQL database, pgAdmin settings,
echo and stored upload volume. Other Docker projects are not deleted.
echo.
set /p "CONFIRM=Type DELETE to continue: "
if /I not "%CONFIRM%"=="DELETE" (
  echo Cancelled. Nothing was deleted.
  pause
  exit /b 0
)

docker compose down -v --remove-orphans
if errorlevel 1 (
  echo Reset failed. Make sure Docker Desktop is running.
  pause
  exit /b 1
)

echo.
echo MissionGuard Docker data was reset.
echo Run START_DOCKER_WINDOWS.bat to create a fresh database.
pause
