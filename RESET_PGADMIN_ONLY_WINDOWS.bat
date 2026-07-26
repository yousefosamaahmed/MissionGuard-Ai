@echo off
setlocal EnableExtensions
cd /d "%~dp0"

echo This resets pgAdmin login/settings only.
echo PostgreSQL database data will NOT be deleted.
echo.
set /p "CONFIRM=Type RESET to continue: "
if /I not "%CONFIRM%"=="RESET" (
  echo Cancelled.
  pause
  exit /b 0
)

docker compose stop pgadmin >nul 2>&1
docker compose rm -f pgadmin >nul 2>&1
docker volume rm missionguard_missionguard_pgadmin_data >nul 2>&1
docker compose up -d pgadmin
if errorlevel 1 (
  echo pgAdmin reset failed. Make sure Docker Desktop is running.
  pause
  exit /b 1
)

echo.
echo pgAdmin was reset. Open http://localhost:5050 after it starts.
echo Email: admin@missionguard.com
echo Password: MissionGuardAdmin2026_Local
pause
