@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo [1/6] Creating virtual environment...
    python -m venv .venv || exit /b 1
)

call .venv\Scripts\activate.bat

echo [2/6] Installing required packages...
python -m pip install --upgrade pip
pip install -r requirements.txt || exit /b 1

if not exist ".env" (
    echo [3/6] Creating local-mode .env from .env.example...
    copy /Y .env.example .env >nul
) else (
    echo [3/6] Environment file found.
)

if not exist "models\opssat_model.joblib" (
    echo [4/6] Training the real OPSSAT model...
    python scripts\train_opssat.py || exit /b 1
) else (
    echo [4/6] OPSSAT model artifact found.
)

findstr /R /I /C:"^[ ]*DATABASE_ENABLED[ ]*=[ ]*true" .env >nul
if errorlevel 1 (
    echo [5/6] PostgreSQL disabled - using local analysis mode.
) else (
    findstr /C:"replace_with_a_strong_password" .env >nul
    if not errorlevel 1 (
        echo Replace the placeholder PostgreSQL password in .env, then run again.
        exit /b 1
    )

    echo [5/6] Creating or verifying PostgreSQL schema...
    python scripts\bootstrap_database.py || exit /b 1
    python scripts\initialize_database.py || exit /b 1
)

echo [6/6] Starting MissionGuard AI at http://localhost:8501 ...
python -m streamlit run app.py --server.address localhost --server.port 8501
endlocal
