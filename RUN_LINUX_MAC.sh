#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -x ".venv/bin/python" ]; then
  echo "[1/6] Creating virtual environment..."
  python3 -m venv .venv
fi

source .venv/bin/activate

echo "[2/6] Installing required packages..."
python -m pip install --upgrade pip
pip install -r requirements.txt

if [ ! -f ".env" ]; then
  echo "[3/6] Creating local-mode .env from .env.example..."
  cp .env.example .env
else
  echo "[3/6] Environment file found."
fi

if [ ! -f "models/opssat_model.joblib" ]; then
  echo "[4/6] Training the real OPSSAT model..."
  python scripts/train_opssat.py
else
  echo "[4/6] OPSSAT model artifact found."
fi

if grep -Eiq '^[[:space:]]*DATABASE_ENABLED[[:space:]]*=[[:space:]]*true' .env; then
  if grep -q 'replace_with_a_strong_password' .env; then
    echo "Replace the placeholder PostgreSQL password in .env, then run again."
    exit 1
  fi

  echo "[5/6] Creating or verifying PostgreSQL schema..."
  python scripts/bootstrap_database.py
  python scripts/initialize_database.py
else
  echo "[5/6] PostgreSQL disabled - using local analysis mode."
fi

echo "[6/6] Starting MissionGuard AI at http://localhost:8501 ..."
python -m streamlit run app.py --server.address localhost --server.port 8501
