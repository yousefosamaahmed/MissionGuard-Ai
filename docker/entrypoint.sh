#!/usr/bin/env sh
set -eu

python scripts/wait_for_database.py
python scripts/bootstrap_database.py
python scripts/initialize_database.py

exec python -m streamlit run app.py \
  --server.address=0.0.0.0 \
  --server.port="${PORT:-8501}" \
  --server.headless=true
