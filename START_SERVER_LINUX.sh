#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

if ! command -v docker >/dev/null 2>&1; then
  echo "Docker is not installed. Install Docker Engine and the Compose plugin first."
  exit 1
fi

if [[ ! -f .env ]]; then
  cp .env.server.example .env
  echo "Created .env. Replace every CHANGE_ME value, then run this script again."
  exit 1
fi

if grep -q 'CHANGE_ME' .env; then
  echo "Replace every CHANGE_ME value in .env before starting."
  exit 1
fi

docker compose up -d --build

echo "MissionGuard: http://SERVER_IP:${APP_PORT:-8501}"
echo "pgAdmin is bound to 127.0.0.1:${PGADMIN_PORT:-5050} by default."
echo "Use an SSH tunnel to access pgAdmin securely."
