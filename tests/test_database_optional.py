from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_database_modules_import_when_postgresql_is_disabled() -> None:
    script = """
from database.connection import database_is_enabled, database_status_message
from database.repositories.analysis import list_incidents
from database.services.opssat_inference_service import run_real_opssat_analysis

assert database_is_enabled() is False
assert 'disabled' in database_status_message().lower()
assert callable(list_incidents)
assert callable(run_real_opssat_analysis)
"""

    environment = os.environ.copy()
    environment["DATABASE_ENABLED"] = "false"
    environment.pop("DATABASE_URL", None)
    environment.pop("POSTGRES_USER", None)
    environment.pop("POSTGRES_PASSWORD", None)

    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=PROJECT_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
