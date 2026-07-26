from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import psycopg

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from database.config import get_database_settings


def main() -> None:
    settings = get_database_settings()
    attempts = int(os.getenv("DATABASE_WAIT_ATTEMPTS", "60"))
    delay_seconds = float(os.getenv("DATABASE_WAIT_DELAY_SECONDS", "2"))
    last_error: Exception | None = None

    for attempt in range(1, attempts + 1):
        try:
            connection_kwargs = settings.psycopg_connect_kwargs(
                database_name=settings.database,
            )
            connection_kwargs.setdefault("connect_timeout", 3)

            with psycopg.connect(
                **connection_kwargs,
            ) as connection:
                connection.execute("SELECT 1").fetchone()

            print("PostgreSQL is ready.")
            return
        except Exception as error:
            last_error = error
            print(
                f"Waiting for PostgreSQL ({attempt}/{attempts}): "
                f"{type(error).__name__}"
            )
            time.sleep(delay_seconds)

    raise RuntimeError(
        "PostgreSQL did not become ready before the startup timeout."
    ) from last_error


if __name__ == "__main__":
    main()
