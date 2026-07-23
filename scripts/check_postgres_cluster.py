from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import cast

import psycopg
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from database.schema_requirements import REQUIRED_TABLES

_ = load_dotenv(
    PROJECT_ROOT / ".env",
    override=True,
)


def get_required_env(name: str) -> str:
    """
    Read a required environment variable.

    Raises:
        RuntimeError:
            If the value is missing or empty.
    """

    value = os.getenv(name)

    if value is None or not value.strip():
        raise RuntimeError(
            f"{name} is missing or empty in the .env file."
        )

    return value.strip()


def get_optional_env(
    name: str,
    default: str,
) -> str:
    """
    Read an optional environment variable.
    """

    value = os.getenv(name)

    if value is None or not value.strip():
        return default

    return value.strip()


def get_postgres_port() -> int:
    """
    Read and validate the PostgreSQL port.
    """

    raw_port = get_optional_env(
        "POSTGRES_PORT",
        "5432",
    )

    try:
        return int(raw_port)

    except ValueError as error:
        raise RuntimeError(
            "POSTGRES_PORT must contain a valid integer."
        ) from error


host: str = get_optional_env(
    "POSTGRES_HOST",
    "127.0.0.1",
)

port: int = get_postgres_port()

user: str = get_optional_env(
    "POSTGRES_USER",
    "postgres",
)

password: str = get_required_env(
    "POSTGRES_PASSWORD"
)

configured_database: str = get_required_env(
    "POSTGRES_DB"
)

configured_schema: str = get_optional_env(
    "POSTGRES_SCHEMA",
    "missionguard",
)


print("=" * 70)
print("MissionGuard PostgreSQL Cluster Diagnostic")
print("=" * 70)

print(f"Host: {host!r}")
print(f"Port: {port}")
print(f"User: {user!r}")
print(f"Configured database: {configured_database!r}")
print(f"Configured schema: {configured_schema!r}")


# ============================================================
# Connect to PostgreSQL maintenance database
# ============================================================

with psycopg.connect(
    host=host,
    port=port,
    user=user,
    password=password,
    dbname="postgres",
) as connection:

    server_cursor = connection.execute(
        """
        SELECT
            current_database(),
            current_user,
            inet_server_addr(),
            inet_server_port(),
            current_setting('data_directory'),
            version()
        """
    )

    server_row = server_cursor.fetchone()

    if server_row is None:
        raise RuntimeError(
            "PostgreSQL returned no server information."
        )

    connected_database = cast(str, server_row[0])
    connected_user = cast(str, server_row[1])
    server_address = cast(str, server_row[2])
    server_port = cast(str, server_row[3])
    data_directory = cast(str, server_row[4])
    postgres_version = cast(str, server_row[5])

    databases_cursor = connection.execute(
        """
        SELECT
            datname,
            length(datname)
        FROM pg_database
        WHERE datistemplate = FALSE
        ORDER BY datname
        """
    )

    database_rows = databases_cursor.fetchall()


print("\nServer seen by Python:")
print(f"Connected database: {connected_database}")
print(f"Connected user: {connected_user}")
print(f"Server address: {server_address}")
print(f"Server port: {server_port}")
print(f"Data directory: {data_directory}")
print(f"PostgreSQL version: {postgres_version}")


print("\nDatabases seen by Python:")

database_names: list[str] = []

for database_row in database_rows:
    database_name = cast(str, database_row[0])
    database_name_length = cast(int, database_row[1])

    database_names.append(database_name)

    print(f"- name={database_name!r}, length={database_name_length}")


if configured_database not in database_names:
    print("\nERROR: Python is connected to a PostgreSQL instance that does not contain the configured database.")
    print(f"Missing database: {configured_database!r}")
    raise SystemExit(1)


print("\nThe configured database exists. Testing the MissionGuard schema...")


# ============================================================
# Connect to MissionGuard database
# ============================================================

with psycopg.connect(
    host=host,
    port=port,
    user=user,
    password=password,
    dbname=configured_database,
    options=f"-c search_path={configured_schema},public",
) as connection:

    database_info_cursor = connection.execute(
        """
        SELECT
            current_database(),
            current_schema()
        """
    )

    database_info_row = database_info_cursor.fetchone()

    if database_info_row is None:
        raise RuntimeError(
            "PostgreSQL returned no database information."
        )

    current_database = cast(str, database_info_row[0])
    current_schema = cast(str, database_info_row[1])

    table_cursor = connection.execute(
        """
        SELECT table_name
        FROM information_schema.tables
        WHERE table_schema = %s
          AND table_type = 'BASE TABLE'
        ORDER BY table_name
        """,
        (configured_schema,),
    )

    table_rows = table_cursor.fetchall()
    table_names = {
        str(table_row[0])
        for table_row in table_rows
    }
    table_count = len(table_names)
    missing_required_tables = sorted(
        set(REQUIRED_TABLES) - table_names
    )


print("\nMissionGuard connection result:")
print(f"Database: {current_database}")
print(f"Schema: {current_schema}")
print(f"Tables: {table_count}")


if missing_required_tables:
    print(
        "Connection succeeded, but required tables are missing: "
        + ", ".join(missing_required_tables)
    )
    raise SystemExit(1)

print(
    "Connection successful. Verified all "
    f"{len(REQUIRED_TABLES)} required MissionGuard tables."
)
