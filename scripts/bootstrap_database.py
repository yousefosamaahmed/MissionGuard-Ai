from __future__ import annotations

import sys
from pathlib import Path

import psycopg
from psycopg import sql

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from database.config import get_database_settings
from database.schema_requirements import REQUIRED_TABLES


def _database_missing(error: Exception) -> bool:
    sqlstate = getattr(error, "sqlstate", None)
    return sqlstate == "3D000" or "does not exist" in str(error).lower()


def _ensure_database_exists() -> None:
    settings = get_database_settings()
    database_name = settings.database

    if not database_name:
        raise RuntimeError("The target PostgreSQL database name is missing.")

    try:
        with psycopg.connect(
            **settings.psycopg_connect_kwargs(database_name=database_name),
        ):
            return
    except psycopg.Error as error:
        if not _database_missing(error):
            raise

    print(f"Database {database_name!r} does not exist; creating it...")

    maintenance_kwargs = settings.psycopg_connect_kwargs(
        database_name=settings.maintenance_database,
    )

    try:
        with psycopg.connect(
            **maintenance_kwargs,
            autocommit=True,
        ) as connection:
            connection.execute(
                sql.SQL("CREATE DATABASE {}").format(
                    sql.Identifier(database_name)
                )
            )
    except psycopg.Error as error:
        raise RuntimeError(
            "The target database is missing and the configured PostgreSQL "
            "account could not create it. Create the database in pgAdmin or "
            "use a connection string for an existing managed database."
        ) from error


def main() -> None:
    settings = get_database_settings()
    _ensure_database_exists()

    schema_file = PROJECT_ROOT / "database" / "schema.sql"
    schema_sql = schema_file.read_text(encoding="utf-8")
    schema_sql = schema_sql.replace(
        "__SCHEMA__",
        '"' + settings.schema + '"',
    )

    with psycopg.connect(
        **settings.psycopg_connect_kwargs(
            database_name=settings.database,
            include_search_path=True,
        ),
        autocommit=True,
    ) as connection:
        connection.execute(schema_sql)

        table_rows = connection.execute(
            """
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = %s
              AND table_type = 'BASE TABLE'
            ORDER BY table_name
            """,
            (settings.schema,),
        ).fetchall()

    found_tables = {str(row[0]) for row in table_rows}
    missing_tables = sorted(set(REQUIRED_TABLES) - found_tables)

    if missing_tables:
        raise RuntimeError(
            "Database bootstrap completed with missing tables: "
            + ", ".join(missing_tables)
        )

    print(
        "MissionGuard PostgreSQL schema is ready. "
        f"Verified {len(REQUIRED_TABLES)} required tables "
        f"in {settings.database}.{settings.schema}."
    )


if __name__ == "__main__":
    main()
