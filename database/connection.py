from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Generator

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine, URL
from sqlalchemy.orm import Session, sessionmaker

from database.config import (
    DatabaseSettings,
    database_requested,
    get_database_settings,
)
from database.schema_requirements import REQUIRED_TABLES

DATABASE_REQUESTED = database_requested()
DATABASE_CONFIGURATION_ERROR: str | None = None
settings: DatabaseSettings | None = None
engine: Engine | None = None
SessionLocal: sessionmaker[Session] | None = None

POSTGRES_USER: str | None = None
POSTGRES_PASSWORD: str | None = None
POSTGRES_HOST: str | None = None
POSTGRES_PORT = 5432
POSTGRES_DB = "missionguard_ai"
POSTGRES_SCHEMA = "missionguard"
DATABASE_URL: URL | None = None

if DATABASE_REQUESTED:
    try:
        settings = get_database_settings()

        POSTGRES_USER = settings.user
        POSTGRES_PASSWORD = settings.password
        POSTGRES_HOST = settings.host
        POSTGRES_PORT = settings.port or 5432
        POSTGRES_DB = settings.database or "missionguard_ai"
        POSTGRES_SCHEMA = settings.schema
        DATABASE_URL = settings.url

        engine = create_engine(
            DATABASE_URL,
            pool_pre_ping=True,
            pool_size=settings.pool_size,
            max_overflow=settings.max_overflow,
            pool_timeout=settings.pool_timeout,
            pool_recycle=settings.pool_recycle,
            connect_args=settings.sqlalchemy_connect_args(),
        )

        SessionLocal = sessionmaker(
            bind=engine,
            class_=Session,
            autoflush=False,
            expire_on_commit=False,
        )
    except Exception as exc:  # Keep the UI available in local-only mode.
        DATABASE_CONFIGURATION_ERROR = f"{type(exc).__name__}: {exc}"
        settings = None
        engine = None
        SessionLocal = None


def database_is_enabled() -> bool:
    """Return whether a usable PostgreSQL engine is configured."""

    return bool(
        DATABASE_REQUESTED
        and DATABASE_CONFIGURATION_ERROR is None
        and engine is not None
        and SessionLocal is not None
    )


def database_status_message() -> str:
    """Return a user-facing explanation of the database mode."""

    if not DATABASE_REQUESTED:
        return (
            "PostgreSQL persistence is disabled. The full local analysis, "
            "CSV upload, model validation, reports, and dashboards remain "
            "available. Set DATABASE_ENABLED=true in .env to enable storage."
        )

    if DATABASE_CONFIGURATION_ERROR:
        return (
            "PostgreSQL was requested but its configuration is invalid: "
            f"{DATABASE_CONFIGURATION_ERROR}"
        )

    return "PostgreSQL is configured."


def _require_database() -> tuple[Engine, sessionmaker[Session]]:
    if not database_is_enabled() or engine is None or SessionLocal is None:
        raise RuntimeError(database_status_message())

    return engine, SessionLocal


@contextmanager
def database_session() -> Generator[Session, None, None]:
    _, session_factory = _require_database()
    session = session_factory()

    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def check_database_connection() -> dict[str, Any]:
    active_engine, _ = _require_database()

    with active_engine.connect() as connection:
        database_name = connection.execute(
            text("SELECT current_database()")
        ).scalar_one()

        current_schema = connection.execute(
            text("SELECT current_schema()")
        ).scalar_one()

        postgres_version = connection.execute(
            text("SELECT version()")
        ).scalar_one()

        table_rows = connection.execute(
            text(
                """
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema = :schema_name
                  AND table_type = 'BASE TABLE'
                ORDER BY table_name
                """
            ),
            {"schema_name": POSTGRES_SCHEMA},
        ).scalars().all()

    table_names = [str(table_name) for table_name in table_rows]
    missing_required_tables = sorted(
        set(REQUIRED_TABLES) - set(table_names)
    )

    return {
        "database_name": database_name,
        "schema_name": current_schema,
        "configured_schema": POSTGRES_SCHEMA,
        "table_count": len(table_names),
        "table_names": table_names,
        "required_table_count": len(REQUIRED_TABLES),
        "missing_required_tables": missing_required_tables,
        "postgres_version": postgres_version,
    }
