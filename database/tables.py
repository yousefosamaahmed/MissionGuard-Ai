from __future__ import annotations

from functools import lru_cache

from sqlalchemy import MetaData, Table

from database.connection import (
    POSTGRES_SCHEMA,
    database_status_message,
    engine,
)

metadata = MetaData(
    schema=POSTGRES_SCHEMA,
)


@lru_cache(maxsize=None)
def get_table(table_name: str) -> Table:
    """
    Load an existing PostgreSQL table using SQLAlchemy reflection.

    This function does not create the table.
    It reads the table structure already stored in PostgreSQL.
    """

    if not table_name.strip():
        raise ValueError(
            "Table name cannot be empty."
        )

    if engine is None:
        raise RuntimeError(database_status_message())

    return Table(
        table_name,
        metadata,
        schema=POSTGRES_SCHEMA,
        autoload_with=engine,
        extend_existing=True,
    )