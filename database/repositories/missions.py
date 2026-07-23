from __future__ import annotations

from typing import Any
from uuid import UUID

from sqlalchemy import insert, select

from database.connection import database_session
from database.tables import get_table


def create_mission(
    name: str,
    spacecraft_name: str | None = None,
    description: str | None = None,
    status: str = "active",
    mission_code: str | None = None,
) -> UUID:
    """
    Create a new mission and return its generated UUID.
    """

    clean_name = name.strip()

    if not clean_name:
        raise ValueError(
            "Mission name cannot be empty."
        )

    valid_statuses = {
        "planned",
        "active",
        "paused",
        "completed",
        "archived",
    }

    if status not in valid_statuses:
        raise ValueError(
            f"Invalid mission status: {status}"
        )

    missions_table = get_table(
        "missions"
    )

    statement = (
        insert(missions_table)
        .values(
            name=clean_name,
            mission_code=mission_code,
            spacecraft_name=spacecraft_name,
            description=description,
            status=status,
        )
        .returning(
            missions_table.c.id
        )
    )

    with database_session() as session:
        result = session.execute(
            statement
        )

        mission_id_value = (
            result.scalar_one()
        )

    if isinstance(
        mission_id_value,
        UUID,
    ):
        return mission_id_value

    return UUID(
        str(mission_id_value)
    )


def list_missions() -> list[dict[str, Any]]:
    """
    Return all stored missions ordered from newest to oldest.
    """

    missions_table = get_table(
        "missions"
    )

    statement = (
        select(missions_table)
        .order_by(
            missions_table
            .c
            .created_at
            .desc()
        )
    )

    with database_session() as session:
        result = session.execute(
            statement
        )

        rows = (
            result
            .mappings()
            .all()
        )

    return [
        dict(row)
        for row in rows
    ]


def get_mission(
    mission_id: UUID,
) -> dict[str, Any] | None:
    """
    Return one mission by its UUID.
    """

    missions_table = get_table(
        "missions"
    )

    statement = (
        select(missions_table)
        .where(
            missions_table.c.id
            == mission_id
        )
    )

    with database_session() as session:
        result = session.execute(
            statement
        )

        row = (
            result
            .mappings()
            .one_or_none()
        )

    if row is None:
        return None

    return dict(row)